"""Expected behaviour and change detection.

expected  = median of the same hours on each of the previous 7 days (robust to a bad day in the history)
noise     = 1.4826 x MAD of those 7 values, floored by sampling noise (Poisson / binomial) and 2% of expected
anomaly   = |z| >= Z_MIN and |relative change| >= REL_MIN and enough volume
complete  = every source feeding the metric had all rows for the window (else: 'incomplete', not a change)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .store import NotApplicable, Store

H = pd.Timedelta(hours=1)
WINDOW_H = 3
Z_MIN = 4.5
Z_STRONG = 8.0        # above this, one window is enough; below, the previous window must agree (persistence)
REL_MIN = 0.10
EUR_CV = 1.2          # order-level variation of order values (category mix): noise of EUR metrics ~ 1.2/sqrt(orders)
MIN_VOLUME = 100


def day_factor(store: Store, day: pd.Timestamp) -> float:
    """Traffic level of a calendar day relative to the average complete day (weekday rhythm).
    Counts are compared after dividing by it; rates need no correction."""
    key = ("day_factor", str(pd.Timestamp(day).normalize()))
    if key in store._cache:
        return store._cache[key]
    s = store.series("sessions")
    daily = s.groupby(s.index.normalize()).agg(["sum", "count"])
    full = daily[daily["count"] == 24]["sum"]
    d = pd.Timestamp(day).normalize()
    prev = d - pd.Timedelta(days=7)
    # the weekday level comes from the same weekday one week earlier, never from the day itself:
    # a campaign or outage on the day must not leak into what we expect for that day
    if prev in full.index and len(full) >= 3:
        f = float(full.loc[prev] / full.mean())
    elif d in full.index and len(full) >= 3:      # first week of history: no earlier weekday to use
        f = float(full.loc[d] / full.mean())
    else:
        f = None        # weekday level not learnable yet (no earlier same weekday, day not complete)
    store._cache[key] = f
    return f


def baseline_days(store: Store, start) -> int:
    """How many of the previous 7 days have this window fully inside the data."""
    first = store.hours[0]
    return int(min(7, max(0, (pd.Timestamp(start) - first) // pd.Timedelta(days=1))))


def is_count(store: Store, metric: str) -> bool:
    return store.metrics[metric].get("unit") in ("count", "EUR") and store.expr(metric).kind != "ratio"


@dataclass
class Measurement:
    metric: str
    segment: dict
    start: str
    end: str
    actual: float
    expected: float
    pooled: float           # pooled-baseline value (for exact decompositions)
    rel: float
    z: float
    volume: float
    completeness: float
    status: str             # anomaly | normal | low_volume | no_baseline | incomplete | n/a
    direction: str          # up | down | ""

    def seg_text(self) -> str:
        return seg_text(self.segment)

    def short(self) -> str:
        return (f"{self.metric}[{self.seg_text()}] {fmt(self.actual, self.metric)} vs expected "
                f"{fmt(self.expected, self.metric)} ({self.rel:+.1%}, z={self.z:+.1f})")

    def to_dict(self) -> dict:
        return asdict(self)


def seg_text(seg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(seg.items())) or "total"


def fmt(v: float, metric: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if abs(v) < 1 and v != 0:
        return f"{v:.2%}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def window(now: pd.Timestamp, hours: int = WINDOW_H) -> tuple[pd.Timestamp, pd.Timestamp]:
    return now - (hours - 1) * H, now


def completeness(store: Store, metric: str, start, end) -> float:
    vals = []
    for src in store.sources(metric):
        c = store.completeness[(store.completeness.source == src) & (store.completeness.ts >= start)
                               & (store.completeness.ts <= end)]
        if len(c):
            vals.append(float(c.completeness.min()))
    return min(vals) if vals else 1.0


def volume(store: Store, metric: str, start, end, seg: dict) -> float:
    e = store.expr(metric)
    if e.kind == "ratio":
        return float(store.window_value(e.tree.right.id, start, end, seg))
    if e.kind == "base":
        return float(store.window_value(metric, start, end, seg))
    # composite / other ratios: use the smallest base input as the volume
    vols = [float(store.window_value(m, start, end, seg)) for m in e.metrics if store.expr(m).kind == "base"]
    return min(vols) if vols else MIN_VOLUME


def measure(store: Store, metric: str, seg: dict, now: pd.Timestamp, hours: int = WINDOW_H,
            persistence: bool = True) -> Measurement:
    m = _measure(store, metric, seg, now, hours)
    if persistence and m.status == "anomaly" and abs(m.z) < Z_STRONG:
        prev = _measure(store, metric, seg, now - H, hours)
        if not (prev.status == "anomaly" and prev.direction == m.direction):
            m.status = "candidate"      # one window only: watch, do not investigate yet
    return m


def _measure(store: Store, metric: str, seg: dict, now: pd.Timestamp, hours: int = WINDOW_H) -> Measurement:
    start, end = window(now, hours)
    base = dict(metric=metric, segment=dict(seg), start=str(start), end=str(end))
    try:
        nb = baseline_days(store, start)
        actual = float(store.window_value(metric, start, end, seg))
        daily = [float(store.window_value(metric, start - pd.Timedelta(days=k), end - pd.Timedelta(days=k), seg))
                 for k in range(1, nb + 1)]
        pooled = float(store.pooled_value(metric, start, end, max(nb, 1), seg))
        vol = volume(store, metric, start, end, seg)
    except NotApplicable:
        return Measurement(**base, actual=np.nan, expected=np.nan, pooled=np.nan, rel=np.nan, z=np.nan, volume=0,
                           completeness=1.0, status="n/a", direction="")
    comp = completeness(store, metric, start, end)
    if is_count(store, metric):   # remove the weekday rhythm before comparing days
        today = day_factor(store, start)
        if today is None:         # first week: a normal Saturday would look like a +10% anomaly
            return Measurement(**base, actual=actual, expected=np.nan, pooled=pooled, rel=np.nan, z=np.nan,
                               volume=vol, completeness=comp, status="no_baseline", direction="")
        daily = [d / (day_factor(store, start - pd.Timedelta(days=k)) or 1.0) for k, d in enumerate(daily, start=1)]
    else:
        today = 1.0
    daily = [d for d in daily if d == d and not math.isinf(d)]
    if len(daily) < 4:
        return Measurement(**base, actual=actual, expected=np.nan, pooled=pooled, rel=np.nan, z=np.nan, volume=vol,
                           completeness=comp, status="no_baseline", direction="")
    exp = float(np.median(daily)) * today
    mad = 1.4826 * float(np.median(np.abs(np.array(daily) * today - exp)))
    basis = store.metrics[metric].get("noise_basis")
    if basis:
        n = float(store.pooled_value(basis, start, end, max(nb, 1), seg)) / max(nb, 1)
        floor = abs(exp) * EUR_CV / math.sqrt(max(n, 1.0))
    elif store.expr(metric).kind == "base":
        floor = math.sqrt(max(exp, 1.0))
    else:
        p = min(max(abs(exp), 1e-6), 1 - 1e-6) if abs(exp) < 1 else abs(exp)
        floor = math.sqrt(p * (1 - p) / max(vol, 1)) if abs(exp) < 1 else 0.0
    sd = max(mad, floor, 0.02 * abs(exp), 1e-9)
    rel = actual / exp - 1 if exp else np.nan
    z = (actual - exp) / sd
    status = "normal"
    if comp < 0.95:
        status = "incomplete"
    elif vol < MIN_VOLUME:
        status = "low_volume"
    elif abs(z) >= Z_MIN and abs(rel) >= REL_MIN:
        status = "anomaly"
    return Measurement(**base, actual=actual, expected=exp, pooled=pooled, rel=rel, z=z, volume=vol,
                       completeness=comp, status=status, direction="up" if actual > exp else "down")


def hourly_deviation(store: Store, metric: str, seg: dict, now: pd.Timestamp, back: int = 30) -> pd.Series:
    """Relative deviation per hour vs the median of the same hour on the previous 7 days."""
    s = store.series(metric, seg)
    s = s.loc[:now]
    hist = pd.concat([s.shift(24 * k) for k in range(1, 8)], axis=1)
    med = hist.median(axis=1)
    dev = (s / med - 1).replace([np.inf, -np.inf], np.nan)
    return dev.iloc[-back:]


def onset(store: Store, metric: str, seg: dict, now: pd.Timestamp, rel_min: float = REL_MIN,
          direction: str = "down") -> pd.Timestamp | None:
    """First hour of the current run of hours deviating in `direction` by more than rel_min."""
    dev = hourly_deviation(store, metric, seg, now)
    sign = -1 if direction == "down" else 1
    hit = (dev * sign) >= rel_min
    if not hit.iloc[-1] and not hit.iloc[-2:].any():
        return None
    first = None
    for ts in reversed(dev.index):
        if hit.loc[ts]:
            first = ts
        elif first is not None:
            # allow one noisy hour inside the run
            prev = ts - H
            if prev in hit.index and hit.loc[prev]:
                continue
            break
    return first
