"""SELF-CHECK: is it real? (SKILL.md section 2). Every check that ran is recorded in
`checks` so the alert can show it; a check that could not run is reported, never skipped."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .detectors import Detection
from .loader import MarketData
from .state import State


@dataclass
class CheckResult:
    verdict: str                 # alert | suppressed | insufficient | no_baseline
    reason: str = ""
    checks: dict = field(default_factory=dict)


def freshness(data: MarketData, now: pd.Timestamp, cfg: dict) -> tuple[bool, str]:
    latest = data.latest_ts
    max_h = cfg["thresholds"]["max_data_staleness_hours"]
    if latest is None:
        return False, "no data loaded"
    age_h = (now - latest) / pd.Timedelta(hours=1)
    if age_h > max_h:
        return False, f"STALE: latest data {latest} is {age_h:.0f}h old (max {max_h}h)"
    return True, "OK"


def known_event(det: Detection, cfg: dict) -> dict | None:
    for ev in cfg.get("known_events") or []:
        start, end = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        scope = ev.get("scope", ["*"])
        if det.signal not in ev.get("suppress_signals", []):
            continue
        if "*" not in scope and det.entity not in scope:
            continue
        if ev.get("expected_effect") == "volume_drop" and det.direction != "down":
            continue  # a holiday explains a drop, never a spike
        if det.window_start < end and det.window_end >= start:
            return ev
    return None


def check(det: Detection, fresh: tuple[bool, str], cfg: dict) -> CheckResult:
    checks = {"freshness": fresh[1]}
    if not det.baseline_ok:
        checks["seasonality"] = f"NO BASELINE ({det.note})"
        return CheckResult("no_baseline", det.note, checks)
    checks["seasonality"] = "OK (same weekday/hour)" if det.signal != "S7" else "OK (trailing 7d)"
    if det.insufficient:
        checks["sample"] = f"INSUFFICIENT ({det.note})"
        return CheckResult("insufficient", det.note, checks)
    checks["sample"] = "OK"
    ev = known_event(det, cfg)
    if ev:
        checks["known_events"] = f"MATCH: {ev['name']}"
        return CheckResult("suppressed", f"known event: {ev['name']}", checks)
    checks["known_events"] = "none"
    checks["tracking_vs_reality"] = _tracking_vs_reality(det)
    checks["change_log"] = "NOT AVAILABLE (no change log connected)"
    return CheckResult("alert", "", checks)


def _tracking_vs_reality(det: Detection) -> str:
    if det.signal == "S4":
        r = det.metrics.get("click_ratio")
        return f"clicks normal ({r:.0%}) -> tracking suspected" if r else "n/a"
    return "n/a for this signal"


def dedupe(det: Detection, severity: int, state: State, now: pd.Timestamp, cfg: dict
           ) -> tuple[str, dict]:
    """Returns (action, record): new | realert (severity rose) | update | cooldown."""
    rec = state.find_open(det.signal, det.entity)
    if rec is None:
        return "new", state.open_anomaly(det.signal, det.entity, now, severity, det.title)
    rec["last_seen"] = str(now)
    rec["in_range_checks"] = 0
    if severity > rec["severity"]:
        rec["severity"] = severity
        rec["last_alert"] = str(now)
        rec["alerts"] += 1
        return "realert", rec
    if state.in_cooldown(rec, now, cfg["cooldown_hours"]):
        return "cooldown", rec
    return "update", rec
