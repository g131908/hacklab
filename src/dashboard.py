"""Build the demo dashboard: one self-contained HTML file (no network, opens via double-click).

python src/dashboard.py            -> dashboard/index.html  (run replay.py first)
python src/dashboard.py --open     also opens it in the default browser

Reads: data (via the configured loader), .agents/monitoring/alerts.jsonl,
data/injected.json (optional ground truth), .agents/monitoring/config.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ALERTS, INJECTED, compute, load_log  # noqa: E402
from monitor.loader import MarketData, make_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / ".agents" / "monitoring" / "config.yaml"
TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
OUT = ROOT / "dashboard" / "index.html"
WEEK = pd.Timedelta(days=7)
H = pd.Timedelta(hours=1)
SIGNAL_NAMES = {"S1": "Spend pacing", "S4": "Tracking break", "S6": "Creative fatigue", "S7": "Invalid traffic"}


def _clean(values) -> list:
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(None)
        else:
            out.append(round(float(v), 6))
    return out


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M")


def _hourly_pair(series: pd.Series, idx: pd.DatetimeIndex) -> tuple[list, list]:
    s = series.reindex(idx)
    base = series.reindex(idx - WEEK)
    base.index = idx
    return _clean(s.values), _clean(base.values)


def _chart(title, unit, idx, series, fmt="int", note=""):
    return dict(title=title, unit=unit, fmt=fmt, note=note, ts=[_iso(t) for t in idx], series=series)


def focus_charts(signal: str, entity: str, ts: pd.Timestamp, data: MarketData) -> list[dict]:
    """Two small multiples per anomaly, each one axis: actual vs its baseline."""
    hours = data.wide("adv", "spend").index
    lo, hi = max(ts - pd.Timedelta(hours=36), hours.min()), min(ts + pd.Timedelta(hours=35), hours.max())
    idx = pd.date_range(lo, hi, freq="h")
    if signal == "S4":
        out = []
        for metric, label in (("conversions", "Conversions per hour"), ("clicks", "Clicks per hour")):
            a, b = _hourly_pair(data.wide("adv", metric)[entity], idx)
            out.append(_chart(label, "", idx, [dict(name=entity, role="actual", values=a),
                                               dict(name="same hour last week", role="baseline", values=b)]))
        out[1]["note"] = "Clicks stay on baseline while conversions drop: tracking, not demand."
        return out
    if signal == "S1":
        sp = data.wide("adv", "spend")[entity]
        im = data.wide("adv", "impressions")[entity]
        cum = sp.groupby(sp.index.normalize()).cumsum()
        a, b = _hourly_pair(cum, idx)
        cpm = sp / im * 1000
        ca, cb = _hourly_pair(cpm, idx)
        return [
            _chart("Spend to date (resets at midnight)", "EUR", idx,
                   [dict(name=entity, role="actual", values=a),
                    dict(name="same weekday last week", role="baseline", values=b)], fmt="eur"),
            _chart("CPM per hour", "EUR", idx,
                   [dict(name=entity, role="actual", values=ca),
                    dict(name="same hour last week", role="baseline", values=cb)], fmt="eur2",
                   note="Price per impression up, volume flat: the overspend is bid-driven."),
        ]
    if signal == "S6":
        days = pd.date_range(hours.min().normalize(), hours.max().normalize(), freq="D")
        imp = data.wide("adv", "impressions")[entity].groupby(lambda t: t.normalize()).sum()
        clk = data.wide("adv", "clicks")[entity].groupby(lambda t: t.normalize()).sum()
        frq = data.wide("adv", "frequency")[entity].groupby(lambda t: t.normalize()).mean()
        ctr = (clk / imp).reindex(days)
        return [
            _chart("CTR per day", "%", days, [dict(name=entity, role="actual", values=_clean(ctr.values * 100))],
                   fmt="pct"),
            _chart("Average frequency per day", "", days,
                   [dict(name=entity, role="actual", values=_clean(frq.reindex(days).values))], fmt="dec2",
                   note="Frequency climbs while CTR slides: same people, same ad."),
        ]
    if signal == "S7":
        clk = data.wide("sup", "clicks")[entity]
        imp = data.wide("sup", "impressions")[entity]
        ctr = clk / imp * 100
        trailing = ctr.shift(1).rolling(168, min_periods=48).mean()
        ip = data.wide("sup", "top_ip_click_share")[entity] * 100
        return [
            _chart("CTR per hour", "%", idx, [dict(name=entity, role="actual", values=_clean(ctr.reindex(idx).values)),
                                              dict(name="trailing 7-day mean", role="baseline",
                                                   values=_clean(trailing.reindex(idx).values))], fmt="pct"),
            _chart("Top-IP share of clicks", "%", idx,
                   [dict(name=entity, role="actual", values=_clean(ip.reindex(idx).values))], fmt="pct",
                   note="One IP sends a third of all clicks during the burst."),
        ]
    return []


def build(open_browser: bool = False) -> Path:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    data = make_source(cfg, ROOT).load()
    if not ALERTS.exists():
        sys.exit("missing .agents/monitoring/alerts.jsonl - run: python src/replay.py")
    log = load_log()
    truth = json.loads(INJECTED.read_text(encoding="utf-8")) if INJECTED.exists() else None

    spend = data.wide("adv", "spend").sum(axis=1)
    hours = spend.index
    ov_a, ov_b = _hourly_pair(spend, hours)

    bands = []
    for ev in cfg.get("known_events") or []:
        bands.append(dict(kind="known_event", label=ev["name"], start=_iso(ev["start"]), end=_iso(ev["end"])))
    if truth:
        for sc in truth["scenarios"]:
            bands.append(dict(kind="injected", label=f"{sc['id']} ({sc['entity']})", start=_iso(sc["start"]),
                              end=_iso(sc["end"]), expected=sc["expected"], signal=sc["signal"]))

    events, focus, sup_groups = [], {}, set()
    for _, r in log.sort_values("ts").iterrows():
        e = dict(ts=_iso(r.ts), kind=r.event, signal=r.get("signal"), entity=r.get("entity"),
                 anomaly_id=r.get("anomaly_id") if isinstance(r.get("anomaly_id"), str) else None)
        trace = r.get("trace") if isinstance(r.get("trace"), list) else None
        if r.event in ("alert", "digest"):
            e.update(severity=int(r.severity), headline=r.get("headline"), text=r.get("text"),
                     options=r.get("options"), recommended=r.get("recommended"), decision=r.get("decision"),
                     realert=bool(r.get("realert")), disagreement=bool(r.get("disagreement")),
                     specialists=r.get("specialists") or [],
                     money=None if pd.isna(r.get("money_at_risk_day")) else float(r.get("money_at_risk_day")),
                     trace=trace)
            if e["anomaly_id"] not in focus:
                focus[e["anomaly_id"]] = dict(signal=r.signal, entity=r.entity,
                                              name=SIGNAL_NAMES.get(r.signal, r.signal),
                                              charts=focus_charts(r.signal, r.entity, r.ts, data))
        elif r.event == "suppressed":
            group = f"{r.ts:%Y-%m-%d}|{r.get('reason')}"
            e.update(reason=r.get("reason"), title=r.get("title"), group="SUP|" + group)
            if group not in sup_groups:  # one example trace + charts per suppressed group keeps the file small
                sup_groups.add(group)
                e["trace"] = trace
                focus["SUP|" + group] = dict(signal=r.signal, entity=r.entity,
                                             name=f"{SIGNAL_NAMES.get(r.signal, r.signal)} (suppressed)",
                                             charts=focus_charts(r.signal, r.entity, r.ts, data))
        events.append(e)

    evaluation = compute(truth, log) if truth else None
    routing = sorted({s.get("source", "?") for e in events for s in (e.get("specialists") or [])})
    payload = dict(
        meta=dict(label=data.label or "LIVE DATA", marketplace=cfg.get("marketplace", ""),
                  currency=cfg.get("currency", "EUR"), start=_iso(hours.min()), end=_iso(hours.max()),
                  advertisers=int(data.advertisers.advertiser_id.nunique()),
                  sources=int(data.supply.source_id.nunique()), routing=routing,
                  generated=datetime.now().strftime("%Y-%m-%d %H:%M")),
        hours=[_iso(t) for t in hours],
        overview=dict(actual=ov_a, baseline=ov_b),
        bands=bands, events=events, focus=focus, evaluation=evaluation,
    )
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[{payload['meta']['label']}] dashboard written: {OUT}  ({len(html) / 1024:.0f} KB, "
          f"{sum(e['kind'] == 'alert' for e in events)} alerts, "
          f"{sum(e['kind'] == 'suppressed' for e in events)} suppressed detections)")
    if open_browser:
        webbrowser.open(OUT.as_uri())
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    build(ap.parse_args().open)
