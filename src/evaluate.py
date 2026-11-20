"""Compare alerts raised by the replay with the injected ground truth.

python src/evaluate.py   (run python src/replay.py first)
Output: injected / detected / false alarms / time-to-detect, per scenario and in total.
`compute()` is shared with the dashboard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INJECTED = ROOT / "data" / "injected.json"
ALERTS = ROOT / ".agents" / "monitoring" / "alerts.jsonl"
GRACE = pd.Timedelta(hours=24)  # detection may come shortly after the injected window ends


def load_log(path: Path = ALERTS) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    log = pd.DataFrame(rows)
    if log.empty:
        log = pd.DataFrame(columns=["event", "ts", "signal", "entity", "anomaly_id", "severity", "realert"])
    if "realert" not in log:
        log["realert"] = False
    log["ts"] = pd.to_datetime(log["ts"])
    return log


def load() -> tuple[dict, pd.DataFrame]:
    for p, hint in ((INJECTED, "python src/generate_demo_data.py"), (ALERTS, "python src/replay.py")):
        if not p.exists():
            sys.exit(f"missing {p.relative_to(ROOT)} - run: {hint}")
    return json.loads(INJECTED.read_text(encoding="utf-8")), load_log()


def matches(log: pd.DataFrame, sc: dict) -> pd.Series:
    start, end = pd.Timestamp(sc["start"]), pd.Timestamp(sc["end"])
    m = log.signal.isin(sc["accept_signals"]) & (log.ts >= start) & (log.ts < end + GRACE)
    if sc["entity"] != "*":
        m &= log.entity == sc["entity"]
    return m


def compute(truth: dict, log: pd.DataFrame) -> dict:
    alerts = log[log.event.isin(["alert", "digest"])]
    first_alerts = alerts[~alerts.realert.fillna(False).astype(bool)]
    suppressed = log[log.event == "suppressed"]
    rows, matched_ids, ttds = [], set(), []
    for sc in truth["scenarios"]:
        start = pd.Timestamp(sc["start"])
        hit = first_alerts[matches(first_alerts, sc)].sort_values("ts")
        sup = suppressed[matches(suppressed, sc)]
        row = dict(id=sc["id"], signal=sc["signal"], entity=sc["entity"], start=sc["start"], end=sc["end"],
                   expected=sc["expected"], description=sc.get("description", ""),
                   detected_ts=None, severity=None, ttd_h=None, false_alarms=0,
                   suppressed_count=int(len(sup)), suppressed_entities=int(sup.entity.nunique()))
        if sc["expected"] == "detect" and len(hit):
            first = hit.iloc[0]
            matched_ids |= set(hit.anomaly_id)
            row.update(detected_ts=str(first.ts), severity=int(first.severity),
                       ttd_h=(first.ts - start) / pd.Timedelta(hours=1))
            ttds.append(row["ttd_h"])
        elif sc["expected"] == "suppress":
            row["false_alarms"] = int(hit.anomaly_id.nunique())
        rows.append(row)
    unmatched = first_alerts[~first_alerts.anomaly_id.isin(matched_ids)]
    raised = int(first_alerts.anomaly_id.nunique())
    n_detect = sum(r["expected"] == "detect" for r in rows)
    detected = sum(r["detected_ts"] is not None for r in rows)
    return dict(
        scenarios=rows,
        unmatched=[dict(anomaly_id=r.anomaly_id, ts=str(r.ts), severity=int(r.severity))
                   for _, r in unmatched.iterrows()],
        summary=dict(injected=n_detect, detected=detected, raised=raised,
                     false_alarms=int(unmatched.anomaly_id.nunique()),
                     precision=(raised - unmatched.anomaly_id.nunique()) / raised if raised else None,
                     ttd_median=float(pd.Series(ttds).median()) if ttds else None,
                     ttd_min=min(ttds) if ttds else None, ttd_max=max(ttds) if ttds else None,
                     realerts=int(alerts.realert.fillna(False).astype(bool).sum()),
                     false_alarm_scenario_suppressed=all(r["false_alarms"] == 0 for r in rows
                                                         if r["expected"] == "suppress")),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    truth, log = load()
    res = compute(truth, log)
    s = res["summary"]

    print(f"=== EVALUATION [{truth.get('label', '')}] {truth['start']} .. {truth['end']} ===\n")
    print(f"{'scenario':<18}{'signal':<8}{'entity':<8}{'injected at':<18}{'expected':<10}{'result':<46}time-to-detect")
    for r in res["scenarios"]:
        start = pd.Timestamp(r["start"])
        if r["expected"] == "detect":
            if r["detected_ts"]:
                result = f"DETECTED {pd.Timestamp(r['detected_ts']):%d.%m %H:%M} SEV {r['severity']}"
                ttd_s = f"{r['ttd_h']:.0f}h"
            else:
                result, ttd_s = "MISSED", "-"
        else:
            result = (f"FALSE ALARM x{r['false_alarms']}" if r["false_alarms"] else
                      f"SUPPRESSED ({r['suppressed_count']} raw detections, {r['suppressed_entities']} accounts)")
            ttd_s = "n/a"
        print(f"{r['id']:<18}{r['signal']:<8}{r['entity']:<8}{start:%d.%m %H:%M}       "
              f"{r['expected']:<10}{result:<46}{ttd_s}")

    print("\n--- summary ---")
    print(f"injected (to detect):   {s['injected']}")
    print(f"detected:               {s['detected']}/{s['injected']}")
    prec = f"{s['precision']:.0%}" if s["precision"] is not None else "n/a"
    print(f"false alarms:           {s['false_alarms']}  (of {s['raised']} alerts raised; precision {prec})")
    print(f"false-alarm scenario:   {'suppressed by self-check' if s['false_alarm_scenario_suppressed'] else 'NOT suppressed'}")
    if s["ttd_median"] is not None:
        print(f"time-to-detect:         median {s['ttd_median']:.0f}h, min {s['ttd_min']:.0f}h, max {s['ttd_max']:.0f}h")
    print(f"re-alerts (sev raised): {s['realerts']}")
    if res["unmatched"]:
        print("\nunmatched alerts:")
        for u in res["unmatched"]:
            print(f"  {u['ts']}  {u['anomaly_id']}  SEV {u['severity']}")
    print("\nNote: S6 (creative fatigue) needs a 3-day CTR trend by design, so its time-to-detect is days, not hours.")


if __name__ == "__main__":
    main()
