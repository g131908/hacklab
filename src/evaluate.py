"""Compare alerts raised by the replay with the injected ground truth.

python src/evaluate.py   (run python src/replay.py first)
Output: injected / detected / false alarms / time-to-detect, per scenario and in total.
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


def load() -> tuple[dict, pd.DataFrame]:
    for p, hint in ((INJECTED, "python src/generate_demo_data.py"), (ALERTS, "python src/replay.py")):
        if not p.exists():
            sys.exit(f"missing {p.relative_to(ROOT)} - run: {hint}")
    truth = json.loads(INJECTED.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in ALERTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    log = pd.DataFrame(rows)
    if log.empty:
        log = pd.DataFrame(columns=["event", "ts", "signal", "entity", "anomaly_id"])
    log["ts"] = pd.to_datetime(log["ts"])
    return truth, log


def matches(log: pd.DataFrame, sc: dict) -> pd.Series:
    start, end = pd.Timestamp(sc["start"]), pd.Timestamp(sc["end"])
    m = log.signal.isin(sc["accept_signals"]) & (log.ts >= start) & (log.ts < end + GRACE)
    if sc["entity"] != "*":
        m &= log.entity == sc["entity"]
    return m


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    truth, log = load()
    alerts = log[log.event.isin(["alert", "digest"])]
    first_alerts = alerts[~alerts.get("realert", pd.Series(False, index=alerts.index)).fillna(False).astype(bool)]
    suppressed = log[log.event == "suppressed"]

    print(f"=== EVALUATION [{truth.get('label', '')}] {truth['start']} .. {truth['end']} ===\n")
    print(f"{'scenario':<18}{'signal':<8}{'entity':<8}{'injected at':<18}{'expected':<10}{'result':<46}time-to-detect")
    matched_ids: set = set()
    detected = n_detect = ttds = 0
    ttd_list = []
    holiday_false = 0
    for sc in truth["scenarios"]:
        start = pd.Timestamp(sc["start"])
        hit = first_alerts[matches(first_alerts, sc)]
        if sc["expected"] == "detect":
            n_detect += 1
            if len(hit):
                detected += 1
                matched_ids |= set(hit.anomaly_id)
                first = hit.sort_values("ts").iloc[0]
                ttd = (first.ts - start) / pd.Timedelta(hours=1)
                ttd_list.append(ttd)
                result = f"DETECTED {first.ts:%d.%m %H:%M} SEV {int(first.severity)}"
                ttd_s = f"{ttd:.0f}h"
            else:
                result, ttd_s = "MISSED", "-"
        else:  # expected suppression (false-alarm scenario)
            sup = suppressed[matches(suppressed, sc)]
            holiday_false = hit.anomaly_id.nunique()
            if holiday_false:
                result = f"FALSE ALARM x{holiday_false}"
            else:
                result = f"SUPPRESSED ({len(sup)} raw detections, {sup.entity.nunique()} accounts)"
            ttd_s = "n/a"
        print(f"{sc['id']:<18}{sc['signal']:<8}{sc['entity']:<8}{start:%d.%m %H:%M}       "
              f"{sc['expected']:<10}{result:<46}{ttd_s}")

    raised = first_alerts.anomaly_id.nunique()
    false_alarms = first_alerts[~first_alerts.anomaly_id.isin(matched_ids)].anomaly_id.nunique()
    precision = (raised - false_alarms) / raised if raised else float("nan")
    print("\n--- summary ---")
    print(f"injected (to detect):   {n_detect}")
    print(f"detected:               {detected}/{n_detect}")
    print(f"false alarms:           {false_alarms}  (of {raised} alerts raised; precision {precision:.0%})")
    print(f"false-alarm scenario:   {'suppressed by self-check' if not holiday_false else 'NOT suppressed'}")
    if ttd_list:
        print(f"time-to-detect:         median {pd.Series(ttd_list).median():.0f}h, "
              f"min {min(ttd_list):.0f}h, max {max(ttd_list):.0f}h")
    re = alerts.get("realert", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    print(f"re-alerts (sev raised): {re}")
    if false_alarms:
        print("\nunmatched alerts:")
        for _, r in first_alerts[~first_alerts.anomaly_id.isin(matched_ids)].iterrows():
            print(f"  {r.ts}  {r.anomaly_id}  SEV {r.severity}")
    print("\nNote: S6 (creative fatigue) needs a 3-day CTR trend by design, so its time-to-detect is days, not hours.")


if __name__ == "__main__":
    main()
