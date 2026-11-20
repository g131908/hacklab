"""Record a human decision for a pending alert (the GATE, outside the replay).

python src/decide.py                         list pending decisions
python src/decide.py <anomaly_id> A|B        decide (staged only: nothing is executed)
python src/decide.py <anomaly_id> A --by Marco
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor.state import State  # noqa: E402

STATE_DIR = Path(__file__).resolve().parent.parent / ".agents" / "monitoring"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anomaly_id", nargs="?")
    ap.add_argument("option", nargs="?", choices=["A", "B", "a", "b"])
    ap.add_argument("--by", default="human (decide.py)")
    args = ap.parse_args()
    state = State(STATE_DIR)
    pending = {k: a for k, a in state.anomalies.items() if a.get("decision") == "pending"}

    if not args.anomaly_id:
        if not pending:
            print("No pending decisions.")
        for k, a in pending.items():
            print(f"{k}  SEV {a['severity']}  {a['title']}\n   A) {a['options']['A']}\n   B) {a['options']['B']}"
                  f"\n   recommended: {a['recommended']}")
        return 0
    rec = state.anomalies.get(args.anomaly_id)
    if rec is None:
        print(f"Unknown anomaly id {args.anomaly_id}. Pending: {', '.join(pending) or 'none'}")
        return 1
    if rec.get("decision") != "pending":
        print(f"{args.anomaly_id} has no pending decision (current: {rec.get('decision', 'none')}).")
        return 1
    if not args.option:
        print("Give an option: A or B")
        return 1
    opt = args.option.upper()
    rec["decision"] = opt
    state.save()
    state.log_decision(args.anomaly_id, opt, rec["options"][opt], args.by)
    print(f"Decision {opt} recorded for {args.anomaly_id}: {rec['options'][opt]}\n"
          "Staged only - nothing was executed (Tier 2 needs a human to carry it out).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
