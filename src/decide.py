"""Record a human decision for a pending alert (the GATE, outside the replay).

python src/decide.py                         list pending decisions
python src/decide.py <anomaly_id> A|B        decide (staged only: nothing is executed)
python src/decide.py <anomaly_id> A --by Marco
python src/decide.py --history               show the decision memory the agent learns from
python src/decide.py --forget                clear the decision memory (and its proposals)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor.learning import DecisionMemory  # noqa: E402
from monitor.state import State  # noqa: E402

STATE_DIR = Path(__file__).resolve().parent.parent / ".agents" / "monitoring"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anomaly_id", nargs="?")
    ap.add_argument("option", nargs="?", choices=["A", "B", "a", "b"])
    ap.add_argument("--by", default="human (decide.py)")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--forget", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load((STATE_DIR / "config.yaml").read_text(encoding="utf-8"))
    mem = DecisionMemory(STATE_DIR, cfg)

    if args.forget:
        mem.forget()
        print("Decision memory cleared (decision-history.jsonl, learning-proposals.md).")
        return 0
    if args.history:
        hist = mem.load()
        if not hist:
            print("Decision memory is empty.")
        for h in hist:
            tag = "human" if h["human"] else "not counted"
            print(f"{h['decided_at']}  {h['anomaly_id']:<26} chose {h['decision']} (recommended {h['recommended']})"
                  f"  by {h['decided_by']} [{tag}]")
        for p in mem.proposals():
            print(f"PROPOSAL: {p['text']}")
        return 0

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
    mem.record(anomaly_id=args.anomaly_id, signal=rec["signal"], entity=rec["entity"], severity=rec["severity"],
               recommended=rec["recommended"], decision=opt, decided_by=args.by, data_ts=rec["first_seen"])
    print(f"Decision {opt} recorded for {args.anomaly_id}: {rec['options'][opt]}\n"
          "Staged only - nothing was executed (Tier 2 needs a human to carry it out).")
    if opt != rec["recommended"]:
        print(f"Learned: you overrode the recommendation ({rec['recommended']} -> {opt}). "
              f"The next {rec['signal']} alert on {rec['entity']} will recommend {opt}.")
    for p in mem.proposals():
        print(f"PROPOSAL (not applied): {p['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
