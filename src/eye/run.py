"""Run the Eye: replay hours through the monitor (demo) or tick once (live loop).

python src/eye/run.py --from "2026-11-16 00:00" --to "2026-11-21 23:00"     replay a range (fresh state)
python src/eye/run.py --tick                                               one tick at the next hour (cursor)
python src/eye/run.py --tick --hours 6                                     six hours in one tick
python src/eye/run.py ... --llm                                            Claude plans the next query and narrates
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye.monitor import STATE, Monitor  # noqa: E402
from eye.store import Store  # noqa: E402

CURSOR = STATE / "cursor.json"
DEFAULT_START = "2026-11-16 00:00"


def print_report(r: dict, quiet: bool = False):
    now = pd.Timestamp(r["now"])
    shown = [i for i in r["investigations"] if i["new"] or i.get("escalated")]
    busy = shown or r["resolved"] or r["notices"]
    if quiet and not busy:
        return
    head = f"[{r['label']}] {now:%a %d.%m %H:%M}  watched {r['watched']}  "
    head += (f"anomalies {len(r['anomalies']) + len(r['known'])} (already explained {len(r['known'])}, "
             f"watching {r.get('candidates', 0)} candidates)  {r['ms']} ms")
    print(head)
    for n in r["notices"]:
        print(f"   DATA   {n['text']}")
    silent = len(r["investigations"]) - len(shown)
    if silent:
        print(f"   ...    {silent} more symptom(s) traced to open incidents (updated silently)")
    for i in shown:
        tag = {"fault": "ALERT", "innocent": "INFO ", "data_incomplete": "DATA ", "unexplained": "ALERT",
               "change": "NOTE "}.get(i["classification"], "NOTE ")
        print(f"   {tag}  {i['incident']} {'NEW' if i['new'] else 'UPDATE'}: {i['headline']}")
        print(f"          chain {' <- '.join(i['chain'])} | {i['steps']} queries | supported: {i['supported'] or '-'}")
        for u in i["uncertainty"]:
            print(f"          ? {u}")
        rv = i.get("review")
        if rv and "summary" in rv:
            print(f"          Claude ({rv['verdict']}{'' if rv['grounded'] else ', UNGROUNDED refs dropped'}): {rv['summary']}")
            if rv["verdict"] == "challenge":
                print(f"          Claude challenges: {rv['challenge']}")
            for c in rv["next_checks"][:2]:
                print(f"          next check: {c['check']}")
        elif rv:
            print(f"          Claude review failed ({rv['error']}); the ledger above is complete")
    for x in r["resolved"]:
        print(f"   RESOLVED {x['incident']}: {x['headline']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--tick", action="store_true")
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--llm", action="store_true", help="Claude plans next queries and writes the narrative")
    ap.add_argument("--quiet", action="store_true", help="only print ticks where something happened")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    full = Store.load()
    from eye import llm
    reviewer, since = None, llm.log_lines()
    if args.llm:
        if llm.available():
            from eye import agent_llm
            reviewer = agent_llm.review
            print(f"Claude review on ({llm.MODEL}, max 1 call per incident)")
        else:
            print("Claude review off: no ANTHROPIC_API_KEY in env or .env")
    mon = Monitor(full, reviewer=reviewer)
    last = full.tables["facts"].ts.max()
    if args.tick:
        cur = json.loads(CURSOR.read_text()) if CURSOR.exists() else {"next": DEFAULT_START}
        if not CURSOR.exists():
            mon.reset()
        hours = pd.date_range(cur["next"], periods=args.hours, freq="h")
    else:
        mon.reset()
        hours = pd.date_range(args.start or DEFAULT_START, args.end or last, freq="h")
    hours = [h for h in hours if h <= last]
    if not hours:
        print("demo data exhausted; delete .agents/eye/cursor.json to restart")
        return
    for h in hours:
        print_report(mon.tick(h), quiet=args.quiet)
    CURSOR.write_text(json.dumps({"next": str(hours[-1] + pd.Timedelta(hours=1))}))
    opn = [i for i in mon.incidents.values() if i.status == "open"]
    print(f"\nopen incidents: {len(opn)}  |  all incidents: {len(mon.incidents)}  |  state: {STATE}")
    print(llm.cost_report(since))


if __name__ == "__main__":
    main()
