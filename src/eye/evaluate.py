"""Scorecard: replay the whole data period, then compare what the agent did with data/eye/injected.json.

python src/eye/evaluate.py            replay all hours (no Claude, costs nothing) and score
python src/eye/evaluate.py --no-replay  score the incidents already in .agents/eye/incidents.json

Per scenario
  detected     an incident was opened for it inside [start, end + 6h)
  time to detect  hours from scenario start to the incident opening
  place        the incident's segment equals the scenario's segment, and the scenario metric is on the
               incident's chain or among the symptoms the agent attributed to it (incident.explains)
  first cause  the scenario's first_cause_event is the incident's named cause (n/a for data scenarios)
  handling     fault -> ALERT incident (someone is paged); innocent -> INFO note only, nobody paged;
               data -> DATA notice only, nobody paged
False alarms = paged incidents (ALERT) that belong to no scenario. Unexplained positive NOTEs and extra
INFO/DATA notes are listed separately; they do not page anyone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye.monitor import STATE, Monitor  # noqa: E402
from eye.store import DATA, Store  # noqa: E402

GRACE = pd.Timedelta(hours=6)
PAGED = ("fault", "unexplained")
EXPECTED = {"fault": "ALERT (paged)", "innocent": "INFO only", "data": "DATA notice only"}


def kind_of(inc) -> str:
    return {"fault": "ALERT", "unexplained": "ALERT", "innocent": "INFO", "data_incomplete": "DATA"}.get(
        inc.classification, "NOTE")


def belongs(inc, sc) -> bool:
    opened = pd.Timestamp(inc.opened)
    if not (pd.Timestamp(sc["start"]) <= opened < pd.Timestamp(sc["end"]) + GRACE):
        return False
    if sc["kind"] == "data":
        return inc.classification == "data_incomplete"
    if sc.get("first_cause_event") and sc["first_cause_event"] in inc.signature:
        return True
    seg_ok = all(inc.segment.get(k) == v for k, v in sc["segment"].items())
    return seg_ok and sc["metric"] in inc.chain


def score(incidents: list, truth: dict) -> dict:
    rows, used = [], set()
    for sc in truth["scenarios"]:
        mine = sorted([i for i in incidents if belongs(i, sc)], key=lambda i: i.opened)
        used |= {i.id for i in mine}
        row = dict(id=sc["id"], kind=sc["kind"], metric=sc["metric"], segment=sc["segment"], start=sc["start"],
                   expected=EXPECTED[sc["kind"]], detected=bool(mine), incidents=[i.id for i in mine],
                   ttd_h=None, place=None, place_found=None, first_cause=None, cause_found=None,
                   handling=None, handling_found=None, problems=[])
        if not mine:
            row["problems"].append("not detected")
            rows.append(row)
            continue
        first = mine[0]
        row["ttd_h"] = (pd.Timestamp(first.opened) - pd.Timestamp(sc["start"])) / pd.Timedelta(hours=1)
        if sc["kind"] != "data":
            best = next((i for i in mine if i.segment == sc["segment"] and
                         (sc["metric"] in i.chain or any(x["metric"] == sc["metric"] for x in i.explains))), None)
            row["place"] = best is not None
            ref = best or first
            row["place_found"] = f"{ref.chain[-1] if ref.chain else '?'} in " + (
                ",".join(f"{k}={v}" for k, v in sorted(ref.segment.items())) or "total")
            if not row["place"]:
                row["problems"].append(f"place: found {row['place_found']}, expected {sc['metric']} in "
                                       + (",".join(f"{k}={v}" for k, v in sorted(sc['segment'].items())) or "total"))
            if sc.get("first_cause_event"):
                hit = next((i for i in mine if sc["first_cause_event"] in i.signature), None)
                row["first_cause"] = hit is not None
                row["cause_found"] = (hit or first).signature.split("|", 1)[-1]
                if not hit:
                    row["problems"].append(f"first cause: named '{row['cause_found']}', expected "
                                           f"'{sc['first_cause_event']}'")
        kinds = sorted({kind_of(i) for i in mine})
        row["handling_found"] = ", ".join(kinds)
        want = {"fault": {"ALERT"}, "innocent": {"INFO"}, "data": {"DATA"}}[sc["kind"]]
        row["handling"] = set(kinds) == want
        if not row["handling"]:
            row["problems"].append(f"handling: got {row['handling_found']}, expected {EXPECTED[sc['kind']]}")
        if len(mine) > 1:
            row["problems"].append(f"{len(mine)} separate incidents for one cause ({', '.join(i.id for i in mine)})")
        rows.append(row)

    other = [i for i in incidents if i.id not in used]
    fa = [dict(id=i.id, opened=i.opened, headline=i.headline) for i in other if i.classification in PAGED]
    notes = [dict(id=i.id, opened=i.opened, kind=kind_of(i), headline=i.headline) for i in other
             if i.classification not in PAGED]
    ok = [r for r in rows if not r["problems"]]
    ttd = [r["ttd_h"] for r in rows if r["kind"] == "fault" and r["ttd_h"] is not None]
    return dict(scenarios=rows, false_alarms=fa, extra_notes=notes,
                summary=dict(scenarios=len(rows), fully_correct=len(ok), detected=sum(r["detected"] for r in rows),
                             false_alarms=len(fa), extra_notes=len(notes), incidents_total=len(incidents),
                             median_ttd_faults_h=float(pd.Series(ttd).median()) if ttd else None))


def yn(v) -> str:
    return "n/a" if v is None else ("yes" if v else "NO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    full = Store.load()
    mon = Monitor(full)                       # no reviewer: no Claude calls, no cost
    if not args.no_replay:
        mon.reset()
        hours = full.hours
        print(f"[{full.label}] replaying {len(hours)} hours {hours[0]} .. {hours[-1]} (no Claude) ...")
        for h in hours:
            mon.tick(h)
    truth = json.loads((DATA / "injected.json").read_text(encoding="utf-8"))
    res = score(mon.all_incidents(), truth)

    print(f"\n=== SCORECARD [{full.label}] ===")
    print(f"{'scenario':<18}{'kind':<10}{'detected':<10}{'TTD':<7}{'place':<7}{'cause':<7}{'handling':<10}found")
    for r in res["scenarios"]:
        ttd = "-" if r["ttd_h"] is None else f"{r['ttd_h']:.0f}h"
        found = r["handling_found"] or "-"
        if r["place_found"]:
            found += f" | {r['place_found']}"
        if r["cause_found"]:
            found += f" | {r['cause_found']}"
        print(f"{r['id']:<18}{r['kind']:<10}{yn(r['detected']):<10}{ttd:<7}{yn(r['place']):<7}"
              f"{yn(r['first_cause']):<7}{yn(r['handling']):<10}{found}")
    s = res["summary"]
    print(f"\nfully correct: {s['fully_correct']}/{s['scenarios']}  |  detected: {s['detected']}/{s['scenarios']}  |  "
          f"false alarms (paged, no scenario): {s['false_alarms']}  |  extra notes: {s['extra_notes']}  |  "
          f"incidents total: {s['incidents_total']}")
    if s["median_ttd_faults_h"] is not None:
        print(f"median time to detect (faults): {s['median_ttd_faults_h']:.0f}h")
    failing = [r for r in res["scenarios"] if r["problems"]]
    print("\nstill failing:" if failing else "\nstill failing: none")
    for r in failing:
        for p in r["problems"]:
            print(f"  - {r['id']}: {p}")
    if res["false_alarms"]:
        print("\nfalse alarms:")
        for f in res["false_alarms"]:
            print(f"  - {f['id']} {f['opened']}: {f['headline']}")
    if res["extra_notes"]:
        print("\nextra notes (nobody paged):")
        for f in res["extra_notes"]:
            print(f"  - {f['id']} {f['opened']} [{f['kind']}]: {f['headline']}")
    out = STATE / "scorecard.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
