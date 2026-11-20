"""Replay data hour by hour through the monitoring pipeline.

python src/replay.py              interactive: asks A/B for every SEV >= 4 alert
python src/replay.py --auto A     non-interactive, always choose A (or B / rec = recommended)
python src/replay.py --quiet      only alerts and suppressions, no daily progress lines
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor.loader import make_source  # noqa: E402
from monitor.pipeline import run_once  # noqa: E402
from monitor.routing import Router  # noqa: E402
from monitor.state import State  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / ".agents" / "monitoring" / "config.yaml"


def make_decider(auto: str | None):
    def decide(al: dict) -> str:
        if auto:
            choice = al["recommended"] if auto == "rec" else auto
            print(f">>> Decision (auto): {choice}")
            return choice
        while True:
            try:
                ans = input(f">>> Your decision [A/B] (Enter = recommended {al['recommended']}): ")
            except EOFError:  # no interactive terminal
                print(f"(no terminal input) -> recommended {al['recommended']}")
                return al["recommended"]
            ans = ans.replace("﻿", "").strip().upper() or al["recommended"]
            if ans in ("A", "B"):
                return ans
            print("    please type A or B")
    return decide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", choices=["A", "B", "rec"], help="answer decisions automatically")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--routing", choices=["auto", "rules", "llm"],
                    help="specialist execution (default: routing.mode from config)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    data = make_source(cfg, ROOT).load()
    state = State(CFG.parent)
    if state.killed():
        print("KILL switch present (.agents/monitoring/KILL) - exiting, nothing done.")
        return
    state.reset()  # every replay starts from clean state

    label = data.label or "LIVE DATA"
    hours = pd.date_range(min(data.advertisers.ts.min(), data.supply.ts.min()),
                          data.latest_ts, freq="h")
    print(f"=== REPLAY [{label}] {hours[0]} .. {hours[-1]} ({len(hours)} hours) ===")
    router = Router(cfg, args.routing)
    print(f"    {data.advertisers.advertiser_id.nunique()} advertisers, "
          f"{data.supply.source_id.nunique()} supply sources; config: {CFG.relative_to(ROOT)}")
    print(f"    specialist routing: {router.mode}"
          + (" (Claude API, falls back to rules on error)" if router.mode == "llm" else
             " (rule-based; set ANTHROPIC_API_KEY or use --routing llm for Claude)") + "\n")
    for err in router.llm_errors:
        print(f"    note: {err}")

    decide = make_decider(args.auto)

    def notify(al: dict) -> None:
        if al["type"] == "page":
            print(f"\n!!! {al['text']}")
            return
        print(f"\n{'=' * 100}\n{al['text']}")
        if al["type"] == "alert" and al["severity"] >= 4:
            print("-" * 100)

    shown_suppressed: set = set()
    totals = dict(alert=0, digest=0, suppressed=0, resolved=0)
    for now in hours:
        if now.hour == 0 and not args.quiet:
            print(f"--- {now:%a %d.%m.%Y} ---")
        events = run_once(data.until(now), now, cfg, state, notify, decide, router)
        for ev in events:
            t = ev["type"]
            totals[t] = totals.get(t, 0) + 1
            if t == "killed":
                print(f"{now}: KILL switch present - stopping replay.")
                return
            if t in ("alert", "digest"):
                if ev.get("decision"):
                    print(f"GATE: decision {ev['decision']} recorded -> staged, NOT executed "
                          "(Tier 2 needs human action)")
                elif t == "alert":
                    print("GATE: SEV 3 -> recommendation staged, no decision required now")
                else:
                    print("GATE: SEV 1-2 -> daily digest only, no interrupt")
                print("=" * 100 + "\n")
            elif t == "suppressed":
                key = (ev["signal"], ev["reason"], now.date())
                if key not in shown_suppressed:
                    shown_suppressed.add(key)
                    print(f"{now:%d.%m %H:%M}  SUPPRESSED {ev['signal']} {ev['title']} "
                          f"({ev['entity']} and others) -> {ev['reason']}; "
                          f"checks: {', '.join(f'{k} {v}' for k, v in ev['checks'].items())}")
            elif t == "resolved":
                print(f"{now:%d.%m %H:%M}  RESOLVED {ev['anomaly_id']} (back in range)")
            elif t in ("pipeline", "error"):
                print(f"{now:%d.%m %H:%M}  [SEV 3] {ev['signal']}: {ev['text']}")

    print(f"\n=== REPLAY DONE [{label}] alerts={totals['alert']} digest={totals['digest']} "
          f"suppressed_by_selfcheck={totals['suppressed']} raw detections resolved={totals['resolved']}")
    if router.llm_errors:
        print(f"    LLM specialist errors (fell back to rules): {len(router.llm_errors)}")
        for err in router.llm_errors[:5]:
            print(f"      - {err}")
    print("    log: .agents/monitoring/alerts.jsonl | decisions: .agents/monitoring/decisions.md")
    print("    next: python src/evaluate.py")


if __name__ == "__main__":
    main()
