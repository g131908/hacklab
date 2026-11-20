"""One live monitoring tick: run the pipeline once and keep state between ticks (no reset).
Meant to be called on a schedule (Claude Code /loop, cron, Task Scheduler).

python src/live.py                 mode from config (live.mode)
python src/live.py --mode demo     next hour(s) of the demo data, cursor in .agents/monitoring/live-cursor.json
python src/live.py --mode live     newest complete data hour; freshness checked against the wall clock
python src/live.py --dashboard     also rebuild dashboard/index.html after the tick
python src/live.py --reset-demo    start the demo loop from live.demo_start with clean state

Never decides on its own: SEV >= 4 alerts are left PENDING in decisions.md
(decide with: python src/decide.py <anomaly_id> A|B). Exit code 0 also when alerts fire.
"""
from __future__ import annotations

import argparse
import json
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
CURSOR = ROOT / ".agents" / "monitoring" / "live-cursor.json"


def pending_decider(al: dict):
    return None  # nobody at a terminal during a scheduled tick


pending_decider.mode = "scheduled tick (no human) -> pending"


def load_cursor() -> dict:
    return json.loads(CURSOR.read_text(encoding="utf-8")) if CURSOR.exists() else {}


def save_cursor(cur: dict) -> None:
    CURSOR.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"])
    ap.add_argument("--hours", type=int, help="demo: hours to process this tick (default live.demo_hours_per_tick)")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--reset-demo", action="store_true")
    ap.add_argument("--routing", choices=["auto", "rules", "llm"])
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    lcfg = cfg.get("live", {})
    mode = args.mode or lcfg.get("mode", "demo")
    state = State(CFG.parent)
    if state.killed():
        print("KILL switch present (.agents/monitoring/KILL) - tick skipped, nothing done.")
        state.log_run(pd.Timestamp.now().floor("h"), 0, 0, 0, 0, note="killed (KILL file present)")
        return 0

    data = make_source(cfg, ROOT).load()
    label = data.label or "LIVE DATA"
    router = Router(cfg, args.routing)
    cur = load_cursor()

    if args.reset_demo or (mode == "demo" and cur.get("mode") != "demo"):
        state.reset()
        cur = dict(mode="demo", next=str(pd.Timestamp(lcfg.get("demo_start", data.advertisers.ts.min()))))
        print(f"[{label}] demo loop (re)started at {cur['next']} with clean state")

    if mode == "demo":
        start = pd.Timestamp(cur["next"])
        n = args.hours or int(lcfg.get("demo_hours_per_tick", 1))
        hours = [h for h in pd.date_range(start, periods=n, freq="h") if h <= data.latest_ts]
        if not hours:
            print(f"[{label}] demo data exhausted (last hour {data.latest_ts}). "
                  "Restart with: python src/live.py --reset-demo")
            return 0
        walls = [None] * len(hours)
    else:
        wall = pd.Timestamp.now(tz=cfg.get("timezone", "Europe/Berlin")).tz_localize(None).floor("h")
        latest = data.latest_ts
        last = pd.Timestamp(cur["last"]) if cur.get("mode") == "live" and cur.get("last") else None
        if last is not None and latest is not None and latest <= last:
            hours, walls = [latest], [wall]  # no new data: run anyway so the freshness check can fire
            print(f"[{label}] no new data since {last}; re-checking freshness at {wall}")
        else:
            hours, walls = [latest], [wall]

    print(f"[{label}] tick mode={mode} routing={router.mode} hours={hours[0]}"
          + (f"..{hours[-1]}" if len(hours) > 1 else ""))
    total = dict(alert=0, digest=0, suppressed=0, resolved=0, pipeline=0, error=0)
    for now, wall in zip(hours, walls):
        shown = []

        def notify(al: dict) -> None:
            if al["type"] in ("alert", "digest"):
                shown.append(al)

        events = run_once(data.until(now), now, cfg, state, notify, pending_decider, router, wall=wall)
        for ev in events:
            total[ev["type"]] = total.get(ev["type"], 0) + 1
            if ev["type"] == "killed":
                print("KILL switch present - stopping.")
                return 0
            if ev["type"] in ("pipeline", "error"):
                print(f"\n[SEV 3] {ev['signal']} at {now}: {ev['text']}")
        for al in shown:
            print("\n" + "=" * 90 + "\n" + al["text"])
            if al["type"] == "alert" and al["severity"] >= 4:
                aid = next(e["anomaly_id"] for e in events if e.get("text") == al["text"])
                print(f"GATE: decision PENDING (recommended {al['recommended']}). "
                      f"Decide: python src/decide.py {aid} A|B")
            print("=" * 90)
        sup = [e for e in events if e["type"] == "suppressed"]
        if sup:
            reasons = sorted({e["reason"] for e in sup})
            print(f"{now:%a %d.%m %H:%M}  suppressed {len(sup)} raw detection(s): {'; '.join(reasons)}")
        for e in events:
            if e["type"] == "resolved":
                print(f"{now:%a %d.%m %H:%M}  resolved {e['anomaly_id']}")

    if mode == "demo":
        cur["next"] = str(hours[-1] + pd.Timedelta(hours=1))
    else:
        cur = dict(mode="live", last=str(hours[-1]))
    save_cursor(cur)

    pending = [a for a in state.anomalies.values() if a.get("decision") == "pending" and a["status"] == "open"]
    quiet = not any(total[k] for k in ("alert", "digest", "pipeline", "error"))
    print(f"\n[{label}] tick done: alerts={total['alert']} suppressed={total['suppressed']} "
          f"resolved={total['resolved']} | open anomalies={sum(a['status'] == 'open' for a in state.anomalies.values())} "
          f"| pending decisions={len(pending)}" + (" | checked, nothing to do" if quiet else ""))
    for a in pending:
        print(f"  PENDING {a['anomaly_id']} (SEV {a['severity']}, recommended {a.get('recommended')}): "
              f"python src/decide.py {a['anomaly_id']} A|B")
    if mode == "demo":
        print(f"  next demo hour: {cur['next']}")
    if args.dashboard:
        import dashboard
        dashboard.build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
