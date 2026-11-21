"""Eye dashboard: one self-contained HTML file (no network) with the metric graph, the incident feed, each
investigation's evidence ledger and the scorecard.

python src/eye/dashboard.py            build from the current state (.agents/eye/) -> dashboard/eye.html
python src/eye/dashboard.py --replay   replay all hours first (no Claude, free), then build
python src/eye/dashboard.py --open     also open it in the browser
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye.evaluate import score  # noqa: E402
from eye.graph import MetricGraph  # noqa: E402
from eye.monitor import STATE, Monitor  # noqa: E402
from eye.store import DATA, Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
OUT = ROOT / "dashboard" / "eye.html"


def build(replay: bool = False, open_browser: bool = False) -> Path:
    full = Store.load()
    mon = Monitor(full)
    if replay:
        mon.reset()
        for h in full.hours:
            mon.tick(h)
    g = MetricGraph.build(full)
    truth_p = DATA / "injected.json"
    truth = json.loads(truth_p.read_text(encoding="utf-8")) if truth_p.exists() else None
    incidents = mon.all_incidents()
    card = score(incidents, truth) if truth else None
    gj = g.to_json()
    payload = dict(
        meta=dict(label=full.label, start=str(full.hours[0]), end=str(full.hours[-1]),
                  generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                  metrics=len(gj["nodes"]), edges=len(gj["edges"])),
        graph=dict(nodes=gj["nodes"], edges=[e for e in gj["edges"] if e["type"] != "breakdown"], roots=gj["roots"],
                   log=gj["log"], breakdowns={n["name"]: n["dims"] for n in gj["nodes"]}),
        incidents=[asdict(i) for i in incidents],
        scorecard=card,
    )
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, default=str, separators=(",", ":")).replace("</", "<\\/"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[{full.label}] Eye dashboard: {OUT} ({len(html) // 1024} KB, {len(incidents)} incidents)")
    if open_browser:
        webbrowser.open(OUT.as_uri())
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    build(a.replay, a.open)
