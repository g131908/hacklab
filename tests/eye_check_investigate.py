"""Run the investigator on each scenario's trigger and print the trail."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eye.detect import measure  # noqa: E402
from eye.graph import MetricGraph  # noqa: E402
from eye.investigate import Investigator  # noqa: E402
from eye.store import Store  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
full = Store.load()
only = sys.argv[1:]
cases = [
    ("margin", "2026-11-16 15:00", "orders", {"country": "NL"}),
    ("payment", "2026-11-17 16:00", "gmv", {}),
    ("campaign", "2026-11-18 21:00", "conversion_rate", {}),
    ("search", "2026-11-19 12:00", "orders", {"country": "DE"}),
    ("stockout", "2026-11-20 14:00", "cart_rate", {"category": "electronics"}),
    ("late", "2026-11-21 23:00", "orders", {}),
]
for name, ts, metric, seg in cases:
    if only and name not in only:
        continue
    st = full.asof(ts)
    g = MetricGraph.build(st, validate=False)
    for d in g.decomps:
        d.status = "valid"      # validated once on full data (show_graph.py)
    trig = measure(st, metric, seg, pd.Timestamp(ts))
    inv = Investigator(st, g, ts).run(trig)
    print(f"\n=================== {name} @ {ts}: {inv.classification.upper()}")
    print("HEADLINE:", inv.headline)
    for s in inv.steps:
        print(f"  {s.id:<4}{s.kind:<10}{s.title[:70]:<72}| {s.decision[:150]}")
    print("  chain:", " <- ".join(inv.chain), "| segment:", inv.segment, "| onset:", inv.onset)
    for h in sorted(inv.hypotheses, key=lambda h: -h.score):
        print(f"  [{h.status:<10}] {h.score:>4} {h.label:<15} {h.cause}")
    print("  evidence level:", inv.confidence_label, "| stop:", inv.stop_reason, "| branches:", inv.branches)
    for c in inv.changes:
        print(f"  change: {c['name']} (owner {c['owner']}, deployed {c['deployed_at'][5:16]}, exposed "
              f"{c['first_exposure_at'][5:16]}) -> " + "; ".join(f"{k}: {v['status']}" for k, v in c["verdicts"].items()))
    for u in inv.uncertainty:
        print("  ? ", u)
