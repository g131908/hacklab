"""Detection at scenario times (and a quiet time) on the clocked store."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eye.detect import measure  # noqa: E402
from eye.store import Store  # noqa: E402

full = Store.load()
cases = [
    ("2026-11-16 15:00", "quiet", [("orders", {}), ("conversion_rate", {}), ("orders", {"country": "DE"})]),
    ("2026-11-17 16:00", "payment", [("orders", {}), ("gmv", {}), ("orders", {"country": "DE"}),
                                     ("payment_success_rate", {"country": "DE", "platform": "android"})]),
    ("2026-11-18 21:00", "campaign", [("conversion_rate", {}), ("conversion_rate", {"country": "FR"}),
                                      ("orders", {"country": "FR"}), ("sessions", {"country": "FR"})]),
    ("2026-11-19 12:00", "search", [("orders", {}), ("orders", {"country": "DE"}), ("search_ctr", {"country": "DE"}),
                                    ("search_ctr", {"country": "FR"})]),
    ("2026-11-20 14:00", "stockout", [("orders", {}), ("orders", {"category": "electronics"}),
                                      ("cart_rate", {"category": "electronics"}),
                                      ("in_stock_rate", {"category": "electronics"})]),
    ("2026-11-21 23:00", "late data", [("orders", {}), ("sessions", {})]),
]
for ts, name, checks in cases:
    st = full.asof(ts)
    print(f"--- {ts} {name}")
    for m, seg in checks:
        r = measure(st, m, seg, pd.Timestamp(ts))
        print(f"  {r.status:<11} {r.short()}  vol={r.volume:,.0f} comp={r.completeness:.2f}")
