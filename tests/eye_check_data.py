"""Sanity check of the Eye DEMO DATA: do the injected scenarios show up as intended?"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "eye"
f = pd.read_csv(D / "facts.csv", parse_dates=["ts"])
inv = pd.read_csv(D / "inventory.csv", parse_dates=["ts"])


def rate(df, num, den):
    return df[num].sum() / df[den].sum()


def window(df, a, b):
    return df[(df.ts >= a) & (df.ts < b)]


def lastweek(df, a, b):
    return window(df, pd.Timestamp(a) - pd.Timedelta(days=7), pd.Timestamp(b) - pd.Timedelta(days=7))


checks = [
    ("payment DE android", lambda d: d[(d.country == "DE") & (d.platform == "android")], "orders", "payment_attempts",
     "2026-11-17 14:00", "2026-11-18 02:00"),
    ("payment DE ios (control)", lambda d: d[(d.country == "DE") & (d.platform == "ios")], "orders", "payment_attempts",
     "2026-11-17 14:00", "2026-11-18 02:00"),
    ("FR conversion", lambda d: d[d.country == "FR"], "orders", "sessions", "2026-11-18 18:00", "2026-11-20 00:00"),
    ("FR orders vs sessions", lambda d: d[d.country == "FR"], "orders", None, "2026-11-18 18:00", "2026-11-20 00:00"),
    ("search ctr DE", lambda d: d[d.country == "DE"], "product_views", "searches", "2026-11-19 09:00", "2026-11-19 21:00"),
    ("cart rate electronics", lambda d: d[d.category == "electronics"], "add_to_carts", "product_views",
     "2026-11-20 12:00", "2026-11-21 00:00"),
    ("views electronics (demand)", lambda d: d[d.category == "electronics"], "product_views", None,
     "2026-11-20 12:00", "2026-11-21 00:00"),
    ("orders total last 3h (late data)", lambda d: d, "orders", None, "2026-11-21 21:00", "2026-11-22 00:00"),
]
for name, sel, num, den, a, b in checks:
    cur, lw = window(sel(f), a, b), lastweek(sel(f), a, b)
    if den:
        x, y = rate(cur, num, den), rate(lw, num, den)
        print(f"{name:<34} now {x:7.3%}  last week {y:7.3%}  ({x / y - 1:+.0%})")
    else:
        x, y = cur[num].sum(), lw[num].sum()
        print(f"{name:<34} now {x:9,.0f}  last week {y:9,.0f}  ({x / y - 1:+.0%})")
e = inv[inv.category == "electronics"]
cur, lw = window(e, "2026-11-20 12:00", "2026-11-21 00:00"), lastweek(e, "2026-11-20 12:00", "2026-11-21 00:00")
print(f"{'in-stock electronics':<34} now {rate(cur, 'in_stock_skus', 'listed_skus'):7.1%}  last week "
      f"{rate(lw, 'in_stock_skus', 'listed_skus'):7.1%}")
