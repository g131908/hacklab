"""DEMO DATA for "Eye of the System": an Amazon-like marketplace, 14 days hourly.

Outputs in data/eye/ (all synthetic, labelled DEMO DATA):
  facts.csv         hour x country x platform x category: the shopping funnel
                    sessions, searches, zero_result_searches, product_views, add_to_carts,
                    checkouts, payment_attempts, orders, gmv, refunds
  inventory.csv     hour x category: listed_skus, in_stock_skus
  events.csv        change log: releases, campaigns, supply events (planned or not)
  completeness.csv  hour x source: share of rows that had arrived when the hour was published
  injected.json     ground truth for evaluation (faults, innocent changes, data issues)

Run: python src/eye/gen_data.py [--seed 7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eye"
START = pd.Timestamp("2026-11-08 00:00")   # Sunday; 14 days -> 2026-11-21 23:00
DAYS = 14

COUNTRIES = {"DE": 9000, "FR": 6000, "NL": 2500, "PL": 3000}          # avg sessions / hour
PLATFORMS = {"android": 0.45, "ios": 0.30, "web": 0.25}
CATEGORIES = {"electronics": 0.30, "fashion": 0.30, "home": 0.20, "grocery": 0.20}
CART_RATE = {"electronics": 0.08, "fashion": 0.12, "home": 0.10, "grocery": 0.20}
AOV = {"electronics": 180.0, "fashion": 60.0, "home": 85.0, "grocery": 40.0}
PAY_OK = {"android": 0.930, "ios": 0.950, "web": 0.955}
COGS = {"electronics": 0.70, "fashion": 0.55, "home": 0.60, "grocery": 0.72}   # cost of goods, share of list price
DISCOUNT = 0.08                       # normal discount share of list price
AD_CPS = 0.12                         # paid media cost per session (EUR)
LISTED = {"electronics": 4000, "fashion": 9000, "home": 5000, "grocery": 3000}
WEEKLY = np.array([1.00, 0.98, 0.99, 1.01, 1.05, 1.10, 1.12])  # Mon..Sun

# ground truth (end exclusive)
SCENARIOS = [
    dict(id="payment_failure", kind="fault", metric="payment_success_rate",
         segment={"country": "DE", "platform": "android"},
         start="2026-11-17 14:00", end="2026-11-18 02:00",
         cause="release payments-sdk android 7.4.0 breaks the DE local payment method",
         first_cause_event="payments-sdk android 7.4.0"),
    dict(id="fr_campaign", kind="innocent", metric="conversion_rate",
         segment={"country": "FR"}, start="2026-11-18 18:00", end="2026-11-20 00:00",
         cause="planned FR Black Week teaser campaign (email + push, no paid media): +60% low-intent traffic, "
               "conversion rate dilutes, orders and margin up",
         first_cause_event="FR Black Week teaser campaign"),
    dict(id="margin_squeeze", kind="fault", metric="contribution_margin",
         segment={"country": "NL"}, start="2026-11-16 09:00", end="2026-11-17 06:00",
         cause="local metric up, business down: a paid social push buys traffic at 1.6x cost per session and the "
               "For You discount-affinity boost pushes discounted products; orders rise, contribution margin falls",
         first_cause_event=["NL paid social push", "For You: discount-affinity boost (NL)"],
         drivers=["ad_cost", "discount_share"], trap="orders"),
    dict(id="search_relevance", kind="fault", metric="search_ctr",
         segment={"country": "DE"}, start="2026-11-19 09:00", end="2026-11-19 21:00",
         cause="search-ranker 2.3 returns irrelevant / empty results for German queries",
         first_cause_event="search-ranker 2.3 (de locale)"),
    dict(id="stockout", kind="fault", metric="in_stock_rate",
         segment={"category": "electronics"}, start="2026-11-20 06:00", end="2026-11-22 00:00",
         cause="supplier delivery delayed: popular electronics out of stock, demand (views) stable",
         first_cause_event="Delivery delayed: Voltix wholesale PO #4471"),
    dict(id="late_orders_data", kind="data", metric="orders",
         segment={}, start="2026-11-21 21:00", end="2026-11-22 00:00",
         cause="orders_db ingestion lag: only ~40% of order rows arrived for the last hours",
         first_cause_event=None),
]

# Change log. deployed_at = when the change shipped; first_exposure_at = when the first users saw it (staged
# rollouts, cache expiry, campaign go-live). Timing tests use first_exposure_at; `ts` is kept as its alias.
EVENTS = [
    # releases (some harmless: competing explanations the agent must rule out)
    dict(type="release", name="ios-app 5.2 (UI polish)", component="app", owner="mobile-ios",
         deployed_at="2026-11-16 09:30", first_exposure_at="2026-11-16 11:00",
         country="*", platform="ios", category="*", planned=True),
    dict(type="campaign", name="NL paid social push", component="marketing", owner="growth-marketing",
         deployed_at="2026-11-16 08:30", first_exposure_at="2026-11-16 09:00", end="2026-11-17 06:00",
         country="NL", platform="*", category="*", planned=True),
    dict(type="release", name="For You: discount-affinity boost (NL)", component="recommendations", owner="recsys",
         deployed_at="2026-11-16 08:45", first_exposure_at="2026-11-16 09:10",
         country="NL", platform="*", category="*", planned=True),
    dict(type="release", name="web header i18n strings (NL)", component="app", owner="web-platform",
         deployed_at="2026-11-16 08:50", first_exposure_at="2026-11-16 09:00",
         country="NL", platform="web", category="*", planned=True),
    dict(type="release", name="web checkout button restyle", component="checkout", owner="checkout-web",
         deployed_at="2026-11-17 11:50", first_exposure_at="2026-11-17 12:05",
         country="*", platform="web", category="*", planned=True),
    dict(type="release", name="payments-sdk android 7.4.0", component="payments", owner="payments-mobile",
         deployed_at="2026-11-17 12:30", first_exposure_at="2026-11-17 13:40",
         country="*", platform="android", category="*", planned=True),
    dict(type="rollback", name="payments-sdk android 7.4.0 rolled back to 7.3.9", component="payments",
         owner="payments-mobile", deployed_at="2026-11-18 01:45", first_exposure_at="2026-11-18 02:00",
         country="*", platform="android", category="*", planned=False),
    dict(type="campaign", name="FR Black Week teaser campaign", component="marketing", owner="crm-fr",
         deployed_at="2026-11-18 17:30", first_exposure_at="2026-11-18 18:00", end="2026-11-20 00:00",
         country="FR", platform="*", category="*", planned=True),
    dict(type="release", name="search-ranker 2.3 (de locale)", component="search", owner="search-relevance",
         deployed_at="2026-11-19 08:40", first_exposure_at="2026-11-19 08:50",
         country="DE", platform="*", category="*", planned=True),
    dict(type="release", name="recommendations model refresh (For You)", component="recommendations", owner="recsys",
         deployed_at="2026-11-19 10:00", first_exposure_at="2026-11-19 10:30",
         country="*", platform="*", category="*", planned=True),
    dict(type="rollback", name="search-ranker 2.3 rolled back", component="search", owner="search-relevance",
         deployed_at="2026-11-19 20:55", first_exposure_at="2026-11-19 21:00",
         country="DE", platform="*", category="*", planned=False),
    dict(type="supply", name="Delivery delayed: Voltix wholesale PO #4471", component="wholesale",
         owner="supply-chain", deployed_at="2026-11-19 16:00", first_exposure_at="2026-11-19 16:00",
         country="*", platform="*", category="electronics", planned=False),
    dict(type="purchasing", name="Wholesale reorder run (electronics) at normal volume", component="wholesale",
         owner="purchasing-system", deployed_at="2026-11-20 05:30", first_exposure_at="2026-11-20 05:30",
         country="*", platform="*", category="electronics", planned=True),
]
for _e in EVENTS:
    _e["ts"] = _e["first_exposure_at"]


def hourly_shape() -> np.ndarray:
    h = np.arange(24)
    s = (0.12 + 0.35 * np.exp(-((h - 12.5) / 2.8) ** 2) + 0.75 * np.exp(-((h - 20.5) / 2.6) ** 2)
         + 0.10 * np.exp(-((h - 8) / 1.5) ** 2))
    return s / s.mean()


def win(ts: pd.Series, sid: str) -> pd.Series:
    sc = next(s for s in SCENARIOS if s["id"] == sid)
    return (ts >= pd.Timestamp(sc["start"])) & (ts < pd.Timestamp(sc["end"]))


def make(seed: int):
    rng = np.random.default_rng(seed)
    hours = pd.date_range(START, periods=DAYS * 24, freq="h")
    shape = hourly_shape()

    # inventory first (cart rate depends on availability)
    inv = []
    for cat, listed in LISTED.items():
        base_stock = rng.normal(0.955, 0.006, len(hours)).clip(0.9, 0.99)
        inv.append(pd.DataFrame(dict(ts=hours, category=cat, listed_skus=listed, stock=base_stock)))
    inv = pd.concat(inv, ignore_index=True)
    so = win(inv.ts, "stockout") & (inv.category == "electronics")
    ramp = ((inv.ts - pd.Timestamp("2026-11-20 06:00")) / pd.Timedelta(hours=1)).clip(0, 6) / 6
    inv.loc[so, "stock"] = 0.955 - 0.34 * ramp[so]           # 95% -> ~62% in 6h
    inv["in_stock_skus"] = np.round(inv.listed_skus * inv.stock).astype(int)
    stock = inv.set_index(["ts", "category"]).stock

    idx = pd.MultiIndex.from_product([hours, list(COUNTRIES), list(PLATFORMS), list(CATEGORIES)],
                                     names=["ts", "country", "platform", "category"])
    f = idx.to_frame(index=False)
    n = len(f)
    season = shape[f.ts.dt.hour] * WEEKLY[f.ts.dt.day_of_week]
    base = f.country.map(COUNTRIES) * f.platform.map(PLATFORMS) * f.category.map(CATEGORIES)
    lam_sessions = base * season * rng.lognormal(0, 0.05, n)
    search_rate = np.full(n, 0.55)
    ctr = np.full(n, 0.60)
    zero_share = np.full(n, 0.05)
    cart = f.category.map(CART_RATE).to_numpy() * (stock.reindex(pd.MultiIndex.from_arrays([f.ts, f.category])).to_numpy() / 0.955)
    checkout = np.full(n, 0.50)
    attempt = np.full(n, 0.92)
    pay = f.platform.map(PAY_OK).to_numpy().copy()

    # --- FR campaign (innocent): +60% sessions of low-intent paid traffic
    camp = win(f.ts, "fr_campaign") & (f.country == "FR")
    lam_sessions = np.where(camp, lam_sessions * 1.6, lam_sessions)
    cart = np.where(camp, cart * 0.72, cart)                 # paid traffic browses, buys less
    # --- margin squeeze (NL): paid social buys traffic at 1.6x cost per session; recommendations push
    #     discounted products, which convert better -> orders up, contribution margin down
    ms = win(f.ts, "margin_squeeze") & (f.country == "NL")
    lam_sessions = np.where(ms, lam_sessions * 1.35, lam_sessions)
    cart = np.where(ms, cart * 1.15, cart)
    disc = np.where(ms, 0.21, DISCOUNT) * rng.lognormal(0, 0.04, n)
    cps = np.where(ms, AD_CPS * 1.6, AD_CPS) * rng.lognormal(0, 0.06, n)
    # --- payment failure: DE android
    pf = win(f.ts, "payment_failure") & (f.country == "DE") & (f.platform == "android")
    pay = np.where(pf, 0.52, pay)
    # --- search relevance: DE all platforms
    sr = win(f.ts, "search_relevance") & (f.country == "DE")
    ctr = np.where(sr, ctr * 0.58, ctr)
    zero_share = np.where(sr, 0.21, zero_share)

    f["sessions"] = rng.poisson(lam_sessions)
    f["searches"] = rng.binomial(f.sessions, search_rate)
    f["zero_result_searches"] = rng.binomial(f.searches, zero_share)
    f["product_views"] = rng.poisson(f.searches * ctr)
    f["add_to_carts"] = rng.binomial(f.product_views, np.clip(cart, 0, 1))
    f["checkouts"] = rng.binomial(f.add_to_carts, checkout)
    f["payment_attempts"] = rng.binomial(f.checkouts, attempt)
    f["orders"] = rng.binomial(f.payment_attempts, pay)
    aov = f.category.map(AOV).to_numpy() * rng.lognormal(0, 0.08, n)
    f["gmv"] = (f.orders * aov).round(2)
    f["refunds"] = rng.binomial(f.orders, 0.04)
    f["discounts"] = (f.gmv * disc).round(2)
    f["cogs"] = (f.gmv * f.category.map(COGS).to_numpy() * rng.lognormal(0, 0.02, n)).round(2)
    # only paid sessions cost media money: the FR teaser runs on email + push, its extra sessions are free
    paid_frac = np.where(camp, 1 / 1.6, 1.0)
    f["ad_cost"] = (f.sessions * paid_frac * cps).round(2)

    # --- late data: orders_db rows only ~40% arrived for the last 3 hours
    comp = []
    sources = {"web_analytics": ["sessions", "searches", "zero_result_searches", "product_views", "add_to_carts"],
               "orders_db": ["checkouts", "payment_attempts", "orders", "gmv", "refunds", "discounts", "cogs"],
               "ads_platform": ["ad_cost"],
               "inventory": ["listed_skus", "in_stock_skus"]}
    for src in sources:
        c = pd.DataFrame(dict(ts=hours, source=src, completeness=1.0))
        if src == "orders_db":
            late = win(c.ts, "late_orders_data")
            c.loc[late, "completeness"] = [0.55, 0.42, 0.38][: late.sum()]
        comp.append(c)
    comp = pd.concat(comp, ignore_index=True)
    late_rows = win(f.ts, "late_orders_data")
    cmap = comp[comp.source == "orders_db"].set_index("ts").completeness
    frac = f.loc[late_rows, "ts"].map(cmap)
    for col in sources["orders_db"]:
        f.loc[late_rows, col] = (f.loc[late_rows, col] * frac).round(2 if col in ("gmv", "discounts", "cogs") else 0)

    inv = inv[["ts", "category", "listed_skus", "in_stock_skus"]]
    ev = pd.DataFrame(EVENTS)
    return f, inv, ev, comp


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT), help="output folder (default data/eye)")
    args = ap.parse_args()
    f, inv, ev, comp = make(args.seed)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT / "facts.csv", index=False)
    inv.to_csv(OUT / "inventory.csv", index=False)
    ev.to_csv(OUT / "events.csv", index=False)
    comp.to_csv(OUT / "completeness.csv", index=False)
    (OUT / "injected.json").write_text(json.dumps(dict(label="DEMO DATA", seed=args.seed, scenarios=SCENARIOS),
                                                  indent=2), encoding="utf-8")
    print(f"DEMO DATA written to {OUT}")
    print(f"  facts.csv: {len(f):,} rows ({f.ts.nunique()} hours x {f.groupby(['country','platform','category']).ngroups} segments)")
    print(f"  inventory.csv: {len(inv):,} rows | events.csv: {len(ev)} events | completeness.csv: {len(comp):,} rows")
    for s in SCENARIOS:
        print(f"  - {s['kind']:<9} {s['id']:<18} {s['start']} -> {s['end']}  {s['segment']}")


if __name__ == "__main__":
    main()

