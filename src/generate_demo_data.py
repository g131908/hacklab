"""Generate 14 days of hourly DEMO DATA for an ads marketplace.

Outputs (all synthetic, labelled DEMO DATA):
  data/advertisers.csv  one row per hour x advertiser
  data/supply.csv       one row per hour x supply source
  data/injected.json    ground truth: the 5 injected scenarios and their timestamps

Run: python src/generate_demo_data.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

START = pd.Timestamp("2026-11-06 00:00")  # Friday; local time Europe/Berlin
DAYS = 14

# daily volume, CTR, CVR, CPM (EUR), avg order value (EUR), base frequency
ADVERTISERS = {
    "adv_01": dict(imps=900_000, ctr=0.012, cvr=0.035, cpm=4.5, aov=60.0, freq=2.0),
    "adv_02": dict(imps=600_000, ctr=0.015, cvr=0.030, cpm=3.8, aov=45.0, freq=2.2),
    "adv_03": dict(imps=750_000, ctr=0.011, cvr=0.040, cpm=4.0, aov=80.0, freq=1.8),
    "adv_04": dict(imps=400_000, ctr=0.010, cvr=0.025, cpm=3.2, aov=35.0, freq=1.6),
    "adv_05": dict(imps=500_000, ctr=0.013, cvr=0.030, cpm=5.0, aov=70.0, freq=2.1),
    "adv_06": dict(imps=250_000, ctr=0.009, cvr=0.030, cpm=2.8, aov=50.0, freq=1.5),
}

# daily ad requests, fill rate, CTR, CVR, publisher payout CPM (EUR)
SUPPLY = {
    "src_A": dict(req=1_400_000, fill=0.82, ctr=0.012, cvr=0.030, payout_cpm=2.6),
    "src_B": dict(req=1_000_000, fill=0.75, ctr=0.011, cvr=0.032, payout_cpm=2.4),
    "src_C": dict(req=600_000, fill=0.70, ctr=0.010, cvr=0.028, payout_cpm=2.0),
    "src_D": dict(req=450_000, fill=0.88, ctr=0.013, cvr=0.035, payout_cpm=3.0),
}

# Mon..Sun
WEEKLY = np.array([1.00, 1.02, 1.03, 1.00, 0.95, 0.85, 0.90])

# Injected scenarios (end is exclusive)
SCENARIOS = [
    dict(id="tracking_break", signal="S4", accept_signals=["S4"], entity="adv_03",
         entity_type="advertiser", start="2026-11-13 10:00", end="2026-11-13 16:00",
         expected="detect", description="Conversions drop to 0 while clicks stay normal"),
    dict(id="click_fraud", signal="S7", accept_signals=["S7"], entity="src_C",
         entity_type="supply", start="2026-11-14 22:00", end="2026-11-15 02:00",
         expected="detect", description="Click burst from one source: CTR spike, sub-10s conversions, IP concentration"),
    dict(id="runaway_spend", signal="S1", accept_signals=["S1", "S2"], entity="adv_05",
         entity_type="advertiser", start="2026-11-16 08:00", end="2026-11-17 00:00",
         expected="detect", description="Spend at ~180% of normal pace"),
    dict(id="creative_fatigue", signal="S6", accept_signals=["S6"], entity="adv_02",
         entity_type="advertiser", start="2026-11-15 00:00", end="2026-11-20 00:00",
         expected="detect", description="CTR -11%/day with rising frequency"),
    dict(id="fill_rate_drop", signal="S8", accept_signals=["S8"], entity="src_B",
         entity_type="supply", start="2026-11-17 13:00", end="2026-11-17 20:00",
         expected="detect", description="Fill rate falls from ~75% to ~45% on one source, ad requests unchanged"),
    dict(id="holiday_dip", signal="S1", accept_signals=["S1", "S2", "S4", "S8"], entity="*",
         entity_type="all", start="2026-11-18 00:00", end="2026-11-19 00:00",
         expected="suppress", description="Holiday (Buss- und Bettag): volume -50%, listed in known_events"),
]


def hourly_shape() -> np.ndarray:
    h = np.arange(24)
    shape = (0.15
             + 0.15 * np.exp(-((h - 8.5) / 1.5) ** 2)
             + 0.55 * np.exp(-((h - 12.5) / 2.5) ** 2)
             + 0.85 * np.exp(-((h - 20.5) / 2.5) ** 2))
    return shape / shape.mean()


def window(ts: pd.Series, name: str) -> pd.Series:
    sc = next(s for s in SCENARIOS if s["id"] == name)
    return (ts >= pd.Timestamp(sc["start"])) & (ts < pd.Timestamp(sc["end"]))


def make_advertisers(rng: np.random.Generator, ts: pd.DatetimeIndex) -> pd.DataFrame:
    shape = hourly_shape()
    frames = []
    for adv, p in ADVERTISERS.items():
        n = len(ts)
        season = shape[ts.hour] * WEEKLY[ts.day_of_week]
        imps = p["imps"] / 24 * season * rng.lognormal(0, 0.08, n)
        ctr = p["ctr"] * rng.lognormal(0, 0.05, n)
        freq = p["freq"] + rng.normal(0, 0.05, n)
        spend_noise = rng.lognormal(0, 0.03, n)
        aov = p["aov"] * rng.lognormal(0, 0.10, n)
        frames.append(pd.DataFrame(dict(ts=ts, advertiser_id=adv, imps=imps, ctr=ctr,
                                        freq=freq, spend_noise=spend_noise, aov=aov,
                                        cvr=p["cvr"], cpm=p["cpm"])))
    df = pd.concat(frames, ignore_index=True)

    # creative fatigue: adv_02 CTR -11% per day, frequency +0.35 per day
    fat = window(df.ts, "creative_fatigue") & (df.advertiser_id == "adv_02")
    day_idx = (df.ts - pd.Timestamp("2026-11-15")).dt.days + 1
    df.loc[fat, "ctr"] *= 0.89 ** day_idx[fat]
    df.loc[fat, "freq"] += 0.35 * day_idx[fat]

    # holiday: all volumes -50%
    hol = window(df.ts, "holiday_dip")
    df.loc[hol, "imps"] *= 0.5

    df["impressions"] = np.round(df.imps).astype(int)
    df["clicks"] = rng.poisson(df.impressions * df.ctr)
    df["conversions"] = rng.poisson(df.clicks * df.cvr)
    df["spend"] = df.impressions / 1000 * df.cpm * df.spend_noise

    # runaway spend: adv_05 at ~180% pace
    run = window(df.ts, "runaway_spend") & (df.advertiser_id == "adv_05")
    df.loc[run, "spend"] *= 1.8

    # tracking break: adv_03 conversions not recorded, clicks unaffected
    brk = window(df.ts, "tracking_break") & (df.advertiser_id == "adv_03")
    df.loc[brk, "conversions"] = 0

    df["revenue"] = df.conversions * df.aov
    df["frequency"] = df.freq
    df["spend"] = df.spend.round(2)
    df["revenue"] = df.revenue.round(2)
    df["frequency"] = df.frequency.round(3)
    return df[["ts", "advertiser_id", "impressions", "clicks", "conversions",
               "spend", "revenue", "frequency"]]


def make_supply(rng: np.random.Generator, ts: pd.DatetimeIndex) -> pd.DataFrame:
    shape = hourly_shape()
    frames = []
    for src, p in SUPPLY.items():
        n = len(ts)
        season = shape[ts.hour] * WEEKLY[ts.day_of_week]
        req = p["req"] / 24 * season * rng.lognormal(0, 0.07, n)
        fill = np.clip(p["fill"] * rng.lognormal(0, 0.02, n), 0, 1)
        ctr = p["ctr"] * rng.lognormal(0, 0.05, n)
        ip_share = np.clip(rng.normal(0.04, 0.01, n), 0.005, 1)
        frames.append(pd.DataFrame(dict(ts=ts, source_id=src, req=req, fill_rate=fill,
                                        ctr=ctr, cvr=p["cvr"], payout_cpm=p["payout_cpm"],
                                        top_ip_click_share=ip_share, fast_p=0.03)))
    df = pd.concat(frames, ignore_index=True)

    hol = window(df.ts, "holiday_dip")
    df.loc[hol, "req"] *= 0.5

    # fill rate drop: src_B fills only ~60% of its usual share (e.g. a broken SSP/demand integration)
    fd = window(df.ts, "fill_rate_drop") & (df.source_id == "src_B")
    df.loc[fd, "fill_rate"] *= 0.6

    df["ad_requests"] = np.round(df.req).astype(int)
    df["impressions"] = np.round(df.ad_requests * df.fill_rate).astype(int)
    df["clicks"] = rng.poisson(df.impressions * df.ctr)

    # click fraud: src_C, extra bot clicks (~5x), sub-10s conversions, single-IP cluster
    fr = window(df.ts, "click_fraud") & (df.source_id == "src_C")
    df.loc[fr, "clicks"] += rng.poisson(df.loc[fr, "clicks"] * 5)
    df.loc[fr, "fast_p"] = 0.6
    df.loc[fr, "top_ip_click_share"] = rng.uniform(0.30, 0.45, fr.sum())

    df["conversions"] = rng.poisson(df.clicks * df.cvr)
    df["fast_conversions"] = rng.binomial(df.conversions, df.fast_p)
    df["payouts"] = (df.impressions / 1000 * df.payout_cpm).round(2)
    df["fill_rate"] = df.fill_rate.round(4)
    df["top_ip_click_share"] = df.top_ip_click_share.round(4)
    return df[["ts", "source_id", "ad_requests", "impressions", "fill_rate", "clicks",
               "conversions", "fast_conversions", "top_ip_click_share", "payouts"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ts = pd.date_range(START, periods=DAYS * 24, freq="h")
    adv = make_advertisers(rng, ts)
    sup = make_supply(rng, ts)

    DATA.mkdir(exist_ok=True)
    adv.to_csv(DATA / "advertisers.csv", index=False)
    sup.to_csv(DATA / "supply.csv", index=False)
    meta = dict(label="DEMO DATA", seed=args.seed, start=str(ts[0]), end=str(ts[-1]),
                timezone="Europe/Berlin", scenarios=SCENARIOS)
    (DATA / "injected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("DEMO DATA generated")
    print(f"  {DATA / 'advertisers.csv'}: {len(adv)} rows, {adv.advertiser_id.nunique()} advertisers")
    print(f"  {DATA / 'supply.csv'}: {len(sup)} rows, {sup.source_id.nunique()} supply sources")
    print(f"  {ts[0]} .. {ts[-1]} ({DAYS} days hourly)")
    print(f"  {DATA / 'injected.json'}: {len(SCENARIOS)} scenarios")
    for s in SCENARIOS:
        print(f"    - {s['id']:<17} {s['signal']:<3} {s['entity']:<7} {s['start']} -> {s['end']}  ({s['expected']})")


if __name__ == "__main__":
    main()
