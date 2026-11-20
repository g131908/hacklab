"""WATCH: signal detectors (SKILL.md section 1). Implemented: S1, S4, S6, S7.

Every detector returns one `Detection` per entity it looked at, also when nothing fired,
so state can auto-resolve open anomalies. Baselines use the same weekday/hour last week
(seasonality) unless noted. Detectors never alert themselves; self-check and triage decide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import MarketData

WEEK = pd.Timedelta(days=7)
HOUR = pd.Timedelta(hours=1)


@dataclass
class Detection:
    signal: str
    entity: str
    entity_type: str            # advertiser | supply
    ts: pd.Timestamp            # check time
    window_start: pd.Timestamp
    window_end: pd.Timestamp    # inclusive (last hour in window)
    fired: bool = False
    insufficient: bool = False  # below minimum sample
    baseline_ok: bool = True    # seasonality baseline available
    direction: str = ""         # up | down
    title: str = ""
    metrics: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    money_at_risk_day: float | None = None
    money_note: str = ""
    note: str = ""


def _sum(w: pd.DataFrame, entity: str, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Sum of an entity over [start, end]; None if any hour is missing."""
    if entity not in w.columns:
        return None
    s = w[entity].loc[start:end]
    expected_hours = int((end - start) / HOUR) + 1
    if len(s) < expected_hours or s.isna().any():
        return None
    return float(s.sum())


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


# ---------------------------------------------------------------- S1 spend pacing
def detect_s1(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    th = cfg["thresholds"]
    spend = data.wide("adv", "spend")
    day_start = now.normalize()
    out = []
    for adv in spend.columns:
        d = Detection("S1", adv, "advertiser", now, day_start, now)
        actual = _sum(spend, adv, day_start, now)
        expected = _sum(spend, adv, day_start - WEEK, now - WEEK)
        full_day_lw = _sum(spend, adv, day_start - WEEK, day_start - WEEK + pd.Timedelta(hours=23))
        if actual is None or expected is None or full_day_lw is None:
            d.baseline_ok = False
            d.note = "no same-weekday baseline (need 7 days history)"
            out.append(d)
            continue
        ratio = actual / expected if expected > 0 else np.nan
        d.metrics = dict(spend_to_date=actual, expected_to_date=expected, pacing=ratio,
                         expected_full_day=full_day_lw)
        if expected < th["s1_min_expected_spend"]:
            d.insufficient = True
            d.note = f"expected spend-to-date {expected:.0f} < {th['s1_min_expected_spend']}"
            out.append(d)
            continue
        win = f"{day_start:%H:%M}-{now + HOUR:%H:%M}"
        d.evidence = [f"spend-to-date EUR {_fmt(actual)} vs EUR {_fmt(expected)} expected "
                      f"(same weekday last week, {win}) = {ratio:.0%} pace"]
        if ratio > th["s1_pacing_high"]:
            d.fired, d.direction = True, "up"
            d.title = f"pacing {ratio:.0%} of expected"
            d.money_at_risk_day = (ratio - 1) * full_day_lw
            d.money_note = (f"~EUR {_fmt(d.money_at_risk_day)}/day overspend if pace holds "
                            f"(last week's day: EUR {_fmt(full_day_lw)})")
        elif ratio < th["s1_pacing_low"]:
            d.fired, d.direction = True, "down"
            d.title = f"under-pacing {ratio:.0%} of expected"
            d.money_at_risk_day = (1 - ratio) * full_day_lw
            d.money_note = f"~EUR {_fmt(d.money_at_risk_day)}/day of planned delivery not spent"
        out.append(d)
    return out


# ---------------------------------------------------------------- S4 tracking break
def detect_s4(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    th = cfg["thresholds"]
    hrs = th["s4_window_hours"]
    start = now - (hrs - 1) * HOUR
    conv, clicks, spend = (data.wide("adv", m) for m in ("conversions", "clicks", "spend"))
    lo, hi = th["s4_clicks_normal_band"]
    out = []
    for adv in conv.columns:
        d = Detection("S4", adv, "advertiser", now, start, now)
        c_now, c_exp = _sum(conv, adv, start, now), _sum(conv, adv, start - WEEK, now - WEEK)
        k_now, k_exp = _sum(clicks, adv, start, now), _sum(clicks, adv, start - WEEK, now - WEEK)
        day_lw = _sum(spend, adv, now.normalize() - WEEK, now.normalize() - WEEK + 23 * HOUR)
        if None in (c_now, c_exp, k_now, k_exp, day_lw):
            d.baseline_ok = False
            d.note = "no same-weekday baseline"
            out.append(d)
            continue
        d.metrics = dict(conversions=c_now, expected_conversions=c_exp, clicks=k_now,
                         expected_clicks=k_exp)
        if c_exp < th["s4_min_expected_conversions"]:
            d.insufficient = True
            d.note = f"expected conversions {c_exp:.0f} < {th['s4_min_expected_conversions']}"
            out.append(d)
            continue
        click_ratio = k_now / k_exp if k_exp else np.nan
        conv_ratio = c_now / c_exp
        d.metrics.update(click_ratio=click_ratio, conv_ratio=conv_ratio)
        win = f"{start:%H:%M}-{now + HOUR:%H:%M}"
        d.evidence = [
            f"conversions {c_now:.0f} in {hrs}h ({win}) vs ~{c_exp:.0f} expected (same weekday/hour last week)",
            f"clicks {_fmt(k_now)} vs {_fmt(k_exp)} baseline ({click_ratio:.0%})",
        ]
        if conv_ratio <= 1 - th["s4_conv_drop"] and lo <= click_ratio <= hi:
            d.fired, d.direction = True, "down"
            d.title = f"conversions {c_now:.0f} in {hrs}h vs ~{c_exp:.0f} expected, clicks normal"
            d.money_at_risk_day = day_lw
            d.money_note = (f"~EUR {_fmt(day_lw)}/day of spend optimizing against no signal "
                            "(last week's same-weekday spend); bidding degrades within 24-48h")
        out.append(d)
    return out


# ---------------------------------------------------------------- S6 creative fatigue
def detect_s6(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    """Daily check: yesterday's CTR vs the day 3 days before it, with rising frequency."""
    th = cfg["thresholds"]
    d1 = now.normalize() - pd.Timedelta(days=1)
    d4 = d1 - pd.Timedelta(days=3)
    imps, clicks, freq, spend = (data.wide("adv", m)
                                 for m in ("impressions", "clicks", "frequency", "spend"))
    out = []
    for adv in imps.columns:
        d = Detection("S6", adv, "advertiser", now, d4, d1 + 23 * HOUR)
        vals = {}
        for name, day in (("d1", d1), ("d4", d4)):
            i = _sum(imps, adv, day, day + 23 * HOUR)
            c = _sum(clicks, adv, day, day + 23 * HOUR)
            f = freq[adv].loc[day:day + 23 * HOUR].mean() if adv in freq else np.nan
            vals[name] = (i, c, f)
        if any(v[0] is None or v[1] is None for v in vals.values()):
            d.baseline_ok = False
            d.note = "need 4 full days of history"
            out.append(d)
            continue
        (i1, c1, f1), (i4, c4, f4) = vals["d1"], vals["d4"]
        if min(i1, i4) < th["s6_min_daily_impressions"]:
            d.insufficient = True
            d.note = "impressions below minimum"
            out.append(d)
            continue
        ctr1, ctr4 = c1 / i1, c4 / i4
        change = ctr1 / ctr4 - 1
        d.metrics = dict(ctr_now=ctr1, ctr_before=ctr4, ctr_change=change, freq_now=f1, freq_before=f4)
        d.evidence = [f"CTR {ctr1:.2%} on {d1:%a %d.%m} vs {ctr4:.2%} on {d4:%a %d.%m} ({change:+.0%})",
                      f"avg frequency {f4:.2f} -> {f1:.2f}"]
        if change <= -th["s6_ctr_drop"] and f1 > f4:
            day_spend = _sum(spend, adv, d1, d1 + 23 * HOUR) or 0.0
            d.fired, d.direction = True, "down"
            d.title = f"CTR {change:+.0%} in 3 days, frequency up"
            d.money_at_risk_day = -change * day_spend
            d.money_note = (f"~EUR {_fmt(d.money_at_risk_day)}/day of spend buying fewer clicks "
                            f"(EUR {_fmt(day_spend)} spend on {d1:%d.%m} x {-change:.0%} CTR loss)")
        out.append(d)
    return out


# ---------------------------------------------------------------- S7 invalid traffic
def detect_s7(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    th = cfg["thresholds"]
    imps, clicks, conv, fast, ip = (data.wide("sup", m) for m in
                                    ("impressions", "clicks", "conversions",
                                     "fast_conversions", "top_ip_click_share"))
    # marketplace CPC (advertiser side, trailing 7 days) to price excess clicks
    a_sp, a_cl = data.wide("adv", "spend"), data.wide("adv", "clicks")
    lw = slice(now - WEEK, now - HOUR)
    cpc = float(a_sp.loc[lw].sum().sum() / max(a_cl.loc[lw].sum().sum(), 1))
    out = []
    for src in imps.columns:
        d = Detection("S7", src, "supply", now, now, now)
        if now not in imps.index:
            d.baseline_ok = False
            out.append(d)
            continue
        hist_ctr = (clicks[src] / imps[src]).loc[lw].dropna()
        if len(hist_ctr) < 48:
            d.baseline_ok = False
            d.note = "need >= 48h CTR history"
            out.append(d)
            continue
        k, i = float(clicks.at[now, src]), float(imps.at[now, src])
        if k < th["s7_min_clicks"] or i <= 0:
            d.insufficient = True
            d.note = f"clicks {k:.0f} < {th['s7_min_clicks']}"
            out.append(d)
            continue
        mu, sd = hist_ctr.mean(), hist_ctr.std()
        ctr = k / i
        z = (ctr - mu) / sd if sd > 0 else 0.0
        cv = float(conv.at[now, src])
        fast_share = float(fast.at[now, src]) / cv if cv >= 5 else 0.0
        ip_share = float(ip.at[now, src])
        excess = max(k - mu * i, 0.0)
        d.metrics = dict(ctr=ctr, ctr_baseline=mu, z=z, fast_conv_share=fast_share,
                         top_ip_share=ip_share, excess_clicks=excess, cpc=cpc)
        d.evidence = [
            f"CTR {ctr:.2%} vs {mu:.2%} baseline (trailing 7d, {z:+.1f} sigma); clicks {k:,.0f}, ~{excess:,.0f} above normal",
            f"sub-10s conversions {fast_share:.0%} of {cv:.0f}; top IP share {ip_share:.0%} of clicks",
        ]
        if z > th["s7_ctr_sigma"] and (fast_share > th["s7_fast_conv_share"]
                                       or ip_share > th["s7_ip_concentration"]):
            d.fired, d.direction = True, "up"
            d.title = f"CTR {z:+.0f} sigma, sub-10s conversions / IP cluster"
            d.money_at_risk_day = excess * cpc * 24
            d.money_note = (f"~EUR {_fmt(excess * cpc)}/h billed for suspect clicks "
                            f"({excess:,.0f} x EUR {cpc:.2f} avg CPC); ~EUR {_fmt(d.money_at_risk_day)}/day if sustained")
        out.append(d)
    return out


# ---------------------------------------------------------------- S2 budget cap breach risk
def detect_s2(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    """Projected end-of-day spend = spend-to-date / share of a normal day's spend done by this hour
    (same weekday last week). Fires when the projection exceeds the account's hard daily cap."""
    th = cfg["thresholds"]
    caps = (cfg.get("caps") or {}).get("account_daily_caps") or {}
    spend = data.wide("adv", "spend")
    day_start = now.normalize()
    out = []
    for adv in spend.columns:
        d = Detection("S2", adv, "advertiser", now, day_start, now)
        cap = caps.get(adv)
        if cap is None:
            d.baseline_ok = False
            d.note = "no daily cap configured for this account"
            out.append(d)
            continue
        actual = _sum(spend, adv, day_start, now)
        lw_to = _sum(spend, adv, day_start - WEEK, now - WEEK)
        lw_day = _sum(spend, adv, day_start - WEEK, day_start - WEEK + 23 * HOUR)
        if None in (actual, lw_to, lw_day) or not lw_day:
            d.baseline_ok = False
            d.note = "no same-weekday baseline (need 7 days history)"
            out.append(d)
            continue
        share = lw_to / lw_day
        projected = actual / share if share > 0 else np.nan
        d.metrics = dict(spend_to_date=actual, day_share=share, projected=projected, cap=float(cap),
                         last_week_day=lw_day)
        if share < th["s2_min_day_share"]:
            d.insufficient = True
            d.note = f"only {share:.0%} of a normal day's spend done by this hour"
            out.append(d)
            continue
        d.evidence = [f"projected end-of-day spend EUR {_fmt(projected)} vs hard cap EUR {_fmt(cap)} "
                      f"(EUR {_fmt(actual)} spent by {now + HOUR:%H:%M}; last week {share:.0%} of the day was spent by then)"]
        if projected > cap:
            d.fired, d.direction = True, "up"
            d.title = f"projected EUR {_fmt(projected)} vs daily cap EUR {_fmt(cap)}"
            d.money_at_risk_day = projected - cap
            d.money_note = (f"~EUR {_fmt(projected - cap)} above the hard cap by end of day if the pace holds "
                            f"(last week's same weekday: EUR {_fmt(lw_day)})")
        out.append(d)
    return out


# ---------------------------------------------------------------- S8 fill rate / inventory drop
def detect_s8(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    th = cfg["thresholds"]
    hrs = th["s8_window_hours"]
    start = now - (hrs - 1) * HOUR
    req, imp = data.wide("sup", "ad_requests"), data.wide("sup", "impressions")
    a_sp, a_im = data.wide("adv", "spend"), data.wide("adv", "impressions")
    lw = slice(now - WEEK, now - HOUR)
    cpm = float(a_sp.loc[lw].sum().sum() / max(a_im.loc[lw].sum().sum(), 1) * 1000)
    out = []
    for src in req.columns:
        d = Detection("S8", src, "supply", now, start, now)
        r, r_exp = _sum(req, src, start, now), _sum(req, src, start - WEEK, now - WEEK)
        i, i_exp = _sum(imp, src, start, now), _sum(imp, src, start - WEEK, now - WEEK)
        if None in (r, r_exp, i, i_exp):
            d.baseline_ok = False
            d.note = "no same-weekday baseline"
            out.append(d)
            continue
        if r_exp < th["s8_min_expected_requests"]:
            d.insufficient = True
            d.note = f"expected ad requests {r_exp:.0f} < {th['s8_min_expected_requests']}"
            out.append(d)
            continue
        fill, fill_exp = i / r if r else 0.0, i_exp / r_exp
        fill_ratio, req_ratio = fill / fill_exp if fill_exp else np.nan, r / r_exp
        lost = max(i_exp - i, 0.0)
        d.metrics = dict(fill_rate=fill, fill_rate_baseline=fill_exp, fill_ratio=fill_ratio,
                         ad_requests=r, ad_requests_baseline=r_exp, requests_ratio=req_ratio,
                         impressions=i, impressions_baseline=i_exp, lost_impressions=lost, cpm=cpm)
        win = f"{start:%H:%M}-{now + HOUR:%H:%M}"
        d.evidence = [f"fill rate {fill:.1%} vs {fill_exp:.1%} same hours last week ({win}, {fill_ratio - 1:+.0%})",
                      f"ad requests {_fmt(r)} vs {_fmt(r_exp)} ({req_ratio - 1:+.0%}); impressions {_fmt(i)} vs {_fmt(i_exp)}"]
        fill_hit = fill_ratio <= 1 - th["s8_fill_drop"]
        req_hit = req_ratio <= 1 - th["s8_requests_drop"]
        if fill_hit or req_hit:
            d.fired, d.direction = True, "down"
            d.title = (f"fill rate {fill:.0%} vs {fill_exp:.0%} last week" if fill_hit
                       else f"ad requests {req_ratio - 1:+.0%} vs last week")
            per_h = lost / hrs
            d.money_at_risk_day = per_h * 24 * cpm / 1000
            d.money_note = (f"~{_fmt(per_h)} impressions/h not delivered x EUR {cpm:.2f} avg advertiser CPM = "
                            f"~EUR {_fmt(per_h * cpm / 1000)}/h of advertiser demand unserved; "
                            f"~EUR {_fmt(d.money_at_risk_day)}/day if sustained")
        out.append(d)
    return out


HOURLY = {"S1": detect_s1, "S2": detect_s2, "S4": detect_s4, "S7": detect_s7, "S8": detect_s8}
DAILY = {"S6": detect_s6}


def run_detectors(data: MarketData, now: pd.Timestamp, cfg: dict) -> list[Detection]:
    dets: list[Detection] = []
    for fn in HOURLY.values():
        dets += fn(data, now, cfg)
    if now.hour == cfg["thresholds"]["s6_check_hour"]:
        for fn in DAILY.values():
            dets += fn(data, now, cfg)
    return dets
