"""Rule-based specialists: each skill's method (SKILL.md section 4) applied to one anomaly,
deterministically and offline. Every function returns the specialist output contract:

{skill, anomaly_id, diagnosis, evidence[], hypotheses[{cause, confidence, how_to_verify}],
 recommended_action{action, tier, reversible, est_impact}, stance, missing_data[], source}

stance: act_now | hold | investigate (used by synthesis to surface disagreements).
Facts (evidence), hypotheses and missing data stay separate; numbers only come from the data.
"""
from __future__ import annotations

import pandas as pd

from .detectors import Detection
from .loader import MarketData

WEEK = pd.Timedelta(days=7)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def _out(skill, aid, diagnosis, evidence, hypotheses, action, stance, missing):
    return dict(skill=skill, anomaly_id=aid, diagnosis=diagnosis, evidence=evidence,
                hypotheses=[dict(cause=c, confidence=k, how_to_verify=v) for c, k, v in hypotheses],
                recommended_action=action, stance=stance, missing_data=missing, source="rules")


def _act(action, tier, reversible, impact):
    return dict(action=action, tier=tier, reversible=reversible, est_impact=impact)


def _sum(data: MarketData, side: str, metric: str, entity: str, start, end) -> float:
    w = data.wide(side, metric)
    return float(w[entity].loc[start:end].sum()) if entity in w else float("nan")


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


# ====================================================================== S4 tracking break
def conversion_tracking_s4(det: Detection, data: MarketData, aid: str) -> dict:
    e, now = det.entity, det.ts
    conv, clicks, rev = (data.wide("adv", m)[e] for m in ("conversions", "clicks", "revenue"))
    # walk back to find when the current near-zero streak started
    start = now
    t = now
    while t - WEEK in conv.index and t in conv.index:
        base = conv.loc[t - WEEK]
        if base > 0 and conv.loc[t] <= 0.2 * base:
            start, t = t, t - HOUR
        else:
            break
    hours = int((now - start) / HOUR) + 1
    c = conv.loc[start:now].sum()
    c_exp = conv.loc[start - WEEK:now - WEEK].sum()
    k = clicks.loc[start:now].sum()
    k_exp = clicks.loc[start - WEEK:now - WEEK].sum()
    r = rev.loc[start:now].sum()
    evidence = [
        f"conversions {c:.0f} vs {c_exp:.0f} expected since {start:%H:%M} ({hours}h, same weekday last week)",
        f"clicks {k:,.0f} vs {k_exp:,.0f} ({_ratio(k, k_exp):.0%}) in the same hours: traffic is intact",
        f"tracked revenue EUR {r:,.0f} in the window" + (" - value is missing too, so the whole purchase event is gone"
                                                          if r == 0 else ""),
    ]
    return _out(
        "conversion-tracking", aid,
        f"Conversion event for {e} stopped recording at {start:%H:%M} while clicks stayed normal: "
        "a tracking break, not a demand drop.",
        evidence,
        [("Purchase tag/pixel removed or broken by a site deploy or tag-manager change", "medium",
          "Fire a test conversion and watch it in the platform's event debugger / GTM preview; check the deploy log for changes before " + f"{start:%H:%M}"),
         ("Consent banner change now blocks the conversion tag (ad_storage denied by default)", "low",
          "Load the confirmation page with consent granted vs denied and compare tag firing"),
         ("Platform-side ingestion outage", "low",
          "Check the ad platform status page and whether other advertisers' conversions are normal")],
        _act(f"Pause automated bidding on {e} until a test conversion arrives again", 2, True,
             "stops smart bidding from learning that clicks do not convert"),
        "act_now",
        ["backend orders/installs for the same hours (to confirm the real conversions still happen)",
         "deploy / tag-manager change log"],
    )


def analytics_s4(det: Detection, data: MarketData, aid: str) -> dict:
    e, now = det.entity, det.ts
    start = det.window_start
    c, c_exp = det.metrics["conversions"], det.metrics["expected_conversions"]
    k, k_exp = det.metrics["clicks"], det.metrics["expected_clicks"]
    cvr, cvr_exp = _ratio(c, k), _ratio(c_exp, k_exp)
    # is the baseline itself sane? compare last week's day CVR with the week before when available
    d_lw = (now - WEEK).normalize()
    lw_c = _sum(data, "adv", "conversions", e, d_lw, d_lw + 23 * HOUR)
    lw_k = _sum(data, "adv", "clicks", e, d_lw, d_lw + 23 * HOUR)
    evidence = [
        f"conversion rate {cvr:.2%} vs {cvr_exp:.2%} baseline ({start:%H:%M}-{now + HOUR:%H:%M}, same weekday last week)",
        f"baseline day {d_lw:%a %d.%m} CVR {_ratio(lw_c, lw_k):.2%} over {lw_k:,.0f} clicks - no known event, usable as baseline",
    ]
    return _out(
        "analytics", aid,
        f"CVR collapsed ({cvr:.2%} vs {cvr_exp:.2%}) on normal click volume; the metric is computed correctly "
        "and the baseline is valid, so the gap is in measurement.",
        evidence,
        [("Measurement gap in the conversion event (see conversion-tracking)", "high",
          "Reconcile platform conversions with backend orders for the window")],
        _act(f"Annotate the window as a tracking gap for {e} and exclude it from baselines and reports", 1, True,
             "keeps future baselines and CPA reports clean"),
        "act_now",
        ["backend orders for reconciliation"],
    )


# ====================================================================== S1 spend pacing
def ads_s1(det: Detection, data: MarketData, aid: str) -> dict:
    e, now = det.entity, det.ts
    d0 = now.normalize()
    sp = _sum(data, "adv", "spend", e, d0, now)
    sp_lw = _sum(data, "adv", "spend", e, d0 - WEEK, now - WEEK)
    im = _sum(data, "adv", "impressions", e, d0, now)
    im_lw = _sum(data, "adv", "impressions", e, d0 - WEEK, now - WEEK)
    rv = _sum(data, "adv", "revenue", e, d0, now)
    rv_lw = _sum(data, "adv", "revenue", e, d0 - WEEK, now - WEEK)
    cpm, cpm_lw = _ratio(sp, im) * 1000, _ratio(sp_lw, im_lw) * 1000
    roas, roas_lw = _ratio(rv, sp), _ratio(rv_lw, sp_lw)
    imp_ratio, cpm_ratio = _ratio(im, im_lw), _ratio(cpm, cpm_lw)
    evidence = [
        f"spend EUR {sp:,.0f} vs EUR {sp_lw:,.0f} same weekday last week (00:00-{now + HOUR:%H:%M})",
        f"impressions {imp_ratio:.0%} of last week, CPM EUR {cpm:.2f} vs EUR {cpm_lw:.2f} ({cpm_ratio:.0%})",
        f"ROAS {roas:.2f} vs {roas_lw:.2f} last week (revenue EUR {rv:,.0f} vs EUR {rv_lw:,.0f})",
    ]
    if det.direction == "up":
        price_driven = cpm_ratio > 1.15 and imp_ratio < 1.15
        cause = ("Bid or bid-strategy change raised the price paid per impression (CPM up, volume flat)"
                 if price_driven else "Budget raised or targeting widened: more volume bought at similar prices")
        roas_down = roas < 0.85 * roas_lw
        stance = "act_now" if roas_down else "hold"
        diag = (f"{e} is overspending because it pays more per impression, not because it buys more reach"
                if price_driven else f"{e} is buying more volume than usual")
        diag += "; ROAS is down, so the extra spend is not paying back." if roas_down else \
                "; ROAS is holding, so the extra spend may be paying back."
        return _out(
            "ads", aid, diag, evidence,
            [(cause, "high" if price_driven else "medium",
              "Check the change history for bid/budget edits on this account today"),
             ("Auction pressure from a competitor entering the same inventory", "low",
              "Compare CPM on other advertisers in the same placements")],
            _act(f"Lower {e}'s bids or daily budget by 20% (cap per run) and re-check pacing in 2h", 2, True,
                 f"brings pace back toward last week's EUR {sp_lw:,.0f} to this hour"),
            stance,
            ["campaign change history (bid/budget edits)", "hard daily budget for this account"],
        )
    return _out(
        "ads", aid, f"{e} is under-delivering vs last week's same weekday.", evidence,
        [("Budget exhausted early or bids below the clearing price", "medium",
          "Check budget-limited status and impression share lost to rank"),
         ("Demand dip (seasonal or event)", "medium", "Compare with other advertisers in the same hour")],
        _act(f"Review bids/budget for {e}; raise delivery only if ROAS supports it", 2, True,
             "restores planned delivery"),
        "investigate",
        ["impression share / budget-limited status"],
    )


def attribution_s1(det: Detection, data: MarketData, aid: str) -> dict:
    e, now = det.entity, det.ts
    d0 = now.normalize()
    sp = _sum(data, "adv", "spend", e, d0, now)
    sp_lw = _sum(data, "adv", "spend", e, d0 - WEEK, now - WEEK)
    cv = _sum(data, "adv", "conversions", e, d0, now)
    cv_lw = _sum(data, "adv", "conversions", e, d0 - WEEK, now - WEEK)
    sp_r, cv_r = _ratio(sp, sp_lw), _ratio(cv, cv_lw)
    evidence = [f"spend {sp_r:.0%} of last week vs conversions {cv_r:.0%} ({cv:.0f} vs {cv_lw:.0f}) to this hour"]
    if cv_r < 1 + 0.5 * (sp_r - 1):
        return _out(
            "attribution", aid,
            "The extra spend did not bring proportional conversions: this is a real cost increase, not a credit shift.",
            evidence,
            [("Incremental spend is buying no incremental conversions", "medium",
              "Holdout or geo-split test if the overspend persists")],
            _act("Treat as cost problem; no attribution change needed", 1, True, "-"),
            "act_now", ["conversion lag (late conversions may still arrive)"],
        )
    return _out(
        "attribution", aid,
        "Conversions rose with spend; the extra spend may be incremental - judge on revenue before cutting.",
        evidence,
        [("Spend increase is buying incremental conversions", "medium", "Compare new-customer share vs last week")],
        _act("Hold budget changes until a full day of revenue is in", 1, True, "avoids cutting profitable spend"),
        "hold", ["new vs returning customer split"],
    )


# ====================================================================== S6 creative fatigue
def marketing_psychology_s6(det: Detection, data: MarketData, aid: str) -> dict:
    m = det.metrics
    return _out(
        "marketing-psychology", aid,
        f"Classic creative wear-out: frequency {m['freq_before']:.2f} -> {m['freq_now']:.2f} while CTR fell "
        f"{-m['ctr_change']:.0%}; the same people see the same ad and stop noticing it.",
        [f"CTR {m['ctr_before']:.2%} -> {m['ctr_now']:.2%} over 3 days",
         f"frequency +{m['freq_now'] - m['freq_before']:.2f} in the same period"],
        [("Habituation / banner blindness: repeated exposure lowers attention (mere-exposure effect has turned negative)",
          "high", "A/B test: current creative vs 2 fresh variants on the same audience, CTR over 48h"),
         ("Novelty and curiosity gap used up: the hook no longer creates tension", "medium",
          "Test a variant with a new hook/headline but the same offer"),
         ("Audience too small for the budget, so frequency climbs", "medium",
          "Check reach vs audience size; widen targeting in a test cell")],
        _act(f"Rotate in 2-3 new creatives (contrast/novelty-based hooks) and cap frequency at ~3/day for {det.entity}",
             2, True, "CTR typically recovers toward baseline if wear-out is the cause; verify via A/B test"),
        "act_now",
        ["creative-level data (which ad IDs are fatiguing)", "reach / unique users"],
    )


def ads_s6(det: Detection, data: MarketData, aid: str) -> dict:
    e, now = det.entity, det.ts
    d1 = now.normalize() - DAY
    d4 = d1 - 3 * DAY

    def day(metric, d):
        return _sum(data, "adv", metric, e, d, d + 23 * HOUR)

    cvr1, cvr4 = _ratio(day("conversions", d1), day("clicks", d1)), _ratio(day("conversions", d4), day("clicks", d4))
    roas1, roas4 = _ratio(day("revenue", d1), day("spend", d1)), _ratio(day("revenue", d4), day("spend", d4))
    stable_cvr = cvr1 >= 0.85 * cvr4  # stable or up: post-click is fine
    roas_drop = roas1 < 0.85 * roas4
    return _out(
        "ads", aid,
        ("CTR is falling but post-click CVR is stable or up, so the problem sits in the ad, not the landing page"
         if stable_cvr else "CTR and CVR both fell; not purely a creative problem")
        + ("; ROAS is down, act now." if roas_drop else "; ROAS still holds."),
        [f"CVR {cvr4:.2%} on {d4:%d.%m} -> {cvr1:.2%} on {d1:%d.%m}",
         f"ROAS {roas4:.2f} -> {roas1:.2f} (revenue / spend per day)"],
        [("Creative fatigue (CTR-only decline)" if stable_cvr else "Landing page or offer change", "high" if stable_cvr else "medium",
          "Split CTR by creative ID; the fatigued ones show the steepest decline")],
        _act(f"Refresh creatives for {e}; keep budget unchanged until new creatives have 48h of data", 2, True,
             "protects ROAS without cutting reach"),
        "act_now" if roas_drop else "hold",
        ["per-creative breakdown"],
    )


# ====================================================================== S7 invalid traffic
def analytics_s7(det: Detection, data: MarketData, aid: str) -> dict:
    src, now = det.entity, det.ts
    m = det.metrics
    imps, clicks = data.wide("sup", "impressions"), data.wide("sup", "clicks")
    lw = slice(now - WEEK, now - HOUR)
    others = []
    for s in imps.columns:
        if s == src or now not in imps.index:
            continue
        h = (clicks[s] / imps[s]).loc[lw]
        z = (clicks.at[now, s] / imps.at[now, s] - h.mean()) / h.std()
        others.append(f"{s} {z:+.1f}")
    payout = float(data.wide("sup", "payouts").at[now, src])
    return _out(
        "analytics", aid,
        f"Invalid-traffic pattern isolated to {src}: CTR {m['z']:+.0f} sigma while other sources are normal, "
        f"plus {m['fast_conv_share']:.0%} sub-10s conversions and {m['top_ip_share']:.0%} of clicks from one IP.",
        [f"other sources this hour (CTR sigma): {', '.join(others)}",
         f"~{m['excess_clicks']:,.0f} excess clicks x EUR {m['cpc']:.2f} avg CPC = ~EUR {m['excess_clicks'] * m['cpc']:,.0f} billed to advertisers this hour",
         f"{src} payout this hour EUR {payout:,.0f} (impression-based)"],
        [("Bot / click-farm traffic on one source", "high",
          "Pull raw click logs for the hour: IP/ASN/device distribution and click-to-conversion times"),
         ("Click injection by a publisher SDK", "medium", "Check click timestamps vs install/conversion timestamps per device"),
         ("Legit viral placement", "low", "Would not produce sub-10s conversions or a 20%+ single-IP share")],
        _act(f"Block {src} for affected campaigns, hold its payouts and credit advertisers for invalid clicks after review",
             2, False, "stops further billing for invalid clicks; credits touch billing (always escalate)"),
        "act_now",
        ["raw IP/ASN/device click logs", "click-to-conversion time distribution", "which campaigns received the clicks"],
    )


# ====================================================================== S2 budget cap breach risk
def ads_s2(det: Detection, data: MarketData, aid: str) -> dict:
    out = ads_s1(det, data, aid)  # same pacing analysis (CPM vs volume, ROAS), framed against the cap
    m = det.metrics
    out["evidence"].insert(0, f"projected EUR {m['projected']:,.0f} vs hard cap EUR {m['cap']:,.0f} "
                              f"(EUR {m['spend_to_date']:,.0f} spent, {m['day_share']:.0%} of a normal day done)")
    needed = 1 - m["cap"] / m["projected"] if m["projected"] else 0
    out["recommended_action"] = _act(
        f"Lower {det.entity}'s bids/daily budget by {min(needed, 0.20):.0%} now (cap per run 20%)"
        + (f"; {needed:.0%} needed in total, re-check in 1h and repeat" if needed > 0.20 else ""),
        2, True, f"keeps end-of-day spend under the EUR {m['cap']:,.0f} cap")
    out["stance"] = "act_now"  # a hard cap is a hard cap, whatever ROAS says
    return out


# ====================================================================== S8 fill rate / inventory drop
def analytics_s8(det: Detection, data: MarketData, aid: str) -> dict:
    src, now, m = det.entity, det.ts, det.metrics
    start = det.window_start
    req, imp = data.wide("sup", "ad_requests"), data.wide("sup", "impressions")
    others = []
    for s in req.columns:
        if s == src:
            continue
        r, r_lw = req[s].loc[start:now].sum(), req[s].loc[start - WEEK:now - WEEK].sum()
        f = imp[s].loc[start:now].sum() / r if r else 0
        f_lw = imp[s].loc[start - WEEK:now - WEEK].sum() / r_lw if r_lw else 0
        others.append((s, _ratio(f, f_lw) - 1, _ratio(r, r_lw) - 1))
    isolated = all(abs(fr) < 0.10 and rr > -0.15 for _, fr, rr in others)
    fill_drop = m["fill_ratio"] <= 0.80
    if fill_drop:
        diag = (f"{src} receives its usual ad requests ({m['requests_ratio'] - 1:+.0%}) but fills only "
                f"{m['fill_rate']:.0%} instead of {m['fill_rate_baseline']:.0%}: ")
        diag += ("demand is not reaching this source (bid-request/SSP integration or a blocklist), "
                 "since the other sources fill normally." if isolated else
                 "the drop is marketplace-wide, so it is demand-side (fewer bids/budgets), not this source.")
        hyps = ([("Broken SSP/bid-request integration or timeout on this source", "high",
                  "Check bid-request error rates and timeouts for this source since the drop started"),
                 ("A new blocklist/brand-safety rule excludes this source's inventory", "medium",
                  "Check recent targeting/blocklist changes affecting this source")] if isolated else
                [("Advertiser demand fell (budgets exhausted, campaigns paused)", "medium",
                  "Compare active campaigns and bids vs last week")])
    else:
        diag = (f"{src} receives {m['requests_ratio'] - 1:+.0%} ad requests vs last week: "
                + ("the publisher's traffic or integration dropped." if isolated else
                   "all sources dropped, which points to a seasonal/event effect."))
        hyps = [("Publisher traffic loss or SDK/tag outage", "medium" if isolated else "low",
                 "Ask the publisher for their traffic; check SDK version/error rates")]
    return _out(
        "analytics", aid, diag,
        [f"other sources, same window (fill change / request change): "
         + ", ".join(f"{s} {fr:+.0%} / {rr:+.0%}" for s, fr, rr in others),
         f"~{m['lost_impressions']:,.0f} impressions not delivered in {int((now - start) / HOUR) + 1}h "
         f"at EUR {m['cpm']:.2f} avg advertiser CPM"],
        hyps,
        _act(f"Open a supply incident for {src} and move affected campaigns' demand to the other sources", 2, True,
             "restores delivery for advertisers while the source is fixed"),
        "act_now" if isolated else "investigate",
        ["bid-request / SSP error logs", "publisher-side traffic numbers", "change log for targeting and blocklists"],
    )


RULES = {
    ("ads", "S2"): ads_s2,
    ("attribution", "S2"): attribution_s1,
    ("analytics", "S8"): analytics_s8,
    ("conversion-tracking", "S4"): conversion_tracking_s4,
    ("analytics", "S4"): analytics_s4,
    ("ads", "S1"): ads_s1,
    ("attribution", "S1"): attribution_s1,
    ("marketing-psychology", "S6"): marketing_psychology_s6,
    ("ads", "S6"): ads_s6,
    ("analytics", "S7"): analytics_s7,
}


def run(skill: str, det: Detection, data: MarketData, aid: str) -> dict:
    fn = RULES.get((skill, det.signal))
    if fn is None:
        return _out(skill, aid, None, [], [], None, "investigate",
                    [f"no rule-based method for {skill} on {det.signal}"])
    return fn(det, data, aid)
