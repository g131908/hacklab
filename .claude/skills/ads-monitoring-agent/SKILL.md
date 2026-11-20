---
name: ads-monitoring-agent
description: "Use when building, running, or demoing an agent that monitors an ads marketplace app: detecting spend, performance, tracking, fraud, and supply-side anomalies on a loop, triaging them, routing to specialist skills (ads, analytics, conversion-tracking, attribution, marketing-psychology), and staging decisions for human approval."
metadata:
  version: 0.1.0
  built_for: HackLab Berlin, Nov 2026 (software track: agentic monitoring for an ads marketplace app)
  builds_on: marketing-loops, ads, analytics, conversion-tracking, attribution, marketing-psychology, chief-of-staff (routing pattern only)
---

# Ads Monitoring Agent

You run an **always-on monitoring loop** for an ads marketplace: advertisers buy placements, the app serves them to users, and money, traffic, and conversions flow between both sides. Your job is to notice when something breaks or drifts, figure out *why*, and hand a human a decision, not a wall of charts.

Core pipeline, every run:

```
WATCH -> SELF-CHECK -> TRIAGE -> ROUTE -> SYNTHESIZE -> GATE -> LOG
(loop)   (is it real?)  (score)   (skills)  (one alert)  (human)  (state)
```

Most runs should end at WATCH or SELF-CHECK with "checked, nothing to do." That is the system working, not failing.

## 0. Before the first run

1. Read `.agents/monitoring/config.yaml` if it exists. If it doesn't, create it from the template in section 10 and ask only for what's missing: data source(s), accounts/campaigns in scope, currency, timezone, spend caps, and who approves.
2. Confirm the data source. In order of preference: a connected MCP/API, a database/warehouse query, a CSV/JSON export, or the **demo data generator** (section 11). Never invent numbers. If data is missing, say what is missing.
3. Confirm the kill switch exists (section 9) before scheduling anything.

## 1. WATCH: the signal catalog

Each signal has a cadence matched to how fast it really moves (rule from `marketing-loops`: over-frequent checks are the most common failure). Thresholds are **starting defaults**; tune them in config.

| # | Signal | Side | Cadence | Fires when (default) | Severity floor |
|---|--------|------|---------|----------------------|----------------|
| S1 | **Spend pacing** | Advertiser | Hourly | Spend-to-date > 130% or < 60% of expected pace for this hour | 3 |
| S2 | **Budget cap breach risk** | Advertiser | Hourly | Projected daily spend > hard cap | 5 |
| S3 | **CPA / ROAS drift** | Advertiser | Every 2-3 days | CPA +30% or ROAS -25% vs 7-day baseline, with >= 30 conversions in window | 3 |
| S4 | **Tracking break** | Both | Hourly | Conversions = 0 (or -80%) for >= 3h while clicks/impressions are within normal range | 4 |
| S5 | **Conversion double-count** | Both | Daily | Conversions/clicks ratio jumps > 2x baseline, or duplicate transaction IDs | 3 |
| S6 | **Creative fatigue** | Advertiser | Every 2-3 days | CTR -25% over 3 days while frequency rises | 2 |
| S7 | **Invalid traffic / fraud** | Supply | Hourly | CTR > baseline + 3 sigma, click-to-conversion time < 10s clusters, single IP/device/ASN concentration > 20% of clicks | 4 |
| S8 | **Fill rate / inventory drop** | Supply | Hourly | Fill rate -20% vs same hour last week, or ad requests -30% | 3 |
| S9 | **Landing page / app-store regression** | Advertiser | Daily | LP error rate up, load time +50%, or LP conversion rate -30% with stable traffic | 3 |
| S10 | **Revenue / payout mismatch** | Marketplace | Daily | Advertiser billed spend vs publisher payouts vs platform take deviates > 2% from expected margin | 5 |

Adapt the catalog to the actual marketplace data. Drop signals you can't measure; don't fake them.

## 2. SELF-CHECK: is it real?

Before raising anything, rule out the boring explanations. Record which checks ran.

- **Data freshness**: is the latest data point recent? Stale pipelines look like drops. A stale source is itself an alert (severity 3, type `data_pipeline`), not a performance problem.
- **Minimum sample**: below the config's minimum volume, label it "insufficient signal" and don't alert. (40 visitors a week is noise.)
- **Seasonality**: compare against the same weekday/hour, not just yesterday. Check config for known events (holidays, launches, sales).
- **Tracking vs reality**: if conversions fall but clicks, sessions, and backend orders/installs don't, suspect tracking (S4) before performance (S3). Use the `conversion-tracking` skill's discrepancy logic: platform numbers never match each other exactly, and that alone isn't an anomaly.
- **Known changes**: was there a deploy, campaign edit, bid change, or creative swap in the window? Check the change log if one exists.
- **Dedupe**: is this the same anomaly already open in state (section 8)? If yes, update it; don't re-alert.

If a self-check can't be completed, say so in the alert and lower confidence. Never silently skip it.

## 3. TRIAGE: score it

Adapted from the chief-of-staff complexity score. Start from the signal's severity floor, then add:

| Factor | +1 if... |
|--------|----------|
| Money at risk | > the config's `money_at_risk_threshold` per day if unaddressed |
| Cross-side impact | Affects both advertisers and supply/publishers |
| Multi-domain | Needs 2+ specialist skills to explain |
| Irreversible / external | Fix would touch live campaigns, billing, or users |
| Strategic account | Account is on the config's `strategic_accounts` list |

Cap at 5. Then:

- **1-2: Log only.** Goes in the daily digest, no interrupt.
- **3: Single specialist.** Route to one skill, stage one recommendation.
- **4: Panel.** Route to 2-3 specialists in parallel, synthesize, and notify now.
- **5: Page a human.** Notify immediately with the facts you have, before the diagnosis is finished. Never self-correct (see the always-escalate list in section 7).

## 4. ROUTE: which skill explains it

Load the specialist skill and apply its method to *this* anomaly. If subagents are available and the score is 4+, run specialists in parallel. Each one gets only the anomaly record and the relevant data slice.

| Anomaly type | Primary skill | Secondary | Question to answer |
|--------------|---------------|-----------|--------------------|
| Spend pacing, budget, CPA/ROAS drift | `ads` | `attribution` | Is this a bid/budget/auction problem, or a measurement one? Kill, hold, or scale? |
| Tracking break, double-count | `conversion-tracking` | `analytics` | Which tag/pixel/event broke, and since when? What's the real number? |
| Metric definitions, funnel math | `analytics` | | Is the metric computed correctly? Is the baseline valid? |
| Channel credit / "is it really working" | `attribution` | `ads` | Is the drop real revenue loss or a credit shift? |
| Creative fatigue, CTR/CR drop on stable traffic | `marketing-psychology` | `ads` | Which 2-3 behavioral principles plausibly explain it? What's the testable fix? |
| Fraud / invalid traffic | (no skill: use section 1 S7 heuristics) | `analytics` | Which sources/placements? How much spend is affected? |
| Fill rate, supply, payouts | (no skill: marketplace logic) | `analytics` | Demand-side or supply-side cause? |

**Loop prevention** (from chief-of-staff): max routing depth 2. A specialist may consult one other skill, but never route back to the monitor, and specialists don't call each other in a circle. If depth would exceed 2, stop and escalate with what you have.

### Specialist output contract

Every specialist returns exactly this shape so synthesis stays mechanical:

```json
{
  "skill": "conversion-tracking",
  "anomaly_id": "S4-2026-11-21-acct42",
  "diagnosis": "one sentence",
  "evidence": ["fact with number and source", "..."],
  "hypotheses": [{"cause": "...", "confidence": "high|medium|low", "how_to_verify": "..."}],
  "recommended_action": {"action": "...", "tier": 1, "reversible": true, "est_impact": "..."},
  "missing_data": ["..."]
}
```

Keep facts, hypotheses, and missing data separate (rule from `ads` audit guardrails). Never present a hypothesis as a fact.

## 5. SYNTHESIZE: one alert, one decision

Merge specialist outputs into a single alert. Surface disagreements instead of hiding them: if `ads` says "scale back" and `attribution` says "the drop is a credit shift, hold," that conflict *is* the alert's decision point.

Alert format:

```
[SEV 4] Tracking break: Advertiser "acct42" - conversions 0 since 09:00
What happened: 0 conversions in 4h vs ~38 expected; clicks normal (1,210 vs 1,180 baseline).
Why (most likely): Purchase event stopped firing after 08:52 deploy (conversion-tracking, high confidence).
Money at risk: ~EUR 640/day of spend optimizing against no signal; smart bidding will degrade within 24-48h.
Specialists disagree on: (none) | or: ads says pause, attribution says hold - because ...
Decision for you:
  A) Pause automated bidding on acct42 until fixed (reversible, stops wasted spend)
  B) Keep running, fix the tag first (risk: bidding learns from bad data)
Recommended: A. Owner: on-call. Re-check: in 2h.
Checks run: freshness OK, sample OK, seasonality OK, recent deploy FOUND.
```

Rules: lead with the number, one recommended option, at most 5 owned actions, every number has a source and a time window.

## 6. GATE: what the agent may do on its own

Two-tier action model (from `marketing-loops` guardrails):

**Tier 1, autonomous:** read, query, compare, score, diagnose, draft recommendations, stage changes, write logs, send internal notifications.

**Tier 2, gated (human approval required):** pause/resume campaigns, change bids or budgets, move spend, block placements/publishers, change tracking config, contact advertisers or publishers, issue refunds/credits, anything in billing.

A Tier-2 action may run unattended **only** if the config explicitly authorizes that action type, the target is on the allowlist, and caps hold:
- per-run budget change <= `max_budget_change_pct` (default 20%)
- never exceed `hard_daily_spend_cap`
- judge paid changes on revenue/ROAS, not CTR/CPA alone

## 7. Always escalate (never self-correct)

- Revenue or spend anomalies of severity 5 (S2, S10)
- Suspected fraud affecting payouts or billing
- Strategic accounts
- Anything that deletes data, issues money, or contacts many users at once
- Any case where a self-check failed and the proposed action is Tier 2

## 8. LOG: state and idempotency

State lives in `.agents/monitoring/`:

```
.agents/monitoring/
  config.yaml          # thresholds, caps, allowlist, approvers, timezone
  baselines.json       # rolling baselines per signal x account (updated each run)
  open-anomalies.json  # dedupe set: anomaly_id -> status, first_seen, last_seen, severity
  decisions.md         # approved/rejected decisions with timestamp, approver, review date
  runs.log             # one line per run: time, signals checked, fired, suppressed, errors
  KILL                 # if this file exists, do nothing and exit (kill switch)
```

- **anomaly_id** = `<signal>-<date>-<account>`. The same ID never alerts twice. Update `last_seen` and only re-notify if severity rises.
- **Cooldown**: after an alert, the same signal x account is silent for `cooldown_hours` (default 6) unless severity increases.
- **Auto-resolve**: if the signal is back in range for 2 consecutive checks, mark it resolved and note it in the digest.
- **No raw PII** in state or logs. Use IDs or hashes.
- At session end, append decisions and anything learned (e.g. "acct42 deploys on Tuesdays, expect tracking blips") to `decisions.md`.

## 9. Kill switch and scheduling

- The kill switch is checked first, every run: `if exists .agents/monitoring/KILL -> log "killed" and exit`.
- **Claude Code**: use `/loop` for the hourly watch during a demo, or a cron schedule for real use. Hourly signals and 2-3 day signals are separate schedules, not one fast loop.
- **Anything else**: cron calling the same prompt/script.
- On error: log it, don't retry more than once, and alert severity 3 `monitor_error`. A silent monitor is worse than a noisy one.

## 10. Config template

```yaml
marketplace: "<name>"
timezone: "Europe/Berlin"
currency: "EUR"
data_sources:
  - type: "csv|api|mcp|sql|demo"
    location: "<path/url/connector>"
scope:
  accounts: ["*"]              # or explicit list
  strategic_accounts: []
thresholds:                     # override section 1 defaults
  min_conversions_window: 30
  money_at_risk_threshold: 500  # per day
caps:
  hard_daily_spend_cap: 10000
  max_budget_change_pct: 20
autonomy:                       # Tier-2 actions allowed without approval (default: none)
  allowed_actions: []
  allowlist: []
cooldown_hours: 6
notify:
  channel: "stdout|slack|email|file"
  approvers: ["<name>"]
known_events: []                # holidays, launches, planned changes
```

## 11. Demo mode (hackathon)

When there's no live data, build and use a synthetic dataset. Label every output "DEMO DATA".

1. Generate 14 days of hourly data for 5-10 advertisers and 3-5 supply sources: impressions, clicks, conversions, spend, revenue, fill rate, ad requests, payouts. Add realistic daily and weekly seasonality and noise.
2. Inject one anomaly per scenario at a known timestamp, so the demo can prove detection:
   - tracking break (conversions to 0, clicks normal) -> S4
   - click fraud burst from one source (CTR spike, sub-10s conversions) -> S7
   - runaway spend (pacing 180%) -> S1/S2
   - creative fatigue (CTR slide plus rising frequency) -> S6
   - one **false alarm** (a holiday dip listed in `known_events`) to show the self-check suppressing it
3. Replay the data hour by hour through the pipeline, and show: detected, suppressed, triaged, routed, the alert, and the human decision.
4. Report precision on the injected set: N injected, N detected, N false alarms, time-to-detect.

The false-alarm scenario is the strongest demo moment. It shows the agent thinks before it pages someone.

## Anti-patterns

- **Dashboard-in-a-loop**: alerts nobody acts on. Every alert ends in a decision or it's a digest line.
- **Alerting on noise**: skipping the minimum sample or the seasonality check.
- **Optimizing a proxy**: "CTR is up" while revenue falls.
- **Self-correcting money**: auto-pausing or moving budget without explicit authorization and caps.
- **Re-alerting**: no dedupe or cooldown, so the on-call mutes the channel.
- **One cadence for everything**: CPA checked hourly is noise. Pacing checked daily is too late.
- **Invented numbers**: if data is missing, the alert says "missing," not an estimate.

## Related skills

- `marketing-loops`: loop anatomy, cadence rule, guardrails. Read `references/loop-guardrails.md` and `references/loop-state.md` before going live.
- `ads`: `references/operating-check.md` for the weekly decision review, and `references/audit-guardrails.md` for facts vs hypotheses.
- `conversion-tracking`: `references/discrepancies.md` for why numbers never match.
- `analytics`, `attribution`: metric correctness and channel credit.
- `marketing-psychology`: behavioral diagnosis for creative or CTR/CR drops (outputs are hypotheses to A/B test).
- `chief-of-staff`: source of the triage/routing/synthesis pattern; not required at runtime.
- `marketing-council`: for strategic "what should we do about this trend" questions. It's not for incident response.
