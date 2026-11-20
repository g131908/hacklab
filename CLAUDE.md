# HackLab Berlin – Agentic Monitoring for an Ads Marketplace App

## Context
- Hackathon: HackLab Berlin (Nov 2026), software track: **agentic monitoring for an ads marketplace app**.
- Team lead: Marco. Communicate in German, keep answers short and direct.
- Environment: Windows 10, run commands so they work in CMD/PowerShell (or use Python for scripts).

## Goal
Build an agent that watches an ads marketplace (advertisers + supply/publishers), detects anomalies,
explains *why*, and hands a human ONE decision per alert. Nothing that spends, pauses, changes budgets,
or touches billing runs without human approval.

## Main skill
Use **ads-monitoring-agent** (.claude/skills/ads-monitoring-agent/SKILL.md) as the blueprint:
WATCH -> SELF-CHECK -> TRIAGE (score 1-5) -> ROUTE (specialist skills) -> SYNTHESIZE (one alert) -> GATE (human) -> LOG.

Specialist skills for routing: ads, analytics, conversion-tracking, attribution, marketing-psychology.
marketing-loops = loop anatomy, cadence, guardrails. chief-of-staff = routing pattern only.
marketing-council = pitch/positioning questions, not incident response.

## Build order (do not skip ahead)
1. Demo data: `src/generate_demo_data.py` -> 14 days hourly CSV in `data/`, 5 injected scenarios
   (tracking break S4, click-fraud burst S7, runaway spend S1/S2, creative fatigue S6, holiday false alarm).
2. Detectors: start with S4 and S7 incl. self-checks (freshness, min sample, seasonality, known events).
3. Triage + routing + alert in the format of SKILL.md section 5.
4. Human approval step (simple A/B choice in terminal is enough).
5. Only then: more signals, UI/dashboard, real challenge data.

## Folder layout
- `src/` code, `data/` datasets (CSVs are git-ignored), `.agents/monitoring/` config + state.
- Kill switch: a file `.agents/monitoring/KILL` stops the agent. Do not create it unless asked.

## Demo goal (end of day)
Replay data hour by hour: detection -> diagnosis -> alert with decision -> human approves.
Show the false alarm being suppressed by the self-check. Final slide: injected / detected / false alarms / time-to-detect.

## Rules
- Never invent numbers; if data is missing, say so. Label synthetic output "DEMO DATA".
- Small steps, run and test after each, commit working states.
