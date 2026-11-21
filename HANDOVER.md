# HANDOVER – Eye of the System (Almedia track)

Status: Sat 21.11.2026, 12:50. Read this first. Details: `docs/EYE.md` (architecture, German).

## 1. The challenge in one sentence
An agent that monitors a marketplace on its own: decides what to watch, detects real anomalies,
investigates the cause through a metric graph, reports with evidence, and tracks incidents until resolved.
Focus from the CEO: **what is an anomaly, and how is it resolved.**
Prize: 700 EUR. Submission: Sun 22.11, 17:45. Judging 18:00, pitches 19:00.

## 2. Setup (Windows, PowerShell)
```powershell
cd hacklab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-llm.txt
python src\eye\gen_data.py          # create DEMO DATA (data/eye/, git-ignored)
python src\eye\evaluate.py          # scorecard, no API calls, free
```
Optional Claude review: put `ANTHROPIC_API_KEY=...` in `.env` (git-ignored, never commit, never paste in chat).
Our API credits are currently EMPTY (-0.40 USD) – everything runs without them (deterministic fallback).

## 3. What exists (src/eye/)
| File | What it does |
|---|---|
| `gen_data.py` | 14 days hourly, funnel x country x platform x category, inventory, change log, completeness. 5 scenarios + decoy releases. Ground truth in `data/eye/injected.json` |
| `store.py` | data access; ratios = ratio of sums; `asof(ts)` = data as it looked at that hour |
| `graph.py` | metric graph from `config/eye/metrics.yaml`; edges: formula, composite, breakdown, possible_cause; each edge validated on data |
| `detect.py` | expected = median of same hours, last 7 days (weekday-normalised); noise = 1.4826 x MAD (floored); anomaly: z>=4.5, change>=10%, volume>=100, 2 windows (1 if z>=8); incomplete data = DATA notice |
| `monitor.py` | per tick: all metrics at total + 70 segment checks ranked by GMV share x recent trouble x staleness; max 2 investigations/tick; incidents open -> persisting -> resolved (2 normal checks), dedupe, reopen |
| `investigate.py` | WHERE (segment: explanatory power + surprise, joint check e.g. DE x android, Kitagawa mix/rate) -> WHAT (formula factor) -> WHY (change log, control groups, cause edges); explicit stop rule; evidence ledger q1, q2, ... |
| `extend.py` | new metric places itself in the graph (examples: `config/eye/new_metrics/*.yaml`) |
| `agent_llm.py`, `llm.py` | Claude reviews the ledger: max 1 call/incident, Sonnet default, must cite ledger ids, cannot overwrite |
| `run.py` | replay / live tick. `evaluate.py` = scorecard vs ground truth |

Commands:
```powershell
python src\eye\show_graph.py                                   # graph as tree
python src\eye\extend.py config\eye\new_metrics\zero_result_rate.yaml
python src\eye\run.py --from "2026-11-16 00:00" --quiet        # replay
python src\eye\evaluate.py                                     # scorecard
```

## 4. Scenarios and current score
| Scenario | Expected | Status |
|---|---|---|
| payment_failure (DE android, payments-sdk 7.4.0) | ALERT + release | OK |
| fr_campaign (planned, conv rate down, orders up) | INFO only | OK |
| search_relevance (DE, search-ranker 2.3) | ALERT + release | OK |
| stockout (electronics, Voltix PO #4471 delayed) | ALERT + delivery | **REGRESSION: stops at category=electronics** |
| late_orders_data (40% rows) | DATA notice, no page | OK |

Score: 4/5 fully correct (was 5/5 before the last investigate.py change), 0 false alarms, median time-to-detect 2h.

**Not pushed yet** (only on Marco's laptop until committed): big investigate.py upgrade, gen_data.py, graph.py, store.py, metrics.yaml, docs/EYE.md, this file.

## 5. Open work, in priority order
1. **Fix stockout regression** -> back to 5/5 on seeds 7 and 11. (owner: Giannis – investigation)
2. **Baseline protection**: hours of an open incident must not enter the 7-day median, otherwise a fault lasting >=4 days becomes "normal" and auto-resolves. Test: stockout extended to 7 days must NOT resolve. (owner: Marco – monitoring)
3. **Resolution states** instead of "2 normal checks": fixed (recovery follows a fix/rollback event) | recovered_unexplained | new_normal (human confirms, re-baseline) | still_open. (owner: Marco)
4. **Anomaly classes + EUR impact**: noise | data_issue | expected | innocent_change | incident | positive_anomaly; impact in EUR/h via the graph; prioritise by impact x surprise. (owner: Marco)
5. **Confidence labels**: descriptive | associational | causal_evidence (only with diff-in-diff effect + interval) | inconclusive. (owner: Giannis)
6. **Relationship anomalies**: connected metrics moving in opposite directions (feed clicks up, purchases down). New scenario "For You" + "orders up, contribution margin down". (owner: Nikos – graph/data)
7. **Dashboard** (single offline HTML: graph tree, incident feed, investigation trail, ground-truth toggle, scorecard) + `DEMO.md` 3-min script. (owner: Nikos)
8. **METHOD.md** (2 pages): how we choose what to monitor, which edges we follow, which groups we examine, when evidence is enough. (owner: Marco, last)

Definition for the pitch: an anomaly is a change that is **real, unexpected, unexplained and matters**.
Resolved means we can show **why** it is over (fixed / recovered / new normal) – not that the baseline got used to it.

## 6. Rules (to save tokens and avoid conflicts)
- **Claude Code on your Claude.ai Pro login, NOT the Console/API account** (API credits are gone). Check with `/status`.
- Use **Sonnet** (`/model`) for normal work; Opus only for hard bugs.
- **One task per session**, then `/clear`. Long sessions re-read everything and burn tokens.
- Give Claude Code exact file names (`src/eye/detect.py`), not "look through the project".
- Always test with `python src\eye\evaluate.py` (free). Use `run.py --llm` only for the final demo.
- **Each person owns different files** (see owners above). Pull before you start, commit + push small and often.
- Never commit `.env`, `data/`, `.agents/eye/llm_calls.jsonl`.
- Ignore the old ads monitor (`src/monitor/`, `src/replay.py`, `.agents/monitoring/`) – not part of the Eye submission.
