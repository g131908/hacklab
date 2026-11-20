# HackLab Berlin – Ads Marketplace Monitoring Agent

Agent, der einen Ads-Marktplatz (Advertiser + Supply/Publisher) überwacht, Anomalien erkennt,
erklärt *warum* und dem Menschen pro Alarm **eine** Entscheidung vorlegt.
Pipeline nach `.claude/skills/ads-monitoring-agent/SKILL.md`:
WATCH -> SELF-CHECK -> TRIAGE -> ROUTE -> SYNTHESIZE -> GATE -> LOG.

## Setup (Windows, PowerShell oder CMD)

```powershell
python -m venv .venv
.\.venv\Scripts\activate          # CMD: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

Falls PowerShell das Aktivieren blockiert:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Optional, für Specialist-Diagnosen per Claude API:

```powershell
pip install -r requirements-llm.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # CMD: set ANTHROPIC_API_KEY=sk-ant-...
```

## Specialist-Routing (hybrid)

Jeder Alarm ab SEV 3 wird an die passenden Skills geroutet (SEV 3: ein Specialist, SEV 4-5: Panel parallel):

| Signal | Specialists |
|---|---|
| S1 Spend pacing | ads, attribution |
| S4 Tracking break | conversion-tracking, analytics |
| S6 Creative fatigue | marketing-psychology, ads |
| S7 Invalid traffic | analytics |

- `routing.mode` in der Config bzw. `--routing`: `rules` (offline, deterministisch), `llm` (Claude mit der jeweiligen
  `SKILL.md` als System-Prompt, Structured Output im Specialist-Contract), `auto` (llm wenn `ANTHROPIC_API_KEY` gesetzt, sonst rules).
- Schlägt ein LLM-Call fehl (kein Key, Netz, Refusal), fällt genau dieser Specialist auf die Regeln zurück.
- LLM-Modus kostet pro Specialist-Call; der Demo-Replay macht 7 Calls (4 Alarme).
- Widersprechen sich Specialists (z. B. `act_now` vs. `hold`), steht das als Entscheidungspunkt im Alarm.
- Test ohne API-Calls: `python tests\test_routing.py`

## Ausführen

```powershell
python src\generate_demo_data.py   # 14 Tage stündliche DEMO DATA -> data\*.csv + data\injected.json
python src\replay.py               # Stunde für Stunde abspielen, Alarme + A/B-Entscheidung bei SEV>=4
python src\replay.py --auto A      # ohne Rückfrage, wählt immer Option A (für Tests/CI)
python src\evaluate.py             # injected / detected / false alarms / time-to-detect
python src\dashboard.py --open     # Demo-Dashboard -> dashboard\index.html (offline, Doppelklick reicht)
```

## Demo-Dashboard

`dashboard\index.html` ist eine einzelne Datei ohne Internet-Abhängigkeit. Sie wird aus dem letzten Replay erzeugt,
also nach jedem `replay.py` neu bauen.

- **Play** spielt die 14 Tage ab (Tempo 6–48 h/s). Bei SEV ≥ 4 pausiert es mit der A/B-Entscheidung
  (bei SEV 5 mit Paging-Banner). Die Wahl wird nur angezeigt, nichts wird ausgeführt.
- **Next event** springt zum nächsten Alarm oder zur nächsten Unterdrückung, ideal für eine kurze Demo.
- **Show ground truth** blendet die injizierten Szenarien ein, und zwar nur bis zur aktuellen Replay-Zeit.
- Deep Links: `index.html#t=end`, `index.html#t=2026-11-18T07:00` (Feiertag, Unterdrückung).
- Tasten: `Space` Play/Pause, `←`/`→` ±1 h, `N` nächstes Event, `Esc` Dialog schließen.
- Jede Grafik hat eine Tabellenansicht, Hell-/Dunkelmodus folgt dem System.
- **Agent-Trace** pro Alarm (und pro Gruppe unterdrückter Detektionen) als Zeitleiste unter den Fokus-Charts:
  Detektor → Self-Checks mit Einzelergebnis → Triage-Score mit Faktoren → Dedupe → Routing mit Leitfragen
  → Specialist-Antworten (Fakten, Hypothesen, Empfehlung, fehlende Daten) → Synthese → Gate → Log, jeweils mit
  Status und gemessener Dauer. Bei unterdrückten Detektionen sieht man, an welchem Check die Pipeline stoppt.
  Der Trace steht auch im Log (`trace`-Feld in `alerts.jsonl`).

Demo-Ablauf (ca. 3 Min.): Reset → Play → Tracking-Break (SEV 5, Entscheidung A) → Klickbetrug
→ Runaway Spend → Feiertag: „115 suppressed“, niemand wird gepaged → Creative Fatigue mit
Specialist-Widerspruch → End: Scorecard 4/4, 0 Fehlalarme.

## Struktur

```
src/
  generate_demo_data.py   synthetische Daten + 5 Szenarien (DEMO DATA)
  replay.py               stündlicher Replay durch die Pipeline
  evaluate.py             Vergleich Alarme vs. data/injected.json
  dashboard.py            baut dashboard/index.html aus Daten + Alarm-Log + Ground Truth
  dashboard_template.html Dashboard-UI (Vanilla JS/SVG, keine externen Abhängigkeiten)
  monitor/
    loader.py             Datenquelle (CSV, später austauschbar gegen API)
    detectors.py          S1 Pacing, S2 Budget-Cap-Risiko, S4 Tracking-Break, S6 Creative Fatigue,
                          S7 Fraud, S8 Fill-Rate/Inventory-Drop (S1 wird bei gleichzeitigem S2 in den S2-Alarm gemergt)
    selfcheck.py          Freshness, Min-Sample, Saisonalität, Known Events, Dedupe
    triage.py             Score 1-5
    routing.py            Routing an Specialist-Skills (hybrid, Depth-Guard, Panel parallel)
    specialists_rules.py  regelbasierte Methode je Skill x Signal
    specialists_llm.py    Claude-API-Specialist mit SKILL.md + Structured Output
    alert.py              Synthese: ein Alarm, eine Entscheidung (SKILL.md Abschnitt 5)
    pipeline.py           ein Monitoring-Lauf (geteilt von Replay und späterem Live-Loop)
    trace.py              Agent-Trace: ein Eintrag pro Pipeline-Schritt mit Status, Dauer, Details
    state.py              open-anomalies.json, Cooldown, KILL-Check
.agents/monitoring/
  config.yaml             Schwellen, Caps, Known Events
  open-anomalies.json     Dedupe-State (wird erzeugt)
  KILL                    falls vorhanden: Agent tut nichts (Kill Switch)
```

## Echte Challenge-Daten anschließen

1. Daten als CSV im gleichen Schema ablegen (siehe `src/monitor/loader.py`, `ADV_COLUMNS` / `SUP_COLUMNS`)
   oder in `loader.py` eine neue Quelle implementieren (`load_data` liefert zwei DataFrames).
2. `data_sources` in `.agents/monitoring/config.yaml` anpassen.
3. Schwellen und `known_events` in der Config tunen.
