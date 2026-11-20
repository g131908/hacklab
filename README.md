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

## Ausführen

```powershell
python src\generate_demo_data.py   # 14 Tage stündliche DEMO DATA -> data\*.csv + data\injected.json
python src\replay.py               # Stunde für Stunde abspielen, Alarme + A/B-Entscheidung bei SEV>=4
python src\replay.py --auto A      # ohne Rückfrage, wählt immer Option A (für Tests/CI)
python src\evaluate.py             # injected / detected / false alarms / time-to-detect
```

## Struktur

```
src/
  generate_demo_data.py   synthetische Daten + 5 Szenarien (DEMO DATA)
  replay.py               stündlicher Replay durch die Pipeline
  evaluate.py             Vergleich Alarme vs. data/injected.json
  monitor/
    loader.py             Datenquelle (CSV, später austauschbar gegen API)
    detectors.py          S1 Pacing, S4 Tracking-Break, S6 Creative Fatigue, S7 Fraud
    selfcheck.py          Freshness, Min-Sample, Saisonalität, Known Events, Dedupe
    triage.py             Score 1-5 (+ Platzhalter fürs Routing an Specialist-Skills)
    alert.py              Alarm-Format (SKILL.md Abschnitt 5)
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
