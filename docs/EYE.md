# Eye of the System: Architektur

Agentisches Monitoring für einen Marketplace (Almedia-Track). Code in `src/eye/`, Katalog in `config/eye/`,
synthetische Daten in `data/eye/` (DEMO DATA), Laufzeit-State in `.agents/eye/`.

## Datenfluss

```
metrics.yaml ──► graph.py ──► Metrik-Graph (typisierte Kanten, numerisch validiert)
(+ added_metrics.yaml)   ▲                     │
                         │                     ▼
neue Metrik ──► extend.py (Extension-Agent)   monitor.py  ── pro Stunde ──►  detect.py
                                               │  Attention-Policy            erwartet vs. ist
                                               │                              (Saison, Vollständigkeit)
                                               ▼
                                         investigate.py  ──► Evidence Ledger (q1, q2, ...)
                                               │  WHERE / WHAT / WHY
                                               ▼
                                         Incident (offen → besteht → gelöst)
                                               │
                                               ├─► agent_llm.py: Claude prüft das Ledger, 1 Call pro Incident (optional)
                                               └─► run.py / evaluate.py: Bericht, Scorecard
store.py liegt unter allem: jede Zahl kommt aus Rohsummen der Tabellen, für jeden Filter und jedes Zeitfenster.
```

## Bausteine

| Datei | Aufgabe |
|---|---|
| `gen_data.py` | DEMO DATA: 14 Tage stündlich, Funnel × Land × Plattform × Kategorie, Inventory, Change-Log, Datenvollständigkeit. Injiziert 3 Fehler, 1 harmlose Änderung, 1 Datenproblem, dazu harmlose Releases als konkurrierende Erklärungen. |
| `store.py` | Datenzugriff. Ratios sind immer Ratio der Summen. `asof(ts)` gibt die Sicht „Stand dieser Stunde“ (für Replay und Live-Loop). |
| `graph.py` | Leitet den Graphen aus den Metrik-Definitionen ab und validiert jede Formel gegen die Daten. |
| `extend.py` | Nimmt eine neue Metrik (Name, Beschreibung, Formel) auf, ohne dass jemand ihren Platz angibt. |
| `detect.py` | Erwartungswert (Median derselben Stunden der letzten 7 Tage, Wochentag herausgerechnet), Rauschmaß, Persistenz, Vollständigkeit. |
| `investigate.py` | Autonome Untersuchung einer Abweichung bis zur gestützten Ursache oder ehrlichen Hypothese. |
| `monitor.py` | Tick-Schleife: was beobachten, was ist neu, was ist schon erklärt, Incident-Lebenszyklus. |
| `agent_llm.py`, `llm.py` | Claude als Prüfer des fertigen Ledgers. Kostenlog in `.agents/eye/llm_calls.jsonl`. |
| `run.py` | Replay eines Zeitraums oder ein Tick (`--tick`), mit Kostenanzeige. |
| `evaluate.py` | Replay über alle Daten und Vergleich mit `data/eye/injected.json` (Scorecard). |

## Der Metrik-Graph

Kantentypen, und welche Art Schluss jede erlaubt:

- **formula**: exakte Arithmetik, z. B. `orders = payment_attempts × payment_success_rate`. Eine Änderung zerfällt
  exakt in Log-Beiträge der Faktoren. Wird aus den Definitionen abgeleitet, nie von Hand eingetragen.
- **composite**: eine Metrik ist das Produkt mehrerer anderer, z. B. `conversion_rate` = Produkt der sechs
  Funnel-Raten. Der Agent findet das durch Ablaufen der Formelkette und akzeptiert es nur, wenn es auf den Daten stimmt.
- **breakdown**: Metrik nach Dimension (Land, Plattform, Kategorie). Summen sind additiv, Ratios zerfallen in
  Mix- und Rate-Effekt.
- **possible_cause**: Hypothesen-Kante (Lagerbestand → Cart-Rate, Change-Log-Komponente → Metrik). Wird in der
  Untersuchung mit Timing und Segment getestet, nie als Rechnung benutzt.

Jede Formel-, Composite- und Breakdown-Kante trägt ihr Residuum aus der Validierung. Eine Definition, die nicht
mehr zu den Daten passt, fällt damit auf, statt stillschweigend benutzt zu werden.

## Automatische Erweiterung (`extend.py`)

Schritte, jeweils mit Beleg protokolliert: Parsen und Schema-Check, Normalisieren (`sum(spalte)` wird zur bestehenden
Metrik oder zu einer neuen Basismetrik), Formelkanten, Composite-Suche, Duplikat-Check (numerisch identisch zu einer
bestehenden Metrik → abgelehnt), Dimensionen und Additivität, datengetriebene Links (Korrelation der Abweichungen
zur Vorwoche mit Lead/Lag, als **mehrdeutig** markiert), Change-Log-Komponenten, Aufnahme ins Monitoring.

## Monitoring (`monitor.py`, `detect.py`)

- **Was wird beobachtet:** jede Metrik auf Gesamtebene, dazu ein Budget an Segment-Checks, gewichtet nach
  GMV-Anteil, kürzlichem Ärger im Segment, Zeit seit dem letzten Check und Neuheit der Metrik.
- **Was ist eine Abweichung:** deutlich außerhalb des Rauschens und über zwei Fenster stabil (sehr starke
  Abweichungen sofort). Unvollständige Quellen gelten nicht als Abweichung, sondern als Datenhinweis.
- **Was ist schon erklärt:** eine Abweichung auf der Kette eines offenen Incidents, oder eine, die verschwindet,
  wenn man das Segment des Incidents herausrechnet, wird dem Incident zugeordnet und nicht neu gemeldet.

## Untersuchung (`investigate.py`)

1. **Daten prüfen:** Vollständigkeit der Quellen, Gegenprobe auf einer vollständigen Quelle.
2. **WHERE:** pro Dimension Explanatory Power (Anteil am Gap) und Surprise je Wert (nach Adtributor). Zeigen zwei
   Dimensionen je auf einen Wert, wird die Kombination direkt gegen ihre Kontrollzellen getestet (Schutz vor
   Simpson-Effekten). Ratios nutzen die symmetrische Kitagawa-Zerlegung; Mix + Rate werden gegen das direkt
   berechnete Gesamt-Delta geprüft, ein Residuum kommt als Warnung ins Ledger.
3. **WHAT:** entlang der Formelkanten zum Faktor, der die Änderung trägt, bis zu einer Blattmetrik.
4. **WHY:** Change-Log-Events (Komponente, Segment, Timing, Kontrollgruppen) und Ursache-Kanten testen.
5. **Geschäftsprüfung:** Steigt die auslösende Metrik, prüft der Agent den Deckungsbeitrag im selben Segment.
   Fällt dieser deutlich, untersucht er die Marge statt den Anstieg (Szenario „Orders hoch, Marge runter“).
   Summen wie `contribution_margin = gmv − discounts − cogs − ad_cost` zerfallen in einen Volumen-Effekt und einen
   Raten-Effekt je Kostenblock; jeder Treiber bekommt einen eigenen Zweig bis zu seinem Blatt.
6. **Difference-in-Differences je Blatt:** betroffenes Segment gegen den Rest, 24h vor dem Onset gegen die Stunden
   danach, mit 95%-Intervall im Ledger.
7. **Stopp-Regel, im Ledger begründet, pro Zweig:** (a) führende Erklärung hat mindestens zwei unabhängige Belege,
   (b) Alternativen geprüft und widerlegt, (c) Daten können nicht unterscheiden → Hypothese mit Unsicherheit.
8. **Ergebnis:** Klassifikation (fault, innocent, data_incomplete, unexplained), Kette Symptom → Ursache,
   Auswirkung auf Orders, GMV und Deckungsbeitrag, Evidenzstufe (`descriptive | associational |
   causal_evidence | inconclusive`, siehe `docs/METHOD.md`), alle Changes im selben Zeitfenster mit Owner,
   `deployed_at` und `first_exposure_at`, offene Unsicherheit.

## Gegen das Auseinanderlaufen von Agent und Realität

- Jede Aussage hängt an einer Abfrage im Evidence Ledger. Es gibt keinen frei fortgeschriebenen Text als Zustand.
- Jeder Tick leitet alles neu aus den Daten ab; gemerkt werden nur Incidents, nicht Schlussfolgerungen.
- Der Graph wird gegen die Daten validiert, nicht geglaubt.
- Claude sieht nur das Ledger, darf keine Zahlen hinzufügen und muss Ledger-IDs zitieren. Zitate, die es nicht
  gibt, werden verworfen. Claude kann dem Ergebnis widersprechen, es aber nicht überschreiben.

## Befehle

```powershell
python src\eye\gen_data.py                 # DEMO DATA erzeugen
python src\eye\show_graph.py               # Graph als Baum, mit Validierung
python src\eye\extend.py config\eye\new_metrics\zero_result_rate.yaml   # neue Metrik aufnehmen
python src\eye\run.py --from "2026-11-16 00:00" --quiet                 # Replay
python src\eye\run.py --from "2026-11-17 00:00" --quiet --llm           # mit Claude-Review, zeigt Kosten
python src\eye\run.py --tick                                            # ein Tick für den Live-Loop
python src\eye\evaluate.py                                              # Scorecard (ohne Claude, kostenlos)
```

API-Key: `ANTHROPIC_API_KEY=...` in `.env` im Repo-Root (gitignored). Modell über `EYE_MODEL`, Standard ist
`claude-sonnet-5-5`.
