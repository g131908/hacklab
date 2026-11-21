# METHOD – how the Eye decides

Short version of the reasoning rules. Architecture and file overview: `docs/EYE.md`. All numbers below come
from `src/eye/` (DEMO DATA).

**What is an anomaly?** A change that is **real** (not noise, not missing data), **unexpected** (not seasonality,
not a planned event), **unexplained** (not a symptom of an incident we already track) and **matters** (moves a
business outcome). **Resolved** means we can show why it is over, not that the baseline got used to it.

## 1. What we monitor

- Every metric at total level, every hour.
- Plus a budget of 70 segment checks per hour (country, platform, category), ranked by GMV share × recent
  trouble in the segment × time since the last check, with a boost for metrics added in the last 24h.
- A deviation needs |z| ≥ 4.5 and ≥ 10% change against the median of the same hours on the previous 7 days
  (weekday rhythm removed for counts), enough volume, and confirmation in a second window (unless |z| ≥ 8).
- A source that is less than 95% complete is a data notice, never an anomaly.

## 2. Which edges we follow

- **WHERE** (breakdown edges): per dimension, every value gets its explanatory power (share of the gap) and its
  surprise (Adtributor). If two dimensions each point at one value, the combination is tested directly against
  its control cells (Simpson guard). Ratios are split with the symmetric Kitagawa decomposition into a mix effect
  and a rate effect; the parts are checked against the directly computed total.
- **WHAT** (formula edges): products are split into exact log-contributions; we follow the factor that carries
  the change. Sums (contribution margin = gmv − discounts − cogs − ad_cost) are split into a **volume effect**
  and a **rate effect per term**; every term whose share of the base rose and carries ≥ 20% of the change is a
  driver with its own branch. Costs that only grew with volume are not drivers.
- **Business check:** if the trigger went *up*, the agent checks contribution margin in the same segment. If the
  margin is clearly down (≥ 10%, z ≤ −2.5), it pivots and investigates the margin instead of reporting a win.
- **WHY** (possible-cause edges and the change log): a change is a candidate for a leaf only if its component can
  move that leaf in the graph, its scope covers the affected segment, and its `first_exposure_at` lies 0..lag
  hours before the onset (lag 3h, campaigns 2h, wholesale 30h). Every change first exposed within 6h before the
  onset is tested and listed in the incident with owner, `deployed_at` and `first_exposure_at`.

## 3. Which groups we compare

- **Control groups:** the leaf metric in the other values of each localized dimension (other countries,
  platforms, categories), inside and outside the change's scope.
- **Difference-in-differences:** affected segment vs everything else, 24h before the onset vs the hours since.
  Effect on the log scale with a 95% interval from the hourly differences (Welch). Hour-of-day and day effects
  shared by both groups cancel out.

## 4. When evidence is enough (stop rule)

The agent never stops at the first anomaly: it always checks the data, localizes, follows the formula to a leaf
and tests every candidate. It then stops, and writes why, when for each leaf:

- **(a)** the leading explanation has ≥ 2 independent supports (timing fits, segment fits, control group
  normal, upstream metric moved, source incomplete), or
- **(b)** the alternatives were tested and refuted, or
- **(c)** the data cannot tell explanations apart: the leader is reported as a hypothesis, the rivals are named.

## 5. Evidenzstufen (evidence levels)

Every ledger step, every hypothesis and every incident carries a `confidence_label`. The incident takes the
weakest label among its supported causes.

| Label | Meaning | When the agent assigns it |
|---|---|---|
| `descriptive` | What happened, where and how much. No claim about why. | Measurements, localization, decomposition, control-group readings, impact; data-completeness notices; incidents with no candidate cause. |
| `associational` | A cause fits in time, place and mechanism, but its effect is not separated from the background. | A change-log entry or cause edge that passes component, scope and timing; a supported cause whose DiD interval includes 0 or cannot be estimated (no segment, no clear onset, fewer than 2 hours since onset). |
| `causal_evidence` | The affected segment moved against its control exactly at the onset, and the cause is the only one left standing. | Only if the cause is supported by the stop rule (a) **and** the difference-in-differences effect on its leaf has a 95% interval that excludes 0 in the observed direction. Effect and interval are written to the ledger. |
| `inconclusive` | Several explanations fit and the data cannot separate them. | Stop rule (c) with rivals of similar strength; a DiD step whose interval includes 0. |

What `causal_evidence` does **not** mean: proof. It means the observed data rules out the common confounders we
can test (seasonality, a market-wide move, a different segment) and leaves one change as the explanation. What
would make it stronger, e.g. a rollback that restores the metric, or logs from the changed system, is listed
under "next checks".

## 6. Resolution

An incident resolves after 2 consecutive normal checks of its proximate metric in its segment (a planned
campaign stays open until it ends). Coming back within 12h reopens the same incident. Symptoms of an open
incident (on its chain, or normal once its segment is excluded) are attributed to it silently, not re-reported.
