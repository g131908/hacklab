"""Continuous monitoring: decide what to watch, detect, investigate, track incidents.

Attention policy per tick
  - every metric at total level (cheap, broad)
  - plus a budget of segment checks, ranked by business weight (GMV share) x recency of trouble x staleness,
    with a boost for metrics added in the last 24h (their baseline is still being established)
Triage
  - anomalies already explained by an open incident (metric on its chain or an ancestor of it, segment
    overlapping) are "known, persisting": no new investigation, no new report
  - new anomalies are investigated most important first (outcomes > KPIs > others, total before segments)
Incidents
  - keyed by signature = proximate metric | segment; open -> persisting -> resolved after 2 normal checks
  - innocent changes become INFO notes, incomplete data becomes a DATA notice; neither pages anyone
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import detect
from .detect import seg_text
from .graph import MetricGraph
from .investigate import LABEL_RANK, Investigator
from .store import NotApplicable, Store

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".agents" / "eye"
SEGMENT_BUDGET = 70
MAX_INVESTIGATIONS = 2
REOPEN_H = 12
TIER = {"outcome": 0, "kpi": 1}


@dataclass
class Incident:
    id: str
    signature: str
    classification: str
    status: str                 # open | resolved
    opened: str
    last_seen: str
    severity: int
    headline: str
    segment: dict
    chain: list
    investigation: dict
    updates: list = field(default_factory=list)
    normal_checks: int = 0
    reports: int = 1
    review: dict | None = None
    resolved_at: str | None = None
    explains: list = field(default_factory=list)   # symptoms attributed to this incident without a new report
    confidence_label: str = ""                     # descriptive | associational | causal_evidence | inconclusive
    changes: list = field(default_factory=list)    # change-log entries around the onset, all tested (with owner)


def severity(inv) -> int:
    if inv.classification == "innocent":
        return 1
    if inv.classification == "data_incomplete":
        return 2
    lost = -(inv.impact.get("gmv", {}).get("per_hour") or 0)
    return 5 if lost >= 3000 else 4 if lost >= 1000 else 3


class Monitor:
    def __init__(self, full: Store, state_dir: Path = STATE, reviewer=None):
        self.full = full
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.incidents: dict[str, Incident] = {}
        self.last_checked: dict[str, str] = {}
        self.trouble: dict[str, str] = {}          # segment key -> last anomaly time
        self.reviewer = reviewer            # optional: Claude review of the ledger, once per incident
        self.graph_full = MetricGraph.build(full)   # validated once on all data available
        self.added_at: dict[str, str] = {}
        self.closed: list[Incident] = []     # older incidents whose signature was opened again later
        self.seq = 0
        self._load()

    # ------------------------------------------------------------------ state
    def _load(self):
        p = self.dir / "incidents.json"
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            self.incidents = {k: Incident(**v) for k, v in raw.get("incidents", {}).items()}
            self.closed = [Incident(**v) for v in raw.get("closed", [])]
            self.seq = raw.get("seq", len(self.incidents) + len(self.closed))
            self.last_checked = raw.get("last_checked", {})
            self.trouble = raw.get("trouble", {})
            self.added_at = raw.get("added_at", {})

    def save(self):
        (self.dir / "incidents.json").write_text(json.dumps(dict(
            incidents={k: asdict(v) for k, v in self.incidents.items()}, closed=[asdict(v) for v in self.closed],
            seq=self.seq, last_checked=self.last_checked,
            trouble=self.trouble, added_at=self.added_at), default=str, indent=1), encoding="utf-8")

    def reset(self):
        for f in ("incidents.json", "ticks.jsonl"):
            p = self.dir / f
            if p.exists():
                p.unlink()
        self.incidents, self.last_checked, self.trouble = {}, {}, {}
        self.closed, self.seq = [], 0

    # ------------------------------------------------------------------ attention
    def attention(self, st: Store, now: pd.Timestamp) -> list[tuple[str, dict, float, str]]:
        plan = [(m, {}, 10.0, "total level: always watched") for m in st.metrics]
        gmv_share = {}
        tot = float(st.window_value("gmv", now - pd.Timedelta(days=1), now)) or 1.0
        for dim in ("country", "platform", "category"):
            vals = st.window_value("gmv", now - pd.Timedelta(days=1), now, None, dim)
            for v, x in vals.items():
                gmv_share[(dim, v)] = float(x) / tot
        cands = []
        for m in st.metrics:
            for dim in st.dims(m):
                for v in st.values(dim):
                    key = f"{m}|{dim}={v}"
                    w = gmv_share.get((dim, v), 0.1)
                    last = self.last_checked.get(key)
                    stale = 3.0 if last is None else min((now - pd.Timestamp(last)) / pd.Timedelta(hours=1), 6) / 2
                    hot = 3.0 if (t := self.trouble.get(f"{dim}={v}")) and now - pd.Timestamp(t) <= pd.Timedelta(hours=6) else 1.0
                    new = 2.0 if m in self.added_at and now - pd.Timestamp(self.added_at[m]) <= pd.Timedelta(hours=24) else 1.0
                    rate_bonus = 1.3 if st.expr(m).kind == "ratio" else 1.0
                    score = (0.5 + 3 * w) * hot * new * rate_bonus * (0.5 + stale)
                    why = []
                    if hot > 1:
                        why.append("recent trouble in this segment")
                    if new > 1:
                        why.append("new metric")
                    if last is None or stale >= 3:
                        why.append("not checked recently")
                    why.append(f"{w:.0%} of GMV")
                    cands.append((m, {dim: v}, score, ", ".join(why)))
        cands.sort(key=lambda c: -c[2])
        return plan + cands[:SEGMENT_BUDGET]

    # ------------------------------------------------------------------ coverage
    def _ancestors(self, g: MetricGraph, metric: str) -> set:
        out, frontier = {metric}, [metric]
        while frontier:
            m = frontier.pop()
            for p in g.parents(m, ratios=True):
                if p not in out:
                    out.add(p)
                    frontier.append(p)
        return out

    def _related(self, g: MetricGraph, chain: list) -> set:
        """Metrics an incident can move: its chain, everything computed from it (formula ancestors), and the
        effects of chain metrics along possible-cause edges (with their ancestors)."""
        rel = set()
        frontier = list(chain)
        while frontier:
            c = frontier.pop()
            if c in rel:
                continue
            rel |= self._ancestors(g, c)
            frontier += [x["to"] for x in g.causes if x["from"] == c and x["to"] not in rel]
        return rel

    def covered_by(self, st: Store, g: MetricGraph, m: detect.Measurement, now) -> tuple[Incident | None, str]:
        for inc in self.incidents.values():
            if inc.status != "open":
                continue
            seg, iseg = m.segment, inc.segment
            compatible = all(iseg.get(k) == v for k, v in seg.items() if k in iseg)
            if not compatible:
                continue
            if m.metric in self._related(g, inc.chain):
                return inc, "on the incident's chain"
            # explained-away test: is the metric normal once the incident's segment is taken out?
            if iseg and not set(iseg) & set(seg):
                try:
                    rest = detect.measure(st, m.metric, {**seg, "__not__": iseg}, now, persistence=False)
                except NotApplicable:
                    continue
                if rest.status not in ("anomaly", "candidate"):
                    return inc, f"normal without {seg_text(iseg)} ({rest.rel:+.1%})"
        return None, ""

    # ------------------------------------------------------------------ tick
    def tick(self, now, extra_store: Store | None = None) -> dict:
        t0 = time.perf_counter()
        now = pd.Timestamp(now)
        st = (extra_store or self.full).asof(now)
        g = self.graph_full          # built and validated once; rebuilt when the catalog changes
        report = dict(now=str(now), label=st.label, watched=0, anomalies=[], known=[], investigations=[],
                      resolved=[], notices=[], attention=[], candidates=0)

        # data freshness first
        for src in st.catalog["sources"]:
            c = st.completeness_at(src, now)
            if c == c and c < 0.95:
                report["notices"].append(dict(kind="data", text=f"{src}: only {c:.0%} of rows for {now:%d.%m %H:%M} "
                                                                 "have arrived; metrics from it are held back"))

        plan = self.attention(st, now)
        report["watched"] = len(plan)
        report["attention"] = [dict(metric=m, segment=seg_text(s), score=round(sc, 2), why=w) for m, s, sc, w in plan[-SEGMENT_BUDGET:][:12]]
        found = []
        for m, seg, _, _ in plan:
            try:
                r = detect.measure(st, m, seg, now)
            except NotApplicable:
                continue
            if seg:
                self.last_checked[f"{m}|{seg_text(seg)}"] = str(now)
            if r.status in ("anomaly", "incomplete", "candidate"):
                for k, v in seg.items():
                    self.trouble[f"{k}={v}"] = str(now)      # look closer at this segment next tick
            if r.status == "candidate":
                report["candidates"] += 1
            if r.status in ("anomaly", "incomplete"):
                found.append(r)
        found.sort(key=lambda r: (TIER.get(st.metrics[r.metric].get("tier"), 2), len(r.segment), -abs(r.z)))

        investigated = 0
        for r in found:
            inc, why = self.covered_by(st, g, r, now)
            entry = dict(metric=r.metric, segment=r.seg_text(), rel=r.rel, z=r.z, status=r.status)
            if inc:
                entry["incident"], entry["why"] = inc.id, why
                if not any(x["metric"] == r.metric and x["segment"] == r.seg_text() for x in inc.explains):
                    inc.explains.append(dict(metric=r.metric, segment=r.seg_text(), first=str(now), rel=r.rel, why=why))
                report["known"].append(entry)
                continue
            if investigated >= MAX_INVESTIGATIONS:
                entry["deferred"] = True
                report["anomalies"].append(entry)
                continue
            investigated += 1
            inv = Investigator(st, g, now).run(r)
            entry["investigation"] = inv.signature
            report["anomalies"].append(entry)
            inc = self.incidents.get(inv.signature)
            sev = severity(inv)
            if inc and inc.status == "open":
                inc.last_seen = str(now)
                inc.normal_checks = 0
                inc.chain = inc.chain + [c for c in inv.chain if c not in inc.chain]
                escalated = sev > inc.severity
                inc.updates.append(dict(at=str(now), text=f"{r.metric}[{r.seg_text()}] also traced to this incident"
                                        + (f"; severity raised to {sev}" if escalated else "")))
                stronger = LABEL_RANK.get(inv.confidence_label, 0) > LABEL_RANK.get(inc.confidence_label, 0)
                if escalated or stronger:   # re-report only when it got worse or the evidence got stronger
                    inc.severity, inc.reports, inc.investigation = max(sev, inc.severity), inc.reports + 1, inv.to_dict()
                    inc.headline, inc.confidence_label = inv.headline, inv.confidence_label
                    inc.changes = inv.changes or inc.changes
                    if stronger and not escalated:
                        inc.updates.append(dict(at=str(now), text=f"evidence now {inv.confidence_label}"))
                        escalated = True
                report["investigations"].append(dict(incident=inc.id, new=False, escalated=escalated, **self._brief(inv)))
            elif inc and inc.resolved_at and now - pd.Timestamp(inc.resolved_at) <= pd.Timedelta(hours=REOPEN_H):
                # same root cause back within hours of resolving (e.g. a quiet night in between): same incident
                inc.status, inc.normal_checks, inc.last_seen, inc.resolved_at = "open", 0, str(now), None
                inc.updates.append(dict(at=str(now), text="reopened: same cause seen again shortly after resolving"))
                report["investigations"].append(dict(incident=inc.id, new=False, escalated=False, **self._brief(inv)))
            else:
                if inc:
                    self.closed.append(inc)     # an older, resolved incident with the same signature
                self.seq += 1
                iid = f"I{self.seq:03d}"
                inc = Incident(iid, inv.signature, inv.classification, "open", str(now), str(now), sev, inv.headline,
                               inv.segment or r.segment, inv.chain or [r.metric], inv.to_dict(),
                               confidence_label=inv.confidence_label, changes=inv.changes)
                # Claude reviews the ledger once, when the incident opens (never on updates): 1 call per incident
                if self.reviewer and inv.classification != "data_incomplete":
                    try:
                        inc.review = self.reviewer(inv)
                    except Exception as exc:
                        inc.review = dict(error=f"{type(exc).__name__}: {str(exc)[:200]}")
                self.incidents[inv.signature] = inc
                report["investigations"].append(dict(incident=iid, new=True, escalated=False, review=inc.review,
                                                     **self._brief(inv)))

        # lifecycle of open incidents: persist or resolve
        for inc in self.incidents.values():
            if inc.status != "open" or inc.last_seen == str(now):
                continue
            leaf = inc.chain[-1] if inc.chain else inc.investigation["trigger"]["metric"]
            if inc.classification == "data_incomplete":
                srcs = inc.signature.split("|", 1)[1].split(",")
                ok = all(st.completeness_at(s, now) >= 0.95 for s in srcs)
                state = "normal" if ok else "anomaly"
            else:
                try:
                    state = detect.measure(st, leaf, inc.segment, now).status
                except NotApplicable:
                    state = "normal"
            if state in ("anomaly", "incomplete", "candidate") or self._event_running(st, inc, now):
                inc.normal_checks, inc.last_seen = 0, str(now)
            else:
                inc.normal_checks += 1
                if inc.normal_checks >= 2:
                    inc.status, inc.resolved_at = "resolved", str(now)
                    inc.updates.append(dict(at=str(now), text=f"{leaf} back to normal for 2 checks"))
                    report["resolved"].append(dict(incident=inc.id, headline=inc.headline))
        report["open"] = [dict(id=i.id, severity=i.severity, classification=i.classification, headline=i.headline,
                               since=i.opened) for i in self.incidents.values() if i.status == "open"]
        report["ms"] = round((time.perf_counter() - t0) * 1000)
        with (self.dir / "ticks.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, default=str) + "\n")
        self.save()
        return report

    @staticmethod
    def _event_running(st: Store, inc: Incident, now) -> bool:
        """A planned event with an end time (a campaign) keeps its note open until it ends."""
        if not inc.signature.startswith("cause|") or "end" not in st.events:
            return False
        name = inc.signature.split(": ", 1)[-1]
        ev = st.events[st.events.name == name]
        return bool(len(ev)) and pd.notna(ev.iloc[0]["end"]) and now < ev.iloc[0]["end"]

    def all_incidents(self) -> list[Incident]:
        return sorted(self.closed + list(self.incidents.values()), key=lambda i: i.id)

    @staticmethod
    def _brief(inv) -> dict:
        sup = next((h.cause for h in inv.hypotheses if h.status == "supported"), None)
        return dict(classification=inv.classification, headline=inv.headline, segment=seg_text(inv.segment),
                    chain=inv.chain, supported=sup, steps=len(inv.steps), uncertainty=inv.uncertainty[:3],
                    label=inv.confidence_label, stop=inv.stop_reason, changes=inv.changes, branches=inv.branches)
