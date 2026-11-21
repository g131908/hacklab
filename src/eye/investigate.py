"""Autonomous investigation: from one detected change to a supported cause (or an honest hypothesis).

The agent alternates between two questions and lets the graph decide which query comes next:
  WHERE  drill a dimension: which country / platform / category carries the change?
         (sums: segment deltas; ratios: mix effect + rate effect, so a traffic shift is not mistaken for a
         rate problem)
  WHAT   follow a formula edge: which factor of the metric carries the change? (exact log-contributions
         against a pooled baseline, so the contributions add up to the whole change)
At a leaf (no further formula) it tests explanations: change-log events (component, segment, timing,
control groups), possible-cause edges (is the cause metric itself off, and earlier?), planned events.

Grounding: every query and its numbers go into the evidence ledger (q1, q2, ...). Findings cite ledger
ids. The agent re-derives everything from data each run; nothing is carried over from earlier narratives.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import detect
from .detect import Measurement, fmt, seg_text
from .graph import MetricGraph
from .store import NotApplicable, Store

H = pd.Timedelta(hours=1)
LAG = {"wholesale": 30, "marketing": 2, "default": 3}       # hours an event may precede the onset
OUTCOMES = ["orders", "gmv"]


@dataclass
class Step:
    id: str
    kind: str          # check | measure | localize | decompose | explain | control | impact | conclude
    title: str
    query: dict
    result: dict
    decision: str
    options: list = field(default_factory=list)   # candidate next queries considered (for "where next")


@dataclass
class Hypothesis:
    cause: str
    kind: str          # event | metric | planned | data
    status: str        # supported | hypothesis | ruled_out
    score: float
    evidence: list[str]
    against: list[str]
    refs: list[str]


@dataclass
class Investigation:
    trigger: dict
    now: str
    steps: list[Step] = field(default_factory=list)
    segment: dict = field(default_factory=dict)
    chain: list[str] = field(default_factory=list)        # trigger metric -> ... -> proximate cause metric
    symptoms: list[str] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    classification: str = ""                              # fault | innocent | data_incomplete | unexplained
    impact: dict = field(default_factory=dict)
    uncertainty: list[str] = field(default_factory=list)
    explained_share: float | None = None
    onset: str | None = None
    signature: str = ""
    headline: str = ""
    summary: str = ""
    planner: str = "graph policy"

    def add(self, kind, title, query, result, decision, options=None) -> str:
        sid = f"q{len(self.steps) + 1}"
        self.steps.append(Step(sid, kind, title, query, _clean(result), decision, options or []))
        return sid

    def to_dict(self) -> dict:
        d = asdict(self)
        return _clean(d)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o != o or math.isinf(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


class Investigator:
    def __init__(self, store: Store, graph: MetricGraph, now, budget: int = 40, planner=None):
        self.s, self.g, self.now, self.budget = store, graph, pd.Timestamp(now), budget
        self.start, self.end = detect.window(self.now)
        self.nb = max(detect.baseline_days(store, self.start), 1)   # baseline days that exist in the data
        self.planner = planner          # optional callable(inv, options) -> index of chosen option

    # ------------------------------------------------------------------ helpers
    def _scale(self, metric: str) -> float:
        """Pooled baseline is 7 days of sums; counts are rescaled to one day at today's traffic level."""
        if self.s.expr(metric).kind == "base" or detect.is_count(self.s, metric):
            days = [detect.day_factor(self.s, self.start - pd.Timedelta(days=k)) or 1.0 for k in range(1, self.nb + 1)]
            return (detect.day_factor(self.s, self.start) or 1.0) / sum(days)
        return 1.0

    def actual(self, metric, seg, by=None):
        return self.s.window_value(metric, self.start, self.end, seg, by)

    def base(self, metric, seg, by=None):
        return self.s.pooled_value(metric, self.start, self.end, self.nb, seg, by) * self._scale(metric)

    # ------------------------------------------------------------------ WHERE
    def drill(self, metric: str, seg: dict, dim: str) -> dict | None:
        e = self.s.expr(metric)
        try:
            if e.kind == "base" or (e.kind != "ratio"):
                a, b = self.actual(metric, seg, dim), self.base(metric, seg, dim)
                delta = (a - b)
                total = float(delta.sum())
                rows = {v: dict(actual=float(a[v]), expected=float(b[v]), delta=float(delta[v]),
                                rel=float(a[v] / b[v] - 1) if b[v] else None) for v in a.index}
                mix = None
            else:
                num, den = e.tree.left.id, e.tree.right.id
                Na, Da = self.actual(num, seg, dim), self.actual(den, seg, dim)
                Nb, Db = self.base(num, seg, dim), self.base(den, seg, dim)
                ra, rb = Na / Da, Nb / Db
                wa, wb = Da / Da.sum(), Db / Db.sum()
                r_tot_b = Nb.sum() / Db.sum()
                # centred mix effect: gaining weight only matters if the segment's rate differs from the average
                mix_v = (wa - wb) * (rb - r_tot_b)
                rate_v = wa * (ra - rb)
                delta = (mix_v + rate_v)
                total = float(delta.sum())
                rows = {v: dict(actual=float(ra[v]), expected=float(rb[v]), delta=float(delta[v]),
                                rel=float(ra[v] / rb[v] - 1) if rb[v] else None,
                                mix=float(mix_v[v]), rate=float(rate_v[v]),
                                weight_now=float(wa[v]), weight_before=float(wb[v])) for v in ra.index}
                mix = dict(mix=float(mix_v.sum()), rate=float(rate_v.sum()))
        except NotApplicable:
            return None
        if not total:
            return None
        for v in rows:
            rows[v]["share"] = rows[v]["delta"] / total
        top = max(rows, key=lambda v: rows[v]["share"])
        others = [r["rel"] for k, r in rows.items() if k != top and r["rel"] is not None]
        rest = float(np.median(others)) if others else 0.0
        return dict(metric=metric, dim=dim, total_delta=total, rows=rows, top=top, top_share=rows[top]["share"],
                    top_rel=rows[top]["rel"], rest_rel=rest, mix=mix)

    def localize(self, inv: Investigation, metric: str, seg: dict) -> dict:
        while len(seg) < 3 and len(inv.steps) < self.budget:
            dims = [d for d in self.s.dims(metric) if d not in seg]
            options = []
            for d in dims:
                r = self.drill(metric, seg, d)
                if r:
                    options.append(r)
            if not options:
                break
            opts_txt = [dict(query=f"drill {o['dim']}", score=round(o["top_share"], 3),
                             why=f"{o['top']} carries {o['top_share']:.0%} of the change "
                                 f"({o['top_rel']:+.1%} vs {o['rest_rel']:+.1%} elsewhere)") for o in options]
            best = max(options, key=lambda o: o["top_share"])
            concentrated = (best["top_share"] >= 0.6 and best["top_rel"] is not None
                            and abs(best["top_rel"]) >= 2 * abs(best["rest_rel"]) + 0.03)
            res = dict(dim=best["dim"], top=best["top"], top_share=best["top_share"], top_rel=best["top_rel"],
                       rest_rel=best["rest_rel"], segments={k: dict(rel=v["rel"], share=v["share"],
                                                                        **({"mix": v["mix"], "rate": v["rate"]} if "mix" in v else {}))
                                                            for k, v in best["rows"].items()},
                       mix=best["mix"])
            if concentrated:
                seg = {**seg, best["dim"]: best["top"]}
                inv.add("localize", f"WHERE: {metric} by {best['dim']}", dict(metric=metric, segment=seg_text(seg)),
                        res, f"localized to {best['dim']}={best['top']}: {best['top_share']:.0%} of the change, "
                             f"{best['top_rel']:+.1%} there vs {best['rest_rel']:+.1%} elsewhere -> narrow down", opts_txt)
                if best["mix"] and abs(best["mix"]["mix"]) > abs(best["mix"]["rate"]):
                    inv.add("localize", f"WHERE: mix check for {metric}", dict(dim=best["dim"]), best["mix"],
                            "most of the change is a MIX effect (traffic shifted between segments), not a rate change")
            else:
                inv.add("localize", f"WHERE: {metric} by {', '.join(o['dim'] for o in options)}",
                        dict(metric=metric, segment=seg_text(seg)), res,
                        f"not concentrated (best: {best['dim']}={best['top']} with {best['top_share']:.0%}): "
                        f"the change is spread across {', '.join(o['dim'] for o in options)} -> stop narrowing", opts_txt)
                break
        return seg

    # ------------------------------------------------------------------ WHAT
    def decompose(self, inv: Investigation, metric: str, seg: dict) -> str | None:
        d = self.g.decomposition(metric)
        if not d or d.op != "product":
            return None
        try:
            ma, mb = float(self.actual(metric, seg)), float(self.base(metric, seg))
            facs = {f: (float(self.actual(f, seg)), float(self.base(f, seg))) for f in d.factors}
        except NotApplicable:
            return None
        if not (ma > 0 and mb > 0):
            return None
        total = math.log(ma / mb)
        contrib = {f: math.log(a / b) if a > 0 and b > 0 else float("nan") for f, (a, b) in facs.items()}
        shares = {f: (c / total if total else 0.0) for f, c in contrib.items()}
        rows = {f: dict(actual=facs[f][0], expected=facs[f][1], change=math.exp(contrib[f]) - 1,
                        log_contribution=contrib[f], share=shares[f]) for f in d.factors}
        lead = max(d.factors, key=lambda f: shares[f])
        stable = [f for f in d.factors if abs(rows[f]["change"]) < 0.03]
        opts = [dict(query=f"follow {f}", score=round(shares[f], 3), why=f"{rows[f]['change']:+.1%}, "
                     f"{shares[f]:.0%} of the log-change") for f in d.factors]
        decision = (f"{lead} explains {shares[lead]:.0%} of the {math.exp(total) - 1:+.1%} change in {metric} "
                    f"({d.text}, {d.kind}) -> follow {lead}")
        if stable:
            decision += f"; stable: {', '.join(stable)}"
        inv.add("decompose", f"WHAT: {metric}[{seg_text(seg)}] = {' x '.join(d.factors)}",
                dict(metric=metric, segment=seg_text(seg), formula=d.text), dict(total_change=math.exp(total) - 1,
                                                                                factors=rows), decision, opts)
        return lead if shares[lead] >= 0.5 else (lead if shares[lead] >= 0.35 else None)

    # ------------------------------------------------------------------ explanations
    def _event_scope_matches(self, ev, seg) -> tuple[bool, list[str]]:
        notes = []
        for dim in ("country", "platform", "category"):
            v = ev.get(dim, "*")
            if v in ("*", None) or (isinstance(v, float) and math.isnan(v)):
                if dim in seg:
                    notes.append(f"{dim}: event covers all, change only in {seg[dim]}")
                continue
            if dim in seg and seg[dim] != v:
                return False, [f"{dim}: event is for {v}, change is in {seg[dim]}"]
            if dim not in seg:
                notes.append(f"{dim}: event only for {v}, change seen across {dim}")
        return True, notes

    def explain(self, inv: Investigation, leaf: str, seg: dict, direction: str, onset: pd.Timestamp | None):
        comps_for = set()
        for m in inv.chain:
            comps_for |= set(self.g.components_for(m))
        look_from = (onset or self.start) - pd.Timedelta(hours=max(LAG.values()))
        evs = self.s.events_between(look_from, self.end)
        ref_t = onset or self.start
        for _, ev in evs.iterrows():
            ev = ev.to_dict()
            comp = ev.get("component")
            lag = LAG.get(comp, LAG["default"])
            ok_scope, scope_notes = self._event_scope_matches(ev, seg)
            gap_h = (ref_t - ev["ts"]) / H
            ev_end = ev.get("end")
            active = isinstance(ev_end, pd.Timestamp) and not pd.isna(ev_end) and ev["ts"] <= ref_t <= ev_end
            timing_ok = (-1.0 <= gap_h <= lag) or active
            comp_ok = comp in comps_for
            evidence, against = [], []
            if comp_ok:
                evidence.append(f"component '{comp}' can move {', '.join(m for m in inv.chain if comp in self.g.components_for(m))}")
            else:
                against.append(f"component '{comp}' is not connected to {leaf} or its chain in the graph")
            if ok_scope:
                evidence.append("scope matches the affected segment" + (f" ({'; '.join(scope_notes)})" if scope_notes else ""))
            else:
                against += scope_notes
            if timing_ok:
                evidence.append(f"{'active since' if active else 'at'} {ev['ts']:%d.%m %H:%M}, "
                                f"{gap_h:+.1f}h before the onset {ref_t:%d.%m %H:%M}" if gap_h >= 0 else
                                f"at {ev['ts']:%H:%M}, within the onset hour")
            else:
                against.append(f"timing: {ev['ts']:%d.%m %H:%M} is {gap_h:+.1f}h from the onset "
                               f"(allowed 0..{lag}h for {comp})")
            if ev["type"] in ("rollback",) and ev["ts"] > ref_t:
                against.append("this is a later rollback (a fix, not a cause)")
            score = 2 * comp_ok + 2 * ok_scope + 2 * timing_ok + (0.5 if not ev.get("planned", True) else 0)
            status = "hypothesis" if (comp_ok and ok_scope and timing_ok) else "ruled_out"
            ref = inv.add("explain", f"EVENT: {ev['name']}", dict(type=ev["type"], component=comp,
                          scope={k: ev.get(k) for k in ("country", "platform", "category")}, ts=str(ev["ts"])),
                          dict(component_match=comp_ok, scope_match=ok_scope, timing_ok=timing_ok, gap_h=gap_h),
                          f"{status}: " + "; ".join(evidence if status != "ruled_out" else against))
            kind = "planned" if ev["type"] == "campaign" else "event"
            h = Hypothesis(f"{ev['type']}: {ev['name']}", kind, status, score, evidence, against, [ref])
            if status != "ruled_out" and not ok_scope is False:
                self.control(inv, h, ev, leaf, seg, direction)
            inv.hypotheses.append(h)

        # possible-cause edges (metric -> metric), incl. ambiguous ones from the extension agent
        for c in [c for c in self.s.catalog.get("causes") or [] if c["to"] in inv.chain]:
            cause = c["from"]
            cseg = {k: v for k, v in seg.items() if k in self.s.dims(cause)}
            m = detect.measure(self.s, cause, cseg, self.now)
            ref = inv.add("measure", f"CAUSE?: {cause}[{seg_text(cseg)}] -> {c['to']}",
                          dict(edge=c.get("reason", ""), status=c.get("status", "hypothesis")), m.to_dict(),
                          f"{cause} is {m.status} ({m.rel:+.1%})" if m.rel == m.rel else f"{cause}: {m.status}")
            if m.status == "anomaly":
                o = detect.onset(self.s, cause, cseg, self.now, direction=m.direction)
                ok_time = o is None or onset is None or o <= onset + H
                ev_ = [f"{cause} itself moved {m.rel:+.1%} in {seg_text(cseg)} (z={m.z:+.1f})",
                       f"graph edge: {c.get('reason', '')}"]
                if o is not None:
                    ev_.append(f"its onset {o:%d.%m %H:%M} {'precedes or matches' if ok_time else 'is AFTER'} "
                               f"the {c['to']} onset {onset:%d.%m %H:%M}" if onset is not None else f"onset {o:%d.%m %H:%M}")
                ambiguous = c.get("status") == "ambiguous"
                inv.hypotheses.append(Hypothesis(f"metric: {cause} ({'co-moving, direction unknown' if ambiguous else 'upstream cause'})",
                                                 "metric", "hypothesis" if (ok_time and not ambiguous) else "hypothesis",
                                                 5 + (0 if ambiguous else 2) + (1 if ok_time else -2), ev_,
                                                 [] if ok_time else ["cause moved after the effect"], [ref]))
                if not ambiguous and ok_time:
                    # the cause metric is itself a node: look for what moved it (one level, depth-limited)
                    inv.chain.append(cause)
                    co = o or onset
                    sub = [h for h in inv.hypotheses]
                    self.explain_events_only(inv, cause, cseg, co)

    def explain_events_only(self, inv: Investigation, metric: str, seg: dict, onset):
        comps = set(self.g.components_for(metric))
        look_from = (onset or self.start) - pd.Timedelta(hours=max(LAG.values()))
        for _, ev in self.s.events_between(look_from, self.end).iterrows():
            ev = ev.to_dict()
            if ev["component"] not in comps:
                continue
            ok_scope, notes = self._event_scope_matches(ev, seg)
            gap_h = ((onset or self.start) - ev["ts"]) / H
            lag = LAG.get(ev["component"], LAG["default"])
            timing_ok = -1.0 <= gap_h <= lag
            routine = bool(ev.get("planned", True)) and ev["type"] in ("purchasing",)
            ev_txt = [f"component '{ev['component']}' moves {metric}", f"{gap_h:+.1f}h before the {metric} onset"]
            against = [] if timing_ok and ok_scope else notes + ([] if timing_ok else [f"timing {gap_h:+.1f}h"])
            if routine:
                against.append("routine planned run reported at normal volume; order quantities are not in the data")
            status = "ruled_out" if not (timing_ok and ok_scope) else "hypothesis"
            ref = inv.add("explain", f"EVENT (upstream of {metric}): {ev['name']}",
                          dict(type=ev["type"], component=ev["component"], ts=str(ev["ts"])),
                          dict(timing_ok=timing_ok, scope_match=ok_scope, gap_h=gap_h, planned=bool(ev.get("planned"))),
                          f"{status}: " + "; ".join(ev_txt if status != "ruled_out" else against)
                          + (" (competing explanation, weaker: routine)" if routine and status != "ruled_out" else ""))
            score = 6 + (1 if not ev.get("planned", True) else 0) - (2 if routine else 0)
            inv.hypotheses.append(Hypothesis(f"{ev['type']}: {ev['name']}", "event", status, score, ev_txt,
                                             against, [ref]))

    def control(self, inv: Investigation, h: Hypothesis, ev: dict, leaf: str, seg: dict, direction: str):
        """Event scope wider than the affected segment: is the leaf metric normal in the rest of the scope?"""
        for dim in ("country", "platform", "category"):
            if dim in seg and ev.get(dim, "*") in ("*", None):
                others = [v for v in self.s.values(dim) if v != seg[dim]]
                cs = {k: v for k, v in seg.items() if k != dim}
                rows = {}
                for v in others:
                    try:
                        m = detect.measure(self.s, leaf, {**cs, dim: v}, self.now)
                        rows[v] = dict(rel=m.rel, status=m.status)
                    except NotApplicable:
                        pass
                normal = [v for v, r in rows.items() if r["status"] != "anomaly"]
                ref = inv.add("control", f"CONTROL: {leaf} in other {dim} values within the event scope",
                              dict(metric=leaf, base_segment=seg_text(cs), dim=dim), rows,
                              f"{len(normal)}/{len(rows)} unaffected -> the change is specific to {dim}={seg[dim]}"
                              if len(normal) == len(rows) else
                              f"also affected: {', '.join(v for v in rows if v not in normal)}")
                h.refs.append(ref)
                if len(normal) == len(rows) and rows:
                    h.evidence.append(f"control: {leaf} normal in {', '.join(normal)} -> only {dim}={seg[dim]} is hit")
                    inv.uncertainty.append(f"'{ev['name']}' covers all {dim} values but only {seg[dim]} is affected: "
                                           f"mechanism specific to {seg[dim]} is inferred, not verified")

    # ------------------------------------------------------------------ main
    def run(self, trigger: Measurement) -> Investigation:
        inv = Investigation(trigger=trigger.to_dict(), now=str(self.now))
        if self.planner is not None:
            inv.planner = getattr(self.planner, "name", "planner")
        inv.add("measure", f"TRIGGER: {trigger.metric}[{trigger.seg_text()}]", dict(window=f"{self.start} - {self.end}"),
                trigger.to_dict(), f"{trigger.short()} -> investigate")

        # data completeness first: an incomplete source looks like a drop
        comp = {src: min((self.s.completeness_at(src, t) for t in pd.date_range(self.start, self.end, freq="h")),
                         default=1.0) for src in self.s.sources(trigger.metric)}
        ref = inv.add("check", "DATA: completeness of sources feeding the trigger", dict(sources=list(comp)), comp,
                      "complete" if min(comp.values(), default=1) >= 0.95 else
                      f"INCOMPLETE: {', '.join(f'{k} {v:.0%}' for k, v in comp.items() if v < 0.95)}")
        if min(comp.values(), default=1) < 0.95:
            ctrl = detect.measure(self.s, "sessions", trigger.segment, self.now)
            r2 = inv.add("check", "DATA: same window on a complete source (sessions)", {}, ctrl.to_dict(),
                         f"sessions {ctrl.rel:+.1%}: traffic is normal -> the drop is a data artefact")
            inv.classification = "data_incomplete"
            inv.hypotheses.append(Hypothesis("data: late / missing rows in " +
                                             ", ".join(k for k, v in comp.items() if v < 0.95), "data", "supported", 10,
                                             [f"completeness {', '.join(f'{k} {v:.0%}' for k, v in comp.items())}",
                                              f"sessions (complete source) {ctrl.rel:+.1%}"], [], [ref, r2]))
            inv.uncertainty.append("real orders for these hours are unknown until the data is complete; re-check next tick")
            inv.signature = f"data|{','.join(k for k, v in comp.items() if v < 0.95)}"
            return self.finish(inv)

        metric = trigger.metric
        direction = trigger.direction
        inv.chain = [metric]
        if trigger.segment:
            # zoom out first: the same change may sit in a bigger (or different) segment than the one that tripped
            inv.add("localize", "ZOOM OUT: start from the whole marketplace", dict(trigger_segment=trigger.seg_text()), {},
                    "a segment tripped the alarm; check from total where the change really concentrates")
            seg = self.localize(inv, metric, {})
            if not seg:
                seg = self.localize(inv, metric, dict(trigger.segment))
        else:
            seg = self.localize(inv, metric, {})
        narrowed = len(seg) > 0
        for _ in range(8):
            if len(inv.steps) >= self.budget:
                inv.uncertainty.append("query budget exhausted before a leaf metric was reached")
                break
            nxt = self.decompose(inv, metric, seg)
            if not nxt:
                break
            share = inv.steps[-1].result["factors"][nxt]["share"]
            metric = nxt
            inv.chain.append(metric)
            if not narrowed and share >= 0.85:
                # the factor carries (almost) the whole change and the parent did not concentrate anywhere:
                # the same drill-down would only repeat itself, so spend the query budget further down instead
                continue
            before = len(seg)
            seg = self.localize(inv, metric, seg)
            narrowed = len(seg) > before
        inv.segment = seg
        leaf = inv.chain[-1]
        leaf_m = detect.measure(self.s, leaf, seg, self.now)
        o = detect.onset(self.s, leaf, seg, self.now, direction=leaf_m.direction or direction)
        inv.onset = str(o) if o is not None else None
        inv.add("measure", f"LEAF: {leaf}[{seg_text(seg)}]", {}, dict(**leaf_m.to_dict(), onset=inv.onset),
                f"proximate metric: {leaf_m.short()}, onset {o:%d.%m %H:%M}" if o is not None else
                f"proximate metric: {leaf_m.short()} (onset not clear)")
        self.explain(inv, leaf, seg, leaf_m.direction or direction, o)
        return self.finish(inv)

    # ------------------------------------------------------------------ conclusion
    def finish(self, inv: Investigation) -> Investigation:
        seg = inv.segment or inv.trigger["segment"]
        # one entry per explanation: keep the strongest verdict (an event can be ruled out at the leaf but
        # live as the upstream cause of a cause metric)
        rank = {"supported": 2, "hypothesis": 1, "ruled_out": 0}
        best = {}
        for h in inv.hypotheses:
            k = h.cause
            if k not in best or (rank[h.status], h.score) > (rank[best[k].status], best[k].score):
                if k in best:
                    h.refs = best[k].refs + h.refs
                best[k] = h
            else:
                best[k].refs += h.refs
        inv.hypotheses = list(best.values())
        # business impact on outcomes in the affected segment (not computed on incomplete data)
        imp = {}
        for m in (OUTCOMES if inv.classification != "data_incomplete" else []):
            try:
                a, b = float(self.actual(m, seg)), float(self.base(m, seg))
            except NotApplicable:
                continue
            hours = int((self.end - self.start) / H) + 1
            imp[m] = dict(actual=a, expected=b, delta=a - b, rel=a / b - 1 if b else None, per_hour=(a - b) / hours)
        inv.impact = imp
        if imp:
            inv.add("impact", f"IMPACT on outcomes in {seg_text(seg)}", dict(window=f"{self.start} - {self.end}"), imp,
                    "; ".join(f"{m} {v['delta']:+,.0f} ({v['rel']:+.1%}) over the window" for m, v in imp.items()
                              if v["rel"] is not None))
        # share of the trigger's change that sits in the localized segment
        t = inv.trigger
        if seg != t["segment"] and t["metric"] in OUTCOMES + ["conversion_rate"] and inv.classification != "data_incomplete":
            try:
                tot_a, tot_b = float(self.actual(t["metric"], t["segment"])), float(self.base(t["metric"], t["segment"]))
                sa, sb = float(self.actual(t["metric"], seg)), float(self.base(t["metric"], seg))
                if self.s.expr(t["metric"]).kind != "ratio" and tot_a != tot_b:
                    inv.explained_share = (sa - sb) / (tot_a - tot_b)
            except NotApplicable:
                pass

        live = [h for h in inv.hypotheses if h.status != "ruled_out"]
        live.sort(key=lambda h: -h.score)
        if inv.classification != "data_incomplete":
            harmful = any(v["rel"] is not None and v["rel"] <= -0.03 for v in imp.values())
            planned = [h for h in live if h.kind == "planned"]
            if planned and not harmful:
                inv.classification = "innocent"
                planned[0].status = "supported"
                planned[0].evidence.append("outcomes are not down: " + ", ".join(
                    f"{m} {v['rel']:+.1%}" for m, v in imp.items() if v["rel"] is not None))
            elif live:
                inv.classification = "fault" if harmful or inv.trigger["direction"] == "down" else "change"
                best = live[0]
                strong = best.score >= 6       # component, scope and timing all match (or a cause metric moved first)
                unique = len(live) == 1 or live[1].score < best.score
                if strong and unique:
                    best.status = "supported"
                    for h in live[1:]:
                        h.against.append(f"weaker than '{best.cause}' ({h.score:g} vs {best.score:g})")
                if len(live) > 1:
                    inv.uncertainty.append("competing explanations still open: " +
                                           ", ".join(f"{h.cause} ({h.score:g})" for h in live[1:3]))
                if not strong:
                    inv.uncertainty.append(f"best explanation '{best.cause}' does not match on component, scope and "
                                           "timing together: reported as a hypothesis")
            else:
                inv.classification = "unexplained" if harmful or inv.trigger["direction"] == "down" else "change"
                inv.uncertainty.append("no event or upstream metric explains the change; data shows WHERE and WHAT, "
                                       "not yet WHY")
        inv.symptoms = [m for m in inv.chain[:-1]] + [m for m in OUTCOMES if m not in inv.chain and m in imp
                                                       and imp[m]["rel"] is not None and abs(imp[m]["rel"]) > 0.03]
        sup = next((h for h in live if h.status == "supported"), None)
        leaf = inv.chain[-1] if inv.chain else inv.trigger["metric"]
        # one incident per root cause: a supported event names the incident; otherwise proximate metric + segment
        first_event = sup if sup and sup.kind in ("event", "planned") else None
        if sup and sup.kind == "metric":
            first_event = next((h for h in live if h.kind == "event"), None)
        inv.signature = inv.signature or (f"cause|{first_event.cause}" if first_event else f"{leaf}|{seg_text(seg)}")
        if inv.classification == "data_incomplete":
            inv.headline = f"Data incomplete: {inv.trigger['metric']} looks {inv.trigger['rel']:+.0%} but the source is still loading"
        elif inv.classification == "innocent":
            inv.headline = (f"Innocent change: {inv.trigger['metric']} {inv.trigger['rel']:+.0%} in {seg_text(seg)} "
                            f"explained by {sup.cause if sup else 'a planned event'}; outcomes not hurt")
        else:
            lost = imp.get("gmv", {}).get("per_hour")
            first = None
            if sup and sup.kind == "metric":    # cause metric supported: name the best upstream event, if any
                first = next((h for h in live if h.kind == "event" and h is not sup), None)
            inv.headline = (f"{inv.trigger['metric']} {inv.trigger['rel']:+.0%}: {leaf} in {seg_text(seg)} "
                            + (f"-> {sup.cause.split(' (')[0]}" if sup else "(cause not confirmed)")
                            + (f" -> likely {first.cause} (hypothesis)" if first else "")
                            + (f"; GMV {lost:+,.0f} EUR/h" if lost is not None else ""))
        inv.add("conclude", "CONCLUSION", dict(classification=inv.classification),
                dict(chain=inv.chain, segment=seg_text(seg), supported=sup.cause if sup else None,
                     hypotheses=[(h.cause, h.status, h.score) for h in inv.hypotheses]),
                inv.headline)
        return inv
