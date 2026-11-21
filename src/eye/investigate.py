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
OUTCOMES = ["orders", "gmv", "contribution_margin"]
BUSINESS = "contribution_margin"     # the outcome that decides whether a change is good news
CONCURRENT_H = 6            # changes first exposed within this many hours before the onset count as concurrent

# Evidence levels (see docs/METHOD.md). Every ledger step, every hypothesis and every incident carries one.
DESCRIPTIVE, ASSOCIATIONAL, CAUSAL, INCONCLUSIVE = "descriptive", "associational", "causal_evidence", "inconclusive"
LABEL_RANK = {CAUSAL: 3, ASSOCIATIONAL: 2, INCONCLUSIVE: 1, DESCRIPTIVE: 0}
STEP_LABEL = {"check": DESCRIPTIVE, "measure": DESCRIPTIVE, "localize": DESCRIPTIVE, "decompose": DESCRIPTIVE,
              "control": DESCRIPTIVE, "impact": DESCRIPTIVE, "explain": ASSOCIATIONAL}


@dataclass
class Step:
    id: str
    kind: str          # check | measure | localize | decompose | explain | control | impact | conclude
    title: str
    query: dict
    result: dict
    decision: str
    options: list = field(default_factory=list)   # candidate next queries considered (for "where next")
    label: str = ""                               # evidence level of this finding


@dataclass
class Hypothesis:
    cause: str
    kind: str          # event | metric | planned | data
    status: str        # supported | hypothesis | ruled_out
    score: float
    evidence: list[str]
    against: list[str]
    refs: list[str]
    checks: dict = field(default_factory=dict)   # independent supports: timing, scope, control, upstream_anomaly ...
    label: str = ""                              # evidence level: descriptive | associational | causal_evidence | inconclusive


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
    stop_reason: str = ""
    branches: list = field(default_factory=list)     # one chain per driver (e.g. margin <- discounts <- discount_share)
    changes: list = field(default_factory=list)      # change-log entries first exposed around the onset, all tested
    did: dict = field(default_factory=dict)          # leaf metric -> difference-in-differences estimate
    confidence_label: str = ""
    trap: str | None = None                          # a local metric that went UP while the business went down

    def add(self, kind, title, query, result, decision, options=None, label=None) -> str:
        sid = f"q{len(self.steps) + 1}"
        self.steps.append(Step(sid, kind, title, query, _clean(result), decision, options or [],
                               label if label is not None else STEP_LABEL.get(kind, "")))
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
    EP_MIN, EP_SET, EP_JOINT = 0.10, 0.67, 0.60     # Adtributor thresholds: per element, per candidate set, joint cell

    @staticmethod
    def _surprise(p: pd.Series, q: pd.Series) -> pd.Series:
        """Adtributor surprise: per-element Jensen-Shannon term between the prior share p (expected) and
        the posterior share q (actual). High when a value's share of the whole moved unexpectedly."""
        p, q = p.clip(lower=0).astype(float), q.clip(lower=0).astype(float)
        m = (p + q) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(p > 0, p * np.log(p / m), 0.0)
            b = np.where(q > 0, q * np.log(q / m), 0.0)
        return pd.Series(0.5 * (a + b), index=p.index)

    def drill(self, metric: str, seg: dict, dim: str) -> dict | None:
        """One dimension: per-value explanatory power (share of the gap) and surprise.
        Sums are additive. Ratios use the symmetric Kitagawa decomposition, which is exact:
          delta = sum_v (wa-wb) * ((ra+rb)/2 - mid)   [mix]   +   sum_v (wa+wb)/2 * (ra-rb)   [rate]
        The parts are checked against the directly computed total; a residual goes into the ledger."""
        e = self.s.expr(metric)
        try:
            if e.kind != "ratio":
                a, b = self.actual(metric, seg, dim).astype(float), self.base(metric, seg, dim).astype(float)
                tot_a, tot_b = float(self.actual(metric, seg)), float(self.base(metric, seg))
                delta = a - b
                surprise = self._surprise(b / b.sum(), a / a.sum()) if a.sum() and b.sum() else delta * 0
                rows = {v: dict(actual=float(a[v]), expected=float(b[v]), delta=float(delta[v]),
                                rel=float(a[v] / b[v] - 1) if b[v] else None) for v in a.index}
                mix = None
            else:
                num, den = e.tree.left.id, e.tree.right.id
                Na, Da = self.actual(num, seg, dim).astype(float), self.actual(den, seg, dim).astype(float)
                Nb, Db = self.base(num, seg, dim).astype(float), self.base(den, seg, dim).astype(float)
                tot_a, tot_b = float(Na.sum() / Da.sum()), float(Nb.sum() / Db.sum())
                with np.errstate(divide="ignore", invalid="ignore"):
                    ra, rb = (Na / Da), (Nb / Db)
                ra, rb = ra.fillna(rb).fillna(tot_a), rb.fillna(ra).fillna(tot_b)   # empty cells carry no weight
                wa, wb = Da / Da.sum(), Db / Db.sum()
                mid = (tot_a + tot_b) / 2
                mix_v = (wa - wb) * ((ra + rb) / 2 - mid)
                rate_v = (wa + wb) / 2 * (ra - rb)
                delta = mix_v + rate_v
                surprise = self._surprise(Nb / Nb.sum(), Na / Na.sum()) + self._surprise(Db / Db.sum(), Da / Da.sum())
                rows = {v: dict(actual=float(ra[v]), expected=float(rb[v]), delta=float(delta[v]),
                                rel=float(ra[v] / rb[v] - 1) if rb[v] else None,
                                mix=float(mix_v[v]), rate=float(rate_v[v]),
                                weight_now=float(wa[v]), weight_before=float(wb[v])) for v in ra.index}
                mix = dict(mix=float(mix_v.sum()), rate=float(rate_v.sum()))
        except NotApplicable:
            return None
        total = tot_a - tot_b                       # computed directly, not from the parts
        if not total or total != total:
            return None
        residual = abs(float(delta.sum()) - total) / abs(total)
        for v in rows:
            rows[v]["ep"] = rows[v]["delta"] / total
            rows[v]["share"] = rows[v]["ep"]
            rows[v]["surprise"] = float(surprise[v])
        # Adtributor candidate set: most surprising values first, keep those that explain >= EP_MIN of the
        # gap, stop once the set explains >= EP_SET
        cand, cum = [], 0.0
        for v in sorted(rows, key=lambda v: -rows[v]["surprise"]):
            if rows[v]["ep"] >= self.EP_MIN:
                cand.append(v)
                cum += rows[v]["ep"]
                if cum >= self.EP_SET:
                    break
        top = cand[0] if cand else max(rows, key=lambda v: rows[v]["ep"])
        others = [r["rel"] for k, r in rows.items() if k != top and r["rel"] is not None]
        rest = float(np.median(others)) if others else 0.0
        top_rel = rows[top]["rel"]
        contrast = top_rel is not None and abs(top_rel) >= 2 * abs(rest) + 0.03
        return dict(metric=metric, dim=dim, total_delta=total, rows=rows, top=top, top_share=rows[top]["ep"],
                    top_rel=top_rel, rest_rel=rest, mix=mix, cand=cand, cand_ep=cum,
                    dim_surprise=float(sum(rows[v]["surprise"] for v in cand)), residual=residual,
                    single=len(cand) == 1 and cum >= self.EP_JOINT and contrast)

    def joint(self, metric: str, seg: dict, d1: str, v1: str, d2: str, v2: str, total: float) -> dict | None:
        """Test the combination d1=v1 x d2=v2 directly against its control cells (v1 x other d2 values,
        other d1 values x v2). Guards against Simpson effects: two marginal winners do not prove the cell."""
        e = self.s.expr(metric)

        def cell(s):
            if e.kind != "ratio":
                a, b = float(self.actual(metric, s)), float(self.base(metric, s))
                return a, b, a - b
            num, den = e.tree.left.id, e.tree.right.id
            Na, Da, Nb, Db = (float(self.actual(num, s)), float(self.actual(den, s)),
                              float(self.base(num, s)), float(self.base(den, s)))
            DA, DB = float(self.actual(den, seg)), float(self.base(den, seg))
            ra, rb = (Na / Da if Da else float("nan")), (Nb / Db if Db else float("nan"))
            contrib = (Da / DA + Db / DB) / 2 * (ra - rb) if DA and DB and ra == ra and rb == rb else 0.0
            return ra, rb, contrib

        try:
            a, b, contrib = cell({**seg, d1: v1, d2: v2})
            controls = {}
            for v in self.s.values(d2):
                if v != v2:
                    ca, cb, _ = cell({**seg, d1: v1, d2: v})
                    controls[f"{d1}={v1},{d2}={v}"] = (ca / cb - 1) if cb else None
            for v in self.s.values(d1):
                if v != v1:
                    ca, cb, _ = cell({**seg, d1: v, d2: v2})
                    controls[f"{d1}={v},{d2}={v2}"] = (ca / cb - 1) if cb else None
        except NotApplicable:
            return None
        rel = (a / b - 1) if b else None
        ctrl = [abs(c) for c in controls.values() if c is not None and c == c]
        ctrl_med = float(np.median(ctrl)) if ctrl else 0.0
        ctrl_max = max(ctrl) if ctrl else 0.0
        ep = contrib / total if total else 0.0
        ok = ep >= self.EP_JOINT and rel is not None and abs(rel) >= 2 * ctrl_med + 0.03
        return dict(cell=f"{d1}={v1},{d2}={v2}", rel=rel, ep=ep, controls=controls, control_median=ctrl_med,
                    control_max=ctrl_max, accepted=ok)

    def localize(self, inv: Investigation, metric: str, seg: dict) -> dict:
        while len(seg) < 3 and len(inv.steps) < self.budget:
            dims = [d for d in self.s.dims(metric) if d not in seg]
            options = [r for d in dims if (r := self.drill(metric, seg, d))]
            if not options:
                break
            options.sort(key=lambda o: -o["dim_surprise"])
            for o in options:
                if o["residual"] > 1e-6:
                    inv.add("check", f"DECOMPOSITION RESIDUAL: {metric} by {o['dim']}",
                            dict(metric=metric, dim=o["dim"]), dict(residual=o["residual"], total=o["total_delta"]),
                            f"WARNING: parts do not add up to the total change (residual {o['residual']:.2%}); "
                            "shares for this dimension are approximate")
                    inv.uncertainty.append(f"{metric} by {o['dim']}: decomposition residual {o['residual']:.2%}")
            opts_txt = [dict(query=f"drill {o['dim']}", score=round(o["dim_surprise"], 5),
                             why=f"{'+'.join(o['cand']) or o['top']} explains {o['cand_ep']:.0%} of the gap "
                                 f"(surprise {o['dim_surprise']:.4f}; {o['top']} {o['top_rel']:+.1%} vs "
                                 f"{o['rest_rel']:+.1%} elsewhere)") for o in options]

            def res_of(o):
                return dict(dim=o["dim"], top=o["top"], top_share=o["top_share"], top_rel=o["top_rel"],
                            rest_rel=o["rest_rel"], candidate_set=o["cand"], set_explains=o["cand_ep"],
                            surprise=o["dim_surprise"], residual=o["residual"], mix=o["mix"],
                            segments={k: dict(rel=v["rel"], ep=v["ep"], surprise=v["surprise"],
                                              **({"mix": v["mix"], "rate": v["rate"]} if "mix" in v else {}))
                                      for k, v in o["rows"].items()})

            singles = [o for o in options if o["single"]]
            if not singles:
                best = options[0]
                spread = (f"{'+'.join(best['cand'])} together explain {best['cand_ep']:.0%}" if len(best["cand"]) > 1
                          else f"best single value {best['dim']}={best['top']} explains {best['top_share']:.0%}, "
                               f"{best['top_rel']:+.1%} vs {best['rest_rel']:+.1%} elsewhere")
                inv.add("localize", f"WHERE: {metric} by {', '.join(o['dim'] for o in options)}",
                        dict(metric=metric, segment=seg_text(seg)), res_of(best),
                        f"not concentrated in one value ({spread}) -> stop narrowing", opts_txt)
                break
            best = singles[0]
            if len(singles) >= 2:      # two dimensions each point at one value: test the combination itself
                o1, o2 = singles[0], singles[1]
                j = self.joint(metric, seg, o1["dim"], o1["top"], o2["dim"], o2["top"], o1["total_delta"])
                if j:
                    worst = max(j["controls"], key=lambda k: abs(j["controls"][k] or 0)) if j["controls"] else "-"
                    if j["accepted"]:
                        seg = {**seg, o1["dim"]: o1["top"], o2["dim"]: o2["top"]}
                        inv.add("localize", f"JOINT: {metric} in {j['cell']} vs control cells",
                                dict(metric=metric, cell=j["cell"]), j,
                                f"combination accepted: the cell explains {j['ep']:.0%} of the gap and moved "
                                f"{j['rel']:+.1%}, control cells median {j['control_median']:.1%} "
                                f"(largest: {worst} {j['controls'].get(worst) or 0:+.1%}) -> narrow to both", opts_txt)
                        continue
                    inv.add("localize", f"JOINT: {metric} in {j['cell']} vs control cells",
                            dict(metric=metric, cell=j["cell"]), j,
                            f"combination rejected: the cell explains {j['ep']:.0%} and moved {j['rel']:+.1%}, but "
                            f"control cells moved too (median {j['control_median']:.1%}, e.g. {worst} "
                            f"{j['controls'].get(worst) or 0:+.1%}) -> keep only {best['dim']}={best['top']}", opts_txt)
            seg = {**seg, best["dim"]: best["top"]}
            inv.add("localize", f"WHERE: {metric} by {best['dim']}", dict(metric=metric, segment=seg_text(seg)),
                    res_of(best), f"localized to {best['dim']}={best['top']}: explains {best['top_share']:.0%} of the "
                                  f"gap (surprise {best['dim_surprise']:.4f}), {best['top_rel']:+.1%} there vs "
                                  f"{best['rest_rel']:+.1%} elsewhere -> narrow down", opts_txt)
            if best["mix"] and abs(best["mix"]["mix"]) > abs(best["mix"]["rate"]):
                inv.add("localize", f"WHERE: mix check for {metric}", dict(dim=best["dim"]), best["mix"],
                        "most of the change is a MIX effect (traffic shifted between segments), not a rate change")
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

    def decompose_sum(self, inv: Investigation, metric: str, seg: dict, d) -> list[str]:
        """Additive formula (margin = gmv - discounts - cogs - ad_cost): which terms carry the change?
        Signed contributions add up exactly to the change; every term pushing in the change's direction with
        >= 20% is a driver and gets its own branch. Terms pushing the other way are reported as offsets."""
        try:
            ma, mb = float(self.actual(metric, seg)), float(self.base(metric, seg))
            terms = {f: (float(self.actual(f, seg)), float(self.base(f, seg))) for f in d.factors}
        except NotApplicable:
            return []
        total = ma - mb
        if not total:
            return []
        # Volume vs rate: costs that grow in proportion to the base (first positive term, e.g. gmv) are not
        # drivers. With r_i = term_i / base:  delta = (base_a - base_b) * margin_rate_b  [volume]
        #                                             + sum_i sign_i * base_a * (r_ia - r_ib)  [rate, per term]
        # This split is exact. Drivers are terms whose RATE effect carries >= 20% of the change.
        base_f = next((f for s_, f in zip(d.signs, d.factors) if s_ > 0), d.factors[0])
        ba, bb = terms[base_f]
        rows = {}
        volume = (ba - bb) * (mb / bb) if bb else 0.0
        for sign, f in zip(d.signs, d.factors):
            a, b = terms[f]
            raw = sign * (a - b)
            if f == base_f or not (ba and bb):
                rate_eff = 0.0 if f == base_f else raw
            else:
                rate_eff = sign * ba * (a / ba - b / bb)
            rows[f] = dict(actual=a, expected=b, change=(a / b - 1) if b else None, sign=sign,
                           contribution=raw, share_raw=raw / total,
                           share_of_base_now=(a / ba) if ba else None, share_of_base_before=(b / bb) if bb else None,
                           rate_effect=rate_eff, share=rate_eff / total)
        rows["(volume)"] = dict(actual=None, expected=None, change=(ba / bb - 1) if bb else None, sign=1,
                                contribution=volume, share_raw=None, rate_effect=volume, share=volume / total)
        resid = abs(sum(r["rate_effect"] for r in rows.values()) - total) / abs(total)
        drivers = sorted([f for f in d.factors if rows[f]["share"] >= 0.20], key=lambda f: -rows[f]["share"])
        offsets = [f for f in rows if rows[f]["share"] <= -0.10]
        def ratetxt(f):
            r = rows[f]
            if f == base_f or r.get("share_of_base_before") is None:
                return f"{r['change']:+.0%}"
            return f"{r['share_of_base_before']:.1%} -> {r['share_of_base_now']:.1%} of {base_f}"

        opts = [dict(query=f"follow {f}", score=round(rows[f]["share"], 3),
                     why=f"{'+' if rows[f]['sign'] > 0 else '-'}{f}: {ratetxt(f)}, rate effect "
                         f"{rows[f]['share']:.0%} of the change") for f in d.factors]
        drv = ", ".join(f"{f} ({ratetxt(f)}, {rows[f]['share']:.0%} of the change)" for f in drivers)
        off = ", ".join(f"{'volume: ' + base_f if f == '(volume)' else f} ({rows[f]['change']:+.0%}, "
                        f"{rows[f]['share']:.0%})" for f in offsets)
        decision = f"{metric} {total:+,.0f} ({ma / mb - 1:+.1%}): driven by {drv or 'no single term'}"
        if off:
            decision += f"; offset by {off}"
        decision += (f" -> follow {len(drivers)} drivers separately" if len(drivers) > 1 else
                     f" -> follow {drivers[0]}" if drivers else "")
        if resid > 1e-6:
            decision += f" (WARNING: terms leave a residual of {resid:.2%})"
        inv.add("decompose", f"WHAT: {metric}[{seg_text(seg)}] = {d.text.split(' = ', 1)[1]}",
                dict(metric=metric, segment=seg_text(seg), formula=d.text),
                dict(total_change=total, terms=rows, residual=resid), decision, opts)
        return drivers

    def did(self, inv: Investigation, metric: str, seg: dict, onset, direction: str) -> dict:
        """Difference-in-differences: affected segment vs everything else (control), the 24h before the onset
        vs the hours since. Hour-of-day and day effects shared by both groups cancel out. Effect on the log
        scale (relative), 95% interval from the hourly differences (Welch). Recorded in the ledger."""
        q = dict(metric=metric, treated=seg_text(seg), control=f"not ({seg_text(seg)})",
                 pre="24h before onset", post="onset .. now")
        if not seg or onset is None:
            why = "no localized segment" if not seg else "no clear onset"
            out = dict(metric=metric, ok=False, reason=why, significant=False)
            out["ref"] = inv.add("did", f"DiD: {metric}", q, out, f"not estimable: {why}", label=INCONCLUSIVE)
            inv.did[metric] = out
            return out
        post = pd.date_range(onset, self.now, freq="h")
        pre = pd.date_range(onset - pd.Timedelta(hours=24), onset - H, freq="h")
        try:
            T = self.s.series(metric, seg)
            C = self.s.series(metric, {"__not__": seg})
        except NotApplicable:
            out = dict(metric=metric, ok=False, reason="no control segment for this metric", significant=False)
            out["ref"] = inv.add("did", f"DiD: {metric}", q, out, "not estimable: no control segment", label=INCONCLUSIVE)
            inv.did[metric] = out
            return out
        idx = pre.append(post)
        t, c = T.reindex(idx), C.reindex(idx)
        log = bool((t > 0).all() and (c > 0).all())
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (np.log(t) - np.log(c)) if log else (t - c)
        dpre, dpost = d.reindex(pre).dropna(), d.reindex(post).dropna()
        if len(dpost) < 2 or len(dpre) < 6:
            out = dict(metric=metric, ok=False, reason=f"too few hours (pre {len(dpre)}, post {len(dpost)})",
                       significant=False)
            out["ref"] = inv.add("did", f"DiD: {metric}[{seg_text(seg)}]", q, out,
                                 f"not estimable yet: {out['reason']}", label=INCONCLUSIVE)
            inv.did[metric] = out
            return out
        eff = float(dpost.mean() - dpre.mean())
        se = float(math.sqrt(dpost.var(ddof=1) / len(dpost) + dpre.var(ddof=1) / len(dpre)))
        lo, hi = eff - 1.96 * se, eff + 1.96 * se
        signif = lo > 0 or hi < 0
        same_dir = (eff < 0) == (direction == "down")

        def conv(x):
            return math.exp(x) - 1 if log else x

        out = dict(metric=metric, ok=True, scale="log" if log else "level", effect=eff, se=se, ci95=[lo, hi],
                   effect_pct=conv(eff) if log else None, ci95_pct=[conv(lo), conv(hi)] if log else None,
                   n_pre=len(dpre), n_post=len(dpost), significant=signif and same_dir,
                   control=f"not ({seg_text(seg)})")
        txt = (f"{conv(eff):+.1%} [95% CI {conv(lo):+.1%} .. {conv(hi):+.1%}]" if log else
               f"{eff:+,.4g} [95% CI {lo:+,.4g} .. {hi:+,.4g}]")
        verdict = ("interval excludes 0: the affected segment moved against its control at the onset"
                   if out["significant"] else "interval includes 0 (or wrong sign): no separable effect")
        out["ref"] = inv.add("did", f"DiD: {metric} in {seg_text(seg)} vs rest, before vs after {onset:%d.%m %H:%M}",
                             q, out, f"effect {txt}, {len(dpre)}h pre / {len(dpost)}h post -> {verdict}",
                             label=CAUSAL if out["significant"] else INCONCLUSIVE)
        inv.did[metric] = out
        return out

    def _note_change(self, inv: Investigation, ev: dict, leaf: str, status: str, reason: str, gap_h: float):
        if not (-1.0 <= gap_h <= CONCURRENT_H):
            return
        entry = next((x for x in inv.changes if x["name"] == ev["name"]), None)
        if entry is None:
            entry = dict(name=ev["name"], type=ev["type"], owner=ev.get("owner") or "unknown",
                         deployed_at=str(ev.get("deployed_at") or ev["ts"]),
                         first_exposure_at=str(ev.get("first_exposure_at") or ev["ts"]),
                         hours_before_onset=round(gap_h, 2), verdicts={})
            inv.changes.append(entry)
        entry["verdicts"][leaf] = dict(status=status, reason=reason)

    def explain(self, inv: Investigation, leaf: str, seg: dict, direction: str, onset: pd.Timestamp | None,
                branch: list | None = None):
        branch = branch or inv.chain
        # a change must be able to move the LEAF: the decomposition already showed that the metrics above it
        # changed only through the leaf, so a component acting on those (e.g. product_views) is not a candidate
        comps_for = set(self.g.components_for(leaf))
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
            routine = bool(ev.get("planned", True)) and ev["type"] in ("purchasing",)
            evidence, against = [], []
            if comp_ok:
                evidence.append(f"component '{comp}' can move {leaf}")
            else:
                against.append(f"component '{comp}' cannot move {leaf} in the graph")
            narrower = [x for x in scope_notes if "event only for" in x]
            if ok_scope and not narrower:
                evidence.append("scope matches the affected segment" + (f" ({'; '.join(scope_notes)})" if scope_notes else ""))
            elif ok_scope:
                against.append("scope covers only part of the change: " + "; ".join(narrower))
            else:
                against += scope_notes
            if timing_ok:
                evidence.append(f"{'active since' if active else 'first exposed'} {ev['ts']:%d.%m %H:%M}, "
                                f"{gap_h:+.1f}h before the onset {ref_t:%d.%m %H:%M}" if gap_h >= 0 else
                                f"first exposed {ev['ts']:%H:%M}, within the onset hour")
            else:
                against.append(f"timing: first exposure {ev['ts']:%d.%m %H:%M} is {gap_h:+.1f}h from the onset "
                               f"(allowed 0..{lag}h for {comp})")
            if routine:
                against.append("routine planned run reported at normal volume; order quantities are not in the data")
            if ev["type"] in ("rollback",) and ev["ts"] > ref_t:
                against.append("this is a later rollback (a fix, not a cause)")
            full_scope = ok_scope and not narrower
            score = (2 * comp_ok + 2 * ok_scope + 2 * timing_ok + (0.5 if not ev.get("planned", True) else 0)
                     - (2 if routine else 0) - (1.5 if ok_scope and narrower else 0))
            status = "hypothesis" if (comp_ok and ok_scope and timing_ok) else "ruled_out"
            owner = ev.get("owner") or "unknown"
            ref = inv.add("explain", f"EVENT: {ev['name']} ({owner})",
                          dict(type=ev["type"], component=comp, owner=owner,
                               deployed_at=str(ev.get("deployed_at") or ev["ts"]),
                               first_exposure_at=str(ev.get("first_exposure_at") or ev["ts"]), for_leaf=leaf,
                               scope={k: ev.get(k) for k in ("country", "platform", "category")}),
                          dict(component_match=comp_ok, scope_match=ok_scope, timing_ok=timing_ok, gap_h=gap_h),
                          f"{status} for {leaf}: " + "; ".join(evidence if status != "ruled_out" else against)
                          + (" (competing explanation, weaker: routine)" if routine and status != "ruled_out" else ""))
            self._note_change(inv, ev, leaf, status, "; ".join(evidence if status != "ruled_out" else against), gap_h)
            kind = "planned" if ev["type"] == "campaign" else "event"
            h = Hypothesis(f"{ev['type']}: {ev['name']}", kind, status, score, evidence, against, [ref],
                           dict(component=comp_ok, scope=full_scope, timing=timing_ok, routine=routine,
                                level="leaf", leaf=leaf))
            if status != "ruled_out":
                self.control(inv, h, ev, leaf, seg, direction)
            inv.hypotheses.append(h)

        # possible-cause edges (metric -> metric), incl. ambiguous ones from the extension agent
        for c in [c for c in self.s.catalog.get("causes") or [] if c["to"] in branch]:
            cause = c["from"]
            cseg = {k: v for k, v in seg.items() if k in self.s.dims(cause)}
            m = detect.measure(self.s, cause, cseg, self.now)
            ref = inv.add("measure", f"CAUSE?: {cause}[{seg_text(cseg)}] -> {c['to']}",
                          dict(edge=c.get("reason", ""), status=c.get("status", "hypothesis")), m.to_dict(),
                          f"{cause} is {m.status} ({m.rel:+.1%})" if m.rel == m.rel else f"{cause}: {m.status}")
            if m.status == "anomaly":
                o = detect.onset(self.s, cause, cseg, self.now, direction=m.direction)
                # hourly onsets of gradual changes are only precise to ~2h: within that they count as simultaneous
                ok_time = o is None or onset is None or o <= onset + 2 * H
                ev_ = [f"{cause} itself moved {m.rel:+.1%} in {seg_text(cseg)} (z={m.z:+.1f})",
                       f"graph edge: {c.get('reason', '')}"]
                if o is not None and onset is not None:
                    rel_t = ("precedes" if o < onset else "matches" if o == onset else
                             "follows within 2h (simultaneous at hourly resolution)" if ok_time else "is AFTER")
                    ev_.append(f"its onset {o:%d.%m %H:%M} {rel_t} the {c['to']} onset {onset:%d.%m %H:%M}")
                elif o is not None:
                    ev_.append(f"onset {o:%d.%m %H:%M}")
                self.did(inv, cause, cseg, o or onset, m.direction)
                ambiguous = c.get("status") == "ambiguous"
                inv.hypotheses.append(Hypothesis(f"metric: {cause} ({'co-moving, direction unknown' if ambiguous else 'upstream cause'})",
                                                 "metric", "hypothesis",
                                                 5 + (0 if ambiguous else 2) + (1 if ok_time else -2), ev_,
                                                 [] if ok_time else ["cause moved after the effect"], [ref],
                                                 dict(upstream_anomaly=True, timing=ok_time, scope=True,
                                                      ambiguous=ambiguous, level="leaf", leaf=leaf, metric=cause,
                                                      segment=cseg)))
                if not ambiguous and ok_time:
                    # the cause metric is itself a node: look for what moved it (one level, depth-limited)
                    inv.chain.append(cause)
                    co = o or onset
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
            self._note_change(inv, ev, metric, status, "; ".join(ev_txt if status != "ruled_out" else against), gap_h)
            score = 6 + (1 if not ev.get("planned", True) else 0) - (2 if routine else 0)
            inv.hypotheses.append(Hypothesis(f"{ev['type']}: {ev['name']}", "event", status, score, ev_txt,
                                             against, [ref], dict(component=True, scope=ok_scope, timing=timing_ok,
                                                                  routine=routine, level="upstream", leaf=metric)))

    def control(self, inv: Investigation, h: Hypothesis, ev: dict, leaf: str, seg: dict, direction: str):
        """Control groups: is the leaf metric normal in the other values of each localized dimension?
        Inside the event's scope this shows the change is specific to the segment; outside its scope it shows
        nothing else moved the metric at the same time. Either way it is evidence independent of timing."""
        ran, all_normal = 0, True
        for dim in ("country", "platform", "category"):
            if dim in seg:
                wide = ev.get(dim, "*") in ("*", None) or ev.get(dim) != ev.get(dim)
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
                ran += 1
                all_normal = all_normal and bool(rows) and len(normal) == len(rows)
                ref = inv.add("control", f"CONTROL: {leaf} in other {dim} values "
                                         f"({'inside' if wide else 'outside'} the event scope)",
                              dict(metric=leaf, base_segment=seg_text(cs), dim=dim), rows,
                              f"{len(normal)}/{len(rows)} unaffected -> the change is specific to {dim}={seg[dim]}"
                              if len(normal) == len(rows) else
                              f"also affected: {', '.join(v for v in rows if v not in normal)}")
                h.refs.append(ref)
                if len(normal) == len(rows) and rows:
                    h.evidence.append(f"control: {leaf} normal in {', '.join(normal)} -> only {dim}={seg[dim]} is hit")
                    if wide:
                        inv.uncertainty.append(f"'{ev['name']}' covers all {dim} values but only {seg[dim]} is "
                                               f"affected: mechanism specific to {seg[dim]} is inferred, not verified")
        if ran:
            h.checks["control"] = all_normal

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
                                              f"sessions (complete source) {ctrl.rel:+.1%}"], [], [ref, r2],
                                             dict(completeness=True, control=True, level="leaf")))
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

        # business check: a local metric going UP is only good news if the business outcome agrees
        if direction == "up" and metric != BUSINESS and BUSINESS in self.s.metrics:
            try:
                b = detect.measure(self.s, BUSINESS, seg, self.now, persistence=False)
            except NotApplicable:
                b = None
            if b is not None:
                # a guard, not a detector: a clear drop of the business outcome is enough to contradict the rise
                pivot = b.direction == "down" and (b.status in ("anomaly", "candidate")
                                                   or (b.rel == b.rel and b.rel <= -0.10 and b.z <= -2.5))
                inv.add("check", f"BUSINESS CHECK: {BUSINESS}[{seg_text(seg)}] while {metric} rose",
                        dict(metric=BUSINESS, segment=seg_text(seg)), b.to_dict(),
                        (f"PIVOT: {metric} is up but {BUSINESS} is down {b.rel:+.1%} (z={b.z:+.1f}): the local win is "
                         f"a business loss -> investigate {BUSINESS} instead of stopping at {metric}") if pivot else
                        f"{BUSINESS} {b.rel:+.1%}: the business outcome does not contradict the {metric} increase")
                if pivot:
                    inv.trap = metric
                    metric, direction = BUSINESS, "down"
                    inv.chain.append(metric)

        # WHAT: follow formula edges. An additive formula can have several drivers: one branch per driver.
        stems = [[metric]]
        narrowed = len(seg) > 0
        for _ in range(8):
            if len(inv.steps) >= self.budget:
                inv.uncertainty.append("query budget exhausted before a leaf metric was reached")
                break
            d = self.g.decomposition(metric)
            if d is not None and d.op == "sum":
                drivers = self.decompose_sum(inv, metric, seg, d)
                if drivers:
                    stems = [[metric, drv] for drv in drivers]
                break
            nxt = self.decompose(inv, metric, seg)
            if not nxt:
                break
            share = inv.steps[-1].result["factors"][nxt]["share"]
            metric = nxt
            inv.chain.append(metric)
            stems = [list(inv.chain)]
            if not narrowed and share >= 0.85:
                # the factor carries (almost) the whole change and the parent did not concentrate anywhere:
                # the same drill-down would only repeat itself, so spend the query budget further down instead
                continue
            before = len(seg)
            seg = self.localize(inv, metric, seg)
            narrowed = len(seg) > before
        # each driver branch: follow its own product formula down to a leaf (segment stays fixed)
        branches = []
        for stem in stems:
            branch = list(stem)
            m = branch[-1]
            for _ in range(6):
                d = self.g.decomposition(m)
                if len(stems) == 1 or d is None or d.op != "product":
                    break
                nxt = self.decompose(inv, m, seg)
                if not nxt:
                    break
                branch.append(nxt)
                m = nxt
            branches.append(branch)
        inv.branches = branches
        for br in branches:
            for m in br:
                if m not in inv.chain:
                    inv.chain.append(m)
        inv.segment = seg

        # WHY: per branch leaf - onset, competing changes, cause edges, difference-in-differences
        onsets = []
        for br in branches:
            leaf = br[-1]
            leaf_m = detect.measure(self.s, leaf, seg, self.now)
            ldir = leaf_m.direction or direction
            o = detect.onset(self.s, leaf, seg, self.now, direction=ldir)
            if o is not None:
                onsets.append(o)
            inv.add("measure", f"LEAF: {leaf}[{seg_text(seg)}]", dict(branch=" <- ".join(br)),
                    dict(**leaf_m.to_dict(), onset=str(o) if o is not None else None),
                    f"proximate metric: {leaf_m.short()}, onset {o:%d.%m %H:%M}" if o is not None else
                    f"proximate metric: {leaf_m.short()} (onset not clear)")
            self.explain(inv, leaf, seg, ldir, o, branch=br)
            self.did(inv, leaf, seg, o, ldir)
        inv.onset = str(min(onsets)) if onsets else None
        return self.finish(inv)

    # ------------------------------------------------------------------ conclusion
    def finish(self, inv: Investigation) -> Investigation:
        seg = inv.segment or inv.trigger["segment"]
        # one entry per explanation: keep the strongest verdict (an event can be ruled out for one leaf but
        # live for another, or for the upstream cause of a cause metric)
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
            if m not in self.s.metrics:
                continue
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
        t = inv.trigger
        if seg != t["segment"] and t["metric"] in OUTCOMES + ["conversion_rate"] and inv.classification != "data_incomplete":
            try:
                tot_a, tot_b = float(self.actual(t["metric"], t["segment"])), float(self.base(t["metric"], t["segment"]))
                sa, sb = float(self.actual(t["metric"], seg)), float(self.base(t["metric"], seg))
                if self.s.expr(t["metric"]).kind != "ratio" and tot_a != tot_b:
                    inv.explained_share = (sa - sb) / (tot_a - tot_b)
            except NotApplicable:
                pass

        # ---- explicit stop rule, per branch leaf. The agent never stops at the first anomaly: it always checked
        # the data, tried to localize, followed the formula to a leaf and tested every candidate explanation.
        #  (a) the leading explanation has >= 2 independent supports (timing, segment, control group normal,
        #      upstream metric moved first, source completeness)
        #  (b) the alternatives were tested and refuted
        #  (c) the data cannot tell explanations apart -> hypothesis with stated uncertainty
        SUPPORTS = ("timing", "scope", "control", "upstream_anomaly", "completeness")
        names = dict(timing="timing fits", scope="segment fits", control="control group normal",
                     upstream_anomaly="upstream metric moved", completeness="source incomplete")

        def supports(h):
            return [k for k in SUPPORTS if h.checks.get(k)]

        refuted = [h for h in inv.hypotheses if h.status == "ruled_out"]
        live = [h for h in inv.hypotheses if h.status != "ruled_out"]
        harmful = any(v["rel"] is not None and v["rel"] <= -0.03 for v in imp.values())
        codes, why, leaders, rivals_all = [], [], [], []
        if inv.classification == "data_incomplete":
            codes, why = ["a"], ["source completeness below 95% and a complete control source is normal"]
            for h in live:
                h.label = DESCRIPTIVE
        else:
            leaves = [br[-1] for br in inv.branches] or [inv.chain[-1] if inv.chain else t["metric"]]
            for leaf in dict.fromkeys(leaves):
                group = [h for h in live if h.checks.get("level", "leaf") == "leaf" and h.checks.get("leaf", leaf) == leaf]
                group.sort(key=lambda h: (-len(supports(h)), -h.score))
                if not group:
                    tested = [h for h in refuted if h.checks.get("leaf") == leaf]
                    codes.append("b" if tested else "c")
                    why.append(f"{leaf}: all {len(tested)} candidate explanations refuted" if tested else
                               f"{leaf}: no candidate explanation in the change log or the graph")
                    continue
                planned = [h for h in group if h.kind == "planned"]
                leader = planned[0] if planned and not harmful else group[0]
                sup_l = supports(leader)
                rivals = [h for h in group if h is not leader and len(supports(h)) >= 2
                          and h.score >= leader.score - 1 and not h.checks.get("ambiguous")]
                if len(sup_l) >= 2 and not rivals and not leader.checks.get("ambiguous"):
                    leader.status = "supported"
                    leaders.append(leader)
                    codes.append("a")
                    why.append(f"{leaf}: '{leader.cause}' has {len(sup_l)} independent supports "
                               f"({', '.join(names[k] for k in sup_l)})")
                    others = [h for h in refuted if h.checks.get("leaf") == leaf]
                    if others:
                        codes.append("b")
                        why.append(f"{leaf}: {len(others)} alternative(s) tested and refuted")
                    for h in group:
                        if h is not leader:
                            h.against.append(f"weaker than '{leader.cause}'")
                else:
                    codes.append("c")
                    leaders.append(leader)          # named, but stays a hypothesis
                    rivals_all += [leader] + rivals
                    if rivals:
                        why.append(f"{leaf}: the data cannot tell '{leader.cause}' from "
                                   f"{', '.join(repr(h.cause) for h in rivals[:2])}")
                        inv.uncertainty.append("competing explanations the data cannot separate: " +
                                               ", ".join(h.cause for h in [leader] + rivals[:2]))
                    else:
                        why.append(f"{leaf}: '{leader.cause}' has only {len(sup_l)} independent support(s) "
                                   f"({', '.join(names[k] for k in sup_l) or 'none'}): reported as a hypothesis")
                        inv.uncertainty.append(f"'{leader.cause}' is not confirmed by two independent checks")
            # one level further up (what moved a cause metric): confirmed only if a single candidate is left
            up_live = [h for h in live if h.checks.get("level") == "upstream"]
            strong_up = [h for h in up_live if len(supports(h)) >= 2 and not h.checks.get("routine")]
            if len(strong_up) == 1 and len([h for h in up_live if not h.checks.get("routine")]) == 1:
                strong_up[0].status = "supported"
                if len(up_live) > 1:
                    inv.uncertainty.append("competing explanation not excluded by data: " + ", ".join(
                        h.cause for h in up_live if h is not strong_up[0]) + " (ranked lower: routine run)")
            elif up_live:
                codes.append("c")
                why.append("first cause one level up stays a hypothesis: " +
                           ", ".join(h.cause for h in up_live[:3]) + " are not separable with this data")
                inv.uncertainty.append("competing explanations still open: " +
                                       ", ".join(f"{h.cause} ({h.score:g})" for h in up_live[:3]))
            # classification
            sup = [h for h in leaders if h.status == "supported"]
            if not leaders:
                inv.classification = "unexplained" if harmful or t["direction"] == "down" else "change"
                inv.uncertainty.append("no event or upstream metric explains the change; data shows WHERE and WHAT, "
                                       "not yet WHY")
            elif sup and all(h.kind == "planned" for h in sup) and not harmful:
                inv.classification = "innocent"
                for h in sup:
                    h.evidence.append("outcomes are not down: " + ", ".join(
                        f"{m} {v['rel']:+.1%}" for m, v in imp.items() if v["rel"] is not None))
            else:
                inv.classification = "fault" if harmful or t["direction"] == "down" else "change"
                if inv.trap:
                    inv.uncertainty.append(f"{inv.trap} went up, but {BUSINESS} went down: do not read the "
                                           f"{inv.trap} increase as success")

        # ---- evidence levels: DiD on the hypothesis' leaf, the stop rule's verdict, and whether rivals remain
        for h in inv.hypotheses:
            if h.status == "ruled_out":
                h.label = ""
                continue
            if h.kind == "data":
                h.label = DESCRIPTIVE
                continue
            leaf = h.checks.get("metric") if h.kind == "metric" else h.checks.get("leaf")
            if h.checks.get("level") == "upstream":
                leaf = h.checks.get("leaf")
            dd = inv.did.get(leaf) if leaf else None
            if h in rivals_all:
                h.label = INCONCLUSIVE
            elif h.status == "supported" and dd and dd.get("significant"):
                h.label = CAUSAL
                h.evidence.append(f"DiD on {leaf}: {_did_text(dd)} (interval excludes 0)")
                h.refs.append(dd["ref"])
            elif h.status == "supported":
                h.label = ASSOCIATIONAL
                if dd:
                    h.evidence.append(f"DiD on {leaf}: {_did_text(dd) if dd.get('ok') else dd.get('reason')} "
                                      "(no separable effect: association only)")
                    h.refs.append(dd["ref"])
            else:
                h.label = ASSOCIATIONAL
        named = [h for h in leaders if h.status == "supported"]
        if inv.classification == "data_incomplete":
            inv.confidence_label = DESCRIPTIVE
        elif named:
            inv.confidence_label = min((h.label for h in named), key=lambda x: LABEL_RANK[x])
        elif leaders or rivals_all:
            inv.confidence_label = INCONCLUSIVE
        else:
            inv.confidence_label = DESCRIPTIVE

        n_loc = sum(s.kind == "localize" for s in inv.steps)
        n_dec = sum(s.kind == "decompose" for s in inv.steps)
        inv.stop_reason = "+".join(f"({c})" for c in dict.fromkeys(codes))
        conc = [c for c in inv.changes if len(c["verdicts"])]
        inv.add("stop", "STOP: why the investigation ends here",
                dict(rule="(a) >=2 independent supports | (b) alternatives refuted | (c) data cannot distinguish"),
                dict(reason=list(dict.fromkeys(codes)), queries=len(inv.steps), localization_steps=n_loc,
                     decomposition_steps=n_dec, explanations_tested=len(inv.hypotheses), refuted=len(refuted),
                     still_open=len([h for h in inv.hypotheses if h.status == "hypothesis"]),
                     concurrent_changes=[c["name"] for c in conc], evidence_level=inv.confidence_label),
                f"stop {inv.stop_reason}: " + "; ".join(why)
                + f". Before stopping: {n_loc} localization and {n_dec} decomposition steps, "
                  f"{len(inv.hypotheses)} explanations tested ({len(refuted)} refuted)"
                + (f", {len(conc)} concurrent changes checked" if conc else "") + ".",
                label=inv.confidence_label)

        # ---- naming: one incident per root cause
        live = sorted([h for h in inv.hypotheses if h.status != "ruled_out"],
                      key=lambda h: (h.status != "supported", -h.score))
        sup_events = [h for h in named if h.kind in ("event", "planned")]
        up_sup = [h for h in inv.hypotheses if h.status == "supported" and h.checks.get("level") == "upstream"]
        up_lead = sorted([h for h in inv.hypotheses if h.status != "ruled_out" and h.checks.get("level") == "upstream"],
                         key=lambda h: (h.status != "supported", h.checks.get("routine", False), -h.score))
        if sup_events:
            cause_names = sorted(h.cause for h in sup_events)
        elif up_sup or up_lead:
            cause_names = [(up_sup or up_lead)[0].cause]
        elif leaders and leaders[0].kind in ("event", "planned"):
            cause_names = [leaders[0].cause]
        else:
            cause_names = []
        leaf = inv.chain[-1] if inv.chain else t["metric"]
        inv.symptoms = ([inv.trap] if inv.trap else []) + [m for m in inv.chain[:-1] if m != inv.trap] + \
            [m for m in OUTCOMES if m not in inv.chain and m in imp and imp[m]["rel"] is not None and abs(imp[m]["rel"]) > 0.03]
        inv.signature = inv.signature or (f"cause|{' + '.join(cause_names)}" if cause_names else f"{leaf}|{seg_text(seg)}")
        lab = f"[{inv.confidence_label}] "
        if inv.classification == "data_incomplete":
            inv.headline = f"{lab}Data incomplete: {t['metric']} looks {t['rel']:+.0%} but the source is still loading"
        elif inv.classification == "innocent":
            inv.headline = (f"{lab}Innocent change: {t['metric']} {t['rel']:+.0%} in {seg_text(seg)} explained by "
                            f"{' + '.join(h.cause for h in named) or 'a planned event'}; outcomes not hurt")
        else:
            lost = imp.get(BUSINESS, imp.get("gmv", {})).get("per_hour")
            lost_m = BUSINESS if BUSINESS in imp else "gmv"
            drivers = " + ".join(br[-1] for br in inv.branches) if inv.branches else leaf
            head = (f"{inv.trap} {t['rel']:+.0%} but {BUSINESS} {imp[BUSINESS]['rel']:+.0%}" if inv.trap and BUSINESS in imp
                    else f"{t['metric']} {t['rel']:+.0%}")
            cause_txt = (" + ".join(h.cause.split(' (')[0] for h in named) if named else
                         (f"likely {cause_names[0]} (hypothesis)" if cause_names else "cause not confirmed"))
            if named and all(h.kind == "metric" for h in named) and (up_sup or up_lead):
                cause_txt += f" -> {'likely ' if not up_sup else ''}{(up_sup or up_lead)[0].cause}" + \
                             ("" if up_sup else " (hypothesis)")
            inv.headline = (f"{lab}{head}: {drivers} in {seg_text(seg)} -> {cause_txt}"
                            + (f"; {lost_m} {lost:+,.0f} EUR/h" if lost is not None else ""))
        inv.add("conclude", "CONCLUSION", dict(classification=inv.classification),
                dict(chain=inv.chain, branches=inv.branches, segment=seg_text(seg),
                     supported=[h.cause for h in named], evidence_level=inv.confidence_label,
                     hypotheses=[(h.cause, h.status, h.score, h.label) for h in inv.hypotheses]),
                inv.headline, label=inv.confidence_label)
        return inv


def _did_text(dd: dict) -> str:
    if dd.get("scale") == "log":
        lo, hi = dd["ci95_pct"]
        return f"{dd['effect_pct']:+.1%} [95% CI {lo:+.1%} .. {hi:+.1%}]"
    lo, hi = dd["ci95"]
    return f"{dd['effect']:+,.4g} [95% CI {lo:+,.4g} .. {hi:+,.4g}]"
