"""The agent's model of the business: a typed metric graph derived from metric definitions.

Edge types and the reasoning each one supports
  formula        exact arithmetic. a = b * (a/b): a change of `a` splits exactly into log-contributions
                 of its factors. Derived from expressions, never typed in by hand.
  composite      a metric equals a product of other metrics (e.g. conversion_rate = product of the six
                 funnel rates). Found by walking formula chains, accepted only if it validates on data.
  breakdown      a metric split by a dimension (country, platform, category). Sums are additive;
                 ratios split into a mix effect and a rate effect.
  possible_cause a hypothesis edge (inventory -> cart rate, change-log component -> metric). Tested with
                 timing and segment overlap during investigation, never used as arithmetic.

Every formula/composite/breakdown edge is validated against the data (total and every dimension value)
and keeps its residual, so a definition that stops matching reality is detected instead of trusted.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .store import NotApplicable, Store


@dataclass
class Decomp:
    parent: str
    factors: list[str]
    op: str                      # product | ratio | sum
    kind: str                    # identity | definition | composite
    text: str
    status: str = "unvalidated"  # valid | invalid | unvalidated
    residual: float | None = None
    checked: int = 0             # number of (segment, hour) points checked
    primary: bool = False
    signs: list = field(default_factory=list)    # op == "sum": +1 / -1 per factor


@dataclass
class Edge:
    src: str                     # the metric being explained
    dst: str                     # the component / breakdown / cause
    type: str                    # formula | composite | breakdown | possible_cause | definition
    label: str = ""
    status: str = "unvalidated"
    meta: dict = field(default_factory=dict)


class MetricGraph:
    def __init__(self, store: Store):
        self.store = store
        self.decomps: list[Decomp] = []
        self.edges: list[Edge] = []
        self.causes: list[dict] = []
        self.log: list[str] = []

    # ------------------------------------------------------------------ build
    @staticmethod
    def build(store: Store, validate: bool = True) -> "MetricGraph":
        g = MetricGraph(store)
        g._derive()
        if validate:
            g.validate()
        g._edges()
        return g

    @property
    def metrics(self) -> dict:
        return self.store.metrics

    def _derive(self):
        s = self.store
        identities: dict[str, list[Decomp]] = {}
        for name in self.metrics:
            e = s.expr(name)
            if e.kind == "ratio":
                a, b = e.tree.left.id, e.tree.right.id
                self.decomps.append(Decomp(name, [a, b], "ratio", "definition", f"{name} = {a} / {b}"))
                identities.setdefault(a, []).append(
                    Decomp(a, [b, name], "product", "identity", f"{a} = {b} x {name}"))
            elif e.kind == "product":
                self.decomps.append(Decomp(name, e.metrics, "product", "identity", f"{name} = {e.text}", primary=True))
            elif e.kind == "sum" and e.terms:
                # a metric's own formula is its primary decomposition (e.g. margin = gmv - discounts - cogs - ad_cost)
                self.decomps.append(Decomp(name, [t[1] for t in e.terms], "sum", "identity", f"{name} = {e.text}",
                                           primary=True, signs=[t[0] for t in e.terms]))
        own = {d.parent for d in self.decomps if d.primary}

        # primary decomposition: the one whose first factor can itself be decomposed furthest
        def depth(m, seen=()):
            if m in seen or m not in identities:
                return 0
            return 1 + max(depth(d.factors[0], seen + (m,)) for d in identities[m])

        for parent, cands in identities.items():
            cands.sort(key=lambda d: -depth(d.factors[0]))
            if parent in own:       # its own formula wins; ratio identities stay as alternatives
                self.decomps.extend(cands)
                continue
            cands[0].primary = True
            self.decomps.extend(cands)
            for d in cands[1:]:
                self.log.append(f"{parent}: chose '{cands[0].text}' as primary (deeper chain) over '{d.text}'")

        # composite: a ratio a/b whose identity is not primary = product of the primary chain a -> ... -> b
        primary = {d.parent: d for d in self.decomps if d.primary and d.op == "product" and len(d.factors) == 2}
        for d in [d for d in self.decomps if d.kind == "identity" and not d.primary and d.op == "product"]:
            a, b, r = d.parent, d.factors[0], d.factors[1]
            chain, cur = [], a
            while cur in primary and cur != b and len(chain) < 12:
                chain.append(primary[cur].factors[1])
                cur = primary[cur].factors[0]
            if cur == b and len(chain) > 1:
                self.decomps.append(Decomp(r, chain[::-1], "product", "composite",
                                           f"{r} = " + " x ".join(chain[::-1])))
                self.log.append(f"{r}: found composite of {len(chain)} metrics along {a} -> {b}")

    # ------------------------------------------------------------------ validation
    def _rhs(self, d: Decomp, filters=None) -> pd.Series:
        parts = [self.store.series(f, filters) for f in d.factors]
        if d.op == "product":
            out = parts[0]
            for p in parts[1:]:
                out = out * p
            return out
        if d.op == "ratio":
            return parts[0] / parts[1]
        signs = d.signs or [1] * len(parts)
        return sum(s * p for s, p in zip(signs, parts))

    def validate_decomp(self, d: Decomp, tol: float = 1e-6) -> Decomp:
        s = self.store
        segs = [None] + [{dim: v} for dim in s.dims(d.parent) for v in s.values(dim)]
        worst, n = 0.0, 0
        for f in segs:
            try:
                lhs, rhs = s.series(d.parent, f), self._rhs(d, f)
            except NotApplicable:
                continue
            ok = lhs.notna() & rhs.notna() & (lhs.abs() > 0)
            if not ok.any():
                continue
            err = ((lhs[ok] - rhs[ok]).abs() / lhs[ok].abs()).max()
            worst, n = max(worst, float(err)), n + int(ok.sum())
        d.residual, d.checked = worst, n
        d.status = "valid" if n and worst <= tol else ("invalid" if n else "unvalidated")
        return d

    def breakdown_check(self, metric: str, dim: str) -> dict:
        """Sums must add up over a dimension; ratios are checked through numerator and denominator."""
        s = self.store
        e = s.expr(metric)
        cols = [metric] if e.kind == "base" else [m for m in e.metrics]
        worst = 0.0
        for m in cols:
            if s.expr(m).kind != "base":
                continue
            total = s.series(m)
            parts = s.series(m, by=dim).sum(axis=1)
            ok = total > 0
            if ok.any():
                worst = max(worst, float(((total[ok] - parts[ok]).abs() / total[ok]).max()))
        additive = e.kind == "base"
        return dict(metric=metric, dim=dim, residual=worst, status="valid" if worst <= 1e-6 else "invalid",
                    reasoning="additive (segments sum to total)" if additive else
                    "ratio: change splits into mix effect + rate effect per segment")

    def validate(self):
        for d in self.decomps:
            self.validate_decomp(d)
        bad = [d for d in self.decomps if d.status == "invalid"]
        for d in bad:
            self.log.append(f"INVALID: {d.text} (max residual {d.residual:.2%}) - excluded from reasoning")

    # ------------------------------------------------------------------ edges
    def _edges(self):
        s = self.store
        self.edges = []
        for d in self.decomps:
            if d.kind == "definition":
                for i, f in enumerate(d.factors):
                    self.edges.append(Edge(d.parent, f, "definition", "numerator" if i == 0 else "denominator",
                                           d.status, dict(text=d.text)))
            elif d.kind == "identity" and d.primary:
                for f in d.factors:
                    self.edges.append(Edge(d.parent, f, "formula", d.text, d.status, dict(op=d.op)))
            elif d.kind == "composite":
                for f in d.factors:
                    self.edges.append(Edge(d.parent, f, "composite", d.text, d.status))
        for m in self.metrics:
            for dim in s.dims(m):
                bc = self.breakdown_check(m, dim)
                self.edges.append(Edge(m, f"{m} by {dim}", "breakdown", bc["reasoning"], bc["status"],
                                       dict(dim=dim, values=s.values(dim), residual=bc["residual"])))
        self.causes = []
        for c in s.catalog.get("causes") or []:
            shared = set(s.dims(c["from"])) & set(s.dims(c["to"]))
            ok = set(c.get("join", [])) <= shared
            self.causes.append(dict(c, status="hypothesis" if ok else "invalid (no shared dimension)"))
            self.edges.append(Edge(c["to"], c["from"], "possible_cause", c["reason"],
                                   "hypothesis" if ok else "invalid", dict(join=c.get("join", []))))
        for comp, mets in (s.catalog.get("event_components") or {}).items():
            for m in mets:
                if m in self.metrics:
                    self.edges.append(Edge(m, f"change:{comp}", "possible_cause",
                                           f"a {comp} change can move {m}", "hypothesis", dict(component=comp)))

    # ------------------------------------------------------------------ navigation
    def primary(self, metric: str) -> Decomp | None:
        return next((d for d in self.decomps if d.parent == metric and d.primary and d.status == "valid"), None)

    def composite(self, metric: str) -> Decomp | None:
        return next((d for d in self.decomps if d.parent == metric and d.kind == "composite" and d.status == "valid"), None)

    def decomposition(self, metric: str) -> Decomp | None:
        """What to split a change of `metric` into: primary identity, else a validated composite."""
        return self.primary(metric) or self.composite(metric)

    def causes_of(self, metric: str) -> list[dict]:
        return [c for c in self.causes if c["to"] == metric and c["status"] == "hypothesis"]

    def components_for(self, metric: str) -> list[str]:
        comps = self.store.catalog.get("event_components") or {}
        return [c for c, ms in comps.items() if metric in ms]

    def parents(self, metric: str, ratios: bool = False) -> list[str]:
        """Metrics computed from `metric` via formula/composite edges; with ratios=True also every ratio that
        uses it (definition edges). The monitor uses ratios=True to recognise symptoms of an open incident."""
        types = ("formula", "composite", "definition") if ratios else ("formula", "composite")
        return [e.src for e in self.edges if e.dst == metric and e.type in types]

    def roots(self) -> list[str]:
        return [m for m, d in self.metrics.items() if d.get("tier") in ("outcome", "kpi") and not
                any(e.dst == m and e.type == "formula" for e in self.edges)]

    def tree(self, root: str, indent: str = "", seen=None) -> list[str]:
        seen = seen or set()
        m = self.metrics[root]
        dims = self.store.dims(root)
        line = f"{indent}{root}  [{m.get('unit', '')}]" + (f"  by {', '.join(dims)}" if dims else "")
        out = [line]
        if root in seen:
            return out
        seen.add(root)
        d = self.decomposition(root)
        if d:
            rhs = d.text.split(' = ', 1)[1] if d.op == "sum" else ' x '.join(d.factors)
            out.append(f"{indent}  = {rhs}   ({d.kind}, {d.status})")
            for f in d.factors:
                out += self.tree(f, indent + "    ", seen)
        for c in self.causes_of(root):
            out.append(f"{indent}  ? possible cause: {c['from']} (join {', '.join(c['join'])}): {c['reason']}")
        comps = self.components_for(root)
        if comps:
            out.append(f"{indent}  ? change log components: {', '.join(comps)}")
        return out

    def to_json(self) -> dict:
        s = self.store
        nodes = [dict(name=n, description=m.get("description", ""), unit=m.get("unit", ""),
                      tier=m.get("tier", ""), kind=s.expr(n).kind, expr=m["expr"], dims=s.dims(n),
                      sources=s.sources(n), added_by=m.get("added_by", "catalog"))
                 for n, m in self.metrics.items()]
        return dict(nodes=nodes, edges=[asdict(e) for e in self.edges],
                    decomps=[asdict(d) for d in self.decomps], causes=self.causes, log=self.log,
                    roots=self.roots())
