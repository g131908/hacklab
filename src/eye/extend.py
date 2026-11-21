"""Extension agent: a contributor hands over a new metric (name, description, expression). The agent works
out where it belongs, validates it on the data and adds it to monitoring. The contributor never says where
the metric goes in the graph.

python src/eye/extend.py config/eye/new_metrics/zero_result_rate.yaml          incorporate
python src/eye/extend.py config/eye/new_metrics/zero_result_rate.yaml --dry-run  report only

Steps (each one logged with its evidence):
 1 parse + schema check      unknown columns/metrics are rejected with a reason
 2 normalise                 sum(col) of an existing base metric -> that metric; unknown raw column -> new base metric
 3 formula connections       ratio a/b -> definition + identity a = b x new; product/sum -> identity
 4 composite discovery       a/b where a and b sit on the same formula chain -> product of the rates between them
 5 duplicate check           numerically identical to an existing metric -> flagged as alias, not added twice
 6 dimensions                inferred from the tables it reads; additivity validated per dimension
 7 data-driven links         correlation of deviations vs last week (with lead/lag) against existing metrics,
                             per shared segment -> AMBIGUOUS possible_cause edges (direction from lead/lag)
 8 change-log components     description keywords (and Claude, if a key is set) -> which change types can move it
 9 incorporate               written to config/eye/added_metrics.yaml, monitored from the next tick
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye.graph import Decomp, MetricGraph  # noqa: E402
from eye.store import ADDED, Expr, NotApplicable, Store  # noqa: E402

COMPONENT_WORDS = {
    "search": ["search", "query", "queries", "result", "ranking", "relevance"],
    "payments": ["payment", "pay", "card", "authori", "psp"],
    "checkout": ["checkout", "basket"],
    "recommendations": ["recommend", "for you", "feed", "personali"],
    "marketing": ["campaign", "marketing", "traffic", "acquisition"],
    "wholesale": ["stock", "inventory", "supplier", "purchase order", "wholesale", "delivery"],
    "app": ["app", "crash", "release"],
    "refunds": ["refund", "refunds", "returned orders", "returns"],
}


def _formula_family(g: MetricGraph, metric: str) -> set[str]:
    """Metrics arithmetically tied to `metric`: its inputs (numerator/denominator), composite factors, and
    everything computed from any of those (their formula ancestors). Siblings that merely share an input
    (zero_result_rate and search_ctr both divide by searches) stay candidates. Correlation with family
    members adds no information, so they are not proposed as links."""
    core = {metric}
    core |= {e.dst for e in g.edges if e.src == metric and e.type in ("definition", "composite")}
    fam = set(core)
    for m in core:
        fam |= _ancestors(g, m)
    return fam


def _ancestors(g: MetricGraph, metric: str) -> set[str]:
    out, frontier = set(), [metric]
    while frontier:
        m = frontier.pop()
        for p in g.parents(m):
            if p not in out:
                out.add(p)
                frontier.append(p)
    return out
WEEK = 168


class ExtensionReport:
    def __init__(self, spec: dict):
        self.spec = spec
        self.steps: list[dict] = []
        self.accepted = False
        self.metrics: list[dict] = []
        self.causes: list[dict] = []
        self.components: dict[str, list[str]] = {}

    def step(self, name: str, status: str, text: str, evidence: dict | None = None):
        self.steps.append(dict(step=name, status=status, text=text, evidence=evidence or {}))

    def to_dict(self) -> dict:
        return dict(spec=self.spec, accepted=self.accepted, steps=self.steps, metrics=self.metrics,
                    causes=self.causes, components=self.components,
                    at=datetime.now().strftime("%Y-%m-%d %H:%M"))


def _deviation(s: pd.Series) -> pd.Series:
    """log change vs the same hour last week (removes daily and weekly seasonality)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(s / s.shift(WEEK))
    return d.replace([np.inf, -np.inf], np.nan)


def extend(store: Store, spec: dict, use_llm: bool = True) -> tuple[ExtensionReport, Store]:
    rep = ExtensionReport(spec)
    name = spec["name"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        rep.step("parse", "rejected", f"invalid metric name {name!r} (use snake_case)")
        return rep, store
    if name in store.metrics:
        rep.step("parse", "rejected", f"a metric named {name} already exists")
        return rep, store

    # 1 parse + schema check -------------------------------------------------------------
    try:
        e = Expr.parse(spec["expr"])
    except (SyntaxError, ValueError) as exc:
        rep.step("parse", "rejected", f"expression does not parse: {exc}")
        return rep, store
    known_cols = {c for t in store.tables.values() for c in t.columns}
    unknown_cols = [c for c in e.columns if c not in known_cols]
    unknown_mets = [m for m in e.metrics if m not in store.metrics]
    if unknown_cols or unknown_mets:
        rep.step("parse", "rejected", "unknown references: "
                 + ", ".join([f"column {c}" for c in unknown_cols] + [f"metric {m}" for m in unknown_mets]),
                 dict(available_columns=sorted(known_cols)))
        return rep, store
    rep.step("parse", "ok", f"parsed '{spec['expr']}': columns {e.columns or '-'}, metrics {e.metrics or '-'}")

    # 2 normalise sum(col) to metric names --------------------------------------------------
    base_of = {store.expr(m).columns[0]: m for m in store.metrics if store.expr(m).kind == "base"}
    new_bases = []
    text = spec["expr"]
    if e.kind != "base":
        for col in e.columns:
            if col in base_of:
                text = re.sub(rf"sum\(\s*{col}\s*\)", base_of[col], text)
                rep.step("normalise", "ok", f"sum({col}) is the existing metric '{base_of[col]}'")
            else:
                src = store.column_source(col) or "unknown source"
                new_bases.append(dict(name=col, description=f"Raw column {col} (auto-added for {name})",
                                      expr=f"sum({col})", unit="count", added_by="extension agent"))
                text = re.sub(rf"sum\(\s*{col}\s*\)", col, text)
                rep.step("normalise", "ok", f"sum({col}) has no metric yet: added base metric '{col}' from {src}")
    new = dict(name=name, description=spec.get("description", ""), expr=text, unit=spec.get("unit", "ratio"),
               added_by="extension agent")
    if spec.get("tier"):
        new["tier"] = spec["tier"]
    trial = store.with_metrics(new_bases + [new])
    ne = trial.expr(name)

    # 3+4 formula connections and composites (by building the graph with the new metric) ----------
    g = MetricGraph.build(trial)
    mine = [d for d in g.decomps if name in [d.parent] + d.factors]
    for d in mine:
        rep.step("formula", "ok" if d.status == "valid" else "warning",
                 f"{d.kind}: {d.text} -> {d.status}"
                 + (f" (residual {d.residual:.1e} over {d.checked:,} points)" if d.residual is not None else ""),
                 dict(kind=d.kind, primary=d.primary, residual=d.residual))
    comp = g.composite(name)
    if comp:
        rep.step("composite", "ok", f"{name} sits on the formula chain: it equals {' x '.join(comp.factors)} "
                                    f"(validated, residual {comp.residual:.1e})")
    parents = g.parents(name)
    rep.step("place", "ok", f"place in the graph: component of {', '.join(parents) or '(none)'}; "
                            f"decomposes into {', '.join((g.decomposition(name) or Decomp(name, [], '', '', '')).factors) or 'raw data'}")

    # 5 duplicate check ---------------------------------------------------------------------
    s_new = trial.series(name)
    for other in store.metrics:
        s_o = store.series(other)
        ok = s_new.notna() & s_o.notna() & (s_o.abs() > 0)
        if ok.sum() > 24 and float(((s_new[ok] - s_o[ok]).abs() / s_o[ok].abs()).max()) < 1e-9:
            rep.step("duplicate", "rejected", f"{name} is numerically identical to existing metric '{other}' "
                                              "on all hours: register it as an alias instead")
            return rep, store
    rep.step("duplicate", "ok", "no existing metric has the same values")

    # 6 dimensions -------------------------------------------------------------------------
    dims = trial.dims(name)
    checks = [g.breakdown_check(name, d) for d in dims]
    rep.step("dimensions", "ok" if all(c["status"] == "valid" for c in checks) else "warning",
             f"breakable by {', '.join(dims) or 'nothing'}; " + "; ".join(f"{c['dim']}: {c['status']}, {c['reasoning']}"
                                                                          for c in checks),
             dict(dims=dims, checks=checks))

    # 7 data-driven links (ambiguous) --------------------------------------------------------
    cands = []
    family = _formula_family(g, name)
    for other in store.metrics:
        if other in family:   # already explained by arithmetic
            continue
        shared = [d for d in dims if d in store.dims(other)]
        segs = [None] + [{d: v} for d in shared for v in store.values(d)]
        best = None
        for seg in segs:
            try:
                a, b = _deviation(trial.series(name, seg)), _deviation(store.series(other, seg))
            except NotApplicable:
                continue
            for lag in range(-3, 4):
                x, y = a, b.shift(lag)
                ok = x.notna() & y.notna()
                if ok.sum() < 48 or x[ok].std() == 0 or y[ok].std() == 0:
                    continue
                r = float(np.corrcoef(x[ok], y[ok])[0, 1])
                if best is None or abs(r) > abs(best["r"]):
                    best = dict(other=other, segment=seg or "total", lag=lag, r=r, n=int(ok.sum()))
        if best and abs(best["r"]) >= 0.6:
            cands.append(best)
    cands.sort(key=lambda c: -abs(c["r"]))
    # a candidate whose movement is already a downstream (formula) consequence of a stronger candidate
    # is a symptom of that candidate, not a separate link
    kept, dropped = [], []
    for c in cands:
        stronger = next((k for k in kept if c["other"] in _ancestors(g, k["other"])), None)
        (dropped if stronger else kept).append((c, stronger))
    kept = [c for c, _ in kept]
    for c, s in dropped:
        rep.step("links", "ok", f"{c['other']} (r={c['r']:+.2f}) dropped: downstream of {s['other']} via formula, "
                                "so it is a symptom, not a separate link")
    for c in kept[:3]:
        if c["lag"] > 0:
            cause, effect, direction = name, c["other"], f"{name} moves {c['lag']}h before {c['other']}"
        elif c["lag"] < 0:
            cause, effect, direction = c["other"], name, f"{name} moves {-c['lag']}h after {c['other']}"
        else:
            cause, effect, direction = name, c["other"], "same hour: direction unknown"
        rep.causes.append(dict(**{"from": cause, "to": effect},
                               join=list(c["segment"]) if isinstance(c["segment"], dict) else [],
                               reason=f"data: deviations correlate r={c['r']:+.2f} in {c['segment']} ({direction})",
                               status="ambiguous", origin="correlation"))
        rep.step("links", "ambiguous", f"{c['other']}: r={c['r']:+.2f} in segment {c['segment']} over {c['n']} hours "
                                       f"({direction}) -> possible_cause {cause} -> {effect}; correlation only, "
                                       "tested again during investigations", c)
    if not cands:
        rep.step("links", "ok", "no strong data-driven link to existing metrics (|r| < 0.6)")

    # 8 change-log components ----------------------------------------------------------------
    words = (spec.get("description", "") + " " + name.replace("_", " ")).lower()
    comps = [c for c, ws in COMPONENT_WORDS.items()
             if any(re.search(rf"\b{re.escape(w)}", words) for w in ws)]
    llm_note = ""
    if use_llm:
        try:
            from eye import llm
            extra = llm.propose_components(spec, sorted(store.catalog.get("event_components", {})), cands[:4])
            if extra:
                comps = sorted(set(comps) | set(extra["components"]))
                llm_note = f"; Claude: {extra['why']}"
        except Exception as exc:  # no key, no network: keywords only
            llm_note = f" (Claude not used: {type(exc).__name__})"
    rep.components = {c: [name] for c in comps if c in (store.catalog.get("event_components") or {}) or c == "refunds"}
    rep.step("components", "ok" if comps else "warning",
             (f"change types that can move {name}: {', '.join(comps)}" if comps else
              "no change-log component recognised from the description") + llm_note)

    # 9 incorporate -----------------------------------------------------------------------
    rep.metrics = new_bases + [new]
    rep.accepted = True
    final = store.with_metrics(rep.metrics, [c for c in rep.causes], rep.components)
    rep.step("monitor", "ok", f"{name} enters monitoring at the next tick (total + {', '.join(dims)} segments)")
    return rep, final


def save(rep: ExtensionReport, path: Path = ADDED):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    data.setdefault("metrics", []).extend(rep.metrics)
    data.setdefault("causes", []).extend(rep.causes)
    for comp, mets in rep.components.items():
        data.setdefault("event_components", {}).setdefault(comp, [])
        data["event_components"][comp] += [m for m in mets if m not in data["event_components"][comp]]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log = path.parent / "extension_log.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rep.to_dict(), default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="YAML file with name, description, expr (optional unit)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    rep, _ = extend(Store.load(), spec, use_llm=not args.no_llm)
    print(f"[DEMO DATA] extension agent: new metric '{spec['name']}' = {spec['expr']}")
    for s in rep.steps:
        print(f"  {s['step']:<11} {s['status']:<9} {s['text']}")
    if rep.accepted and not args.dry_run:
        save(rep)
        print(f"\nincorporated -> {ADDED.relative_to(ADDED.parents[2])} (+ extension_log.jsonl)")
    elif rep.accepted:
        print("\n(dry run: nothing written)")
    else:
        print("\nnot incorporated")


if __name__ == "__main__":
    main()
