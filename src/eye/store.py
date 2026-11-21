"""Data access + metric evaluation. Every metric is evaluated from raw sums (ratios are ratios of sums,
never averages of ratios), for any filter and time window, so all reasoning goes back to real rows.

The store can be "clocked": `asof(ts)` returns a view that only sees data published up to that hour,
with the completeness that data had at the time (used by the live loop and the demo)."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "config" / "eye" / "metrics.yaml"
ADDED = ROOT / "config" / "eye" / "added_metrics.yaml"     # written by the extension agent
DATA = ROOT / "data" / "eye"
H = pd.Timedelta(hours=1)


class NotApplicable(Exception):
    """A filter uses a dimension the metric does not have (e.g. in_stock_rate by country)."""


# ----------------------------------------------------------------------------- expressions
@dataclass
class Expr:
    text: str
    tree: ast.AST
    metrics: list[str]          # referenced metric names
    columns: list[str]          # sum(<column>) references
    kind: str                   # base | ratio | product | sum | other

    @staticmethod
    def parse(text: str) -> "Expr":
        tree = ast.parse(text, mode="eval").body
        metrics, columns = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id == "sum" and len(node.args) == 1
                        and isinstance(node.args[0], ast.Name)):
                    raise ValueError(f"only sum(<column>) calls are allowed: {text}")
                columns.append(node.args[0].id)
            elif isinstance(node, ast.Name) and not _is_call_arg(tree, node):
                if node.id != "sum":
                    metrics.append(node.id)
            elif isinstance(node, (ast.BinOp, ast.Name, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult,
                                   ast.Div, ast.UnaryOp, ast.USub, ast.Call)):
                continue
            elif not isinstance(node, (ast.expr_context,)):
                raise ValueError(f"unsupported syntax in {text!r}: {type(node).__name__}")
        if columns and not metrics and isinstance(tree, ast.Call):
            kind = "base"
        elif isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Div) and all(
                isinstance(x, ast.Name) for x in (tree.left, tree.right)):
            kind = "ratio"
        elif isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Mult):
            kind = "product"
        elif isinstance(tree, ast.BinOp) and isinstance(tree.op, (ast.Add, ast.Sub)):
            kind = "sum"
        else:
            kind = "other"
        return Expr(text, tree, sorted(set(metrics)), sorted(set(columns)), kind)


def _is_call_arg(tree, name_node) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and name_node in node.args:
            return True
        if isinstance(node, ast.Call) and node.func is name_node:
            return True
    return False


# ----------------------------------------------------------------------------- store
@dataclass
class Store:
    catalog: dict
    tables: dict[str, pd.DataFrame]
    events: pd.DataFrame
    completeness: pd.DataFrame
    now: pd.Timestamp | None = None          # clock (None = all data)
    label: str = "DEMO DATA"
    _cache: dict = field(default_factory=dict, repr=False)

    # --------------------------------------------------------------- loading
    @staticmethod
    def load(catalog_path: Path = CATALOG, data_dir: Path = DATA, added_path: Path | None = ADDED) -> "Store":
        cat = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        if added_path and added_path.exists():
            extra = yaml.safe_load(added_path.read_text(encoding="utf-8")) or {}
            cat["metrics"] += extra.get("metrics") or []
            cat["causes"] = (cat.get("causes") or []) + (extra.get("causes") or [])
            for comp, mets in (extra.get("event_components") or {}).items():
                cat.setdefault("event_components", {}).setdefault(comp, [])
                cat["event_components"][comp] += [m for m in mets if m not in cat["event_components"][comp]]
        tables = {}
        for name, t in cat["tables"].items():
            tables[name] = pd.read_csv(data_dir / t["file"], parse_dates=["ts"])
        ev = pd.read_csv(data_dir / "events.csv", parse_dates=["ts"])
        if "end" in ev:
            ev["end"] = pd.to_datetime(ev["end"])
        comp = pd.read_csv(data_dir / "completeness.csv", parse_dates=["ts"])
        return Store(cat, tables, ev, comp)

    def with_metrics(self, metrics: list[dict], causes: list[dict] | None = None,
                     components: dict | None = None) -> "Store":
        """A copy of the store whose catalog also contains `metrics` (used to test a new metric)."""
        import copy
        cat = copy.deepcopy(self.catalog)
        cat["metrics"] = cat["metrics"] + metrics
        cat["causes"] = (cat.get("causes") or []) + (causes or [])
        for comp, mets in (components or {}).items():
            cat.setdefault("event_components", {}).setdefault(comp, [])
            cat["event_components"][comp] += [m for m in mets if m not in cat["event_components"][comp]]
        return Store(cat, self.tables, self.events, self.completeness, self.now, self.label)

    def asof(self, ts) -> "Store":
        ts = pd.Timestamp(ts)
        return Store(self.catalog, {k: v[v.ts <= ts] for k, v in self.tables.items()},
                     self.events[self.events.ts <= ts], self.completeness[self.completeness.ts <= ts], ts, self.label)

    # --------------------------------------------------------------- catalog helpers
    @property
    def metrics(self) -> dict[str, dict]:
        return {m["name"]: m for m in self.catalog["metrics"]}

    def expr(self, metric: str) -> Expr:
        key = ("expr", metric)
        if key not in self._cache:
            self._cache[key] = Expr.parse(self.metrics[metric]["expr"])
        return self._cache[key]

    def column_table(self, column: str) -> str:
        for name, t in self.tables.items():
            if column in t.columns:
                return name
        raise KeyError(f"unknown column {column}")

    def column_source(self, column: str) -> str | None:
        for name, s in self.catalog["sources"].items():
            if column in s["columns"]:
                return name
        return None

    def base_columns(self, metric: str, _seen=None) -> set[str]:
        e = self.expr(metric)
        cols = set(e.columns)
        for m in e.metrics:
            cols |= self.base_columns(m)
        return cols

    def dims(self, metric: str) -> list[str]:
        """Dimensions a metric can be broken down by = dims shared by all tables it reads."""
        sets = [set(self.catalog["tables"][self.column_table(c)]["dims"]) for c in self.base_columns(metric)]
        common = set.intersection(*sets) if sets else set()
        order = ["country", "platform", "category"]
        return [d for d in order if d in common]

    def sources(self, metric: str) -> list[str]:
        return sorted({s for c in self.base_columns(metric) if (s := self.column_source(c))})

    @property
    def hours(self) -> pd.DatetimeIndex:
        t = self.tables["facts"].ts
        return pd.date_range(t.min(), t.max(), freq="h")

    # --------------------------------------------------------------- evaluation
    def column_series(self, column: str, filters: dict | None = None, by: str | None = None) -> pd.Series | pd.DataFrame:
        filters = {k: v for k, v in (filters or {}).items() if v not in (None, "*")}
        tname = self.column_table(column)
        df = self.tables[tname]
        tdims = self.catalog["tables"][tname]["dims"]
        for k, v in filters.items():
            if k not in tdims:
                raise NotApplicable(f"{column} ({tname}) has no dimension {k}")
            df = df[df[k] == v]
        if by:
            if by not in tdims:
                raise NotApplicable(f"{column} ({tname}) has no dimension {by}")
            return df.groupby(["ts", by])[column].sum().unstack(by).reindex(self.hours).fillna(0.0)
        return df.groupby("ts")[column].sum().reindex(self.hours).fillna(0.0)

    def series(self, metric: str, filters: dict | None = None, by: str | None = None):
        """Hourly series of a metric (or a DataFrame with one column per `by` value)."""
        key = ("series", metric, tuple(sorted((filters or {}).items())), by)
        if key in self._cache:
            return self._cache[key]
        e = self.expr(metric)
        out = self._eval(e.tree, filters, by)
        self._cache[key] = out
        return out

    def _eval(self, node, filters, by):
        if isinstance(node, ast.Call):
            return self.column_series(node.args[0].id, filters, by)
        if isinstance(node, ast.Name):
            return self.series(node.id, filters, by)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            return -self._eval(node.operand, filters, by)
        a, b = self._eval(node.left, filters, by), self._eval(node.right, filters, by)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / b
        return r.replace([np.inf, -np.inf], np.nan) if hasattr(r, "replace") else r

    def window_value(self, metric: str, start, end, filters: dict | None = None, by: str | None = None):
        """Metric over [start, end] (inclusive hours): ratios are recomputed from window sums."""
        e = self.expr(metric)
        return self._eval_window(e.tree, pd.Timestamp(start), pd.Timestamp(end), filters, by)

    def _eval_window(self, node, start, end, filters, by):
        if isinstance(node, ast.Call):
            s = self.column_series(node.args[0].id, filters, by)
            return s.loc[start:end].sum()
        if isinstance(node, ast.Name):
            return self._eval_window(self.expr(node.id).tree, start, end, filters, by)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            return -self._eval_window(node.operand, start, end, filters, by)
        a = self._eval_window(node.left, start, end, filters, by)
        b = self._eval_window(node.right, start, end, filters, by)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        with np.errstate(divide="ignore", invalid="ignore"):
            return a / b

    def values(self, dim: str) -> list[str]:
        for name, t in self.catalog["tables"].items():
            if dim in t["dims"]:
                return sorted(self.tables[name][dim].unique())
        return []

    # --------------------------------------------------------------- data quality
    def completeness_at(self, source: str, ts) -> float:
        c = self.completeness[(self.completeness.source == source) & (self.completeness.ts == pd.Timestamp(ts))]
        return float(c.completeness.iloc[0]) if len(c) else float("nan")

    def events_between(self, start, end) -> pd.DataFrame:
        ev = self.events
        return ev[(ev.ts >= pd.Timestamp(start)) & (ev.ts <= pd.Timestamp(end))]
