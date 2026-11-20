"""ROUTE: send an anomaly to the specialist skills that can explain it (SKILL.md section 4).

Hybrid execution per specialist:
  mode rules -> rule-based method (specialists_rules), offline and deterministic
  mode llm   -> Claude with the skill's SKILL.md (specialists_llm); on any error -> rules
  mode auto  -> llm if ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set and `anthropic` is installed, else rules

Loop prevention: max routing depth 2 (monitor -> specialist -> at most one consult); specialists
never route back to the monitor. Panels (score >= 4) run their specialists in parallel.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import specialists_llm, specialists_rules
from .detectors import Detection
from .loader import MarketData

# signal -> (specialists in priority order, question each one answers)
ROUTES = {
    "S1": [("ads", "Is this a bid/budget/auction problem or a measurement one? Kill, hold, or scale?"),
           ("attribution", "Is the extra/missing spend buying real conversions, or is it a credit shift?")],
    "S4": [("conversion-tracking", "Which tag/pixel/event broke, and since when? What's the real number?"),
           ("analytics", "Is the metric computed correctly? Is the baseline valid?")],
    "S6": [("marketing-psychology", "Which 2-3 behavioral principles plausibly explain it? What's the testable fix?"),
           ("ads", "Is the CTR loss hurting ROAS enough to act now, or can it wait?")],
    "S7": [("analytics", "Which sources/placements? How much spend is affected?")],
}


def resolve_mode(cfg: dict, override: str | None = None) -> str:
    mode = override or cfg.get("routing", {}).get("mode", "auto")
    if mode == "auto":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        try:
            import anthropic  # noqa: F401
            has_sdk = True
        except ImportError:
            has_sdk = False
        return "llm" if has_key and has_sdk else "rules"
    return mode


def specialists_for(signal: str, severity: int) -> list[tuple[str, str]]:
    route = ROUTES.get(signal, [])
    if severity <= 3:
        return route[:1]       # single specialist
    return route[:3]           # panel: 2-3 specialists


def data_slice(det: Detection, data: MarketData, hours: int = 36) -> dict:
    """The relevant slice only: this entity, the last `hours` hours and the same hours last week."""
    side, col, df = (("adv", "advertiser_id", data.advertisers) if det.entity_type == "advertiser"
                     else ("sup", "source_id", data.supply))
    start = det.ts - pd.Timedelta(hours=hours - 1)
    wk = pd.Timedelta(days=7)
    ent = df[df[col] == det.entity].drop(columns=[col])
    recent = ent[(ent.ts >= start) & (ent.ts <= det.ts)]
    last_week = ent[(ent.ts >= start - wk) & (ent.ts <= det.ts - wk)]
    return {"entity": det.entity, "side": side,
            "recent_hourly_csv": recent.to_csv(index=False),
            "same_hours_last_week_csv": last_week.to_csv(index=False)}


def anomaly_payload(det: Detection, severity: int, aid: str, data: MarketData, label: str) -> dict:
    return {"anomaly_id": aid, "data_label": label, "signal": det.signal, "entity": det.entity,
            "entity_type": det.entity_type, "detected_at": str(det.ts),
            "window": [str(det.window_start), str(det.window_end)], "direction": det.direction,
            "title": det.title, "metrics": det.metrics, "detector_evidence": det.evidence,
            "money_at_risk_per_day_estimate": det.money_at_risk_day, "money_note": det.money_note,
            "severity": severity, "data": data_slice(det, data)}


class Router:
    def __init__(self, cfg: dict, mode: str | None = None, client=None):
        self.cfg = cfg
        self.mode = resolve_mode(cfg, mode)
        self.max_depth = cfg.get("routing", {}).get("max_depth", 2)
        self.client = client
        self.llm_errors: list[str] = []
        if self.mode == "llm" and self.client is None:
            try:
                self.client = specialists_llm.make_client(cfg.get("routing", {}).get("timeout_seconds", 90))
            except Exception as exc:  # SDK missing or no credentials -> rules
                self.llm_errors.append(f"LLM client unavailable: {exc!r}")
                self.mode = "rules"

    def _one(self, skill: str, question: str, det: Detection, data: MarketData, aid: str,
             payload: dict | None) -> dict:
        t0 = time.perf_counter()
        out = self._run_one(skill, question, det, data, aid, payload)
        out["question"] = question
        out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def _run_one(self, skill: str, question: str, det: Detection, data: MarketData, aid: str,
                 payload: dict | None) -> dict:
        if self.mode == "llm":
            try:
                out = specialists_llm.run(skill, question, payload, self.cfg, self.client)
                out["anomaly_id"] = aid
                return out
            except Exception as exc:
                self.llm_errors.append(f"{skill}: {exc!r}")
                out = specialists_rules.run(skill, det, data, aid)
                out["source"] = "rules, llm failed"
                out["llm_error"] = repr(exc)[:300]
                return out
        return specialists_rules.run(skill, det, data, aid)

    def route(self, det: Detection, severity: int, aid: str, data: MarketData, depth: int = 1) -> list[dict]:
        if depth > self.max_depth:
            return [dict(skill="monitor", anomaly_id=aid, diagnosis=None, evidence=[], hypotheses=[],
                         recommended_action=None, stance="investigate", source="depth-guard",
                         missing_data=[f"routing depth {depth} > {self.max_depth}: escalated with facts only"])]
        picks = specialists_for(det.signal, severity)
        if not picks:
            return []
        payload = anomaly_payload(det, severity, aid, data, data.label) if self.mode == "llm" else None
        if len(picks) == 1:
            return [self._one(*picks[0], det, data, aid, payload)]
        with ThreadPoolExecutor(max_workers=len(picks)) as pool:
            futures = [pool.submit(self._one, s, q, det, data, aid, payload) for s, q in picks]
            return [f.result() for f in futures]
