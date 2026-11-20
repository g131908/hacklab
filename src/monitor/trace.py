"""Agent trace: one record per pipeline step for each fired detection, so a human (and the
dashboard) can see exactly how the agent got from a raw signal to an alert or a suppression.

Step: {key, label, status, ms, summary, detail}
status: pass (step went through) | stop (pipeline ended here, e.g. suppressed) |
        warn (ran with a gap) | info | skip (not run)
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from .detectors import Detection
from .triage import SIGNALS, action_for

ACTION_TEXT = {"log": "log only (daily digest)", "single": "single specialist",
               "panel": "specialist panel, notify now", "page": "page a human now"}
PIPELINE = ["detect", "selfcheck", "triage", "dedupe", "route", "specialists", "synthesize", "memory", "gate", "log"]
LABELS = {"detect": "Detector", "selfcheck": "Self-checks", "dedupe": "Dedupe & cooldown",
          "triage": "Triage score", "route": "Routing", "specialists": "Specialist answers",
          "synthesize": "Synthesis", "memory": "Decision memory", "gate": "Human gate", "log": "Log & state"}


class Trace:
    def __init__(self):
        self.steps: list[dict] = []

    @contextmanager
    def timed(self):
        box = {}
        t0 = time.perf_counter()
        yield box
        box["ms"] = round((time.perf_counter() - t0) * 1000, 2)

    def add(self, key: str, status: str, summary: str, detail: dict | None = None, ms: float | None = None):
        self.steps.append(dict(key=key, label=LABELS[key], status=status, ms=ms, summary=summary,
                               detail=detail or {}))

    def skip_rest(self, reason: str):
        done = {s["key"] for s in self.steps}
        for k in PIPELINE:
            if k not in done:
                self.add(k, "skip", reason)

    def to_list(self) -> list[dict]:
        """Steps in pipeline order (triage is computed before dedupe, which needs the severity)."""
        return sorted(self.steps, key=lambda s: PIPELINE.index(s["key"]))


def detect_step(tr: Trace, det: Detection, run_ms: float, n_checks: int):
    tr.add("detect", "pass",
           f"{det.signal} {SIGNALS.get(det.signal, {}).get('name', '')} fired on {det.entity}: {det.title}",
           dict(signal=det.signal, entity=det.entity, entity_type=det.entity_type,
                window=[str(det.window_start), str(det.window_end)], direction=det.direction,
                evidence=det.evidence, metrics={k: _num(v) for k, v in det.metrics.items()},
                money_at_risk_day=_num(det.money_at_risk_day), money_note=det.money_note,
                detectors_run=n_checks),
           ms=run_ms)


def check_status(name: str, value: str) -> str:
    v = value.upper()
    if v.startswith("OK") or v == "NONE":
        return "pass"
    if v.startswith("MATCH") or v.startswith("INSUFFICIENT") or v.startswith("NO BASELINE") or v.startswith("STALE"):
        return "stop"
    if "NOT AVAILABLE" in v:
        return "warn"
    if v.startswith("N/A"):
        return "skip"
    return "info"


def selfcheck_step(tr: Trace, res, ms: float):
    checks = [dict(name=k, result=v, status=check_status(k, v)) for k, v in res.checks.items()]
    if res.verdict == "alert":
        gaps = [c["name"] for c in checks if c["status"] == "warn"]
        tr.add("selfcheck", "warn" if gaps else "pass",
               "real: all checks passed" + (f" (could not run: {', '.join(gaps)})" if gaps else ""),
               dict(verdict=res.verdict, checks=checks), ms)
    else:
        tr.add("selfcheck", "stop", f"{res.verdict}: {res.reason} - no alert, nobody paged",
               dict(verdict=res.verdict, reason=res.reason, checks=checks), ms)


def dedupe_step(tr: Trace, action: str, rec: dict, ms: float):
    text = {"new": f"new anomaly {rec['anomaly_id']}",
            "realert": f"{rec['anomaly_id']} already open, severity rose -> re-alert",
            "cooldown": f"{rec['anomaly_id']} already open and in cooldown -> silent",
            "update": f"{rec['anomaly_id']} already open -> last_seen updated, no new alert"}[action]
    tr.add("dedupe", "pass" if action in ("new", "realert") else "stop", text,
           dict(action=action, anomaly_id=rec["anomaly_id"], first_seen=rec["first_seen"],
                alerts=rec["alerts"]), ms)


def triage_step(tr: Trace, sev: int, factors: list[str], ms: float):
    act = action_for(sev)
    tr.add("triage", "pass", f"SEV {sev} -> {ACTION_TEXT[act]}",
           dict(severity=sev, factors=factors, action=act), ms)


def route_step(tr: Trace, router, picks: list[tuple[str, str]], sev: int):
    if not picks:
        tr.add("route", "skip", "SEV 1-2: no specialists, digest only")
        return
    tr.add("route", "pass",
           f"{len(picks)} specialist{'s' if len(picks) > 1 else ''} via {router.mode}"
           + (" in parallel" if len(picks) > 1 else "") + ": " + ", ".join(p[0] for p in picks),
           dict(mode=router.mode, depth=1, max_depth=router.max_depth, parallel=len(picks) > 1,
                specialists=[dict(skill=s, question=q) for s, q in picks]))


def specialists_step(tr: Trace, outs: list[dict], ms: float):
    if not outs:
        tr.add("specialists", "skip", "no specialists asked")
        return
    stances = {o.get("stance") for o in outs if o.get("diagnosis")}
    failed = [o["skill"] for o in outs if "llm failed" in str(o.get("source", ""))]
    summary = "; ".join(f"{o['skill']}: {(o.get('stance') or '?').replace('_', ' ')}" for o in outs)
    if failed:
        summary += f" (LLM failed for {', '.join(failed)}, used rules)"
    tr.add("specialists", "warn" if failed or len(stances) > 1 else "pass", summary,
           dict(outputs=outs), ms)


def synth_step(tr: Trace, al: dict, why: str, ms: float):
    tr.add("synthesize", "warn" if al["disagreement"] else "pass",
           ("specialists disagree -> decision point for the human; " if al["disagreement"] else "")
           + f"recommended option {al['recommended']}",
           dict(headline=al["headline"], why=why, disagreement=al["disagreement"],
                options=al["options"], recommended=al["recommended"]), ms)


def memory_step(tr: Trace, memory: dict | None, analysis_rec: str, ms: float):
    if memory is None:
        tr.add("memory", "skip", "no decision needed at this severity")
        return
    basis = [dict(anomaly_id=b["anomaly_id"], decision=b["decision"], recommended=b["recommended"],
                  decided_by=b["decided_by"], decided_at=b["decided_at"], entity=b["entity"])
             for b in memory.get("basis", [])]
    status = "info" if memory.get("applied") else "pass"
    summary = memory["text"] if not memory.get("applied") else \
        f"learned: {memory['text']}"
    tr.add("memory", status, summary,
           dict(applied=bool(memory.get("applied")), scope=memory.get("scope"), analysis_recommended=analysis_rec,
                final_recommended=memory["recommended"], basis=basis), ms)


def gate_step(tr: Trace, sev: int, kind: str, decision: str | None, decided_by: str, ms: float):
    if kind == "digest":
        tr.add("gate", "skip", "SEV 1-2: no interrupt, goes to the daily digest")
    elif decision is None and sev >= 4:
        tr.add("gate", "warn", "awaiting human decision: pending in decisions.md (python src/decide.py)",
               dict(decision="pending", decided_by=decided_by, executed=False), ms)
    elif decision is None:
        tr.add("gate", "info", "SEV 3: recommendation staged, no decision required now",
               dict(staged=True), ms)
    else:
        tr.add("gate", "pass", f"decision {decision} by {decided_by} -> staged, NOT executed (Tier 2)",
               dict(decision=decision, decided_by=decided_by, executed=False), ms)


def log_step(tr: Trace, rec: dict | None, extra: str = ""):
    tr.add("log", "pass", (f"written to alerts.jsonl, open-anomalies.json ({rec['anomaly_id']})" if rec
                           else "written to alerts.jsonl") + extra)


def _num(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)
    except (TypeError, ValueError):
        return v
