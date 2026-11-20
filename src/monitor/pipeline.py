"""One monitoring run: WATCH -> SELF-CHECK -> TRIAGE -> ROUTE -> SYNTHESIZE -> GATE -> LOG.
Shared by replay.py (demo) and a future live loop. Every fired detection carries an agent
trace (monitor/trace.py) that is written to the log alongside the alert or suppression."""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from . import alert as alert_mod
from . import trace as tr_mod
from .detectors import run_detectors
from .learning import DecisionMemory
from .loader import MarketData
from .routing import Router, specialists_for
from .selfcheck import check, dedupe, freshness
from .state import State
from .triage import SIGNALS, action_for, score

# notify(alert_dict) shows the alert; decide(alert_dict) -> "A" | "B", only for SEV >= 4 (the GATE).
# A decider may carry a `mode` attribute ("terminal", "auto") that the trace records as decided_by.
Notifier = Callable[[dict], None]
Decider = Callable[[dict], str]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def run_once(data: MarketData, now: pd.Timestamp, cfg: dict, state: State,
             notify: Notifier, decide: Decider, router: Router | None = None,
             wall: pd.Timestamp | None = None) -> list[dict]:
    """Returns events for display: alert | suppressed | resolved | pipeline | error | killed.
    `now` is the data hour to evaluate; `wall` (live mode) is the real clock for the freshness check."""
    if state.killed():
        state.log_run(now, 0, 0, 0, 0, note="killed (KILL file present)")
        return [dict(type="killed", ts=now)]

    events: list[dict] = []
    fresh = freshness(data, wall if wall is not None else now, cfg)
    if not fresh[0]:
        ev = dict(type="pipeline", ts=now, severity=3, signal="data_pipeline", text=fresh[1])
        state.log_event(dict(event="alert", **ev))
        state.log_run(now, 0, 0, 1, 0, note="stale data")
        return [ev]

    t0 = time.perf_counter()
    try:
        dets = run_detectors(data, now, cfg)
    except Exception as exc:  # a silent monitor is worse than a noisy one
        ev = dict(type="error", ts=now, severity=3, signal="monitor_error", text=repr(exc))
        state.log_event(dict(event="alert", **ev))
        state.log_run(now, 0, 0, 1, 0, errors=1, note=repr(exc))
        return [ev]
    detect_ms = _ms(t0)

    # related signals: an overspend that threatens the hard cap is one incident, not two alerts.
    # S1 (pacing up) on an account where S2 (cap risk) fired this hour becomes evidence of the S2 alert.
    s2 = {d.entity: d for d in dets if d.signal == "S2" and d.fired}
    merged = []
    for d in dets:
        if d.signal == "S1" and d.fired and d.direction == "up" and d.entity in s2:
            s2[d.entity].evidence.append(f"merged S1: {d.evidence[0] if d.evidence else d.title}")
            merged.append(d)
    dets = [d for d in dets if d not in merged]

    router = router or Router(cfg)
    mem = DecisionMemory(state.folder, cfg)
    fired = alerted = suppressed = 0
    for det in dets:
        t0 = time.perf_counter()
        res = check(det, fresh, cfg)
        check_ms = _ms(t0)
        if not det.fired:
            if res.verdict == "alert":  # measurable and in range -> auto-resolve counter
                rec = state.mark_in_range(det.signal, det.entity, now, cfg["auto_resolve_checks"])
                if rec:
                    events.append(dict(type="resolved", ts=now, anomaly_id=rec["anomaly_id"]))
                    state.log_event(dict(event="resolved", ts=now, anomaly_id=rec["anomaly_id"],
                                         signal=det.signal, entity=det.entity))
            continue

        fired += 1
        tr = tr_mod.Trace()
        tr_mod.detect_step(tr, det, detect_ms, len(dets))
        tr_mod.selfcheck_step(tr, res, check_ms)
        if res.verdict != "alert":
            suppressed += 1
            tr_mod.log_step(tr, None, " as suppressed (counts toward the false-alarm check)")
            tr.skip_rest("not run: self-check stopped the pipeline")
            ev = dict(type="suppressed", ts=now, signal=det.signal, entity=det.entity,
                      reason=res.reason, title=det.title, checks=res.checks)
            events.append(ev)
            state.log_event(dict(event="suppressed", ts=now, signal=det.signal, entity=det.entity,
                                 reason=res.reason, title=det.title, trace=tr.to_list()))
            continue

        t0 = time.perf_counter()
        sev, factors = score(det, cfg)
        triage_ms = _ms(t0)
        t0 = time.perf_counter()
        action, rec = dedupe(det, sev, state, now, cfg)
        tr_mod.dedupe_step(tr, action, rec, _ms(t0))
        if action in ("cooldown", "update"):
            suppressed += 1
            continue  # same anomaly already alerted; not logged again (keeps the log one-per-alert)
        tr_mod.triage_step(tr, sev, factors, triage_ms)

        if action_for(sev) == "page":  # SEV 5: page with the facts before the diagnosis is finished
            notify(dict(type="page", severity=sev, anomaly_id=rec["anomaly_id"],
                        text=(f"[{data.label}] " if data.label else "") +
                        f"PAGE [SEV 5] {SIGNALS[det.signal]['name']}: {det.entity} - {det.title}. "
                        f"Facts: {'; '.join(det.evidence)}. Diagnosis running..."))
        picks = specialists_for(det.signal, sev) if action_for(sev) != "log" else []
        tr_mod.route_step(tr, router, picks, sev)
        t0 = time.perf_counter()
        specialists = router.route(det, sev, rec["anomaly_id"], data) if picks else []
        tr_mod.specialists_step(tr, specialists, _ms(t0))

        t0 = time.perf_counter()
        al = alert_mod.build(det, sev, factors, res.checks, specialists,
                             label=data.label, realert=(action == "realert"))
        tr_mod.synth_step(tr, al, alert_mod._why(specialists), _ms(t0))

        # decision memory: past HUMAN decisions on this signal (+ account) may change the recommendation
        t0 = time.perf_counter()
        memory = None
        if action_for(sev) != "log":
            # asked before this alert's own decision exists, so it only sees earlier decisions
            memory = mem.advise(det.signal, det.entity, al["recommended"])
            if memory["basis"]:
                al = alert_mod.build(det, sev, factors, res.checks, specialists, label=data.label,
                                     realert=(action == "realert"), memory=memory)
        tr_mod.memory_step(tr, memory, al.get("analysis_recommended", al["recommended"]), _ms(t0))

        decision = None
        kind = "digest" if action_for(sev) == "log" else "alert"
        notify(dict(type=kind, severity=sev, **al))
        t0 = time.perf_counter()
        if kind == "alert":
            alerted += 1
            if sev >= 4:
                decision = decide(al)  # None = no human available now -> pending approval
                if decision in ("A", "B"):
                    rec["decision"] = decision
                    state.log_decision(rec["anomaly_id"], decision, al["options"][decision],
                                       getattr(decide, "mode", "human"), data.label)
                    mem.record(anomaly_id=rec["anomaly_id"], signal=det.signal, entity=det.entity,
                               severity=sev, recommended=al["recommended"], decision=decision,
                               decided_by=getattr(decide, "mode", "human"), data_ts=str(now),
                               learned=al.get("learned", False), data_label=data.label)
                else:
                    decision = None
                    rec["decision"] = "pending"
                    rec["options"] = al["options"]
                    rec["recommended"] = al["recommended"]
                    state.log_pending(rec["anomaly_id"], al["headline"], al["options"], al["recommended"],
                                      ", ".join(cfg["notify"]["approvers"]), data.label)
        tr_mod.gate_step(tr, sev, kind, decision, getattr(decide, "mode", "human"), _ms(t0))
        tr_mod.log_step(tr, rec, ", decisions.md" if decision else "")

        events.append(dict(type=kind, ts=now, severity=sev, anomaly_id=rec["anomaly_id"],
                           text=al["text"], decision=decision, recommended=al["recommended"]))
        state.log_event(dict(event=kind, ts=now, anomaly_id=rec["anomaly_id"], signal=det.signal,
                             entity=det.entity, severity=sev, realert=(action == "realert"),
                             money_at_risk_day=det.money_at_risk_day, decision=decision,
                             specialists=[dict(skill=s["skill"], source=s.get("source"),
                                               stance=s.get("stance"), diagnosis=s.get("diagnosis"))
                                          for s in specialists],
                             disagreement=al["disagreement"], headline=al["headline"],
                             text=al["text"], options=al["options"], recommended=al["recommended"],
                             learned=al.get("learned", False), trace=tr.to_list()))

    state.save()
    state.log_run(now, len(dets), fired, alerted, suppressed)
    return events
