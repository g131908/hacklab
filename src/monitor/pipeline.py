"""One monitoring run: WATCH -> SELF-CHECK -> TRIAGE -> ROUTE -> SYNTHESIZE -> GATE -> LOG.
Shared by replay.py (demo) and a future live loop."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import alert as alert_mod
from .detectors import run_detectors
from .loader import MarketData
from .routing import Router
from .selfcheck import check, dedupe, freshness
from .state import State
from .triage import SIGNALS, action_for, score

# notify(alert_dict) shows the alert; decide(alert_dict) -> "A" | "B", only for SEV >= 4 (the GATE)
Notifier = Callable[[dict], None]
Decider = Callable[[dict], str]


def run_once(data: MarketData, now: pd.Timestamp, cfg: dict, state: State,
             notify: Notifier, decide: Decider, router: Router | None = None) -> list[dict]:
    """Returns events for display: alert | suppressed | resolved | pipeline | error | killed."""
    if state.killed():
        state.log_run(now, 0, 0, 0, 0, note="killed (KILL file present)")
        return [dict(type="killed", ts=now)]

    events: list[dict] = []
    fresh = freshness(data, now, cfg)
    if not fresh[0]:
        ev = dict(type="pipeline", ts=now, severity=3, signal="data_pipeline", text=fresh[1])
        state.log_event(dict(event="alert", **ev))
        state.log_run(now, 0, 0, 1, 0, note="stale data")
        return [ev]

    try:
        dets = run_detectors(data, now, cfg)
    except Exception as exc:  # a silent monitor is worse than a noisy one
        ev = dict(type="error", ts=now, severity=3, signal="monitor_error", text=repr(exc))
        state.log_event(dict(event="alert", **ev))
        state.log_run(now, 0, 0, 1, 0, errors=1, note=repr(exc))
        return [ev]

    router = router or Router(cfg)
    fired = alerted = suppressed = 0
    for det in dets:
        res = check(det, fresh, cfg)
        if not det.fired:
            if res.verdict == "alert":  # measurable and in range -> auto-resolve counter
                rec = state.mark_in_range(det.signal, det.entity, now, cfg["auto_resolve_checks"])
                if rec:
                    events.append(dict(type="resolved", ts=now, anomaly_id=rec["anomaly_id"]))
                    state.log_event(dict(event="resolved", ts=now, anomaly_id=rec["anomaly_id"],
                                         signal=det.signal, entity=det.entity))
            continue
        fired += 1
        if res.verdict != "alert":
            suppressed += 1
            ev = dict(type="suppressed", ts=now, signal=det.signal, entity=det.entity,
                      reason=res.reason, title=det.title, checks=res.checks)
            events.append(ev)
            state.log_event(dict(event="suppressed", ts=now, signal=det.signal,
                                 entity=det.entity, reason=res.reason, title=det.title))
            continue

        sev, factors = score(det, cfg)
        action, rec = dedupe(det, sev, state, now, cfg)
        if action in ("cooldown", "update"):
            suppressed += 1
            continue
        if action_for(sev) == "page":  # SEV 5: page with the facts before the diagnosis is finished
            notify(dict(type="page", severity=sev, anomaly_id=rec["anomaly_id"],
                        text=(f"[{data.label}] " if data.label else "") +
                        f"PAGE [SEV 5] {SIGNALS[det.signal]['name']}: {det.entity} - {det.title}. "
                        f"Facts: {'; '.join(det.evidence)}. Diagnosis running..."))
        specialists = router.route(det, sev, rec["anomaly_id"], data) if action_for(sev) != "log" else []
        al = alert_mod.build(det, sev, factors, res.checks, specialists,
                             label=data.label, realert=(action == "realert"))
        decision = None
        kind = "digest" if action_for(sev) == "log" else "alert"
        notify(dict(type=kind, severity=sev, **al))
        if kind == "alert":
            alerted += 1
            if sev >= 4:
                decision = decide(al)
                state.log_decision(rec["anomaly_id"], decision, al["options"][decision],
                                   ", ".join(cfg["notify"]["approvers"]), data.label)
        events.append(dict(type=kind, ts=now, severity=sev, anomaly_id=rec["anomaly_id"],
                           text=al["text"], decision=decision, recommended=al["recommended"]))
        state.log_event(dict(event=kind, ts=now, anomaly_id=rec["anomaly_id"], signal=det.signal,
                             entity=det.entity, severity=sev, realert=(action == "realert"),
                             money_at_risk_day=det.money_at_risk_day, decision=decision,
                             specialists=[dict(skill=s["skill"], source=s.get("source"),
                                               stance=s.get("stance"), diagnosis=s.get("diagnosis"))
                                          for s in specialists],
                             disagreement=al["disagreement"], headline=al["headline"],
                             text=al["text"], options=al["options"], recommended=al["recommended"]))

    state.save()
    state.log_run(now, len(dets), fired, alerted, suppressed)
    return events
