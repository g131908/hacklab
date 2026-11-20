"""SYNTHESIZE: merge specialist outputs into one alert with one decision (SKILL.md section 5).
Disagreements between specialists are surfaced as the decision point, not hidden."""
from __future__ import annotations

from .detectors import Detection
from .triage import SIGNALS

# Decision options per signal: A = intervene now, B = hold / keep running.
PLAYBOOK = {
    "S1": dict(
        up=dict(A="Cap today's budget/bids for {e} (-20% per run, reversible)",
                B="Keep running and re-check pacing in 2h (risk: overspend continues)",
                rec="A", owner="campaign manager", recheck="2h"),
        down=dict(A="Check bids/budget for {e} and raise delivery (reversible)",
                  B="Hold and re-check in 2h (risk: under-delivery vs plan)",
                  rec="B", owner="campaign manager", recheck="2h"),
    ),
    "S2": dict(up=dict(
        A="Lower {e}'s bids/daily budget now so the projection stays under the hard cap (-20% per run, reversible)",
        B="Keep running and re-check in 1h (risk: the hard cap is breached, overspend billed)",
        rec="A", owner="campaign manager + finance", recheck="1h")),
    "S8": dict(down=dict(
        A="Open an incident with {e} / its SSP integration and shift demand to the other sources (reversible)",
        B="Hold and re-check in 2h (risk: advertisers keep losing delivery on {e})",
        rec="A", owner="supply ops", recheck="2h")),
    "S4": dict(down=dict(
        A="Pause automated bidding on {e} until tracking is fixed (reversible, stops wasted spend)",
        B="Keep running, fix the tag first (risk: bidding learns from bad data)",
        rec="A", owner="on-call", recheck="2h")),
    "S6": dict(down=dict(
        A="Rotate in fresh creatives / cap frequency for {e} (reversible)",
        B="Keep current creatives and re-check in 2 days (risk: CTR keeps sliding)",
        rec="A", owner="creative lead", recheck="2 days")),
    "S7": dict(up=dict(
        A="Block {e} for affected campaigns and hold its payouts pending review (needs approval, touches billing)",
        B="Keep serving, flag clicks as suspect and review tomorrow (risk: advertisers billed for invalid clicks)",
        rec="A", owner="supply ops + finance", recheck="1h")),
}

STANCE_TO_OPTION = {"act_now": "A", "hold": "B"}
CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def _ran(specialists: list[dict]) -> list[dict]:
    return [s for s in specialists if s.get("diagnosis")]


def _why(specialists: list[dict]) -> str:
    ran = _ran(specialists)
    if not ran:
        return "UNKNOWN - no specialist produced a diagnosis (see missing data)"
    primary = ran[0]
    hyps = sorted(primary.get("hypotheses") or [], key=lambda h: CONF_RANK.get(h["confidence"], 3))
    top = f" Most likely cause: {hyps[0]['cause']} ({hyps[0]['confidence']} confidence)." if hyps else ""
    return f"{primary['diagnosis']}{top} [{primary['skill']}, {primary['source']}]"


def _disagreement(specialists: list[dict]) -> tuple[str, str | None]:
    """Returns (text, recommended option or None if specialists disagree)."""
    ran = [s for s in _ran(specialists) if s.get("stance")]
    stances = {s["stance"] for s in ran}
    if len(ran) < 2 or len(stances) == 1:
        rec = STANCE_TO_OPTION.get(ran[0]["stance"]) if ran else None
        return "(none)", rec
    parts = [f"{s['skill']} says {s['stance'].replace('_', ' ')} ({s['diagnosis'].rstrip('.')})" for s in ran]
    return "; ".join(parts), None


def build(det: Detection, severity: int, factors: list[str], checks: dict,
          specialists: list[dict], label: str = "", realert: bool = False) -> dict:
    pb = PLAYBOOK[det.signal][det.direction]
    e = det.entity
    kind = "Advertiser" if det.entity_type == "advertiser" else "Supply source"
    headline = f"[SEV {severity}] {SIGNALS[det.signal]['name']}: {kind} \"{e}\" - {det.title}"
    if realert:
        headline += " (severity raised)"

    disagree, spec_rec = _disagreement(specialists)
    if disagree != "(none)":
        rec, rec_note = pb["rec"], " (specialists disagree - default from playbook, your call)"
    else:
        rec, rec_note = spec_rec or pb["rec"], ""

    spec_evidence = [ev for s in _ran(specialists) for ev in s.get("evidence", [])][:3]
    missing = []
    for s in specialists:
        for m in s.get("missing_data") or []:
            if m not in missing:
                missing.append(m)
    verify = []
    for s in _ran(specialists):
        for h in s.get("hypotheses") or []:
            if h["confidence"] != "low" and len(verify) < 3:
                verify.append(f"{h['how_to_verify']} ({s['skill']})")
    actions = [f"{s['skill']}: {s['recommended_action']['action']} (tier {s['recommended_action']['tier']})"
               for s in _ran(specialists) if s.get("recommended_action")]
    routed = ", ".join(f"{s['skill']} ({s.get('source', '?')})" for s in specialists) or "none"

    lines = [
        f"{'[' + label + '] ' if label else ''}{headline}",
        f"When: {det.ts:%a %d.%m.%Y %H:%M} (window {det.window_start:%d.%m %H:%M} - {det.window_end:%d.%m %H:%M})",
        "What happened: " + "; ".join(det.evidence) + ".",
    ]
    if spec_evidence:
        lines.append("Specialist facts: " + "; ".join(spec_evidence) + ".")
    lines += [
        f"Why (most likely): {_why(specialists)}",
        f"Money at risk: {det.money_note or 'MISSING (not computable from available data)'}.",
        f"Specialists disagree on: {disagree}",
        "Decision for you:",
        f"  A) {pb['A'].format(e=e)}",
        f"  B) {pb['B'].format(e=e)}",
        f"Recommended: {rec}{rec_note}. Owner: {pb['owner']}. Re-check: in {pb['recheck']}.",
    ]
    if actions:
        lines.append("Staged by specialists: " + " | ".join(actions[:3]))
    if verify:
        lines.append("Verify: " + " | ".join(verify))
    lines += [
        "Missing data: " + ("; ".join(missing[:5]) if missing else "none") + ".",
        "Checks run: " + ", ".join(f"{k} {v}" for k, v in checks.items()) + ".",
        "Triage: " + ", ".join(factors) + f". Routed: {routed}.",
    ]
    return dict(text="\n".join(lines), options={"A": pb["A"].format(e=e), "B": pb["B"].format(e=e)},
                recommended=rec, headline=headline, disagreement=disagree != "(none)")
