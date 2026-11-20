"""SYNTHESIZE: one alert, one decision (SKILL.md section 5)."""
from __future__ import annotations

from .detectors import Detection
from .triage import SIGNALS

# Rule-based "most likely" cause + decision options per signal, until specialists are wired in.
PLAYBOOK = {
    "S1": dict(
        up=dict(why="Bid/budget change or auction shift raised CPM/spend (rule-based; ads specialist not yet routed)",
                A="Cap today's budget for {e} at last week's level (reversible, max -20% per run)",
                B="Keep running and re-check pacing in 2h (risk: overspend continues)",
                rec="A", owner="campaign manager", recheck="2h"),
        down=dict(why="Delivery dropped: budget exhausted, bids too low, or demand dip (rule-based)",
                  A="Check bids/budget for {e} and raise delivery (reversible)",
                  B="Hold and re-check in 2h (risk: under-delivery vs plan)",
                  rec="B", owner="campaign manager", recheck="2h"),
    ),
    "S4": dict(down=dict(
        why="Conversion event stopped firing while traffic is normal (rule-based; conversion-tracking not yet routed)",
        A="Pause automated bidding on {e} until tracking is fixed (reversible, stops wasted spend)",
        B="Keep running, fix the tag first (risk: bidding learns from bad data)",
        rec="A", owner="on-call", recheck="2h")),
    "S6": dict(down=dict(
        why="Audience saturation: same creative shown more often, CTR decays (rule-based; marketing-psychology not yet routed)",
        A="Rotate in fresh creatives / cap frequency for {e} (reversible)",
        B="Keep current creatives and re-check in 2 days (risk: CTR keeps sliding)",
        rec="A", owner="creative lead", recheck="2 days")),
    "S7": dict(up=dict(
        why="Bot / click-farm traffic from one source: CTR spike, sub-10s conversions, IP cluster (rule-based)",
        A="Block {e} for affected campaigns and hold its payouts pending review (needs approval, touches billing)",
        B="Keep serving, flag clicks as suspect and review tomorrow (risk: advertisers billed for invalid clicks)",
        rec="A", owner="supply ops + finance", recheck="1h")),
}


def build(det: Detection, severity: int, factors: list[str], checks: dict,
          specialists: list[dict], label: str = "", realert: bool = False) -> dict:
    pb = PLAYBOOK[det.signal][det.direction]
    e = det.entity
    kind = "Advertiser" if det.entity_type == "advertiser" else "Supply source"
    headline = f"[SEV {severity}] {SIGNALS[det.signal]['name']}: {kind} \"{e}\" - {det.title}"
    if realert:
        headline += " (severity raised)"
    pending = [s["skill"] for s in specialists if s["status"] != "done"]
    lines = [
        f"{'[' + label + '] ' if label else ''}{headline}",
        f"When: {det.ts:%a %d.%m.%Y %H:%M} (window {det.window_start:%d.%m %H:%M} - {det.window_end:%d.%m %H:%M})",
        "What happened: " + "; ".join(det.evidence) + ".",
        f"Why (most likely): {pb['why']}.",
        f"Money at risk: {det.money_note or 'MISSING (not computable from available data)'}.",
        "Specialists disagree on: (none - routing to " + ", ".join(pending) + " pending)",
        "Decision for you:",
        f"  A) {pb['A'].format(e=e)}",
        f"  B) {pb['B'].format(e=e)}",
        f"Recommended: {pb['rec']}. Owner: {pb['owner']}. Re-check: in {pb['recheck']}.",
        "Checks run: " + ", ".join(f"{k} {v}" for k, v in checks.items()) + ".",
        "Triage: " + ", ".join(factors) + ".",
    ]
    return dict(text="\n".join(lines), options={"A": pb["A"].format(e=e), "B": pb["B"].format(e=e)},
                recommended=pb["rec"], headline=headline)
