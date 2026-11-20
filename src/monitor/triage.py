"""TRIAGE: score 1-5 (SKILL.md section 3) + ROUTE placeholder (section 4)."""
from __future__ import annotations

from .detectors import Detection

# floor = severity floor from the signal catalog.
# cross_side: affects advertisers and supply. multi_domain: needs 2+ specialist skills.
# irreversible: the recommended fix touches billing/users or cannot simply be undone.
SIGNALS = {
    "S1": dict(name="Spend pacing", floor=3, cross_side=False, multi_domain=False,
               irreversible=False, route=["ads", "attribution"]),
    "S4": dict(name="Tracking break", floor=4, cross_side=True, multi_domain=True,
               irreversible=False, route=["conversion-tracking", "analytics"]),
    "S6": dict(name="Creative fatigue", floor=2, cross_side=False, multi_domain=True,
               irreversible=False, route=["marketing-psychology", "ads"]),
    "S7": dict(name="Invalid traffic", floor=4, cross_side=True, multi_domain=False,
               irreversible=True, route=["analytics"]),   # payouts/billing involved
}


def score(det: Detection, cfg: dict) -> tuple[int, list[str]]:
    meta = SIGNALS[det.signal]
    s = meta["floor"]
    factors = [f"floor {meta['floor']} ({det.signal})"]
    thr = cfg["thresholds"]["money_at_risk_threshold"]
    if det.money_at_risk_day is not None and det.money_at_risk_day > thr:
        s += 1
        factors.append(f"+1 money at risk > EUR {thr}/day")
    if meta["cross_side"]:
        s += 1
        factors.append("+1 cross-side")
    if meta["multi_domain"]:
        s += 1
        factors.append("+1 multi-domain")
    if meta["irreversible"]:
        s += 1
        factors.append("+1 irreversible/external")
    if det.entity in (cfg["scope"].get("strategic_accounts") or []):
        s += 1
        factors.append("+1 strategic account")
    return min(s, 5), factors


def action_for(severity: int) -> str:
    if severity <= 2:
        return "log"       # daily digest only
    if severity == 3:
        return "single"    # one specialist, one recommendation
    if severity == 4:
        return "panel"     # 2-3 specialists, notify now
    return "page"          # page a human now, before diagnosis is finished


def route(det: Detection, severity: int) -> list[dict]:
    """PLACEHOLDER: specialist routing is not built yet.

    Will load the specialist skills (SKILL.md section 4) and return outputs in the
    specialist output contract. Today it only returns which skills *would* be asked,
    with status "not_implemented", so synthesis can show that no specialist ran.
    """
    skills = SIGNALS[det.signal]["route"]
    if action_for(severity) in ("log", "single"):
        skills = skills[:1]
    return [dict(skill=s, status="not_implemented", diagnosis=None, hypotheses=[],
                 missing_data=["specialist routing not implemented"]) for s in skills]
