"""Claude's one job per incident: review the finished evidence ledger and plan what to check next.

Budget: at most ONE call per incident, made when the incident is opened (never on updates).
Grounding: Claude sees only the ledger (queries + numbers the agent already ran). It may not add
numbers. Its output must cite ledger ids; citations that do not exist are dropped and the review is
marked ungrounded. The deterministic conclusion is never overwritten: Claude can agree, or challenge
it, and a challenge is shown as a challenge.
"""
from __future__ import annotations

import json

from . import llm

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "verdict": {"type": "string", "enum": ["agree", "challenge"]},
        "challenge": {"type": "string"},
        "next_checks": {"type": "array", "items": {
            "type": "object",
            "properties": {"check": {"type": "string"}, "why": {"type": "string"}},
            "required": ["check", "why"], "additionalProperties": False}},
        "refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "verdict", "challenge", "next_checks", "refs"],
    "additionalProperties": False,
}

SYSTEM = """You review investigations made by an autonomous monitoring agent for an online marketplace.
You get the agent's evidence ledger: every query it ran (ids q1, q2, ...) with the numbers it got, its
hypotheses with status (supported / hypothesis / ruled_out), its conclusion and its stated uncertainty.

Your job, in one pass:
1. summary: 2-4 plain sentences for an on-call manager: what changed, where, the supported cause or the
   best hypothesis, the business impact, and what is still uncertain. Use only numbers from the ledger.
2. verdict: "agree" if the evidence supports the conclusion as stated; "challenge" if a step does not
   follow from its numbers, a competing explanation was dismissed too early, or a hypothesis is stated
   as fact. Put the specific objection in `challenge` (empty string when you agree).
3. next_checks: up to 3 checks that would most reduce the remaining uncertainty and that the agent
   could NOT run on this data (e.g. a log, a per-query breakdown, a supplier record). Most valuable first.
4. refs: the ledger ids your summary relies on.

Never invent a number, an event or a cause that is not in the ledger. If the ledger does not decide
something, say so."""


def review(inv) -> dict:
    """One structured call. Returns {summary, verdict, challenge, next_checks, refs, grounded}."""
    d = inv.to_dict()
    ledger = dict(
        trigger=d["trigger"], classification=d["classification"], headline=d["headline"], chain=d["chain"],
        segment=d["segment"], onset=d["onset"], impact=d["impact"], uncertainty=d["uncertainty"],
        hypotheses=[dict(cause=h["cause"], status=h["status"], score=h["score"], evidence=h["evidence"],
                         against=h["against"], refs=h["refs"]) for h in d["hypotheses"]],
        steps=[dict(id=s["id"], kind=s["kind"], title=s["title"], result=s["result"], decision=s["decision"])
               for s in d["steps"]],
    )
    out = llm.call_json("incident.review", SYSTEM, json.dumps(ledger, default=str), SCHEMA,
                        effort="medium", max_tokens=6000)
    ids = {s["id"] for s in d["steps"]}
    bad = [r for r in out["refs"] if r not in ids]
    out["refs"] = [r for r in out["refs"] if r in ids]
    out["grounded"] = bool(out["refs"]) and not bad
    out["model"] = llm.MODEL
    return out
