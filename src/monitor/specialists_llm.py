"""LLM specialists: one Claude call per specialist skill. The skill's SKILL.md is the system
prompt (cached), the anomaly record + a data slice is the user message, and structured outputs
force the specialist output contract. Any failure raises; the router falls back to rules."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "how_to_verify": {"type": "string"},
                },
                "required": ["cause", "confidence", "how_to_verify"],
                "additionalProperties": False,
            },
        },
        "recommended_action": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "tier": {"type": "integer", "enum": [1, 2]},
                "reversible": {"type": "boolean"},
                "est_impact": {"type": "string"},
            },
            "required": ["action", "tier", "reversible", "est_impact"],
            "additionalProperties": False,
        },
        "stance": {"type": "string", "enum": ["act_now", "hold", "investigate"]},
        "missing_data": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnosis", "evidence", "hypotheses", "recommended_action", "stance", "missing_data"],
    "additionalProperties": False,
}

ROLE = """You are the "{skill}" specialist inside an always-on ads marketplace monitoring agent.
The monitor has detected one anomaly and routed it to you. Apply the method of the skill below to
this anomaly only, and answer the question you are given.

Rules:
- Use only numbers that appear in the anomaly record or the data slice; cite source and time window
  for each number in `evidence`. Never estimate a number you do not have: list it in `missing_data`.
- Keep facts (`evidence`), hypotheses (`hypotheses`, with confidence and how to verify) and missing
  data separate. Never state a hypothesis as a fact.
- `recommended_action`: tier 1 = read/analyse/annotate/notify; tier 2 = anything that pauses, changes
  bids or budgets, blocks supply, touches tracking config, billing or contacts customers.
- `stance`: act_now if the decision-maker should intervene now, hold if waiting is the better call,
  investigate if the data cannot decide yet.
- Be brief: diagnosis is one sentence, at most 4 evidence items and 3 hypotheses.

The skill you apply follows.

"""


def _skill_text(skill: str) -> str:
    p = SKILLS / skill / "SKILL.md"
    if not p.exists():
        raise FileNotFoundError(f"skill not found: {p}")
    return p.read_text(encoding="utf-8")


def make_client(timeout: float):
    import anthropic  # optional dependency (requirements-llm.txt)
    return anthropic.Anthropic(timeout=timeout, max_retries=1)


def run(skill: str, question: str, payload: dict, cfg: dict, client) -> dict:
    """Call Claude for one specialist. Raises on any error, refusal or invalid output."""
    rcfg = cfg.get("routing", {})
    system = [{"type": "text", "text": ROLE.format(skill=skill) + _skill_text(skill),
               "cache_control": {"type": "ephemeral"}}]
    user = ("Question to answer: " + question + "\n\nAnomaly record and data slice (JSON):\n"
            + json.dumps(payload, indent=1, default=str))
    resp = client.beta.messages.create(
        model=rcfg.get("model", "claude-opus-5-5"),
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": rcfg.get("effort", "medium"),
                       "format": {"type": "json_schema", "schema": CONTRACT_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"specialist {skill} refused")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"specialist {skill} hit max_tokens")
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    out.update(skill=skill, source=f"llm:{getattr(resp, 'model', rcfg.get('model'))}")
    return out
