"""Claude for the Eye agent: optional, grounded, and never the source of numbers.

- The key is read from the environment or from <repo>/.env (ANTHROPIC_API_KEY=...), never printed.
- Every call uses structured outputs, so the agent gets a schema-valid object or an exception.
- Every call is logged to .agents/eye/llm_calls.jsonl (purpose, model, tokens) for cost and audit.
- Callers always have a deterministic fallback; an exception here never stops monitoring.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / ".agents" / "eye" / "llm_calls.jsonl"
MODEL = os.environ.get("EYE_MODEL", "claude-sonnet-5-5")   # cheap default; override with EYE_MODEL
_client = None

# list prices in USD per million tokens: (input, output, cache read); cache writes cost 1.25x input
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.10), "claude-opus-5-5": (4.0, 20.0, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01), "claude-fable-5-2": (8.0, 40.0, 0.20),
          "claude-sonnet-5": (2.0, 10.0, 0.20)}


def call_cost(c: dict) -> float | None:
    p = next((v for k, v in PRICES.items() if str(c.get("model", "")).startswith(k)), None)
    if p is None:
        return None
    return (c.get("input_tokens", 0) * p[0] + c.get("cache_write", 0) * p[0] * 1.25
            + c.get("cache_read", 0) * p[2] + c.get("output_tokens", 0) * p[1]) / 1e6


def log_lines() -> int:
    return sum(1 for _ in LOG.open(encoding="utf-8")) if LOG.exists() else 0


def cost_report(since_line: int = 0) -> str:
    """Cost of the calls logged after `since_line`, plus the running total (estimate from list prices)."""
    if not LOG.exists():
        return "Claude: 0 calls, USD 0.00"
    calls = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]

    def tot(cs):
        known = [call_cost(c) for c in cs]
        usd = sum(k for k in known if k is not None)
        unknown = sum(k is None for k in known)
        return (f"{len(cs)} calls, {sum(c.get('input_tokens', 0) + c.get('cache_read', 0) + c.get('cache_write', 0) for c in cs):,} in / "
                f"{sum(c.get('output_tokens', 0) for c in cs):,} out tokens, ~USD {usd:.4f}"
                + (f" ({unknown} calls on a model without a known price)" if unknown else ""))
    run = calls[since_line:]
    models = sorted({c.get("model", "?") for c in run}) or ["-"]
    return (f"Claude this run: {tot(run)} [{', '.join(models)}]\n"
            f"Claude total:    {tot(calls)} (estimate from list prices; log: .agents/eye/llm_calls.jsonl)")


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def available() -> bool:
    _load_dotenv()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def client():
    global _client
    if _client is None:
        _load_dotenv()
        import anthropic
        _client = anthropic.Anthropic(timeout=120, max_retries=1)
    return _client


def call_json(purpose: str, system: str, user: str, schema: dict, effort: str = "medium",
              max_tokens: int = 8000) -> dict:
    """One structured call. Raises on refusal, truncation, bad JSON or any API error."""
    if not available():
        raise RuntimeError("no ANTHROPIC_API_KEY (env or .env)")
    t0 = time.perf_counter()
    resp = client().beta.messages.create(
        model=MODEL, max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{purpose}: refused")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"{purpose}: hit max_tokens")
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    u = resp.usage
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(purpose=purpose, model=getattr(resp, "model", MODEL),
                                input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                                cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                                cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
                                ms=round((time.perf_counter() - t0) * 1000))) + "\n")
    return out


# ----------------------------------------------------------------------------- uses
def ping() -> dict:
    return call_json("ping", "Answer with the requested JSON only.", "Say ok.",
                     {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                      "additionalProperties": False}, effort="low", max_tokens=1000)


def propose_components(spec: dict, components: list[str], data_links: list[dict]) -> dict:
    """Which change-log components can move a new metric? Grounded in its definition and data links."""
    schema = {"type": "object", "properties": {
        "components": {"type": "array", "items": {"type": "string", "enum": components}},
        "why": {"type": "string"}}, "required": ["components", "why"], "additionalProperties": False}
    system = ("You maintain the metric graph of an online marketplace. Given a new metric's definition, pick "
              "which kinds of change (from the allowed list) could plausibly move it. Be conservative: only "
              "components with a direct mechanism. One short sentence of reasoning.")
    user = json.dumps(dict(metric=spec, allowed_components=components, data_links_found=data_links), default=str)
    return call_json("extension.components", system, user, schema, effort="low", max_tokens=2000)
