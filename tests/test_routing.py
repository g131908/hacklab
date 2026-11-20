"""Offline test of the LLM specialist path with a fake Anthropic client (no API calls)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from monitor.detectors import run_detectors
from monitor.loader import make_source
from monitor.routing import Router

cfg = yaml.safe_load((ROOT / ".agents/monitoring/config.yaml").read_text(encoding="utf-8"))
data = make_source(cfg, ROOT).load()
now = pd.Timestamp("2026-11-13 11:00")
view = data.until(now)
det = next(d for d in run_detectors(view, now, cfg) if d.fired and d.signal == "S4")

calls = []


class FakeMessages:
    def __init__(self, fail_skill=None):
        self.fail_skill = fail_skill

    def create(self, **kw):
        calls.append(kw)
        skill = kw["system"][0]["text"].split('"')[1]
        if skill == self.fail_skill:
            raise RuntimeError("simulated API error")
        body = dict(diagnosis=f"LLM diagnosis from {skill}", evidence=["e1"],
                    hypotheses=[dict(cause="c", confidence="high", how_to_verify="v")],
                    recommended_action=dict(action="a", tier=1, reversible=True, est_impact="i"),
                    stance="act_now", missing_data=[])
        return NS(stop_reason="end_turn", model=kw["model"],
                  content=[NS(type="thinking", thinking=""), NS(type="text", text=json.dumps(body))])


def client(fail=None):
    return NS(beta=NS(messages=FakeMessages(fail)))


# 1) all specialists succeed
r = Router(cfg, "llm", client=client())
out = r.route(det, 5, "S4-test", view)
assert [o["skill"] for o in out] == ["conversion-tracking", "analytics"], out
assert all(o["source"].startswith("llm:") for o in out), out
kw = calls[0]
assert kw["model"] == "claude-opus-5-5"
assert kw["output_config"]["format"]["type"] == "json_schema"
assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
assert "# Conversion Tracking" in kw["system"][0]["text"] or "conversion" in kw["system"][0]["text"].lower()
payload = json.loads(kw["messages"][0]["content"].split("(JSON):\n", 1)[1])
assert payload["entity"] == "adv_03" and "recent_hourly_csv" in payload["data"]
print("ok: llm path, request shape, payload")

# 2) one specialist fails -> that one falls back to rules, the other stays llm
calls.clear()
r = Router(cfg, "llm", client=client(fail="analytics"))
out = r.route(det, 5, "S4-test", view)
src = {o["skill"]: o["source"] for o in out}
assert src["conversion-tracking"].startswith("llm:") and src["analytics"] == "rules, llm failed", src
assert r.llm_errors and "analytics" in r.llm_errors[0]
print("ok: per-specialist fallback to rules", src)

# 3) depth guard
out = Router(cfg, "rules").route(det, 5, "S4-test", view, depth=3)
assert out[0]["source"] == "depth-guard"
print("ok: depth guard")

# 4) auto without key -> rules; single specialist for SEV 3
r = Router(cfg, "auto")
assert r.mode == "rules"
assert len(r.route(det, 3, "S4-test", view)) == 1
print("ok: auto->rules without key, SEV3 single specialist")
