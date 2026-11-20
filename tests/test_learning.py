"""Decision memory tests in a temp folder (does not touch .agents/monitoring)."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from monitor.learning import DecisionMemory, is_human  # noqa: E402

CFG = {"learning": {"enabled": True, "min_entity_decisions": 1, "min_signal_decisions": 3, "propose_after": 2}}


def rec(mem, aid, signal, entity, recommended, decision, by="human (decide.py)"):
    mem.record(anomaly_id=aid, signal=signal, entity=entity, severity=4, recommended=recommended,
               decision=decision, decided_by=by, data_ts="2026-11-18 06:00")


with tempfile.TemporaryDirectory() as d:
    mem = DecisionMemory(Path(d), CFG)

    # 0) only humans count
    assert is_human("terminal (human)") and is_human("Marco")
    assert not is_human("auto (rec)") and not is_human("no terminal input -> recommended")
    assert not is_human("scheduled tick (no human) -> pending")
    rec(mem, "S6-x1-adv_02", "S6", "adv_02", "A", "B", by="auto (B)")
    a = mem.advise("S6", "adv_02", "A")
    assert not a["applied"] and not a["basis"], a
    print("ok: automatic decisions are recorded but never learned from")

    # 1) one human override on the same signal+account flips the next recommendation
    rec(mem, "S6-x2-adv_02", "S6", "adv_02", "A", "B")
    a = mem.advise("S6", "adv_02", "A")
    assert a["applied"] and a["recommended"] == "B" and a["scope"] == "account", a
    print("ok: override learned ->", a["text"])

    # 2) the same situation seen again (e.g. a second replay, same anomaly id) learns from the earlier decision
    a = mem.advise("S6", "adv_02", "A")
    assert a["applied"] and a["basis"][0]["anomaly_id"] == "S6-x2-adv_02", a
    print("ok: a repeat of the same anomaly id uses the earlier decision")

    # 3) another account: needs 3 human decisions across accounts
    a = mem.advise("S6", "adv_04", "A")
    assert not a["applied"] and not a["basis"], a
    rec(mem, "S6-x3-adv_01", "S6", "adv_01", "A", "B")
    rec(mem, "S6-x4-adv_06", "S6", "adv_06", "A", "B")
    a = mem.advise("S6", "adv_04", "A")
    assert a["applied"] and a["scope"] == "signal" and a["recommended"] == "B", a
    print("ok: cross-account learning only after 3 human decisions ->", a["text"])

    # 4) confirmation and split votes do not change anything
    rec(mem, "S4-y1-adv_03", "S4", "adv_03", "A", "A")
    a = mem.advise("S4", "adv_03", "A")
    assert not a["applied"] and "confirmed" in a["text"], a
    rec(mem, "S4-y2-adv_03", "S4", "adv_03", "A", "B")
    a = mem.advise("S4", "adv_03", "A")
    assert not a["applied"] and "split" in a["text"], a
    print("ok: confirmations and split votes keep the analysis' recommendation")

    # 5) proposals after 2x hold on the same signal+account, written but never applied
    assert not mem.proposals()
    rec(mem, "S6-x5-adv_02", "S6", "adv_02", "B", "B")
    props = mem.proposals()
    assert len(props) == 1 and props[0]["signal"] == "S6" and props[0]["entity"] == "adv_02", props
    assert mem.proposals_path.exists() and "Not applied" in mem.proposals_path.read_text(encoding="utf-8")
    print("ok: proposal written ->", props[0]["text"][:90], "...")

    mem.forget()
    assert not mem.path.exists() and not mem.proposals_path.exists()
    print("ok: forget clears memory and proposals")
