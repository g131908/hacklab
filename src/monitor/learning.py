"""Decision memory: the agent learns from HUMAN decisions, never from its own.

- decision-history.jsonl (survives replay resets) records every gate decision.
- advise(): if humans overrode the recommendation for the same signal + account before,
  the next alert recommends what they chose and says so. Across accounts only after
  `min_signal_decisions` human decisions on that signal.
- proposals(): repeated "hold" (option B) answers become a written proposal to adjust a
  threshold. Proposals are never applied automatically (Tier 2: config changes need a human).

Option semantics are the same for every signal: A = intervene now, B = hold / keep running.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

NON_HUMAN = ("auto", "no terminal input", "scheduled tick")

THRESHOLD_FOR = {
    "S1": ("s1_pacing_high", "raise to e.g. 1.45 for this account"),
    "S2": ("account_daily_caps", "raise this account's daily cap"),
    "S4": ("s4_conv_drop", "require a deeper drop, e.g. 0.90"),
    "S6": ("s6_ctr_drop", "require a steeper CTR slide, e.g. 0.35"),
    "S7": ("s7_ctr_sigma", "raise to e.g. 4.0"),
    "S8": ("s8_fill_drop", "require a deeper fill drop, e.g. 0.30"),
}


def is_human(decided_by: str | None) -> bool:
    return bool(decided_by) and not str(decided_by).lower().startswith(NON_HUMAN)


class DecisionMemory:
    def __init__(self, folder: Path, cfg: dict):
        self.path = Path(folder) / "decision-history.jsonl"
        self.proposals_path = Path(folder) / "learning-proposals.md"
        lc = cfg.get("learning") or {}
        self.enabled = lc.get("enabled", True)
        self.min_entity = int(lc.get("min_entity_decisions", 1))
        self.min_signal = int(lc.get("min_signal_decisions", 3))
        self.propose_after = int(lc.get("propose_after", 2))

    # ------------------------------------------------------------ storage
    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def record(self, *, anomaly_id: str, signal: str, entity: str, severity: int, recommended: str,
               decision: str, decided_by: str, data_ts: str, learned: bool = False, data_label: str = "") -> dict:
        rec = dict(anomaly_id=anomaly_id, signal=signal, entity=entity, severity=severity,
                   recommended=recommended, decision=decision, decided_by=decided_by,
                   human=is_human(decided_by), override=decision != recommended, learned_rec=learned,
                   data_ts=data_ts, decided_at=datetime.now().strftime("%Y-%m-%d %H:%M"), data_label=data_label)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["human"]:
            self.write_proposals()
        return rec

    def forget(self) -> None:
        for p in (self.path, self.proposals_path):
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------ advice
    def advise(self, signal: str, entity: str, recommended: str, exclude_id: str | None = None) -> dict:
        """Returns {applied, recommended, scope, basis, text}. `applied` False = no change, but the
        basis (if any) is still reported so the trace shows what the memory looked at."""
        if not self.enabled:
            return dict(applied=False, recommended=recommended, scope=None, basis=[], text="learning disabled")
        human = [h for h in self.load() if h["human"] and h["signal"] == signal and h["anomaly_id"] != exclude_id]
        same = [h for h in human if h["entity"] == entity]
        for scope, basis, need in (("account", same, self.min_entity), ("signal", human, self.min_signal)):
            if len(basis) < need or not basis:
                continue
            votes = Counter(h["decision"] for h in basis)
            choice, n = votes.most_common(1)[0]
            if n * 2 <= len(basis):  # no clear majority -> keep the analysis' recommendation
                return dict(applied=False, recommended=recommended, scope=scope, basis=basis,
                            text=f"{len(basis)} past human decision(s) on {signal}"
                                 f"{' / ' + entity if scope == 'account' else ''} are split ({dict(votes)}); kept {recommended}")
            where = f"{signal} / {entity}" if scope == "account" else f"{signal} (all accounts)"
            if choice == recommended:
                return dict(applied=False, recommended=recommended, scope=scope, basis=basis,
                            text=f"confirmed by {n} of {len(basis)} past human decision(s) on {where}")
            return dict(applied=True, recommended=choice, scope=scope, basis=basis,
                        text=f"you chose {choice} in {n} of {len(basis)} past decision(s) on {where}; "
                             f"recommending {choice} instead of {recommended}")
        return dict(applied=False, recommended=recommended, scope=None, basis=[],
                    text=f"no past human decisions on {signal} / {entity} yet")

    # ------------------------------------------------------------ proposals
    def proposals(self) -> list[dict]:
        human = [h for h in self.load() if h["human"]]
        out = []
        for (signal, entity), n in Counter((h["signal"], h["entity"]) for h in human if h["decision"] == "B").items():
            total = sum(1 for h in human if h["signal"] == signal and h["entity"] == entity)
            if n >= self.propose_after and n * 2 > total:
                key, how = THRESHOLD_FOR.get(signal, ("(none)", "review the signal for this account"))
                out.append(dict(signal=signal, entity=entity, holds=n, total=total, threshold=key,
                                text=f"{signal} on {entity}: you chose 'hold' (B) {n} of {total} times. "
                                     f"Proposal: {how} (config key `{key}`), or add a known event if it is planned. "
                                     "Not applied - edit .agents/monitoring/config.yaml to accept."))
        return out

    def write_proposals(self) -> None:
        props = self.proposals()
        if not props:
            if self.proposals_path.exists():
                self.proposals_path.unlink()
            return
        lines = ["# Learning proposals (from human decisions, not applied)", ""]
        lines += [f"- {p['text']}" for p in props]
        self.proposals_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
