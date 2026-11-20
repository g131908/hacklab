"""LOG: state and idempotency (SKILL.md section 8). Files in .agents/monitoring/:
open-anomalies.json (dedupe set), alerts.jsonl (alert/suppression log for evaluation),
decisions.md (human decisions), runs.log (one line per run), KILL (kill switch)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class State:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.anomalies_path = self.folder / "open-anomalies.json"
        self.alerts_path = self.folder / "alerts.jsonl"
        self.decisions_path = self.folder / "decisions.md"
        self.runs_path = self.folder / "runs.log"
        self.kill_path = self.folder / "KILL"
        self.anomalies: dict[str, dict] = {}
        if self.anomalies_path.exists():
            self.anomalies = json.loads(self.anomalies_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ kill switch
    def killed(self) -> bool:
        return self.kill_path.exists()

    # ------------------------------------------------------------ reset (replay)
    def reset(self) -> None:
        self.anomalies = {}
        for p in (self.anomalies_path, self.alerts_path, self.runs_path):
            if p.exists():
                p.unlink()
        self.save()

    def save(self) -> None:
        self.anomalies_path.write_text(json.dumps(self.anomalies, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ anomalies
    @staticmethod
    def make_id(signal: str, ts: pd.Timestamp, entity: str) -> str:
        return f"{signal}-{ts:%Y-%m-%d}-{entity}"

    def find_open(self, signal: str, entity: str) -> dict | None:
        for a in self.anomalies.values():
            if a["signal"] == signal and a["entity"] == entity and a["status"] == "open":
                return a
        return None

    def open_anomaly(self, signal: str, entity: str, ts: pd.Timestamp, severity: int,
                     title: str) -> dict:
        aid = self.make_id(signal, ts, entity)
        rec = dict(anomaly_id=aid, signal=signal, entity=entity, status="open",
                   first_seen=str(ts), last_seen=str(ts), severity=severity,
                   last_alert=str(ts), alerts=1, in_range_checks=0, title=title)
        self.anomalies[aid] = rec
        return rec

    def in_cooldown(self, rec: dict, now: pd.Timestamp, hours: float) -> bool:
        return now - pd.Timestamp(rec["last_alert"]) < pd.Timedelta(hours=hours)

    def mark_in_range(self, signal: str, entity: str, now: pd.Timestamp, needed: int) -> dict | None:
        """Auto-resolve after `needed` consecutive in-range checks. Returns record if resolved."""
        rec = self.find_open(signal, entity)
        if rec is None:
            return None
        rec["in_range_checks"] += 1
        if rec["in_range_checks"] >= needed:
            rec["status"] = "resolved"
            rec["resolved_at"] = str(now)
            return rec
        return None

    # ------------------------------------------------------------ logs
    def log_event(self, event: dict) -> None:
        with self.alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def log_run(self, now: pd.Timestamp, checked: int, fired: int, alerted: int,
                suppressed: int, errors: int = 0, note: str = "") -> None:
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(f"{now} checked={checked} fired={fired} alerted={alerted} "
                    f"suppressed={suppressed} errors={errors} {note}".rstrip() + "\n")

    def log_decision(self, anomaly_id: str, option: str, text: str, approver: str,
                     data_label: str = "") -> None:
        new = not self.decisions_path.exists()
        with self.decisions_path.open("a", encoding="utf-8") as f:
            if new:
                f.write("# Decisions\n\n")
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            label = f" [{data_label}]" if data_label else ""
            f.write(f"- {stamp}{label} `{anomaly_id}`: option {option} ({text}) - approver: {approver}. "
                    "Staged only, not executed (Tier 2). Review: next run.\n")
