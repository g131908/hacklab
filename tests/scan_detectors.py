"""Run every detector over every hour of the data and list raw fires (no self-check, no state).
Used to check thresholds for false positives: python tests/scan_detectors.py [S2 S8 ...]"""
import collections
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from monitor.detectors import run_detectors  # noqa: E402
from monitor.loader import make_source  # noqa: E402

only = set(sys.argv[1:])
cfg = yaml.safe_load((ROOT / ".agents/monitoring/config.yaml").read_text(encoding="utf-8"))
data = make_source(cfg, ROOT).load()
fires = collections.defaultdict(list)
for now in pd.date_range(data.advertisers.ts.min(), data.latest_ts, freq="h"):
    for d in run_detectors(data.until(now), now, cfg):
        if d.fired and (not only or d.signal in only):
            fires[(d.signal, d.entity, str(now.date()))].append((now, d.title))
for k, v in sorted(fires.items()):
    print(f"{k[0]} {k[1]} {k[2]}: {len(v)}x, first {v[0][0]:%H:%M} '{v[0][1]}'")
