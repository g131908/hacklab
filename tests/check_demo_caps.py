"""Sanity check for DEMO DATA: daily spend per advertiser vs the S2 caps in config, and the S8 scenario rows."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / ".agents/monitoring/config.yaml").read_text(encoding="utf-8"))
caps = cfg["caps"]["account_daily_caps"]
adv = pd.read_csv(ROOT / "data/advertisers.csv", parse_dates=["ts"])
day = adv.groupby([adv.ts.dt.date, "advertiser_id"]).spend.sum().unstack()
runaway = day.index.astype(str) == "2026-11-16"
normal = day[~runaway]
print("max normal-day spend vs cap (headroom):")
for a, cap in caps.items():
    print(f"  {a}: max {normal[a].max():7.0f}  cap {cap:5d}  headroom {cap / normal[a].max() - 1:+.0%}")
print(f"adv_05 on 2026-11-16 (runaway): {day.loc[runaway, 'adv_05'].iloc[0]:.0f} vs cap {caps['adv_05']}")

sup = pd.read_csv(ROOT / "data/supply.csv", parse_dates=["ts"])
b = sup[(sup.source_id == "src_B") & (sup.ts >= "2026-11-17 11:00") & (sup.ts < "2026-11-17 22:00")]
print(b[["ts", "ad_requests", "impressions", "fill_rate"]].to_string(index=False))
