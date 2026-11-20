"""Data access. Everything downstream only sees `MarketData`, so the CSV source can be
swapped for an API/SQL/MCP source by implementing `DataSource.load()`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ADV_COLUMNS = ["ts", "advertiser_id", "impressions", "clicks", "conversions",
               "spend", "revenue", "frequency"]
SUP_COLUMNS = ["ts", "source_id", "ad_requests", "impressions", "fill_rate", "clicks",
               "conversions", "fast_conversions", "top_ip_click_share", "payouts"]


@dataclass
class MarketData:
    """Hourly data, long format. `wide()` gives metric pivots (index ts, columns entity)."""
    advertisers: pd.DataFrame
    supply: pd.DataFrame
    label: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def latest_ts(self) -> pd.Timestamp | None:
        stamps = [df.ts.max() for df in (self.advertisers, self.supply) if len(df)]
        return max(stamps) if stamps else None

    def wide(self, side: str, metric: str) -> pd.DataFrame:
        key = (side, metric)
        if key not in self._cache:
            df, ent = ((self.advertisers, "advertiser_id") if side == "adv"
                       else (self.supply, "source_id"))
            self._cache[key] = df.pivot(index="ts", columns=ent, values=metric).sort_index()
        return self._cache[key]

    def until(self, now: pd.Timestamp) -> "MarketData":
        """View of the data as it would have looked at `now` (used by replay)."""
        sub = MarketData(self.advertisers[self.advertisers.ts <= now],
                         self.supply[self.supply.ts <= now], self.label)
        for key, w in self._cache.items():
            sub._cache[key] = w.loc[:now]
        return sub


class DataSource:
    def load(self) -> MarketData:  # pragma: no cover - interface
        raise NotImplementedError


class CsvSource(DataSource):
    def __init__(self, folder: Path, advertisers_file: str, supply_file: str, label: str = ""):
        self.folder = Path(folder)
        self.advertisers_file = advertisers_file
        self.supply_file = supply_file
        self.label = label

    def load(self) -> MarketData:
        paths = [self.folder / self.advertisers_file, self.folder / self.supply_file]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing data file(s): {missing}. "
                                    "Run: python src/generate_demo_data.py")
        adv = pd.read_csv(paths[0], parse_dates=["ts"])
        sup = pd.read_csv(paths[1], parse_dates=["ts"])
        _require(adv, ADV_COLUMNS, paths[0])
        _require(sup, SUP_COLUMNS, paths[1])
        return MarketData(adv[ADV_COLUMNS], sup[SUP_COLUMNS], self.label)


class ApiSource(DataSource):
    """Placeholder for the challenge API. Must return the same columns as the CSVs."""

    def __init__(self, location: str):
        self.location = location

    def load(self) -> MarketData:
        raise NotImplementedError(f"API source not connected yet ({self.location})")


def _require(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def make_source(cfg: dict, root: Path) -> DataSource:
    src = cfg["data_sources"][0]
    kind = src["type"]
    if kind in ("csv", "demo"):
        return CsvSource(root / src["location"], src.get("advertisers_file", "advertisers.csv"),
                         src.get("supply_file", "supply.csv"),
                         label="DEMO DATA" if kind == "demo" else "")
    if kind == "api":
        return ApiSource(src["location"])
    raise ValueError(f"unsupported data source type: {kind}")
