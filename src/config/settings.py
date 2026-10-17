"""Typed settings loaded from config/config.yaml (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repository root (two levels above src/config)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root: Path

    # --- paths ---------------------------------------------------------------
    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]

    # --- scalars -------------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def target(self) -> str:
        return self.raw["target"]

    @property
    def test_size(self) -> float:
        return float(self.raw["split"]["test_size"])

    @property
    def cv_folds(self) -> int:
        return int(self.raw["split"]["cv_folds"])

    # --- column groups ---------------------------------------------------------
    def cols(self, key: str) -> list[str]:
        return list(self.raw["columns"][key])

    @property
    def raw_feature_columns(self) -> list[str]:
        c = self.raw["columns"]
        return c["numeric_counts"] + c["numeric_durations"] + c["numeric_rates"] + c["categorical"]

    @property
    def drop_page_values(self) -> bool:
        return bool(self.raw["features"]["drop_page_values"])

    def section(self, key: str) -> dict[str, Any]:
        return dict(self.raw[key])


def load_settings(path: str | Path | None = None) -> Settings:
    root = project_root()
    cfg_path = Path(path) if path else root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings(raw=raw, root=root)
