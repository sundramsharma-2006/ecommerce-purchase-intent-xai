"""Data access layer: load + validate the sessions CSV, fail loudly on bad input (PRD 4.1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Settings, load_settings
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when the sessions file does not match the expected schema."""


EXPECTED_DTYPE_KIND = {
    "Administrative": "i", "Administrative_Duration": "f", "Informational": "i", "Informational_Duration": "f",
    "ProductRelated": "i", "ProductRelated_Duration": "f", "BounceRates": "f", "ExitRates": "f", "PageValues": "f",
    "SpecialDay": "f", "OperatingSystems": "i", "Browser": "i", "Region": "i", "TrafficType": "i", "Converted": "i",
}  # kind codes: i = integer, f = float


def validate_sessions(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Check columns, dtypes, ranges and the target; return the frame unchanged if valid."""
    expected = settings.raw_feature_columns + [settings.target]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    for col, kind in EXPECTED_DTYPE_KIND.items():
        actual = df[col].dtype.kind
        ok = actual == kind or (kind == "f" and actual in "if")
        if not ok:
            raise DataValidationError(f"Column {col!r} has dtype {df[col].dtype}, expected kind {kind!r}")
    if df[expected].isna().any().any():
        bad = df[expected].columns[df[expected].isna().any()].tolist()
        raise DataValidationError(f"Missing values in: {bad}")
    if not set(df[settings.target].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary 0/1")
    for col in ("BounceRates", "ExitRates", "SpecialDay"):
        if df[col].min() < 0 or df[col].max() > 1:
            raise DataValidationError(f"{col} outside [0, 1]")
    count_cols = settings.cols("numeric_counts") + settings.cols("numeric_durations")
    if (df[count_cols] < 0).any().any():
        raise DataValidationError("Negative page counts or durations")
    unknown_month = set(df["Month"].unique()) - set(settings.cols("months"))
    if unknown_month:
        raise DataValidationError(f"Unknown Month values: {sorted(unknown_month)}")
    return df


def load_sessions(settings: Settings | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """Single entry point used by notebooks, training, API and dashboard."""
    settings = settings or load_settings()
    csv_path = Path(path) if path else settings.path("raw_data")
    if not csv_path.exists():
        raise FileNotFoundError(f"Sessions file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # malformed CSV -> clear message
        raise DataValidationError(f"Could not parse {csv_path}: {exc}") from exc
    if "Weekend" in df.columns:        # a missing column is reported by validate_sessions, not as a KeyError here
        df["Weekend"] = df["Weekend"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    validate_sessions(df, settings)
    log.info("Loaded %d sessions from %s (conversion rate %.3f)", len(df), csv_path.name, df[settings.target].mean())
    return df


def stratified_split(df: pd.DataFrame, settings: Settings):
    """Deterministic stratified train/test split (same result on every run)."""
    import numpy as np

    rng = np.random.RandomState(settings.seed)
    y = df[settings.target].to_numpy()
    test_idx = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        test_idx.extend(idx[: int(round(len(idx) * settings.test_size))])
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
