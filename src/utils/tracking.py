"""Experiment tracking: MLflow when installed, otherwise an append-only JSON-lines log in experiments/.

The MLflow branch (params, metrics, model artifact, registry entry) was written to the MLflow API but could not
be executed in the build sandbox (mlflow not installable there); the JSONL branch is what actually ran.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import Settings
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def log_run(settings: Settings, run_name: str, params: dict, metrics: dict, artifact: Path | None = None) -> str:
    try:
        import mlflow  # type: ignore

        mlflow.set_experiment("purchase-intent-xai")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if artifact and Path(artifact).exists():
                mlflow.log_artifact(str(artifact))
        return "mlflow"
    except ImportError:
        d = settings.path("experiments_dir")
        d.mkdir(parents=True, exist_ok=True)
        rec = {"run": run_name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "params": params, "metrics": metrics,
               "artifact": str(artifact) if artifact else None}
        with open(d / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return "jsonl"
