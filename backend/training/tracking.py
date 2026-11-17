"""
Experiment tracking. MLflow when it is installed and works; otherwise a JSON file in the run folder, so
training never depends on MLflow being importable. The run folder (backend/ml_runs/<run_id>/) is always
written: model, ONNX, meta, report, model card. MLflow is a searchable index on top of it.

Each run records: parameters, headline metrics, the DATASET (sha256, label source, size, split manifest),
the code version (git sha), and the artifacts.
"""
import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EXPERIMENT = "triage-esi"


def git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def _metric_key(k: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-. /]", "_", k)[:240]


class JsonTracker:
    backend = "json"

    def __init__(self, run_dir: str):
        self.run_dir, self.record = run_dir, {"params": {}, "metrics": {}, "tags": {}}

    def start(self, run_name: str, params: Dict, tags: Dict) -> None:
        self.record.update({"run_name": run_name, "params": flatten(params), "tags": tags})

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        self.record["metrics"].update({k: v for k, v in flatten(metrics).items() if isinstance(v, (int, float)) and v is not None})

    def log_artifacts(self, path: str) -> None:
        pass

    def end(self) -> None:
        with open(os.path.join(self.run_dir, "tracker.json"), "w", encoding="utf-8") as f:
            json.dump(self.record, f, indent=2, default=str)


class MlflowTracker:
    backend = "mlflow"

    def __init__(self, run_dir: str, store_dir: str):
        import mlflow

        self.mlflow, self.run_dir = mlflow, run_dir
        os.makedirs(store_dir, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + os.path.join(store_dir, "mlflow.db").replace("\\", "/"))
        mlflow.set_experiment(EXPERIMENT)

    def start(self, run_name: str, params: Dict, tags: Dict) -> None:
        self.mlflow.start_run(run_name=run_name)
        self.mlflow.log_params({k: str(v)[:250] for k, v in flatten(params).items()})
        self.mlflow.set_tags({k: str(v)[:5000] for k, v in tags.items()})

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        for k, v in flatten(metrics).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
                self.mlflow.log_metric(_metric_key(k), float(v))

    def log_artifacts(self, path: str) -> None:
        self.mlflow.log_artifacts(path)

    def end(self) -> None:
        self.mlflow.end_run()


def get_tracker(run_dir: str, store_dir: str, use_mlflow: bool = True):
    if use_mlflow:
        try:
            return MlflowTracker(run_dir, store_dir)
        except Exception as e:  # not installed, or its store failed to initialise
            logger.warning("MLflow unavailable (%s); using JSON tracking.", type(e).__name__)
    return JsonTracker(run_dir)
