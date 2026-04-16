import json
from pathlib import Path

from qat.config import RunMode, RuntimeConfig, SplitConfig, make_manifest
from qat.runner import (
    _prediction_log_path,
    append_metrics_once,
    ensure_split_manifest,
    resolve_model_path,
)


def test_append_metrics_once_dedupes_rows(tmp_path) -> None:
    row = {
        "model_name": "demo/model",
        "quantization_artifact": "run-1",
        "quantization_dtype": "bf16/bf16",
        "quantization_granularity": "none",
        "quantization_method": "none",
        "metric_name": "accuracy",
        "metric_value": 1.0,
    }
    path = tmp_path / "metrics.csv"
    assert append_metrics_once(path, row)
    assert not append_metrics_once(path, row)


def test_resolve_model_path_defaults_to_artifact_dir(tmp_path) -> None:
    config = RuntimeConfig(
        split=SplitConfig("smoke", 2, 1, 17),
        mode=RunMode.BASELINE,
        artifact_root=tmp_path,
    )
    resolved = resolve_model_path(config)
    assert resolved == tmp_path / "baseline-smoke-baseline-seed17"


def test_prediction_log_path_uses_output_directory(tmp_path) -> None:
    output_path = tmp_path / "metrics.csv"
    model_path = Path("/tmp/demo-model")
    assert _prediction_log_path(output_path, model_path) == (
        tmp_path / "predictions_demo-model.json"
    )


def test_ensure_split_manifest_reuses_existing_path(tmp_path) -> None:
    config = RuntimeConfig(
        split=SplitConfig("smoke", 2, 1, 17),
        mode=RunMode.BASELINE,
        artifact_root=tmp_path,
    )
    path = tmp_path / "splits" / "numinamath_cot-smoke-seed17-9d8d210c.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"train_indices": [0], "test_indices": [1]}))
    assert ensure_split_manifest(config) == path


def test_manifest_round_trip_shape(tmp_path) -> None:
    config = RuntimeConfig(
        split=SplitConfig("smoke", 2, 1, 17),
        mode=RunMode.BASELINE,
        artifact_root=tmp_path,
    )
    manifest = make_manifest(config)
    payload = manifest.to_dict()
    assert payload["mode"] == "baseline"
