from __future__ import annotations

import json
from types import SimpleNamespace

from qat.config import (
    QuantizationVariant,
    RuntimeConfig,
    SplitConfig,
    make_manifest,
    make_resume_fingerprint,
)
from qat.runner import (
    EvaluationSummary,
    RunStage,
    append_metrics_once,
    artifact_dir_for_run,
    build_matrix,
    resume_or_validate,
    run_single,
)


def test_build_matrix_expands_smoke_in_stable_order() -> None:
    config = RuntimeConfig(task="smoke", split=SplitConfig("smoke", 2, 1, 17))
    matrix = build_matrix(config)
    assert len(matrix) == 7
    assert matrix[0].task == "baseline"
    assert matrix[0].quantization_variant is None
    assert [item.quantization_variant for item in matrix[1:]] == list(
        QuantizationVariant
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


def test_resume_or_validate_rejects_mismatched_fingerprint(tmp_path) -> None:
    config = RuntimeConfig(
        task="baseline",
        split=SplitConfig("smoke", 2, 1, 17),
        artifact_root=tmp_path,
    )
    artifact_dir = artifact_dir_for_run(config)
    artifact_dir.mkdir(parents=True)
    manifest = make_manifest(config)
    payload = manifest.to_dict()
    payload["resume_fingerprint"] = "wrong"
    (artifact_dir / "manifest.json").write_text(json.dumps(payload))
    try:
        resume_or_validate(config, artifact_dir)
    except ValueError as exc:
        assert "resume fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatch to raise")


def test_run_single_completed_run_noops(tmp_path) -> None:
    config = RuntimeConfig(
        task="baseline",
        split=SplitConfig("smoke", 2, 1, 17),
        artifact_root=tmp_path,
    )
    artifact_dir = artifact_dir_for_run(config)
    artifact_dir.mkdir(parents=True)
    manifest = make_manifest(config)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )
    (artifact_dir / "runner_state.json").write_text(
        json.dumps(
            {
                "stage": RunStage.COMPLETED.value,
                "resume_fingerprint": make_resume_fingerprint(config),
            }
        )
    )
    result = run_single(config)
    assert result.stage == RunStage.COMPLETED
    assert result.resumed


def test_run_single_skips_retraining_after_trained_state(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = RuntimeConfig(
        task="baseline",
        split=SplitConfig("smoke", 2, 1, 17),
        artifact_root=tmp_path,
    )
    work_root = tmp_path / ".work" / artifact_dir_for_run(config).name
    checkpoint_dir = work_root / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    manifest = make_manifest(config, split_manifest_path=tmp_path / "split.json")
    (work_root / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    (work_root / "runner_state.json").write_text(
        json.dumps(
            {
                "stage": RunStage.TRAINED.value,
                "resume_fingerprint": make_resume_fingerprint(config),
            }
        )
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(json.dumps({"train_indices": [0], "test_indices": [1]}))
    calls = {"train": 0, "export": 0, "eval": 0}

    def fake_preflight(config):  # noqa: ANN001
        return None

    def fake_train(*args, **kwargs):  # noqa: ANN001, ARG001
        calls["train"] += 1
        return None

    def fake_export(config, checkpoint_dir):  # noqa: ANN001
        calls["export"] += 1
        artifact_dir = artifact_dir_for_run(config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()))
        return type("ExportResult", (), {"compile_status": "disabled"})()

    def fake_eval(config, artifact_dir, split_manifest_path):  # noqa: ANN001
        calls["eval"] += 1
        return EvaluationSummary(
            decisions=[],
            metrics_rows=[
                {
                    "model_name": config.model_id,
                    "quantization_artifact": artifact_dir.name,
                    "quantization_dtype": "bf16/bf16",
                    "quantization_granularity": "none",
                    "quantization_method": "none",
                    "metric_name": "accuracy",
                    "metric_value": 1.0,
                }
            ],
            prediction_log_path=str(artifact_dir / "predictions.json"),
        )

    monkeypatch.setattr(
        "qat.runner.ensure_split_manifest",
        lambda config: split_manifest,
    )
    monkeypatch.setattr(
        "qat.runner.verify_vllm_loadability",
        lambda artifact_dir, config: type(
            "Load",
            (),
            {"loaded": True, "error": None},
        )(),
    )
    services = SimpleNamespace(
        preflight=fake_preflight,
        train_baseline=fake_train,
        train_qat=fake_train,
        export_model=fake_export,
        evaluate_model=fake_eval,
    )
    result = run_single(config, services=services)
    assert result.stage == RunStage.COMPLETED
    assert calls["train"] == 0
    assert calls["export"] == 1
    assert calls["eval"] == 1


def test_run_single_releases_gpu_memory_around_vllm_steps(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    config = RuntimeConfig(
        task="baseline",
        split=SplitConfig("smoke", 2, 1, 17),
        artifact_root=tmp_path,
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(json.dumps({"train_indices": [0], "test_indices": [1]}))
    calls: list[str] = []
    manifest = make_manifest(config, split_manifest_path=split_manifest)

    def fake_preflight(config):  # noqa: ANN001, ARG001
        return None

    def fake_train(config, split_manifest_path, checkpoint_dir):  # noqa: ANN001, ARG001
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fake_export(config, checkpoint_dir):  # noqa: ANN001, ARG001
        artifact_dir = artifact_dir_for_run(config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()))
        return type("ExportResult", (), {"compile_status": "disabled"})()

    def fake_verify(artifact_dir, config):  # noqa: ANN001, ARG001
        calls.append("verify")
        return type("Load", (), {"loaded": True, "error": None})()

    def fake_eval(config, artifact_dir, split_manifest_path):  # noqa: ANN001, ARG001
        calls.append("eval")
        return EvaluationSummary(
            decisions=[],
            metrics_rows=[],
            prediction_log_path=str(artifact_dir / "predictions.json"),
        )

    monkeypatch.setattr(
        "qat.runner.ensure_split_manifest",
        lambda config: split_manifest,
    )
    monkeypatch.setattr(
        "qat.runner.verify_vllm_loadability",
        fake_verify,
    )
    monkeypatch.setattr(
        "qat.runner._release_gpu_memory",
        lambda: calls.append("release"),
    )
    services = SimpleNamespace(
        preflight=fake_preflight,
        train_baseline=fake_train,
        train_qat=fake_train,
        export_model=fake_export,
        evaluate_model=fake_eval,
    )
    result = run_single(config, services=services)
    assert result.stage == RunStage.COMPLETED
    assert calls == ["release", "verify", "release", "eval", "release"]
