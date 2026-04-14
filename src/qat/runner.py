from __future__ import annotations

import csv
import gc
import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from qat.config import (
    QuantizationVariant,
    RunManifest,
    RuntimeConfig,
    artifact_dir_for_run,
    make_manifest,
    make_resume_fingerprint,
)
from qat.data import (
    build_split_manifest,
    load_numinamath_train_dataset,
    save_split_manifest,
)
from qat.eval.core import (
    EvaluationDecision,
    append_metrics_row,
    evaluate_prediction,
    make_metrics_row,
    write_prediction_log,
)
from qat.eval.vllm_runner import (
    build_generation_prompts,
    generate_with_vllm,
    verify_vllm_loadability,
)
from qat.export import export_model_artifact
from qat.preflight import format_report, run_preflight
from qat.train import train_baseline
from qat.train.qat import train_qat


class RunStage(StrEnum):
    PREFLIGHT = "preflight"
    SPLIT_READY = "split_ready"
    TRAINED = "trained"
    EXPORTED = "exported"
    EVALUATED = "evaluated"
    COMPLETED = "completed"
    FAILED = "failed"


STAGE_ORDER = {
    RunStage.PREFLIGHT: 0,
    RunStage.SPLIT_READY: 1,
    RunStage.TRAINED: 2,
    RunStage.EXPORTED: 3,
    RunStage.EVALUATED: 4,
    RunStage.COMPLETED: 5,
    RunStage.FAILED: 6,
}


@dataclass(frozen=True)
class RunResult:
    run_id: str
    artifact_dir: str
    stage: RunStage
    compile_status: str
    metrics_rows_written: int
    prediction_log_path: str | None
    resumed: bool


@dataclass(frozen=True)
class ExistingRunState:
    manifest: RunManifest | None
    stage: RunStage | None
    resume_allowed: bool
    completed: bool


@dataclass(frozen=True)
class EvaluationSummary:
    decisions: list[EvaluationDecision]
    metrics_rows: list[dict[str, Any]]
    prediction_log_path: str


@dataclass(frozen=True)
class RunnerServices:
    preflight: Callable[[RuntimeConfig], None]
    train_baseline: Callable[..., Any]
    train_qat: Callable[..., Any]
    export_model: Callable[..., Any]
    evaluate_model: Callable[[RuntimeConfig, Path, Path], EvaluationSummary]


def _default_services() -> RunnerServices:
    return RunnerServices(
        preflight=_run_preflight_or_raise,
        train_baseline=train_baseline,
        train_qat=train_qat,
        export_model=export_model_artifact,
        evaluate_model=evaluate_exported_model,
    )


def _run_preflight_or_raise(config: RuntimeConfig) -> None:
    checks = run_preflight(
        gpu_index=config.gpu_index,
        variant=config.quantization_variant,
    )
    failures = [check for check in checks if not check.ok]
    if failures:
        raise RuntimeError(format_report(checks))


def _split_manifest_filename(config: RuntimeConfig) -> str:
    revision_prefix = config.dataset_revision[:8]
    return (
        f"numinamath_cot-{config.split.name}-seed{config.split.seed}-"
        f"{revision_prefix}.json"
    )


def ensure_split_manifest(config: RuntimeConfig) -> Path:
    path = config.artifact_root / "splits" / _split_manifest_filename(config)
    if path.exists():
        return path
    dataset = load_numinamath_train_dataset(revision=config.dataset_revision)
    manifest = build_split_manifest(dataset, config.split)
    save_split_manifest(manifest, path)
    return path


def build_matrix(task_config: RuntimeConfig) -> list[RuntimeConfig]:
    if task_config.task not in {"smoke", "full"}:
        return [task_config]
    configs = [
        RuntimeConfig(
            task="baseline",
            split=task_config.split,
            seed=task_config.seed,
            gpu_index=task_config.gpu_index,
            artifact_root=task_config.artifact_root,
            model_id=task_config.model_id,
            model_revision=task_config.model_revision,
            dataset_id=task_config.dataset_id,
            dataset_revision=task_config.dataset_revision,
            compile_policy=task_config.compile_policy,
            training=task_config.training,
            quantization_variant=None,
        )
    ]
    for variant in QuantizationVariant:
        configs.append(
            RuntimeConfig(
                task="qat",
                split=task_config.split,
                seed=task_config.seed,
                gpu_index=task_config.gpu_index,
                artifact_root=task_config.artifact_root,
                model_id=task_config.model_id,
                model_revision=task_config.model_revision,
                dataset_id=task_config.dataset_id,
                dataset_revision=task_config.dataset_revision,
                compile_policy=task_config.compile_policy,
                training=task_config.training,
                quantization_variant=variant,
            )
        )
    return configs


def _work_root_for_run(config: RuntimeConfig) -> Path:
    return config.artifact_root / ".work" / artifact_dir_for_run(config).name


def _runner_state_path(config: RuntimeConfig) -> Path:
    artifact_dir = artifact_dir_for_run(config)
    if artifact_dir.exists():
        return artifact_dir / "runner_state.json"
    return _work_root_for_run(config) / "runner_state.json"


def _work_manifest_path(config: RuntimeConfig) -> Path:
    return _work_root_for_run(config) / "manifest.json"


def _manifest_from_json(path: Path) -> RunManifest:
    payload = json.loads(path.read_text())
    field_names = RunManifest.__dataclass_fields__  # type: ignore[attr-defined]
    manifest_payload = {name: payload[name] for name in field_names}
    return RunManifest(**manifest_payload)


def _stage_at_least(stage: RunStage | None, target: RunStage) -> bool:
    if stage is None:
        return False
    return STAGE_ORDER[stage] >= STAGE_ORDER[target]


def resume_or_validate(config: RuntimeConfig, artifact_dir: Path) -> ExistingRunState:
    expected = make_resume_fingerprint(config)
    manifest_path = artifact_dir / "manifest.json"
    work_manifest_path = _work_manifest_path(config)
    state_path = _runner_state_path(config)

    manifest: RunManifest | None = None
    if manifest_path.exists():
        manifest = _manifest_from_json(manifest_path)
    elif work_manifest_path.exists():
        manifest = _manifest_from_json(work_manifest_path)

    stage = None
    if state_path.exists():
        payload = json.loads(state_path.read_text())
        stage = RunStage(payload["stage"])
    if manifest is None:
        return ExistingRunState(
            manifest=None,
            stage=stage,
            resume_allowed=True,
            completed=False,
        )
    if manifest.resume_fingerprint != expected:
        raise ValueError(
            f"resume fingerprint mismatch for {artifact_dir.name}: "
            f"{manifest.resume_fingerprint} != {expected}"
        )
    completed = stage == RunStage.COMPLETED
    return ExistingRunState(
        manifest=manifest,
        stage=stage,
        resume_allowed=True,
        completed=completed,
    )


def _write_runner_state(
    config: RuntimeConfig,
    *,
    stage: RunStage,
    error: str | None = None,
    prediction_log_path: str | None = None,
) -> None:
    path = _runner_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage.value,
        "error": error,
        "prediction_log_path": prediction_log_path,
        "resume_fingerprint": make_resume_fingerprint(config),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_metrics_once(metrics_path: Path, row: dict[str, Any]) -> bool:
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for existing in reader:
                if (
                    existing["model_name"] == str(row["model_name"])
                    and existing["quantization_artifact"]
                    == str(row["quantization_artifact"])
                    and existing["metric_name"] == str(row["metric_name"])
                ):
                    return False
    append_metrics_row(metrics_path, row)
    return True


def _reference_answer(row: dict[str, Any]) -> str:
    messages = row["messages"]
    if messages and messages[-1]["role"] == "assistant":
        return str(messages[-1]["content"])
    raise ValueError("evaluation row is missing a terminal assistant message")


def evaluate_exported_model(
    config: RuntimeConfig,
    artifact_dir: Path,
    split_manifest_path: Path,
) -> EvaluationSummary:
    from transformers import AutoTokenizer

    split_payload = json.loads(split_manifest_path.read_text())
    dataset = load_numinamath_train_dataset(revision=config.dataset_revision)
    rows = [dataset[int(index)] for index in split_payload["test_indices"]]
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    prompts = build_generation_prompts(rows, tokenizer=tokenizer)
    generations = generate_with_vllm(
        artifact_dir,
        prompts=prompts,
        config=config,
    )
    decisions: list[EvaluationDecision] = []
    for row, generation in zip(rows, generations, strict=True):
        decisions.append(
            evaluate_prediction(
                generation.prediction_text,
                _reference_answer(row),
            )
        )
    prediction_log_path = artifact_dir / "predictions_numinamath_cot.json"
    write_prediction_log(prediction_log_path, decisions)
    accuracy = sum(decision.is_correct for decision in decisions) / max(
        1,
        len(decisions),
    )
    metrics_rows = [
        make_metrics_row(
            model_name=config.model_id,
            quantization_artifact=artifact_dir.name,
            variant=config.quantization_variant,
            metric_name="accuracy",
            metric_value=accuracy,
        )
    ]
    return EvaluationSummary(
        decisions=decisions,
        metrics_rows=metrics_rows,
        prediction_log_path=str(prediction_log_path),
    )


def _stage_from_result(result: RunResult) -> RunStage:
    return result.stage


def _promote_checkpoint_dir(config: RuntimeConfig, checkpoint_dir: Path) -> None:
    target = artifact_dir_for_run(config) / "_work" / "checkpoint"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, target)


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def run_single(
    config: RuntimeConfig,
    *,
    services: RunnerServices | None = None,
) -> RunResult:
    if config.task not in {"baseline", "qat"}:
        raise ValueError(
            f"run_single only supports baseline/qat tasks, not {config.task}"
        )

    services = services or _default_services()
    artifact_dir = artifact_dir_for_run(config)
    work_root = _work_root_for_run(config)
    checkpoint_dir = work_root / "checkpoint"
    metrics_path = config.artifact_root / "metrics_numinamath_cot.csv"

    existing = resume_or_validate(config, artifact_dir)
    if existing.completed:
        return RunResult(
            run_id=artifact_dir.name,
            artifact_dir=str(artifact_dir),
            stage=RunStage.COMPLETED,
            compile_status="completed",
            metrics_rows_written=0,
            prediction_log_path=None,
            resumed=True,
        )

    split_manifest_path = ensure_split_manifest(config)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(config, split_manifest_path=split_manifest_path)
    _work_manifest_path(config).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    _write_runner_state(config, stage=RunStage.SPLIT_READY)
    services.preflight(config)
    _write_runner_state(config, stage=RunStage.PREFLIGHT)

    if not _stage_at_least(existing.stage, RunStage.TRAINED):
        trainer = (
            services.train_baseline
            if config.task == "baseline"
            else services.train_qat
        )
        trainer(
            config,
            split_manifest_path=split_manifest_path,
            checkpoint_dir=checkpoint_dir,
        )
        _write_runner_state(config, stage=RunStage.TRAINED)

    if not _stage_at_least(existing.stage, RunStage.EXPORTED):
        export_result = services.export_model(
            config,
            checkpoint_dir=checkpoint_dir,
        )
        _promote_checkpoint_dir(config, checkpoint_dir)
        _write_runner_state(config, stage=RunStage.EXPORTED)
        compile_status = export_result.compile_status
    else:
        compile_status = "resumed"

    if not _stage_at_least(existing.stage, RunStage.EVALUATED):
        _release_gpu_memory()
        loadability = verify_vllm_loadability(artifact_dir, config)
        if not loadability.loaded:
            _write_runner_state(
                config,
                stage=RunStage.FAILED,
                error=loadability.error,
            )
            raise RuntimeError(loadability.error or "vLLM loadability check failed")
        _release_gpu_memory()
        evaluation = services.evaluate_model(
            config,
            artifact_dir,
            split_manifest_path,
        )
        _release_gpu_memory()
        rows_written = 0
        for row in evaluation.metrics_rows:
            rows_written += int(append_metrics_once(metrics_path, row))
        _write_runner_state(
            config,
            stage=RunStage.EVALUATED,
            prediction_log_path=evaluation.prediction_log_path,
        )
        _write_runner_state(
            config,
            stage=RunStage.COMPLETED,
            prediction_log_path=evaluation.prediction_log_path,
        )
        return RunResult(
            run_id=artifact_dir.name,
            artifact_dir=str(artifact_dir),
            stage=RunStage.COMPLETED,
            compile_status=compile_status,
            metrics_rows_written=rows_written,
            prediction_log_path=evaluation.prediction_log_path,
            resumed=existing.stage is not None,
        )

    return RunResult(
        run_id=artifact_dir.name,
        artifact_dir=str(artifact_dir),
        stage=RunStage.COMPLETED,
        compile_status=compile_status,
        metrics_rows_written=0,
        prediction_log_path=None,
        resumed=True,
    )


def _config_for_eval_target(config: RuntimeConfig) -> RuntimeConfig:
    return RuntimeConfig(
        task="qat" if config.quantization_variant is not None else "baseline",
        split=config.split,
        seed=config.seed,
        gpu_index=config.gpu_index,
        artifact_root=config.artifact_root,
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        compile_policy=config.compile_policy,
        training=config.training,
        quantization_variant=config.quantization_variant,
    )


def run_baseline_task(config: RuntimeConfig) -> int:
    run_single(config)
    return 0


def run_qat_task(config: RuntimeConfig) -> int:
    run_single(config)
    return 0


def run_eval_task(config: RuntimeConfig) -> int:
    run_single(_config_for_eval_target(config))
    return 0


def _run_matrix(config: RuntimeConfig) -> int:
    for child in build_matrix(config):
        run_single(child)
    return 0


def run_smoke_task(config: RuntimeConfig) -> int:
    return _run_matrix(config)


def run_full_task(config: RuntimeConfig) -> int:
    return _run_matrix(config)
