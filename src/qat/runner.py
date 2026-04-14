from __future__ import annotations

import csv
import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qat.config import (
    DEFAULT_METRICS_OUTPUT,
    RuntimeConfig,
    artifact_dir_for_run,
    dump_json,
    launch_config_payload,
    make_run_id,
    split_manifest_path,
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


@dataclass(frozen=True)
class TrainResult:
    artifact_dir: str
    split_manifest_path: str
    launch_config_path: str
    compile_status: str


@dataclass(frozen=True)
class EvaluationSummary:
    decisions: list[EvaluationDecision]
    metrics_row: dict[str, Any]
    prediction_log_path: str
    metrics_path: str


def _run_preflight_or_raise(config: RuntimeConfig) -> None:
    checks = run_preflight(variant=config.quantization_variant)
    failures = [check for check in checks if not check.ok]
    if failures:
        raise RuntimeError(format_report(checks))


def ensure_split_manifest(config: RuntimeConfig) -> Path:
    path = split_manifest_path(config)
    if path.exists():
        return path
    dataset = load_numinamath_train_dataset(revision=config.dataset_revision)
    manifest = build_split_manifest(dataset, config.split)
    save_split_manifest(manifest, path)
    return path


def _temp_checkpoint_root(config: RuntimeConfig) -> Path:
    return config.artifact_root / ".tmp" / make_run_id(config)


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


def _dump_train_config(config: RuntimeConfig, artifact_dir: Path) -> Path:
    path = artifact_dir / "train_config.json"
    dump_json(path, launch_config_payload(config))
    return path


def _dump_eval_config(
    config: RuntimeConfig,
    *,
    output_path: Path,
    model_path: Path,
) -> Path:
    path = output_path.parent / f"eval_config_{model_path.name}.json"
    payload = launch_config_payload(config)
    payload["model_path"] = str(model_path)
    payload["output_path"] = str(output_path)
    dump_json(path, payload)
    return path


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


def _prediction_log_path(output_path: Path, model_path: Path) -> Path:
    return output_path.parent / f"predictions_{model_path.name}.json"


def evaluate_exported_model(
    config: RuntimeConfig,
    *,
    artifact_dir: Path,
    split_manifest: Path,
    output_path: Path = DEFAULT_METRICS_OUTPUT,
) -> EvaluationSummary:
    from transformers import AutoTokenizer

    split_payload = json.loads(split_manifest.read_text())
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
    prediction_log = _prediction_log_path(output_path, artifact_dir)
    write_prediction_log(prediction_log, decisions)
    accuracy = sum(decision.is_correct for decision in decisions) / max(
        1,
        len(decisions),
    )
    metrics_row = make_metrics_row(
        model_name=config.model_id,
        quantization_artifact=artifact_dir.name,
        variant=config.quantization_variant,
        metric_name="accuracy",
        metric_value=accuracy,
    )
    append_metrics_once(output_path, metrics_row)
    _dump_eval_config(config, output_path=output_path, model_path=artifact_dir)
    return EvaluationSummary(
        decisions=decisions,
        metrics_row=metrics_row,
        prediction_log_path=str(prediction_log),
        metrics_path=str(output_path),
    )


def train_and_export(config: RuntimeConfig) -> TrainResult:
    _run_preflight_or_raise(config)
    split_manifest = ensure_split_manifest(config)
    checkpoint_root = _temp_checkpoint_root(config)
    checkpoint_dir = checkpoint_root / "checkpoint"
    trainer = train_baseline if config.mode.value == "baseline" else train_qat

    try:
        trainer(
            config,
            split_manifest_path=split_manifest,
            checkpoint_dir=checkpoint_dir,
        )
        _release_gpu_memory()
        export_result = export_model_artifact(
            config,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        _release_gpu_memory()
        shutil.rmtree(checkpoint_root, ignore_errors=True)

    artifact_dir = Path(export_result.artifact_dir)
    launch_config_path = _dump_train_config(config, artifact_dir)
    return TrainResult(
        artifact_dir=str(artifact_dir),
        split_manifest_path=str(split_manifest),
        launch_config_path=str(launch_config_path),
        compile_status=export_result.compile_status,
    )


def resolve_model_path(
    config: RuntimeConfig,
    *,
    model_path: str | Path | None = None,
) -> Path:
    if model_path is not None:
        return Path(model_path)
    return artifact_dir_for_run(config)


def evaluate_model(
    config: RuntimeConfig,
    *,
    model_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_METRICS_OUTPUT,
) -> EvaluationSummary:
    artifact_dir = resolve_model_path(config, model_path=model_path)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"model path does not exist: {artifact_dir}")
    _run_preflight_or_raise(config)
    split_manifest = ensure_split_manifest(config)
    _release_gpu_memory()
    loadability = verify_vllm_loadability(artifact_dir, config)
    if not loadability.loaded:
        raise RuntimeError(loadability.error or "vLLM loadability check failed")
    _release_gpu_memory()
    return evaluate_exported_model(
        config,
        artifact_dir=artifact_dir,
        split_manifest=split_manifest,
        output_path=Path(output_path),
    )
