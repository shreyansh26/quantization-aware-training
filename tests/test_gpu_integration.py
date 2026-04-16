import json
import os
from pathlib import Path

import pytest
import torch

from qat.config import (
    CompilePolicy,
    QuantizationVariant,
    RunMode,
    RuntimeConfig,
    SplitConfig,
    TrainingConfig,
)
from qat.eval.vllm_runner import verify_vllm_loadability
from qat.export import export_model_artifact
from qat.train.baseline import train_baseline
from qat.train.qat import train_qat

RUN_GPU_TESTS = os.environ.get("RUN_QAT_GPU_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_GPU_TESTS,
    reason="set RUN_QAT_GPU_TESTS=1 to run GPU integration coverage",
)


def _tiny_runtime_config(
    artifact_root: Path,
    *,
    mode: RunMode,
    variant: QuantizationVariant | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        split=SplitConfig(name="smoke", train_size=2, test_size=1, seed=17),
        mode=mode,
        artifact_root=artifact_root,
        compile_policy=CompilePolicy.DISABLED,
        training=TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        ),
        quantization_variant=variant,
    )


def _write_tiny_split(path: Path) -> None:
    path.write_text(
        json.dumps({"train_indices": [0, 1], "test_indices": [2]}, indent=2)
    )


@pytest.mark.gpu
def test_gpu_baseline_export_loadability(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    work_dir = tmp_path / "baseline"
    work_dir.mkdir(parents=True)
    split_manifest = work_dir / "split.json"
    _write_tiny_split(split_manifest)
    config = _tiny_runtime_config(tmp_path, mode=RunMode.BASELINE)
    summary = train_baseline(
        config,
        split_manifest_path=split_manifest,
        checkpoint_dir=work_dir / "checkpoint",
        max_length=256,
        max_steps=1,
    )
    export_result = export_model_artifact(
        config,
        checkpoint_dir=Path(summary.checkpoint_dir),
    )
    loadability = verify_vllm_loadability(Path(export_result.artifact_dir), config)
    assert loadability.loaded


@pytest.mark.gpu
def test_gpu_qat_int8_bf16_export_loadability(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    work_dir = tmp_path / "qat-int8-bf16"
    work_dir.mkdir(parents=True)
    split_manifest = work_dir / "split.json"
    _write_tiny_split(split_manifest)
    config = _tiny_runtime_config(
        tmp_path,
        mode=RunMode.QAT,
        variant=QuantizationVariant.INT8_BF16,
    )
    summary = train_qat(
        config,
        split_manifest_path=split_manifest,
        checkpoint_dir=work_dir / "checkpoint",
        max_length=256,
        max_steps=1,
    )
    export_result = export_model_artifact(
        config,
        checkpoint_dir=Path(summary.checkpoint_dir),
    )
    loadability = verify_vllm_loadability(Path(export_result.artifact_dir), config)
    assert loadability.loaded
