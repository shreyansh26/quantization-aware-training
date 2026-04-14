from __future__ import annotations

import json

from torch import nn

from qat.config import CompilePolicy, RunMode, RuntimeConfig, SplitConfig
from qat.export import (
    export_model_artifact,
    load_checkpoint_manifest,
    verify_export_completeness,
)


class TinyTokenizer:
    def save_pretrained(self, save_directory):  # noqa: ANN001
        import os

        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/tokenizer.json", "w", encoding="utf-8") as handle:
            handle.write("{}")


class TinyExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def save_pretrained(self, save_directory, safe_serialization=True):  # noqa: ANN001
        import os

        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/config.json", "w", encoding="utf-8") as handle:
            json.dump({"architectures": ["TinyExportModel"]}, handle)
        with open(f"{save_directory}/model.safetensors", "wb") as handle:
            handle.write(b"weights")


def test_verify_export_completeness_reports_missing_files(tmp_path) -> None:
    missing = verify_export_completeness(tmp_path)
    assert missing == ["config.json", "model-weights", "tokenizer", "manifest.json"]


def test_load_checkpoint_manifest_round_trips(tmp_path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    payload = {
        "run_id": "demo",
        "mode": "baseline",
        "split_name": "smoke",
        "quantization_variant": None,
        "model_id": "demo/model",
        "model_revision": "rev",
        "dataset_id": "demo/dataset",
        "dataset_revision": "ds-rev",
        "split_manifest_path": None,
        "artifact_dir": str(tmp_path / "artifact"),
        "compile_policy": "disabled",
        "seed": 17,
        "package_versions": {},
        "git_sha": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (checkpoint_dir / "manifest.json").write_text(json.dumps(payload))
    manifest = load_checkpoint_manifest(checkpoint_dir)
    assert manifest.run_id == "demo"
    assert manifest.mode == "baseline"


def test_export_model_artifact_writes_final_directory(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = RuntimeConfig(
        split=SplitConfig(name="smoke", train_size=2, test_size=1, seed=17),
        mode=RunMode.BASELINE,
        artifact_root=tmp_path,
        compile_policy=CompilePolicy.DISABLED,
    )
    monkeypatch.setattr(
        "qat.export.probe_exported_model_compile",
        lambda artifact_dir, config: "disabled",
    )
    result = export_model_artifact(
        config,
        model=TinyExportModel(),
        tokenizer=TinyTokenizer(),
    )
    artifact_dir = tmp_path / "baseline-smoke-baseline-seed17"
    assert result.artifact_dir == str(artifact_dir)
    assert (artifact_dir / "config.json").exists()
    assert (artifact_dir / "tokenizer.json").exists()
    assert (artifact_dir / "manifest.json").exists()
