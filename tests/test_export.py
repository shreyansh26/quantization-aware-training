import json

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_qparams
from torch import nn

from qat.config import CompilePolicy, RunMode, RuntimeConfig, SplitConfig
from qat.export import (
    _min_max_for_weight,
    export_model_artifact,
    load_checkpoint_manifest,
    probe_exported_model_compile,
    verify_export_completeness,
)
from qat.quantization.qat import _calculate_qparams


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


def test_probe_exported_model_compile_uses_runtime_device(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    config = RuntimeConfig(
        split=SplitConfig(name="smoke", train_size=2, test_size=1, seed=17),
        mode=RunMode.BASELINE,
        artifact_root=tmp_path,
        compile_policy=CompilePolicy.REQUIRED,
    )
    device = torch.device("cuda")

    class FakeTensor:
        def __init__(self) -> None:
            self.device = None

        def to(self, target_device):  # noqa: ANN001
            self.device = target_device
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.input_ids = FakeTensor()
            self.attention_mask = FakeTensor()

        def __call__(self, *args, **kwargs):  # noqa: ANN001
            return {
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
            }

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.encoded = None

        def to(self, target_device):  # noqa: ANN001
            self.device = target_device
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):  # noqa: ANN003
            self.encoded = encoded

    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr("qat.export.runtime_device", lambda: device)
    monkeypatch.setattr(
        "qat.export.compile_model_for_training",
        lambda model, compile_policy: (model, "compiled"),
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )

    status = probe_exported_model_compile(tmp_path, config)

    assert status == "compiled"
    assert fake_model.device == device
    assert fake_model.encoded == {
        "input_ids": fake_tokenizer.input_ids,
        "attention_mask": fake_tokenizer.attention_mask,
    }
    assert fake_tokenizer.input_ids.device == device
    assert fake_tokenizer.attention_mask.device == device


def test_local_qparams_match_compressed_tensors_for_supported_weight_schemes() -> None:
    weight = torch.tensor(
        [
            [0.83, -0.41, 0.06, -0.12],
            [-0.24, 0.71, -0.35, 0.18],
        ],
        dtype=torch.bfloat16,
    )
    args_list = [
        QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            symmetric=True,
            strategy=QuantizationStrategy.CHANNEL,
        ),
        QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP,
            group_size=2,
        ),
        QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL,
        ),
    ]

    for args in args_list:
        min_vals, max_vals = _min_max_for_weight(weight, args)
        expected_scales, expected_zero_points = calculate_qparams(
            min_vals,
            max_vals,
            args,
        )
        actual_scales, actual_zero_points = _calculate_qparams(
            min_vals,
            max_vals,
            bits=args.num_bits,
            dtype="float" if args.type == QuantizationType.FLOAT else "int",
            symmetric=args.symmetric if args.symmetric is not None else True,
        )
        torch.testing.assert_close(actual_scales, expected_scales)
        torch.testing.assert_close(
            actual_zero_points.to(expected_zero_points.dtype),
            expected_zero_points,
        )
