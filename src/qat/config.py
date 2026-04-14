from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET_ID = "AI-MO/NuminaMath-CoT"
DATASET_REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
ARTIFACT_ROOT = Path("/mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat")
DEFAULT_METRICS_OUTPUT = ARTIFACT_ROOT / "metrics_numinamath_cot.csv"

SMOKE_TRAIN_SIZE = 500
SMOKE_TEST_SIZE = 100
FULL_TRAIN_SIZE = 5000
FULL_TEST_SIZE = 500


class RunType(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


class RunMode(StrEnum):
    BASELINE = "baseline"
    QAT = "qat"


class QuantizationVariant(StrEnum):
    FP8_BF16 = "fp8_bf16"
    INT8_BF16 = "int8_bf16"
    FP8_FP8 = "fp8_fp8"
    INT8_INT8 = "int8_int8"
    INT4_FP8 = "int4_fp8"
    INT4_BF16 = "int4_bf16"


SUPPORTED_VARIANTS = frozenset(variant for variant in QuantizationVariant)
UNSUPPORTED_VARIANTS = frozenset({"int4_int8"})


@dataclass(frozen=True)
class VariantMetadata:
    weight_dtype: str
    activation_dtype: str
    serving_scheme: str
    source: str
    notes: str


VARIANT_METADATA = {
    QuantizationVariant.FP8_BF16: VariantMetadata(
        weight_dtype="fp8",
        activation_dtype="bf16",
        serving_scheme="torchao_fp8_weight_only",
        source="TorchAO",
        notes=(
            "Reference-only training variant; export path must remain "
            "vLLM-compatible."
        ),
    ),
    QuantizationVariant.INT8_BF16: VariantMetadata(
        weight_dtype="int8",
        activation_dtype="bf16",
        serving_scheme="W8A16",
        source="compressed_tensors",
        notes="Supported broadly as 8-bit weights with 16-bit activations.",
    ),
    QuantizationVariant.FP8_FP8: VariantMetadata(
        weight_dtype="fp8",
        activation_dtype="fp8",
        serving_scheme="W8A8-FP8",
        source="compressed_tensors",
        notes="Recommended high-throughput Hopper path.",
    ),
    QuantizationVariant.INT8_INT8: VariantMetadata(
        weight_dtype="int8",
        activation_dtype="int8",
        serving_scheme="W8A8-INT8",
        source="compressed_tensors",
        notes="Supported on Turing and later.",
    ),
    QuantizationVariant.INT4_FP8: VariantMetadata(
        weight_dtype="int4",
        activation_dtype="fp8",
        serving_scheme="W4AFP8",
        source="compressed_tensors",
        notes="Documented Hopper-target mixed-precision path.",
    ),
    QuantizationVariant.INT4_BF16: VariantMetadata(
        weight_dtype="int4",
        activation_dtype="bf16",
        serving_scheme="W4A16",
        source="compressed_tensors",
        notes="Supported on Turing and later as weight-only INT4.",
    ),
}


class CompilePolicy(StrEnum):
    DISABLED = "disabled"
    TRY = "try"
    REQUIRED = "required"


@dataclass(frozen=True)
class SplitConfig:
    name: str
    train_size: int
    test_size: int
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    split: SplitConfig
    mode: RunMode
    seed: int = 17
    artifact_root: Path = ARTIFACT_ROOT
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dataset_id: str = DATASET_ID
    dataset_revision: str = DATASET_REVISION
    compile_policy: CompilePolicy = CompilePolicy.DISABLED
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization_variant: QuantizationVariant | None = None

    def __post_init__(self) -> None:
        if self.mode == RunMode.BASELINE and self.quantization_variant is not None:
            raise ValueError(
                "baseline mode does not accept a quantization variant"
            )
        if self.mode == RunMode.QAT and self.quantization_variant is None:
            raise ValueError("qat mode requires a quantization variant")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    mode: str
    split_name: str
    quantization_variant: str | None
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    split_manifest_path: str | None
    artifact_dir: str
    compile_policy: str
    seed: int
    package_versions: dict[str, str] = field(default_factory=dict)
    git_sha: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_variant(
    value: QuantizationVariant | str | None,
) -> QuantizationVariant | None:
    if value is None:
        return None
    if isinstance(value, QuantizationVariant):
        return value
    if value in UNSUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported quantization variant: {value}")
    try:
        return QuantizationVariant(value)
    except ValueError as exc:
        supported = ", ".join(sorted(variant.value for variant in SUPPORTED_VARIANTS))
        raise ValueError(
            f"Unknown quantization variant '{value}'. Supported variants: {supported}"
        ) from exc


def get_variant_metadata(
    variant: QuantizationVariant | str | None,
) -> VariantMetadata | None:
    parsed = parse_variant(variant)
    if parsed is None:
        return None
    return VARIANT_METADATA[parsed]


def get_split_config(name: RunType | str, *, seed: int = 17) -> SplitConfig:
    split_name = RunType(name).value
    if split_name == RunType.SMOKE.value:
        return SplitConfig(
            name=split_name,
            train_size=SMOKE_TRAIN_SIZE,
            test_size=SMOKE_TEST_SIZE,
            seed=seed,
        )
    if split_name == RunType.FULL.value:
        return SplitConfig(
            name=split_name,
            train_size=FULL_TRAIN_SIZE,
            test_size=FULL_TEST_SIZE,
            seed=seed,
        )
    raise ValueError(f"Unknown split '{name}'")


def make_run_id(config: RuntimeConfig) -> str:
    variant_name = (
        config.quantization_variant.value
        if config.quantization_variant is not None
        else "baseline"
    )
    return f"{config.mode.value}-{config.split.name}-{variant_name}-seed{config.seed}"


def artifact_dir_for_run(config: RuntimeConfig) -> Path:
    return config.artifact_root / make_run_id(config)


def split_manifest_path(config: RuntimeConfig) -> Path:
    revision_prefix = config.dataset_revision[:8]
    filename = (
        f"numinamath_cot-{config.split.name}-seed{config.seed}-{revision_prefix}.json"
    )
    return config.artifact_root / "splits" / filename


def launch_config_payload(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "type": config.split.name,
        "mode": config.mode.value,
        "seed": config.seed,
        "quantization_variant": (
            config.quantization_variant.value
            if config.quantization_variant is not None
            else None
        ),
        "training": asdict(config.training),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "compile_policy": config.compile_policy.value,
        "artifact_root": str(config.artifact_root),
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_manifest(
    config: RuntimeConfig,
    *,
    split_manifest_path_value: Path | None = None,
    package_versions: dict[str, str] | None = None,
    git_sha: str | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=make_run_id(config),
        mode=config.mode.value,
        split_name=config.split.name,
        quantization_variant=(
            config.quantization_variant.value
            if config.quantization_variant is not None
            else None
        ),
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        split_manifest_path=(
            str(split_manifest_path_value) if split_manifest_path_value else None
        ),
        artifact_dir=str(artifact_dir_for_run(config)),
        compile_policy=config.compile_policy.value,
        seed=config.seed,
        package_versions=package_versions or {},
        git_sha=git_sha,
    )
