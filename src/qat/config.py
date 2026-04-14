from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET_ID = "AI-MO/NuminaMath-CoT"
DATASET_REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
ARTIFACT_ROOT = Path("/mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat")

SMOKE_TRAIN_SIZE = 500
SMOKE_TEST_SIZE = 100
FULL_TRAIN_SIZE = 5000
FULL_TEST_SIZE = 500


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
    task: str
    split: SplitConfig
    seed: int = 17
    gpu_index: int = 5
    artifact_root: Path = ARTIFACT_ROOT
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dataset_id: str = DATASET_ID
    dataset_revision: str = DATASET_REVISION
    compile_policy: CompilePolicy = CompilePolicy.TRY
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization_variant: QuantizationVariant | None = None

    def with_variant(self, variant: QuantizationVariant | str | None) -> RuntimeConfig:
        parsed = parse_variant(variant)
        return RuntimeConfig(
            task=self.task,
            split=self.split,
            seed=self.seed,
            gpu_index=self.gpu_index,
            artifact_root=self.artifact_root,
            model_id=self.model_id,
            model_revision=self.model_revision,
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            compile_policy=self.compile_policy,
            training=self.training,
            quantization_variant=parsed,
        )


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    task: str
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
    gpu_index: int
    package_versions: dict[str, str] = field(default_factory=dict)
    git_sha: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    )
    resume_fingerprint: str = ""

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


def get_split_config(name: str, *, seed: int = 17) -> SplitConfig:
    split_map = {
        "smoke": SplitConfig(
            name="smoke",
            train_size=SMOKE_TRAIN_SIZE,
            test_size=SMOKE_TEST_SIZE,
            seed=seed,
        ),
        "full": SplitConfig(
            name="full",
            train_size=FULL_TRAIN_SIZE,
            test_size=FULL_TEST_SIZE,
            seed=seed,
        ),
    }
    try:
        return split_map[name]
    except KeyError as exc:
        valid = ", ".join(sorted(split_map))
        raise ValueError(f"Unknown split '{name}'. Expected one of: {valid}") from exc


def make_run_id(
    *,
    task: str,
    split: SplitConfig,
    variant: QuantizationVariant | str | None,
    seed: int,
) -> str:
    parsed = parse_variant(variant)
    variant_name = parsed.value if parsed is not None else "baseline"
    return f"{task}-{split.name}-{variant_name}-seed{seed}"


def artifact_dir_for_run(config: RuntimeConfig) -> Path:
    run_id = make_run_id(
        task=config.task,
        split=config.split,
        variant=config.quantization_variant,
        seed=config.seed,
    )
    return config.artifact_root / run_id


def make_resume_fingerprint(config: RuntimeConfig) -> str:
    payload = {
        "task": config.task,
        "split": asdict(config.split),
        "seed": config.seed,
        "gpu_index": config.gpu_index,
        "artifact_root": str(config.artifact_root),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "compile_policy": config.compile_policy.value,
        "quantization_variant": (
            config.quantization_variant.value
            if config.quantization_variant is not None
            else None
        ),
        "training": asdict(config.training),
    }
    fingerprint = sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
    return fingerprint


def make_manifest(
    config: RuntimeConfig,
    *,
    split_manifest_path: Path | None = None,
    package_versions: dict[str, str] | None = None,
    git_sha: str | None = None,
) -> RunManifest:
    run_id = make_run_id(
        task=config.task,
        split=config.split,
        variant=config.quantization_variant,
        seed=config.seed,
    )
    return RunManifest(
        run_id=run_id,
        task=config.task,
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
            str(split_manifest_path) if split_manifest_path is not None else None
        ),
        artifact_dir=str(artifact_dir_for_run(config)),
        compile_policy=config.compile_policy.value,
        seed=config.seed,
        gpu_index=config.gpu_index,
        package_versions=package_versions or {},
        git_sha=git_sha,
        resume_fingerprint=make_resume_fingerprint(config),
    )


def default_runtime_config(task: str, *, split_name: str) -> RuntimeConfig:
    return RuntimeConfig(task=task, split=get_split_config(split_name))
