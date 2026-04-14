from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET_ID = "AI-MO/NuminaMath-CoT"
DATASET_REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
ARTIFACT_ROOT = Path("/mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat")


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
