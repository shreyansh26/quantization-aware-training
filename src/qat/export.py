import json
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.utils import calculate_qparams
from torch import nn

from qat.config import RunManifest, RuntimeConfig, artifact_dir_for_run
from qat.quantization.qat import (
    convert_model_from_qat,
    default_linear_filter,
    get_qat_spec,
)
from qat.train.baseline import compile_model_for_training, load_baseline_tokenizer


@dataclass(frozen=True)
class ExportResult:
    artifact_dir: str
    manifest_path: str
    compile_status: str
    completeness_status: str
    source_checkpoint_dir: str | None
    quantization_variant: str | None


def load_checkpoint_manifest(checkpoint_dir: Path) -> RunManifest:
    payload = json.loads((checkpoint_dir / "manifest.json").read_text())
    field_names = {item.name for item in fields(RunManifest)}
    filtered = {key: value for key, value in payload.items() if key in field_names}
    return RunManifest(**filtered)


def resolve_export_artifact_dir(config: RuntimeConfig) -> Path:
    return artifact_dir_for_run(config)


def verify_export_completeness(artifact_dir: Path) -> list[str]:
    missing: list[str] = []
    if not (artifact_dir / "config.json").exists():
        missing.append("config.json")
    if not any(
        (artifact_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    ):
        missing.append("model-weights")
    if not any(
        (artifact_dir / filename).exists()
        for filename in ("tokenizer.json", "tokenizer_config.json")
    ):
        missing.append("tokenizer")
    if not (artifact_dir / "manifest.json").exists():
        missing.append("manifest.json")
    return missing


def _load_hf_model_and_tokenizer_from_checkpoint(
    checkpoint_dir: Path,
    config: RuntimeConfig,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir / "model",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = load_baseline_tokenizer(config)
    tokenizer_source = checkpoint_dir / "tokenizer"
    if tokenizer_source.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _activation_quant_args(spec) -> QuantizationArgs | None:  # noqa: ANN001
    """Translate local QAT activation settings into compressed-tensors metadata."""

    if spec.activation_dtype == "bf16":
        return None
    if spec.activation_dtype == "fp8":
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TOKEN,
            dynamic=True,
        )
    if spec.activation_dtype == "int8":
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            symmetric=False,
            strategy=QuantizationStrategy.TOKEN,
            dynamic=True,
        )
    raise ValueError(f"unsupported activation dtype: {spec.activation_dtype}")


def _weight_quant_args(spec) -> QuantizationArgs:  # noqa: ANN001
    """Translate local QAT weight settings into compressed-tensors metadata."""

    if spec.weight_dtype == "fp8":
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL,
        )
    if spec.weight_dtype == "int8":
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            symmetric=True,
            strategy=QuantizationStrategy.CHANNEL,
        )
    if spec.weight_dtype == "int4":
        return QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP,
            group_size=spec.group_size,
        )
    raise ValueError(f"unsupported weight dtype: {spec.weight_dtype}")


def _compression_format_for_variant(config: RuntimeConfig) -> CompressionFormat:
    """Choose the serialized compression format expected for the variant."""

    variant = config.quantization_variant
    if variant is None:
        return CompressionFormat.dense
    if variant == variant.FP8_BF16:
        return CompressionFormat.naive_quantized
    if variant == variant.FP8_FP8:
        return CompressionFormat.float_quantized
    if variant == variant.INT8_INT8:
        return CompressionFormat.int_quantized
    return CompressionFormat.pack_quantized


def _min_max_for_weight(
    weight: torch.Tensor,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect per-tensor, per-channel, or per-group ranges for a weight matrix."""

    if args.strategy == QuantizationStrategy.TENSOR:
        min_vals = weight.amin().reshape(1)
        max_vals = weight.amax().reshape(1)
    elif args.strategy == QuantizationStrategy.CHANNEL:
        min_vals = weight.amin(dim=-1, keepdim=True)
        max_vals = weight.amax(dim=-1, keepdim=True)
    elif args.strategy == QuantizationStrategy.GROUP:
        assert args.group_size is not None
        if weight.shape[-1] % args.group_size != 0:
            raise ValueError(
                "weight shape is not divisible by the configured group size"
            )
        grouped = weight.reshape(*weight.shape[:-1], -1, args.group_size)
        min_vals = grouped.amin(dim=-1)
        max_vals = grouped.amax(dim=-1)
    else:
        raise ValueError(f"unsupported weight strategy: {args.strategy}")
    return min_vals, max_vals


def _attach_weight_qparams(module: nn.Linear, args: QuantizationArgs) -> None:
    """Populate the module buffers that compressed-tensors reads during export."""

    min_vals, max_vals = _min_max_for_weight(module.weight.detach(), args)
    # compressed_tensors.calculate_qparams takes the observed min/max range and
    # converts it into the affine quantization parameters that the export path
    # expects: one scale per tensor/channel/group plus an optional zero point.
    #
    # In the symmetric INT4/INT8 cases used here this yields zero-centered ranges
    # with zero_point == 0. In asymmetric cases, it would compute a nonzero zero
    # point so that the observed floating-point interval maps into the integer
    # range. The helper also guarantees that 0.0 stays representable and clamps
    # degenerate scales away from exact zero.
    scales, zero_points = calculate_qparams(min_vals, max_vals, args)
    module.weight_scale.data.copy_(scales.to(module.weight_scale.dtype))
    if hasattr(module, "weight_zero_point"):
        module.weight_zero_point.data.copy_(
            zero_points.to(module.weight_zero_point.dtype)
        )


def _attach_quantization_metadata(model: nn.Module, config: RuntimeConfig) -> None:
    """Attach the quant buffers and scheme metadata needed before compression."""

    if config.quantization_variant is None:
        return
    spec = get_qat_spec(config.quantization_variant)
    compression_format = _compression_format_for_variant(config)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=_weight_quant_args(spec),
        input_activations=_activation_quant_args(spec),
        format=compression_format,
    )
    for name, module in model.named_modules():
        if not default_linear_filter(name, module):
            continue
        # compressed_tensors.initialize_module_for_quantization does the module
        # bookkeeping that its compressor expects before it can rewrite weights:
        #
        # 1. it clears any existing qparam buffers on the module
        # 2. it allocates fresh buffers such as weight_scale / weight_zero_point
        #    and any activation-side metadata implied by the scheme
        # 3. it stores quantization_scheme / quantization_status on the module
        # 4. it wraps the forward path with compressed-tensors quantization hooks
        #
        # In this repo we only rely on a narrow subset of that behavior for
        # export-time preparation of nn.Linear layers. We do not use the wrapped
        # forward path for training here, but we do need the allocated buffers
        # and attached scheme metadata before ModelCompressor.compress_model().
        initialize_module_for_quantization(module, scheme)
        # initialize_module_for_quantization allocates the qparam buffers, but it
        # does not populate the weight scales from our trained floating-point
        # weights. We do that explicitly from the current weight tensor so the
        # compressor sees concrete channel/group statistics during export.
        _attach_weight_qparams(module, scheme.weights)


def _save_with_compressed_tensors_adapter(
    model: nn.Module,
    artifact_dir: Path,
    *,
    config: RuntimeConfig,
) -> None:
    """Save either a dense model or a compressed-tensors quantized artifact."""

    if config.quantization_variant is None:
        model.save_pretrained(artifact_dir, safe_serialization=True)
        return

    # The export flow is:
    # 1. attach quantization metadata/buffers to the plain Linear modules
    # 2. let ModelCompressor rewrite/compress the weight tensors in-place
    # 3. save the rewritten model as a standalone Hugging Face artifact
    # 4. update the saved config with the compressed-tensors metadata required by
    #    downstream loaders such as vLLM
    _attach_quantization_metadata(model, config)
    compressor = ModelCompressor.from_pretrained_model(model)
    compressor.compress_model(model)
    model.save_pretrained(artifact_dir, safe_serialization=True)
    compressor.update_config(str(artifact_dir))


def _write_export_manifest(
    artifact_dir: Path,
    manifest: RunManifest,
    *,
    checkpoint_dir: Path | None,
    compile_status: str,
    completeness_status: str,
) -> Path:
    payload = manifest.to_dict()
    payload["export"] = {
        "source_checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "compile_status": compile_status,
        "completeness_status": completeness_status,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def probe_exported_model_compile(
    artifact_dir: Path,
    config: RuntimeConfig,
    *,
    max_length: int = 16,
) -> str:
    if config.compile_policy.value == "disabled":
        return "disabled"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model, status = compile_model_for_training(model, config.compile_policy)
    encoded = tokenizer(
        "2 + 2 =",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        model(**encoded)
    return status


def export_model_artifact(
    config: RuntimeConfig,
    *,
    checkpoint_dir: Path | None = None,
    model: nn.Module | None = None,
    tokenizer: Any | None = None,
    source_manifest: RunManifest | None = None,
) -> ExportResult:
    artifact_dir = resolve_export_artifact_dir(config)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{artifact_dir.name}-", dir=str(artifact_dir.parent))
    )
    try:
        if checkpoint_dir is not None:
            model, tokenizer = _load_hf_model_and_tokenizer_from_checkpoint(
                checkpoint_dir,
                config,
            )
            source_manifest = source_manifest or load_checkpoint_manifest(
                checkpoint_dir
            )
        else:
            if model is None or tokenizer is None:
                raise ValueError(
                    "export requires either a checkpoint_dir or both model "
                    "and tokenizer"
                )
        model = convert_model_from_qat(model)
        _save_with_compressed_tensors_adapter(model, temp_dir, config=config)
        tokenizer.save_pretrained(temp_dir)

        compile_status = probe_exported_model_compile(temp_dir, config)
        manifest = source_manifest or RunManifest(
            run_id=artifact_dir.name,
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
            split_manifest_path=None,
            artifact_dir=str(artifact_dir),
            compile_policy=config.compile_policy.value,
            seed=config.seed,
        )
        _write_export_manifest(
            temp_dir,
            manifest,
            checkpoint_dir=checkpoint_dir,
            compile_status=compile_status,
            completeness_status="pending",
        )
        missing = verify_export_completeness(temp_dir)
        completeness_status = (
            "complete" if not missing else f"missing:{','.join(missing)}"
        )
        _write_export_manifest(
            temp_dir,
            manifest,
            checkpoint_dir=checkpoint_dir,
            compile_status=compile_status,
            completeness_status=completeness_status,
        )
        if missing:
            raise ValueError(
                f"exported artifact is incomplete: {', '.join(missing)}"
            )
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        os.replace(temp_dir, artifact_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return ExportResult(
        artifact_dir=str(artifact_dir),
        manifest_path=str(artifact_dir / "manifest.json"),
        compile_status=compile_status,
        completeness_status=completeness_status,
        source_checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        quantization_variant=(
            config.quantization_variant.value
            if config.quantization_variant is not None
            else None
        ),
    )
