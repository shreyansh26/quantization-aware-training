from __future__ import annotations

import importlib.metadata
from collections.abc import Callable

import torch

VLLM_W4A8_FP8_PATCH_ENV = "QAT_PATCH_VLLM_W4A8_FP8_VIEW"


def reshape_channel_scales(
    chan_scales: torch.Tensor,
    orig_shape: torch.Size,
) -> torch.Tensor:
    return chan_scales.view((*orig_shape[:-1], -1))


def _convert_bf16_scales_to_fp8(
    quant_fp8: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert scales.is_contiguous(), (
        f"scale tensor must be contiguous, got {scales.stride()=}"
    )
    assert scales.is_cuda, "scales must be on gpu"

    orig_shape = scales.shape
    k_groups = orig_shape[-1]
    flat_scales = scales.view(-1, k_groups)

    fp8_scales, chan_scales = quant_fp8(flat_scales)
    fp8_scales = (fp8_scales.float() / 8.0).to(torch.float8_e4m3fn)
    chan_scales *= 8.0

    return fp8_scales.view(orig_shape), reshape_channel_scales(
        chan_scales,
        orig_shape,
    )


def patch_vllm_w4a8_fp8_scale_view() -> None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return
    if version != "0.19.0":
        return

    try:
        from vllm.model_executor.kernels.linear.mixed_precision import cutlass
        from vllm.model_executor.layers.quantization.utils import quant_utils
    except (AttributeError, ImportError):
        return

    quant_utils.convert_bf16_scales_to_fp8 = _convert_bf16_scales_to_fp8
    cutlass.convert_bf16_scales_to_fp8 = _convert_bf16_scales_to_fp8
