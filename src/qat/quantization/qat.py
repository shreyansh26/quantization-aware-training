from dataclasses import dataclass
from typing import Callable, Self

import torch
import torch.nn.functional as F
from torch import nn

from qat.config import QuantizationVariant, parse_variant

ModuleFilter = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class QATSpec:
    """Quantization settings used to wrap a train-time fake-quant linear layer."""

    variant: QuantizationVariant
    weight_dtype: str
    activation_dtype: str
    weight_granularity: str
    activation_granularity: str
    group_size: int | None = None


def get_qat_spec(variant: QuantizationVariant | str) -> QATSpec:
    """Map a configured variant to the fake-quant settings used during training."""

    parsed = parse_variant(variant)
    assert parsed is not None
    specs = {
        QuantizationVariant.FP8_BF16: QATSpec(
            variant=parsed,
            weight_dtype="fp8",
            activation_dtype="bf16",
            weight_granularity="per_channel",
            activation_granularity="none",
        ),
        QuantizationVariant.INT8_BF16: QATSpec(
            variant=parsed,
            weight_dtype="int8",
            activation_dtype="bf16",
            weight_granularity="per_channel",
            activation_granularity="none",
        ),
        QuantizationVariant.FP8_FP8: QATSpec(
            variant=parsed,
            weight_dtype="fp8",
            activation_dtype="fp8",
            weight_granularity="per_channel",
            activation_granularity="per_token",
        ),
        QuantizationVariant.INT8_INT8: QATSpec(
            variant=parsed,
            weight_dtype="int8",
            activation_dtype="int8",
            weight_granularity="per_channel",
            activation_granularity="per_token",
        ),
        QuantizationVariant.INT4_FP8: QATSpec(
            variant=parsed,
            weight_dtype="int4",
            activation_dtype="fp8",
            weight_granularity="per_group",
            activation_granularity="per_token",
            group_size=128,
        ),
        QuantizationVariant.INT4_BF16: QATSpec(
            variant=parsed,
            weight_dtype="int4",
            activation_dtype="bf16",
            weight_granularity="per_group",
            activation_granularity="none",
            group_size=128,
        ),
    }
    return specs[parsed]


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Return quantized values in forward while keeping identity gradients."""

    return original + (quantized - original).detach()


def _reshape_groups(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Split the last dimension into fixed-size groups for grouped weight quant."""

    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
        )
    shape = x.shape
    grouped = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    return grouped, shape


def _calculate_range(
    *,
    bits: int,
    dtype: str,
    device: torch.device,
) -> tuple[float, float]:
    """Match the quantized value range conventions used by compressed-tensors."""

    if dtype == "int":
        bit_range = 2**bits
        return -float(bit_range / 2), float(bit_range / 2 - 1)
    if dtype == "float":
        if bits != 8:
            raise NotImplementedError("only fp8 is supported in the local QAT path")
        return (
            float(torch.finfo(torch.float8_e4m3fn).min),
            float(torch.finfo(torch.float8_e4m3fn).max),
        )
    raise ValueError(f"unsupported quantized dtype: {dtype}")


def _calculate_qparams(
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    *,
    bits: int,
    dtype: str,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the compressed-tensors min/max -> scale/zero-point calculation."""

    min_vals = torch.minimum(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.maximum(max_vals, torch.zeros_like(max_vals))
    qmin, qmax = _calculate_range(bits=bits, dtype=dtype, device=min_vals.device)
    bit_range = qmax - qmin

    if symmetric:
        max_val_pos = torch.maximum(min_vals.abs(), max_vals.abs())
        scales = max_val_pos / (bit_range / 2.0)
        zero_points = torch.zeros_like(scales)
    else:
        scales = (max_vals - min_vals) / bit_range
        scales = scales.clamp_min(1e-8)
        zero_points = qmin - (min_vals / scales)
        zero_points = torch.clamp(torch.round(zero_points), qmin, qmax)

    eps = torch.finfo(scales.dtype).eps
    scales = torch.where(
        scales == 0,
        torch.tensor(eps, dtype=scales.dtype, device=scales.device),
        scales,
    )
    if symmetric:
        zero_points = torch.round(zero_points)
    return scales, zero_points


def _qdq(
    x: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    bits: int,
    dtype: str,
) -> torch.Tensor:
    """Quantize then dequantize with compressed-tensors-compatible rules."""

    qmin, qmax = _calculate_range(bits=bits, dtype=dtype, device=x.device)
    scaled = x / scale
    if zero_point is not None:
        scaled = scaled + zero_point.to(x.dtype)

    scaled = torch.clamp(scaled, qmin, qmax)
    if dtype == "int":
        quantized = torch.round(scaled)
    elif dtype == "float":
        quantized = scaled.to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"unsupported quantized dtype: {dtype}")

    quantized = quantized.to(x.dtype)
    if zero_point is not None:
        quantized = quantized - zero_point.to(x.dtype)
    return quantized * scale


def _compute_dynamic_qparams(
    x: torch.Tensor,
    *,
    bits: int,
    dtype: str,
    granularity: str,
    symmetric: bool,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dynamic qparams using the same reduction rules as compressed-tensors."""

    if granularity == "per_token":
        reduce_dims = tuple(index for index in range(x.ndim) if index not in {0, 1})
        if not reduce_dims:
            min_vals, max_vals = torch.aminmax(x)
        else:
            min_vals = torch.amin(x, dim=reduce_dims, keepdim=True)
            max_vals = torch.amax(x, dim=reduce_dims, keepdim=True)
        return _calculate_qparams(
            min_vals,
            max_vals,
            bits=bits,
            dtype=dtype,
            symmetric=symmetric,
        )
    if granularity == "per_tensor":
        min_vals, max_vals = torch.aminmax(x)
        return _calculate_qparams(
            min_vals,
            max_vals,
            bits=bits,
            dtype=dtype,
            symmetric=symmetric,
        )
    if granularity == "per_group":
        assert group_size is not None
        grouped, _ = _reshape_groups(x, group_size)
        min_vals = torch.amin(grouped, dim=-1)
        max_vals = torch.amax(grouped, dim=-1)
        return _calculate_qparams(
            min_vals,
            max_vals,
            bits=bits,
            dtype=dtype,
            symmetric=symmetric,
        )
    raise ValueError(f"unsupported dynamic granularity: {granularity}")


def fake_quantize_int(
    x: torch.Tensor,
    *,
    bits: int,
    granularity: str,
    group_size: int | None = None,
    symmetric: bool = True,
) -> torch.Tensor:
    """Fake-quantize integer tensors and dequantize back into the original dtype.

    The returned tensor stays in floating point for training, but the values are
    snapped to the integer grid implied by `bits`, `granularity`, and `symmetric`.
    """

    if granularity == "per_channel":
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
    elif granularity == "per_group":
        assert group_size is not None
        grouped, shape = _reshape_groups(x, group_size)
        grouped_qdq = fake_quantize_int(
            grouped,
            bits=bits,
            granularity="per_token",
            symmetric=symmetric,
        )
        return _ste(x, grouped_qdq.reshape(shape))
    elif granularity == "per_token":
        scales, zero_points = _compute_dynamic_qparams(
            x,
            bits=bits,
            dtype="int",
            granularity=granularity,
            symmetric=symmetric,
        )
        dq = _qdq(
            x,
            scale=scales,
            zero_point=zero_points,
            bits=bits,
            dtype="int",
        )
        return _ste(x, dq)
    else:
        raise ValueError(f"unsupported integer granularity: {granularity}")

    scale, zero_point = _calculate_qparams(
        x_min,
        x_max,
        bits=bits,
        dtype="int",
        symmetric=symmetric,
    )
    dq = _qdq(
        x,
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        dtype="int",
    )
    return _ste(x, dq)


def fake_quantize_fp8(x: torch.Tensor, *, granularity: str) -> torch.Tensor:
    """Fake-quantize FP8 tensors using compressed-tensors-style scaled QDQ."""

    if granularity == "none":
        return x
    if granularity not in {"per_row", "per_channel", "per_tensor", "per_token"}:
        raise ValueError(f"unsupported fp8 granularity: {granularity}")

    if granularity in {"per_token", "per_tensor"}:
        scales, zero_points = _compute_dynamic_qparams(
            x,
            bits=8,
            dtype="float",
            granularity=granularity,
            symmetric=True,
        )
    else:
        reduce_dims = None if granularity == "per_tensor" else (-1,)
        keepdim = granularity in {"per_row", "per_channel"}
        min_vals = torch.amin(x, dim=reduce_dims, keepdim=keepdim)
        max_vals = torch.amax(x, dim=reduce_dims, keepdim=keepdim)
        scales, zero_points = _calculate_qparams(
            min_vals,
            max_vals,
            bits=8,
            dtype="float",
            symmetric=True,
        )
    dq = _qdq(
        x,
        scale=scales,
        zero_point=zero_points,
        bits=8,
        dtype="float",
    )
    return _ste(x, dq)


def apply_activation_fake_quant(x: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """Apply the activation side of the configured QAT scheme."""

    if spec.activation_dtype == "bf16":
        return x
    if spec.activation_dtype == "fp8":
        return fake_quantize_fp8(x, granularity=spec.activation_granularity)
    if spec.activation_dtype == "int8":
        return fake_quantize_int(
            x,
            bits=8,
            granularity=spec.activation_granularity,
            symmetric=False,
        )
    raise ValueError(f"unsupported activation dtype: {spec.activation_dtype}")


def apply_weight_fake_quant(weight: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """Apply the weight side of the configured QAT scheme."""

    if spec.weight_dtype == "fp8":
        return fake_quantize_fp8(weight, granularity=spec.weight_granularity)
    if spec.weight_dtype == "int8":
        return fake_quantize_int(
            weight,
            bits=8,
            granularity=spec.weight_granularity,
            symmetric=True,
        )
    if spec.weight_dtype == "int4":
        return fake_quantize_int(
            weight,
            bits=4,
            granularity=spec.weight_granularity,
            group_size=spec.group_size,
            symmetric=True,
        )
    raise ValueError(f"unsupported weight dtype: {spec.weight_dtype}")


class FakeQuantLinear(nn.Module):
    """Linear layer wrapper that fake-quantizes activations and weights in forward."""

    def __init__(self, linear: nn.Linear, spec: QATSpec) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.spec = spec
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, spec: QATSpec) -> Self:
        """Clone an existing Linear into a QAT wrapper with the same parameters."""

        return cls(linear, spec)

    def to_linear(self) -> nn.Linear:
        """Materialize a plain Linear for checkpoint export or serving conversion."""

        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight = nn.Parameter(self.weight.detach().clone())
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.detach().clone())
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear projection using fake-quantized activations and weights."""

        qx = apply_activation_fake_quant(x, self.spec)
        qw = apply_weight_fake_quant(self.weight, self.spec)
        return F.linear(qx, qw, self.bias)


def default_linear_filter(name: str, module: nn.Module) -> bool:
    """Select transformer Linear layers while leaving lm_head untouched."""

    if not isinstance(module, nn.Linear):
        return False
    if "lm_head" in name:
        return False
    return True


def prepare_model_for_qat(
    model: nn.Module,
    variant: QuantizationVariant | str,
    *,
    module_filter: ModuleFilter | None = None,
    prefix: str = "",
) -> nn.Module:
    """Replace eligible Linear modules with FakeQuantLinear recursively."""

    spec = get_qat_spec(variant)
    filter_fn = module_filter or default_linear_filter
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if filter_fn(full_name, child):
            setattr(model, name, FakeQuantLinear.from_linear(child, spec))
            continue
        prepare_model_for_qat(
            child,
            spec.variant,
            module_filter=filter_fn,
            prefix=full_name,
        )
    return model


def convert_model_from_qat(model: nn.Module) -> nn.Module:
    """Strip FakeQuantLinear wrappers back to plain Linear modules."""

    for name, child in list(model.named_children()):
        if isinstance(child, FakeQuantLinear):
            setattr(model, name, child.to_linear())
            continue
        convert_model_from_qat(child)
    return model
