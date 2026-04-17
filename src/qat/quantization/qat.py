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
    """Return the fake-quantized forward value while preserving float gradients.

    `quantized` here is slightly overloaded: in this file it is the output of a
    quantize-dequantize (QDQ) step, not a packed INT4/INT8 tensor that can be
    consumed directly by an inference kernel.

    That distinction matters:
    - during QAT training we still run ordinary floating-point PyTorch ops
    - so we quantize onto the target grid and immediately dequantize back to a
      float tensor with the quantization error baked in
    - the forward pass therefore sees values that behave like quantized values,
      but are represented as normal float tensors

    The STE then makes autograd treat this as identity with respect to the
    original float tensor. Forward uses the dequantized, grid-snapped values;
    backward updates the original float weights/activations.
    """

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
    """Mirror the compressed-tensors min/max -> scale/zero-point calculation.

    Symmetric vs asymmetric here follows standard affine quantization rules:

    - symmetric:
      used for weights in this repo. We quantize around zero and keep
      `zero_point == 0`. This is a natural fit for model weights because they
      are usually distributed around zero, and many weight-only kernels assume
      or prefer zero-centered quantization.

    - asymmetric:
      used for INT8 activations in this repo. Activations can be shifted away
      from zero, so allowing a learned zero-point gives better coverage of the
      observed min/max interval.

    FP8 also reuses this helper for its zero-centered scale computation. That
    does not mean FP8 is following the integer affine path end to end: the real
    divergence happens later in `_qdq()`, where integers use `round(...)` onto
    an integer grid while FP8 uses a cast onto the FP8 floating-point grid.

    We also force 0.0 into the representable range before computing qparams.
    That matches compressed-tensors and avoids pathological cases where the
    affine mapping would exclude exact zero.
    """

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
    """Quantize then dequantize with compressed-tensors-compatible rules.

    QAT does not run a true integer/fp8 matmul in forward. Instead it simulates
    the quantization error that inference will see:

    1. divide by scale and add zero-point
    2. clamp to the target quantized range
    3. round/cast onto that grid
    4. dequantize back to the original floating-point dtype

    The result is still a float tensor, but its values are exactly those that
    would come back from a quantize -> dequantize round trip. That is what lets
    the training forward pass "feel" quantization error while still using normal
    PyTorch kernels and autograd.

    Actual inference is different:
    - weights are exported in compressed form by compressed-tensors
    - activations are quantized dynamically by the serving runtime
    - inference kernels then operate on those runtime quantized values
    """

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

    This is why the QAT forward pass uses dequantized values rather than packed
    integer tensors: training wants the quantization noise, not a separate
    integer execution engine. Export/inference is where we turn the learned float
    weights into a true compressed INT8/INT4 artifact.
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
    """Fake-quantize FP8 tensors using compressed-tensors-style scaled QDQ.

    As with the integer path, the output is dequantized back to float for the
    training forward pass. The values lie on the FP8 grid, but the tensor is not
    stored/executed as a persistent FP8 parameter inside the training model.

    This path intentionally reuses `_calculate_qparams(..., symmetric=True)` to
    get a zero-centered scale. The quantizer itself is still different from the
    integer path: `_qdq()` casts onto the FP8 lattice instead of rounding onto
    an integer lattice with affine zero-point handling.
    """

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
    """Apply the activation side of the configured QAT scheme.

    Activation policy in this repo:
    - `bf16`: no activation quantization
    - `fp8`: scaled FP8 fake quant
    - `int8`: asymmetric fake quant

    We use asymmetric INT8 activations because runtime activations are
    input-dependent and can be shifted away from zero. Allowing a nonzero
    zero-point gives better coverage of the observed range than forcing a
    symmetric, zero-centered interval.
    """

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
    """Apply the weight side of the configured QAT scheme.

    Weight policy in this repo:
    - `fp8`: scaled FP8 fake quant
    - `int8`: symmetric per-channel fake quant
    - `int4`: symmetric per-group fake quant

    Integer weights stay symmetric because model weights are generally centered
    around zero and the exported compressed formats also use zero-centered weight
    quantization. This keeps train-time fake quant aligned with export-time
    compression semantics.
    """

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
    """Linear layer wrapper that fake-quantizes activations and weights in forward.

    Important mental model:
    - `self.weight` remains the trainable floating-point parameter
    - each forward pass produces a fake-quantized/dequantized view of that weight
      and, optionally, of the input activations
    - the matmul therefore runs on float tensors whose values have been snapped
      to the target quantized grid

    This is the standard QAT compromise:
    - training keeps float parameters and normal autograd
    - forward simulates inference-time quantization error
    - export later materializes the true compressed weight representation
    """

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
        """Run linear projection using fake-quantized activations and weights.

        `qx` and `qw` are QDQ outputs, not packed quantized tensors. They are
        floating-point tensors whose values have been snapped onto the target
        quantization grid. That is why `F.linear` can consume them directly
        during training.

        Inference uses a different path: export serializes compressed weights,
        and the inference runtime performs the actual quantized execution.
        """

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
