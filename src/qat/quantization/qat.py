from dataclasses import dataclass
from typing import Callable, Self

import torch
import torch.nn.functional as F
from torch import nn

from qat.config import QuantizationVariant, parse_variant

ModuleFilter = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class QATSpec:
    variant: QuantizationVariant
    weight_dtype: str
    activation_dtype: str
    weight_granularity: str
    activation_granularity: str
    group_size: int | None = None


def get_qat_spec(variant: QuantizationVariant | str) -> QATSpec:
    parsed = parse_variant(variant)
    assert parsed is not None
    specs = {
        QuantizationVariant.FP8_BF16: QATSpec(
            variant=parsed,
            weight_dtype="fp8",
            activation_dtype="bf16",
            weight_granularity="per_row",
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
            weight_granularity="per_row",
            activation_granularity="per_row",
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
            activation_granularity="per_row",
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
    return original + (quantized - original).detach()


def _reshape_groups(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
        )
    shape = x.shape
    grouped = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    return grouped, shape


def fake_quantize_int(
    x: torch.Tensor,
    *,
    bits: int,
    granularity: str,
    group_size: int | None = None,
    symmetric: bool = True,
) -> torch.Tensor:
    qmin = -(2 ** (bits - 1)) if symmetric else 0
    qmax = (2 ** (bits - 1)) - 1 if symmetric else (2**bits) - 1

    if granularity == "per_token":
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
    elif granularity == "per_channel":
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
    else:
        raise ValueError(f"unsupported integer granularity: {granularity}")

    if symmetric:
        scale = torch.maximum(x_min.abs(), x_max.abs()) / float(qmax)
        zero_point = torch.zeros_like(scale)
    else:
        scale = (x_max - x_min).clamp_min(1e-8) / float(qmax - qmin)
        zero_point = qmin - torch.round(x_min / scale)
    scale = scale.clamp_min(1e-8)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    dq = (q - zero_point) * scale
    return _ste(x, dq)


def fake_quantize_fp8(x: torch.Tensor, *, granularity: str) -> torch.Tensor:
    if granularity == "none":
        return x
    if granularity not in {"per_row", "per_tensor"}:
        raise ValueError(f"unsupported fp8 granularity: {granularity}")

    try:
        dq = x.to(torch.float8_e4m3fn).to(x.dtype)
    except RuntimeError:
        scale_dims = None if granularity == "per_tensor" else (-1,)
        max_abs = x.abs().amax(dim=scale_dims, keepdim=granularity == "per_row")
        max_abs = max_abs.clamp_min(1e-8)
        dq = (x / max_abs).clamp(-1, 1) * max_abs
    return _ste(x, dq)


def apply_activation_fake_quant(x: torch.Tensor, spec: QATSpec) -> torch.Tensor:
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
        return cls(linear, spec)

    def to_linear(self) -> nn.Linear:
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
        qx = apply_activation_fake_quant(x, self.spec)
        qw = apply_weight_fake_quant(self.weight, self.spec)
        return F.linear(qx, qw, self.bias)


def default_linear_filter(name: str, module: nn.Module) -> bool:
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
    for name, child in list(model.named_children()):
        if isinstance(child, FakeQuantLinear):
            setattr(model, name, child.to_linear())
            continue
        convert_model_from_qat(child)
    return model
