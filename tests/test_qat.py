import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import (
    compute_dynamic_scales_and_zp,
)
from compressed_tensors.quantization.lifecycle.forward import (
    fake_quantize as ct_fake_quantize,
)
from compressed_tensors.quantization.utils import calculate_qparams
from torch import nn

from qat.config import QuantizationVariant
from qat.quantization.qat import (
    FakeQuantLinear,
    convert_model_from_qat,
    fake_quantize_fp8,
    fake_quantize_int,
    get_qat_spec,
    prepare_model_for_qat,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 16)
        self.mlp = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(x))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock()])
        self.lm_head = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        return self.lm_head(x)


def _weight_qparams_from_args(
    weight: torch.Tensor,
    *,
    args: QuantizationArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.strategy == QuantizationStrategy.CHANNEL:
        min_vals = weight.amin(dim=-1, keepdim=True)
        max_vals = weight.amax(dim=-1, keepdim=True)
    elif args.strategy == QuantizationStrategy.GROUP:
        assert args.group_size is not None
        grouped = weight.reshape(*weight.shape[:-1], -1, args.group_size)
        min_vals = grouped.amin(dim=-1)
        max_vals = grouped.amax(dim=-1)
    else:
        raise ValueError(f"unsupported test strategy: {args.strategy}")
    return calculate_qparams(min_vals, max_vals, args)


def test_prepare_model_for_qat_replaces_transformer_linears_only() -> None:
    model = TinyModel()
    prepare_model_for_qat(model, QuantizationVariant.INT8_BF16)
    assert isinstance(model.layers[0].proj, FakeQuantLinear)
    assert isinstance(model.layers[0].mlp, FakeQuantLinear)
    assert isinstance(model.lm_head, nn.Linear)


def test_fake_quant_linear_preserves_shape() -> None:
    linear = nn.Linear(128, 16)
    module = FakeQuantLinear.from_linear(
        linear,
        get_qat_spec(QuantizationVariant.INT4_BF16),
    )
    output = module(torch.randn(2, 128))
    assert output.shape == (2, 16)


def test_convert_model_from_qat_restores_linear_modules() -> None:
    model = TinyModel()
    prepare_model_for_qat(model, QuantizationVariant.FP8_FP8)
    convert_model_from_qat(model)
    assert isinstance(model.layers[0].proj, nn.Linear)
    assert isinstance(model.layers[0].mlp, nn.Linear)


def test_qat_modules_are_compile_friendly() -> None:
    model = TinyBlock()
    prepare_model_for_qat(model, QuantizationVariant.INT8_INT8)
    compiled = torch.compile(model)
    output = compiled(torch.randn(2, 16))
    assert output.shape == (2, 16)


def test_int4_weight_group_size_must_divide_input_dimension() -> None:
    linear = nn.Linear(30, 8)
    module = FakeQuantLinear.from_linear(
        linear,
        get_qat_spec(QuantizationVariant.INT4_BF16),
    )
    with pytest.raises(ValueError):
        module(torch.randn(2, 30))


def test_int8_weight_fake_quant_matches_compressed_tensors() -> None:
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
    )
    scales, zero_points = _weight_qparams_from_args(weight, args=args)
    expected = ct_fake_quantize(weight, scale=scales, zero_point=zero_points, args=args)
    actual = fake_quantize_int(
        weight,
        bits=8,
        granularity="per_channel",
        symmetric=True,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_int4_weight_fake_quant_matches_compressed_tensors() -> None:
    weight = torch.randn(8, 128, dtype=torch.bfloat16)
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
    )
    scales, zero_points = _weight_qparams_from_args(weight, args=args)
    expected = ct_fake_quantize(weight, scale=scales, zero_point=zero_points, args=args)
    actual = fake_quantize_int(
        weight,
        bits=4,
        granularity="per_group",
        group_size=128,
        symmetric=True,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_int8_activation_fake_quant_matches_compressed_tensors() -> None:
    activation = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=False,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )
    scales, zero_points = compute_dynamic_scales_and_zp(
        value=activation,
        args=args,
        module=nn.Identity(),
    )
    expected = ct_fake_quantize(
        activation,
        scale=scales,
        zero_point=zero_points,
        args=args,
    )
    actual = fake_quantize_int(
        activation,
        bits=8,
        granularity="per_token",
        symmetric=False,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fp8_weight_fake_quant_matches_compressed_tensors() -> None:
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
    )
    scales, zero_points = _weight_qparams_from_args(weight, args=args)
    expected = ct_fake_quantize(weight, scale=scales, zero_point=zero_points, args=args)
    actual = fake_quantize_fp8(weight, granularity="per_channel")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fp8_activation_fake_quant_matches_compressed_tensors() -> None:
    activation = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )
    scales, zero_points = compute_dynamic_scales_and_zp(
        value=activation,
        args=args,
        module=nn.Identity(),
    )
    expected = ct_fake_quantize(
        activation,
        scale=scales,
        zero_point=zero_points,
        args=args,
    )
    actual = fake_quantize_fp8(activation, granularity="per_token")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
