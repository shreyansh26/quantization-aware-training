import pytest
import torch
from torch import nn

from qat.config import QuantizationVariant
from qat.quantization.qat import (
    FakeQuantLinear,
    convert_model_from_qat,
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
