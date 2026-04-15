from argparse import Namespace

from qat.cli import build_parser
from qat.config import QuantizationVariant


def test_cli_builds_train_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--type",
            "smoke",
            "--mode",
            "qat",
            "--seed",
            "21",
            "--quantization-variant",
            "int8_bf16",
            "--training.learning-rate",
            "1e-5",
            "--compile",
            "required",
        ]
    )
    assert isinstance(args, Namespace)
    assert args.task == "train"
    assert args.type == "smoke"
    assert args.mode == "qat"
    assert args.seed == 21
    assert args.quantization_variant == QuantizationVariant.INT8_BF16.value
    assert args.training_learning_rate == 1e-5
    assert args.compile == "required"


def test_cli_builds_eval_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--type",
            "full",
            "--mode",
            "baseline",
            "--seed",
            "17",
            "--output-path",
            "/tmp/metrics.csv",
        ]
    )
    assert args.task == "eval"
    assert args.output_path == "/tmp/metrics.csv"
    assert args.model_path is None
