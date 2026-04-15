from __future__ import annotations

import argparse

from qat.config import (
    DEFAULT_METRICS_OUTPUT,
    CompilePolicy,
    RunMode,
    RuntimeConfig,
    TrainingConfig,
    get_split_config,
    parse_variant,
)
from qat.runner import evaluate_model, train_and_export


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--type",
        required=True,
        choices=("smoke", "full"),
        help="Named data split to use.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(RunMode.BASELINE.value, RunMode.QAT.value),
        help="Run baseline or qat mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Seed used for split generation and run naming.",
    )
    parser.add_argument(
        "--quantization-variant",
        default=None,
        help="Required for qat mode. Omit for baseline mode.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qat.cli")
    subparsers = parser.add_subparsers(dest="task", required=True)

    train_parser = subparsers.add_parser("train")
    _add_common_args(train_parser)
    train_parser.add_argument(
        "--training.learning-rate",
        dest="training_learning_rate",
        type=float,
        default=2e-5,
    )
    train_parser.add_argument(
        "--training.weight-decay",
        dest="training_weight_decay",
        type=float,
        default=0.01,
    )
    train_parser.add_argument(
        "--training.warmup-ratio",
        dest="training_warmup_ratio",
        type=float,
        default=0.03,
    )
    train_parser.add_argument(
        "--training.max-grad-norm",
        dest="training_max_grad_norm",
        type=float,
        default=1.0,
    )
    train_parser.add_argument(
        "--training.num-epochs",
        dest="training_num_epochs",
        type=int,
        default=1,
    )
    train_parser.add_argument(
        "--compile",
        choices=[policy.value for policy in CompilePolicy],
        default=CompilePolicy.DISABLED.value,
        help="Training compile policy.",
    )

    eval_parser = subparsers.add_parser("eval")
    _add_common_args(eval_parser)
    eval_parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Explicit trained model path. Overrides path resolution from "
            "mode/type/seed/variant."
        ),
    )
    eval_parser.add_argument(
        "--output-path",
        default=str(DEFAULT_METRICS_OUTPUT),
        help="Metrics CSV path. Predictions are written next to it.",
    )
    return parser


def _runtime_config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    mode = RunMode(args.mode)
    variant = parse_variant(args.quantization_variant)
    training = TrainingConfig(
        learning_rate=getattr(args, "training_learning_rate", 2e-5),
        weight_decay=getattr(args, "training_weight_decay", 0.01),
        warmup_ratio=getattr(args, "training_warmup_ratio", 0.03),
        max_grad_norm=getattr(args, "training_max_grad_norm", 1.0),
        num_epochs=getattr(args, "training_num_epochs", 1),
    )
    return RuntimeConfig(
        split=get_split_config(args.type, seed=args.seed),
        mode=mode,
        seed=args.seed,
        compile_policy=CompilePolicy(getattr(args, "compile", "disabled")),
        training=training,
        quantization_variant=variant,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _runtime_config_from_args(args)
    if args.task == "train":
        train_and_export(config)
        return 0
    evaluate_model(
        config,
        model_path=args.model_path,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
