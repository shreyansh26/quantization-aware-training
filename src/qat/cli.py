from __future__ import annotations

import argparse
from collections.abc import Callable

from qat.config import (
    CompilePolicy,
    RuntimeConfig,
    default_runtime_config,
    parse_variant,
)

Handler = Callable[[RuntimeConfig], int]


def _not_implemented(task_name: str) -> Handler:
    def _runner(config: RuntimeConfig) -> int:
        raise NotImplementedError(
            f"Task '{task_name}' is not wired yet. Config: {config}"
        )

    return _runner


TASK_HANDLERS: dict[str, Handler] = {
    "baseline": _not_implemented("baseline"),
    "qat": _not_implemented("qat"),
    "eval": _not_implemented("eval"),
    "smoke": _not_implemented("smoke"),
    "full": _not_implemented("full"),
}


def _default_split_for_task(task_name: str) -> str:
    if task_name == "full":
        return "full"
    return "smoke"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qat.cli")
    subparsers = parser.add_subparsers(dest="task", required=True)
    for task_name in TASK_HANDLERS:
        task_parser = subparsers.add_parser(task_name)
        task_parser.add_argument(
            "--variant",
            default=None,
            help="Quantization variant, e.g. int8_bf16. Omit for baseline-style runs.",
        )
        task_parser.add_argument(
            "--split",
            default=_default_split_for_task(task_name),
            choices=("smoke", "full"),
            help="Named data split to use.",
        )
        task_parser.add_argument(
            "--seed",
            type=int,
            default=17,
            help="Global run seed used for split generation and runtime determinism.",
        )
        task_parser.add_argument(
            "--gpu-index",
            type=int,
            default=5,
            help="Physical GPU index to target for single-GPU runs.",
        )
        task_parser.add_argument(
            "--compile-policy",
            default=CompilePolicy.TRY.value,
            choices=tuple(policy.value for policy in CompilePolicy),
            help="Compile policy for eager/try/required execution modes.",
        )
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = default_runtime_config(args.task, split_name=args.split)
    variant = parse_variant(args.variant)
    return RuntimeConfig(
        task=config.task,
        split=config.split,
        seed=args.seed,
        gpu_index=args.gpu_index,
        artifact_root=config.artifact_root,
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        compile_policy=CompilePolicy(args.compile_policy),
        training=config.training,
        quantization_variant=variant,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    handler = TASK_HANDLERS[args.task]
    return handler(config)


if __name__ == "__main__":
    raise SystemExit(main())
