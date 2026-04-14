from argparse import Namespace

from qat.cli import TASK_HANDLERS, build_parser, config_from_args
from qat.config import CompilePolicy, QuantizationVariant
from qat.runner import (
    run_baseline_task,
    run_eval_task,
    run_full_task,
    run_qat_task,
    run_smoke_task,
)


def test_cli_builds_runtime_config_for_qat() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "qat",
            "--variant",
            "int8_bf16",
            "--split",
            "smoke",
            "--seed",
            "21",
            "--gpu-index",
            "5",
            "--compile-policy",
            "required",
        ]
    )
    config = config_from_args(args)
    assert config.task == "qat"
    assert config.quantization_variant == QuantizationVariant.INT8_BF16
    assert config.seed == 21
    assert config.gpu_index == 5
    assert config.compile_policy == CompilePolicy.REQUIRED


def test_cli_defaults_split_by_task() -> None:
    parser = build_parser()
    full_args = parser.parse_args(["full"])
    smoke_args = parser.parse_args(["baseline"])
    assert isinstance(full_args, Namespace)
    assert full_args.split == "full"
    assert smoke_args.split == "smoke"


def test_cli_task_handlers_are_runner_wired() -> None:
    assert TASK_HANDLERS["baseline"] is run_baseline_task
    assert TASK_HANDLERS["qat"] is run_qat_task
    assert TASK_HANDLERS["eval"] is run_eval_task
    assert TASK_HANDLERS["smoke"] is run_smoke_task
    assert TASK_HANDLERS["full"] is run_full_task
