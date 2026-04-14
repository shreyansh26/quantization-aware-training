from __future__ import annotations

import argparse
import importlib.metadata
import sys
from dataclasses import dataclass

from qat.config import QuantizationVariant, VariantMetadata, get_variant_metadata

REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "vllm",
    "compressed-tensors",
)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


def check_python_version() -> PreflightCheck:
    version = sys.version_info
    ok = version >= (3, 12)
    detail = f"found Python {version.major}.{version.minor}.{version.micro}"
    if not ok:
        detail = f"{detail}; require Python 3.12+"
    return PreflightCheck(name="python", ok=ok, detail=detail)


def check_required_packages() -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for package_name in REQUIRED_PACKAGES:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                PreflightCheck(
                    name=f"package:{package_name}",
                    ok=False,
                    detail="not installed in the active environment",
                )
            )
            continue
        checks.append(
            PreflightCheck(
                name=f"package:{package_name}",
                ok=True,
                detail=f"version {version}",
            )
        )
    return checks


def check_variant_support(
    variant: QuantizationVariant | None,
) -> list[PreflightCheck]:
    if variant is None:
        return [
            PreflightCheck(
                name="variant",
                ok=True,
                detail="baseline run selected",
            )
        ]
    metadata = get_variant_metadata(variant)
    assert metadata is not None
    checks = [
        PreflightCheck(
            name="variant",
            ok=True,
            detail=(
                f"{variant.value} -> {metadata.serving_scheme} "
                f"({metadata.weight_dtype}/{metadata.activation_dtype})"
            ),
        )
    ]
    if (
        metadata.weight_dtype == "int4"
        and metadata.activation_dtype == "int8"
    ):
        checks.append(
            PreflightCheck(
                name="variant:int4_int8",
                ok=False,
                detail="int4/int8 is explicitly unsupported for this project",
            )
        )
    return checks


def _fp8_required(metadata: VariantMetadata | None) -> bool:
    if metadata is None:
        return False
    return "fp8" in {metadata.weight_dtype, metadata.activation_dtype}


def check_cuda(
    *,
    gpu_index: int,
    variant: QuantizationVariant | None,
) -> list[PreflightCheck]:
    try:
        import torch
    except ImportError:
        return [
            PreflightCheck(
                name="cuda",
                ok=False,
                detail="torch is not importable, cannot validate CUDA state",
            )
        ]

    checks: list[PreflightCheck] = []
    if not torch.cuda.is_available():
        return [
            PreflightCheck(
                name="cuda",
                ok=False,
                detail="CUDA is not available through torch",
            )
        ]

    device_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= device_count:
        return [
            PreflightCheck(
                name="cuda:device",
                ok=False,
                detail=(
                    f"requested GPU {gpu_index} is outside the visible range "
                    f"0..{device_count - 1}"
                ),
            )
        ]

    device_name = torch.cuda.get_device_name(gpu_index)
    is_h100 = "H100" in device_name.upper()
    checks.append(
        PreflightCheck(
            name="cuda:device",
            ok=is_h100,
            detail=f"GPU {gpu_index}: {device_name}",
        )
    )

    metadata = get_variant_metadata(variant)
    if _fp8_required(metadata):
        major, minor = torch.cuda.get_device_capability(gpu_index)
        checks.append(
            PreflightCheck(
                name="cuda:fp8",
                ok=(major, minor) >= (9, 0),
                detail=f"compute capability {major}.{minor}",
            )
        )
    return checks


def run_preflight(
    *,
    gpu_index: int,
    variant: QuantizationVariant | None,
) -> list[PreflightCheck]:
    checks = [check_python_version()]
    checks.extend(check_required_packages())
    checks.extend(check_variant_support(variant))
    checks.extend(check_cuda(gpu_index=gpu_index, variant=variant))
    return checks


def format_report(checks: list[PreflightCheck]) -> str:
    lines = []
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qat.preflight")
    parser.add_argument("--gpu-index", type=int, default=5)
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in QuantizationVariant],
        default=None,
    )
    args = parser.parse_args(argv)

    variant = (
        QuantizationVariant(args.variant) if args.variant is not None else None
    )
    checks = run_preflight(gpu_index=args.gpu_index, variant=variant)
    print(format_report(checks))
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
