from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace

import pytest

from qat.config import QuantizationVariant
from qat.preflight import check_required_packages, format_report, main, run_preflight


def test_check_required_packages_reports_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(package_name: str) -> str:
        if package_name == "vllm":
            raise importlib.metadata.PackageNotFoundError
        return "1.0.0"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    checks = check_required_packages()
    vllm_check = next(check for check in checks if check.name == "package:vllm")
    assert not vllm_check.ok
    assert "not installed" in vllm_check.detail


def test_run_preflight_rejects_int4_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0.0")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "NVIDIA H100 80GB HBM3",
            get_device_capability=lambda _: (9, 0),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    checks = run_preflight(variant=None)
    assert any(check.ok for check in checks)


def test_run_preflight_requires_capability_90_for_fp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0.0")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "NVIDIA A100 80GB PCIe",
            get_device_capability=lambda _: (8, 0),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    checks = run_preflight(
        variant=QuantizationVariant.FP8_FP8,
    )
    assert any(check.name == "cuda:device" and not check.ok for check in checks)
    assert any(check.name == "cuda:fp8" and not check.ok for check in checks)


def test_run_preflight_accepts_non_h100_with_capability_90(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0.0")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "Future Hopper-class GPU",
            get_device_capability=lambda _: (9, 0),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    checks = run_preflight(
        variant=QuantizationVariant.FP8_FP8,
    )
    assert any(check.name == "cuda:device" and check.ok for check in checks)
    assert any(check.name == "cuda:fp8" and check.ok for check in checks)


def test_format_report_contains_pass_and_fail() -> None:
    report = format_report(
        checks=[
            SimpleNamespace(name="one", ok=True, detail="alpha"),
            SimpleNamespace(name="two", ok=False, detail="beta"),
        ]
    )
    assert "[PASS] one: alpha" in report
    assert "[FAIL] two: beta" in report


def test_preflight_main_passes_for_supported_variant() -> None:
    exit_code = main(["--variant", "fp8_fp8"])
    assert exit_code == 0
