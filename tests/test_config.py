from qat.config import (
    RunMode,
    RuntimeConfig,
    SplitConfig,
    artifact_dir_for_run,
    dump_json,
    get_split_config,
    launch_config_payload,
    make_manifest,
    make_run_id,
    parse_variant,
)


def test_parse_variant_rejects_unsupported_variant() -> None:
    try:
        parse_variant("int4_int8")
    except ValueError as exc:
        assert "Unsupported quantization variant" in str(exc)
    else:
        raise AssertionError("expected parse_variant to reject int4_int8")


def test_make_manifest_contains_artifact_dir() -> None:
    config = RuntimeConfig(
        split=get_split_config("smoke"),
        mode=RunMode.QAT,
        quantization_variant=parse_variant("int8_bf16"),
    )
    manifest = make_manifest(config)
    assert manifest.artifact_dir.endswith("qat-smoke-int8_bf16-seed17")


def test_make_run_id_uses_baseline_when_variant_is_missing() -> None:
    config = RuntimeConfig(
        split=SplitConfig("smoke", 2, 1, 17),
        mode=RunMode.BASELINE,
    )
    assert make_run_id(config) == "baseline-smoke-baseline-seed17"


def test_artifact_dir_for_run_is_stable() -> None:
    config = RuntimeConfig(
        split=get_split_config("full"),
        mode=RunMode.BASELINE,
    )
    artifact_dir = artifact_dir_for_run(config)
    assert artifact_dir.name == "baseline-full-baseline-seed17"


def test_runtime_config_rejects_baseline_with_variant() -> None:
    try:
        RuntimeConfig(
            split=get_split_config("smoke"),
            mode=RunMode.BASELINE,
            quantization_variant=parse_variant("int8_bf16"),
        )
    except ValueError as exc:
        assert "baseline mode" in str(exc)
    else:
        raise AssertionError("expected baseline config to reject variant")


def test_dump_json_writes_payload(tmp_path) -> None:  # noqa: ANN001
    config = RuntimeConfig(
        split=get_split_config("smoke"),
        mode=RunMode.BASELINE,
    )
    path = tmp_path / "config.json"
    dump_json(path, launch_config_payload(config))
    assert path.exists()
