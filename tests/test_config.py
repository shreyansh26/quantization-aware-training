from qat.config import (
    QuantizationVariant,
    artifact_dir_for_run,
    default_runtime_config,
    get_split_config,
    make_manifest,
    make_resume_fingerprint,
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


def test_manifest_contains_resume_fingerprint_and_artifact_dir() -> None:
    config = default_runtime_config("qat", split_name="smoke").with_variant(
        QuantizationVariant.INT8_BF16
    )
    manifest = make_manifest(config)
    assert manifest.resume_fingerprint == make_resume_fingerprint(config)
    assert manifest.artifact_dir.endswith("qat-smoke-int8_bf16-seed17")


def test_make_run_id_uses_baseline_when_variant_is_missing() -> None:
    split = get_split_config("smoke")
    run_id = make_run_id(task="baseline", split=split, variant=None, seed=17)
    assert run_id == "baseline-smoke-baseline-seed17"


def test_artifact_dir_for_run_is_stable() -> None:
    config = default_runtime_config("eval", split_name="full")
    artifact_dir = artifact_dir_for_run(config)
    assert artifact_dir.name == "eval-full-baseline-seed17"
