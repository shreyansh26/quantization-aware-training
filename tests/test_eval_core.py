from __future__ import annotations

import csv
import json

from qat.eval.core import (
    append_metrics_row,
    evaluate_prediction,
    extract_boxed_answer,
    extract_final_answer,
    write_prediction_log,
)


def test_extract_boxed_answer_returns_last_boxed_value() -> None:
    text = "work... \\boxed{1} and finally \\boxed{\\frac{3}{10}}"
    assert extract_boxed_answer(text) == r"\frac{3}{10}"


def test_extract_final_answer_falls_back_to_last_line() -> None:
    text = "reasoning\nmore reasoning\n15/150"
    assert extract_final_answer(text) == "15/150"


def test_evaluate_prediction_accepts_equivalent_fractions() -> None:
    decision = evaluate_prediction(
        prediction=r"The final answer is \boxed{\frac{15}{50}}",
        reference=r"\boxed{\frac{3}{10}}",
    )
    assert decision.is_correct
    assert decision.match_method == "sympy"


def test_evaluate_prediction_handles_non_matching_answers() -> None:
    decision = evaluate_prediction(
        prediction=r"\boxed{7}",
        reference=r"\boxed{8}",
    )
    assert not decision.is_correct
    assert decision.match_method == "none"


def test_append_metrics_row_creates_header(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "metrics_numinamath_cot.csv"
    append_metrics_row(
        path,
        {
            "model_name": "demo",
            "quantization_artifact": "artifact",
            "quantization_dtype": "bf16/bf16",
            "quantization_granularity": "none",
            "quantization_method": "none",
            "metric_name": "exact_match",
            "metric_value": 0.5,
        },
    )
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["metric_name"] == "exact_match"


def test_write_prediction_log_serializes_decisions(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "predictions.json"
    decision = evaluate_prediction(r"\boxed{2}", r"\boxed{2}")
    write_prediction_log(path, [decision])
    payload = json.loads(path.read_text())
    assert payload[0]["is_correct"] is True
