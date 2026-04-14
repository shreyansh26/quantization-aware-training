from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sympy

from qat.config import QuantizationVariant, get_variant_metadata, parse_variant

BOXED_MARKER = r"\boxed{"
FRACTION_PATTERN = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")


@dataclass(frozen=True)
class EvaluationDecision:
    prediction_text: str
    reference_text: str
    extracted_prediction: str
    extracted_reference: str
    normalized_prediction: str
    normalized_reference: str
    is_correct: bool
    match_method: str


def extract_boxed_answer(text: str) -> str | None:
    start = text.rfind(BOXED_MARKER)
    if start == -1:
        return None
    cursor = start + len(BOXED_MARKER)
    depth = 1
    chars: list[str] = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        cursor += 1
    return None


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    for pattern in (
        r"final answer\s*[:=]\s*(.+)",
        r"answer\s*[:=]\s*(.+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def _replace_fractions(text: str) -> str:
    previous = text
    while True:
        current = FRACTION_PATTERN.sub(r"(\1)/(\2)", previous)
        if current == previous:
            return current
        previous = current


def normalize_answer(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.strip("$")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\cdot", "*")
    normalized = normalized.replace("^", "**")
    normalized = _replace_fractions(normalized)
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.strip(".;,")
    return normalized


def _sympy_parseable(text: str) -> str:
    candidate = normalize_answer(text)
    if "=" in candidate:
        candidate = candidate.split("=")[-1]
    return candidate


def are_answers_equivalent(prediction: str, reference: str) -> tuple[bool, str]:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    if normalized_prediction == normalized_reference:
        return True, "exact"

    try:
        lhs = sympy.sympify(_sympy_parseable(prediction))
        rhs = sympy.sympify(_sympy_parseable(reference))
    except (sympy.SympifyError, TypeError, ValueError):
        return False, "none"

    try:
        equivalent = bool(sympy.simplify(lhs - rhs) == 0)
    except TypeError:
        return False, "none"
    return equivalent, "sympy" if equivalent else "none"


def evaluate_prediction(prediction: str, reference: str) -> EvaluationDecision:
    extracted_prediction = extract_final_answer(prediction)
    extracted_reference = extract_final_answer(reference)
    normalized_prediction = normalize_answer(extracted_prediction)
    normalized_reference = normalize_answer(extracted_reference)
    is_correct, method = are_answers_equivalent(
        extracted_prediction,
        extracted_reference,
    )
    return EvaluationDecision(
        prediction_text=prediction,
        reference_text=reference,
        extracted_prediction=extracted_prediction,
        extracted_reference=extracted_reference,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        is_correct=is_correct,
        match_method=method,
    )


def make_metrics_row(
    *,
    model_name: str,
    quantization_artifact: str,
    variant: QuantizationVariant | str | None,
    metric_name: str,
    metric_value: float,
) -> dict[str, Any]:
    parsed = parse_variant(variant)
    metadata = get_variant_metadata(parsed)
    return {
        "model_name": model_name,
        "quantization_artifact": quantization_artifact,
        "quantization_dtype": (
            f"{metadata.weight_dtype}/{metadata.activation_dtype}"
            if metadata is not None
            else "bf16/bf16"
        ),
        "quantization_granularity": (
            metadata.serving_scheme if metadata is not None else "none"
        ),
        "quantization_method": metadata.source if metadata is not None else "none",
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = list(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_prediction_log(path: Path, decisions: list[EvaluationDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(decision) for decision in decisions]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
