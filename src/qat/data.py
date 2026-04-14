from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

from qat.config import DATASET_ID, DATASET_REVISION, SplitConfig


@dataclass(frozen=True)
class SplitAllocation:
    source: str
    requested: int
    allocated: int
    available: int


@dataclass(frozen=True)
class SplitManifest:
    dataset_id: str
    dataset_revision: str
    source_split: str
    split_name: str
    train_size: int
    test_size: int
    seed: int
    train_indices: list[int]
    test_indices: list[int]
    allocations: list[SplitAllocation]
    redistribution_log: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allocations"] = [asdict(allocation) for allocation in self.allocations]
        return payload


def load_numinamath_train_dataset(
    *,
    revision: str = DATASET_REVISION,
):
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=revision, split="train")
    required_columns = {"messages", "source"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"dataset is missing required columns: {missing_str}")
    return dataset


def source_to_indices(dataset: Any) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        mapping[str(row["source"])].append(index)
    return dict(mapping)


def _base_allocations(
    total: int,
    available_by_source: dict[str, int],
) -> dict[str, int]:
    total_available = sum(available_by_source.values())
    if total > total_available:
        raise ValueError(
            f"requested {total} examples from a pool of only {total_available}"
        )

    allocations: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    assigned = 0
    for source, available in sorted(available_by_source.items()):
        exact = (total * available) / total_available
        count = min(available, int(exact))
        allocations[source] = count
        assigned += count
        remainder = exact - count
        remainders.append((available - count, int(remainder * 1_000_000), source))

    deficit = total - assigned
    for capacity, _, source in sorted(
        remainders,
        key=lambda item: (-item[1], -item[0], item[2]),
    ):
        if deficit == 0:
            break
        if capacity <= 0:
            continue
        allocations[source] += 1
        deficit -= 1
    if deficit != 0:
        raise ValueError(f"failed to allocate the requested total of {total}")
    return allocations


def _sample_indices(
    *,
    grouped_indices: dict[str, list[int]],
    allocations: dict[str, int],
    rng: Random,
) -> tuple[list[int], list[SplitAllocation]]:
    sampled: list[int] = []
    summary: list[SplitAllocation] = []
    for source in sorted(grouped_indices):
        indices = list(grouped_indices[source])
        rng.shuffle(indices)
        requested = allocations.get(source, 0)
        chosen = indices[:requested]
        sampled.extend(chosen)
        summary.append(
            SplitAllocation(
                source=source,
                requested=requested,
                allocated=len(chosen),
                available=len(indices),
            )
        )
    rng.shuffle(sampled)
    return sampled, summary


def build_split_manifest(
    dataset: Any,
    split: SplitConfig,
) -> SplitManifest:
    grouped = source_to_indices(dataset)
    available_by_source = {source: len(indices) for source, indices in grouped.items()}
    total_required = split.train_size + split.test_size
    selection_allocations = _base_allocations(total_required, available_by_source)

    rng = Random(split.seed)
    selected_indices, allocation_summary = _sample_indices(
        grouped_indices=grouped,
        allocations=selection_allocations,
        rng=rng,
    )
    test_indices = selected_indices[: split.test_size]
    train_indices = selected_indices[split.test_size : total_required]
    redistribution_log = [
        (
            "Allocated split counts by source using deterministic proportional "
            "sampling with capped availability and stable tie-breaking."
        )
    ]

    return SplitManifest(
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        source_split="train",
        split_name=split.name,
        train_size=len(train_indices),
        test_size=len(test_indices),
        seed=split.seed,
        train_indices=train_indices,
        test_indices=test_indices,
        allocations=allocation_summary,
        redistribution_log=redistribution_log,
    )


def save_split_manifest(manifest: SplitManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


def _fallback_assistant_mask(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    input_length: int,
    max_length: int | None,
) -> list[int]:
    conversation_prefix = messages[:-1]
    if not conversation_prefix:
        raise ValueError("cannot infer assistant mask without any prompt messages")
    prompt_tokens = tokenizer.apply_chat_template(
        conversation_prefix,
        tokenize=True,
        add_generation_prompt=True,
        max_length=max_length,
        truncation=max_length is not None,
    )
    prefix_length = len(prompt_tokens)
    assistant_length = max(0, input_length - prefix_length)
    return ([0] * prefix_length) + ([1] * assistant_length)


def encode_messages_for_training(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int | None = None,
) -> dict[str, list[int]]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        max_length=max_length,
        truncation=max_length is not None,
    )
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded.get("assistant_tokens_mask")
    if assistant_mask is None:
        assistant_mask = _fallback_assistant_mask(
            tokenizer=tokenizer,
            messages=messages,
            input_length=len(encoded["input_ids"]),
            max_length=max_length,
        )

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    labels = [
        token if is_assistant else -100
        for token, is_assistant in zip(input_ids, assistant_mask, strict=True)
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_tokens_mask": list(assistant_mask),
    }
