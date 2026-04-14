from __future__ import annotations

import json

from qat.config import SplitConfig
from qat.data import (
    build_split_manifest,
    encode_messages_for_training,
    save_split_manifest,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **_: object):  # noqa: ANN001
        assert messages[0]["role"] == "user"
        return {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 1],
        }


class FallbackTokenizer:
    def apply_chat_template(self, messages, **kwargs: object):  # noqa: ANN001
        if kwargs.get("add_generation_prompt"):
            return [10, 11]
        assert len(messages) == 2
        return {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
        }


class ZeroMaskTokenizer:
    def apply_chat_template(self, messages, **kwargs: object):  # noqa: ANN001
        if kwargs.get("add_generation_prompt"):
            return [10, 11]
        return {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 0, 0],
        }


def test_build_split_manifest_is_deterministic() -> None:
    dataset = [
        {"source": "a", "messages": []},
        {"source": "a", "messages": []},
        {"source": "b", "messages": []},
        {"source": "b", "messages": []},
        {"source": "c", "messages": []},
        {"source": "c", "messages": []},
    ]
    split = SplitConfig(name="smoke", train_size=3, test_size=2, seed=17)
    left = build_split_manifest(dataset, split)
    right = build_split_manifest(dataset, split)
    assert left.train_indices == right.train_indices
    assert left.test_indices == right.test_indices


def test_encode_messages_for_training_builds_assistant_only_labels() -> None:
    encoded = encode_messages_for_training(
        tokenizer=FakeTokenizer(),
        messages=[{"role": "user", "content": "solve this"}],
    )
    assert encoded["labels"] == [-100, -100, 12, 13]
    assert encoded["assistant_tokens_mask"] == [0, 0, 1, 1]


def test_encode_messages_for_training_falls_back_without_assistant_mask() -> None:
    encoded = encode_messages_for_training(
        tokenizer=FallbackTokenizer(),
        messages=[
            {"role": "user", "content": "solve this"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    assert encoded["assistant_tokens_mask"] == [0, 0, 1, 1]
    assert encoded["labels"] == [-100, -100, 12, 13]


def test_encode_messages_for_training_replaces_all_zero_assistant_mask() -> None:
    encoded = encode_messages_for_training(
        tokenizer=ZeroMaskTokenizer(),
        messages=[
            {"role": "user", "content": "solve this"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    assert encoded["assistant_tokens_mask"] == [0, 0, 1, 1]
    assert encoded["labels"] == [-100, -100, 12, 13]


def test_save_split_manifest_writes_json(tmp_path) -> None:  # noqa: ANN001
    dataset = [
        {"source": "a", "messages": []},
        {"source": "a", "messages": []},
        {"source": "b", "messages": []},
        {"source": "b", "messages": []},
        {"source": "c", "messages": []},
    ]
    split = SplitConfig(name="smoke", train_size=2, test_size=2, seed=19)
    manifest = build_split_manifest(dataset, split)
    path = tmp_path / "split.json"
    save_split_manifest(manifest, path)
    payload = json.loads(path.read_text())
    assert payload["split_name"] == "smoke"
    assert sorted(payload) == sorted(manifest.to_dict())
