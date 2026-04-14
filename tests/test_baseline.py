from __future__ import annotations

import json

import torch
from torch import nn

from qat.config import (
    CompilePolicy,
    RunMode,
    RuntimeConfig,
    SplitConfig,
    TrainingConfig,
)
from qat.train.baseline import (
    TrainerState,
    collate_encoded_examples,
    compile_model_for_training,
    save_training_checkpoint,
    train_baseline,
)


class TinyTokenizer:
    def save_pretrained(self, save_directory):  # noqa: ANN001
        import os

        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/tokenizer.json", "w", encoding="utf-8") as handle:
            handle.write("{}")

    def apply_chat_template(self, messages, **_: object):  # noqa: ANN001
        assert messages
        return {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 1],
        }


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.proj = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: ANN001
        hidden = self.embedding(input_ids)
        logits = self.proj(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return type("Output", (), {"loss": loss, "logits": logits})()

    def save_pretrained(self, save_directory):  # noqa: ANN001
        import os

        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")


def test_collate_encoded_examples_returns_tensors() -> None:
    batch = collate_encoded_examples(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [-100, 4]},
        ]
    )
    assert batch["input_ids"].shape == (2, 2)
    assert batch["labels"].dtype == torch.long


def test_compile_model_for_training_falls_back_to_eager(monkeypatch) -> None:  # noqa: ANN001
    model = TinyLM()

    def explode(_model):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "compile", explode)
    compiled, status = compile_model_for_training(model, CompilePolicy.TRY)
    assert compiled is model
    assert status.startswith("eager_fallback:")


def test_train_baseline_writes_checkpoint(tmp_path) -> None:  # noqa: ANN001
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps({"train_indices": [0, 1], "test_indices": [2]})
    )
    dataset = [
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
        {"messages": [{"role": "user", "content": "c"}]},
    ]
    config = RuntimeConfig(
        split=SplitConfig(name="smoke", train_size=2, test_size=1, seed=17),
        mode=RunMode.BASELINE,
        compile_policy=CompilePolicy.DISABLED,
        training=TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=1,
        ),
    )
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    try:
        summary = train_baseline(
            config,
            split_manifest_path=split_manifest,
            checkpoint_dir=tmp_path / "ckpt",
            max_steps=1,
            model=TinyLM(),
            tokenizer=TinyTokenizer(),
            dataset=dataset,
            max_length=16,
        )
    finally:
        torch.cuda.is_available = original_is_available  # type: ignore[method-assign]
    assert summary.steps_completed == 1
    assert (tmp_path / "ckpt" / "training_state.pt").exists()


def test_save_training_checkpoint_persists_manifest(tmp_path) -> None:  # noqa: ANN001
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_training_checkpoint(
        checkpoint_dir=tmp_path / "ckpt",
        state=TrainerState(epoch=1, step=2, optimizer_step=2),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=TinyTokenizer(),
        manifest_payload={"run_id": "demo"},
    )
    payload = json.loads((tmp_path / "ckpt" / "manifest.json").read_text())
    assert payload["run_id"] == "demo"
