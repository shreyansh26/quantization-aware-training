from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import AdamW

from qat.config import RuntimeConfig, make_manifest
from qat.data import load_numinamath_train_dataset
from qat.quantization.qat import prepare_model_for_qat
from qat.train.baseline import (
    BaselineTrainingSummary,
    TrainerState,
    build_scheduler,
    collate_encoded_examples,
    collect_package_versions,
    compile_model_for_training,
    encode_split_examples,
    git_sha,
    load_baseline_model,
    load_baseline_tokenizer,
    runtime_device,
    save_training_checkpoint,
    train_one_epoch,
)


def load_qat_model(config: RuntimeConfig) -> nn.Module:
    if config.quantization_variant is None:
        raise ValueError("QAT training requires a quantization variant")
    model = load_baseline_model(config)
    return prepare_model_for_qat(model, config.quantization_variant)


def train_qat(
    config: RuntimeConfig,
    *,
    split_manifest_path: Path,
    checkpoint_dir: Path | None = None,
    max_length: int = 2048,
    max_steps: int | None = None,
    model: nn.Module | None = None,
    tokenizer: Any | None = None,
    dataset: Any | None = None,
) -> BaselineTrainingSummary:
    if config.quantization_variant is None:
        raise ValueError("QAT training requires a quantization variant")

    device = runtime_device()
    tokenizer = tokenizer or load_baseline_tokenizer(config)
    model = model or load_qat_model(config)
    model = model.to(device)
    model, compile_status = compile_model_for_training(model, config.compile_policy)

    split_payload = json.loads(split_manifest_path.read_text())
    train_indices = split_payload["train_indices"]
    dataset = dataset or load_numinamath_train_dataset(
        revision=config.dataset_revision
    )
    encoded_examples = encode_split_examples(
        dataset=dataset,
        indices=train_indices,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = __import__("torch").utils.data.DataLoader(
        encoded_examples,
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        collate_fn=collate_encoded_examples,
    )
    steps_per_epoch = max(
        1,
        len(dataloader) // config.training.gradient_accumulation_steps,
    )
    total_steps = max_steps or max(1, steps_per_epoch * config.training.num_epochs)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, total_steps),
        warmup_ratio=config.training.warmup_ratio,
    )
    state = TrainerState()
    final_loss = 0.0
    for epoch in range(1, config.training.num_epochs + 1):
        state, final_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
            epoch=epoch,
            start_step=state.step,
            max_steps=max_steps,
        )
        if max_steps is not None and state.step >= max_steps:
            break

    checkpoint_dir = checkpoint_dir or Path("artifacts") / "qat-checkpoint"
    manifest = make_manifest(
        config,
        split_manifest_path_value=split_manifest_path,
        package_versions=collect_package_versions(),
        git_sha=git_sha(),
    )
    save_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        state=state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        manifest_payload=manifest.to_dict(),
    )
    return BaselineTrainingSummary(
        checkpoint_dir=str(checkpoint_dir),
        compile_status=compile_status,
        steps_completed=state.step,
        final_loss=final_loss,
    )
