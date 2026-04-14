from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from qat.config import CompilePolicy, RuntimeConfig, make_manifest
from qat.data import encode_messages_for_training, load_numinamath_train_dataset


class SavePretrainedModel(Protocol):
    def save_pretrained(self, save_directory: str | Path) -> None: ...


class SavePretrainedTokenizer(Protocol):
    def save_pretrained(self, save_directory: str | Path) -> None: ...


@dataclass(frozen=True)
class TrainerState:
    epoch: int = 0
    step: int = 0
    optimizer_step: int = 0


@dataclass(frozen=True)
class BaselineTrainingSummary:
    checkpoint_dir: str
    compile_status: str
    steps_completed: int
    final_loss: float


def runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_baseline_tokenizer(config: RuntimeConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_baseline_model(config: RuntimeConfig):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
    )
    if config.training.gradient_checkpointing and hasattr(
        model,
        "gradient_checkpointing_enable",
    ):
        model.gradient_checkpointing_enable()
    return model


def encode_split_examples(
    *,
    dataset: Any,
    indices: list[int],
    tokenizer: Any,
    max_length: int = 2048,
) -> list[dict[str, list[int]]]:
    encoded_examples: list[dict[str, list[int]]] = []
    for index in indices:
        row = dataset[int(index)]
        encoded_examples.append(
            encode_messages_for_training(
                tokenizer=tokenizer,
                messages=row["messages"],
                max_length=max_length,
            )
        )
    return encoded_examples


def collate_encoded_examples(
    batch: list[dict[str, list[int]]],
) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "attention_mask", "labels")
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        output[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    return output


def compile_model_for_training(
    model: nn.Module,
    compile_policy: CompilePolicy,
) -> tuple[nn.Module, str]:
    if compile_policy == CompilePolicy.DISABLED:
        return model, "disabled"
    try:
        return torch.compile(model), "compiled"
    except Exception as exc:  # pragma: no cover
        if compile_policy == CompilePolicy.REQUIRED:
            raise
        return model, f"eager_fallback:{type(exc).__name__}"


def _warmup_factor(step: int, *, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return 1.0


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_factor(
            step,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
        ),
    )


def save_training_checkpoint(
    *,
    checkpoint_dir: Path,
    state: TrainerState,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    tokenizer: SavePretrainedTokenizer | None = None,
    manifest_payload: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": asdict(state),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_dir / "model")
    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    if manifest_payload is not None:
        (checkpoint_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
        )


def load_training_checkpoint(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> TrainerState:
    payload = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return TrainerState(**payload["state"])


def collect_package_versions() -> dict[str, str]:
    packages = [
        "torch",
        "transformers",
        "datasets",
        "trl",
        "sympy",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    grad_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    start_step: int = 0,
    max_steps: int | None = None,
) -> tuple[TrainerState, float]:
    model.train()
    step = start_step
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / grad_accumulation_steps
        if torch.isnan(loss):
            raise ValueError("encountered NaN loss during baseline training")
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        should_step = (batch_index + 1) % grad_accumulation_steps == 0
        if should_step:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
    final_loss = losses[-1] if losses else 0.0
    return TrainerState(epoch=epoch, step=step, optimizer_step=step), final_loss


def train_baseline(
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
    device = runtime_device()
    tokenizer = tokenizer or load_baseline_tokenizer(config)
    model = model or load_baseline_model(config)
    model = model.to(device)
    model, compile_status = compile_model_for_training(model, config.compile_policy)

    manifest_payload = json.loads(split_manifest_path.read_text())
    train_indices = manifest_payload["train_indices"]
    dataset = dataset or load_numinamath_train_dataset(
        revision=config.dataset_revision
    )
    encoded_examples = encode_split_examples(
        dataset=dataset,
        indices=train_indices,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(
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

    checkpoint_dir = checkpoint_dir or Path("artifacts") / "baseline-checkpoint"
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
