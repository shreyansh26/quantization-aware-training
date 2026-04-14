# QAT

Lightweight quantization-aware training experiments for `Qwen/Qwen3-4B` on `AI-MO/NuminaMath-CoT`.

## Overview

This repo provides a small single-GPU workflow for:

- baseline bf16 supervised fine-tuning
- QAT fine-tuning for supported quantization variants
- export to standalone model artifacts
- vLLM-based inference and NuminaMath evaluation

The interface is intentionally split into two stages:

- `train`: train + export only
- `eval`: vLLM loadability check + inference + evaluation only

That keeps long-running training separate from inference/evaluation and makes it easy to re-evaluate an existing exported artifact.

## Environment

- Python `3.12+`
- CUDA-visible GPU through `CUDA_VISIBLE_DEVICES`
- Hopper-class or newer GPU for this project’s preflight gate: compute capability `>= 9.0`
- Installed runtime packages: `torch`, `transformers`, `datasets`, `trl`, `vllm`, `compressed-tensors`

The code does not take a `gpu_index` flag. Set the device externally, for example:

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.preflight --variant int8_bf16
```

## Data Splits

Named splits are built from `AI-MO/NuminaMath-CoT` with deterministic seeds:

- `smoke`: train `500`, test `100`
- `full`: train `5000`, test `500`

## Supported Variants

QAT mode supports:

- `fp8_bf16`
- `int8_bf16`
- `fp8_fp8`
- `int8_int8`
- `int4_fp8`
- `int4_bf16`

Baseline mode does not take a quantization variant.

## Artifact Layout

The default artifact root is:

```text
/mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat
```

Train artifacts are written under:

```text
<artifact_root>/<run_id>/
```

Examples:

- `baseline-smoke-baseline-seed17`
- `qat-smoke-int8_bf16-seed17`

Each trained artifact includes `train_config.json`.

Evaluation writes:

- metrics CSV to the chosen `--output-path`
- `predictions_<run_id>.json` next to the metrics file
- `eval_config_<run_id>.json` next to the metrics file

## Commands

### Train baseline smoke

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.cli train \
  --type smoke \
  --mode baseline \
  --seed 17
```

### Train QAT smoke

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.cli train \
  --type smoke \
  --mode qat \
  --seed 17 \
  --quantization-variant int8_bf16
```

### Train QAT with custom optimizer settings

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.cli train \
  --type full \
  --mode qat \
  --seed 17 \
  --quantization-variant int8_int8 \
  --training.learning-rate 2e-5 \
  --training.weight-decay 0.01 \
  --training.warmup-ratio 0.03 \
  --training.max-grad-norm 1.0 \
  --training.num-epochs 1
```

### Evaluate from resolved artifact path

If `--model-path` is omitted, the code resolves the artifact path from `--type`, `--mode`, `--seed`, and `--quantization-variant`.

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.cli eval \
  --type smoke \
  --mode qat \
  --seed 17 \
  --quantization-variant int8_int8
```

### Evaluate an explicit artifact

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.cli eval \
  --type smoke \
  --mode baseline \
  --seed 17 \
  --model-path /mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat/baseline-smoke-baseline-seed17 \
  --output-path /mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat/metrics_numinamath_cot.csv
```

### Run preflight only

```bash
CUDA_VISIBLE_DEVICES=5 uv run python -m qat.preflight --variant fp8_fp8
```

## Notes

- `eval` uses the repo’s vLLM evaluation flow, not a separate ad hoc inference script.
- The eval path currently performs a loadability check before the actual evaluation pass, so the model is loaded twice per eval run.
- Known FP8 serving failures will surface during the loadability gate before generation starts.
- QAT uses fake quantization with an explicit straight-through estimator in [`src/qat/quantization/qat.py`](src/qat/quantization/qat.py): the forward pass uses the quantized value while the backward pass flows through the original tensor via `original + (quantized - original).detach()`.
