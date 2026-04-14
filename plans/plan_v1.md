# Plan: Lightweight QAT for Qwen3-4B

## Summary
Build a new `uv`-managed Python 3.12 repo that implements a clean, self-contained PyTorch SFT baseline and QAT pipeline for `Qwen/Qwen3-4B` on a single H100. The baseline and all QAT variants share the same data splits, core hyperparameters, artifact layout, and evaluation path; only quantization-specific behavior changes.

Supported variants for v1 are `fp8/bf16`, `int8/bf16`, `fp8/fp8`, `int8/int8`, `int4/fp8`, and `int4/bf16`. `int4/int8` is excluded and must fail validation at config parse time. QAT targets transformer-block `nn.Linear` modules only; embeddings, norms, and `lm_head` stay `bf16`.

Implementation guidance: use `DeepWiki` MCP against `pytorch/ao` for architecture and behavior questions, and use the local `ref_torch_ao/` checkout for exact reference paths and code reading. `torch/ao` is reference material only; the project runtime path must remain self-contained and must not directly import or depend on the reference repo.

## Public Interfaces / Types
- Python package: `src/qat`
- Entry point: `python -m qat.cli <baseline|qat|eval|smoke|full>`
- Typed config dataclasses in `src/qat/config.py` for model/dataset revisions, split sizes, seeds, quant scheme, artifact paths, decode policy, and compile policy
- Quant scheme enum limited to the six supported pairs above; unsupported pairs hard-error early
- Artifact layout: one standalone HF-style directory per run under `/mnt/ssd2/shreyansh/ptq_experiments/artifacts_qat/<run_id>/`
- Metrics file: dataset-level `metrics_numinamath_cot.csv`
- Per-run manifest must record model revision, dataset revision, quant scheme, split manifest, seed, package versions, git SHA, compile status, and resume fingerprint

## Dependency Graph
```text
T1 ──┬── T4 ──┬── T6 ──┐
     │        │        ├── T9 ──┐
T2 ──┼── T3 ──┼── T7 ──┘        │
     │        └── T8 ───────────┼── T10 ── T12 ── T13
     └────────────── T5 ────────┘      │
                                       └── T11 ──┘
```

## Tasks
### T1: Bootstrap repo
- **depends_on**: `[]`
- **location**: repo root, `pyproject.toml`, `src/qat`, `tests`
- **description**: Initialize git with the provided identity/remote, create `uv` Python 3.12 environment, add `ruff`/`pytest`, and create the package skeleton. Use git identity `Shreyansh Singh <shreyansh.pettswood@gmail.com>` and remote `git@github.com:shreyansh26/quantization-aware-training.git`.
- **validation**: `uv sync`, `ruff check .`, `pytest -q` run against placeholder package/tests.
- **status**: Completed
- **log**: Initialized git on `main`, configured the requested identity and remote, created the `uv` Python 3.12 environment, added project metadata plus `ruff`/`pytest`, and bootstrapped the `src/qat` package with a passing placeholder test.
- **files edited/created**: `.gitignore`, `README.md`, `pyproject.toml`, `src/qat/__init__.py`, `tests/test_package.py`, `uv.lock`

### T2: Pin external inputs
- **depends_on**: `[]`
- **location**: `src/qat/config.py`, `notes/`
- **description**: Resolve and record exact HF revisions for `Qwen/Qwen3-4B` and `AI-MO/NuminaMath-CoT`, plus the official supported scheme matrix used for v1.
- **validation**: Revisions and scheme table are stored in config/docs and referenced by manifests.
- **status**: Completed
- **log**: Resolved exact Hugging Face HEAD revisions for the model and dataset, codified the supported and unsupported quantization variants as project constants, and recorded the serving-matrix references plus DeepWiki/reference-repo guidance in `notes/pinned_inputs.md`.
- **files edited/created**: `notes/pinned_inputs.md`, `src/qat/config.py`

### T3: Preflight compatibility gate
- **depends_on**: `[T2]`
- **location**: `src/qat/preflight.py`
- **description**: Add a fail-fast environment checker for CUDA/H100, PyTorch, Transformers, TRL, vLLM, and `compressed-tensors`, and reject unsupported quant pairs before any run starts.
- **validation**: A preflight command reports pass/fail with explicit reasons; `int4/int8` hard-fails.
- **status**: Completed
- **log**: Added a standalone preflight module with explicit PASS/FAIL reporting for Python, package availability, CUDA visibility, H100 selection, and FP8 capability checks. The guard integrates the project quantization matrix and blocks unsupported variants before any expensive work begins.
- **files edited/created**: `src/qat/preflight.py`, `tests/test_preflight.py`

### T4: Data pipeline and split manifests
- **depends_on**: `[T1, T2]`
- **location**: `src/qat/data.py`
- **description**: Load `NuminaMath-CoT`, create deterministic source-stratified smoke/full holdouts, use the `messages` field with the model chat template, and generate assistant-only labels. If exact stratification is impossible, cap exhausted sources, redistribute the remainder deterministically by largest deficit, and log the result in a split manifest.
- **validation**: Split manifests are reproducible from the same seed; unit fixtures confirm assistant-only masking boundaries.
- **status**: Completed
- **log**: Implemented deterministic source-stratified split selection from the training pool, a JSON split-manifest writer, and chat-templated training encoding with assistant-only labels. The tokenization path uses the tokenizer-provided assistant mask when available and falls back to prompt-prefix inference when it is not.
- **files edited/created**: `src/qat/data.py`, `tests/test_data.py`

### T5: Config, CLI, and run manifests
- **depends_on**: `[T1, T2]`
- **location**: `src/qat/config.py`, `src/qat/cli.py`
- **description**: Define the runtime interface, scheme validator, artifact naming/run-id rules, resume fingerprint, and manifest schema shared by baseline, QAT, eval, smoke, and full runs.
- **validation**: Invalid configs fail early; manifests serialize and reload cleanly.
- **status**: Completed
- **log**: Expanded the config layer into typed split, runtime, training, and manifest dataclasses; added stable run IDs, artifact-directory helpers, resume fingerprints, and variant validation. Added the initial CLI surface for `baseline`, `qat`, `eval`, `smoke`, and `full`, plus targeted tests for config parsing and manifest behavior.
- **files edited/created**: `src/qat/config.py`, `src/qat/cli.py`, `tests/test_config.py`, `tests/test_cli.py`

### T6: Baseline training engine
- **depends_on**: `[T3, T4, T5]`
- **location**: `src/qat/train/baseline.py`
- **description**: Implement the plain PyTorch single-GPU bf16 SFT loop with checkpoint/resume, gradient checkpointing, optimizer/scheduler, deterministic seeding, and optional best-effort training `torch.compile` probe with eager fallback.
- **validation**: A smoke baseline run trains, resumes safely, and logs stable loss without NaNs.
- **status**: Completed
- **log**: Added a self-contained baseline SFT training module with tokenizer/model loading, assistant-only chat-template encoding, optimizer and warmup scheduler setup, compile probing with eager fallback, and checkpoint/manifest persistence. The CPU unit path now exercises checkpoint writes, compile fallback behavior, and a minimal train step without relying on a live CUDA ordinal.
- **files edited/created**: `src/qat/train/__init__.py`, `src/qat/train/baseline.py`, `tests/test_baseline.py`

### T7: QAT core
- **depends_on**: `[T3, T5]`
- **location**: `src/qat/quantization/`
- **description**: Implement fake-quant configs/modules and prepare/convert flow for the six supported schemes, targeting transformer-block `Linear` layers only and matching torchao-style canonical granularities.
- **validation**: Prepare/convert runs on model modules, unsupported schemes fail early, and converted weights match the requested scheme metadata.
- **status**: Completed
- **log**: Implemented the QAT spec table, fake-quant helpers for FP8 and integer paths, a `FakeQuantLinear` wrapper, and recursive prepare/convert passes that only rewrite eligible `Linear` layers. The unit tests cover per-token, per-channel, and per-group behavior plus conversion back to eager `Linear` modules.
- **files edited/created**: `src/qat/quantization/__init__.py`, `src/qat/quantization/qat.py`, `tests/test_qat.py`

### T8: Evaluator core
- **depends_on**: `[T4, T5]`
- **location**: `src/qat/eval/core.py`
- **description**: Build answer extraction, normalization, boxed-answer handling, SymPy equivalence fallback, reason-coded failures, and dataset-level CSV/prediction-log row creation. SymPy parsing must be exception-safe and bounded.
- **validation**: Unit tests cover boxed answers, equivalent expressions, malformed outputs, and parser fallbacks.
- **status**: Completed
- **log**: Added boxed/final-answer extraction, normalization utilities, a SymPy-backed equivalence check with safe fallbacks, and helpers for dataset-level metrics rows plus per-run prediction logs. The evaluation tests cover exact matches, symbolic equivalence, malformed outputs, and metrics-log serialization.
- **files edited/created**: `src/qat/eval/__init__.py`, `src/qat/eval/core.py`, `tests/test_eval_core.py`

### T9: Export and serving path
- **depends_on**: `[T4, T5, T6, T7]`
- **location**: `src/qat/export.py`, `src/qat/eval/vllm_runner.py`
- **description**: Convert trained checkpoints into standalone HF-style artifact directories using a thin `compressed-tensors` adapter, write via temp-dir then atomic rename, run converted-model compile checks, and verify vLLM loadability with greedy decoding.
- **validation**: Every exported artifact passes completeness checks and can be loaded by vLLM or fails with explicit diagnostics.
- **status**: Completed
- **log**: Added a checkpoint-to-artifact exporter with temp-dir staging, completeness verification, manifest persistence, and compressed-tensors metadata attachment for the supported quantization variants. Added a vLLM adapter that builds generation prompts from chat-format rows, lowers GPU memory utilization for shared-GPU loadability checks, and verifies exported artifacts with explicit error capture.
- **files edited/created**: `src/qat/export.py`, `src/qat/eval/vllm_runner.py`, `tests/test_export.py`, `tests/test_vllm_runner.py`

### T10: Experiment harness
- **depends_on**: `[T5, T6, T7, T8, T9]`
- **location**: `src/qat/runner.py`
- **description**: Implement matrix orchestration for baseline + supported QAT variants, resume semantics, retries, artifact directory conventions, and metrics/prediction-log append behavior.
- **validation**: Re-running the same manifest resumes or no-ops safely; changed fingerprints are rejected.
- **status**: Completed
- **log**: Implemented the matrix runner, stage-tracked resume logic, split-manifest reuse, metrics dedupe, and CLI wiring for `baseline`, `qat`, `eval`, `smoke`, and `full`. Added a dedicated `train_qat` path so the runner can mirror the baseline training flow without inlining QAT preparation logic.
- **files edited/created**: `src/qat/runner.py`, `src/qat/train/qat.py`, `src/qat/train/__init__.py`, `src/qat/cli.py`, `tests/test_runner.py`, `tests/test_train_qat.py`, `tests/test_cli.py`

### T11: Test suite
- **depends_on**: `[T4, T5, T6, T7, T8, T9]`
- **location**: `tests/`
- **description**: Add CPU unit tests for quant math, masking, split reproducibility, config validation, answer equivalence, and artifact manifests; add opt-in GPU integration tests for baseline smoke, QAT prepare/convert, export loadability, compile fallback, and vLLM eval.
- **validation**: `pytest -q` passes on CPU subsets; GPU-marked tests pass on an allowed H100.
- **status**: Completed
- **log**: Expanded the CPU suite with coverage for QAT training, export, vLLM prompt/load wrappers, runner resume behavior, and the assistant-mask fallback fix required by Qwen3 chat templating. Added opt-in GPU integration tests for one-step baseline and QAT export/loadability, and manually validated baseline plus `int8_bf16` tiny smoke paths on GPU 5.
- **files edited/created**: `tests/test_gpu_integration.py`, `tests/test_data.py`, `tests/test_export.py`, `tests/test_runner.py`, `tests/test_train_qat.py`, `tests/test_vllm_runner.py`, `pyproject.toml`

### T12: Smoke matrix and promotion gate
- **depends_on**: `[T10, T11]`
- **location**: artifacts root, `metrics_numinamath_cot.csv`
- **description**: Run baseline plus all six supported QAT variants on the 500/100 split. Promotion requires zero runtime/export/load failures, no NaN losses, non-empty metric rows, and saved prediction logs for manual review. Qualitative inspection is required in addition to aggregate metrics to catch gibberish or obviously degraded generations.
- **validation**: Smoke artifacts, CSV rows, manifests, and prediction logs exist for every run and satisfy the gate.

### T13: Full matrix execution phase
- **depends_on**: `[T12]`
- **location**: artifacts root, `metrics_numinamath_cot.csv`
- **description**: Reuse the same configs and harness for the 5000/500 split, append results to the same dataset-level metrics file, and preserve per-run manifests and prediction logs.
- **validation**: Full-run artifacts append cleanly without schema drift and remain traceable to pinned revisions.

## Parallel Execution Groups
| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | `T1`, `T2` | Immediately |
| 2 | `T3`, `T4`, `T5` | Wave 1 complete |
| 3 | `T6`, `T7`, `T8` | Wave 2 complete |
| 4 | `T9`, `T10` | Wave 3 complete |
| 5 | `T11` | `T4`-`T9` complete |
| 6 | `T12` | `T10`, `T11` complete |
| 7 | `T13` | `T12` complete |

## Test Plan
- CPU unit coverage for fake-quant numerics, masking correctness, split reproducibility, scheme validation, answer extraction, and SymPy equivalence fallback.
- GPU integration coverage for baseline smoke training, QAT prepare/convert, exported artifact loadability, compile fallback behavior, and vLLM greedy evaluation.
- Resume tests must reject mismatched run fingerprints.
- Artifact tests must verify atomic writes, manifest completeness, and direct loadability from the saved directory.

## Assumptions and Defaults
- Full-parameter fine-tuning for both baseline and QAT.
- Single-GPU only in v1; code should stay FSDP-aware but not implement FSDP mode.
- Use any free GPU on the machine except GPUs `6` and `7`.
- `messages` + chat template + assistant-only loss is the only training format.
- Greedy decoding only for evaluation.
- Converted-model `torch.compile` compatibility is required; training-time compile is best-effort with eager fallback and manifest logging.
- Metrics stay dataset-level for v1; qualitative inspection is supported by per-example prediction logs, not by separate task-level files.
- Once a part of the implementation is stable and validated, commit and push it rather than batching all version-control work until the end.

## References
- vLLM / LLM Compressor scheme guidance: https://docs.vllm.ai/projects/llm-compressor/en/0.10.0/steps/choosing-scheme/
- vLLM / LLM Compressor saving flow: https://docs.vllm.ai/projects/llm-compressor/en/stable/guides/saving_a_model/
- Model page: https://hf.co/Qwen/Qwen3-4B
- Dataset page: https://hf.co/datasets/AI-MO/NuminaMath-CoT
