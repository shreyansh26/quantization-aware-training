# Pinned Inputs

## Hugging Face revisions

- Model: `Qwen/Qwen3-4B`
  - Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
  - Resolved with: `git ls-remote https://huggingface.co/Qwen/Qwen3-4B HEAD`
- Dataset: `AI-MO/NuminaMath-CoT`
  - Revision: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
  - Resolved with: `git ls-remote https://huggingface.co/datasets/AI-MO/NuminaMath-CoT HEAD`

## Supported quantization matrix for v1

Project variants:

- `fp8_bf16`
- `int8_bf16`
- `fp8_fp8`
- `int8_int8`
- `int4_fp8`
- `int4_bf16`

Explicitly unsupported:

- `int4_int8`

Supporting references used to lock the matrix:

- vLLM / LLM Compressor scheme table lists `W4A16/W8A16`, `W8A8-INT8`, `W8A8-FP8`, `W4AFP8`, and `W4AINT8`, with `W4AFP8` called out as Hopper-targeted and `W4AINT8` called out as Arm-targeted:
  - `https://docs.vllm.ai/projects/llm-compressor/en/0.10.0/steps/choosing-scheme/`
- vLLM TorchAO docs cover the native TorchAO path and are the reference point for TorchAO-specific exported formats:
  - `https://docs.vllm.ai/usage/quantization/torchao/`

## Reference implementation guidance

- Use DeepWiki MCP against `pytorch/ao` for architecture and behavior questions.
- Use the local `ref_torch_ao/` checkout for exact code paths and local source inspection.
- `torch/ao` is reference-only for this project. Reimplement the required functionality locally instead of importing runtime code from the reference repo.
