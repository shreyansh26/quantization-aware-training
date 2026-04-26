import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qat.config import RuntimeConfig
from qat.eval.vllm_compat import (
    VLLM_W4A8_FP8_PATCH_ENV,
    patch_vllm_w4a8_fp8_scale_view,
)


@dataclass(frozen=True)
class VLLMGeneration:
    index: int
    prompt_text: str
    prediction_text: str


@dataclass(frozen=True)
class VLLMValidationResult:
    artifact_dir: str
    loaded: bool
    engine_model_name: str
    error: str | None = None


def _messages_to_prompt_messages(
    messages: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    if messages and messages[-1].get("role") == "assistant":
        return list(messages[:-1])
    return list(messages)


def build_generation_prompts(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        prompt_messages = _messages_to_prompt_messages(row["messages"])
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def _prepare_vllm_env(config: RuntimeConfig) -> None:
    _ = config
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault(VLLM_W4A8_FP8_PATCH_ENV, "1")
    src_path = str(Path(__file__).resolve().parents[2])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        entries = pythonpath.split(os.pathsep)
        if src_path not in entries:
            os.environ["PYTHONPATH"] = os.pathsep.join([src_path, *entries])
    else:
        os.environ["PYTHONPATH"] = src_path
    patch_vllm_w4a8_fp8_scale_view()


def verify_vllm_loadability(
    artifact_dir: Path,
    config: RuntimeConfig,
) -> VLLMValidationResult:
    try:
        _prepare_vllm_env(config)
        from vllm import LLM

        llm = LLM(
            model=str(artifact_dir),
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.75,
            enforce_eager=True,
        )
        del llm
        return VLLMValidationResult(
            artifact_dir=str(artifact_dir),
            loaded=True,
            engine_model_name=str(artifact_dir),
        )
    except Exception as exc:
        return VLLMValidationResult(
            artifact_dir=str(artifact_dir),
            loaded=False,
            engine_model_name=str(artifact_dir),
            error=f"{type(exc).__name__}: {exc}",
        )


def generate_with_vllm(
    artifact_dir: Path,
    *,
    prompts: Sequence[str],
    config: RuntimeConfig,
    max_new_tokens: int = 256,
) -> list[VLLMGeneration]:
    _prepare_vllm_env(config)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(artifact_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(list(prompts), sampling_params)
    generations: list[VLLMGeneration] = []
    for index, output in enumerate(outputs):
        text = output.outputs[0].text if output.outputs else ""
        generations.append(
            VLLMGeneration(
                index=index,
                prompt_text=prompts[index],
                prediction_text=text,
            )
        )
    return generations
