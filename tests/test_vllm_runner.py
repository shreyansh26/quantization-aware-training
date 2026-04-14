from __future__ import annotations

import sys
import types

from qat.config import RuntimeConfig, SplitConfig
from qat.eval.vllm_runner import (
    build_generation_prompts,
    generate_with_vllm,
    verify_vllm_loadability,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001
        rendered = " | ".join(f"{item['role']}:{item['content']}" for item in messages)
        if add_generation_prompt:
            rendered += " | assistant:"
        return rendered


def test_build_generation_prompts_drops_terminal_assistant() -> None:
    prompts = build_generation_prompts(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "\\boxed{4}"},
                ]
            }
        ],
        tokenizer=TinyTokenizer(),
    )
    assert prompts == ["user:What is 2+2? | assistant:"]


def test_verify_vllm_loadability_surfaces_errors(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    class BrokenLLM:
        def __init__(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=BrokenLLM))
    result = verify_vllm_loadability(
        tmp_path,
        RuntimeConfig(task="baseline", split=SplitConfig("smoke", 2, 1, 17)),
    )
    assert not result.loaded
    assert "RuntimeError" in (result.error or "")


def test_generate_with_vllm_uses_fake_module(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    class FakeSamplingParams:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    class FakeOutput:
        def __init__(self, text: str) -> None:
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeLLM:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def generate(self, prompts, sampling_params):  # noqa: ANN001
            assert sampling_params.kwargs["temperature"] == 0.0
            return [FakeOutput(f"answer:{prompt}") for prompt in prompts]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    outputs = generate_with_vllm(
        tmp_path,
        prompts=["hello"],
        config=RuntimeConfig(task="baseline", split=SplitConfig("smoke", 2, 1, 17)),
        max_new_tokens=8,
    )
    assert outputs[0].prediction_text == "answer:hello"
