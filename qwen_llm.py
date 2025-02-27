from unittest.mock import patch
from collections.abc import Sequence
import re
import json

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _message_to_openai,
    _function_to_openai,
    _openai_to_assistant_message,
)

from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatAssistantMessage, FunctionCall

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def chat_completion_request(
    llm: LLM,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam] | None,
    sampling_params: SamplingParams,
):
    return llm.chat(
        messages=messages,
        tools=tools,
        sampling_params=sampling_params,
        use_tqdm=False,
    )


def _output_to_assistant_message(request_output: RequestOutput) -> ChatAssistantMessage:
    content = request_output.outputs[0].text

    # Sourced from https://qwen.readthedocs.io/en/latest/framework/function_call.html
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append(
                FunctionCall(
                    function=func["name"],
                    args=func["arguments"],
                    id=f"tool_call_{i}",
                )
            )
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        c = None
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        return ChatAssistantMessage(
            role="assistant",
            content=c,
            tool_calls=tool_calls,
        )
    return ChatAssistantMessage(
        role="assistant",
        content=re.sub(r"<\|im_end\|>$", "", content),
        tool_calls=None,
    )


class QwenLLM(OpenAILLM):
    def __init__(
        self,
        model: str,
        sampling_params: SamplingParams,
        device: str = "cuda",
        vllm_args: dict = {},
    ) -> None:
        self.model = model
        self.sampling_params = sampling_params

        gpu_memory_utilization = vllm_args.get("vllm_gpu_memory_utilization", 0.9)
        max_model_len = vllm_args.get("vllm_max_model_len", 20544)
        dtype = vllm_args.get("vllm_dtype", "auto")

        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None,
        )
        with world_size_patch, profiling_patch:
            self.llm = LLM(
                model=model,
                device=device,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                # This is particularly useful here because we generate completions from the same prompts.
                enable_prefix_caching=True,
                enforce_eager=True,
                max_model_len=max_model_len,
            )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message) for message in messages]
        openai_tools = [
            _function_to_openai(tool) for tool in runtime.functions.values()
        ]
        completion = chat_completion_request(
            self.llm,
            openai_messages,
            openai_tools,
            self.sampling_params,
        )
        assert len(completion[0].outputs) == 1, completion[0].outputs
        output = _output_to_assistant_message(completion[0])
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
