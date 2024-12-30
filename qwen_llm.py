import openai
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from collections.abc import Sequence

from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM, _message_to_openai, _function_to_openai, _openai_to_assistant_message

from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    temperature: float | None = 0.0,
    **kwargs,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature,
        **kwargs,
    )

class QwenLLM(OpenAILLM):

    def __init__(self, client: openai.OpenAI, model: str, temperature: float | None = 0.0, **kwargs) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = chat_completion_request(self.client, self.model, openai_messages, openai_tools, self.temperature, **self.kwargs)
        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
