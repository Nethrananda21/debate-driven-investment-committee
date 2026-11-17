import json

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments


def make_agent(service, name, instructions):
    options = {"temperature": 0, "max_tokens": 700}
    if service.ai_model_id == "nvidia/nemotron-3.5-lightning-30b-a3b":
        options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif service.ai_model_id == "qwen/qwen3.8-27b":
        options["extra_body"] = {"reasoning_effort": "none"}
    elif service.ai_model_id in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        options["reasoning_effort"] = "low"
    settings = OpenAIChatPromptExecutionSettings(**options)
    return ChatCompletionAgent(
        service=service,
        name=name,
        instructions=instructions,
        arguments=KernelArguments(settings=settings),
    )


async def ask_json(agent, prompt):
    response = await agent.get_response(messages=prompt)
    text = str(response.content).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)
