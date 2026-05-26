"""Microsoft Agent Framework (MAF) agent for TAU Bench."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from agent_framework import (
    Message, Content, FunctionTool, ChatOptions,
)
from agent_framework_openai import OpenAIChatClient

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from src.metrics import TokenTracker


def _make_dummy_func(name: str, params: dict):
    """Create a dummy callable for FunctionTool so MAF doesn't complain."""
    def _func(**kwargs):
        return ""
    _func.__name__ = name
    return _func


def _convert_tools(tools_info: List[Dict[str, Any]]) -> List[FunctionTool]:
    """Convert tau-bench OpenAI-format tools to MAF FunctionTool."""
    tools = []
    for tool in tools_info:
        func = tool["function"]
        ft = FunctionTool(
            name=func["name"],
            description=func.get("description", ""),
            input_model=func.get("parameters", {}),
            func=_make_dummy_func(func["name"], func.get("parameters", {})),
        )
        tools.append(ft)
    return tools


class MAFAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str = "gpt-4o",
        model_provider: str = "openai",
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.tracker = TokenTracker()

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        return asyncio.run(self._solve_async(env, task_index, max_num_steps))

    async def _solve_async(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        self.tracker.reset()

        client = OpenAIChatClient(model=self.model)
        tools = _convert_tools(self.tools_info)

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        reward = 0.0

        messages = [
            Message(role="system", contents=[self.wiki]),
            Message(role="user", contents=[obs]),
        ]
        raw_messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        options = ChatOptions(
            model=self.model,
            temperature=self.temperature,
            tools=tools,
        )

        for step in range(max_num_steps):
            response = await client.get_response(messages, options=options)

            # Track tokens
            usage = response.usage_details
            if usage:
                if isinstance(usage, dict):
                    self.tracker.prompt_tokens += usage.get("input_token_count", 0) or 0
                    self.tracker.completion_tokens += usage.get("output_token_count", 0) or 0
                    self.tracker.total_tokens += usage.get("total_token_count", 0) or 0
                else:
                    self.tracker.prompt_tokens += getattr(usage, "input_token_count", 0) or 0
                    self.tracker.completion_tokens += getattr(usage, "output_token_count", 0) or 0
                    self.tracker.total_tokens += getattr(usage, "total_token_count", 0) or 0

            # Collect all response messages
            resp_msgs = response.messages if isinstance(response.messages, list) else ([response.messages] if response.messages else [])
            if not resp_msgs:
                break

            # Find function calls and text in response messages
            func_call_content = None
            text_content = None

            for resp_msg in resp_msgs:
                for c in (resp_msg.contents or []):
                    if c.type == "function_call" and func_call_content is None:
                        func_call_content = c
                    elif c.type == "function_result":
                        # MAF auto-executed our dummy — skip
                        continue
                    elif c.type == "text" and c.text and text_content is None:
                        text_content = c

            if func_call_content:
                # Tool call — execute via env.step, then add result to messages
                args = func_call_content.parse_arguments() if func_call_content.arguments else {}
                if isinstance(args, str):
                    args = json.loads(args)
                action = Action(name=func_call_content.name, kwargs=args)

                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                # Build messages: assistant with function_call + tool result
                call_content = Content.from_function_call(
                    call_id=func_call_content.call_id or f"call_{step}",
                    name=func_call_content.name,
                    arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                )
                result_content = Content.from_function_result(
                    call_id=func_call_content.call_id or f"call_{step}",
                    result=env_response.observation,
                )
                messages.append(Message(role="assistant", contents=[call_content]))
                messages.append(Message(role="tool", contents=[result_content]))

                raw_messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": func_call_content.call_id or f"call_{step}",
                        "type": "function",
                        "function": {
                            "name": func_call_content.name,
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    }],
                })
                raw_messages.append({
                    "role": "tool",
                    "tool_call_id": func_call_content.call_id or f"call_{step}",
                    "name": func_call_content.name,
                    "content": env_response.observation,
                })

                if env_response.done:
                    break
            else:
                # Text response to user
                content = text_content.text if text_content else ""
                if not content:
                    # Try to get text from the first response message
                    content = resp_msgs[0].text or "" if resp_msgs else ""
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                messages.append(Message(role="assistant", contents=[content]))
                raw_messages.append({"role": "assistant", "content": content})

                if env_response.done:
                    break

                messages.append(Message(role="user", contents=[env_response.observation]))
                raw_messages.append({"role": "user", "content": env_response.observation})

        return SolveResult(reward=reward, info=info, messages=raw_messages)
