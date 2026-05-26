"""LangGraph-based agent for TAU Bench."""

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from src.metrics import TokenTracker


def _create_llm(model: str, model_provider: str, temperature: float):
    """Create a LangChain chat model based on provider."""
    if model_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, max_retries=5)
    elif model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, max_retries=5)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


class LangGraphAgent(Agent):
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
        self.tracker.reset()

        llm = _create_llm(self.model, self.model_provider, self.temperature)
        llm_with_tools = llm.bind_tools(self.tools_info, parallel_tool_calls=False)

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        reward = 0.0

        messages = [
            SystemMessage(content=self.wiki),
            HumanMessage(content=obs),
        ]

        for step in range(max_num_steps):
            response: AIMessage = llm_with_tools.invoke(messages)
            self.tracker.track(response)

            if response.tool_calls:
                # Only keep the first tool call (same as tau-bench's approach)
                first_tc = response.tool_calls[0]
                response = AIMessage(
                    content=response.content,
                    tool_calls=[first_tc],
                    id=response.id,
                )
                messages.append(response)

                action = Action(name=first_tc["name"], kwargs=first_tc["args"])
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                messages.append(
                    ToolMessage(
                        content=env_response.observation,
                        tool_call_id=first_tc["id"],
                        name=first_tc["name"],
                    )
                )

                if env_response.done:
                    break
            else:
                content = response.content
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                if env_response.done:
                    break

                messages.append(HumanMessage(content=env_response.observation))

        # Convert messages to dicts for SolveResult
        messages_dicts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                messages_dicts.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                messages_dicts.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                d = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages_dicts.append(d)
            elif isinstance(msg, ToolMessage):
                messages_dicts.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "content": msg.content,
                })

        return SolveResult(reward=reward, info=info, messages=messages_dicts)
