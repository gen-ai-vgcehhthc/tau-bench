"""CrewAI-based agent for TAU Bench.

Uses litellm (CrewAI's underlying LLM layer) for direct tool-calling control,
since CrewAI's Crew/Task abstraction doesn't support tau-bench's interactive
multi-turn conversation model.
"""

import json
from typing import Any, Dict, List, Optional

import litellm

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from src.metrics import TokenTracker


def _convert_tools_to_openai(tools_info: List[Dict[str, Any]]) -> list:
    """Convert tau-bench tools_info to OpenAI function-calling format."""
    tools = list(tools_info)  # already in OpenAI format
    # Add respond_to_user tool
    tools.append({
        "type": "function",
        "function": {
            "name": RESPOND_ACTION_NAME,
            "description": "Send a message to the user and receive their reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The message to send to the user",
                    }
                },
                "required": ["content"],
            },
        },
    })
    return tools


class CrewAIAgent(Agent):
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

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        reward = 0.0

        # Build model string for litellm
        if self.model_provider == "openai":
            model_str = f"openai/{self.model}"
        elif self.model_provider == "anthropic":
            model_str = f"anthropic/{self.model}"
        else:
            model_str = self.model

        tools = _convert_tools_to_openai(self.tools_info)

        messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        for step in range(max_num_steps):
            response = litellm.completion(
                model=model_str,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                num_retries=5,
            )

            # Track tokens
            usage = response.usage
            if usage:
                self.tracker.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.tracker.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                self.tracker.total_tokens += getattr(usage, "total_tokens", 0) or 0

            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                # Process only the first tool call
                tc = msg.tool_calls[0]
                func_name = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                # Add assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }],
                })

                # Execute via env.step
                action = Action(name=func_name, kwargs=args)
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                # Add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": func_name,
                    "content": env_response.observation,
                })

                if env_response.done:
                    break
            else:
                # Text response — send to user via respond action
                content = msg.content or ""
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                messages.append({"role": "assistant", "content": content})

                if env_response.done:
                    break

                # Add user reply
                messages.append({"role": "user", "content": env_response.observation})

        return SolveResult(reward=reward, info=info, messages=messages)
