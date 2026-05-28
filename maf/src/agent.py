"""Microsoft Agent Framework (MAF) agent for TAU Bench.

This implementation uses MAF's OFFICIAL recommended pattern: the native ``Agent``
abstraction with ``agent.run()``. The framework auto-executes Python-callable tools
and loops autonomously until the model stops calling tools.

Key idea for tau-bench: BOTH tool execution AND talking to the user go through
``env.step()``. We expose "talk to the user" as just another tool (``talk_to_user``),
so a single ``agent.run()`` drives the entire multi-turn conversation. An
``EnvBridge`` routes every tool invocation (including ``talk_to_user``) to
``env.step()`` and captures the final reward / info / done state.

Token aggregation: ``agent.run()`` makes multiple internal LLM calls. We attach a
``ChatMiddleware`` that, after each chat completion, reads ``context.result.usage_details``
(``input_token_count`` / ``output_token_count`` / ``total_token_count``) and accumulates
it into ``self.tracker``. This reliably yields the cumulative usage for the full episode.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agent_framework import (
    Agent,
    ChatContext,
    ChatMiddleware,
    ChatOptions,
    FunctionTool,
)
from agent_framework_openai import OpenAIChatClient

from tau_bench.agents.base import Agent as TauAgent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from src.metrics import TokenTracker

# MAF's OpenAIChatClient does not auto-read .env; load it at import time so that
# `uv run python run_benchmark.py ...` works directly without sourcing the env.
load_dotenv()

TALK_TOOL_NAME = "talk_to_user"


class EnvBridge:
    """Routes every MAF tool call to ``env.step()`` and captures episode state.

    Both real tau-bench tools and the synthetic ``talk_to_user`` tool flow through
    here. The last observed reward / info / done are stored so the driver can build
    the final ``SolveResult`` regardless of how/when ``agent.run()`` terminates.
    """

    def __init__(self, env: Env, max_num_steps: int = 30):
        self.env = env
        self.reward = 0.0
        self.info: Dict[str, Any] = {}
        self.done = False
        self.max_num_steps = max_num_steps
        self.num_steps = 0
        # OpenAI-format trajectory for logging.
        self.trajectory: List[Dict[str, Any]] = []

    def call_tool(self, name: str, kwargs: Dict[str, Any]) -> str:
        if self.done:
            return "Conversation has ended."
        if self.num_steps >= self.max_num_steps:
            self.done = True
            return "Maximum number of steps reached. Conversation has ended."

        self.num_steps += 1
        r = self.env.step(Action(name=name, kwargs=kwargs))
        self.reward = r.reward
        self.info = {**self.info, **r.info.model_dump()}
        self.done = r.done

        # Log the step in OpenAI format for the trajectory.
        if name == RESPOND_ACTION_NAME:
            # Assistant spoke to the user; observation is the user's reply.
            self.trajectory.append(
                {"role": "assistant", "content": kwargs.get("content", "")}
            )
            if not self.done:
                self.trajectory.append({"role": "user", "content": r.observation})
        else:
            call_id = f"call_{self.num_steps}"
            self.trajectory.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(kwargs),
                            },
                        }
                    ],
                }
            )
            self.trajectory.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": r.observation,
                }
            )

        return r.observation

    def respond(self, content: str) -> str:
        return self.call_tool(RESPOND_ACTION_NAME, {"content": content})


class _TokenTrackingMiddleware(ChatMiddleware):
    """Accumulates per-LLM-call token usage across the whole ``agent.run()``.

    ``agent.run()`` performs multiple internal chat completions (one per tool-calling
    turn). After each completion we read ``context.result.usage_details`` and add it
    to the shared ``TokenTracker`` so the final totals cover the entire episode.
    """

    def __init__(self, tracker: TokenTracker):
        self._tracker = tracker

    async def process(self, context: ChatContext, call_next) -> None:
        await call_next()
        result = getattr(context, "result", None)
        if result is None:
            return
        usage = getattr(result, "usage_details", None)
        if not usage:
            return
        # UsageDetails is a TypedDict (dict-like).
        self._tracker.prompt_tokens += usage.get("input_token_count", 0) or 0
        self._tracker.completion_tokens += usage.get("output_token_count", 0) or 0
        self._tracker.total_tokens += usage.get("total_token_count", 0) or 0


def _make_env_tool(tool_info: Dict[str, Any], bridge: EnvBridge) -> FunctionTool:
    """Build an auto-executable MAF tool from a tau-bench OpenAI-format tool.

    The callable receives the model-supplied arguments as kwargs and forwards them to
    ``env.step()`` via the bridge, returning the env observation string.
    """
    func = tool_info["function"]
    name = func["name"]
    description = func.get("description", "")
    parameters = func.get("parameters", {"type": "object", "properties": {}})

    def _call(**kwargs: Any) -> str:
        return bridge.call_tool(name, kwargs)

    return FunctionTool(
        name=name,
        description=description,
        input_model=parameters,
        func=_call,
    )


def _make_talk_tool(bridge: EnvBridge) -> FunctionTool:
    """The synthetic tool the agent uses for ALL customer communication."""

    def _talk(content: str) -> str:
        return bridge.respond(content)

    schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message to send to the customer.",
            }
        },
        "required": ["content"],
    }
    return FunctionTool(
        name=TALK_TOOL_NAME,
        description=(
            "Send a message to the customer and receive their reply. "
            "Use this for ALL communication with the customer."
        ),
        input_model=schema,
        func=_talk,
    )


class MAFAgent(TauAgent):
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

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        bridge = EnvBridge(env, max_num_steps=max_num_steps)
        bridge.info = env_reset.info.model_dump()

        # Build the auto-executable tool set: every tau-bench tool + talk_to_user.
        tools: List[FunctionTool] = [
            _make_env_tool(t, bridge) for t in self.tools_info
        ]
        tools.append(_make_talk_tool(bridge))

        instructions = (
            self.wiki
            + "\n\n"
            + "You are a customer service agent. To communicate with the customer you "
            + f"MUST call the `{TALK_TOOL_NAME}` tool — never reply with plain text. "
            + "Use the other tools to look up and modify data as needed. "
            + "Make at most ONE tool call at a time and wait for its result before "
            + "deciding the next step. The conversation continues until the customer's "
            + "request is fully resolved."
        )

        client = OpenAIChatClient(
            model=self.model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        agent = Agent(
            client=client,
            instructions=instructions,
            tools=tools,
            middleware=[_TokenTrackingMiddleware(self.tracker)],
            default_options=ChatOptions(
                temperature=self.temperature,
                # Enforce tau-bench's "one tool call at a time" policy
                # (maps to OpenAI parallel_tool_calls=False).
                allow_multiple_tool_calls=False,
            ),
        )

        # Seed the trajectory with the standard system + first-user messages.
        bridge.trajectory.append({"role": "system", "content": self.wiki})
        bridge.trajectory.append({"role": "user", "content": obs})

        # An AgentSession (backed by the default in-memory history provider) keeps
        # the conversation context across successive run() calls.
        session = agent.create_session()

        # `agent.run()` autonomously auto-executes tools (real tau-bench tools AND
        # talk_to_user) and loops until the model stops calling tools. Each run()
        # ends when the model emits a final plain-text turn. In tau-bench, ALL user
        # communication must go through env.step(); models occasionally answer with
        # plain text instead of the talk_to_user tool, which would not advance the
        # conversation. So we wrap run() in an outer loop: whenever a run() finishes
        # with plain text (and the conversation is not yet done), we route that text
        # through the bridge as a RESPOND action and feed the customer's reply back
        # into the next run() — preserving history via the session.
        next_input: Any = obs
        # Generous outer-loop ceiling; the bridge's own max_num_steps + done guard
        # are the real terminators.
        for _ in range(max_num_steps + 5):
            if bridge.done:
                break
            try:
                response = await agent.run(next_input, session=session)
            except Exception:
                # Episode reward/info are captured in the bridge regardless of how
                # run() terminates (e.g. iteration cap or transient model error).
                break

            if bridge.done:
                break

            # run() ended on a plain-text turn that never reached the user. Treat it
            # as a message to the customer and continue with their reply.
            final_text = (response.text or "").strip() if response is not None else ""
            if final_text:
                reply = bridge.respond(final_text)
                if bridge.done:
                    break
                next_input = reply
            else:
                # No text and not done — nothing more to drive the loop with.
                break

        return SolveResult(
            reward=bridge.reward,
            info=bridge.info,
            messages=bridge.trajectory,
        )
