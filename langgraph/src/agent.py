"""LangGraph-based agent for TAU Bench.

Uses LangGraph's official recommended pattern (`create_react_agent`) instead of a
hand-rolled tool-calling loop. The key idea for tau-bench is that BOTH executing a
real tool AND talking to the user go through `env.step()`. So we expose "talk to the
user" as a tool (`talk_to_user`) alongside the real tau-bench tools, and let
`create_react_agent` autonomously drive the entire multi-turn conversation.
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

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


# JSON-schema primitive type -> Python type.
_JSON_PRIMITIVE_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
}


def _json_type_to_python(schema: Dict[str, Any]) -> Any:
    """Map a JSON-schema property definition to a Python type annotation."""
    jtype = schema.get("type")
    if jtype == "array":
        items = schema.get("items") or {}
        inner = _json_type_to_python(items) if items else Any
        return List[inner]
    return _JSON_PRIMITIVE_TO_PY.get(jtype, Any)


def _build_args_schema(tool_name: str, parameters: Dict[str, Any]):
    """Dynamically build a pydantic model for a tool's args from its JSON schema."""
    properties = (parameters or {}).get("properties", {}) or {}
    required = set((parameters or {}).get("required", []) or [])

    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _json_type_to_python(prop_schema)
        description = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            # Optional fields default to None.
            fields[prop_name] = (Optional[py_type], Field(default=None, description=description))

    model_name = f"{tool_name}_Args"
    return create_model(model_name, **fields)


class EnvBridge:
    """Bridges tool/user calls from the LangGraph agent into tau-bench's env.step()."""

    def __init__(self, env: Env):
        self.env = env
        self.reward = 0.0
        self.info: Dict[str, Any] = {}
        self.done = False

    def call_tool(self, name: str, kwargs: Dict[str, Any]) -> str:
        if self.done:
            return "Conversation has ended."
        # Drop None-valued optional kwargs so tau-bench tools receive only what was set.
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        r = self.env.step(Action(name=name, kwargs=clean_kwargs))
        self.reward = r.reward
        self.info = {**self.info, **r.info.model_dump()}
        self.done = r.done
        return r.observation

    def respond(self, content: str) -> str:
        return self.call_tool(RESPOND_ACTION_NAME, {"content": content})


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

    def _build_tools(self, bridge: EnvBridge) -> List[StructuredTool]:
        """Build StructuredTools for every tau-bench tool plus a talk_to_user tool."""
        tools: List[StructuredTool] = []

        for spec in self.tools_info:
            fn = spec["function"]
            tool_name = fn["name"]
            description = fn.get("description", "")
            parameters = fn.get("parameters", {}) or {}
            args_schema = _build_args_schema(tool_name, parameters)

            def make_func(name: str):
                def _func(**kwargs):
                    return bridge.call_tool(name, kwargs)
                return _func

            tools.append(
                StructuredTool.from_function(
                    func=make_func(tool_name),
                    name=tool_name,
                    description=description,
                    args_schema=args_schema,
                )
            )

        def _talk_to_user(content: str) -> str:
            return bridge.respond(content)

        tools.append(
            StructuredTool.from_function(
                func=_talk_to_user,
                name="talk_to_user",
                description=(
                    "Send a message to the customer and receive their reply. "
                    "Use this for ALL communication with the customer."
                ),
            )
        )

        return tools

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        self.tracker.reset()

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation

        bridge = EnvBridge(env)
        bridge.info = env_reset.info.model_dump()

        llm = _create_llm(self.model, self.model_provider, self.temperature)
        tools = self._build_tools(bridge)

        system_prompt = (
            self.wiki
            + "\n\n"
            + "You communicate with the customer ONLY through the `talk_to_user` tool: "
            + "use it to send any message and to receive the customer's reply. Do not "
            + "write replies as plain assistant text. Make at most ONE tool call at a time "
            + "and wait for its result before deciding the next action."
        )

        # Disable parallel tool calls at the model level so only one tool runs per turn.
        try:
            model_for_agent = llm.bind(parallel_tool_calls=False)
        except Exception:
            model_for_agent = llm

        # Safety net: create_react_agent stops the loop the moment the model emits an AI
        # message with no tool calls. In tau-bench, plain assistant text never reaches the
        # customer (everything must go through env.step()). So if the model replies with
        # bare text instead of calling talk_to_user, rewrite that message in-place into a
        # talk_to_user tool call (matching message id => the reducer replaces it), keeping
        # the conversation going through the env.
        def _post_model_hook(state):
            msgs = state["messages"]
            if not msgs:
                return {}
            last = msgs[-1]
            if not isinstance(last, AIMessage):
                return {}
            if last.tool_calls:
                return {}
            text = last.content
            if isinstance(text, list):
                text = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )
            if not text or not str(text).strip():
                return {}
            rewritten = AIMessage(
                content="",
                id=last.id,
                tool_calls=[{
                    "name": "talk_to_user",
                    "args": {"content": text},
                    "id": f"talk_{last.id}",
                }],
            )
            return {"messages": [rewritten]}

        graph = create_react_agent(
            model=model_for_agent,
            tools=tools,
            prompt=system_prompt,
            post_model_hook=_post_model_hook,
        )

        # Collect messages as the graph runs; streaming with stream_mode="values" lets us
        # capture the latest message state even if the run terminates early.
        final_messages: List[Any] = []
        config = {"recursion_limit": max_num_steps * 2}
        try:
            for state in graph.stream(
                {"messages": [HumanMessage(content=obs)]},
                config=config,
                stream_mode="values",
            ):
                if isinstance(state, dict) and "messages" in state:
                    final_messages = state["messages"]
                if bridge.done:
                    break
        except GraphRecursionError:
            # Conversation likely ended (bridge.done) or hit the step cap. The reward is
            # already captured in the bridge; keep whatever messages we collected.
            pass
        except Exception:
            # Be defensive: never let an agent-internal failure crash the benchmark run.
            pass

        # Accumulate token usage from every AI message in the trajectory.
        for msg in final_messages:
            self.tracker.track(msg)

        messages_dicts = self._to_openai_messages(obs, final_messages)

        return SolveResult(
            reward=bridge.reward,
            info=bridge.info,
            messages=messages_dicts,
        )

    def _to_openai_messages(self, obs: str, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert LangGraph message objects to OpenAI-format trajectory dicts."""
        out: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System prompt already captured above; skip duplicates.
                continue
            elif isinstance(msg, HumanMessage):
                # The initial user message is already captured; only emit subsequent ones.
                if msg.content == obs:
                    continue
                out.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                d: Dict[str, Any] = {"role": "assistant", "content": content}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                out.append(d)
            elif isinstance(msg, ToolMessage):
                out.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": getattr(msg, "name", None),
                    "content": msg.content,
                })

        return out
