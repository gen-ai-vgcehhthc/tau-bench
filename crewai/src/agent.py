"""CrewAI-based agent for TAU Bench.

Uses CrewAI's native Agent abstraction (``crewai.Agent`` driven by
``Agent.kickoff()`` and ``crewai.tools.BaseTool``) rather than calling
``litellm.completion()`` directly.

Key idea: in tau-bench BOTH tool execution AND talking to the user go through
``env.step()`` (an ``EnvBridge`` routes everything through it). Each tau-bench
tool is wrapped as a native ``crewai.tools.BaseTool`` with a dynamically-built
pydantic ``args_schema``, and the agent is a native ``crewai.Agent`` whose
internal LiteAgent loop calls those tools.

PREFERRED design (item 5 of the spec): expose "talk to the user" as a
``talk_to_user`` BaseTool and let a single ``agent.kickoff()`` drive the whole
multi-turn conversation. We tested this first but it terminated after ONE turn:
CrewAI's LiteAgent treats any plain-text model output as the agent's final
answer, so gpt-4o-mini emitted its greeting as text and the kickoff finished
without ever looping (~3k tokens, one assistant message).

FALLBACK design (item 6, what we actually use): drive the conversation
turn-by-turn ourselves. Each turn we call ``agent.kickoff(messages=history)``;
the agent uses the real tau-bench tools internally during that kickoff, then
returns its reply in ``result.raw``. We relay that reply to the customer via the
bridge, append assistant+user to ``history``, and repeat until the env signals
done or we hit ``max_num_steps``. Token usage is accumulated from each turn's
``result.usage_metrics``. In this mode ``talk_to_user`` is unnecessary (we own
the loop), so the env tools are the only tools given to the agent.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model

from crewai import Agent as CrewAgent, LLM
from crewai.tools import BaseTool

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME

from src.metrics import TokenTracker


# ---------------------------------------------------------------------------
# Environment bridge: holds episode state and routes everything through env.step
# ---------------------------------------------------------------------------
class EnvBridge:
    """Holds the tau-bench env plus running episode state.

    Both real tool calls and "talk to the user" go through ``env.step`` here.
    Replaces the module-level globals the previous implementation would need.
    """

    def __init__(self, env: Env):
        self.env = env
        self.reward = 0.0
        self.info: Dict[str, Any] = {}
        self.done = False
        # OpenAI-format trajectory, populated as tools fire (for logging only).
        self.messages: List[Dict[str, Any]] = []

    def call_tool(self, name: str, kwargs: Dict[str, Any]) -> str:
        if self.done:
            return "Conversation has ended."
        r = self.env.step(Action(name=name, kwargs=kwargs))
        self.reward = r.reward
        self.info = {**self.info, **r.info.model_dump()}
        self.done = r.done

        # Log the assistant tool call + tool result (or user reply for respond).
        if name == RESPOND_ACTION_NAME:
            self.messages.append({"role": "assistant", "content": kwargs.get("content", "")})
            if not self.done:
                self.messages.append({"role": "user", "content": r.observation})
        else:
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(kwargs)},
                }],
            })
            self.messages.append({"role": "tool", "name": name, "content": r.observation})
        return r.observation

    def respond(self, content: str) -> str:
        return self.call_tool(RESPOND_ACTION_NAME, {"content": content})


# ---------------------------------------------------------------------------
# JSON-schema -> pydantic args_schema (for building BaseTool subclasses)
# ---------------------------------------------------------------------------
def _json_schema_to_python(prop: Dict[str, Any]) -> Any:
    """Map a single JSON-schema property to a Python type annotation."""
    t = prop.get("type", "string")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        items = prop.get("items", {})
        inner = _json_schema_to_python(items) if items else Any
        return List[inner]
    if t == "object":
        return Dict[str, Any]
    return Any


def _build_args_schema(name: str, parameters: Dict[str, Any]) -> Type[BaseModel]:
    """Build a pydantic model from a JSON-schema ``parameters`` block."""
    props = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])

    fields: Dict[str, Any] = {}
    for pname, pschema in props.items():
        py_type = _json_schema_to_python(pschema)
        desc = pschema.get("description", "")
        if pname in required:
            fields[pname] = (py_type, Field(..., description=desc))
        else:
            fields[pname] = (Optional[py_type], Field(default=None, description=desc))

    if not fields:
        # create_model needs at least the model name; empty schema is fine.
        return create_model(f"{name}_Args")
    return create_model(f"{name}_Args", **fields)


def _make_tool_class(tool_name: str, tool_desc: str, schema: Type[BaseModel], bridge: EnvBridge) -> BaseTool:
    """Create a BaseTool instance whose _run routes through the bridge."""
    _tool_name = tool_name
    _tool_desc = tool_desc or f"Call the {tool_name} tool."
    _bridge = bridge

    class _EnvTool(BaseTool):
        name: str = _tool_name
        description: str = _tool_desc
        args_schema: Type[BaseModel] = schema

        def _run(self, **kwargs: Any) -> str:
            # Drop None values that pydantic injected for optional args.
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            return _bridge.call_tool(_tool_name, cleaned)

    return _EnvTool()


def _build_talk_to_user_tool(bridge: EnvBridge) -> BaseTool:
    """The `talk_to_user` tool for the single-kickoff (PREFERRED) design.

    Kept for the PREFERRED approach where CrewAI's tool loop drives the whole
    conversation. We ended up on the FALLBACK turn-by-turn design (see module
    docstring and ``solve``) because CrewAI's LiteAgent treats any plain-text
    model output as a final answer, so the single kickoff terminated after one
    turn instead of looping. This helper is therefore unused in FALLBACK mode
    but retained to document the intended PREFERRED tool.
    """

    class _TalkArgs(BaseModel):
        content: str = Field(..., description="The message to send to the customer.")

    class _TalkTool(BaseTool):
        name: str = "talk_to_user"
        description: str = (
            "Send a message to the customer and receive their reply. "
            "Use this for ALL communication with the customer."
        )
        args_schema: Type[BaseModel] = _TalkArgs

        def _run(self, content: str) -> str:
            return bridge.respond(content)

    return _TalkTool()


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

    def _model_str(self) -> str:
        if self.model_provider == "openai":
            return f"openai/{self.model}"
        if self.model_provider == "anthropic":
            return f"anthropic/{self.model}"
        return self.model

    def _build_env_tools(self, bridge: EnvBridge) -> List[BaseTool]:
        tools: List[BaseTool] = []
        for entry in self.tools_info:
            func = entry["function"]
            tname = func["name"]
            tdesc = func.get("description", "")
            params = func.get("parameters", {}) or {}
            args_schema = _build_args_schema(tname, params)
            tools.append(_make_tool_class(tname, tdesc, args_schema, bridge))
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
        info = env_reset.info.model_dump()

        bridge = EnvBridge(env)
        bridge.info = info

        # FALLBACK mode (see module docstring / report): the env tools are real
        # BaseTools routed through the bridge so the agent CAN look up / modify
        # data within each kickoff. talk_to_user is NOT used here -- WE drive the
        # back-and-forth conversation loop by relaying each turn's reply to the
        # user via the bridge.
        env_tools = self._build_env_tools(bridge)

        llm = LLM(model=self._model_str(), temperature=self.temperature)

        goal = (
            "Resolve the customer's request by following the policy exactly.\n\n"
            "RULES:\n"
            "- Use the data tools (find_user_id_by_name_zip, get_order_details, "
            "get_product_details, etc.) to look up and modify data as needed.\n"
            "- Make at most ONE data-modifying tool call at a time, and always get "
            "explicit customer confirmation before any write/modify/cancel action.\n"
            "- Your Final Answer for each turn is the single message that will be sent "
            "to the customer: ask for any info you still need, confirm actions, or "
            "deliver the result. Keep it concise and do not invent information.\n"
            "- Do not transfer to a human unless the policy requires it."
        )

        agent = CrewAgent(
            role="Customer Service Agent",
            goal=goal,
            backstory=self.wiki,
            tools=env_tools,
            llm=llm,
            max_iter=max_num_steps,
            allow_delegation=False,
            verbose=False,
        )

        def _track(result_obj) -> None:
            usage = getattr(result_obj, "usage_metrics", None)
            if not usage:
                return
            if not isinstance(usage, dict):
                usage = getattr(usage, "model_dump", lambda: {})()
            self.tracker.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.tracker.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.tracker.total_tokens += int(usage.get("total_tokens", 0) or 0)

        # Conversation history we feed back into each kickoff. The agent calls the
        # real tau-bench tools internally during a kickoff (logged via the bridge),
        # then returns the turn's reply in result.raw, which we relay to the user.
        history: List[Dict[str, str]] = [{"role": "user", "content": obs}]

        for _ in range(max_num_steps):
            result = agent.kickoff(messages=list(history))
            _track(result)

            reply = (getattr(result, "raw", None) or "").strip()

            # Send this turn's reply to the customer and get their next message.
            user_reply = bridge.respond(reply)
            history.append({"role": "assistant", "content": reply})

            if bridge.done:
                break

            history.append({"role": "user", "content": user_reply})

        # Build the OpenAI-format trajectory for logging: system + first user,
        # then everything the bridge recorded (real tool calls interleaved with
        # the assistant replies and user messages).
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        messages.extend(bridge.messages)

        return SolveResult(reward=bridge.reward, info=bridge.info, messages=messages)
