import time
from dataclasses import dataclass, field


@dataclass
class TaskMetrics:
    task_id: int = 0
    reward: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    num_steps: int = 0
    info: dict = field(default_factory=dict)


class TokenTracker:
    """Accumulates token usage across multiple LLM calls within a single task."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def track(self, response):
        """Extract token usage from a ChatModel response (AIMessage with usage_metadata)."""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.prompt_tokens += usage.get("input_tokens", 0)
            self.completion_tokens += usage.get("output_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class Timer:
    """Simple wall-clock timer context manager."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
