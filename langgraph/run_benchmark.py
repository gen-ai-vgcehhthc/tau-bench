"""Run TAU Bench with LangGraph agent."""

import argparse
import json
import os
import traceback
from datetime import datetime

from tqdm import tqdm

from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult

from src.agent import LangGraphAgent
from src.metrics import TaskMetrics, Timer

FRAMEWORK = "langgraph"


def run_benchmark(
    env_name: str = "retail",
    model: str = "gpt-4o",
    model_provider: str = "openai",
    user_model: str = "gpt-4o",
    user_model_provider: str = "openai",
    temperature: float = 0.0,
    task_split: str = "test",
    start_index: int = 0,
    end_index: int = -1,
    task_ids: list[int] | None = None,
    log_dir: str = "../results",
):
    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    result_path = f"{log_dir}/{FRAMEWORK}-{model.split('/')[-1]}_{env_name}_{time_str}.json"

    env = get_env(
        env_name,
        user_strategy="llm",
        user_model=user_model,
        user_provider=user_model_provider,
        task_split=task_split,
        task_index=0,
    )

    agent = LangGraphAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=model,
        model_provider=model_provider,
        temperature=temperature,
    )

    total_tasks = len(env.tasks)
    actual_end = total_tasks if end_index == -1 else min(end_index, total_tasks)
    idxs = task_ids if task_ids else list(range(start_index, actual_end))

    print(f"Framework: {FRAMEWORK}")
    print(f"Running {len(idxs)} tasks from {env_name} domain")
    print(f"Model: {model} ({model_provider})")
    print(f"Results: {result_path}")
    print("=" * 60)

    all_metrics: list[dict] = []
    all_results: list[dict] = []
    pass_count = 0

    pbar = tqdm(idxs, desc=FRAMEWORK, unit="task")
    for idx in pbar:
        isolated_env = get_env(
            env_name,
            user_strategy="llm",
            user_model=user_model,
            user_provider=user_model_provider,
            task_split=task_split,
            task_index=idx,
        )

        try:
            with Timer() as timer:
                result = agent.solve(env=isolated_env, task_index=idx)

            metrics = TaskMetrics(
                task_id=idx,
                reward=result.reward,
                prompt_tokens=agent.tracker.prompt_tokens,
                completion_tokens=agent.tracker.completion_tokens,
                total_tokens=agent.tracker.total_tokens,
                wall_time_seconds=timer.elapsed,
                num_steps=len([m for m in result.messages if m.get("role") == "assistant"]),
                info=result.info,
            )
            passed = result.reward >= 1.0 - 1e-6
            status = "PASS" if passed else "FAIL"
            if passed:
                pass_count += 1
            tqdm.write(f"Task {idx:>3} [{status}] reward={result.reward:.2f} "
                       f"tokens={metrics.total_tokens} time={timer.elapsed:.1f}s")
            done = len(all_metrics) + 1
            pbar.set_postfix(passed=f"{pass_count}/{done}")
        except Exception as e:
            tqdm.write(f"Task {idx:>3} [ERROR] {e}")
            traceback.print_exc()
            metrics = TaskMetrics(task_id=idx, reward=0.0, info={"error": str(e)})
            result = None

        all_metrics.append({
            "task_id": metrics.task_id,
            "reward": metrics.reward,
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "total_tokens": metrics.total_tokens,
            "wall_time_seconds": round(metrics.wall_time_seconds, 2),
            "num_steps": metrics.num_steps,
        })

        if result:
            all_results.append(EnvRunResult(
                task_id=idx, reward=result.reward, info=result.info,
                traj=result.messages, trial=0,
            ).model_dump())

        _save(result_path, all_metrics, all_results)

    _print_summary(all_metrics)
    print(f"\nResults saved to: {result_path}")


def _save(path, metrics, results):
    summary = _summary(metrics)
    with open(path, "w") as f:
        json.dump({"framework": FRAMEWORK, "metrics": metrics, "results": results, "summary": summary}, f, indent=2, default=str)


def _summary(metrics):
    total = len(metrics)
    if total == 0:
        return {}
    passed = sum(1 for m in metrics if m["reward"] >= 1.0 - 1e-6)
    return {
        "total_tasks": total, "passed": passed,
        "success_rate": round(passed / total, 4),
        "avg_tokens_per_task": round(sum(m["total_tokens"] for m in metrics) / total, 1),
        "avg_prompt_tokens": round(sum(m["prompt_tokens"] for m in metrics) / total, 1),
        "avg_completion_tokens": round(sum(m["completion_tokens"] for m in metrics) / total, 1),
        "avg_time_per_task": round(sum(m["wall_time_seconds"] for m in metrics) / total, 2),
        "total_tokens": sum(m["total_tokens"] for m in metrics),
        "total_time": round(sum(m["wall_time_seconds"] for m in metrics), 2),
    }


def _print_summary(all_metrics):
    s = _summary(all_metrics)
    if not s:
        return
    print("\n" + "=" * 60)
    print(f"SUMMARY ({FRAMEWORK})")
    print("=" * 60)
    print(f"Tasks: {s['total_tasks']}")
    print(f"Success rate: {s['passed']}/{s['total_tasks']} ({s['success_rate']*100:.1f}%)")
    print(f"Avg tokens/task: {s['avg_tokens_per_task']:.0f}")
    print(f"Avg time/task: {s['avg_time_per_task']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run TAU Bench with {FRAMEWORK}")
    parser.add_argument("--env", dest="env_name", default="retail", choices=["retail", "airline"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--user-model", default="gpt-4o")
    parser.add_argument("--user-model-provider", default="openai")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--task-split", default="test", choices=["train", "test", "dev"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--log-dir", default="../results")
    args = parser.parse_args()
    run_benchmark(**vars(args))
