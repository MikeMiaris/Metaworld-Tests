"""
Evaluate final PPO models created by train_push_v3_splits.py.

Run from the folder containing push_v3_ppo_split_runs:
    python evaluate_push_v3_splits_final.py

Useful variants:
    python evaluate_push_v3_splits_final.py --configs base_push careful_push
    python evaluate_push_v3_splits_final.py --groups test --eval-test-episodes-per-task 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import metaworld # type: ignore
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "Meta-World/MT1"
ENV_NAME = "push-v3"
REWARD_TYPE = "v2"
MAX_EPISODE_STEPS = 500
TASK_SEED = 67
DEFAULT_ROOT = "./push_v3_ppo_split_runs"

@dataclass
class RunInfo:
    run_name: str
    config_name: str
    split_id: int
    split_seed: int
    train_seed: int
    total_timesteps: int
    train_idx: List[int]
    test_idx: List[int]
    vecnormalize: bool
    model_path: Path
    vecnormalize_path: Path
    config_path: Path

def parse_str_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for value in values:
        out.extend([p.strip() for p in value.replace(',', ' ').split() if p.strip()])
    return out or None

def parse_int_set(text: str) -> Optional[set[int]]:
    if not text.strip():
        return None
    return {int(x.strip()) for x in text.replace(',', ' ').split() if x.strip()}

def get_task_wrapper(env: gym.Env):
    cur = env
    seen = set()
    while True:
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "tasks") and hasattr(cur, "toggle_sample_tasks_on_reset"):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        unwrapped = getattr(cur, "unwrapped", None)
        if unwrapped is not None and unwrapped is not cur:
            cur = unwrapped
            continue
        break
    raise RuntimeError("Could not find Meta-World task wrapper.")

def make_mt1_env(seed: int, tasks: Optional[Sequence[Any]] = None, sample_on_reset: bool = True,
                 terminate_on_success: bool = True, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(
        ENV_ID,
        env_name=ENV_NAME,
        task_select="pseudorandom",
        terminate_on_success=terminate_on_success,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=seed,
        render_mode=render_mode,
        reward_function_version=REWARD_TYPE,
    )
    wrapper = get_task_wrapper(env)
    if tasks is not None:
        wrapper.tasks = list(tasks)
    wrapper.toggle_sample_tasks_on_reset(sample_on_reset)
    return env

def extract_all_tasks(task_seed: int) -> List[Any]:
    env = make_mt1_env(seed=task_seed, tasks=None, sample_on_reset=True, terminate_on_success=False)
    wrapper = get_task_wrapper(env)
    tasks = list(wrapper.tasks)
    env.close()
    if len(tasks) != 50:
        raise RuntimeError(f"Expected 50 MT1 task variations, got {len(tasks)}")
    return tasks

def make_dummy_env_for_vecnormalize(seed: int = 999) -> DummyVecEnv:
    return DummyVecEnv([lambda: make_mt1_env(seed=seed, tasks=None, sample_on_reset=True, terminate_on_success=True)])

def load_vecnormalize(path: Path) -> Optional[VecNormalize]:
    if not path.exists():
        return None
    dummy = make_dummy_env_for_vecnormalize()
    vecnorm = VecNormalize.load(str(path), dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm

def normalize_obs(vecnorm: Optional[VecNormalize], obs: np.ndarray) -> np.ndarray:
    if vecnorm is None:
        return obs
    return vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))

def discover_runs(root: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for config_path in sorted((root / "models").glob("*/*_config.json")):
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        run_dir = config_path.parent
        run_name = run_dir.name
        runs.append(RunInfo(
            run_name=run_name,
            config_name=str(cfg.get("config_name", "unknown")),
            split_id=int(cfg.get("split_id", -1)),
            split_seed=int(cfg.get("split_seed", -1)),
            train_seed=int(cfg.get("train_seed", -1)),
            total_timesteps=int(cfg.get("total_timesteps", -1)),
            train_idx=[int(x) for x in cfg.get("train_idx", [])],
            test_idx=[int(x) for x in cfg.get("test_idx", [])],
            vecnormalize=bool(cfg.get("vecnormalize", True)),
            model_path=run_dir / f"{run_name}_final.zip",
            vecnormalize_path=run_dir / f"{run_name}_vecnormalize.pkl",
            config_path=config_path,
        ))
    return runs

def evaluate_task_group(model: PPO, vecnorm: Optional[VecNormalize], tasks: Sequence[Any], run: RunInfo,
                        group: str, eval_seed: int, n_episodes_per_task: int,
                        deterministic: bool, terminate_on_success: bool, render: bool) -> List[Dict[str, Any]]:
    env = make_mt1_env(seed=eval_seed, tasks=list(tasks), sample_on_reset=False,
                       terminate_on_success=terminate_on_success, render_mode="human" if render else None)
    base_env = env.unwrapped
    rows: List[Dict[str, Any]] = []
    for task_local_idx, task in enumerate(tasks):
        print(f"    {group}: task {task_local_idx + 1}/{len(tasks)}", end="\r")
        base_env.set_task(task)
        for episode in range(n_episodes_per_task):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0
            episode_steps = 0
            success = 0.0
            first_success_step = np.nan
            while not done:
                model_obs = normalize_obs(vecnorm, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                if render:
                    env.render()
                episode_return += float(reward)
                episode_steps += 1
                if float(info.get("success", 0.0)) > 0.0:
                    success = 1.0
                    if np.isnan(first_success_step):
                        first_success_step = episode_steps
                    if terminate_on_success:
                        done = True
                        break
                done = bool(terminated or truncated)
            rows.append({
                "run_name": run.run_name,
                "config_name": run.config_name,
                "split_id": run.split_id,
                "split_seed": run.split_seed,
                "train_seed": run.train_seed,
                "total_timesteps": run.total_timesteps,
                "group": group,
                "task_local_idx": task_local_idx,
                "episode": episode,
                "success": success,
                "return": episode_return,
                "steps": episode_steps,
                "first_success_step": first_success_step,
                "model_path": str(run.model_path),
                "vecnormalize_path": str(run.vecnormalize_path),
            })
    env.close()
    print(" " * 80, end="\r")
    return rows

def summarize_by_run(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["config_name", "split_id", "split_seed", "train_seed", "total_timesteps", "group"], dropna=False).agg(
        success_rate=("success", "mean"),
        avg_return=("return", "mean"),
        std_return=("return", "std"),
        avg_steps=("steps", "mean"),
        std_steps=("steps", "std"),
        avg_first_success_step=("first_success_step", "mean"),
        median_first_success_step=("first_success_step", "median"),
        episodes=("success", "count"),
        run_name=("run_name", "first"),
        model_path=("model_path", "first"),
        vecnormalize_path=("vecnormalize_path", "first"),
    ).reset_index()

def summarize_across_splits(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(["config_name", "group"], dropna=False).agg(
        mean_success_rate=("success_rate", "mean"),
        std_success_rate=("success_rate", "std"),
        mean_return=("avg_return", "mean"),
        std_return_across_runs=("avg_return", "std"),
        mean_first_success_step=("avg_first_success_step", "mean"),
        std_first_success_step=("avg_first_success_step", "std"),
        runs=("run_name", "count"),
        total_episodes=("episodes", "sum"),
    ).reset_index()

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final push-v3 split models.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--configs", nargs="*", default=None, help="Example: --configs base_push careful_push")
    parser.add_argument("--splits", type=str, default="", help="Example: --splits 0,1")
    parser.add_argument("--groups", type=str, default="train,test", help="train,test or only test")
    parser.add_argument("--eval-train-episodes-per-task", type=int, default=1)
    parser.add_argument("--eval-test-episodes-per-task", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--task-seed", type=int, default=TASK_SEED)
    parser.add_argument("--no-terminate-on-success", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / "eval_final_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_configs = parse_str_list(args.configs)
    selected_splits = parse_int_set(args.splits)
    selected_groups = set(parse_str_list([args.groups]) or ["train", "test"])
    terminate_on_success = not args.no_terminate_on_success
    deterministic = not args.stochastic

    print("=" * 100)
    print("PUSH-V3 FINAL MODEL EVALUATION")
    print("=" * 100)
    print("Root:", root)
    print("Output dir:", output_dir)
    print("Configs:", selected_configs if selected_configs else "ALL")
    print("Splits:", selected_splits if selected_splits is not None else "ALL")
    print("Groups:", selected_groups)
    print("Deterministic:", deterministic)
    print("Terminate on success:", terminate_on_success)
    print("=" * 100)

    print("\nExtracting 50 push-v3 task variations...")
    all_tasks = extract_all_tasks(args.task_seed)
    runs = discover_runs(root)
    if selected_configs is not None:
        runs = [r for r in runs if r.config_name in selected_configs]
    if selected_splits is not None:
        runs = [r for r in runs if r.split_id in selected_splits]
    print(f"Found {len(runs)} candidate runs.")

    all_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        if not run.model_path.exists():
            skipped_rows.append({"run_name": run.run_name, "config_name": run.config_name, "split_id": run.split_id, "reason": "missing_model", "model_path": str(run.model_path)})
            print("Skipping missing model:", run.model_path)
            continue
        if not run.train_idx or not run.test_idx:
            skipped_rows.append({"run_name": run.run_name, "config_name": run.config_name, "split_id": run.split_id, "reason": "missing_split_indices", "model_path": str(run.model_path)})
            print("Skipping run with missing split indices:", run.run_name)
            continue
        train_tasks = [all_tasks[idx] for idx in run.train_idx]
        test_tasks = [all_tasks[idx] for idx in run.test_idx]
        print("\n" + "-" * 100)
        print(f"Run {i}/{len(runs)}: {run.run_name}")
        print(f"Config: {run.config_name} | split={run.split_id} | seed={run.split_seed}")
        print("Model:", run.model_path)
        print("VecNormalize:", run.vecnormalize_path)
        model = PPO.load(str(run.model_path), device="cpu")
        vecnorm = load_vecnormalize(run.vecnormalize_path) if run.vecnormalize else None
        if run.vecnormalize and vecnorm is None:
            print("WARNING: VecNormalize expected but missing. Evaluating without normalization.")
        if "train" in selected_groups:
            print("  Evaluating TRAIN variations...")
            rows = evaluate_task_group(model, vecnorm, train_tasks, run, "train", args.eval_seed + run.split_id, args.eval_train_episodes_per_task, deterministic, terminate_on_success, args.render)
            all_rows.extend(rows)
            tmp = pd.DataFrame(rows)
            print(f"    TRAIN SR={tmp['success'].mean():.3f} | Return={tmp['return'].mean():.2f} | FirstSuccess={tmp['first_success_step'].mean():.1f}")
        if "test" in selected_groups:
            print("  Evaluating TEST variations...")
            rows = evaluate_task_group(model, vecnorm, test_tasks, run, "test", args.eval_seed + 10000 + run.split_id, args.eval_test_episodes_per_task, deterministic, terminate_on_success, args.render)
            all_rows.extend(rows)
            tmp = pd.DataFrame(rows)
            print(f"    TEST  SR={tmp['success'].mean():.3f} | Return={tmp['return'].mean():.2f} | FirstSuccess={tmp['first_success_step'].mean():.1f}")
        if vecnorm is not None:
            vecnorm.close()

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_path = output_dir / "push_v3_eval_skipped.csv"
    skipped_df.to_csv(skipped_path, index=False)
    if not all_rows:
        print("\nNo evaluations were performed.")
        print("Skipped report:", skipped_path)
        return
    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize_by_run(raw_df)
    across_df = summarize_across_splits(summary_df)
    raw_path = output_dir / "push_v3_eval_raw_episodes.csv"
    summary_path = output_dir / "push_v3_eval_summary_by_run.csv"
    across_path = output_dir / "push_v3_eval_summary_across_splits.csv"
    pivot_path = output_dir / "push_v3_eval_success_pivot.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    across_df.to_csv(across_path, index=False)
    pivot = across_df.pivot_table(index="config_name", columns="group", values="mean_success_rate")
    pivot.to_csv(pivot_path)
    print("\n" + "=" * 100)
    print("FINAL SUMMARY ACROSS SPLITS")
    print("=" * 100)
    cols = ["config_name", "group", "mean_success_rate", "std_success_rate", "mean_return", "mean_first_success_step", "runs", "total_episodes"]
    print(across_df[cols].to_string(index=False))
    print("\nSuccess-rate pivot:")
    print(pivot.to_string())
    print("\nSaved files:")
    print("Raw episodes:", raw_path)
    print("Summary by run:", summary_path)
    print("Across splits:", across_path)
    print("Pivot:", pivot_path)
    print("Skipped:", skipped_path)

if __name__ == "__main__":
    main()
