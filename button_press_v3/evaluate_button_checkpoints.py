"""
Evaluate button-press-v3 PPO checkpoints.

This script is designed for the folder produced by train_button_spilts.py:

button-press_v3_ppo_split_runs/
    checkpoints/
        <run_name>/
            <run_name>_100000_steps.zip
            <run_name>_vecnormalize_100000_steps.pkl
            ...
    models/
        <run_name>/
            <run_name>_config.json
            <run_name>_final.zip
            <run_name>_vecnormalize.pkl
    manifests/
    results/

What it does:
    - Finds all runs from models/*/*_config.json
    - For each run, finds all checkpoint models
    - Loads the matching VecNormalize checkpoint
    - Reconstructs the same 45/5 train/test task split
    - Evaluates every checkpoint on train and test task variations
    - Saves raw episode results and checkpoint summary CSVs

Usage:
    python evaluate_button_checkpoints.py

Faster test:
    python evaluate_button_checkpoints.py --configs base_button --splits 0 --steps 100000,200000

Evaluate only test tasks:
    python evaluate_button_checkpoints.py --groups test

More evaluation episodes:
    python evaluate_button_checkpoints.py --eval-train-episodes-per-task 2 --eval-test-episodes-per-task 10

Plot learning curve later using:
    button_checkpoint_eval_results/button_checkpoint_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import metaworld # type: ignore
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =============================================================================
# Defaults matching your button-press training script
# =============================================================================

ENV_ID = "Meta-World/MT1"
ENV_NAME = "button-press-v3"
REWARD_TYPE = "v2"
MAX_EPISODE_STEPS = 500
TASK_SEED = 67


# =============================================================================
# Meta-World task helpers
# =============================================================================

def get_task_wrapper(env: gym.Env):
    """
    Find the Meta-World wrapper that stores .tasks and task sampling controls.
    This mirrors the helper used in the training script.
    """
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


def make_mt1_env(
    seed: int,
    tasks: Optional[Sequence[Any]] = None,
    sample_on_reset: bool = False,
    terminate_on_success: bool = True,
    reward_type: str = REWARD_TYPE,
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> gym.Env:
    env = gym.make(
        ENV_ID,
        env_name=ENV_NAME,
        task_select="pseudorandom",
        terminate_on_success=terminate_on_success,
        max_episode_steps=max_episode_steps,
        seed=seed,
        render_mode=None,
        reward_function_version=reward_type,
    )

    if tasks is not None:
        wrapper = get_task_wrapper(env)
        wrapper.tasks = list(tasks)
        wrapper.toggle_sample_tasks_on_reset(sample_on_reset)
    else:
        try:
            env.get_wrapper_attr("toggle_sample_tasks_on_reset")(sample_on_reset)
        except Exception:
            pass

    return env


def extract_all_tasks(task_seed: int) -> List[Any]:
    env = make_mt1_env(
        seed=task_seed,
        tasks=None,
        sample_on_reset=True,
        terminate_on_success=False,
    )

    wrapper = get_task_wrapper(env)
    tasks = list(wrapper.tasks)

    env.close()

    if len(tasks) != 50:
        raise RuntimeError(f"Expected 50 MT1 task variations, got {len(tasks)}")

    return tasks


def make_dummy_vec_env_for_vecnormalize(seed: int) -> DummyVecEnv:
    """
    VecNormalize.load needs a VecEnv. We only need it to restore normalization stats.
    The actual evaluation is done manually in a normal MT1 env.
    """
    return DummyVecEnv([
        lambda: make_mt1_env(
            seed=seed,
            tasks=None,
            sample_on_reset=True,
            terminate_on_success=True,
        )
    ])


def load_vecnormalize_if_exists(vec_path: Optional[Path]) -> Optional[VecNormalize]:
    if vec_path is None or not vec_path.exists():
        return None

    dummy_env = make_dummy_vec_env_for_vecnormalize(seed=999)
    vecnorm = VecNormalize.load(str(vec_path), dummy_env)

    # Critical for evaluation:
    # Do not update normalization statistics, and report raw rewards.
    vecnorm.training = False
    vecnorm.norm_reward = False

    return vecnorm


def normalize_obs_for_model(vecnorm: Optional[VecNormalize], obs: np.ndarray) -> np.ndarray:
    if vecnorm is None:
        return obs

    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vecnorm.normalize_obs(obs_batch)


# =============================================================================
# Checkpoint discovery
# =============================================================================

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
    config_path: Path
    checkpoint_dir: Path


@dataclass
class CheckpointInfo:
    step: int
    model_path: Path
    vecnormalize_path: Optional[Path]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_runs(root: Path) -> List[RunInfo]:
    model_root = root / "models"
    checkpoint_root = root / "checkpoints"

    config_paths = sorted(model_root.glob("*/*_config.json"))
    runs: List[RunInfo] = []

    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        run_name = config_path.parent.name
        checkpoint_dir = checkpoint_root / run_name

        runs.append(
            RunInfo(
                run_name=run_name,
                config_name=str(cfg["config_name"]),
                split_id=int(cfg["split_id"]),
                split_seed=int(cfg["split_seed"]),
                train_seed=int(cfg["train_seed"]),
                total_timesteps=int(cfg["total_timesteps"]),
                train_idx=[int(x) for x in cfg["train_idx"]],
                test_idx=[int(x) for x in cfg["test_idx"]],
                config_path=config_path,
                checkpoint_dir=checkpoint_dir,
            )
        )

    return runs


def discover_checkpoints(run: RunInfo) -> List[CheckpointInfo]:
    if not run.checkpoint_dir.exists():
        return []

    checkpoints: List[CheckpointInfo] = []

    pattern = re.compile(r"_(\d+)_steps\.zip$")

    for model_path in sorted(run.checkpoint_dir.glob("*.zip")):
        # Skip anything that is not a checkpoint with _<step>_steps.zip
        match = pattern.search(model_path.name)
        if not match:
            continue

        step = int(match.group(1))

        # CheckpointCallback with save_vecnormalize=True usually saves:
        # <run_name>_vecnormalize_<step>_steps.pkl
        vec_path = run.checkpoint_dir / f"{run.run_name}_vecnormalize_{step}_steps.pkl"

        if not vec_path.exists():
            vec_path = None

        checkpoints.append(
            CheckpointInfo(
                step=step,
                model_path=model_path,
                vecnormalize_path=vec_path,
            )
        )

    checkpoints.sort(key=lambda x: x.step)
    return checkpoints


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_checkpoint_on_task_list(
    model: PPO,
    vecnorm: Optional[VecNormalize],
    tasks: Sequence[Any],
    env_seed: int,
    n_episodes_per_task: int,
    deterministic: bool,
    terminate_on_success: bool,
    reward_type: str,
    max_episode_steps: int,
    group_name: str,
    run: RunInfo,
    checkpoint: CheckpointInfo,
) -> List[Dict[str, Any]]:
    env = make_mt1_env(
        seed=env_seed,
        tasks=list(tasks),
        sample_on_reset=False,
        terminate_on_success=terminate_on_success,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
    )

    base_env = env.unwrapped
    rows: List[Dict[str, Any]] = []

    for task_local_idx, task in enumerate(tasks):
        sys.stdout.write(
            f"\r    {group_name}: task {task_local_idx + 1}/{len(tasks)} "
            f"| checkpoint={checkpoint.step} ..."
        )
        sys.stdout.flush()

        # Fixed task variation.
        base_env.set_task(task)

        for ep in range(n_episodes_per_task):
            obs, _ = env.reset()

            done = False
            ep_return = 0.0
            ep_steps = 0
            ep_success = 0.0
            first_success_step = np.nan

            while not done:
                model_obs = normalize_obs_for_model(vecnorm, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)

                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]

                obs, reward, terminated, truncated, info = env.step(action)

                ep_return += float(reward)
                ep_steps += 1

                current_success = float(info.get("success", 0.0))

                if current_success > 0.0:
                    ep_success = 1.0

                    if np.isnan(first_success_step):
                        first_success_step = ep_steps

                done = bool(terminated or truncated)

            rows.append(
                {
                    "run_name": run.run_name,
                    "config_name": run.config_name,
                    "split_id": run.split_id,
                    "split_seed": run.split_seed,
                    "train_seed": run.train_seed,
                    "total_timesteps": run.total_timesteps,
                    "checkpoint_step": checkpoint.step,
                    "group": group_name,
                    "task_local_idx": task_local_idx,
                    "episode": ep,
                    "success": ep_success,
                    "return": ep_return,
                    "steps": ep_steps,
                    "first_success_step": first_success_step,
                    "model_path": str(checkpoint.model_path),
                    "vecnormalize_path": str(checkpoint.vecnormalize_path) if checkpoint.vecnormalize_path else "",
                }
            )

    env.close()

    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()

    return rows


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(
        [
            "config_name",
            "split_id",
            "split_seed",
            "train_seed",
            "checkpoint_step",
            "group",
        ],
        dropna=False,
    )

    summary = grouped.agg(
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

    return summary


def summarize_across_splits(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = summary_df.groupby(
        ["config_name", "checkpoint_step", "group"],
        dropna=False,
    )

    out = grouped.agg(
        mean_success_rate=("success_rate", "mean"),
        std_success_rate=("success_rate", "std"),
        mean_return=("avg_return", "mean"),
        std_return_across_splits=("avg_return", "std"),
        mean_first_success_step=("avg_first_success_step", "mean"),
        std_first_success_step=("avg_first_success_step", "std"),
        n_runs=("run_name", "count"),
    ).reset_index()

    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate button-press-v3 PPO checkpoints across train/test splits."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="./button-press_v3_ppo_split_runs",
        help="Root folder containing models/, checkpoints/, results/.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save checkpoint evaluation CSVs. Default: <root>/results/checkpoint_eval",
    )

    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated config names to include, e.g. base_button,careful_button. Empty = all.",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated split IDs to include, e.g. 0,1. Empty = all.",
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated checkpoint steps to include, e.g. 100000,200000. Empty = all.",
    )

    parser.add_argument(
        "--groups",
        type=str,
        default="train,test",
        help="Comma-separated groups to evaluate: train,test or only test.",
    )

    parser.add_argument(
        "--eval-train-episodes-per-task",
        type=int,
        default=1,
        help="Episodes per train task variation.",
    )

    parser.add_argument(
        "--eval-test-episodes-per-task",
        type=int,
        default=5,
        help="Episodes per test task variation.",
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1000,
        help="Base evaluation seed.",
    )

    parser.add_argument(
        "--task-seed",
        type=int,
        default=TASK_SEED,
        help="Seed used to extract the 50 MT1 task variations.",
    )

    parser.add_argument(
        "--reward-type",
        type=str,
        default=REWARD_TYPE,
        choices=["v1", "v2"],
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=MAX_EPISODE_STEPS,
    )

    parser.add_argument(
        "--no-terminate-on-success",
        action="store_true",
        help="If set, evaluation episodes continue until max_episode_steps instead of stopping on success.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if args.output_dir is None:
        output_dir = root / "results" / "checkpoint_eval"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = set(parse_str_list(args.configs)) if args.configs else None
    selected_splits = set(parse_int_list(args.splits)) if args.splits else None
    selected_steps = set(parse_int_list(args.steps)) if args.steps else None
    selected_groups = set(parse_str_list(args.groups))

    terminate_on_success = not args.no_terminate_on_success
    deterministic = not args.stochastic

    print("=" * 110)
    print("BUTTON-PRESS-V3 CHECKPOINT EVALUATION")
    print("=" * 110)
    print(f"Root: {root}")
    print(f"Output dir: {output_dir}")
    print(f"Configs: {selected_configs if selected_configs is not None else 'ALL'}")
    print(f"Splits: {selected_splits if selected_splits is not None else 'ALL'}")
    print(f"Steps: {selected_steps if selected_steps is not None else 'ALL'}")
    print(f"Groups: {selected_groups}")
    print(f"Terminate on success: {terminate_on_success}")
    print(f"Deterministic actions: {deterministic}")
    print("=" * 110)

    print("\nExtracting 50 button-press-v3 task variations ...")
    all_tasks = extract_all_tasks(args.task_seed)

    runs = load_runs(root)

    if selected_configs is not None:
        runs = [r for r in runs if r.config_name in selected_configs]

    if selected_splits is not None:
        runs = [r for r in runs if r.split_id in selected_splits]

    if not runs:
        raise RuntimeError("No runs found after filtering. Check --root, --configs, or --splits.")

    print(f"Found {len(runs)} runs.")

    all_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for run_idx, run in enumerate(runs, start=1):
        checkpoints = discover_checkpoints(run)

        if selected_steps is not None:
            checkpoints = [c for c in checkpoints if c.step in selected_steps]

        if not checkpoints:
            skipped_rows.append(
                {
                    "run_name": run.run_name,
                    "config_name": run.config_name,
                    "split_id": run.split_id,
                    "reason": "no_checkpoints_found_after_filtering",
                    "checkpoint_dir": str(run.checkpoint_dir),
                }
            )
            continue

        train_tasks = [all_tasks[i] for i in run.train_idx]
        test_tasks = [all_tasks[i] for i in run.test_idx]

        print("\n" + "-" * 110)
        print(f"Run {run_idx}/{len(runs)}: {run.run_name}")
        print(f"Config: {run.config_name} | split_id={run.split_id} | split_seed={run.split_seed}")
        print(f"Checkpoints: {[c.step for c in checkpoints]}")
        print("-" * 110)

        for checkpoint in checkpoints:
            if checkpoint.vecnormalize_path is None:
                print(f"WARNING: missing VecNormalize for checkpoint {checkpoint.step}: {checkpoint.model_path}")

            print(f"\nLoading checkpoint {checkpoint.step:,}:")
            print(f"  model:        {checkpoint.model_path}")
            print(f"  vecnormalize: {checkpoint.vecnormalize_path}")

            model = PPO.load(str(checkpoint.model_path))
            vecnorm = load_vecnormalize_if_exists(checkpoint.vecnormalize_path)

            if "train" in selected_groups:
                print("  Evaluating TRAIN tasks ...")
                rows = evaluate_checkpoint_on_task_list(
                    model=model,
                    vecnorm=vecnorm,
                    tasks=train_tasks,
                    env_seed=args.eval_seed + run.split_id,
                    n_episodes_per_task=args.eval_train_episodes_per_task,
                    deterministic=deterministic,
                    terminate_on_success=terminate_on_success,
                    reward_type=args.reward_type,
                    max_episode_steps=args.max_episode_steps,
                    group_name="train",
                    run=run,
                    checkpoint=checkpoint,
                )
                all_rows.extend(rows)

                tmp = pd.DataFrame(rows)
                print(
                    f"    train SR={tmp['success'].mean():.3f} | "
                    f"return={tmp['return'].mean():.2f} | "
                    f"first_success={tmp['first_success_step'].mean():.2f}"
                )

            if "test" in selected_groups:
                print("  Evaluating TEST tasks ...")
                rows = evaluate_checkpoint_on_task_list(
                    model=model,
                    vecnorm=vecnorm,
                    tasks=test_tasks,
                    env_seed=args.eval_seed + 10_000 + run.split_id,
                    n_episodes_per_task=args.eval_test_episodes_per_task,
                    deterministic=deterministic,
                    terminate_on_success=terminate_on_success,
                    reward_type=args.reward_type,
                    max_episode_steps=args.max_episode_steps,
                    group_name="test",
                    run=run,
                    checkpoint=checkpoint,
                )
                all_rows.extend(rows)

                tmp = pd.DataFrame(rows)
                print(
                    f"    test  SR={tmp['success'].mean():.3f} | "
                    f"return={tmp['return'].mean():.2f} | "
                    f"first_success={tmp['first_success_step'].mean():.2f}"
                )

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_path = output_dir / "button_checkpoint_skipped.csv"
    skipped_df.to_csv(skipped_path, index=False)

    if not all_rows:
        print("\nNo evaluations were performed.")
        print(f"Skipped report saved to: {skipped_path}")
        return

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize(raw_df)
    across_df = summarize_across_splits(summary_df)

    raw_path = output_dir / "button_checkpoint_raw_episodes.csv"
    summary_path = output_dir / "button_checkpoint_summary.csv"
    across_path = output_dir / "button_checkpoint_across_splits_summary.csv"
    pivot_success_path = output_dir / "button_checkpoint_success_rate_pivot.csv"
    pivot_first_success_path = output_dir / "button_checkpoint_first_success_step_pivot.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    across_df.to_csv(across_path, index=False)

    success_pivot = across_df.pivot_table(
        index=["config_name", "checkpoint_step"],
        columns="group",
        values="mean_success_rate",
    ).reset_index()
    success_pivot.to_csv(pivot_success_path, index=False)

    first_success_pivot = across_df.pivot_table(
        index=["config_name", "checkpoint_step"],
        columns="group",
        values="mean_first_success_step",
    ).reset_index()
    first_success_pivot.to_csv(pivot_first_success_path, index=False)

    print("\n" + "=" * 110)
    print("FINAL CHECKPOINT SUMMARY ACROSS SPLITS")
    print("=" * 110)

    display_cols = [
        "config_name",
        "checkpoint_step",
        "group",
        "mean_success_rate",
        "std_success_rate",
        "mean_return",
        "mean_first_success_step",
        "n_runs",
    ]

    print(across_df[display_cols].to_string(index=False))

    print("\nSaved files:")
    print(f"Raw episodes:              {raw_path}")
    print(f"Per-run checkpoint summary:{summary_path}")
    print(f"Across-splits summary:     {across_path}")
    print(f"Success-rate pivot:        {pivot_success_path}")
    print(f"First-success pivot:       {pivot_first_success_path}")
    print(f"Skipped checkpoints:       {skipped_path}")

    print("\nSuggested plot CSV:")
    print(f"  {pivot_success_path}")


if __name__ == "__main__":
    main()
