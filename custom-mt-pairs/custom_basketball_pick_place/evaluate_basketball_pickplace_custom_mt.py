"""
Evaluate trained custom MT PPO models for:
    basketball-v3 + pick-place-v3

Compatible with train_custom_mt_basketball_pickplace.py.

It evaluates each trained custom MT model separately on:
    - basketball-v3
    - pick-place-v3

It loads:
    - PPO model .zip
    - VecNormalize .pkl

It supports:
    - base / careful / explore
    - multiple evaluation seeds
    - stop-on-success evaluation
    - deterministic or stochastic actions
    - raw per-episode CSV
    - summary CSV
    - success-rate pivot CSV

Usage:
    python evaluate_basketball_pickplace_custom_mt.py

Robust evaluation:
    python evaluate_basketball_pickplace_custom_mt.py --episodes 100 --seeds 1000,2000,3000 --stop-on-success

Only careful:
    python evaluate_basketball_pickplace_custom_mt.py --configs careful --episodes 100 --seeds 1000,2000,3000 --stop-on-success

Only basketball:
    python evaluate_basketball_pickplace_custom_mt.py --tasks basketball-v3 --episodes 100 --seeds 1000,2000,3000 --stop-on-success
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import List, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import metaworld  # noqa: F401
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =============================================================================
# Settings
# =============================================================================

ENV_ID = "Meta-World/MT1"

TASKS = ("basketball-v3", "pick-place-v3")
TASK_TO_IDX = {name: i for i, name in enumerate(TASKS)}

OUTPUT_DIR = "./subproc_runs_custom_mt"
RUN_PREFIX = "custom_mt_basketball_pickplace_PPO_10M"

CONFIGS = ["base", "careful", "explore"]

MAX_EPISODE_STEPS = 500
REWARD_TYPE = "v2"


# =============================================================================
# Custom evaluation env
# =============================================================================

class BasketballPickPlaceEvalEnv(gym.Env):
    """
    Evaluation environment for:
        basketball-v3 + pick-place-v3

    reset(options={"task_idx": 0}) -> basketball-v3
    reset(options={"task_idx": 1}) -> pick-place-v3

    Observation:
        original Meta-World observation + one-hot task ID

    basketball-v3 -> [1, 0]
    pick-place-v3 -> [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_names: Sequence[str] = TASKS,
        seed: int = 123,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        reward_type: str = REWARD_TYPE,
        terminate_on_success: bool = False,
        append_task_id: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.task_names = list(task_names)
        self.rng = np.random.default_rng(seed)
        self.append_task_id = append_task_id

        self.envs = []

        for idx, task_name in enumerate(self.task_names):
            env = gym.make(
                ENV_ID,
                env_name=task_name,
                task_select="pseudorandom",
                terminate_on_success=terminate_on_success,
                max_episode_steps=max_episode_steps,
                seed=seed + idx,
                reward_function_version=reward_type,
                render_mode=render_mode,
            )

            try:
                env.get_wrapper_attr("toggle_sample_tasks_on_reset")(True)
            except Exception as exc:
                warnings.warn(f"Could not enable task sampling for {task_name}: {exc}")

            self.envs.append(env)

        self.action_space = self.envs[0].action_space

        for task_name, env in zip(self.task_names[1:], self.envs[1:]):
            if env.action_space.shape != self.action_space.shape:
                raise ValueError(
                    f"Action shape mismatch for {task_name}: "
                    f"{env.action_space.shape} != {self.action_space.shape}"
                )

        base_obs_space = self.envs[0].observation_space
        if not isinstance(base_obs_space, spaces.Box):
            raise TypeError("Expected Box observation space from Meta-World.")

        base_shape = base_obs_space.shape

        for task_name, env in zip(self.task_names[1:], self.envs[1:]):
            if env.observation_space.shape != base_shape:
                raise ValueError(
                    f"Observation shape mismatch for {task_name}: "
                    f"{env.observation_space.shape} != {base_shape}"
                )

        low = np.asarray(base_obs_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_obs_space.high, dtype=np.float32).reshape(-1)

        self.task_id_dim = len(self.task_names) if append_task_id else 0

        if append_task_id:
            low = np.concatenate([low, np.zeros(self.task_id_dim, dtype=np.float32)])
            high = np.concatenate([high, np.ones(self.task_id_dim, dtype=np.float32)])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.active_task_idx = 0
        self.active_env = self.envs[0]

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if not self.append_task_id:
            return obs

        one_hot = np.zeros(self.task_id_dim, dtype=np.float32)
        one_hot[self.active_task_idx] = 1.0
        return np.concatenate([obs, one_hot]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and "task_idx" in options:
            self.active_task_idx = int(options["task_idx"])
        else:
            self.active_task_idx = int(self.rng.integers(0, len(self.envs)))

        self.active_env = self.envs[self.active_task_idx]

        obs, info = self.active_env.reset()

        info = dict(info)
        info["task_name"] = self.task_names[self.active_task_idx]
        info["task_idx"] = self.active_task_idx

        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active_env.step(action)

        info = dict(info)
        info["task_name"] = self.task_names[self.active_task_idx]
        info["task_idx"] = self.active_task_idx

        return self._augment_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.active_env.render()

    def close(self):
        for env in self.envs:
            env.close()


# =============================================================================
# Loading helpers
# =============================================================================

def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_eval_env(seed: int, append_task_id: bool = True, render: bool = False):
    return BasketballPickPlaceEvalEnv(
        task_names=TASKS,
        seed=seed,
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_type=REWARD_TYPE,
        terminate_on_success=False,
        append_task_id=append_task_id,
        render_mode="human" if render else None,
    )


def load_vecnormalize(vecnormalize_path: str, append_task_id: bool = True):
    dummy_env = DummyVecEnv([
        lambda: make_eval_env(seed=999, append_task_id=append_task_id, render=False)
    ])

    vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def normalize_obs(vecnorm: VecNormalize, obs: np.ndarray):
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vecnorm.normalize_obs(obs_batch)


def get_paths(config_name: str):
    run_name = f"{RUN_PREFIX}_{config_name}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)

    model_path = os.path.join(run_dir, f"{run_name}_final.zip")
    vecnormalize_path = os.path.join(run_dir, f"{run_name}_vecnormalize.pkl")

    return run_name, model_path, vecnormalize_path


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_one_task(
    model: PPO,
    vecnorm: VecNormalize,
    config_name: str,
    task_idx: int,
    task_name: str,
    eval_seed: int,
    n_episodes: int,
    deterministic: bool,
    stop_on_success: bool,
    render: bool,
):
    env = make_eval_env(
        seed=eval_seed + 10_000 * task_idx,
        append_task_id=True,
        render=render,
    )

    rows = []

    for ep in range(n_episodes):
        obs, info = env.reset(
            seed=eval_seed + ep,
            options={"task_idx": task_idx},
        )

        done = False
        episode_return = 0.0
        episode_length = 0
        success = 0.0
        first_success_step = np.nan

        while not done:
            norm_obs = normalize_obs(vecnorm, obs)

            action, _ = model.predict(
                norm_obs,
                deterministic=deterministic,
            )

            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            episode_return += float(reward)
            episode_length += 1

            current_success = float(info.get("success", 0.0))

            if current_success > 0.0:
                success = 1.0

                if np.isnan(first_success_step):
                    first_success_step = episode_length

                if stop_on_success:
                    done = True
                    break

            done = bool(terminated or truncated)

        rows.append(
            {
                "config": config_name,
                "task_idx": task_idx,
                "task_name": task_name,
                "eval_seed": eval_seed,
                "episode": ep,
                "success": success,
                "return": episode_return,
                "episode_length": episode_length,
                "first_success_step": first_success_step,
            }
        )

    env.close()
    return rows


def summarize(df: pd.DataFrame):
    return (
        df.groupby(["config", "task_name"])
        .agg(
            success_rate=("success", "mean"),
            avg_return=("return", "mean"),
            std_return=("return", "std"),
            avg_episode_length=("episode_length", "mean"),
            std_episode_length=("episode_length", "std"),
            avg_first_success_step=("first_success_step", "mean"),
            median_first_success_step=("first_success_step", "median"),
            episodes=("success", "count"),
        )
        .reset_index()
    )


def summarize_by_seed(df: pd.DataFrame):
    return (
        df.groupby(["config", "task_name", "eval_seed"])
        .agg(
            success_rate=("success", "mean"),
            avg_return=("return", "mean"),
            std_return=("return", "std"),
            avg_episode_length=("episode_length", "mean"),
            avg_first_success_step=("first_success_step", "mean"),
            episodes=("success", "count"),
        )
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate basketball-v3 + pick-place-v3 custom MT PPO models."
    )

    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(CONFIGS),
        help="Comma-separated configs. Example: base,careful,explore",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASKS),
        help="Comma-separated tasks. Example: basketball-v3,pick-place-v3",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per task per seed.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="1000",
        help="Comma-separated evaluation seeds. Example: 1000,2000,3000",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./basketball_pickplace_eval_results",
        help="Where to save CSV results.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )

    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop each episode as soon as success is reached.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation. Use only with a small number of episodes.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = parse_csv_list(args.configs)
    tasks = parse_csv_list(args.tasks)
    eval_seeds = parse_int_list(args.seeds)
    deterministic = not args.stochastic

    for task in tasks:
        if task not in TASK_TO_IDX:
            raise ValueError(f"Unknown task {task}. Expected one of {list(TASK_TO_IDX.keys())}")

    all_rows = []
    skipped = []

    print("=" * 100)
    print("Evaluating basketball-v3 + pick-place-v3 custom MT PPO models")
    print("Configs:", configs)
    print("Tasks:", tasks)
    print("Episodes per task per seed:", args.episodes)
    print("Evaluation seeds:", eval_seeds)
    print("Deterministic:", deterministic)
    print("Stop on success:", args.stop_on_success)
    print("Render:", args.render)
    print("=" * 100)

    for config_name in configs:
        run_name, model_path, vecnormalize_path = get_paths(config_name)

        if not os.path.exists(model_path) or not os.path.exists(vecnormalize_path):
            print(f"Skipping {config_name}: missing model or VecNormalize file.")
            skipped.append(
                {
                    "config": config_name,
                    "run_name": run_name,
                    "model_exists": os.path.exists(model_path),
                    "vecnormalize_exists": os.path.exists(vecnormalize_path),
                    "model_path": model_path,
                    "vecnormalize_path": vecnormalize_path,
                }
            )
            continue

        print("\n" + "-" * 100)
        print(f"Evaluating config: {config_name}")
        print("Model:", model_path)
        print("VecNormalize:", vecnormalize_path)

        model = PPO.load(model_path, device="cpu")
        vecnorm = load_vecnormalize(vecnormalize_path, append_task_id=True)

        for eval_seed in eval_seeds:
            for task_name in tasks:
                task_idx = TASK_TO_IDX[task_name]

                rows = evaluate_one_task(
                    model=model,
                    vecnorm=vecnorm,
                    config_name=config_name,
                    task_idx=task_idx,
                    task_name=task_name,
                    eval_seed=eval_seed,
                    n_episodes=args.episodes,
                    deterministic=deterministic,
                    stop_on_success=args.stop_on_success,
                    render=args.render,
                )

                all_rows.extend(rows)

                temp_df = pd.DataFrame(rows)

                print(
                    f"  seed={eval_seed} | {task_name:16s} | "
                    f"SR={temp_df['success'].mean():.3f} | "
                    f"Return={temp_df['return'].mean():.2f} | "
                    f"FirstSuccessStep={temp_df['first_success_step'].mean():.1f}"
                )

        if vecnorm is not None:
            vecnorm.close()

    skipped_df = pd.DataFrame(skipped)
    skipped_path = os.path.join(args.output_dir, "skipped_basketball_pickplace_models.csv")
    skipped_df.to_csv(skipped_path, index=False)

    if len(all_rows) == 0:
        print("\nNo models were evaluated. Check paths.")
        print("Skipped report:", skipped_path)
        return

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize(raw_df)
    by_seed_df = summarize_by_seed(raw_df)

    raw_path = os.path.join(args.output_dir, "basketball_pickplace_eval_raw_episodes.csv")
    summary_path = os.path.join(args.output_dir, "basketball_pickplace_eval_summary.csv")
    by_seed_path = os.path.join(args.output_dir, "basketball_pickplace_eval_summary_by_seed.csv")
    pivot_path = os.path.join(args.output_dir, "basketball_pickplace_success_rate_pivot.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    by_seed_df.to_csv(by_seed_path, index=False)

    pivot = summary_df.pivot_table(
        index="config",
        columns="task_name",
        values="success_rate",
    )
    pivot.to_csv(pivot_path)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    display_cols = [
        "config",
        "task_name",
        "success_rate",
        "avg_return",
        "std_return",
        "avg_first_success_step",
        "episodes",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print("\nSuccess-rate pivot:")
    print(pivot.to_string())

    print("\nSaved files:")
    print("Raw episodes:", raw_path)
    print("Summary:", summary_path)
    print("Summary by seed:", by_seed_path)
    print("Pivot:", pivot_path)
    print("Skipped:", skipped_path)


if __name__ == "__main__":
    main()
