"""
Evaluate trained custom MT PPO models for:
    button-press-v3 + push-v3

This script tests:
    - base
    - careful
    - explore

For each model, it evaluates separately:
    - button-press-v3
    - push-v3

It loads BOTH:
    - PPO model .zip
    - VecNormalize .pkl

Usage:
    python evaluate_button_push_custom_mt.py

More episodes:
    python evaluate_button_push_custom_mt.py --episodes 100

Multiple seeds:
    python evaluate_button_push_custom_mt.py --episodes 50 --seeds 1000,2000,3000

Stop when success is reached:
    python evaluate_button_push_custom_mt.py --episodes 100 --seeds 1000,2000,3000 --stop-on-success
"""

import argparse
import os
import warnings
from typing import List, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import metaworld # type: ignore
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =========================
# Evaluation settings
# =========================

ENV_ID = "Meta-World/MT1"

TASKS = ("button-press-v3", "push-v3")

OUTPUT_DIR = "./subproc_runs_custom_mt_button_push"
RUN_PREFIX = "custom_mt_button_push_PPO_10M"

CONFIGS = ["base", "careful", "explore"]

MAX_EPISODE_STEPS = 500
REWARD_TYPE = "v2"


class ButtonPushEvalEnv(gym.Env):
    """
    Evaluation environment for the custom MT pair:
        button-press-v3 + push-v3

    reset(options={"task_idx": 0}) -> button-press-v3
    reset(options={"task_idx": 1}) -> push-v3

    Observation:
        original Meta-World observation + one-hot task ID

    button-press-v3 -> [1, 0]
    push-v3         -> [0, 1]
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

        task_one_hot = np.zeros(self.task_id_dim, dtype=np.float32)
        task_one_hot[self.active_task_idx] = 1.0

        return np.concatenate([obs, task_one_hot]).astype(np.float32)

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

    def close(self):
        for env in self.envs:
            env.close()


def make_eval_env(seed: int, append_task_id: bool = True):
    return ButtonPushEvalEnv(
        task_names=TASKS,
        seed=seed,
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_type=REWARD_TYPE,
        terminate_on_success=False,
        append_task_id=append_task_id,
        render_mode=None,
    )


def load_vecnormalize(vecnormalize_path: str, append_task_id: bool = True):
    dummy_env = DummyVecEnv([
        lambda: make_eval_env(seed=999, append_task_id=append_task_id)
    ])

    vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)

    # Do not update normalization stats during evaluation.
    vecnorm.training = False

    # Return raw rewards/returns during evaluation.
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
):
    env = make_eval_env(
        seed=eval_seed + 10_000 * task_idx,
        append_task_id=True,
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

            action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)

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


def parse_seeds(seed_string: str) -> List[int]:
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate button-press-v3 + push-v3 custom MT PPO models."
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
        default="./button_push_eval_results",
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

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_seeds = parse_seeds(args.seeds)
    deterministic = not args.stochastic

    all_rows = []
    skipped = []

    print("=" * 90)
    print("Evaluating button-press-v3 + push-v3 custom MT PPO models")
    print("Configs:", CONFIGS)
    print("Tasks:", TASKS)
    print("Episodes per task per seed:", args.episodes)
    print("Evaluation seeds:", eval_seeds)
    print("Deterministic:", deterministic)
    print("Stop on success:", args.stop_on_success)
    print("=" * 90)

    for config_name in CONFIGS:
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

        print("\n" + "-" * 90)
        print(f"Evaluating config: {config_name}")
        print("Model:", model_path)
        print("VecNormalize:", vecnormalize_path)

        model = PPO.load(model_path)
        vecnorm = load_vecnormalize(vecnormalize_path, append_task_id=True)

        for eval_seed in eval_seeds:
            for task_idx, task_name in enumerate(TASKS):
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
                )

                all_rows.extend(rows)

                temp_df = pd.DataFrame(rows)

                print(
                    f"  seed={eval_seed} | {task_name:16s} | "
                    f"SR={temp_df['success'].mean():.3f} | "
                    f"Return={temp_df['return'].mean():.2f} | "
                    f"FirstSuccessStep={temp_df['first_success_step'].mean():.1f}"
                )

    skipped_df = pd.DataFrame(skipped)
    skipped_path = os.path.join(args.output_dir, "skipped_button_push_models.csv")
    skipped_df.to_csv(skipped_path, index=False)

    if len(all_rows) == 0:
        print("\nNo models were evaluated. Check paths.")
        print("Skipped report:", skipped_path)
        return

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize(raw_df)

    raw_path = os.path.join(args.output_dir, "button_push_eval_raw_episodes.csv")
    summary_path = os.path.join(args.output_dir, "button_push_eval_summary.csv")
    pivot_path = os.path.join(args.output_dir, "button_push_success_rate_pivot.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    pivot = summary_df.pivot_table(
        index="config",
        columns="task_name",
        values="success_rate",
    )
    pivot.to_csv(pivot_path)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

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
    print("Pivot:", pivot_path)
    print("Skipped:", skipped_path)


if __name__ == "__main__":
    main()
