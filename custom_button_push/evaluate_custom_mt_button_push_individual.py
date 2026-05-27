"""
Evaluate custom multi-task PPO models on the individual Meta-World MT1 envs:
    - button-press-v3
    - push-v3

IMPORTANT:
These custom MT models were trained with:
    observation = Meta-World observation + one-hot task id
    button-press-v3 -> [1, 0]
    push-v3         -> [0, 1]

"""
import argparse
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple
import gymnasium as gym
from gymnasium import spaces
import metaworld # type: ignore
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =============================================================================
# Paths / model setup
# =============================================================================

ENV_ID = "Meta-World/MT1"
OUTPUT_DIR = "./subproc_runs_custom_mt_button_push"
RUN_PREFIX = "custom_mt_button_push_PPO_10M"

# This order MUST match the order used during training.
TASK_ORDER = ["button-press-v3", "push-v3"]
TASK_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(TASK_ORDER)}

DEFAULT_CONFIGS = ["base", "careful", "explore"]
DEFAULT_ENV_NAMES = ["button-press-v3", "push-v3"]

MAX_EPISODE_STEPS = 500
REWARD_TYPE = "v2"


# =============================================================================
# Helpers
# =============================================================================

def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def get_paths(config_name: str) -> Tuple[str, str, str]:
    run_name = f"{RUN_PREFIX}_{config_name}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    model_path = os.path.join(run_dir, f"{run_name}_final.zip")
    vecnormalize_path = os.path.join(run_dir, f"{run_name}_vecnormalize.pkl")
    return run_name, model_path, vecnormalize_path


def get_task_wrapper(env):
    """Find the Meta-World wrapper that contains .tasks and task sampling methods."""
    cur = env
    while True:
        if hasattr(cur, "tasks") and hasattr(cur, "toggle_sample_tasks_on_reset"):
            return cur
        if not hasattr(cur, "env"):
            raise RuntimeError("Could not find Meta-World task wrapper.")
        cur = cur.env


class SingleMT1WithTaskID(gym.Env):
    """
    One real MT1 env, but observation is augmented with the custom MT task one-hot.

    This allows a custom MT model trained on button+push to be evaluated on a
    normal individual env like button-press-v3 or push-v3.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_name: str,
        task_idx: int,
        seed: int = 67,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        reward_type: str = REWARD_TYPE,
        terminate_on_success: bool = True,
        render_mode: Optional[str] = None,
        append_task_id: bool = True,
    ):
        super().__init__()

        self.env_name = env_name
        self.task_idx = int(task_idx)
        self.append_task_id = append_task_id
        self.task_id_dim = len(TASK_ORDER) if append_task_id else 0

        self.env = gym.make(
            ENV_ID,
            env_name=env_name,
            task_select="pseudorandom",
            terminate_on_success=terminate_on_success,
            max_episode_steps=max_episode_steps,
            seed=seed,
            render_mode=render_mode,
            reward_function_version=reward_type,
        )

        self.task_wrapper = get_task_wrapper(self.env)

        try:
            self.task_wrapper.toggle_sample_tasks_on_reset(True)
        except Exception as exc:
            warnings.warn(f"Could not enable task sampling for {env_name}: {exc}")

        self.action_space = self.env.action_space

        base_obs_space = self.env.observation_space
        if not isinstance(base_obs_space, spaces.Box):
            raise TypeError("Expected Box observation space from Meta-World.")

        low = np.asarray(base_obs_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_obs_space.high, dtype=np.float32).reshape(-1)

        if append_task_id:
            low = np.concatenate([low, np.zeros(self.task_id_dim, dtype=np.float32)])
            high = np.concatenate([high, np.ones(self.task_id_dim, dtype=np.float32)])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if not self.append_task_id:
            return obs

        one_hot = np.zeros(self.task_id_dim, dtype=np.float32)
        one_hot[self.task_idx] = 1.0
        return np.concatenate([obs, one_hot]).astype(np.float32)

    def get_tasks(self):
        return list(self.task_wrapper.tasks)

    def set_task(self, task):
        # Disable random task sampling; we will explicitly choose the goal/task.
        self.task_wrapper.toggle_sample_tasks_on_reset(False)
        self.env.unwrapped.set_task(task)

    def enable_random_task_sampling(self):
        self.task_wrapper.toggle_sample_tasks_on_reset(True)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["env_name"] = self.env_name
        info["task_idx"] = self.task_idx
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["env_name"] = self.env_name
        info["task_idx"] = self.task_idx
        return self._augment_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# =============================================================================
# VecNormalize loading / prediction
# =============================================================================

def make_dummy_vecenv_for_vecnormalize(seed: int = 999):
    """
    VecNormalize.load needs a VecEnv with the same observation/action space.
    button and push have the same MT1 obs/action shape, and we append the same
    2D task-id vector, so button is enough as a dummy.
    """
    return DummyVecEnv([
        lambda: SingleMT1WithTaskID(
            env_name=TASK_ORDER[0],
            task_idx=0,
            seed=seed,
            terminate_on_success=True,
            render_mode=None,
            append_task_id=True,
        )
    ])


def load_vecnormalize(vecnormalize_path: str) -> VecNormalize:
    dummy_env = make_dummy_vecenv_for_vecnormalize()
    vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def normalize_obs(vecnorm: VecNormalize, obs: np.ndarray) -> np.ndarray:
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vecnorm.normalize_obs(obs_batch)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model_on_individual_env(
    model: PPO,
    vecnorm: VecNormalize,
    config_name: str,
    env_name: str,
    eval_seed: int,
    num_goals: int,
    episodes_per_goal: int,
    path_length: int,
    deterministic: bool,
    render: bool,
    terminate_on_success: bool,
) -> List[dict]:
    if env_name not in TASK_TO_IDX:
        raise ValueError(
            f"Unknown env_name '{env_name}'. Expected one of {list(TASK_TO_IDX.keys())}."
        )

    task_idx = TASK_TO_IDX[env_name]

    env = SingleMT1WithTaskID(
        env_name=env_name,
        task_idx=task_idx,
        seed=eval_seed,
        max_episode_steps=path_length,
        reward_type=REWARD_TYPE,
        terminate_on_success=terminate_on_success,
        render_mode="human" if render else None,
        append_task_id=True,
    )

    all_tasks = env.get_tasks()
    n_available = len(all_tasks)
    n_goals = min(num_goals, n_available)

    rows = []

    for goal_idx in range(n_goals):
        task = all_tasks[goal_idx]
        env.set_task(task)

        for ep in range(episodes_per_goal):
            obs, info = env.reset()

            episode_return = 0.0
            episode_steps = 0
            success = 0.0
            first_success_step = np.nan
            done = False

            while not done and episode_steps < path_length:
                norm_obs = normalize_obs(vecnorm, obs)

                action, _ = model.predict(norm_obs, deterministic=deterministic)
                action = action[0]

                obs, reward, terminated, truncated, info = env.step(action)

                if render:
                    env.render()

                episode_return += float(reward)
                episode_steps += 1

                current_success = float(info.get("success", 0.0))
                if current_success > 0.0:
                    success = 1.0
                    if np.isnan(first_success_step):
                        first_success_step = episode_steps
                    if terminate_on_success:
                        done = True
                        break

                done = bool(terminated or truncated)

            rows.append(
                {
                    "config": config_name,
                    "env_name": env_name,
                    "task_one_hot_idx": task_idx,
                    "eval_seed": eval_seed,
                    "goal_idx": goal_idx,
                    "episode_for_goal": ep,
                    "success": success,
                    "return": episode_return,
                    "steps": episode_steps,
                    "first_success_step": first_success_step,
                }
            )

    env.close()
    return rows


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["config", "env_name", "eval_seed"])
        .agg(
            success_rate=("success", "mean"),
            avg_return=("return", "mean"),
            std_return=("return", "std"),
            avg_steps=("steps", "mean"),
            std_steps=("steps", "std"),
            avg_first_success_step=("first_success_step", "mean"),
            episodes=("success", "count"),
            goals=("goal_idx", "nunique"),
        )
        .reset_index()
    )


def summarize_over_seeds(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_by_seed.groupby(["config", "env_name"])
        .agg(
            mean_success_rate=("success_rate", "mean"),
            std_success_rate=("success_rate", "std"),
            mean_return=("avg_return", "mean"),
            std_return_across_seeds=("avg_return", "std"),
            mean_steps=("avg_steps", "mean"),
            mean_first_success_step=("avg_first_success_step", "mean"),
            eval_seeds=("eval_seed", "nunique"),
            total_episodes=("episodes", "sum"),
        )
        .reset_index()
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate custom button+push MT PPO models on individual MT1 envs."
    )

    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated configs. Example: base,careful",
    )
    parser.add_argument(
        "--env-names",
        type=str,
        default=",".join(DEFAULT_ENV_NAMES),
        help="Comma-separated envs. Example: button-press-v3,push-v3",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="67,68,69",
        help="Comma-separated evaluation seeds. Example: 67,68,69",
    )
    parser.add_argument(
        "--num-goals",
        type=int,
        default=50,
        help="Number of MT1 task/goal variations to evaluate per env.",
    )
    parser.add_argument(
        "--episodes-per-goal",
        type=int,
        default=1,
        help="Episodes per fixed goal variation.",
    )
    parser.add_argument(
        "--path-length",
        type=int,
        default=MAX_EPISODE_STEPS,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions. Default is deterministic.",
    )
    parser.add_argument(
        "--no-terminate-on-success",
        action="store_true",
        help="Do not terminate episode when success is reached.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation. Use with few goals/episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./individual_env_eval_results",
        help="Directory to save CSV files.",
    )

    args = parser.parse_args()

    configs = parse_csv_list(args.configs)
    env_names = parse_csv_list(args.env_names)
    seeds = parse_int_list(args.seeds)
    deterministic = not args.stochastic
    terminate_on_success = not args.no_terminate_on_success

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("Evaluating custom MT PPO models on individual MT1 envs")
    print("Configs:", configs)
    print("Env names:", env_names)
    print("Eval seeds:", seeds)
    print("Num goals:", args.num_goals)
    print("Episodes per goal:", args.episodes_per_goal)
    print("Deterministic:", deterministic)
    print("Terminate on success:", terminate_on_success)
    print("Render:", args.render)
    print("=" * 100)

    all_rows = []
    skipped = []

    for config_name in configs:
        run_name, model_path, vecnormalize_path = get_paths(config_name)

        if not os.path.exists(model_path) or not os.path.exists(vecnormalize_path):
            print(f"\nSkipping {config_name}: missing model or VecNormalize file")
            print("  model:", model_path)
            print("  vecnorm:", vecnormalize_path)
            skipped.append(
                {
                    "config": config_name,
                    "model_path": model_path,
                    "vecnormalize_path": vecnormalize_path,
                    "model_exists": os.path.exists(model_path),
                    "vecnormalize_exists": os.path.exists(vecnormalize_path),
                }
            )
            continue

        print("\n" + "-" * 100)
        print(f"Loading config: {config_name}")
        print("Model:", model_path)
        print("VecNormalize:", vecnormalize_path)

        model = PPO.load(model_path, device="cpu")
        vecnorm = load_vecnormalize(vecnormalize_path)

        for env_name in env_names:
            for eval_seed in seeds:
                rows = evaluate_model_on_individual_env(
                    model=model,
                    vecnorm=vecnorm,
                    config_name=config_name,
                    env_name=env_name,
                    eval_seed=eval_seed,
                    num_goals=args.num_goals,
                    episodes_per_goal=args.episodes_per_goal,
                    path_length=args.path_length,
                    deterministic=deterministic,
                    render=args.render,
                    terminate_on_success=terminate_on_success,
                )
                all_rows.extend(rows)

                temp = pd.DataFrame(rows)
                print(
                    f"  seed={eval_seed:<5d} | {env_name:16s} | "
                    f"SR={temp['success'].mean():.3f} | "
                    f"Return={temp['return'].mean():.2f} | "
                    f"Steps={temp['steps'].mean():.1f} | "
                    f"FirstSuccessStep={temp['first_success_step'].mean():.1f}"
                )

    skipped_df = pd.DataFrame(skipped)
    skipped_path = os.path.join(args.output_dir, "skipped_models.csv")
    skipped_df.to_csv(skipped_path, index=False)

    if not all_rows:
        print("\nNo models evaluated. Check paths.")
        print("Skipped:", skipped_path)
        return

    raw_df = pd.DataFrame(all_rows)
    by_seed_df = summarize(raw_df)
    final_df = summarize_over_seeds(by_seed_df)

    raw_path = os.path.join(args.output_dir, "individual_env_raw_episodes.csv")
    by_seed_path = os.path.join(args.output_dir, "individual_env_summary_by_seed.csv")
    final_path = os.path.join(args.output_dir, "individual_env_final_summary.csv")
    pivot_path = os.path.join(args.output_dir, "individual_env_success_pivot.csv")

    raw_df.to_csv(raw_path, index=False)
    by_seed_df.to_csv(by_seed_path, index=False)
    final_df.to_csv(final_path, index=False)

    pivot = final_df.pivot_table(
        index="config",
        columns="env_name",
        values="mean_success_rate",
    )
    pivot.to_csv(pivot_path)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY OVER SEEDS")
    print("=" * 100)
    display_cols = [
        "config",
        "env_name",
        "mean_success_rate",
        "std_success_rate",
        "mean_return",
        "mean_steps",
        "mean_first_success_step",
        "eval_seeds",
        "total_episodes",
    ]
    print(final_df[display_cols].to_string(index=False))

    print("\nSUCCESS PIVOT")
    print(pivot.to_string())

    print("\nSaved files:")
    print("Raw episodes:", raw_path)
    print("Summary by seed:", by_seed_path)
    print("Final summary:", final_path)
    print("Pivot:", pivot_path)
    print("Skipped:", skipped_path)


if __name__ == "__main__":
    main()
