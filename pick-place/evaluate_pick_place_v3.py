"""
Evaluate PPO on Meta-World pick-place-v3 using multi-goal evaluation.

This script supports models trained WITH VecNormalize, which is important for
models trained with the SubprocVecEnv training script.

Run base model:
    python evaluate_pick_place_v3.py --model base

Run careful model:
    python evaluate_pick_place_v3.py --model careful

Evaluate without rendering:
    python evaluate_pick_place_v3.py --model base --no-render

Render only a few goals:
    python evaluate_pick_place_v3.py --model base --num-goals 5 --sleep 0.02

Save results:
    python evaluate_pick_place_v3.py --model base --no-render --save-json results_base.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import metaworld  # noqa: F401  # registers Meta-World env IDs
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "Meta-World/MT1"
DEFAULT_ENV_NAME = "pick-place-v3"
DEFAULT_PATH_LENGTH = 500
DEFAULT_NUM_GOALS = 50
DEFAULT_SEED = 0
DEFAULT_REWARD_TYPE = "v2"


MODEL_PATHS = {
    "base": {
        "model_path": "./subproc_runs_pick_place/pick_place_v3_PPO_10M_base/pick_place_v3_PPO_10M_base_final.zip",
        "vecnormalize_path": "./subproc_runs_pick_place/pick_place_v3_PPO_10M_base/pick_place_v3_PPO_10M_base_vecnormalize.pkl",
    },
    "careful": {
        "model_path": "./subproc_runs_pick_place/pick_place_v3_PPO_10M_careful/pick_place_v3_PPO_10M_careful_final.zip",
        "vecnormalize_path": "./subproc_runs_pick_place/pick_place_v3_PPO_10M_careful/pick_place_v3_PPO_10M_careful_vecnormalize.pkl",
    },
}


def make_eval_env(env_name, path_length, seed, reward_type, render):
    kwargs = dict(
        env_name=env_name,
        task_select="pseudorandom",
        terminate_on_success=True,
        max_episode_steps=path_length,
        seed=seed,
        reward_function_version=reward_type,
    )

    if render:
        kwargs["render_mode"] = "human"

    eval_env = gym.make(ENV_ID, **kwargs)

    # Enable sampling new goals on each reset.
    eval_env.get_wrapper_attr("toggle_sample_tasks_on_reset")(True)

    return eval_env


def evaluate_mt1_multi_goal_vecnormalize(
    agent,
    vecnormalize_path,
    env_name,
    path_length,
    num_goals=50,
    seed=0,
    reward_type="v2",
    render=True,
    sleep=0.0,
):
    """
    Same idea as your evaluate_mt1_multi_goal function, but adapted for models
    trained with VecNormalize.

    Important: if the PPO model was trained with normalized observations, it
    should also receive normalized observations during evaluation.
    """
    eval_env = make_eval_env(
        env_name=env_name,
        path_length=path_length,
        seed=seed,
        reward_type=reward_type,
        render=render,
    )

    dummy_env = DummyVecEnv([lambda: eval_env])

    vec_env = VecNormalize.load(vecnormalize_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False

    episode_data = []

    for goal_idx in range(num_goals):
        obs = vec_env.reset()

        episode_reward = 0.0
        episode_steps = 0
        goal_success = 0

        for step in range(path_length):
            episode_steps += 1

            action, _ = agent.predict(obs, deterministic=True)

            obs, reward, done_array, infos = vec_env.step(action)

            if render:
                vec_env.render()
                if sleep > 0:
                    time.sleep(sleep)

            done = bool(done_array[0])
            info = infos[0]

            episode_reward += float(reward[0])

            if info.get("success", False):
                goal_success = 1
                break

            if done:
                break

        episode_data.append(
            {
                "goal_idx": goal_idx,
                "reward_sum": episode_reward,
                "reward_avg": episode_reward / episode_steps if episode_steps > 0 else 0.0,
                "steps": episode_steps,
                "success": goal_success,
            }
        )

        print(
            f"Goal {goal_idx + 1:02d}/{num_goals} | "
            f"success={goal_success} | "
            f"steps={episode_steps} | "
            f"reward={episode_reward:.2f}"
        )

    vec_env.close()

    rewards = [ep["reward_sum"] for ep in episode_data]
    successes = [ep["success"] for ep in episode_data]
    steps = [ep["steps"] for ep in episode_data]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "rewards": rewards,
        "successes": successes,
        "steps": steps,
        "episode_data": episode_data,
        "num_goals": num_goals,
    }


def evaluate_mt1_multi_goal_raw(
    agent,
    env_name,
    path_length,
    num_goals=50,
    seed=0,
    reward_type="v2",
    render=True,
    sleep=0.0,
):
    """
    Raw version close to the function you sent.
    Use this only if your model was trained WITHOUT VecNormalize.
    """
    eval_env = make_eval_env(
        env_name=env_name,
        path_length=path_length,
        seed=seed,
        reward_type=reward_type,
        render=render,
    )

    episode_data = []

    for goal_idx in range(num_goals):
        obs, _ = eval_env.reset()

        episode_reward = 0.0
        episode_steps = 0
        goal_success = 0

        for step in range(path_length):
            episode_steps += 1

            action, _ = agent.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = eval_env.step(action)

            if render:
                eval_env.render()
                if sleep > 0:
                    time.sleep(sleep)

            episode_reward += float(reward)

            if info.get("success", False):
                goal_success = 1
                break

            if terminated or truncated:
                break

        episode_data.append(
            {
                "goal_idx": goal_idx,
                "reward_sum": episode_reward,
                "reward_avg": episode_reward / episode_steps if episode_steps > 0 else 0.0,
                "steps": episode_steps,
                "success": goal_success,
            }
        )

        print(
            f"Goal {goal_idx + 1:02d}/{num_goals} | "
            f"success={goal_success} | "
            f"steps={episode_steps} | "
            f"reward={episode_reward:.2f}"
        )

    eval_env.close()

    rewards = [ep["reward_sum"] for ep in episode_data]
    successes = [ep["success"] for ep in episode_data]
    steps = [ep["steps"] for ep in episode_data]

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "rewards": rewards,
        "successes": successes,
        "steps": steps,
        "episode_data": episode_data,
        "num_goals": num_goals,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on Meta-World pick-place-v3.")

    parser.add_argument(
        "--model",
        choices=["base", "careful"],
        default="base",
        help="Which saved pick-place model to evaluate.",
    )

    parser.add_argument("--model-path", default=None, help="Override model path.")

    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Override VecNormalize path.",
    )

    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Use raw observations. Only use if trained without VecNormalize.",
    )

    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--path-length", type=int, default=DEFAULT_PATH_LENGTH)
    parser.add_argument("--num-goals", type=int, default=DEFAULT_NUM_GOALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reward-type", default=DEFAULT_REWARD_TYPE, choices=["v1", "v2"])

    parser.add_argument("--no-render", action="store_true", help="Disable human rendering.")

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Sleep time after each rendered step. Increase to slow down video.",
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save evaluation results as JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    default_paths = MODEL_PATHS[args.model]

    model_path = args.model_path or default_paths["model_path"]
    vecnormalize_path = args.vecnormalize_path or default_paths["vecnormalize_path"]
    render = not args.no_render

    print("=" * 80)
    print("Pick-place evaluation")
    print("=" * 80)
    print(f"Model key:        {args.model}")
    print(f"Model path:       {model_path}")
    print(f"VecNormalize:     {vecnormalize_path}")
    print(f"Use VecNormalize: {not args.no_vecnormalize}")
    print(f"Env name:         {args.env_name}")
    print(f"Num goals:        {args.num_goals}")
    print(f"Path length:      {args.path_length}")
    print(f"Render:           {render}")
    print("=" * 80)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not args.no_vecnormalize and not os.path.exists(vecnormalize_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnormalize_path}")

    agent = PPO.load(model_path, device=args.device)

    if args.no_vecnormalize:
        results = evaluate_mt1_multi_goal_raw(
            agent=agent,
            env_name=args.env_name,
            path_length=args.path_length,
            num_goals=args.num_goals,
            seed=args.seed,
            reward_type=args.reward_type,
            render=render,
            sleep=args.sleep,
        )
    else:
        results = evaluate_mt1_multi_goal_vecnormalize(
            agent=agent,
            vecnormalize_path=vecnormalize_path,
            env_name=args.env_name,
            path_length=args.path_length,
            num_goals=args.num_goals,
            seed=args.seed,
            reward_type=args.reward_type,
            render=render,
            sleep=args.sleep,
        )

    print("\nEvaluation results")
    print("=" * 80)
    print(f"Success rate:   {results['success_rate']:.3f}")
    print(f"Average reward: {results['avg_reward']:.2f}")
    print(f"Reward std:     {results['std_reward']:.2f}")
    print(f"Average steps:  {results['avg_steps']:.2f}")
    print(f"Steps std:      {results['std_steps']:.2f}")
    print("=" * 80)

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to: {save_path}")


if __name__ == "__main__":
    main()
