import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_custom_mt_basketball_pickplace_subproc import ( # type: ignore
    CustomMTBasketballPickPlaceEnv,
    DEFAULT_TASKS,
    DEFAULT_SEED,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_REWARD_TYPE,
)


# =========================
# Evaluation settings
# =========================

N_EVAL_EPISODES = 50
DETERMINISTIC = True

OUTPUT_DIR = "./subproc_runs_custom_mt"

RUNS = {
    "base": "custom_mt_basketball_pickplace_PPO_10M_base",
    "careful": "custom_mt_basketball_pickplace_PPO_10M_careful",
}


def make_raw_eval_env(seed=123):
    env = CustomMTBasketballPickPlaceEnv(
        task_names=DEFAULT_TASKS,
        seed=seed,
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
        reward_type=DEFAULT_REWARD_TYPE,
        terminate_on_success=False,
        append_task_id=True,
        render_mode="human",
    )
    return env


def load_vecnormalize_stats(vecnormalize_path):
    dummy_env = DummyVecEnv([lambda: make_raw_eval_env(seed=999)])
    vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)

    vecnorm.training = False
    vecnorm.norm_reward = False

    return vecnorm


def normalize_obs(vecnorm, obs):
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    norm_obs = vecnorm.normalize_obs(obs_batch)
    return norm_obs


def evaluate_one_task(model, vecnorm, task_idx, task_name, n_episodes):
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    env = make_raw_eval_env(seed=1000 + task_idx)

    for ep in range(n_episodes):
        obs, info = env.reset(options={"task_idx": task_idx})

        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_success = 0.0

        while not done:
            norm_obs = normalize_obs(vecnorm, obs)

            action, _ = model.predict(
                norm_obs,
                deterministic=DETERMINISTIC,
            )

            action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            env.render()######

            ep_reward += float(reward)
            ep_steps += 1

            if "success" in info:
                ep_success = max(ep_success, float(info["success"]))

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        episode_successes.append(ep_success)

    env.close()

    return {
        "task": task_name,
        "success_rate": float(np.mean(episode_successes)),
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_steps": float(np.mean(episode_lengths)),
        "std_steps": float(np.std(episode_lengths)),
        "episodes": n_episodes,
    }


def evaluate_model(run_label, run_name):
    run_dir = os.path.join(OUTPUT_DIR, run_name)

    model_path = os.path.join(run_dir, f"{run_name}_final.zip")
    vecnormalize_path = os.path.join(run_dir, f"{run_name}_vecnormalize.pkl")

    if not os.path.exists(model_path):
        print(f"SKIPPING {run_label}: model not found:")
        print(model_path)
        return []

    if not os.path.exists(vecnormalize_path):
        print(f"SKIPPING {run_label}: VecNormalize file not found:")
        print(vecnormalize_path)
        return []

    print("=" * 80)
    print(f"Evaluating model: {run_label}")
    print("Model:", model_path)
    print("VecNormalize:", vecnormalize_path)
    print("=" * 80)

    model = PPO.load(model_path)
    vecnorm = load_vecnormalize_stats(vecnormalize_path)

    results = []

    for task_idx, task_name in enumerate(DEFAULT_TASKS):
        task_result = evaluate_one_task(
            model=model,
            vecnorm=vecnorm,
            task_idx=task_idx,
            task_name=task_name,
            n_episodes=N_EVAL_EPISODES,
        )

        task_result["model"] = run_label
        results.append(task_result)

        print(
            f"{run_label} | {task_name} | "
            f"SR={task_result['success_rate']:.3f} | "
            f"Reward={task_result['avg_reward']:.2f} | "
            f"Steps={task_result['avg_steps']:.1f}"
        )

    return results


def main():
    all_results = []

    for run_label, run_name in RUNS.items():
        results = evaluate_model(run_label, run_name)
        all_results.extend(results)

    if len(all_results) == 0:
        print("No models were evaluated. Check your paths.")
        return

    df = pd.DataFrame(all_results)

    output_csv = os.path.join(OUTPUT_DIR, "custom_mt_evaluation_results.csv")
    df.to_csv(output_csv, index=False)

    print("\nFinal results:")
    print(df)

    print("\nSaved results to:")
    print(output_csv)


if __name__ == "__main__":
    main()