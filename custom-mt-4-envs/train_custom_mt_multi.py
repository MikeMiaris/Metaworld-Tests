"""

At reset, one task is sampled uniformly and a one-hot task ID is appended:
    button-press-v3 -> [1, 0, 0, 0]
    push-v3         -> [0, 1, 0, 0]
    pick-place-v3   -> [0, 0, 1, 0]
    basketball-v3   -> [0, 0, 0, 1]


TensorBoard:
    tensorboard --logdir ./runs_custom_mt_all4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
from gymnasium import spaces
import metaworld  # type: ignore  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

ENV_ID = "Meta-World/MT1"
DEFAULT_TASKS = ("button-press-v3", "push-v3", "pick-place-v3", "basketball-v3")
DEFAULT_PAIR_NAME = "all4"
DEFAULT_OUTPUT_DIR = "./runs_custom_mt_all4"
DEFAULT_SEED = 42
DEFAULT_TOTAL_TIMESTEPS = 20_000_000
DEFAULT_N_ENVS = 8
DEFAULT_MAX_EPISODE_STEPS = 500
DEFAULT_REWARD_TYPE = "v2"
CHECKPOINT_EVERY_TIMESTEPS = 500_000


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    net_arch: Tuple[int, int]


CONFIGS: Dict[str, PPOConfig] = {
    "base": PPOConfig(
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.7,
        max_grad_norm=0.5,
        net_arch=(256, 256),
    ),
    "careful": PPOConfig(
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=1024,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.002,
        vf_coef=0.8,
        max_grad_norm=0.3,
        net_arch=(256, 256),
    ),
    "explore": PPOConfig(
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        net_arch=(256, 256),
    ),
}


def parse_tasks(text: str) -> List[str]:
    tasks = [x.strip() for x in text.split(",") if x.strip()]
    if len(tasks) < 2:
        raise ValueError("--tasks must contain at least two env names.")
    return tasks


def make_task_mapping(task_names: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for idx, name in enumerate(task_names):
        one_hot = [0] * len(task_names)
        one_hot[idx] = 1
        mapping[name] = one_hot
    return mapping


class CustomMTMultiEnv(gym.Env):
    """Custom multi-task wrapper around multiple Meta-World MT1 environments."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_names: Sequence[str],
        seed: int = DEFAULT_SEED,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        reward_type: str = DEFAULT_REWARD_TYPE,
        terminate_on_success: bool = False,
        append_task_id: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if len(task_names) < 2:
            raise ValueError("Custom MT env needs at least two tasks.")

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
                print(f"Warning: could not toggle task sampling for {task_name}: {exc}")
            self.envs.append(env)

        self.action_space = self.envs[0].action_space
        base_obs_space = self.envs[0].observation_space
        if not isinstance(base_obs_space, spaces.Box):
            raise TypeError("Expected Box observation space from Meta-World env.")

        for task_name, env in zip(self.task_names[1:], self.envs[1:]):
            if env.action_space.shape != self.action_space.shape:
                raise ValueError(f"Action shape mismatch for {task_name}.")
            if env.observation_space.shape != base_obs_space.shape:
                raise ValueError(f"Observation shape mismatch for {task_name}.")

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
            if self.active_task_idx < 0 or self.active_task_idx >= len(self.envs):
                raise ValueError(f"Invalid task_idx: {self.active_task_idx}")
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


def make_env(rank, task_names, seed, max_episode_steps, reward_type, terminate_on_success, append_task_id):
    def _init():
        return CustomMTMultiEnv(
            task_names=task_names,
            seed=seed + rank * 1000,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
            terminate_on_success=terminate_on_success,
            append_task_id=append_task_id,
            render_mode=None,
        )
    return _init


def make_vec_env(task_names, seed, n_envs, start_method, max_episode_steps, reward_type, terminate_on_success, append_task_id):
    env = SubprocVecEnv(
        [make_env(i, task_names, seed, max_episode_steps, reward_type, terminate_on_success, append_task_id) for i in range(n_envs)],
        start_method=start_method,
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    return env


def train_one_config(
    cfg: PPOConfig,
    config_name: str,
    experiment_name: str,
    task_names: Sequence[str],
    seed: int,
    n_envs: int,
    total_timesteps: int,
    start_method: str,
    device: str,
    output_dir: str,
    max_episode_steps: int,
    reward_type: str,
    terminate_on_success: bool,
    append_task_id: bool,
) -> None:
    run_prefix = f"custom_mt_{experiment_name}_PPO_{total_timesteps // 1_000_000}M"
    run_name = f"{run_prefix}_{config_name}"
    if not append_task_id:
        run_name += "_no_task_id"

    run_dir = os.path.join(output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    task_mapping = make_task_mapping(task_names)
    print("=" * 100)
    print(f"Starting custom MT multi-task run: {run_name}")
    print("Tasks and one-hot IDs:")
    for task_name, one_hot in task_mapping.items():
        print(f"  {task_name:16s} -> {one_hot}")
    print(f"N envs: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Approx timesteps per task if balanced: {total_timesteps / len(task_names):,.0f}")
    print(f"Device: {device}")
    print(f"Rollout size: {cfg.n_steps * n_envs}")
    print("=" * 100)

    with open(os.path.join(run_dir, f"{run_name}_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "task_names": list(task_names),
                "task_mapping": task_mapping,
                "config_name": config_name,
                "ppo_config": asdict(cfg),
                "seed": seed,
                "n_envs": n_envs,
                "total_timesteps": total_timesteps,
                "max_episode_steps": max_episode_steps,
                "reward_type": reward_type,
                "append_task_id": append_task_id,
                "terminate_on_success": terminate_on_success,
                "run_name": run_name,
            },
            f,
            indent=2,
        )

    torch.set_num_threads(1)
    env = make_vec_env(task_names, seed, n_envs, start_method, max_episode_steps, reward_type, terminate_on_success, append_task_id)

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_dir,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch))),
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(CHECKPOINT_EVERY_TIMESTEPS // n_envs, 1),
            save_path=checkpoint_dir,
            name_prefix=run_name,
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=checkpoint_callback, progress_bar=True)

        model_path = os.path.join(run_dir, f"{run_name}_final")
        vecnormalize_path = os.path.join(run_dir, f"{run_name}_vecnormalize.pkl")
        model.save(model_path)
        env.save(vecnormalize_path)
        print(f"Saved model: {model_path}.zip")
        print(f"Saved VecNormalize stats: {vecnormalize_path}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on custom multi-task Meta-World env with 2+ MT1 tasks.")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_PAIR_NAME, help="Name for this experiment, e.g. all4.")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS), help="Comma-separated MT1 env names.")
    parser.add_argument("--combo", choices=list(CONFIGS.keys()), default="careful")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork"], default="spawn")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--reward-type", choices=["v1", "v2"], default=DEFAULT_REWARD_TYPE)
    parser.add_argument("--terminate-on-success", action="store_true")
    parser.add_argument("--no-task-id", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = parse_tasks(args.tasks)
    cfg = CONFIGS[args.combo]
    append_task_id = not args.no_task_id
    os.makedirs(args.output_dir, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    print("Selected config:", args.combo)
    print("Tasks:", task_names)

    train_one_config(
        cfg=cfg,
        config_name=args.combo,
        experiment_name=args.experiment_name,
        task_names=task_names,
        seed=args.seed,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        start_method=args.start_method,
        device=args.device,
        output_dir=args.output_dir,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        terminate_on_success=args.terminate_on_success,
        append_task_id=append_task_id,
    )

    print("=" * 100)
    print("Custom MT multi-task training finished.")
    print(f"Output directory: {args.output_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    mp.freeze_support()
    main()
