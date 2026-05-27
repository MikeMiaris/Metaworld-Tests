"""
Train PPO on a custom Meta-World multi-task environment:
    basketball-v3 + pick-place-v3

The custom environment randomly selects one of the two MT1 environments at the
start of each episode and appends a one-hot task ID to the observation:
    basketball-v3  -> [1, 0]
    pick-place-v3  -> [0, 1]
"""

import os

# Limit per-process CPU threading before importing torch/numpy-heavy libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import metaworld # type: ignore
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize


ENV_ID = "Meta-World/MT1"
DEFAULT_TASKS = ("basketball-v3", "pick-place-v3")
DEFAULT_SEED = 42
DEFAULT_TOTAL_TIMESTEPS = 10_000_000
DEFAULT_N_ENVS = 8
DEFAULT_MAX_EPISODE_STEPS = 500
DEFAULT_REWARD_TYPE = "v2"
CHECKPOINT_EVERY_TIMESTEPS = 500_000


@dataclass(frozen=True)
class PPOConfig:
    run_name: str
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
        run_name="custom_mt_basketball_pickplace_PPO_10M_base",
        learning_rate=1e-4,
        # With n_envs=8, rollout size = 2048 * 8 = 16384.
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
        run_name="custom_mt_basketball_pickplace_PPO_10M_careful",
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
        run_name="custom_mt_basketball_pickplace_PPO_10M_explore",
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


class CustomMTBasketballPickPlaceEnv(gym.Env):
    """
    A simple custom multi-task wrapper around Meta-World MT1 tasks.

    On every reset, one task is sampled uniformly from:
        basketball-v3, pick-place-v3

    Observation returned to PPO:
        original Meta-World observation + task one-hot

    This lets one policy learn both tasks while knowing which task is active.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_names: Sequence[str] = DEFAULT_TASKS,
        seed: int = DEFAULT_SEED,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        reward_type: str = DEFAULT_REWARD_TYPE,
        terminate_on_success: bool = False,
        append_task_id: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if len(task_names) < 2:
            raise ValueError("Custom MT env needs at least two task names.")

        self.task_names = list(task_names)
        self.seed_value = seed
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.terminate_on_success = terminate_on_success
        self.append_task_id = append_task_id
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

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

            # Make sure each reset samples a new goal/task variation.
            try:
                env.get_wrapper_attr("toggle_sample_tasks_on_reset")(True)
            except Exception as exc:
                print(f"Warning: could not toggle task sampling for {task_name}: {exc}")

            self.envs.append(env)

        # All Meta-World Sawyer tasks should have the same action space.
        self.action_space = self.envs[0].action_space
        for env in self.envs[1:]:
            if env.action_space.shape != self.action_space.shape:
                raise ValueError("All task action spaces must have the same shape.")

        base_obs_space = self.envs[0].observation_space
        if not isinstance(base_obs_space, spaces.Box):
            raise TypeError("Expected Box observation space from Meta-World env.")

        base_shape = base_obs_space.shape
        for task_name, env in zip(self.task_names[1:], self.envs[1:]):
            if env.observation_space.shape != base_shape:
                raise ValueError(
                    f"Observation shape mismatch for {task_name}: "
                    f"{env.observation_space.shape} != {base_shape}"
                )

        self.base_obs_dim = int(np.prod(base_shape))
        self.task_id_dim = len(self.task_names) if append_task_id else 0

        low = np.asarray(base_obs_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_obs_space.high, dtype=np.float32).reshape(-1)

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

        # Optional fixed task for debugging/evaluation.
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


def make_env(
    rank: int,
    task_names: Sequence[str],
    seed: int,
    max_episode_steps: int,
    reward_type: str,
    terminate_on_success: bool,
    append_task_id: bool,
):
    """Top-level env factory so SubprocVecEnv + spawn works on Windows."""

    def _init():
        env = CustomMTBasketballPickPlaceEnv(
            task_names=task_names,
            seed=seed + rank * 1000,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
            terminate_on_success=terminate_on_success,
            append_task_id=append_task_id,
            render_mode=None,
        )
        return env

    return _init


def make_vec_env(
    task_names: Sequence[str],
    seed: int,
    n_envs: int,
    start_method: str,
    max_episode_steps: int,
    reward_type: str,
    terminate_on_success: bool,
    append_task_id: bool,
) -> VecNormalize:
    env = SubprocVecEnv(
        [
            make_env(
                rank=i,
                task_names=task_names,
                seed=seed,
                max_episode_steps=max_episode_steps,
                reward_type=reward_type,
                terminate_on_success=terminate_on_success,
                append_task_id=append_task_id,
            )
            for i in range(n_envs)
        ],
        start_method=start_method,
    )

    env = VecMonitor(env)

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    return env


def train_one_config(
    cfg: PPOConfig,
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
    run_name = cfg.run_name if append_task_id else f"{cfg.run_name}_no_task_id"
    run_dir = os.path.join(output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    print("=" * 80)
    print(f"Starting custom MT run: {run_name}")
    print(f"Tasks: {list(task_names)}")
    print(f"N_ENVS: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Start method: {start_method}")
    print(f"Device: {device}")
    print(f"Append task ID: {append_task_id}")
    print(f"Terminate on success during training: {terminate_on_success}")
    print(f"n_steps: {cfg.n_steps}")
    print(f"Rollout size: {cfg.n_steps * n_envs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    print("=" * 80)

    torch.set_num_threads(1)

    env = make_vec_env(
        task_names=task_names,
        seed=seed,
        n_envs=n_envs,
        start_method=start_method,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
        terminate_on_success=terminate_on_success,
        append_task_id=append_task_id,
    )

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
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
            policy_kwargs=dict(
                net_arch=dict(
                    pi=list(cfg.net_arch),
                    vf=list(cfg.net_arch),
                )
            ),
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(CHECKPOINT_EVERY_TIMESTEPS // n_envs, 1),
            save_path=checkpoint_dir,
            name_prefix=run_name,
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        model_path = os.path.join(run_dir, f"{run_name}_final")
        vecnormalize_path = os.path.join(run_dir, f"{run_name}_vecnormalize.pkl")

        model.save(model_path)
        env.save(vecnormalize_path)

        print(f"Finished run: {run_name}")
        print(f"Saved model: {model_path}.zip")
        print(f"Saved VecNormalize stats: {vecnormalize_path}")

    finally:
        env.close()

#gia arguments sto cmd
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on a custom basketball-v3 + pick-place-v3 Meta-World environment."
    )

    parser.add_argument(
        "--combo",
        choices=["base", "careful", "explore"],
        default="base",
        help="PPO configuration to run.",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Number of parallel SubprocVecEnv workers.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total timesteps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. For MuJoCo, CPU can sometimes be faster than CUDA.",
    )

    parser.add_argument(
        "--start-method",
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
        help="Use spawn on Windows. On Linux, forkserver/fork may be faster.",
    )

    parser.add_argument(
        "--output-dir",
        default="./subproc_runs_custom_mt",
        help="Directory where models, checkpoints, and logs are saved.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Maximum episode length.",
    )

    parser.add_argument(
        "--reward-type",
        choices=["v1", "v2"],
        default=DEFAULT_REWARD_TYPE,
        help="Meta-World reward function version.",
    )

    parser.add_argument(
        "--terminate-on-success",
        action="store_true",
        help="Terminate episodes when success is reached during training. Default is False.",
    )

    parser.add_argument(
        "--no-task-id",
        action="store_true",
        help="Do not append one-hot task ID to observations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = CONFIGS[args.combo]
    append_task_id = not args.no_task_id

    print("CUDA available:", torch.cuda.is_available())
    print("Selected config:", cfg.run_name)
    print("Tasks:", list(DEFAULT_TASKS))

    train_one_config(
        cfg=cfg,
        task_names=DEFAULT_TASKS,
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

    print("=" * 80)
    print("Custom MT training finished.")
    print(f"Output directory: {args.output_dir}")
    print("TensorBoard command:")
    print(f"tensorboard --logdir {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support() #gia to spawn
    main()
