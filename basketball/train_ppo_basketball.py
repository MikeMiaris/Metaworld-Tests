"""
PPO training script for Meta-World basketball-v3
Configs : A_basketball_main, B_basketball_entropy
Seeds   : 3 train seeds per config
Steps   : 6 000 000 per run
Envs    : SubprocVecEnv with 4 parallel workers for faster rollout collection
"""
import metaworld # type: ignore
import os
import csv
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# ──────────────────────────────────────────────
# Top-level config  (edit these to change runs)
# ──────────────────────────────────────────────

ENV_NAME         = "basketball-v3"
SPLIT_SEED       = 67
TRAIN_SEEDS      = [11, 22, 33]
TIMESTEP_BUDGET  = 6_000_000
N_PARALLEL_ENVS  = 4          # SubprocVecEnv workers

N_EVAL_TRAIN     = 5
N_EVAL_TEST      = 10
RESULTS_CSV      = "ppo_basketball_results.csv"
MODELS_DIR       = "saved_models_basketball"
TB_DIR           = "tb_basketball"

PPO_CONFIGS = {
    "A_basketball_main": {
        "learning_rate": 3e-4,
        "n_steps":       8192,   # per env; total rollout = 8192 * 4 = 32 768
        "batch_size":    512,
        "n_epochs":      20,
        "gamma":         0.995,
        "gae_lambda":    0.95,
        "clip_range":    0.2,
        "ent_coef":      0.0,
        "vf_coef":       0.5,
        "max_grad_norm": 1.0,
    },
    "B_basketball_entropy": {
        "learning_rate": 3e-4,
        "n_steps":       8192,
        "batch_size":    512,
        "n_epochs":      20,
        "gamma":         0.995,
        "gae_lambda":    0.95,
        "clip_range":    0.2,
        "ent_coef":      0.001,
        "vf_coef":       0.5,
        "max_grad_norm": 1.0,
    },
}

# ──────────────────────────────────────────────
# Progress bar  (stdlib only, no tqdm needed)
# ──────────────────────────────────────────────

class ProgressBar:
    BAR_WIDTH = 42

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc  = desc
        self.n     = 0
        self.start = time.time()
        self._draw()

    def update(self, n: int = 1):
        self.n = min(self.n + n, self.total)
        self._draw()

    def _draw(self):
        frac   = self.n / self.total if self.total else 1.0
        filled = int(self.BAR_WIDTH * frac)
        bar    = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        pct    = frac * 100
        elapsed = time.time() - self.start
        if 0 < frac < 1.0:
            eta_str = f"  ETA {elapsed / frac - elapsed:5.0f}s"
        elif frac >= 1.0:
            eta_str = f"  {elapsed:5.0f}s total"
        else:
            eta_str = ""
        sys.stdout.write(
            f"\r{self.desc}  [{bar}] {pct:5.1f}%  "
            f"{self.n:>9}/{self.total}{eta_str}   "
        )
        sys.stdout.flush()

    def close(self):
        self.n = self.total
        self._draw()
        sys.stdout.write("\n")
        sys.stdout.flush()


class ProgressBarCallback(BaseCallback):
    """
    Updates progress bar each time _on_step is called.
    With SubprocVecEnv(N), _on_step fires once per rollout step,
    but num_envs steps were collected — so we advance by num_envs.
    """

    def __init__(self, total_timesteps: int, n_envs: int, desc: str = ""):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.desc   = desc
        self.pbar   = None

    def _on_training_start(self):
        self.pbar = ProgressBar(self.total_timesteps, self.desc)

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(self.n_envs)
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()


# ──────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────

def get_task_wrapper(env):
    """
    basketball-v3 wrapper depth is exactly 4 levels deep:
    env.env.env.env  — confirmed in the notebook.
    Falls back to dynamic search if the depth ever changes.
    """
    try:
        w = env.env.env.env
        if hasattr(w, "tasks") and hasattr(w, "toggle_sample_tasks_on_reset"):
            return w
    except AttributeError:
        pass
    # Dynamic fallback
    cur = env
    while True:
        if hasattr(cur, "tasks") and hasattr(cur, "toggle_sample_tasks_on_reset"):
            return cur
        if not hasattr(cur, "env"):
            raise RuntimeError("Could not find Meta-World task wrapper.")
        cur = cur.env


def extract_all_tasks(seed: int = SPLIT_SEED):
    env = gym.make("Meta-World/MT1", env_name=ENV_NAME, seed=seed)
    wrapper = get_task_wrapper(env)
    tasks = list(wrapper.tasks)
    env.close()
    assert len(tasks) == 50, f"Expected 50 tasks, got {len(tasks)}"
    return tasks


def build_split(tasks, split_seed: int):
    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(tasks))
    rng.shuffle(indices)
    train_idx = indices[:45]
    test_idx  = indices[45:]
    return train_idx, test_idx, [tasks[i] for i in train_idx], [tasks[i] for i in test_idx]


# ──────────────────────────────────────────────
# VecEnv factory  (must be top-level for pickle)
# ──────────────────────────────────────────────

def _make_worker(env_name: str, seed: int, tasks: list, sample_on_reset: bool):
    """
    Returns a zero-argument callable that builds one env worker.
    Kept at module level so SubprocVecEnv can pickle it.
    """
    def _init():
        env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
        wrapper = get_task_wrapper(env)
        wrapper.tasks = list(tasks)
        wrapper.toggle_sample_tasks_on_reset(sample_on_reset)
        return env
    return _init


def make_subproc_train_env(tasks: list, n_envs: int = N_PARALLEL_ENVS) -> SubprocVecEnv:
    fns = [_make_worker(ENV_NAME, SPLIT_SEED, tasks, sample_on_reset=True)
           for _ in range(n_envs)]
    # "fork" is Linux/Mac only; Windows requires "spawn"
    import platform
    start_method = "fork" if platform.system() != "Windows" else "spawn"
    return SubprocVecEnv(fns, start_method=start_method)


def make_single_env(tasks: list, sample_on_reset: bool = False):
    """Single-env for evaluation (no SubprocVecEnv needed)."""
    env = gym.make("Meta-World/MT1", env_name=ENV_NAME, seed=SPLIT_SEED)
    wrapper = get_task_wrapper(env)
    wrapper.tasks = list(tasks)
    wrapper.toggle_sample_tasks_on_reset(sample_on_reset)
    return env


# ──────────────────────────────────────────────
# Evaluation  (runs in main process, single env)
# ──────────────────────────────────────────────

def evaluate_on_task_list(model, env, tasks: list, n_episodes: int) -> dict:
    wrapper = get_task_wrapper(env)
    wrapper.toggle_sample_tasks_on_reset(False)
    base_env = env.unwrapped

    per_task_success, per_task_return = [], []

    for i, task in enumerate(tasks):
        sys.stdout.write(f"\r    Evaluating task {i + 1}/{len(tasks)} ...")
        sys.stdout.flush()
        base_env.set_task(task)
        successes, returns = [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_ret, ep_suc = 0.0, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += reward
                ep_suc = max(ep_suc, float(info.get("success", 0.0)))
                done = terminated or truncated
            successes.append(ep_suc)
            returns.append(ep_ret)
        per_task_success.append(float(np.mean(successes)))
        per_task_return.append(float(np.mean(returns)))

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    wrapper.toggle_sample_tasks_on_reset(True)

    return {
        "mean_success": float(np.mean(per_task_success)),
        "std_success":  float(np.std(per_task_success)),
        "mean_return":  float(np.mean(per_task_return)),
        "std_return":   float(np.std(per_task_return)),
    }


# ──────────────────────────────────────────────
# CSV logging
# ──────────────────────────────────────────────

def append_row(csv_path: str, row: dict, write_header: bool = False):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    # ── Split ──
    print("Extracting 50 task variations ...")
    all_tasks = extract_all_tasks(SPLIT_SEED)
    train_idx, test_idx, train_tasks, test_tasks = build_split(all_tasks, SPLIT_SEED)

    assert len(train_tasks) == 45
    assert len(test_tasks) == 5
    assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
    print(f"Split OK — 45 train / 5 test  (split_seed={SPLIT_SEED})")
    print(f"Test idx: {sorted(test_idx.tolist())}")

    total_runs     = len(PPO_CONFIGS) * len(TRAIN_SEEDS)
    run_counter    = 0
    header_written = False

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    for config_name, config in PPO_CONFIGS.items():
        for train_seed in TRAIN_SEEDS:
            run_counter += 1
            run_name = (
                f"{ENV_NAME}_{config_name}"
                f"_splitseed{SPLIT_SEED}"
                f"_trainseed{train_seed}"
                f"_{TIMESTEP_BUDGET}"
            )

            print(f"\n{'='*72}")
            print(
                f"Run {run_counter}/{total_runs}  |  {config_name}"
                f"  |  seed={train_seed}  |  {TIMESTEP_BUDGET:,} steps"
                f"  |  {N_PARALLEL_ENVS} parallel envs"
            )
            print(f"{'='*72}")

            # ── Build vectorised training env ──
            print(f"  Spawning {N_PARALLEL_ENVS} SubprocVecEnv workers ...")
            import pickle
            try:
                pickle.dumps(train_tasks[0])
            except Exception as e:
                raise RuntimeError(
                    f"Meta-World Task objects are not picklable — "
                    f"SubprocVecEnv (spawn) requires this on Windows.\n{e}"
                )
            train_vec_env = make_subproc_train_env(train_tasks, N_PARALLEL_ENVS)

            # Single envs for eval (kept alive across seeds to save init time)
            train_eval_env = make_single_env(train_tasks)
            test_eval_env  = make_single_env(test_tasks)

            # ── PPO model ──
            model = PPO(
                policy="MlpPolicy",
                env=train_vec_env,
                verbose=0,
                seed=train_seed,
                device="cpu",
                tensorboard_log=TB_DIR,
                **config,
            )

            cb = ProgressBarCallback(
                total_timesteps=TIMESTEP_BUDGET,
                n_envs=N_PARALLEL_ENVS,
                desc=f"  Training",
            )

            # ── Train ──
            model.learn(
                total_timesteps=TIMESTEP_BUDGET,
                tb_log_name=run_name,
                reset_num_timesteps=True,
                callback=cb,
            )

            train_vec_env.close()

            # ── Eval ──
            print(f"  Evaluating {len(train_tasks)} train tasks ...")
            train_metrics = evaluate_on_task_list(
                model, train_eval_env, train_tasks, N_EVAL_TRAIN
            )

            print(f"  Evaluating {len(test_tasks)} test tasks ...")
            test_metrics = evaluate_on_task_list(
                model, test_eval_env, test_tasks, N_EVAL_TEST
            )

            # ── Save ──
            model_path = os.path.join(MODELS_DIR, f"ppo_{run_name}.zip")
            model.save(model_path)

            success_gap = train_metrics["mean_success"] - test_metrics["mean_success"]
            return_gap  = train_metrics["mean_return"]  - test_metrics["mean_return"]

            print(
                f"\n  ✓  train_success={train_metrics['mean_success']:.3f}"
                f"  test_success={test_metrics['mean_success']:.3f}"
                f"  gap={success_gap:+.3f}"
            )
            print(f"  Model → {model_path}")

            row = {
                "env_name":         ENV_NAME,
                "split_seed":       SPLIT_SEED,
                "train_seed":       train_seed,
                "config_name":      config_name,
                "total_timesteps":  TIMESTEP_BUDGET,
                "n_parallel_envs":  N_PARALLEL_ENVS,
                "train_task_count": len(train_tasks),
                "test_task_count":  len(test_tasks),
                "train_idx":        train_idx.tolist(),
                "test_idx":         test_idx.tolist(),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}":  v for k, v in test_metrics.items()},
                "success_gap":      success_gap,
                "return_gap":       return_gap,
                "tb_log_dir":       TB_DIR,
                "model_path":       model_path,
            }
            append_row(RESULTS_CSV, row, write_header=not header_written)
            header_written = True

            train_eval_env.close()
            test_eval_env.close()

    print(f"\n{'='*72}")
    print(f"All done!  Results → {RESULTS_CSV}")
    print(f"Models   → {MODELS_DIR}/")
    print(f"TBoard   → tensorboard --logdir {TB_DIR}")


if __name__ == "__main__":
    main()
