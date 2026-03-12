"""
PPO Training Script for RL-Based Insulin Control.

Uses Proximal Policy Optimization with parallel environments
and diverse virtual patients.
"""

import os
import sys
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Add parent directory to path so we can find the new package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import InsulinEnv


# ----------------------------
# Patient pool
# ----------------------------

PATIENTS = (
    [f"adult#{i:03d}" for i in range(1, 11)]
    + [f"adolescent#{i:03d}" for i in range(1, 11)]
    + [f"child#{i:03d}" for i in range(1, 11)]
)


# ----------------------------
# Logging callback
# ----------------------------

class MetricsCallback(BaseCallback):
    """Log glucose metrics during training."""

    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_glucoses = []
        self.episode_count = 0

    def _on_step(self):
        # Collect glucose from info dicts
        for info in self.locals.get("infos", []):
            if "glucose" in info:
                self.episode_glucoses.append(info["glucose"])

            # End of episode — log metrics
            if info.get("TimeLimit.truncated", False) or info.get("terminal_observation") is not None:
                if len(self.episode_glucoses) > 0:
                    gl = np.array(self.episode_glucoses)
                    tir = np.mean((gl >= 70) & (gl <= 180)) * 100
                    mean_gl = np.mean(gl)
                    self.episode_count += 1

                    if self.episode_count % 20 == 0:
                        print(
                            f"  [Episode {self.episode_count:4d}] "
                            f"Mean BG: {mean_gl:6.1f} mg/dL | "
                            f"TIR: {tir:5.1f}%"
                        )

                    self.episode_glucoses = []

        return True


# ----------------------------
# Environment factory
# ----------------------------

def make_env(patient_name=None):
    """Create a single environment instance."""
    def _init():
        name = patient_name or random.choice(PATIENTS)
        env = InsulinEnv(patient_name=name)
        env = Monitor(env)
        return env
    return _init


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 1_000_000

    print("=" * 60)
    print("  RL Artificial Pancreas — PPO Training")
    print("=" * 60)
    print(f"  Environments : {NUM_ENVS} parallel")
    print(f"  Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Patients     : {len(PATIENTS)} virtual patients")
    print("=" * 60)

    # Create vectorized environment
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # PPO with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
    )

    callback = MetricsCallback()

    print("\nStarting training...\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False,
    )

    model_path = os.path.join(SAVE_DIR, "ppo_pancreas")
    model.save(model_path)

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Total episodes: {callback.episode_count}")
    print("=" * 60)
