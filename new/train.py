"""
PPO Training Script for RL-Based Insulin Control — Phase 2.

Enhanced: larger network, longer rollouts, observation normalization,
2M timesteps for better convergence.
"""

import os
import sys
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

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

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_glucoses = []
        self.episode_count = 0
        self.best_tir = 0.0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "glucose" in info:
                self.episode_glucoses.append(info["glucose"])

            if info.get("TimeLimit.truncated", False) or info.get("terminal_observation") is not None:
                if len(self.episode_glucoses) > 0:
                    gl = np.array(self.episode_glucoses)
                    tir = np.mean((gl >= 70) & (gl <= 180)) * 100
                    tbr = np.mean(gl < 70) * 100
                    mean_gl = np.mean(gl)
                    self.episode_count += 1

                    if tir > self.best_tir:
                        self.best_tir = tir

                    if self.episode_count % 50 == 0:
                        print(
                            f"  [Ep {self.episode_count:5d}] "
                            f"Mean BG: {mean_gl:6.1f} | "
                            f"TIR: {tir:5.1f}% | "
                            f"TBR: {tbr:4.1f}% | "
                            f"Best TIR: {self.best_tir:5.1f}%"
                        )

                    self.episode_glucoses = []
        return True


# ----------------------------
# Environment factory
# ----------------------------

def make_env(patient_name=None):
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
    TOTAL_TIMESTEPS = 2_000_000

    print("=" * 60)
    print("  RL Artificial Pancreas — PPO Training (Phase 2)")
    print("=" * 60)
    print(f"  Environments : {NUM_ENVS} parallel")
    print(f"  Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Patients     : {len(PATIENTS)} virtual patients")
    print(f"  Network      : [128, 128]")
    print(f"  Rollout len  : 512 steps")
    print("=" * 60)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
    )

    callback = MetricsCallback()

    print("\nStarting training...\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False,
    )

    model_path = os.path.join(SAVE_DIR, "ppo_pancreas_v2")
    model.save(model_path)

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Total episodes: {callback.episode_count}")
    print(f"  Best TIR seen:  {callback.best_tir:.1f}%")
    print("=" * 60)
