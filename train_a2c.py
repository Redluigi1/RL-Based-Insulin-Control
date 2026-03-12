import os
import numpy as np
import random

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

from simglucose_gym_env import SimGlucoseGymEnv


os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -----------------------------
# Virtual patient pool
# -----------------------------

PATIENTS = (
    [f"adult#{i:03d}" for i in range(1,11)] +
    [f"adolescent#{i:03d}" for i in range(1,11)] +
    [f"child#{i:03d}" for i in range(1,11)]
)


# -----------------------------
# Training logger
# -----------------------------

class TrainingLogger(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):

        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]

        return True


# -----------------------------
# Environment factory
# -----------------------------

def make_env():

    patient = random.choice(PATIENTS)

    def _init():
        env = SimGlucoseGymEnv(patient_name=patient)
        env = Monitor(env)
        return env

    return _init


# -----------------------------
# Main training
# -----------------------------

if __name__ == "__main__":

    num_envs = 8   # parallel simulations

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        gamma=0.99,
        n_steps=10,
        tensorboard_log="logs/tensorboard/"
    )

    callback = TrainingLogger()

    model.learn(
        total_timesteps=300000,
        callback=callback
    )

    model.save("models/a2c_pancreas")

    print("\nTraining completed")
    print("Model saved at models/a2c_pancreas")