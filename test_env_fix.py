"""Quick smoke test to verify dosing scale is correct."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'new'))
import numpy as np
from env import InsulinEnv

env = InsulinEnv("adult#001")
obs, _ = env.reset()

glucoses = []
rewards = []
insulins = []

for _ in range(50):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    glucoses.append(info["glucose"])
    rewards.append(r)
    insulins.append(info["insulin_dose"])
    if term or trunc:
        break

gl = np.array(glucoses)
ins = np.array(insulins)
print(f"Max basal setting: {env.max_basal} U/min ({env.max_basal*60:.1f} U/hr)")
print(f"Insulin delivered: {ins.min():.4f} - {ins.max():.4f} U/min")
print(f"Glucose range: {gl.min():.1f} - {gl.max():.1f} mg/dL")
print(f"Mean glucose: {gl.mean():.1f} mg/dL")
print(f"Mean reward: {np.mean(rewards):.3f}")
print("PASS" if gl.mean() > 70 else "FAIL - glucose still crashing")
