"""Quick smoke test for Phase 2 env + safety."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new'))
import numpy as np
from env import InsulinEnv
from safety import SafetyGuardrails

# Test env
env = InsulinEnv("adult#001")
obs, _ = env.reset()
print(f"Obs shape: {obs.shape} (expected 18)")

safety = SafetyGuardrails()

glucoses, insulins = [], []
for i in range(60):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    safe_dose, interventions = safety.apply(info["insulin_dose"], info["glucose"], info["iob"])
    glucoses.append(info["glucose"])
    insulins.append(info["insulin_dose"])
    if term or trunc:
        break

gl = np.array(glucoses)
ins = np.array(insulins)
print(f"Max insulin setting: {env.max_insulin} U/min ({env.max_insulin*60:.1f} U/hr)")
print(f"Insulin range: {ins.min():.5f} - {ins.max():.5f} U/min")
print(f"Mean insulin: {ins.mean():.5f} U/min ({ins.mean()*60:.2f} U/hr)")
print(f"Glucose range: {gl.min():.1f} - {gl.max():.1f} mg/dL")
print(f"Mean glucose: {gl.mean():.1f} mg/dL")
print(f"Safety interventions: {safety.get_stats()}")
print("PASS" if gl.mean() > 60 else "FAIL")
