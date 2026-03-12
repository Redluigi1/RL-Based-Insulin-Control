import os
import numpy as np

from stable_baselines3 import A2C

from simglucose_gym_env import SimGlucoseGymEnv


os.makedirs("results", exist_ok=True)


def compute_metrics(glucose):

    glucose = np.array(glucose)

    tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
    tbr = np.mean(glucose < 70) * 100
    tar = np.mean(glucose > 180) * 100
    mean_glucose = np.mean(glucose)

    return tir, tbr, tar, mean_glucose


patients = (
    [f"adult#{i:03d}" for i in range(1,11)] +
    [f"adolescent#{i:03d}" for i in range(1,11)] +
    [f"child#{i:03d}" for i in range(1,11)]
)


model = A2C.load("models/a2c_pancreas")

all_results = []


for patient in patients:

    env = SimGlucoseGymEnv(patient_name=patient)

    obs, _ = env.reset()

    done = False

    glucose_history = []

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        glucose = env.encoder.glucose_history[-1]

        glucose_history.append(glucose)

        done = terminated or truncated

    tir, tbr, tar, mean_glucose = compute_metrics(glucose_history)

    print(patient,
          "TIR:", round(tir,2),
          "TBR:", round(tbr,2),
          "TAR:", round(tar,2),
          "Mean BG:", round(mean_glucose,2))

    all_results.append([tir, tbr, tar, mean_glucose])


all_results = np.array(all_results)

print("\nFinal Average Results Across Patients")

print("Average TIR:", np.mean(all_results[:,0]))
print("Average TBR:", np.mean(all_results[:,1]))
print("Average TAR:", np.mean(all_results[:,2]))
print("Average Mean Glucose:", np.mean(all_results[:,3]))


np.savetxt(
    "results/evaluation_metrics.csv",
    all_results,
    delimiter=",",
    header="TIR,TBR,TAR,MeanGlucose"
)

print("\nResults saved to results/evaluation_metrics.csv")