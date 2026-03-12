"""
Evaluation script for the trained PPO insulin controller.

Runs the model on all 30 virtual patients and reports metrics.
"""

import os
import sys
import numpy as np

# Add this directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from env import InsulinEnv


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(glucose_trace):
    """Compute clinical glucose control metrics."""
    gl = np.array(glucose_trace)
    return {
        "mean_glucose": np.mean(gl),
        "std_glucose": np.std(gl),
        "min_glucose": np.min(gl),
        "max_glucose": np.max(gl),
        "tir": np.mean((gl >= 70) & (gl <= 180)) * 100,     # Time in Range
        "tbr": np.mean(gl < 70) * 100,                       # Time Below Range
        "tar": np.mean(gl > 180) * 100,                      # Time Above Range
    }


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_pancreas")
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model from:", MODEL_PATH)
    model = PPO.load(MODEL_PATH)

    PATIENTS = (
        [f"adult#{i:03d}" for i in range(1, 11)]
        + [f"adolescent#{i:03d}" for i in range(1, 11)]
        + [f"child#{i:03d}" for i in range(1, 11)]
    )

    all_results = []
    example_glucose_trace = None
    example_patient = None

    print("\n" + "=" * 75)
    print(f"  {'Patient':<18} {'Mean BG':>8} {'TIR%':>7} {'TBR%':>7} {'TAR%':>7} {'Min':>6} {'Max':>6}")
    print("=" * 75)

    for patient in PATIENTS:

        env = InsulinEnv(patient_name=patient, seed=42)
        obs, _ = env.reset()

        glucose_trace = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            glucose_trace.append(info["glucose"])
            done = terminated or truncated

        metrics = compute_metrics(glucose_trace)

        print(
            f"  {patient:<18} "
            f"{metrics['mean_glucose']:>8.1f} "
            f"{metrics['tir']:>6.1f}% "
            f"{metrics['tbr']:>6.1f}% "
            f"{metrics['tar']:>6.1f}% "
            f"{metrics['min_glucose']:>6.1f} "
            f"{metrics['max_glucose']:>6.1f}"
        )

        all_results.append([
            metrics["tir"],
            metrics["tbr"],
            metrics["tar"],
            metrics["mean_glucose"],
        ])

        # Save first adult trace for plotting
        if example_glucose_trace is None:
            example_glucose_trace = glucose_trace
            example_patient = patient

    # Aggregate
    all_results = np.array(all_results)

    print("=" * 75)
    print(f"\n  AGGREGATE RESULTS ({len(PATIENTS)} patients)")
    print(f"  {'Average TIR (70-180):':<28} {np.mean(all_results[:, 0]):>6.1f}%")
    print(f"  {'Average TBR (<70):':<28} {np.mean(all_results[:, 1]):>6.1f}%")
    print(f"  {'Average TAR (>180):':<28} {np.mean(all_results[:, 2]):>6.1f}%")
    print(f"  {'Average Mean Glucose:':<28} {np.mean(all_results[:, 3]):>6.1f} mg/dL")
    print()

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    np.savetxt(
        csv_path,
        all_results,
        delimiter=",",
        header="TIR,TBR,TAR,MeanGlucose",
    )
    print(f"  Results saved to: {csv_path}")

    # Plot example glucose trace
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        time_hours = np.arange(len(example_glucose_trace)) * 5 / 60

        ax.plot(time_hours, example_glucose_trace, color="#2196F3", linewidth=1.5, label="CGM Glucose")
        ax.axhspan(70, 180, alpha=0.15, color="green", label="Target Range (70-180)")
        ax.axhline(70, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(180, color="orange", linestyle="--", alpha=0.5, linewidth=0.8)

        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Blood Glucose (mg/dL)", fontsize=12)
        ax.set_title(f"24-Hour Glucose Trace — {example_patient} (PPO Agent)", fontsize=14)
        ax.legend(loc="upper right")
        ax.set_xlim(0, 24)
        ax.set_ylim(40, max(350, max(example_glucose_trace) + 20))
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(RESULTS_DIR, "glucose_trace.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Glucose trace plot saved to: {plot_path}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    print("\nDone.\n")
