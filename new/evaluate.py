"""
Evaluation script for Phase 2 PPO insulin controller.

Runs on all 30 patients with and without safety guardrails.
Reports per-patient, per-category, and aggregate metrics.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from env import InsulinEnv
from safety import SafetyGuardrails


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(glucose_trace):
    gl = np.array(glucose_trace)
    return {
        "mean": np.mean(gl),
        "std": np.std(gl),
        "min": np.min(gl),
        "max": np.max(gl),
        "tir": np.mean((gl >= 70) & (gl <= 180)) * 100,
        "tbr": np.mean(gl < 70) * 100,
        "tar": np.mean(gl > 180) * 100,
        "tir_tight": np.mean((gl >= 80) & (gl <= 140)) * 100,
    }


def run_episode(model, patient, use_safety=False, seed=42):
    """Run one 24-hour episode and return glucose trace + insulin trace."""
    safety = SafetyGuardrails() if use_safety else None
    env = InsulinEnv(patient_name=patient, seed=seed, safety_layer=safety)
    obs, _ = env.reset()

    glucose_trace = []
    delivered_insulin_trace = []
    proposed_insulin_trace = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # To get the proposed (unfiltered) insulin for plotting, we map it manually
        raw_action = float(action[0])
        proposed_insulin = env._map_action(raw_action)
        proposed_insulin_trace.append(proposed_insulin)

        obs, reward, terminated, truncated, info = env.step(action)

        glucose_trace.append(info["glucose"])
        delivered_insulin_trace.append(info["insulin_dose"])
        done = terminated or truncated

    safety_stats = safety.get_stats() if safety else None
    return glucose_trace, delivered_insulin_trace, proposed_insulin_trace, safety_stats


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_pancreas_v2")
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model from:", MODEL_PATH)
    model = PPO.load(MODEL_PATH)

    PATIENTS = {
        "Adult": [f"adult#{i:03d}" for i in range(1, 11)],
        "Adolescent": [f"adolescent#{i:03d}" for i in range(1, 11)],
        "Child": [f"child#{i:03d}" for i in range(1, 11)],
    }

    # ============ Run with & without safety ============
    for mode_label, use_safety in [("Without Safety", False), ("With Safety", True)]:

        print(f"\n{'=' * 80}")
        print(f"  EVALUATION: {mode_label}")
        print(f"{'=' * 80}")

        header = f"  {'Patient':<18} {'Mean BG':>8} {'TIR%':>7} {'TBR%':>7} {'TAR%':>7} {'Min':>6} {'Max':>6}"
        all_results = []
        category_results = {}
        example_data = {}

        for category, patient_list in PATIENTS.items():
            print(f"\n  --- {category}s ---")
            print(header)
            cat_results = []

            for patient in patient_list:
                gl_trace, del_ins_trace, prop_ins_trace, safety_stats = run_episode(
                    model, patient, use_safety=use_safety
                )
                m = compute_metrics(gl_trace)

                print(
                    f"  {patient:<18} "
                    f"{m['mean']:>8.1f} "
                    f"{m['tir']:>6.1f}% "
                    f"{m['tbr']:>6.1f}% "
                    f"{m['tar']:>6.1f}% "
                    f"{m['min']:>6.1f} "
                    f"{m['max']:>6.1f}"
                    + (f"  [{safety_stats['total_interventions']} interventions]" if safety_stats and safety_stats['total_interventions'] > 0 else "")
                )

                result = [m["tir"], m["tbr"], m["tar"], m["mean"]]
                all_results.append(result)
                cat_results.append(result)

                # Save first adult for plotting
                if category == "Adult" and len(example_data) == 0:
                    example_data = {
                        "patient": patient,
                        "glucose": gl_trace,
                        "insulin_delivered": del_ins_trace,
                        "insulin_proposed": prop_ins_trace,
                    }

            category_results[category] = np.array(cat_results)

        # Aggregate
        all_arr = np.array(all_results)
        print(f"\n{'=' * 80}")
        print(f"  AGGREGATE ({mode_label}, {len(all_arr)} patients)")
        print(f"  {'Avg TIR (70-180):':<28} {np.mean(all_arr[:, 0]):>6.1f}%")
        print(f"  {'Avg TBR (<70):':<28} {np.mean(all_arr[:, 1]):>6.1f}%")
        print(f"  {'Avg TAR (>180):':<28} {np.mean(all_arr[:, 2]):>6.1f}%")
        print(f"  {'Avg Mean Glucose:':<28} {np.mean(all_arr[:, 3]):>6.1f} mg/dL")

        # Per-category
        print(f"\n  Per-category averages:")
        for cat, arr in category_results.items():
            print(f"    {cat + 's:':<15} TIR={np.mean(arr[:, 0]):5.1f}%  Mean BG={np.mean(arr[:, 3]):6.1f}")

        # Save CSV
        suffix = "with_safety" if use_safety else "no_safety"
        csv_path = os.path.join(RESULTS_DIR, f"eval_{suffix}.csv")
        np.savetxt(csv_path, all_arr, delimiter=",", header="TIR,TBR,TAR,MeanGlucose")
        print(f"\n  Saved: {csv_path}")

    # ============ Plot example traces ============
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if example_data:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})

            time_h = np.arange(len(example_data["glucose"])) * 5 / 60

            # Glucose trace
            ax1.plot(time_h, example_data["glucose"], color="#2196F3", linewidth=1.5, label="CGM Glucose")
            ax1.axhspan(70, 180, alpha=0.12, color="green", label="Target (70-180)")
            ax1.axhline(70, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
            ax1.axhline(180, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
            ax1.axhline(120, color="green", linestyle=":", alpha=0.3, linewidth=0.8, label="Target 120")
            ax1.set_ylabel("Blood Glucose (mg/dL)", fontsize=12)
            ax1.set_title(f"24-Hour Trace — {example_data['patient']} (PPO v2)", fontsize=14)
            ax1.legend(loc="upper right", fontsize=9)
            ax1.set_ylim(40, max(350, max(example_data["glucose"]) + 20))
            ax1.grid(True, alpha=0.3)

            # Insulin trace
            ax2.fill_between(time_h, 0, np.array(example_data["insulin_delivered"]) * 60,
                             color="#FF9800", alpha=0.6, label="Delivered (Safe) U/hr")
            ax2.plot(time_h, np.array(example_data["insulin_proposed"]) * 60,
                     color="red", linestyle=":", linewidth=1.5, alpha=0.8, label="Proposed (Raw) U/hr")

            ax2.set_xlabel("Time (hours)", fontsize=12)
            ax2.set_ylabel("Insulin (U/hr)", fontsize=12)
            ax2.set_xlim(0, 24)
            ax2.legend(loc="upper right", fontsize=9)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, "glucose_insulin_trace.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Trace plot saved: {plot_path}")

    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    print("\nDone.\n")
