"""
Enhanced Gymnasium wrapper for simglucose.

Phase 2: Richer state (18-dim), nonlinear action mapping,
Magni risk index reward, and proper normalization.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from datetime import datetime

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action


class InsulinEnv(gym.Env):
    """
    RL environment for blood glucose control via insulin delivery.

    State (18-dim, all normalized):
        [0-7]   : Last 8 glucose readings (normalized to ~[-1,1])
        [8]     : Glucose rate-of-change (mg/dL per 5 min, scaled)
        [9]     : Glucose acceleration (2nd derivative, scaled)
        [10]    : Glucose deviation from target (signed, scaled)
        [11]    : Time-of-day sin
        [12]    : Time-of-day cos
        [13]    : Insulin-on-board (normalized)
        [14-16] : Last 3 insulin doses (normalized)
        [17]    : Binary: glucose > 180 (hyper zone flag)

    Action: continuous [-1, 1] mapped via exponential to [0, max_insulin] U/min

    Reward: Magni risk index + zone-based shaping.
    """

    metadata = {"render_modes": []}

    def __init__(self, patient_name="adult#001", max_insulin=0.2, seed=None, safety_layer=None):
        super().__init__()

        self.patient_name = patient_name
        self.max_insulin = max_insulin  # U/min (12 U/hr max for meal coverage)
        self._seed = seed
        self.safety_layer = safety_layer  # Optional: applied before simulator step

        # Normalization constants
        self.glucose_min = 40.0
        self.glucose_max = 400.0
        self.glucose_target = 120.0
        self.max_iob = 1.0  # max IOB for normalization

        # Spaces
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Episode config
        self.max_steps = 288  # 24 hours at 5-min intervals

        # Internal state
        self.sim_env = None
        self.step_count = 0
        self.glucose_history = deque(maxlen=10)  # extra buffer for derivatives
        self.insulin_history = deque(maxlen=3)
        self.iob = 0.0

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _norm_glucose(self, glucose):
        """Normalize glucose to ~[-1, 1]."""
        return 2.0 * (glucose - self.glucose_min) / (self.glucose_max - self.glucose_min) - 1.0

    def _norm_iob(self, iob):
        return np.clip(iob / self.max_iob, 0.0, 1.5)

    def _norm_insulin(self, dose):
        return np.clip(dose / self.max_insulin, 0.0, 1.0) if self.max_insulin > 0 else 0.0

    # ------------------------------------------------------------------
    # Action mapping: exponential squashing
    # ------------------------------------------------------------------

    def _map_action(self, raw_action):
        """
        Map [-1, 1] → [0, max_insulin] with exponential bias toward low doses.

        power=2 means: action=0 → ~25% of max, action=-1 → 0, action=1 → max
        Random exploration (mean ~0) → ~0.05 * max = ~0.01 U/min = 0.6 U/hr
        """
        # Linear map to [0, 1]
        a = np.clip((raw_action + 1.0) / 2.0, 0.0, 1.0)
        # Exponential squashing (bias toward low)
        a_exp = a ** 2.5
        return float(self.max_insulin * a_exp)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self):
        """Build 18-dim observation vector."""
        gl = list(self.glucose_history)

        # -- Glucose history (last 8, padded) --
        hist = gl[-8:] if len(gl) >= 8 else ([gl[0]] * (8 - len(gl))) + gl
        glucose_norm = [self._norm_glucose(g) for g in hist]

        # -- Rate of change (mg/dL per 5 min, scaled by 50) --
        if len(gl) >= 2:
            roc = (gl[-1] - gl[-2]) / 50.0
        else:
            roc = 0.0
        roc = np.clip(roc, -1.5, 1.5)

        # -- Acceleration (2nd derivative) --
        if len(gl) >= 3:
            acc = ((gl[-1] - gl[-2]) - (gl[-2] - gl[-3])) / 50.0
        else:
            acc = 0.0
        acc = np.clip(acc, -1.5, 1.5)

        # -- Deviation from target (signed, scaled) --
        deviation = (gl[-1] - self.glucose_target) / 100.0
        deviation = np.clip(deviation, -2.0, 2.0)

        # -- Time of day --
        time_hours = (self.step_count * 5.0) / 60.0
        time_sin = np.sin(2 * np.pi * time_hours / 24.0)
        time_cos = np.cos(2 * np.pi * time_hours / 24.0)

        # -- IOB --
        iob_norm = self._norm_iob(self.iob)

        # -- Last 3 insulin doses --
        ins = list(self.insulin_history)
        while len(ins) < 3:
            ins.insert(0, 0.0)
        ins_norm = [self._norm_insulin(d) for d in ins]

        # -- Hyper zone flag --
        hyper_flag = 1.0 if gl[-1] > 180.0 else 0.0

        obs = np.array(
            glucose_norm
            + [roc, acc, deviation, time_sin, time_cos, iob_norm]
            + ins_norm
            + [hyper_flag],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Reward: Magni risk index + zone shaping
    # ------------------------------------------------------------------

    def _compute_reward(self, glucose):
        """
        Reward combining Magni risk index and zone-based shaping.

        - Magni risk: log-transform penalizes deviations asymmetrically
          (hypo ~5× more dangerous than hyper, matching clinical reality)
        - Zone bonus: smooth linear reward for being in target range
        - Hyper gradient: explicit penalty proportional to distance above 180
        """
        # --- Magni risk index (lower is better) ---
        # risk = 10 * (log(glucose) - log(target))^2
        # We negate it and scale to make it a reward component
        if glucose > 1.0:  # guard log
            magni_risk = 10.0 * (np.log(glucose) - np.log(self.glucose_target)) ** 2
        else:
            magni_risk = 100.0  # extreme penalty for impossible values

        # Convert risk to reward: max ~1.0 at target, decays for deviations
        risk_reward = np.exp(-0.1 * magni_risk)  # [0, 1]

        # --- Zone-based shaping ---
        if 70.0 <= glucose <= 180.0:
            # In range: bonus scaled by proximity to target
            dist_to_target = abs(glucose - self.glucose_target)
            zone_reward = 0.5 * (1.0 - dist_to_target / 60.0)  # max 0.5 at target
            zone_reward = max(zone_reward, 0.1)  # minimum 0.1 for being in range
        elif glucose < 70.0:
            # Hypoglycemia: steep penalty
            zone_reward = -2.0 * ((70.0 - glucose) / 30.0) ** 1.5
        else:
            # Hyperglycemia: linear penalty proportional to excess
            zone_reward = -0.3 * (glucose - 180.0) / 100.0

        # --- Combine ---
        reward = risk_reward + zone_reward

        return float(reward)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            rng_seed = seed
        elif self._seed is not None:
            rng_seed = self._seed
        else:
            rng_seed = np.random.randint(0, 2**31)

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName("Dexcom", seed=rng_seed)
        pump = InsulinPump.withName("Insulet")
        scenario = RandomScenario(start_time=start_time, seed=rng_seed)

        self.sim_env = T1DSimEnv(patient, sensor, pump, scenario)

        self.step_count = 0
        self.iob = 0.0
        self.glucose_history.clear()
        self.insulin_history.clear()

        # Reset safety layer if present
        if self.safety_layer is not None:
            self.safety_layer.reset()

        obs, _, _, _ = self.sim_env.reset()
        initial_glucose = obs.CGM

        for _ in range(10):
            self.glucose_history.append(initial_glucose)

        return self._build_obs(), {}

    def step(self, action):
        # Map action with exponential squashing
        raw_action = float(action[0])
        insulin_dose = self._map_action(raw_action)

        # Apply safety guardrails BEFORE the simulator step
        safety_interventions = []
        if self.safety_layer is not None:
            current_glucose = self.glucose_history[-1]
            insulin_dose, safety_interventions = self.safety_layer.apply(
                insulin_dose, current_glucose, iob=self.iob
            )

        # Deliver to simulator (with safety-filtered dose)
        action_obj = Action(basal=insulin_dose, bolus=0)
        obs, _, _, _ = self.sim_env.step(action_obj)
        glucose = obs.CGM

        # Update tracking
        self.glucose_history.append(glucose)
        self.insulin_history.append(insulin_dose)

        # IOB decay (~60 min half-life)
        self.iob = self.iob * 0.95 + insulin_dose

        self.step_count += 1

        reward = self._compute_reward(glucose)

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "glucose": glucose,
            "insulin_dose": insulin_dose,
            "iob": self.iob,
            "safety_interventions": safety_interventions,
        }

        return self._build_obs(), reward, terminated, truncated, info

