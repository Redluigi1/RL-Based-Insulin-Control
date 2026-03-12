"""
Clean Gymnasium wrapper for simglucose.
Compact state, proper normalization, well-shaped reward.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from datetime import datetime, timedelta

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action


class InsulinEnv(gym.Env):
    """
    RL environment for blood glucose control via insulin delivery.

    State (10-dim, all normalized to ~[-1, 1]):
        [0]   : current glucose (normalized)
        [1-3] : glucose rate-of-change over last 3 steps
        [4]   : time-of-day sin
        [5]   : time-of-day cos
        [6]   : insulin-on-board (normalized)
        [7-9] : last 3 insulin doses (normalized)

    Action: continuous [-1, 1] mapped to [0, max_basal] U/min

    Reward: Gaussian-shaped centered at target glucose, with steep hypo penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, patient_name="adult#001", max_basal=0.05, seed=None):
        super().__init__()

        self.patient_name = patient_name
        self.max_basal = max_basal
        self._seed = seed

        # Normalization constants
        self.glucose_min = 40.0
        self.glucose_max = 400.0
        self.glucose_target = 120.0  # ideal center for reward
        self.max_iob = 0.5  # max insulin-on-board for normalization

        # State and action spaces
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Episode config
        self.max_steps = 288  # 24 hours at 5-min intervals

        # Internal state (initialized in reset)
        self.sim_env = None
        self.step_count = 0
        self.glucose_history = deque(maxlen=4)
        self.insulin_history = deque(maxlen=3)
        self.iob = 0.0  # insulin-on-board

    def _normalize_glucose(self, glucose):
        """Normalize glucose to approximately [-1, 1]."""
        return 2.0 * (glucose - self.glucose_min) / (self.glucose_max - self.glucose_min) - 1.0

    def _normalize_iob(self, iob):
        """Normalize insulin-on-board to [0, 1]."""
        return min(iob / self.max_iob, 1.0)

    def _normalize_insulin(self, dose):
        """Normalize insulin dose to [0, 1]."""
        return min(dose / self.max_basal, 1.0) if self.max_basal > 0 else 0.0

    def _build_obs(self):
        """Build the 10-dimensional observation vector."""
        # Current glucose (normalized)
        curr_glucose_norm = self._normalize_glucose(self.glucose_history[-1])

        # Glucose rate of change (last 3 differences, normalized by scale)
        glucose_diffs = []
        gl = list(self.glucose_history)
        for i in range(1, len(gl)):
            diff = (gl[i] - gl[i - 1]) / 50.0  # ±50 mg/dL/5min -> ±1
            glucose_diffs.append(np.clip(diff, -1.0, 1.0))
        # Pad if not enough history
        while len(glucose_diffs) < 3:
            glucose_diffs.insert(0, 0.0)

        # Time of day (circular)
        time_hours = (self.step_count * 5.0) / 60.0
        time_sin = np.sin(2 * np.pi * time_hours / 24.0)
        time_cos = np.cos(2 * np.pi * time_hours / 24.0)

        # Insulin-on-board (normalized)
        iob_norm = self._normalize_iob(self.iob)

        # Last 3 insulin doses (normalized)
        ins_hist = list(self.insulin_history)
        while len(ins_hist) < 3:
            ins_hist.insert(0, 0.0)
        ins_norm = [self._normalize_insulin(d) for d in ins_hist]

        obs = np.array(
            [curr_glucose_norm] + glucose_diffs +
            [time_sin, time_cos, iob_norm] + ins_norm,
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, glucose):
        """
        Well-shaped reward function for glucose control.

        - Gaussian reward centered at target (120 mg/dL), σ=30
          Gives smooth gradient everywhere.
        - Extra penalty for hypo (<70) — exponential drop-off
        - Mild extra penalty for severe hyper (>300)
        """
        # Gaussian component: peaks at 1.0 when glucose == target
        sigma = 30.0
        gaussian_reward = np.exp(-0.5 * ((glucose - self.glucose_target) / sigma) ** 2)

        # Hypoglycemia penalty (steep below 70)
        hypo_penalty = 0.0
        if glucose < 70.0:
            hypo_penalty = -2.0 * np.exp((70.0 - glucose) / 20.0)

        # Severe hyperglycemia penalty (above 300)
        hyper_penalty = 0.0
        if glucose > 300.0:
            hyper_penalty = -0.5 * ((glucose - 300.0) / 100.0)

        reward = gaussian_reward + hypo_penalty + hyper_penalty
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use a random seed for scenario diversity
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

        # Reset internal state
        self.step_count = 0
        self.iob = 0.0
        self.glucose_history.clear()
        self.insulin_history.clear()

        # Get initial observation from simulator
        obs, _, _, _ = self.sim_env.reset()
        initial_glucose = obs.CGM

        # Fill glucose history with initial value
        for _ in range(4):
            self.glucose_history.append(initial_glucose)

        return self._build_obs(), {}

    def step(self, action):
        # Map action from [-1, 1] to [0, max_basal]
        raw_action = float(action[0])
        insulin_dose = self.max_basal * (raw_action + 1.0) / 2.0
        insulin_dose = np.clip(insulin_dose, 0.0, self.max_basal)

        # Apply to simulator
        action_obj = Action(basal=insulin_dose, bolus=0)
        obs, _, _, _ = self.sim_env.step(action_obj)
        glucose = obs.CGM

        # Update internal tracking
        self.glucose_history.append(glucose)
        self.insulin_history.append(insulin_dose)

        # Update insulin-on-board (exponential decay, ~60 min half-life)
        decay = 0.95  # per 5-min step -> half-life ≈ 65 min
        self.iob = self.iob * decay + insulin_dose

        self.step_count += 1

        # Compute reward
        reward = self._compute_reward(glucose)

        # Episode termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "glucose": glucose,
            "insulin_dose": insulin_dose,
            "iob": self.iob,
        }

        return self._build_obs(), reward, terminated, truncated, info
