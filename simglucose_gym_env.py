import gymnasium as gym
from gymnasium import spaces
import numpy as np

from datetime import datetime

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action


from state_encoder import StateEncoder
from safety_layer import SafetyLayer
from reward_shaper import RewardShaper


class SimGlucoseGymEnv(gym.Env):

    def __init__(self, patient_name="adult#001"):

        super().__init__()

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName("Dexcom", seed=1)
        pump = InsulinPump.withName("Insulet")
        scenario = RandomScenario(start_time=start_time, seed=1)

        self.env = T1DSimEnv(patient, sensor, pump, scenario)

        self.encoder = StateEncoder(glucose_history_size=11)
        self.safety = SafetyLayer()
        self.reward_fn = RewardShaper()

        state_dim = self.encoder.get_state_dim()

        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(state_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=5.0,
            shape=(1,),
            dtype=np.float32
        )

        self.max_steps = 288
        self.step_count = 0
        self.prev_action = 0
        self.active_insulin = 0

    def reset(self, seed=None, options=None):

        obs, _, _, _ = self.env.reset()

        self.encoder.reset()
        self.safety.reset()

        self.step_count = 0
        self.prev_action = 0
        self.active_insulin = 0

        glucose = obs.CGM

        state = self.encoder.encode(
            glucose=glucose,
            time_hours=0,
            active_insulin=self.active_insulin,
            activity_level=0,
            routine_context={"meal_likely":0}
        )

        state_vec = self.encoder.flatten_state(state)

        return state_vec, {}

    def step(self, action):

        insulin = float(action[0])

        safe_action, _, _ = self.safety.evaluate_action(
            rl_proposed_insulin=insulin,
            glucose=self.encoder.glucose_history[-1],
            active_insulin=self.active_insulin,
            previous_action=self.prev_action
        )

        action_obj = Action(basal=safe_action, bolus=0)

        obs, _, _, _ = self.env.step(action_obj)

        glucose = obs.CGM

        self.active_insulin = 0.9*self.active_insulin + safe_action

        time_hours = (self.step_count * 5)/60

        state = self.encoder.encode(
            glucose=glucose,
            time_hours=time_hours,
            active_insulin=self.active_insulin,
            activity_level=0,
            routine_context={"meal_likely":0}
        )

        next_state = self.encoder.flatten_state(state)

        # reward = self.reward_fn.compute_reward(glucose)
        reward_dict = self.reward_fn.compute_reward(glucose)
        reward = reward_dict["total"]

        self.prev_action = safe_action
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps

        return next_state, reward, terminated, truncated, {}