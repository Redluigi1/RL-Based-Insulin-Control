import numpy as np

from datetime import datetime
from datetime import timedelta

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action


class SimGlucoseEnvWrapper:
    """
    Wrapper so SimGlucose behaves like a simple RL environment.
    """

    def __init__(self, patient_name="adult#001", seed=1):

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        self.patient = T1DPatient.withName(patient_name)
        self.sensor = CGMSensor.withName("Dexcom", seed=seed)
        self.pump = InsulinPump.withName("Insulet")
        self.scenario = RandomScenario(start_time=start_time, seed=seed)

        self.env = T1DSimEnv(
            self.patient,
            self.sensor,
            self.pump,
            self.scenario
        )

        self.max_steps = 288
        self.step_count = 0

        self.active_insulin = 0.0
        self.last_observation = None

    def reset(self):

        obs, _, _, _ = self.env.reset()

        glucose = obs.CGM

        self.step_count = 0
        self.active_insulin = 0.0
        self.last_observation = obs

        return glucose

    def step(self, insulin_dose):

        action = Action(basal=insulin_dose, bolus=0)

        obs, reward, done, info = self.env.step(action)

        glucose = obs.CGM

        self.step_count += 1

        # simple active insulin approximation
        self.active_insulin = 0.9 * self.active_insulin + insulin_dose

        if self.step_count >= self.max_steps:
            done = True

        self.last_observation = obs

        return glucose, done