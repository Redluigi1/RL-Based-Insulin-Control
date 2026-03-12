import numpy as np

class SafetyLayer:

    def __init__(
        self,
        max_insulin_per_minute=5.0,
        max_rate_of_change=2.0,
        glucose_lower_bound=70.0
    ):
        self.max_insulin_per_minute = max_insulin_per_minute
        self.max_rate_of_change = max_rate_of_change
        self.glucose_lower_bound = glucose_lower_bound

    def evaluate_action(self, insulin, glucose, previous_action):

        insulin = min(insulin, self.max_insulin_per_minute)

        delta = insulin - previous_action
        delta = np.clip(delta, -self.max_rate_of_change, self.max_rate_of_change)
        insulin = previous_action + delta

        if glucose < self.glucose_lower_bound:
            insulin = 0.0

        return max(0.0, insulin)