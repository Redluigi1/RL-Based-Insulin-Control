"""
State Encoder Module
Converts raw sensor data into normalized RL state representation
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class StateEncoder:
    """
    Converts raw glucose, time, and activity signals into a normalized state vector.
    
    Input channels:
    - Continuous Glucose Monitor (CGM): BG readings every 5 minutes
    - Clock: Time of day (0-23 hours)
    - Accelerometer + GSR: Activity/stress signals
    - Active Insulin: Residual insulin from previous doses
    
    Output: Normalized state dict with keys for neural network consumption
    """
    
    def __init__(
        self,
        glucose_history_size: int = 11,
        normalize_glucose_range: Tuple[float, float] = (40, 400),
        target_glucose: float = 130,
    ):
        """
        Args:
            glucose_history_size: Number of past glucose readings to include
            normalize_glucose_range: (min, max) for glucose normalization
            target_glucose: Target glucose level (mg/dL) for baseline
        """
        self.glucose_history_size = glucose_history_size
        self.normalize_glucose_range = normalize_glucose_range
        self.target_glucose = target_glucose
        
        # Circular buffers for history
        self.glucose_history = deque(maxlen=glucose_history_size)
        self.activity_history = deque(maxlen=6)  # Last 30 minutes at 5-min intervals
        
    def encode(
        self,
        glucose: float,
        time_hours: float,
        active_insulin: float,
        activity_level: float = 0.0,
        routine_context: Dict[str, float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Encode all sensor signals into normalized state representation.
        
        Args:
            glucose: Current blood glucose (mg/dL)
            time_hours: Time of day (0-23)
            active_insulin: Remaining active insulin (units)
            activity_level: Physical activity [0, 1] from accelerometer
            routine_context: Dict with routine probabilities, e.g. {'meal_likely': 0.8}
            
        Returns:
            Dictionary with normalized state components suitable for NN input
        """
        
        # Add to histories
        self.glucose_history.append(glucose)
        self.activity_history.append(activity_level)
        
        # === Glucose Features ===
        glucose_trend = self._compute_glucose_trend()
        glucose_norm = glucose
        
        # === Time Features (Circular Encoding) ===
        # Use sine/cosine to make time circular (23:59 -> 0:00 is continuous)
        time_sin = np.sin(2 * np.pi * time_hours / 24)
        time_cos = np.cos(2 * np.pi * time_hours / 24)
        
        # === Activity Features ===
        activity_trend = np.mean(list(self.activity_history)) if self.activity_history else 0.0
        activity_norm = np.clip(activity_level, 0, 1)  # Already [0, 1]
        
        # === Insulin Features ===
        # Normalize active insulin: typically ranges [0, 50] units
        active_insulin_norm = active_insulin  # Allow slight overflow
        
        # === Routine Context ===
        if routine_context is None:
            routine_context = {
                'meal_likely': 0.0,
                'exercise_likely': 0.0,
                'high_bg_likely': 0.0,
            }
        
        # === Compile State ===
        state = {
            # Glucose (normalized to [-1, 1] roughly)
            'glucose_current': np.array([glucose_norm], dtype=np.float32),
            'glucose_history': np.array(
                self._pad_glucose_history(glucose_norm),
                dtype=np.float32
            ),
            'glucose_trend': np.array([glucose_trend], dtype=np.float32),
            
            # Time (circular encoding)
            'time_of_day': np.array([time_sin, time_cos], dtype=np.float32),
            
            # Activity
            'activity_level': np.array([activity_norm], dtype=np.float32),
            'activity_trend': np.array([activity_trend], dtype=np.float32),
            
            # Insulin
            'active_insulin': np.array([active_insulin_norm], dtype=np.float32),
            
            # Routine context (from long-term pattern learner)
            'routine_context': np.array([
                routine_context.get('meal_likely', 0.0),
                routine_context.get('exercise_likely', 0.0),
                routine_context.get('high_bg_likely', 0.0),
            ], dtype=np.float32),
        }
        
        return state
    
    def _compute_glucose_trend(self) -> float:
        """
        Compute glucose rate of change (mg/dL per 5 minutes).
        Positive = rising, Negative = falling.
        """
        if len(self.glucose_history) < 2:
            return 0.0
        
        # Last two readings (5 minutes apart)
        current = self.glucose_history[-1]
        previous = self.glucose_history[-2]
        trend = current - previous
        
        # Normalize: ±50 mg/dL/5min → ±1.0
        trend_normalized = trend
        
        return float(trend_normalized)
    
    def _pad_glucose_history(self, current_glucose: float) -> List[float]:
        """
        Pad glucose history to fixed size.
        If not enough history, repeat the current value.
        """
        history = list(self.glucose_history)
        
        while len(history) < self.glucose_history_size:
            if len(history) == 0:
                history.insert(0, current_glucose)
            else:
                history.insert(0, history[0])  # Repeat oldest
        
        return history[-self.glucose_history_size:]
    
    def flatten_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten state dict into single 1D vector for simple NN architectures.
        Order: glucose_current + glucose_history + glucose_trend 
               + time_of_day + activity + active_insulin + routine_context
        """
        components = [
            state['glucose_current'],
            state['glucose_history'],
            state['glucose_trend'],
            state['time_of_day'],
            state['activity_level'],
            state['activity_trend'],
            state['active_insulin'],
            state['routine_context'],
        ]
        
        flattened = np.concatenate(components, axis=0)
        return flattened.astype(np.float32)
    
    def get_state_dim(self) -> int:
        """Return dimensionality of flattened state vector."""
        # 1 (current) + 11 (history) + 1 (trend) + 2 (time) 
        # + 1 (activity) + 1 (activity_trend) + 1 (insulin) + 3 (routine)
        return 1 + 11 + 1 + 2 + 1 + 1 + 1 + 3
    
    def reset(self):
        """Clear history buffers for new episode."""
        self.glucose_history.clear()
        self.activity_history.clear()


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    
    encoder = StateEncoder(glucose_history_size=11, target_glucose=130)
    
    # Simulate 3 timesteps of data
    print("=== Example: State Encoding ===\n")
    
    for step in range(3):
        # Simulate sensor readings
        glucose = 150 + 20 * np.sin(step * 0.1)  # Oscillating glucose
        time_hours = 12.5 + step * (1/60)  # Around noon
        active_insulin = 3.5 - step * 0.5  # Decaying insulin
        activity = 0.3 * np.sin(step)  # Slight activity
        
        routine = {
            'meal_likely': 0.1,
            'exercise_likely': 0.0,
            'high_bg_likely': 0.2,
        }
        
        # Encode
        state = encoder.encode(glucose, time_hours, active_insulin, activity, routine)
        flattened = encoder.flatten_state(state)
        
        print(f"Step {step}:")
        print(f"  Glucose: {glucose:.1f} mg/dL")
        print(f"  Time: {time_hours:.2f}h")
        print(f"  Active Insulin: {active_insulin:.2f} units")
        print(f"  State vector shape: {flattened.shape}")
        print(f"  State vector (first 5): {flattened[:5]}")
        print()
    
    print(f"Total state dimension: {encoder.get_state_dim()}")