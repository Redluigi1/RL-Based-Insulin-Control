"""
Reward Shaping Module
Defines the objective function (reward) that guides the RL agent.
Balances glucose control (Time-in-Range) with safety.
"""

import numpy as np
from typing import Dict


class RewardShaper:
    """
    Computes reward signal based on glucose control and safety metrics.
    
    Reward components:
    1. Time-in-Range (TIR): Bonus for glucose in [70, 180] mg/dL
    2. Hypoglycemia penalty: Severe penalty for BG < 70
    3. Hyperglycemia penalty: Moderate penalty for BG > 180
    4. Smoothness: Penalty for erratic insulin changes
    5. Efficiency: Bonus for minimal insulin while maintaining TIR
    """
    
    def __init__(
        self,
        target_low: float = 70.0,
        target_high: float = 180.0,
        tight_low: float = 100.0,
        tight_high: float = 160.0,
        hypo_threshold: float = 70.0,
        severe_hypo: float = 54.0,
    ):
        """
        Args:
            target_low/high: Target glucose range (70-180 mg/dL)
            tight_low/high: Tighter "excellent" range for bonus (100-160 mg/dL)
            hypo_threshold: Threshold for hypoglycemia alert
            severe_hypo: Threshold for severe hypoglycemia (requires intervention)
        """
        
        self.target_low = target_low
        self.target_high = target_high
        self.tight_low = tight_low
        self.tight_high = tight_high
        self.hypo_threshold = hypo_threshold
        self.severe_hypo = severe_hypo
        
        # Reward weights
        self.w_tir = 1.0               # Base reward for being in range
        self.w_tight = 0.5             # Bonus for tight control
        self.w_hypo = -10.0            # Severe penalty for hypoglycemia
        self.w_severe_hypo = -100.0    # Catastrophic penalty
        self.w_hyper = -0.5            # Mild penalty for hyperglycemia
        self.w_insulin_smoothness = -0.1  # Small penalty for insulin spikes
        self.w_insulin_efficiency = 0.05  # Small bonus for minimal insulin
    
    def compute_reward(
        self,
        glucose: float,
        previous_glucose: float = None,
        insulin_dose: float = 0.0,
        previous_insulin: float = 0.0,
        active_insulin: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute multi-component reward signal.
        
        Args:
            glucose: Current blood glucose (mg/dL)
            previous_glucose: Previous glucose reading (for trend)
            insulin_dose: Current insulin dose (units/min)
            previous_insulin: Previous insulin dose (for smoothness)
            active_insulin: Total active insulin (for efficiency)
            
        Returns:
            Dictionary with reward components and total
        """
        
        reward_dict = {}
        
        # ===== Time-in-Range (TIR) =====
        if self.target_low <= glucose <= self.target_high:
            reward_dict['tir'] = self.w_tir
        else:
            reward_dict['tir'] = 0.0
        
        # ===== Tight Control Bonus =====
        if self.tight_low <= glucose <= self.tight_high:
            reward_dict['tight_control'] = self.w_tight
        else:
            reward_dict['tight_control'] = 0.0
        
        # ===== Hypoglycemia Penalties =====
        if glucose < self.severe_hypo:
            # Severe hypo: immediate action required
            reward_dict['severe_hypo'] = self.w_severe_hypo
            reward_dict['hypo'] = 0.0
        elif glucose < self.hypo_threshold:
            # Moderate hypo: penalty scaled by severity
            hypo_severity = (self.hypo_threshold - glucose) / self.hypo_threshold
            reward_dict['hypo'] = self.w_hypo * hypo_severity
            reward_dict['severe_hypo'] = 0.0
        else:
            reward_dict['hypo'] = 0.0
            reward_dict['severe_hypo'] = 0.0
        
        # ===== Hyperglycemia Penalty =====
        if glucose > self.target_high:
            # Scale by how far above target
            hyper_severity = min(1.0, (glucose - self.target_high) / 100)
            reward_dict['hyper'] = self.w_hyper * hyper_severity
        else:
            reward_dict['hyper'] = 0.0
        
        # ===== Insulin Smoothness Penalty =====
        if previous_insulin is not None:
            # Penalize large changes in insulin delivery
            insulin_change = abs(insulin_dose - previous_insulin)
            reward_dict['smoothness'] = -self.w_insulin_smoothness * insulin_change
        else:
            reward_dict['smoothness'] = 0.0
        
        # ===== Insulin Efficiency Bonus =====
        # Reward low insulin if still maintaining TIR
        if self.target_low <= glucose <= self.target_high and insulin_dose < 5.0:
            reward_dict['efficiency'] = self.w_insulin_efficiency
        else:
            reward_dict['efficiency'] = 0.0
        
        # ===== Glucose Trend Bonus (optional) =====
        # Bonus if glucose is returning to target after excursion
        if previous_glucose is not None:
            trend = glucose - previous_glucose
            
            if glucose > self.target_high and trend < -5:
                # Falling from high glucose (good!)
                reward_dict['trend'] = 0.2
            elif glucose < self.target_low and trend > 5:
                # Rising from low glucose (good!)
                reward_dict['trend'] = 0.2
            else:
                reward_dict['trend'] = 0.0
        else:
            reward_dict['trend'] = 0.0
        
        # ===== Total Reward =====
        total = sum(reward_dict.values())
        reward_scaling = 1e-2
        reward_dict['total'] = total*reward_scaling
        
        return reward_dict
    
    def get_glucose_category(self, glucose: float) -> str:
        """Categorize glucose level for logging."""
        if glucose < self.severe_hypo:
            return "SEVERE_HYPO"
        elif glucose < self.hypo_threshold:
            return "HYPO"
        elif glucose < self.target_low:
            return "LOW"
        elif self.tight_low <= glucose <= self.tight_high:
            return "TIGHT"
        elif glucose <= self.target_high:
            return "TARGET"
        else:
            return "HIGH"


class CumulativeMetrics:
    """
    Track cumulative metrics over an episode (24 hours).
    """
    
    def __init__(self):
        self.glucose_readings = []
        self.insulin_doses = []
        self.rewards = []
        self.time_steps = 0
        
    def update(
        self,
        glucose: float,
        insulin: float,
        reward: float,
    ):
        """Record a single timestep."""
        self.glucose_readings.append(glucose)
        self.insulin_doses.append(insulin)
        self.rewards.append(reward)
        self.time_steps += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute episode metrics."""
        
        glucose = np.array(self.glucose_readings)
        insulin = np.array(self.insulin_doses)
        rewards = np.array(self.rewards)
        
        metrics = {
            # Glucose control metrics
            'mean_glucose': np.mean(glucose),
            'std_glucose': np.std(glucose),
            'min_glucose': np.min(glucose),
            'max_glucose': np.max(glucose),
            
            # Time-in-Range (TIR)
            'tir_70_180': 100 * np.sum((glucose >= 70) & (glucose <= 180)) / len(glucose),
            'tir_80_140': 100 * np.sum((glucose >= 80) & (glucose <= 140)) / len(glucose),
            'tbr_below_70': 100 * np.sum(glucose < 70) / len(glucose),
            'tar_above_180': 100 * np.sum(glucose > 180) / len(glucose),
            
            # Hypo/Hyper metrics
            'time_below_54': 100 * np.sum(glucose < 54) / len(glucose),
            'time_above_250': 100 * np.sum(glucose > 250) / len(glucose),
            'hypo_episodes': self._count_episodes(glucose, threshold=70),
            'hyper_episodes': self._count_episodes(glucose, threshold=180, direction='above'),
            
            # Insulin metrics
            'total_insulin': np.sum(insulin),
            'mean_insulin': np.mean(insulin),
            'max_insulin': np.max(insulin),
            'insulin_variability': np.std(insulin),
            
            # Reward metrics
            'cumulative_reward': np.sum(rewards),
            'mean_reward': np.mean(rewards),
            'episode_length': self.time_steps,
        }
        
        return metrics
    
    def _count_episodes(self, glucose: np.ndarray, threshold: float, direction: str = 'below') -> int:
        """Count number of separate episodes (continuous sequences) above/below threshold."""
        
        if direction == 'below':
            in_episode = glucose < threshold
        else:
            in_episode = glucose > threshold
        
        episodes = 0
        was_in = False
        
        for is_in in in_episode:
            if is_in and not was_in:
                episodes += 1
            was_in = is_in
        
        return episodes
    
    def reset(self):
        """Clear for new episode."""
        self.glucose_readings = []
        self.insulin_doses = []
        self.rewards = []
        self.time_steps = 0

