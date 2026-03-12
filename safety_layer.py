"""
Safety Layer Module
Implements PID controller fallback + constrained action validation
Based on patent US 2022/0088304 A1 (Harvard College)
"""

import numpy as np
from typing import Tuple, Optional


class SafetyLayer:
    """
    Multi-level safety constraint for insulin delivery:
    1. Hard bounds (max insulin, max rate of change)
    2. PID fallback for unsafe RL actions
    3. Insulin feedback (IFB) to prevent "stacking"
    
    The RL agent proposes doses, but the Safety Layer decides what actually gets delivered.
    """
    
    def __init__(
        self,
        max_insulin_per_minute: float = 10.0,  # units/min
        max_insulin_per_bolus: float = 50.0,   # units total
        max_rate_of_change: float = 5.0,       # max units/min change
        target_glucose: float = 130.0,         # mg/dL
        glucose_lower_bound: float = 70.0,     # Safety threshold
        glucose_upper_bound: float = 180.0,    # Safety threshold
    ):
        """
        Args:
            max_insulin_per_minute: Hard limit on insulin infusion rate
            max_insulin_per_bolus: Hard limit on single bolus
            max_rate_of_change: Max acceleration of insulin delivery
            target_glucose: Target glucose level
            glucose_lower_bound: Below this, no basal insulin
            glucose_upper_bound: Above this, max correction
        """
        
        self.max_insulin_per_minute = max_insulin_per_minute
        self.max_insulin_per_bolus = max_insulin_per_bolus
        self.max_rate_of_change = max_rate_of_change
        self.target_glucose = target_glucose
        self.glucose_lower_bound = glucose_lower_bound
        self.glucose_upper_bound = glucose_upper_bound
        
        # PID controller state
        self.kp = 0.05        # Proportional gain
        self.ki = 0.001       # Integral gain
        self.kd = 0.1         # Derivative gain
        self.error_integral = 0.0
        self.last_glucose = target_glucose
        
        # Insulin feedback (IFB) tracking
        self.active_insulin_buffer = 0.0  # Tracks total active insulin
        self.insulin_decay_rate = 0.02    # Exponential decay per minute
        
    def evaluate_action(
        self,
        rl_proposed_insulin: float,
        glucose: float,
        active_insulin: float,
        previous_action: float = 0.0,
        use_pid_override: bool = False,
    ) -> Tuple[float, str, bool]:
        """
        Evaluate RL-proposed insulin dose against safety constraints.
        
        Args:
            rl_proposed_insulin: Insulin dose (units/min) proposed by RL agent
            glucose: Current blood glucose (mg/dL)
            active_insulin: Current active insulin (units)
            previous_action: Last insulin dose delivered (for rate-of-change check)
            use_pid_override: If True, force PID backup for unsafe actions
            
        Returns:
            (approved_insulin, reason, is_safe) tuple
            - approved_insulin: Actual insulin to deliver (units/min)
            - reason: Human-readable explanation of decision
            - is_safe: Boolean indicating if action is within safe bounds
        """
        
        reasons = []
        is_safe = True
        approved_insulin = rl_proposed_insulin
        
        # ===== Hard Constraint 1: Maximum insulin per minute =====
        if approved_insulin > self.max_insulin_per_minute:
            is_safe = False
            approved_insulin = self.max_insulin_per_minute
            reasons.append(f"RL proposed {rl_proposed_insulin:.2f} u/min (exceeds max {self.max_insulin_per_minute})")
        
        # ===== Hard Constraint 2: Rate of change limit =====
        max_step = self.max_rate_of_change
        if approved_insulin - previous_action > max_step:
            is_safe = False
            approved_insulin = previous_action + max_step
            reasons.append(f"Rate-of-change limited: {approved_insulin:.2f} u/min (max +{max_step}/min)")
        elif approved_insulin - previous_action < -max_step:
            is_safe = False
            approved_insulin = previous_action - max_step
            reasons.append(f"Rate-of-change limited: {approved_insulin:.2f} u/min (max -{max_step}/min)")
        
        # ===== Hard Constraint 3: Hypoglycemia protection =====
        if glucose < self.glucose_lower_bound:
            # Below 70 mg/dL: STOP ALL BASAL, prepare to override
            approved_insulin = 0.0
            is_safe = False
            reasons.append(f"HYPO ALERT (BG={glucose:.0f}): Insulin halted")
        
        # ===== Insulin Feedback (IFB): Prevent stacking =====
        # Update active insulin buffer (exponential decay)
        self.active_insulin_buffer *= (1 - self.insulin_decay_rate)
        self.active_insulin_buffer += approved_insulin
        
        max_active = self.max_insulin_per_bolus
        if self.active_insulin_buffer > max_active:
            # Excessive active insulin: reduce further
            excess = self.active_insulin_buffer - max_active
            approved_insulin = max(0, approved_insulin - excess * 0.5)
            is_safe = False
            reasons.append(f"Insulin stacking prevention: Active {self.active_insulin_buffer:.1f}u (limit {max_active}u)")
        
        # ===== PID Fallback (for unsafe RL actions) =====
        if (not is_safe and use_pid_override) or not self._is_glucose_stable(glucose):
            pid_insulin = self._compute_pid_dose(glucose)
            
            # Only use PID if it's more conservative
            if pid_insulin < approved_insulin:
                approved_insulin = pid_insulin
                reasons.append(f"PID override: {pid_insulin:.2f} u/min")
        
        reason_str = " | ".join(reasons) if reasons else "APPROVED by RL"
        
        # Ensure final output is non-negative
        approved_insulin = max(0.0, approved_insulin)
        
        return approved_insulin, reason_str, is_safe
    
    def _compute_pid_dose(self, glucose: float) -> float:
        """
        Proportional-Integral-Derivative controller.
        Standard formula: u(t) = Kp*e(t) + Ki*∫e(t) + Kd*de/dt
        """
        
        # Proportional error
        error = self.target_glucose - glucose
        
        # Integral error (with anti-windup: cap integral term)
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -100, 100)
        
        # Derivative error
        error_rate = glucose - self.last_glucose
        
        # PID output
        pid_output = (
            self.kp * error +
            self.ki * self.error_integral +
            self.kd * error_rate
        )
        
        # Clamp to valid range
        pid_output = np.clip(pid_output, 0, self.max_insulin_per_minute)
        
        self.last_glucose = glucose
        
        return float(pid_output)
    
    def _is_glucose_stable(self, glucose: float) -> bool:
        """
        Check if glucose is in 'stable' region where RL can safely operate.
        If glucose is trending low or already below threshold, activate PID.
        """
        # Glucose stable if within target ± 50 mg/dL
        stable_lower = self.target_glucose - 50
        stable_upper = self.target_glucose + 50
        
        return stable_lower <= glucose <= stable_upper
    
    def reset(self):
        """Reset PID state for new episode."""
        self.error_integral = 0.0
        self.last_glucose = self.target_glucose
        self.active_insulin_buffer = 0.0
    
    def get_constraints_dict(self) -> dict:
        """Return current safety constraints as dictionary (for logging)."""
        return {
            'max_insulin_per_minute': self.max_insulin_per_minute,
            'max_insulin_per_bolus': self.max_insulin_per_bolus,
            'max_rate_of_change': self.max_rate_of_change,
            'glucose_lower_bound': self.glucose_lower_bound,
            'glucose_upper_bound': self.glucose_upper_bound,
        }
