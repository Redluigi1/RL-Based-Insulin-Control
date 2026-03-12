"""
Safety Guardrails Module for RL Insulin Delivery.

Applied during evaluation (and optionally during late-stage training)
to enforce clinical safety constraints on top of RL decisions.
"""

import numpy as np
from collections import deque


class SafetyGuardrails:
    """
    Multi-layer safety system for insulin delivery:

    1. Hypo Guard: Suspend/reduce insulin when glucose is low
    2. IOB Cap: Prevent insulin stacking
    3. Rate Limiter: Smooth insulin delivery changes
    4. Hard Bounds: Absolute max insulin limit

    The RL agent proposes, the safety layer disposes.
    """

    def __init__(
        self,
        max_insulin_rate=0.2,    # U/min absolute hard limit
        hypo_suspend_threshold=60.0,   # mg/dL: zero insulin below this (lowered from 70)
        hypo_reduce_threshold=80.0,    # mg/dL: reduce insulin below this (lowered from 90)
        hypo_reduce_factor=0.5,        # multiply insulin by this below reduce_threshold
        iob_cap=1.5,                   # max IOB (U) before refusing extra insulin (raised from 0.8)
        max_rate_change=0.1,           # max U/min change per step (raised from 0.05)
    ):
        self.max_insulin_rate = max_insulin_rate
        self.hypo_suspend_threshold = hypo_suspend_threshold
        self.hypo_reduce_threshold = hypo_reduce_threshold
        self.hypo_reduce_factor = hypo_reduce_factor
        self.iob_cap = iob_cap
        self.max_rate_change = max_rate_change

        # State
        self.last_insulin = 0.0
        self.iob = 0.0
        self.intervention_log = []

    def apply(self, proposed_insulin, glucose, iob=None):
        """
        Evaluate proposed insulin dose against safety constraints.

        Args:
            proposed_insulin: RL-proposed dose (U/min)
            glucose: Current blood glucose (mg/dL)
            iob: Insulin-on-board (U). If None, uses internal tracking.

        Returns:
            (safe_insulin, interventions) tuple
            - safe_insulin: Approved insulin dose (U/min)
            - interventions: list of string descriptions of any modifications
        """
        dose = proposed_insulin
        interventions = []

        if iob is not None:
            self.iob = iob

        # --- Layer 1: Hard bounds ---
        if dose < 0:
            dose = 0.0
            interventions.append("Clipped negative dose to 0")
        if dose > self.max_insulin_rate:
            dose = self.max_insulin_rate
            interventions.append(f"Hard cap: {proposed_insulin:.4f} → {dose:.4f} U/min")

        # --- Layer 2: Hypo guard ---
        if glucose < self.hypo_suspend_threshold:
            dose = 0.0
            interventions.append(
                f"HYPO SUSPEND (BG={glucose:.0f} < {self.hypo_suspend_threshold}): insulin halted"
            )
        elif glucose < self.hypo_reduce_threshold:
            reduced = dose * self.hypo_reduce_factor
            interventions.append(
                f"Hypo reduce (BG={glucose:.0f} < {self.hypo_reduce_threshold}): "
                f"{dose:.4f} → {reduced:.4f} U/min"
            )
            dose = reduced

        # --- Layer 3: IOB cap ---
        if self.iob > self.iob_cap and dose > 0:
            excess_ratio = self.iob / self.iob_cap
            # Linearly reduce: at 1× cap → full dose, at 2× cap → zero
            scale = max(0.0, 2.0 - excess_ratio)
            reduced = dose * scale
            if reduced < dose:
                interventions.append(
                    f"IOB cap ({self.iob:.2f}U > {self.iob_cap}U): "
                    f"{dose:.4f} → {reduced:.4f} U/min"
                )
            dose = reduced

        # --- Layer 4: Rate limiter ---
        delta = dose - self.last_insulin
        if abs(delta) > self.max_rate_change:
            clamped = self.last_insulin + np.clip(delta, -self.max_rate_change, self.max_rate_change)
            clamped = max(0.0, clamped)
            if abs(clamped - dose) > 1e-6:
                interventions.append(
                    f"Rate limit: {dose:.4f} → {clamped:.4f} U/min "
                    f"(max Δ={self.max_rate_change})"
                )
            dose = clamped

        # Update internal state
        self.last_insulin = dose
        self.iob = self.iob * 0.95 + dose  # exponential decay + new dose

        # Log intervention
        if interventions:
            self.intervention_log.append({
                "proposed": proposed_insulin,
                "approved": dose,
                "glucose": glucose,
                "reasons": interventions,
            })

        return dose, interventions

    def reset(self):
        """Reset state for new episode."""
        self.last_insulin = 0.0
        self.iob = 0.0
        self.intervention_log = []

    def get_stats(self):
        """Return summary statistics of interventions."""
        total = len(self.intervention_log)
        if total == 0:
            return {"total_interventions": 0}

        hypo_suspends = sum(
            1 for entry in self.intervention_log
            if any("HYPO SUSPEND" in r for r in entry["reasons"])
        )
        iob_caps = sum(
            1 for entry in self.intervention_log
            if any("IOB cap" in r for r in entry["reasons"])
        )
        rate_limits = sum(
            1 for entry in self.intervention_log
            if any("Rate limit" in r for r in entry["reasons"])
        )

        return {
            "total_interventions": total,
            "hypo_suspends": hypo_suspends,
            "iob_caps": iob_caps,
            "rate_limits": rate_limits,
        }
