"""Uncertainty damper: prevent positive feedback of doubt."""

from __future__ import annotations

from typing import List


class UncertaintyDamper:
    """Dimension 6 – prevent uncertainty amplification."""

    def __init__(self, damping: float = 0.65):
        self.damping = damping
        self.last_uncertainty: float = 0.0
        self.emergency: bool = False

    def damp(self, initial_u: float, step_confidences: List[float]) -> float:
        u = float(initial_u)
        self.emergency = False
        for conf in step_confidences:
            u *= 1 - self.damping * float(conf)
            if u > initial_u * 1.1:
                self.emergency = True
                self.last_uncertainty = -1.0
                return -1.0
        self.last_uncertainty = max(0.0, u)
        return self.last_uncertainty
