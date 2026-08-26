"""Zero-Energy Lens: belief-state consistency and energy accounting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class ZeroEnergyLens:
    """Belief-state consistency checking for energy efficiency."""

    def __init__(self, consistency_threshold: float = 0.7):
        self.consistency_threshold = consistency_threshold
        self.belief_history: List[Dict[str, Any]] = []
        self.energy_log: List[float] = []
        self.current_beliefs: Dict[str, str] = {}

    def check_belief_consistency(
        self,
        current_beliefs: Dict[str, str],
        new_belief: str,
    ) -> Tuple[bool, float]:
        """Check if a new belief is consistent with existing beliefs."""
        if current_beliefs:
            self.current_beliefs.update(current_beliefs)
        consistency_score = self._calculate_consistency(self.current_beliefs, new_belief)
        energy_cost = self._calculate_energy_cost(len(self.current_beliefs) + 1)
        self.energy_log.append(energy_cost)
        return consistency_score >= self.consistency_threshold, consistency_score

    def _calculate_consistency(self, current_beliefs: Dict[str, str], new_belief: str) -> float:
        new_terms = set(new_belief.lower().split())
        contradictions = 0.0
        total_checks = 0

        for key, belief in current_beliefs.items():
            if key.lower() in new_belief.lower():
                total_checks += 1
                if str(belief).lower() != new_belief.lower():
                    contradictions += 0.3
                else:
                    contradictions -= 0.1
            for term in self._get_related_terms(key):
                if term in new_terms:
                    total_checks += 1
                    contradictions += 0.05

        if total_checks == 0:
            return 1.0
        return max(0.0, 1.0 - (contradictions / total_checks))

    def _get_related_terms(self, term: str) -> List[str]:
        related_map = {
            "belief": ["opinion", "view", "thought", "perspective"],
            "assumption": ["premise", "supposition", "hypothesis"],
            "conclusion": ["result", "outcome", "finding", "deduction"],
            "evidence": ["data", "proof", "support", "basis"],
            "reasoning": ["logic", "argument", "analysis", "process"],
        }
        return related_map.get(term.lower(), [])

    def _calculate_energy_cost(self, num_beliefs: int) -> float:
        return 1.0 * (max(1, num_beliefs) ** 0.3)

    def log_belief(self, belief: str, timestamp: Optional[datetime] = None) -> None:
        self.belief_history.append(
            {
                "belief": belief,
                "timestamp": timestamp or datetime.now(),
                "energy_cost": self.energy_log[-1] if self.energy_log else 0.0,
            }
        )
        key = belief[:48] if belief else "empty"
        self.current_beliefs[key] = belief

    def check_vibrational_consistency(
        self,
        prior_vibration: float,
        new_vibration: float,
        tolerance: float = 0.35,
    ) -> Tuple[bool, float]:
        """Optional Tesla-lens check: large vibration jumps cost extra energy."""
        delta = abs(float(new_vibration) - float(prior_vibration))
        consistent = delta <= tolerance
        extra = 0.0 if consistent else delta * 0.5
        if extra:
            self.energy_log.append(extra)
        return consistent, 1.0 - min(1.0, delta)

    def get_energy_summary(self) -> Dict[str, Any]:
        total = sum(self.energy_log)
        n = len(self.energy_log)
        recent = []
        for b in self.belief_history[-10:]:
            ts = b["timestamp"]
            recent.append(
                {
                    "belief": b["belief"][:120],
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "energy_cost": b["energy_cost"],
                }
            )
        return {
            "total_energy": total,
            "average_energy": total / n if n else 0.0,
            "belief_count": len(self.belief_history),
            "recent_beliefs": recent,
        }
