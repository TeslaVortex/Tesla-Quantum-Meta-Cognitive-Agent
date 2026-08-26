"""Quantum evaluator: measure coherence, not collapsed accuracy."""

from __future__ import annotations

from typing import List

import numpy as np


class QuantumEvaluator:
    """Dimension 7 – coherence of the uncollapsed field."""

    def __init__(self):
        self.history: List[str] = []

    def coherence(self, responses: List[str]) -> float:
        if len(responses) < 2:
            return 1.0
        lengths = np.array([len(r) for r in responses], dtype=np.float32)
        length_stab = 1.0 - (float(np.std(lengths)) / (float(np.mean(lengths)) + 1e-6))

        # Lexical overlap stability between consecutive answers.
        overlaps = []
        for a, b in zip(responses[:-1], responses[1:]):
            wa, wb = set(a.lower().split()), set(b.lower().split())
            denom = max(len(wa | wb), 1)
            overlaps.append(len(wa & wb) / denom)
        lexical = float(np.mean(overlaps)) if overlaps else 1.0
        return float(max(0.0, min(1.0, 0.5 * length_stab + 0.5 * lexical)))

    def observe(self, response: str) -> float:
        self.history.append(response)
        return self.coherence(self.history[-8:])

    def novelty_alignment(
        self,
        claimed_vibration: float,
        residual_energy: float,
        rare_hits: float,
    ) -> float:
        """How well the vibration signal agrees with SAE/VQ novelty cues."""
        proxy = 0.5 * float(residual_energy) + 0.5 * min(1.0, float(rare_hits) / 4.0)
        return float(1.0 - abs(float(claimed_vibration) - proxy))
