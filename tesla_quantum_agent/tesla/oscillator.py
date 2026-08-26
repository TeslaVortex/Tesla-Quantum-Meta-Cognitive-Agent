"""Temporal oscillator: predict the next query via phase."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


class TemporalOscillator:
    """Dimension 4 – predict the next query via circadian / session phase."""

    def __init__(self):
        self.phase_history: Dict[str, List[Dict]] = defaultdict(list)

    def record(
        self,
        user_id: str,
        embedding: np.ndarray,
        ts: Optional[datetime] = None,
    ) -> None:
        ts = ts or datetime.now()
        phase = ts.hour * 60 + ts.minute
        vec = np.asarray(embedding, dtype=np.float32)
        self.phase_history[user_id].append({"embedding": vec, "phase": phase, "ts": ts})

    def predict_and_prewarm(
        self,
        user_id: str,
        current_embedding: np.ndarray,
    ) -> Optional[np.ndarray]:
        history = self.phase_history.get(user_id, [])
        if len(history) < 5:
            return None
        current_phase = datetime.now().hour * 60 + datetime.now().minute
        similar = [h for h in history if abs(h["phase"] - current_phase) < 45]
        if similar:
            return np.mean([h["embedding"] for h in similar], axis=0)
        return None

    def phase_frequency(self, user_id: str, bins: int = 24) -> np.ndarray:
        """Histogram of query phases — the user's frequency signature."""
        hist = np.zeros(bins, dtype=np.float32)
        for h in self.phase_history.get(user_id, []):
            hist[int(h["phase"] / (24 * 60) * bins) % bins] += 1.0
        total = hist.sum()
        return hist / total if total else hist
