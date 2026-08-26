"""Tesla Perception Layer: resonance detection, LSH, standing-wave cache."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import lancedb

    HAS_LANCE = True
except ImportError:
    HAS_LANCE = False

if TYPE_CHECKING:
    from tesla_quantum_agent.core.embedder import QuantizedVibrationalState, VectorEmbedder


@dataclass
class ResonanceResult:
    hit: bool
    content: Optional[str] = None
    vibration: float = 0.0
    energy_cost: float = 0.0
    codes: Optional[np.ndarray] = None
    source: str = "cache"
    similarity: float = 0.0
    qstate: Optional[Any] = None


class RandomProjectionLSH:
    """Locality-sensitive hash via random hyperplanes (not cryptographic hashing)."""

    def __init__(self, dim: int, n_bits: int = 32, seed: int = 369):
        rng = np.random.RandomState(seed)
        planes = rng.randn(n_bits, dim).astype(np.float32)
        planes /= np.linalg.norm(planes, axis=1, keepdims=True) + 1e-8
        self.planes = planes
        self.n_bits = n_bits
        self.buckets: Dict[int, List[str]] = defaultdict(list)

    def hash_vec(self, vec: np.ndarray) -> int:
        bits = (self.planes @ vec.astype(np.float32)) >= 0
        value = 0
        for b in bits:
            value = (value << 1) | int(bool(b))
        return int(value)

    def add(self, key: str, vec: np.ndarray) -> int:
        h = self.hash_vec(vec)
        if key not in self.buckets[h]:
            self.buckets[h].append(key)
        return h

    def neighbors(self, vec: np.ndarray) -> List[str]:
        return list(self.buckets.get(self.hash_vec(vec), []))


class TeslaPerceptionLayer:
    """Dimension 1 – capture at the point of least resistance."""

    def __init__(
        self,
        embedder: "VectorEmbedder",
        resonance_threshold: float = 0.92,
        lsh_bits: int = 32,
        db_path: str = "./tesla_resonance",
    ):
        self.embedder = embedder
        self.resonance_threshold = resonance_threshold
        self.local_models = {
            "resonant": "all-MiniLM-L6-v2",
            "harmonic": "BAAI/bge-small-en-v1.5",
        }
        self.lsh = RandomProjectionLSH(dim=embedder.dim, n_bits=lsh_bits)
        self._memory: Dict[Tuple[int, ...], str] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._contents: Dict[str, str] = {}
        self.table = None
        if HAS_LANCE:
            try:
                self.db = lancedb.connect(db_path)
                try:
                    self.table = self.db.open_table("standing_waves")
                except Exception:
                    self.table = None
            except Exception:
                self.table = None

    def locality_sensitive_hash(self, text: str) -> str:
        """Stable fingerprint of raw text (exact-match fast path)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def detect_resonance(self, text: str, threshold: Optional[float] = None) -> ResonanceResult:
        threshold = self.resonance_threshold if threshold is None else threshold
        text_key = self.locality_sensitive_hash(text)

        # Exact-text standing wave: no embed, no VQ, no SAE — true zero energy.
        if text_key in self._contents:
            return ResonanceResult(
                hit=True,
                content=self._contents[text_key],
                vibration=0.0,
                energy_cost=0.0,
                codes=None,
                source="exact-cache",
                similarity=1.0,
                qstate=None,
            )

        qstate = self.embedder.embed(text)
        code_key = qstate.code_key()
        vec = qstate.reconstructed if qstate.reconstructed is not None else qstate.embedding

        if code_key in self._memory:
            return ResonanceResult(
                hit=True,
                content=self._memory[code_key],
                vibration=float(qstate.vibration),
                energy_cost=0.0,
                codes=qstate.codes,
                source="code-cache",
                similarity=1.0,
                qstate=qstate,
            )

        if self.table is not None and vec is not None:
            try:
                results = self.table.search(vec.tolist()).limit(1).to_list()
                if results and results[0].get("_distance", 1.0) < (1 - threshold):
                    return ResonanceResult(
                        hit=True,
                        content=results[0].get("content"),
                        vibration=float(qstate.vibration),
                        energy_cost=0.0,
                        codes=qstate.codes,
                        source="lancedb",
                        similarity=1.0 - float(results[0].get("_distance", 0.0)),
                        qstate=qstate,
                    )
            except Exception:
                pass

        if vec is not None:
            candidates = self.lsh.neighbors(vec)
            best_sim, best_content = 0.0, None
            search_keys = candidates or list(self._vectors.keys())
            for key in search_keys:
                stored = self._vectors.get(key)
                if stored is None:
                    continue
                sim = float(np.dot(vec, stored) / (
                    (np.linalg.norm(vec) * np.linalg.norm(stored)) + 1e-9
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_content = self._contents.get(key)
            if best_content is not None and best_sim >= threshold:
                return ResonanceResult(
                    hit=True,
                    content=best_content,
                    vibration=float(qstate.vibration),
                    energy_cost=0.0,
                    codes=qstate.codes,
                    source="lsh-cosine",
                    similarity=best_sim,
                    qstate=qstate,
                )

        return ResonanceResult(
            hit=False,
            vibration=float(qstate.vibration),
            energy_cost=float(qstate.original_energy),
            codes=qstate.codes,
            source="miss",
            qstate=qstate,
        )

    def store_standing_wave(
        self,
        text: str,
        content: str,
        qstate: "QuantizedVibrationalState",
    ) -> None:
        code_key = qstate.code_key()
        text_key = self.locality_sensitive_hash(text)
        vec = qstate.reconstructed if qstate.reconstructed is not None else qstate.embedding
        self._memory[code_key] = content
        self._contents[text_key] = content
        if vec is not None:
            unit = np.asarray(vec, dtype=np.float32)
            unit = unit / (np.linalg.norm(unit) + 1e-9)
            self._vectors[text_key] = unit
            self.lsh.add(text_key, unit)

        if self.table is not None and vec is not None:
            try:
                self.table.add(
                    [
                        {
                            "vector": np.asarray(vec).tolist(),
                            "content": content,
                            "codes": qstate.codes.tolist(),
                            "vibration": float(qstate.vibration),
                        }
                    ]
                )
            except Exception:
                pass
