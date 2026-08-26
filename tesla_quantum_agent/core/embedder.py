"""Vector embedder + residual vector quantization (discrete vibrational modes)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False


def _token_hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic feature-hashed bag-of-tokens embedding (semantic-ish fallback)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i in range(3):
            idx = int.from_bytes(digest[i * 4 : (i + 1) * 4], "little") % dim
            sign = 1.0 if digest[12 + i] % 2 == 0 else -1.0
            vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-9)


@dataclass
class VibrationalState:
    """Energy-frequency-vibration snapshot of a continuous thought."""

    embedding: np.ndarray
    energy: float
    frequency: float
    vibration: float
    text: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QuantizedVibrationalState:
    """Discrete vibrational state after residual vector quantization."""

    codes: np.ndarray
    residual_energy: float
    original_energy: float
    frequency: float
    vibration: float
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    reconstructed: Optional[np.ndarray] = None
    continuous_vibration: float = 0.0
    embedding: Optional[np.ndarray] = None

    def code_key(self) -> Tuple[int, ...]:
        return tuple(int(c) for c in self.codes.tolist())


class VectorQuantizer:
    """
    Residual Vector Quantization (RVQ) with online EMA codebook learning.

    Energy  → integer codes instead of float32 vectors.
    Frequency → codebook usage / resonance.
    Vibration → residual energy + rarity of the code combination.
    """

    def __init__(
        self,
        dim: int = 384,
        n_codes: int = 512,
        n_residuals: int = 4,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        seed: int = 369,
    ):
        self.dim = dim
        self.n_codes = n_codes
        self.n_residuals = n_residuals
        self.commitment_cost = commitment_cost
        self.decay = decay

        rng = np.random.RandomState(seed)
        self.codebooks = [
            rng.randn(n_codes, dim).astype(np.float32) * 0.1 for _ in range(n_residuals)
        ]
        for cb in self.codebooks:
            cb /= np.linalg.norm(cb, axis=1, keepdims=True) + 1e-8

        self.code_usage = [np.zeros(n_codes, dtype=np.int64) for _ in range(n_residuals)]
        self.total_steps = 0

    def _nearest_code(
        self, x: np.ndarray, codebook: np.ndarray
    ) -> Tuple[int, np.ndarray, float]:
        sims = codebook @ x
        idx = int(np.argmax(sims))
        code = codebook[idx]
        residual = x - code
        residual_energy = float(np.linalg.norm(residual))
        return idx, residual, residual_energy

    def quantize(
        self,
        embedding: np.ndarray,
        update_codebook: bool = True,
    ) -> QuantizedVibrationalState:
        x = embedding.astype(np.float32).copy()
        x /= np.linalg.norm(x) + 1e-8
        original = x.copy()

        codes: List[int] = []
        total_residual_energy = 0.0
        reconstructed = np.zeros_like(x)

        for stage in range(self.n_residuals):
            idx, residual, res_energy = self._nearest_code(x, self.codebooks[stage])
            codes.append(idx)
            reconstructed += self.codebooks[stage][idx]
            total_residual_energy += res_energy
            x = residual

            if update_codebook:
                self.codebooks[stage][idx] = (
                    self.decay * self.codebooks[stage][idx]
                    + (1 - self.decay)
                    * (original - reconstructed + self.codebooks[stage][idx])
                )
                self.codebooks[stage][idx] /= (
                    np.linalg.norm(self.codebooks[stage][idx]) + 1e-8
                )
                self.code_usage[stage][idx] += 1

        self.total_steps += 1
        code_arr = np.array(codes, dtype=np.int32)

        freq = 0.0
        for stage, c in enumerate(codes):
            usage = self.code_usage[stage][c]
            freq += usage / max(1, self.total_steps)
        freq /= self.n_residuals

        rarity = 1.0 / (1.0 + freq)
        # Per-stage residual of unit vectors is in ~[0, 2]; normalize to [0, 1].
        norm_residual = min(1.0, total_residual_energy / max(1.0, 2.0 * self.n_residuals))
        vibration = float(min(1.0, norm_residual * 0.6 + rarity * 0.4))

        return QuantizedVibrationalState(
            codes=code_arr,
            residual_energy=float(total_residual_energy),
            original_energy=float(np.linalg.norm(embedding)),
            frequency=float(freq),
            vibration=float(vibration),
            text="",
            reconstructed=reconstructed,
            embedding=embedding.astype(np.float32),
        )

    def encode_batch(self, embeddings: np.ndarray) -> List[QuantizedVibrationalState]:
        return [self.quantize(e, update_codebook=False) for e in embeddings]

    def reinitialize_dead_codes(self, min_usage: int = 1) -> int:
        """Revive unused codes by jittering around live centroids."""
        revived = 0
        rng = np.random.RandomState(self.total_steps + 17)
        for stage, usage in enumerate(self.code_usage):
            live = np.where(usage >= min_usage)[0]
            dead = np.where(usage < min_usage)[0]
            if live.size == 0 or dead.size == 0:
                continue
            for d in dead:
                src = int(live[rng.randint(0, live.size)])
                noise = rng.randn(self.dim).astype(np.float32) * 0.05
                self.codebooks[stage][d] = self.codebooks[stage][src] + noise
                self.codebooks[stage][d] /= (
                    np.linalg.norm(self.codebooks[stage][d]) + 1e-8
                )
                self.code_usage[stage][d] = 1
                revived += 1
        return revived

    def get_codebook_stats(self) -> Dict[str, Any]:
        total_usage = sum(int(u.sum()) for u in self.code_usage)
        dead_codes = sum(int((u == 0).sum()) for u in self.code_usage)
        return {
            "total_steps": self.total_steps,
            "total_usage": int(total_usage),
            "dead_codes": int(dead_codes),
            "avg_usage_per_code": float(
                total_usage / (self.n_codes * self.n_residuals + 1e-8)
            ),
            "codebook_energy": [float(np.linalg.norm(cb)) for cb in self.codebooks],
        }


class VectorEmbedder:
    """
    Continuous embedding → energy/frequency/vibration → residual VQ.

    Uses sentence-transformers when available; otherwise a deterministic
    feature-hashed token embedding so resonance is stable across runs.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        n_codes: int = 512,
        n_residuals: int = 4,
    ):
        self.model_name = model_name
        self.dim = dim
        self.model = None
        if HAS_ST:
            try:
                self.model = SentenceTransformer(model_name)
                self.dim = int(self.model.get_sentence_embedding_dimension())
            except Exception:
                self.model = None

        self.vq = VectorQuantizer(dim=self.dim, n_codes=n_codes, n_residuals=n_residuals)
        self.history: List[QuantizedVibrationalState] = []
        self.continuous_history: List[VibrationalState] = []
        self.centroid: Optional[np.ndarray] = None

    def encode(self, text: str) -> np.ndarray:
        if self.model is not None:
            vec = self.model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return _token_hash_embed(text, self.dim)

    def embed_continuous(self, text: str, energy_cost: float = 1.0) -> VibrationalState:
        vec = self.encode(text)
        frequency = 0.0
        if self.continuous_history:
            sims = [
                float(np.dot(vec, h.embedding)) for h in self.continuous_history[-20:]
            ]
            frequency = float(np.mean(sims))

        if self.centroid is None:
            self.centroid = vec.copy()
            vibration = 1.0
        else:
            cos_sim = float(np.dot(vec, self.centroid))
            vibration = 1.0 - cos_sim
            self.centroid = 0.95 * self.centroid + 0.05 * vec
            self.centroid /= np.linalg.norm(self.centroid) + 1e-9

        state = VibrationalState(
            embedding=vec,
            energy=float(energy_cost * (1.0 + vibration)),
            frequency=float(frequency),
            vibration=float(max(0.0, vibration)),
            text=text,
        )
        self.continuous_history.append(state)
        return state

    def embed(self, text: str, energy_cost: float = 1.0) -> QuantizedVibrationalState:
        continuous = self.embed_continuous(text, energy_cost=energy_cost)
        qstate = self.vq.quantize(continuous.embedding, update_codebook=True)
        qstate.text = text
        qstate.original_energy = float(qstate.original_energy * energy_cost)
        qstate.continuous_vibration = continuous.vibration
        qstate.embedding = continuous.embedding
        self.history.append(qstate)
        return qstate

    def novelty_score(self, text: str) -> float:
        """Vibration after quantization — the discrete novelty signal."""
        return float(self.embed(text).vibration)

    def resonant_cluster(self, threshold: float = 0.75) -> List[VibrationalState]:
        if not self.continuous_history:
            return []
        return [h for h in self.continuous_history if h.frequency >= threshold]

    def most_novel_codes(self, top_k: int = 10) -> List[Tuple[int, float]]:
        rarity: List[Tuple[int, float]] = []
        for stage in range(self.vq.n_residuals):
            usage = self.vq.code_usage[stage]
            for idx, u in enumerate(usage):
                rarity.append((stage * self.vq.n_codes + idx, 1.0 / (1.0 + float(u))))
        rarity.sort(key=lambda x: x[1], reverse=True)
        return rarity[:top_k]


class NoveltyAwareEmbedder(VectorEmbedder):
    """VectorEmbedder + Sparse Autoencoder → combined vibration / novelty."""

    def __init__(
        self,
        *args: Any,
        sae_dict_size: int = 4096,
        sae_k: int = 32,
        train_sae: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        from tesla_quantum_agent.core.sae import create_sae

        self.sae = create_sae(
            input_dim=self.dim,
            dict_size=sae_dict_size,
            k=sae_k,
        )
        self.train_sae = train_sae
        self._sae_optimizer = None
        try:
            import torch

            if hasattr(self.sae, "parameters"):
                self._sae_optimizer = torch.optim.Adam(self.sae.parameters(), lr=1e-4)
        except Exception:
            self._sae_optimizer = None

    def embed_with_novelty(self, text: str, train_sae: Optional[bool] = None) -> Dict[str, Any]:
        qstate = self.embed(text)
        reconstructed = qstate.reconstructed if qstate.reconstructed is not None else qstate.embedding
        x = np.asarray(reconstructed, dtype=np.float32).reshape(1, -1)

        do_train = self.train_sae if train_sae is None else train_sae
        if do_train:
            self._online_sae_step(x)

        novelty = self.sae.novelty_score(x)
        combined_vibration = float(
            min(1.0, 0.5 * float(qstate.vibration) + 0.5 * float(novelty["vibration"]))
        )
        trigger_synth = combined_vibration > 0.55 or novelty.get("rare_feature_hits", 0) > 2

        return {
            "qstate": qstate,
            "sae_novelty": novelty,
            "combined_vibration": float(combined_vibration),
            "trigger_synth": bool(trigger_synth),
        }

    def _online_sae_step(self, x: np.ndarray) -> None:
        if self._sae_optimizer is not None:
            try:
                import torch

                self.sae.train()
                xt = torch.tensor(x, dtype=torch.float32)
                loss_dict = self.sae.loss(xt)
                self._sae_optimizer.zero_grad()
                loss_dict["total"].backward()
                self._sae_optimizer.step()
                return
            except Exception:
                pass
        if hasattr(self.sae, "online_step"):
            self.sae.online_step(x)
