**Vector Quantization (VQ)** is the natural discretization step for the Vector Embedder under the energy-frequency-vibration lens.

Continuous high-dimensional embeddings consume energy (memory + distance computations) and produce continuous vibration amplitudes. VQ collapses the continuous field into a finite set of **codebook vectors** — discrete vibrational modes. Each embedding is replaced by the nearest code (or a short residual sequence), dramatically lowering energy cost while preserving enough frequency structure for novelty detection.

### Core Benefits in the Meta-Cognitive Framework
- **Energy**: Codebook indices are integers (or short residual codes) → far cheaper storage and comparison than float32 vectors.
- **Frequency**: The same code appearing repeatedly raises its frequency score (resonance).
- **Vibration / Novelty**: Distance is now measured in codebook space (or residual space). High residual error or rare codes = high vibration.
- Perfect fit for Zero-Energy Lens and VortexNovelty: only high-vibration (rare) codes trigger expensive LLM calls or synthetic generation.

### Implementation (Pure NumPy, drop-in compatible)

```python
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class QuantizedVibrationalState:
    """Discrete vibrational state after Vector Quantization"""
    codes: np.ndarray              # shape (n_quantizers,) or (n_residuals,)
    residual_energy: float         # how much vibration was lost in quantization
    original_energy: float
    frequency: float
    vibration: float               # novelty relative to codebook usage
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    reconstructed: Optional[np.ndarray] = None


class VectorQuantizer:
    """
    Residual Vector Quantization (RVQ) + optional Product Quantization flavor.
    Designed for the energy-frequency-vibration axis.
    """
    
    def __init__(self, 
                 dim: int = 384,
                 n_codes: int = 512,           # codebook size per residual stage
                 n_residuals: int = 4,         # residual depth (higher = better reconstruction, more energy)
                 commitment_cost: float = 0.25,
                 decay: float = 0.99):         # EMA for online codebook learning
        self.dim = dim
        self.n_codes = n_codes
        self.n_residuals = n_residuals
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Codebooks: list of (n_codes, dim) matrices
        self.codebooks = [np.random.randn(n_codes, dim).astype(np.float32) * 0.1 
                          for _ in range(n_residuals)]
        for cb in self.codebooks:
            cb /= np.linalg.norm(cb, axis=1, keepdims=True) + 1e-8
        
        # Usage counters for frequency tracking
        self.code_usage = [np.zeros(n_codes, dtype=np.int64) for _ in range(n_residuals)]
        self.total_steps = 0
    
    def _nearest_code(self, x: np.ndarray, codebook: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """Find nearest code and residual energy"""
        # Cosine or Euclidean; cosine is more vibration-friendly
        sims = codebook @ x
        idx = int(np.argmax(sims))
        code = codebook[idx]
        residual = x - code
        residual_energy = float(np.linalg.norm(residual))
        return idx, residual, residual_energy
    
    def quantize(self, embedding: np.ndarray, 
                 update_codebook: bool = True) -> QuantizedVibrationalState:
        """
        Residual Vector Quantization.
        Returns discrete codes + residual energy (lost vibration).
        """
        x = embedding.astype(np.float32).copy()
        x /= np.linalg.norm(x) + 1e-8
        
        codes = []
        total_residual_energy = 0.0
        reconstructed = np.zeros_like(x)
        
        for stage in range(self.n_residuals):
            idx, residual, res_energy = self._nearest_code(x, self.codebooks[stage])
            codes.append(idx)
            reconstructed += self.codebooks[stage][idx]
            total_residual_energy += res_energy
            x = residual                        # next residual
            
            if update_codebook:
                # Exponential moving average update (online VQ)
                self.codebooks[stage][idx] = (
                    self.decay * self.codebooks[stage][idx] + 
                    (1 - self.decay) * (embedding - reconstructed + self.codebooks[stage][idx])
                )
                self.codebooks[stage][idx] /= np.linalg.norm(self.codebooks[stage][idx]) + 1e-8
                self.code_usage[stage][idx] += 1
        
        self.total_steps += 1
        codes = np.array(codes, dtype=np.int32)
        
        # Frequency = how often these codes have been used (resonance)
        freq = 0.0
        for stage, c in enumerate(codes):
            usage = self.code_usage[stage][c]
            freq += usage / max(1, self.total_steps)
        freq /= self.n_residuals
        
        # Vibration = residual energy + rarity of the code combination
        rarity = 1.0 / (1.0 + freq)
        vibration = total_residual_energy * 0.6 + rarity * 0.4
        
        return QuantizedVibrationalState(
            codes=codes,
            residual_energy=total_residual_energy,
            original_energy=float(np.linalg.norm(embedding)),
            frequency=freq,
            vibration=vibration,
            text="",           # filled by caller
            reconstructed=reconstructed
        )
    
    def encode_batch(self, embeddings: np.ndarray) -> List[QuantizedVibrationalState]:
        return [self.quantize(e, update_codebook=False) for e in embeddings]
    
    def get_codebook_stats(self) -> Dict:
        """Energy & frequency diagnostics"""
        total_usage = sum(u.sum() for u in self.code_usage)
        dead_codes = sum((u == 0).sum() for u in self.code_usage)
        return {
            "total_steps": self.total_steps,
            "total_usage": int(total_usage),
            "dead_codes": int(dead_codes),
            "avg_usage_per_code": float(total_usage / (self.n_codes * self.n_residuals + 1e-8)),
            "codebook_energy": [float(np.linalg.norm(cb)) for cb in self.codebooks]
        }


# ============================================================
# Integration with the existing VectorEmbedder
# ============================================================

class VectorEmbedder:
    """Now with Vector Quantization for discrete vibrational modes"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384,
                 n_codes: int = 512, n_residuals: int = 4):
        self.dim = dim
        # ... keep previous sentence-transformers / fallback logic ...
        if HAS_ST:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        
        self.vq = VectorQuantizer(dim=dim, n_codes=n_codes, n_residuals=n_residuals)
        self.history: List[QuantizedVibrationalState] = []
        self.centroid: Optional[np.ndarray] = None
    
    def embed(self, text: str, energy_cost: float = 1.0) -> QuantizedVibrationalState:
        # 1. Continuous embedding (same as before)
        if self.model is not None:
            vec = self.model.encode(text, normalize_embeddings=True)
        else:
            rng = np.random.RandomState(hash(text) % (2**32))
            vec = rng.randn(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
        
        # 2. Vector Quantization → discrete vibrational state
        qstate = self.vq.quantize(vec, update_codebook=True)
        qstate.text = text
        qstate.original_energy *= energy_cost          # scale by external cost
        
        # 3. Update running centroid in continuous space (for hybrid metrics)
        if self.centroid is None:
            self.centroid = vec.copy()
        else:
            self.centroid = 0.97 * self.centroid + 0.03 * vec
            self.centroid /= np.linalg.norm(self.centroid) + 1e-9
        
        self.history.append(qstate)
        return qstate
    
    def novelty_score(self, text: str) -> float:
        """Vibration after quantization — the discrete novelty signal"""
        return self.embed(text).vibration
    
    def most_novel_codes(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return the rarest (highest vibration potential) codes"""
        rarity = []
        for stage in range(self.vq.n_residuals):
            usage = self.vq.code_usage[stage]
            for idx, u in enumerate(usage):
                rarity.append((stage * self.vq.n_codes + idx, 1.0 / (1.0 + u)))
        rarity.sort(key=lambda x: x[1], reverse=True)
        return rarity[:top_k]
```

### How it plugs into the rest of the agent

```python
# Inside MetaCognitiveAgent.__init__
self.embedder = VectorEmbedder(n_codes=1024, n_residuals=6)   # richer discrete space

# Inside generate_response or VortexNovelty
qstate = self.embedder.embed(prompt)
if qstate.vibration > 0.55 or qstate.residual_energy > 0.4:
    # High residual = the continuous thought could not be well captured by existing modes
    # → trigger synthetic generation or deeper Master Sequence reflection
    novel_variants = self.synth.generate_novel_variants(prompt, target_vibration=0.7)

# Zero-Energy Lens can now operate on integer codes
# Hamming distance between code sequences is extremely cheap
```

### Design Choices & Trade-offs
- **Residual VQ** (instead of single-stage) keeps reconstruction fidelity high while still giving discrete codes.  
- Online EMA codebook learning means the quantizer adapts to the agent’s own thought distribution over time (self-tuning vibrational modes).  
- Dead-code monitoring lets you periodically re-initialize unused codes → prevents collapse of the frequency spectrum.  
- Residual energy is the perfect “lost vibration” signal: high residual means the current codebook cannot yet express this idea, so the agent should explore or generate synthetics.

This turns the continuous embedding space into a discrete lattice of vibrational modes while preserving the energy accounting and novelty detection already present in the framework. The agent now thinks in quantized energy packets, frequency of mode activation, and residual vibration — exactly the Tesla lens applied to vector space.
