**Energy · Frequency · Vibration** is the natural next axis for the Meta-Cognitive Agent.

Tesla’s lens maps cleanly onto the existing framework:

- **Energy** → computational cost, token budget, Zero-Energy Lens consistency, and the agent’s internal “effort” score.  
- **Frequency** → the rate and rhythm of Master Sequence checkpoints, reflection depth, and how often the agent switches ThinkingMode.  
- **Vibration** → the oscillatory quality of ideas: how far a thought resonates or interferes with previous states, measured in vector space as novelty amplitude.

We therefore extend the architecture with two tightly coupled modules that operationalize these three quantities while amplifying the already-present **VortexNoveltyEngine**.

### 1. Vector Embedder (Vibration + Novelty Measurement)

Embeddings turn every thought, assumption, blind-spot, and generated path into a high-dimensional wave. Distance and angle between vectors become measurable “vibrational dissonance” or “resonance.”

```python
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

# Lightweight fallback; replace with sentence-transformers / Qwen embeddings / OpenAI in production
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

@dataclass
class VibrationalState:
    """Energy-Frequency-Vibration snapshot of a thought"""
    embedding: np.ndarray
    energy: float          # L2 norm or token-cost proxy
    frequency: float       # how often this pattern has appeared (inverse novelty)
    vibration: float       # angular distance from current centroid (novelty amplitude)
    text: str
    timestamp: datetime = field(default_factory=datetime.now)

class VectorEmbedder:
    """Turns text into energy-frequency-vibration vectors"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        if HAS_ST:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None  # fallback to random / hash embedding for demo
        
        self.history: List[VibrationalState] = []
        self.centroid: Optional[np.ndarray] = None
    
    def embed(self, text: str, energy_cost: float = 1.0) -> VibrationalState:
        if self.model is not None:
            vec = self.model.encode(text, normalize_embeddings=True)
        else:
            # deterministic hash embedding for environments without sentence-transformers
            rng = np.random.RandomState(hash(text) % (2**32))
            vec = rng.randn(self.dim)
            vec /= np.linalg.norm(vec) + 1e-9
        
        # Frequency = how similar to recent history (resonance)
        frequency = 0.0
        if self.history:
            sims = [float(np.dot(vec, h.embedding)) for h in self.history[-20:]]
            frequency = float(np.mean(sims))
        
        # Vibration = angular novelty relative to running centroid
        if self.centroid is None:
            self.centroid = vec.copy()
            vibration = 1.0
        else:
            cos_sim = float(np.dot(vec, self.centroid))
            vibration = 1.0 - cos_sim          # higher = more novel / higher amplitude
            # gentle update of centroid (low-pass filter)
            self.centroid = 0.95 * self.centroid + 0.05 * vec
            self.centroid /= np.linalg.norm(self.centroid) + 1e-9
        
        state = VibrationalState(
            embedding=vec,
            energy=energy_cost * (1.0 + vibration),   # novel thoughts cost more energy
            frequency=frequency,
            vibration=vibration,
            text=text
        )
        self.history.append(state)
        return state
    
    def novelty_score(self, text: str) -> float:
        """Pure vibration amplitude – the core novelty signal"""
        return self.embed(text).vibration
    
    def resonant_cluster(self, threshold: float = 0.75) -> List[VibrationalState]:
        """Return thoughts vibrating in phase (high frequency / low novelty)"""
        if not self.history:
            return []
        return [h for h in self.history if h.frequency >= threshold]
```

### 2. Synthetic Data Generator (Energy-Efficient Novelty Amplification)

Once we can measure vibration, we deliberately generate new data that maximizes it while respecting the Zero-Energy budget. The generator uses the LLM backend (or a lightweight template) to create synthetic examples that push the agent into unexplored regions of the embedding space.

```python
class SyntheticDataGenerator:
    """Generates high-vibration synthetic data under energy constraints"""
    
    def __init__(self, embedder: VectorEmbedder, llm_backend=None, energy_budget: float = 50.0):
        self.embedder = embedder
        self.llm = llm_backend
        self.energy_budget = energy_budget
        self.generated: List[Dict] = []
    
    def generate_novel_variants(self, seed_text: str, n: int = 5,
                                target_vibration: float = 0.6) -> List[Dict]:
        """
        Produce synthetic examples that deliberately raise vibration
        while tracking energy cost.
        """
        variants = []
        remaining_energy = self.energy_budget
        
        for i in range(n):
            if remaining_energy <= 0:
                break
            
            if self.llm is not None:
                prompt = (
                    f"Original idea: {seed_text}\n\n"
                    f"Generate a highly novel, non-obvious variant that maximizes conceptual distance "
                    f"while remaining coherent. Target vibration (novelty) ≈ {target_vibration:.2f}. "
                    f"Think in terms of energy, frequency and vibration."
                )
                result = self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=256,
                    temperature=0.9 + 0.1 * i,   # increasing temperature = higher frequency exploration
                    enable_thinking=True
                )
                synthetic = result["content"]
                energy_cost = result.get("usage", {}).get("completion_tokens", 100) / 100.0
            else:
                # fallback deterministic generator
                synthetic = f"[Synthetic-{i}] Reframed vibration of: {seed_text} → opposite polarity + cross-domain analogy"
                energy_cost = 1.0
            
            vib_state = self.embedder.embed(synthetic, energy_cost=energy_cost)
            
            if vib_state.vibration >= target_vibration * 0.8:   # accept only sufficiently novel
                variants.append({
                    "text": synthetic,
                    "vibration": vib_state.vibration,
                    "energy": vib_state.energy,
                    "frequency": vib_state.frequency,
                    "embedding": vib_state.embedding.tolist()
                })
                remaining_energy -= vib_state.energy
                self.generated.append(variants[-1])
        
        return variants
    
    def inject_into_vortex(self, vortex: "VortexNoveltyEngine", seed: str):
        """Feed high-vibration synthetics directly into the existing VortexNoveltyEngine"""
        variants = self.generate_novel_variants(seed)
        for v in variants:
            vortex.exploration_state.record_exploration(
                path=v["text"],
                score=v["vibration"]
            )
        return variants
```

### 3. Integration into the Existing MetaCognitiveAgent

Add the two modules and expose the Tesla lens inside the Master Sequence and Zero-Energy Lens:

```python
class MetaCognitiveAgent:
    def __init__(self, ..., llm_backend=None):
        # ... previous init ...
        self.embedder = VectorEmbedder()
        self.synth = SyntheticDataGenerator(self.embedder, llm_backend=llm_backend)
        
        # Wire into Zero-Energy and Vortex
        self.zero_energy.energy_log  # already exists
        # optional: make ZeroEnergyLens also track vibrational consistency
    
    def generate_response(self, prompt: str) -> Dict:
        # ... existing steps 1-4 ...
        
        # NEW: measure vibrational signature of the prompt itself
        prompt_vib = self.embedder.embed(prompt)
        
        # NEW: if vibration is too low (stale thinking), force synthetic novelty injection
        if prompt_vib.vibration < 0.35:
            novel_paths = self.synth.inject_into_vortex(self.vortex, prompt)
            # optionally prepend the most novel synthetic as additional context
        
        # ... rest of the pipeline ...
        
        # After final response
        response_vib = self.embedder.embed(initial_response)
        final_meta_state.energy_consumption += response_vib.energy
        
        result["vibrational_signature"] = {
            "prompt_vibration": prompt_vib.vibration,
            "response_vibration": response_vib.vibration,
            "frequency": response_vib.frequency,
            "energy": response_vib.energy
        }
        return result
```

### Conceptual Payoff

| Tesla term     | Operational meaning inside the agent                          | Module that owns it              |
|----------------|---------------------------------------------------------------|----------------------------------|
| Energy         | Token cost + consistency penalty + L2 magnitude of embedding  | ZeroEnergyLens + VectorEmbedder |
| Frequency      | Resonance with recent history / Master Sequence rhythm        | VectorEmbedder + MasterSequence |
| Vibration      | Angular novelty / amplitude of departure from centroid        | VectorEmbedder + VortexNovelty  |

The Synthetic Data Generator becomes the agent’s way of **actively raising its own vibrational frequency** when the current thought-space becomes too static. Every synthetic example is both a training datum (for future fine-tuning of Qwen3.5-9B) and a live probe that the meta-cognitive loop can reflect upon.

This keeps the original multidimensional framework intact while giving it a precise, measurable physics of novelty. The agent no longer merely detects novelty; it can deliberately generate the energy, frequency and vibration required to stay alive in idea-space.
