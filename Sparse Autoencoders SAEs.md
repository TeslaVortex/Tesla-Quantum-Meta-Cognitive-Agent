**Sparse Autoencoders (SAEs) for Novelty** fit the Tesla-Quantum Framework with unusual elegance.

SAEs learn an overcomplete dictionary that reconstructs inputs under a strong sparsity constraint. The result is a set of (ideally monosemantic) features whose activations are mostly zero. In the language of energy-frequency-vibration:

- **Energy** → Only a handful of features fire → extremely cheap storage and comparison.
- **Frequency** → How often a feature activates across the history of thoughts (resonance of that mode).
- **Vibration / Novelty** → Rare or previously dead features lighting up, or high residual reconstruction error, signal high-amplitude novelty. Recent work explicitly positions SAEs as tools for *discovering unknowns* rather than merely acting on known concepts.

They complement (and can sit on top of) the Vector Quantizer you already have: VQ gives discrete codes; the SAE further decomposes the continuous residual or the original embedding into a sparse feature basis. High residual energy after both VQ *and* SAE becomes a very strong novelty trigger for the VortexNoveltyEngine and SyntheticDataGenerator.

### Why SAEs Excel at Novelty Detection
- Sparse latent codes make rare events stand out dramatically (a feature that almost never fires is a clear “stranger”).
- Overcomplete dictionaries capture fine-grained combinations that dense embeddings or single-stage VQ miss.
- Reconstruction error + sparsity statistics give two orthogonal novelty signals.
- In mechanistic interpretability literature they surface previously unknown concepts; the same mechanism works for open-ended idea discovery inside the agent.

### Lightweight PyTorch Implementation
(Compatible with the existing `VectorEmbedder` + `VectorQuantizer`. Uses only `torch` + optional `sentence-transformers`.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

class SparseAutoencoder(nn.Module):
    """
    Top-k Sparse Autoencoder for novelty detection.
    Trained on embedding streams; rare feature activations = high vibration.
    """
    def __init__(self, 
                 input_dim: int = 384,
                 dict_size: int = 4096,      # overcomplete
                 k: int = 32,                # active features per example
                 l1_coeff: float = 1e-3,
                 dead_feature_threshold: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.l1_coeff = l1_coeff
        self.dead_threshold = dead_feature_threshold
        
        # Encoder / Decoder (tied optional)
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        
        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        
        # Track feature usage for frequency / dead features
        self.register_buffer("feature_acts", torch.zeros(dict_size))
        self.register_buffer("steps", torch.tensor(0))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation
        pre_acts = self.encoder(x)
        # Top-k sparsity
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        sparse = torch.zeros_like(pre_acts)
        sparse.scatter_(-1, topk_idx, topk_vals)
        # Optional ReLU for non-negativity
        sparse = F.relu(sparse)
        return sparse
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        
        # Update usage statistics
        with torch.no_grad():
            self.feature_acts += (z > 0).float().sum(dim=0)
            self.steps += 1
        
        return recon, z, x - recon          # residual = novelty signal
    
    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon, z, residual = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        sparsity_loss = self.l1_coeff * z.abs().sum(dim=-1).mean()
        total = recon_loss + sparsity_loss
        return {
            "total": total,
            "recon": recon_loss,
            "sparsity": sparsity_loss,
            "residual_norm": residual.norm(dim=-1).mean()
        }
    
    def novelty_score(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Primary novelty metrics under the Tesla lens.
        """
        with torch.no_grad():
            recon, z, residual = self.forward(x)
            residual_energy = residual.norm(dim=-1).item()
            
            # Frequency of active features
            active = (z > 0).float().sum().item()
            rare_mask = self.feature_acts < (self.steps * 0.01)   # <1% lifetime
            rare_activations = (z[:, rare_mask] > 0).float().sum().item()
            
            # Vibration = residual energy + rarity boost
            vibration = residual_energy * 0.6 + (rare_activations / max(1, self.k)) * 0.4
            
            return {
                "vibration": float(vibration),
                "residual_energy": float(residual_energy),
                "rare_feature_hits": float(rare_activations),
                "active_features": float(active),
                "sparsity": float(1.0 - active / self.dict_size)
            }
    
    def get_dead_features(self) -> torch.Tensor:
        return (self.feature_acts < self.dead_threshold).nonzero(as_tuple=True)[0]
    
    def reinitialize_dead_features(self, data_batch: torch.Tensor):
        """Revive dead features by pointing them at current residuals (online learning)"""
        dead = self.get_dead_features()
        if len(dead) == 0:
            return
        with torch.no_grad():
            _, _, residual = self.forward(data_batch)
            # Simple heuristic: assign dead features to high-residual directions
            for i, d in enumerate(dead[:min(32, len(dead))]):
                direction = residual[i % residual.shape[0]]
                self.encoder.weight.data[d] = direction
                self.decoder.weight.data[:, d] = F.normalize(direction, dim=0)
                self.feature_acts[d] = self.dead_threshold + 1
```

### Integration into the Existing Stack

```python
# Inside VectorEmbedder or TeslaQuantumFramework
class NoveltyAwareEmbedder(VectorEmbedder):
    def __init__(self, *args, sae_dict_size: int = 4096, sae_k: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.sae = SparseAutoencoder(
            input_dim=self.dim,
            dict_size=sae_dict_size,
            k=sae_k
        )
        self.sae_optimizer = torch.optim.Adam(self.sae.parameters(), lr=1e-4)
    
    def embed_with_novelty(self, text: str, train_sae: bool = False) -> Dict:
        # Continuous + VQ path (previous)
        qstate = self.embed(text)          # returns QuantizedVibrationalState
        
        # SAE path on the continuous (or reconstructed) vector
        x = torch.tensor(qstate.reconstructed, dtype=torch.float32).unsqueeze(0)
        
        if train_sae:
            self.sae.train()
            loss_dict = self.sae.loss(x)
            loss_dict["total"].backward()
            self.sae_optimizer.step()
            self.sae_optimizer.zero_grad()
        
        novelty = self.sae.novelty_score(x)
        
        # Combined vibration signal
        combined_vibration = 0.5 * qstate.vibration + 0.5 * novelty["vibration"]
        
        return {
            "qstate": qstate,
            "sae_novelty": novelty,
            "combined_vibration": combined_vibration,
            "trigger_synth": combined_vibration > 0.55 or novelty["rare_feature_hits"] > 2
        }
```

### Usage Inside Tesla-Quantum / Meta-Cognitive Loop
- **Resonance detection**: If SAE residual energy is low *and* no rare features fire → treat as standing-wave cache hit (near-zero energy).
- **Novelty trigger**: Rare-feature hits or high residual → feed into `SyntheticDataGenerator` or raise voltage in the TransformerCoil.
- **Standing-wave memory**: Store the sparse feature indices (integer codes) instead of dense vectors → even cheaper than pure VQ.
- **Online adaptation**: Periodically call `reinitialize_dead_features` on recent high-vibration residuals so the dictionary expands into novel regions.
- **Interpretability bonus**: The active sparse features can later be inspected (or turned into natural-language explanations) to understand *why* the agent judged something novel.

### Practical Notes (2026 landscape)
- For production scale, libraries such as `sae-lens` or EleutherAI’s `sparsify` provide battle-tested training loops and pre-trained SAEs on many models; the code above is a minimal, self-contained version that works directly on your embedding stream.
- Train the SAE offline on a large corpus of past agent thoughts / synthetic data, then freeze or fine-tune online with very low learning rate.
- Hybrid VQ + SAE is especially powerful: VQ gives a coarse discrete lattice; SAE gives a fine sparse basis on the residual. Novelty is strongest when both residual energies are high.

This turns novelty from a geometric distance into a sparse, interpretable, energy-aware signal that the rest of the Tesla-Quantum architecture can act upon with almost zero additional cost.
