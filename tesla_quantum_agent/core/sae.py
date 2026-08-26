"""Sparse Autoencoder for novelty: rare features + residual energy."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


ArrayLike = Union[np.ndarray, "torch.Tensor"]


class _NumpySparseAutoencoder:
    """Top-k SAE implemented in NumPy so novelty detection runs without PyTorch."""

    def __init__(
        self,
        input_dim: int = 384,
        dict_size: int = 4096,
        k: int = 32,
        l1_coeff: float = 1e-3,
        dead_feature_threshold: int = 1000,
        seed: int = 369,
    ):
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = min(k, dict_size)
        self.l1_coeff = l1_coeff
        self.dead_threshold = dead_feature_threshold

        rng = np.random.RandomState(seed)
        self.W_enc = rng.randn(dict_size, input_dim).astype(np.float32) * 0.02
        self.b_enc = np.zeros(dict_size, dtype=np.float32)
        self.W_dec = rng.randn(input_dim, dict_size).astype(np.float32) * 0.02
        self.W_dec /= np.linalg.norm(self.W_dec, axis=0, keepdims=True) + 1e-8
        self.feature_acts = np.zeros(dict_size, dtype=np.float32)
        self.steps = 0

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        pre = x @ self.W_enc.T + self.b_enc
        k = min(self.k, pre.shape[-1])
        idx = np.argpartition(pre, -k, axis=-1)[:, -k:]
        sparse = np.zeros_like(pre)
        rows = np.arange(pre.shape[0])[:, None]
        sparse[rows, idx] = pre[rows, idx]
        return np.maximum(sparse, 0.0)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.W_dec.T

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        z = self.encode(x)
        recon = self.decode(z)
        self.feature_acts += (z > 0).sum(axis=0)
        self.steps += 1
        return recon, z, x - recon

    def online_step(self, x: np.ndarray, lr: float = 1e-3) -> Dict[str, float]:
        recon, z, residual = self.forward(x)
        x2 = np.asarray(x, dtype=np.float32)
        if x2.ndim == 1:
            x2 = x2[None, :]
        z_mean = z.mean(axis=0)
        res_mean = residual.mean(axis=0)
        self.W_dec += lr * np.outer(res_mean, z_mean)
        col_norm = np.linalg.norm(self.W_dec, axis=0, keepdims=True) + 1e-8
        self.W_dec /= col_norm
        self.W_enc += lr * np.outer(z_mean, res_mean)
        self.b_enc += lr * z_mean
        recon_loss = float(np.mean(residual ** 2))
        return {
            "total": recon_loss + self.l1_coeff * float(np.mean(np.abs(z))),
            "recon": recon_loss,
            "sparsity": float(np.mean(np.abs(z))),
            "residual_norm": float(np.mean(np.linalg.norm(residual, axis=-1))),
        }

    def loss(self, x: ArrayLike) -> Dict[str, float]:
        x_np = x.detach().cpu().numpy() if HAS_TORCH and hasattr(x, "detach") else np.asarray(x)
        return self.online_step(x_np, lr=0.0)

    def novelty_score(self, x: ArrayLike) -> Dict[str, float]:
        x_np = x.detach().cpu().numpy() if HAS_TORCH and hasattr(x, "detach") else np.asarray(x)
        if x_np.ndim == 1:
            x_np = x_np[None, :]
        recon, z, residual = self.forward(x_np)
        residual_energy = float(np.linalg.norm(residual, axis=-1).mean())
        active = float((z > 0).sum())
        if self.steps > 0:
            rare_mask = self.feature_acts < (self.steps * 0.01)
        else:
            rare_mask = np.ones(self.dict_size, dtype=bool)
        rare_activations = float((z[:, rare_mask] > 0).sum())
        vibration = min(
            1.0,
            residual_energy * 0.6 + (rare_activations / max(1, self.k)) * 0.4,
        )
        return {
            "vibration": float(vibration),
            "residual_energy": float(residual_energy),
            "rare_feature_hits": float(rare_activations),
            "active_features": float(active),
            "sparsity": float(1.0 - active / max(1, self.dict_size)),
        }

    def get_dead_features(self) -> np.ndarray:
        return np.where(self.feature_acts < self.dead_threshold)[0]

    def reinitialize_dead_features(self, data_batch: np.ndarray) -> None:
        dead = self.get_dead_features()
        if dead.size == 0:
            return
        recon, z, residual = self.forward(data_batch)
        for i, d in enumerate(dead[: min(32, dead.size)]):
            direction = residual[i % residual.shape[0]]
            self.W_enc[d] = direction
            norm = float(np.linalg.norm(direction)) + 1e-8
            self.W_dec[:, d] = direction / norm
            self.feature_acts[d] = self.dead_threshold + 1


if HAS_TORCH:

    class SparseAutoencoder(nn.Module):
        """Top-k Sparse Autoencoder for novelty detection (PyTorch)."""

        def __init__(
            self,
            input_dim: int = 384,
            dict_size: int = 4096,
            k: int = 32,
            l1_coeff: float = 1e-3,
            dead_feature_threshold: int = 1000,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.dict_size = dict_size
            self.k = min(k, dict_size)
            self.l1_coeff = l1_coeff
            self.dead_threshold = dead_feature_threshold

            self.encoder = nn.Linear(input_dim, dict_size, bias=True)
            self.decoder = nn.Linear(dict_size, input_dim, bias=False)
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

            self.register_buffer("feature_acts", torch.zeros(dict_size))
            self.register_buffer("steps", torch.tensor(0))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            pre_acts = self.encoder(x)
            topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
            sparse = torch.zeros_like(pre_acts)
            sparse.scatter_(-1, topk_idx, topk_vals)
            return F.relu(sparse)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            z = self.encode(x)
            recon = self.decode(z)
            with torch.no_grad():
                self.feature_acts += (z > 0).float().sum(dim=0)
                self.steps += 1
            return recon, z, x - recon

        def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            recon, z, residual = self.forward(x)
            recon_loss = F.mse_loss(recon, x)
            sparsity_loss = self.l1_coeff * z.abs().sum(dim=-1).mean()
            total = recon_loss + sparsity_loss
            return {
                "total": total,
                "recon": recon_loss,
                "sparsity": sparsity_loss,
                "residual_norm": residual.norm(dim=-1).mean(),
            }

        def novelty_score(self, x: ArrayLike) -> Dict[str, float]:
            if not torch.is_tensor(x):
                x = torch.tensor(np.asarray(x), dtype=torch.float32)
            with torch.no_grad():
                recon, z, residual = self.forward(x)
                residual_energy = residual.norm(dim=-1).mean().item()
                active = (z > 0).float().sum().item()
                rare_mask = self.feature_acts < (self.steps * 0.01)
                rare_activations = (z[:, rare_mask] > 0).float().sum().item()
                vibration = min(
                    1.0,
                    residual_energy * 0.6 + (rare_activations / max(1, self.k)) * 0.4,
                )
                return {
                    "vibration": float(vibration),
                    "residual_energy": float(residual_energy),
                    "rare_feature_hits": float(rare_activations),
                    "active_features": float(active),
                    "sparsity": float(1.0 - active / self.dict_size),
                }

        def get_dead_features(self) -> torch.Tensor:
            return (self.feature_acts < self.dead_threshold).nonzero(as_tuple=True)[0]

        def reinitialize_dead_features(self, data_batch: torch.Tensor) -> None:
            dead = self.get_dead_features()
            if len(dead) == 0:
                return
            with torch.no_grad():
                _, _, residual = self.forward(data_batch)
                for i, d in enumerate(dead[: min(32, len(dead))]):
                    direction = residual[i % residual.shape[0]]
                    self.encoder.weight.data[d] = direction
                    self.decoder.weight.data[:, d] = F.normalize(direction, dim=0)
                    self.feature_acts[d] = self.dead_threshold + 1

        def online_step(self, x: ArrayLike, lr: float = 1e-3) -> Dict[str, float]:
            if not torch.is_tensor(x):
                x = torch.tensor(np.asarray(x), dtype=torch.float32)
            loss_dict = self.loss(x)
            return {k: float(v.detach()) for k, v in loss_dict.items()}

else:
    SparseAutoencoder = _NumpySparseAutoencoder  # type: ignore[misc, assignment]


def create_sae(
    input_dim: int = 384,
    dict_size: int = 4096,
    k: int = 32,
    l1_coeff: float = 1e-3,
    dead_feature_threshold: int = 1000,
    prefer_torch: bool = True,
) -> Any:
    """Factory: PyTorch SAE when available, otherwise NumPy."""
    if prefer_torch and HAS_TORCH:
        return SparseAutoencoder(
            input_dim=input_dim,
            dict_size=dict_size,
            k=k,
            l1_coeff=l1_coeff,
            dead_feature_threshold=dead_feature_threshold,
        )
    return _NumpySparseAutoencoder(
        input_dim=input_dim,
        dict_size=dict_size,
        k=k,
        l1_coeff=l1_coeff,
        dead_feature_threshold=dead_feature_threshold,
    )
