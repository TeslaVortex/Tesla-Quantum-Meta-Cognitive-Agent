"""Standing-wave memory: store interference patterns, not instances."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np

try:
    from sklearn.cluster import HDBSCAN

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if TYPE_CHECKING:
    from tesla_quantum_agent.core.embedder import VectorEmbedder


def _numpy_kmeans(embeddings: np.ndarray, k: int, iters: int = 12, seed: int = 369) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]
    k = max(1, min(k, n))
    centroids = embeddings[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dist = ((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1).astype(np.int32)
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = embeddings[mask].mean(axis=0)
    return labels


class StandingWaveMemory:
    """Dimension 2 – store interference patterns (cluster centroids), not raw instances."""

    def __init__(self, embedder: "VectorEmbedder"):
        self.embedder = embedder
        self.interference_patterns: Dict[int, Dict] = {}
        self.pattern_id = 0
        self._all_texts: List[str] = []

    def add_observation(self, text: str) -> None:
        self._all_texts.append(text)

    def create_standing_waves(self, texts: List[str], min_cluster_size: int = 5) -> int:
        corpus = texts or self._all_texts
        if len(corpus) < max(2, min_cluster_size):
            return len(self.interference_patterns)

        embeddings = np.array([self.embedder.encode(t) for t in corpus], dtype=np.float32)

        if HAS_SKLEARN and len(corpus) >= min_cluster_size:
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
            labels = clusterer.fit_predict(embeddings)
        else:
            k = max(1, len(corpus) // max(min_cluster_size, 2))
            labels = _numpy_kmeans(embeddings, k=k)

        for label in set(labels.tolist()):
            if int(label) == -1:
                continue
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            self.interference_patterns[self.pattern_id] = {
                "centroid": centroid,
                "radius": float(embeddings[mask].std()),
                "cardinality": int(mask.sum()),
                "texts": [corpus[i] for i in np.where(mask)[0][:5]],
            }
            self.pattern_id += 1
        return len(self.interference_patterns)

    def nearest_pattern(self, embedding: np.ndarray) -> Dict:
        if not self.interference_patterns:
            return {}
        vec = np.asarray(embedding, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        best_id, best_sim = None, -1.0
        for pid, pattern in self.interference_patterns.items():
            c = np.asarray(pattern["centroid"], dtype=np.float32)
            c = c / (np.linalg.norm(c) + 1e-9)
            sim = float(np.dot(vec, c))
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        if best_id is None:
            return {}
        out = dict(self.interference_patterns[best_id])
        out["id"] = best_id
        out["similarity"] = best_sim
        return out
