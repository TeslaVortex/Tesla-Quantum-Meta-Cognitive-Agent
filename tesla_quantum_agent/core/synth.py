"""Synthetic data generator: raise vibration under an energy budget."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tesla_quantum_agent.core.embedder import VectorEmbedder
    from tesla_quantum_agent.core.vortex import VortexNoveltyEngine
    from tesla_quantum_agent.backends.llm import LLMBackend


class SyntheticDataGenerator:
    """Generates high-vibration synthetic data under energy constraints."""

    def __init__(
        self,
        embedder: "VectorEmbedder",
        llm_backend: Optional["LLMBackend"] = None,
        energy_budget: float = 50.0,
    ):
        self.embedder = embedder
        self.llm = llm_backend
        self.energy_budget = energy_budget
        self.generated: List[Dict[str, Any]] = []
        self.spent_energy: float = 0.0

    def generate_novel_variants(
        self,
        seed_text: str,
        n: int = 5,
        target_vibration: float = 0.6,
    ) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []
        remaining_energy = self.energy_budget

        polarities = [
            "opposite polarity + cross-domain analogy",
            "temporal inversion of cause and effect",
            "scale shift from quantum to civic",
            "constraint removed, then reconstructed",
            "standing-wave interference of two distant fields",
        ]

        for i in range(n):
            if remaining_energy <= 0:
                break

            if self.llm is not None:
                prompt = (
                    f"Original idea: {seed_text}\n\n"
                    f"Generate a highly novel, non-obvious variant that maximizes "
                    f"conceptual distance while remaining coherent. "
                    f"Target vibration (novelty) ≈ {target_vibration:.2f}. "
                    f"Think in terms of energy, frequency and vibration."
                )
                result = self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=256,
                    temperature=min(1.3, 0.9 + 0.1 * i),
                    enable_thinking=True,
                )
                synthetic = result.get("content") or ""
                usage = result.get("usage") or {}
                energy_cost = float(usage.get("completion_tokens", 100)) / 100.0
            else:
                synthetic = (
                    f"[Synthetic-{i}] Reframed vibration of: {seed_text} → "
                    f"{polarities[i % len(polarities)]}"
                )
                energy_cost = 1.0

            if hasattr(self.embedder, "embed_with_novelty"):
                packed = self.embedder.embed_with_novelty(synthetic, train_sae=True)
                vib_state = packed["qstate"]
                vibration = float(packed["combined_vibration"])
            else:
                vib_state = self.embedder.embed(synthetic, energy_cost=energy_cost)
                vibration = float(vib_state.vibration)

            if vibration >= target_vibration * 0.8:
                item = {
                    "text": synthetic,
                    "vibration": vibration,
                    "energy": float(getattr(vib_state, "original_energy", energy_cost)),
                    "frequency": float(getattr(vib_state, "frequency", 0.0)),
                    "residual_energy": float(getattr(vib_state, "residual_energy", 0.0)),
                    "codes": getattr(vib_state, "codes", None),
                    "embedding": (
                        vib_state.reconstructed.tolist()
                        if getattr(vib_state, "reconstructed", None) is not None
                        else None
                    ),
                }
                variants.append(item)
                remaining_energy -= item["energy"]
                self.spent_energy += item["energy"]
                self.generated.append(item)

        return variants

    def inject_into_vortex(
        self,
        vortex: "VortexNoveltyEngine",
        seed: str,
        n: int = 3,
        target_vibration: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Feed high-vibration synthetics directly into VortexNoveltyEngine."""
        variants = self.generate_novel_variants(
            seed, n=n, target_vibration=target_vibration
        )
        for v in variants:
            vortex.exploration_state.record_exploration(
                path=v["text"],
                score=v["vibration"],
            )
        return variants
