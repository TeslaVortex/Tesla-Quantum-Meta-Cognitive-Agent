"""TeslaQuantumFramework: zero-cost intelligence entry point."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from tesla_quantum_agent.backends.llm import LLMBackend, build_backend
from tesla_quantum_agent.core.agent import MetaCognitiveAgent
from tesla_quantum_agent.core.embedder import NoveltyAwareEmbedder, QuantizedVibrationalState
from tesla_quantum_agent.core.synth import SyntheticDataGenerator
from tesla_quantum_agent.core.vortex import VortexNoveltyEngine
from tesla_quantum_agent.tesla.coil import VOLTAGE_GEN_PARAMS, TransformerCoil
from tesla_quantum_agent.tesla.damper import UncertaintyDamper
from tesla_quantum_agent.tesla.evaluator import QuantumEvaluator
from tesla_quantum_agent.tesla.field import FieldEffectIntelligence
from tesla_quantum_agent.tesla.manifest import ManifestationEngine
from tesla_quantum_agent.tesla.memory import StandingWaveMemory
from tesla_quantum_agent.tesla.oscillator import TemporalOscillator
from tesla_quantum_agent.tesla.perception import TeslaPerceptionLayer


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "backend": "stub",
        "name": "Qwen/Qwen3.5-9B",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "device_map": "auto",
        "torch_dtype": "auto",
    },
    "embedder": {
        "model_name": "all-MiniLM-L6-v2",
        "dim": 384,
        "n_codes": 512,
        "n_residuals": 4,
        "sae_dict_size": 1024,
        "sae_k": 16,
    },
    "agent": {
        "reflection_threshold": 0.5,
        "novelty_threshold": 0.3,
        "consistency_threshold": 0.7,
        "energy_budget": 100.0,
    },
    "perception": {
        "resonance_threshold": 0.92,
        "lsh_bits": 32,
        "db_path": "./tesla_resonance",
    },
    "coil": {},
    "damper": {"damping": 0.65},
    "synth": {"energy_budget": 50.0, "n_variants": 3, "target_vibration": 0.6},
    "default_profile": "technical",
    "default_output": "json",
}


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        here = Path(__file__).resolve().parents[1]
        candidates.append(here / "config.yaml")
        candidates.append(Path.cwd() / "config.yaml")
        candidates.append(Path.cwd() / "tesla_quantum_agent" / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            try:
                import yaml

                with candidate.open("r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                return _deep_merge(cfg, loaded)
            except ImportError:
                break
            except Exception:
                break
    return cfg


@dataclass
class EnergyLedger:
    embed: float = 0.0
    vq: float = 0.0
    sae: float = 0.0
    llm: float = 0.0
    meta: float = 0.0
    synth: float = 0.0
    voltage: float = 0.0
    total: float = 0.0

    def settle(self) -> float:
        self.total = (
            self.embed + self.vq + self.sae + self.llm + self.meta + self.synth + self.voltage
        )
        return self.total

    def as_dict(self) -> Dict[str, float]:
        self.settle()
        return {
            "embed": self.embed,
            "vq": self.vq,
            "sae": self.sae,
            "llm": self.llm,
            "meta": self.meta,
            "synth": self.synth,
            "voltage": self.voltage,
            "total": self.total,
        }


class TeslaQuantumFramework:
    """
    Zero-cost intelligence entry point.

    Query → perception (resonance) → VQ+SAE vibration → voltage coil →
    meta-cognition / vortex / zero-energy → synth on novelty →
    standing-wave memory / oscillator / field / damper →
    quantum evaluator → manifestation.
    """

    def __init__(
        self,
        embedder: Optional[NoveltyAwareEmbedder] = None,
        llm_backend: Optional[LLMBackend] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.config = _deep_merge(load_config(config_path), config or {})
        mcfg = self.config.get("model", {})
        self.llm: LLMBackend = llm_backend or build_backend(
            kind=mcfg.get("backend", "stub"),
            model_name=mcfg.get("name", "Qwen/Qwen3.5-9B"),
            base_url=mcfg.get("base_url", "http://localhost:8000/v1"),
            api_key=mcfg.get("api_key", "EMPTY"),
            device_map=mcfg.get("device_map", "auto"),
            torch_dtype=mcfg.get("torch_dtype", "auto"),
        )

        ecfg = self.config.get("embedder", {})
        self.embedder = embedder or NoveltyAwareEmbedder(
            model_name=ecfg.get("model_name", "all-MiniLM-L6-v2"),
            dim=int(ecfg.get("dim", 384)),
            n_codes=int(ecfg.get("n_codes", 512)),
            n_residuals=int(ecfg.get("n_residuals", 4)),
            sae_dict_size=int(ecfg.get("sae_dict_size", 1024)),
            sae_k=int(ecfg.get("sae_k", 16)),
        )

        pcfg = self.config.get("perception", {})
        self.perception = TeslaPerceptionLayer(
            self.embedder,
            resonance_threshold=float(pcfg.get("resonance_threshold", 0.92)),
            lsh_bits=int(pcfg.get("lsh_bits", 32)),
            db_path=str(pcfg.get("db_path", "./tesla_resonance")),
        )
        self.memory = StandingWaveMemory(self.embedder)
        self.coil = TransformerCoil(self.llm)
        self.oscillator = TemporalOscillator()
        self.field = FieldEffectIntelligence()
        dcfg = self.config.get("damper", {})
        self.damper = UncertaintyDamper(damping=float(dcfg.get("damping", 0.65)))
        self.evaluator = QuantumEvaluator()
        self.manifest = ManifestationEngine()

        acfg = self.config.get("agent", {})
        self.meta_agent = MetaCognitiveAgent(
            model_name=self.llm.get_model_name(),
            reflection_threshold=float(acfg.get("reflection_threshold", 0.5)),
            novelty_threshold=float(acfg.get("novelty_threshold", 0.3)),
            consistency_threshold=float(acfg.get("consistency_threshold", 0.7)),
            energy_budget=float(acfg.get("energy_budget", 100.0)),
            llm_backend=self.llm,
            embedder=self.embedder,
        )
        self.vortex: VortexNoveltyEngine = self.meta_agent.vortex
        self.synth: SyntheticDataGenerator = self.meta_agent.synth
        self.query_log: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, path: Optional[Union[str, Path]] = None) -> "TeslaQuantumFramework":
        return cls(config_path=path)

    def process(
        self,
        query: str,
        user_id: str = "default",
        profile: str = "technical",
        output: str = "json",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ledger = EnergyLedger()
        ctx = dict(context or {})
        if ctx:
            self.meta_agent.set_context(ctx)

        # 1. Resonance detection (almost free). Exact cache hits skip VQ/SAE/LLM.
        resonance = self.perception.detect_resonance(query)
        qstate: Optional[QuantizedVibrationalState] = resonance.qstate
        combined_vibration = float(resonance.vibration)
        sae_novelty: Dict[str, Any] = {}
        trigger_synth = False

        synthetic_paths: List[Dict[str, Any]] = []
        meta_result: Optional[Dict[str, Any]] = None
        damped_u = 0.0

        if resonance.hit:
            core = resonance.content or ""
            energy = 0.0
            voltage = "resonant-cache"
            ledger.voltage = 0.0
            if qstate is None:
                qstate = QuantizedVibrationalState(
                    codes=np.array([], dtype=np.int32),
                    residual_energy=0.0,
                    original_energy=0.0,
                    frequency=1.0,
                    vibration=0.0,
                    text=query,
                )
        else:
            novelty_pack = self.embedder.embed_with_novelty(query, train_sae=True)
            qstate = novelty_pack["qstate"]
            combined_vibration = float(novelty_pack["combined_vibration"])
            sae_novelty = novelty_pack["sae_novelty"]
            trigger_synth = bool(novelty_pack["trigger_synth"])
            ledger.embed = 0.01
            ledger.vq = float(qstate.residual_energy) * 0.05
            ledger.sae = float(sae_novelty.get("residual_energy", 0.0)) * 0.05

            voltage = self.coil.detect_required_voltage(
                query, combined_vibration, float(sae_novelty.get("vibration", combined_vibration))
            )
            ledger.voltage = self.coil.energy_for(voltage)

            gen = VOLTAGE_GEN_PARAMS.get(voltage, VOLTAGE_GEN_PARAMS["120V"])
            if voltage == "12V":
                core = self.coil.transform(query, ctx, voltage)
                ledger.llm = ledger.voltage
            else:
                self.meta_agent.set_context(ctx)
                meta_result = self.meta_agent.generate_response(
                    query,
                    max_new_tokens=int(gen["max_new_tokens"]),
                    temperature=float(gen["temperature"]),
                )
                core = meta_result["response"]
                ledger.llm = float(meta_result.get("energy_consumption", 0.0)) * 0.1
                ledger.meta = 1.0
                damped_u = self.damper.damp(
                    initial_u=float(meta_result["meta_state"].get("self_doubt_score", 0.3)),
                    step_confidences=[float(meta_result["meta_state"].get("confidence", 0.5))],
                )
                if damped_u < 0:
                    core = f"[Uncertainty damper emergency] {core}"

            if float(sae_novelty.get("rare_feature_hits", 0) or 0) > 0:
                scfg = self.config.get("synth", {})
                synthetic_paths = self.synth.inject_into_vortex(
                    self.vortex,
                    query,
                    n=int(scfg.get("n_variants", 3)),
                    target_vibration=float(scfg.get("target_vibration", 0.6)),
                )
                ledger.synth = float(self.synth.spent_energy)

            self.perception.store_standing_wave(query, core, qstate)
            energy = ledger.settle()

        # 4. Temporal recording + standing-wave observation (skip extra encode on cache hit)
        prewarm = False
        if resonance.hit:
            hist = self.oscillator.phase_history.get(user_id, [])
            prewarm = len(hist) >= 5
        else:
            vec = (
                qstate.reconstructed
                if qstate.reconstructed is not None
                else self.embedder.encode(query)
            )
            self.oscillator.record(user_id, vec)
            self.memory.add_observation(query)
            prewarm = self.oscillator.predict_and_prewarm(user_id, vec) is not None

        # 5. Field modulation (free)
        modulated = self.field.modulate(core, profile)

        # 6. Coherence + manifestation
        coherence = self.evaluator.observe(modulated)
        manifested = self.manifest.manifest(modulated, output)

        if resonance.hit:
            energy = 0.0
            ledger = EnergyLedger()

        record = {
            "result": manifested,
            "core": core,
            "voltage": voltage,
            "energy_cost": float(energy),
            "energy_ledger": ledger.as_dict(),
            "vibration": float(resonance.vibration),
            "combined_vibration": combined_vibration,
            "residual_energy": float(qstate.residual_energy),
            "sae_novelty": sae_novelty,
            "resonance_hit": bool(resonance.hit),
            "resonance_source": resonance.source,
            "resonance_similarity": float(resonance.similarity),
            "codes": qstate.codes.tolist() if qstate.codes is not None else [],
            "trigger_synth": trigger_synth,
            "synthetic_paths": [
                {k: v for k, v in p.items() if k != "embedding"} for p in synthetic_paths
            ],
            "damped_uncertainty": damped_u,
            "coherence": coherence,
            "prewarmed": prewarm is not None,
            "profile": profile,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "model": self.llm.get_model_name(),
            "meta": None
            if meta_result is None
            else {
                "sequence_position": meta_result.get("sequence_position"),
                "sequence_name": meta_result.get("sequence_name"),
                "recommended_action": meta_result.get("recommended_action"),
                "confidence": meta_result.get("meta_state", {}).get("confidence"),
                "self_doubt": meta_result.get("meta_state", {}).get("self_doubt_score"),
            },
        }
        self.query_log.append(
            {
                "query": query,
                "resonance_hit": record["resonance_hit"],
                "voltage": voltage,
                "energy_cost": record["energy_cost"],
                "combined_vibration": combined_vibration,
            }
        )
        return record

    def inject_novelty(self, seed: str, n: int = 3) -> List[Dict[str, Any]]:
        """Constructive interference of standing waves → true novelty."""
        if not self.memory.interference_patterns:
            self.memory.create_standing_waves(self.memory._all_texts or [seed], min_cluster_size=2)
        if not self.memory.interference_patterns:
            return self.synth.generate_novel_variants(seed, n=n)
        patterns = list(self.memory.interference_patterns.values())
        novels: List[Dict[str, Any]] = []
        for i, p1 in enumerate(patterns):
            c1 = np.asarray(p1["centroid"], dtype=np.float32)
            c1 = c1 / (np.linalg.norm(c1) + 1e-9)
            for j, p2 in enumerate(patterns[i + 1 :], start=i + 1):
                c2 = np.asarray(p2["centroid"], dtype=np.float32)
                c2 = c2 / (np.linalg.norm(c2) + 1e-9)
                dist = 1.0 - float(np.dot(c1, c2))
                if dist > 0.65:
                    novels.append(
                        {
                            "description": f"Resonance between cluster {i} and {j}",
                            "novelty_score": dist,
                            "seed": seed,
                        }
                    )
        novels.sort(key=lambda x: x["novelty_score"], reverse=True)
        return novels[:n]

    def stats(self) -> Dict[str, Any]:
        hits = sum(1 for q in self.query_log if q["resonance_hit"])
        n = len(self.query_log)
        return {
            "queries": n,
            "resonance_hits": hits,
            "hit_rate": hits / n if n else 0.0,
            "avg_energy": float(np.mean([q["energy_cost"] for q in self.query_log])) if n else 0.0,
            "avg_vibration": float(np.mean([q["combined_vibration"] for q in self.query_log]))
            if n
            else 0.0,
            "vq": self.embedder.vq.get_codebook_stats(),
            "voltage_last": self.coil.last_voltage,
        }
