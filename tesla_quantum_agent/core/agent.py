"""MetaCognitiveAgent: Master Sequence + Vortex + Zero-Energy + LLM + vibration."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from tesla_quantum_agent.backends.llm import LLMBackend, StubBackend
from tesla_quantum_agent.core.embedder import NoveltyAwareEmbedder, VectorEmbedder
from tesla_quantum_agent.core.meta_engine import (
    ActionDecision,
    MetaThinkingEngine,
    MetaThinkingState,
    ThinkingMode,
)
from tesla_quantum_agent.core.sequence import MasterSequencePath, SequenceCheckpoint
from tesla_quantum_agent.core.synth import SyntheticDataGenerator
from tesla_quantum_agent.core.vortex import VortexNoveltyEngine
from tesla_quantum_agent.core.zero_energy import ZeroEnergyLens


class MetaCognitiveAgent:
    """Complete meta-cognitive agent with all integrated components."""

    def __init__(
        self,
        model_name: str = "Qwen3.5-9B",
        reflection_threshold: float = 0.5,
        novelty_threshold: float = 0.3,
        consistency_threshold: float = 0.7,
        energy_budget: float = 100.0,
        llm_backend: Optional[LLMBackend] = None,
        embedder: Optional[VectorEmbedder] = None,
    ):
        self.llm = llm_backend or StubBackend()
        self.model_name = self.llm.get_model_name() or model_name

        self.meta_engine = MetaThinkingEngine(
            model_name=self.model_name,
            reflection_threshold=reflection_threshold,
            energy_budget=energy_budget,
        )
        self.sequence = MasterSequencePath(self.meta_engine)
        self.vortex = VortexNoveltyEngine(novelty_threshold)
        self.zero_energy = ZeroEnergyLens(consistency_threshold)
        self.embedder = embedder or NoveltyAwareEmbedder()
        self.synth = SyntheticDataGenerator(
            self.embedder, llm_backend=self.llm, energy_budget=min(50.0, energy_budget)
        )

        self.context: Dict[str, Any] = {}
        self.current_response = ""
        self.thinking_mode = ThinkingMode.ANALYTICAL
        self.session_id = datetime.now().isoformat()
        self.response_history: List[Dict[str, Any]] = []
        self._last_finish_reason: Optional[str] = None
        self._gen_max_new_tokens: int = 2048
        self._gen_temperature: Optional[float] = None

    def set_context(self, context: Dict[str, Any]) -> None:
        self.context = dict(context or {})

    def _build_system_prompt(self) -> str:
        return (
            f"You are a meta-cognitive AI agent powered by {self.model_name}. "
            f"Current thinking mode: {self.thinking_mode.value}. "
            "Always reason carefully, surface assumptions, and note potential blind spots. "
            "Use the Master Sequence and multi-dimensional thinking when appropriate."
        )

    def _account_tokens(self, usage: Optional[Dict[str, Any]]) -> float:
        if not usage:
            return 1.0
        tokens = usage.get("total_tokens") or (
            int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        )
        return float(tokens) / 100.0

    def _generate_initial_response(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        if self._gen_temperature is not None:
            temperature = float(self._gen_temperature)
        else:
            temperature = 0.9 if self.thinking_mode.value == "creative" else 0.7
        result = self.llm.generate(
            messages,
            max_new_tokens=int(self._gen_max_new_tokens),
            temperature=temperature,
            enable_thinking=True,
        )
        self._last_finish_reason = result.get("finish_reason")
        if result.get("thinking"):
            self.meta_engine.meta_state.add_meta_question(
                f"Model internal reasoning: {str(result['thinking'])[:300]}..."
            )
        self.meta_engine.meta_state.update_energy(self._account_tokens(result.get("usage")))
        return result.get("content") or f"[Initial Response to: {prompt[:50]}...]"

    def _generate_alternative_response(self, prompt: str, assumptions: List[str]) -> str:
        assumption_text = "\n".join(f"- {a}" for a in assumptions[:5])
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Original prompt: {prompt}\n\n"
                    f"Previously flagged assumptions:\n{assumption_text}\n\n"
                    "Generate a reframed, higher-quality response that explicitly "
                    "addresses or challenges these assumptions."
                ),
            },
        ]
        result = self.llm.generate(
            messages,
            max_new_tokens=int(self._gen_max_new_tokens),
            enable_thinking=True,
            temperature=0.6,
        )
        self._last_finish_reason = result.get("finish_reason")
        self.meta_engine.meta_state.update_energy(self._account_tokens(result.get("usage")))
        return result.get("content") or (
            f"[Alternative Response]\nPrompt: {prompt[:50]}...\n"
            f"Assumptions addressed: {', '.join(assumptions[:3])}"
        )

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a response with full meta-cognitive + vibrational processing."""
        start_time = datetime.now()
        self.meta_engine.reset()
        self.vortex.exploration_state.exploration_count = 0
        self.zero_energy.belief_history = []
        self._last_finish_reason = None
        if max_new_tokens is not None:
            self._gen_max_new_tokens = int(max_new_tokens)
        if temperature is not None:
            self._gen_temperature = float(temperature)

        if hasattr(self.embedder, "embed_with_novelty"):
            prompt_pack = self.embedder.embed_with_novelty(prompt)
            prompt_vib = prompt_pack["qstate"]
            combined = float(prompt_pack["combined_vibration"])
            trigger_synth = bool(prompt_pack["trigger_synth"])
            sae_novelty = prompt_pack["sae_novelty"]
        else:
            prompt_vib = self.embedder.embed(prompt)
            combined = float(prompt_vib.vibration)
            trigger_synth = combined > 0.55 or float(prompt_vib.residual_energy) > 0.4
            sae_novelty = {}

        novel_paths: List[Dict[str, Any]] = []
        if combined < 0.35 or trigger_synth:
            novel_paths = self.synth.inject_into_vortex(self.vortex, prompt, n=3)

        initial_response = self._generate_initial_response(prompt)

        sequence_num = self.sequence.get_next_position()
        checkpoint = self.sequence.run_meta_checkpoint(
            sequence_num, initial_response, self.context
        )

        exploration_paths = self.vortex.generate_exploration_paths(prompt, self.context)
        is_consistent, consistency_score = self.zero_energy.check_belief_consistency(
            {}, initial_response
        )
        self.zero_energy.log_belief(initial_response[:200])

        action = checkpoint.recommended_action
        if action == ActionDecision.RETHINK:
            initial_response = self._generate_alternative_response(
                prompt, checkpoint.meta_state.assumptions_flagged
            )
        elif action == ActionDecision.SEEK_FEEDBACK:
            spots = ", ".join(checkpoint.meta_state.blind_spots_detected[:3])
            initial_response = f"[Requires verification] {initial_response}\nBlind spots: {spots}"
        elif action == ActionDecision.REFINED:
            truncated = (self._last_finish_reason or "") == "length"
            empty = not (initial_response or "").strip()
            if empty or truncated:
                qs = ", ".join(checkpoint.meta_state.meta_questions[:2])
                initial_response += f"\n\n[Refinement needed] {qs}"

        final_meta_state = self.meta_engine.run_meta_reflection(initial_response, self.context)

        if hasattr(self.embedder, "embed_with_novelty"):
            resp_pack = self.embedder.embed_with_novelty(initial_response)
            response_vib = resp_pack["qstate"]
            resp_combined = float(resp_pack["combined_vibration"])
        else:
            response_vib = self.embedder.embed(initial_response)
            resp_combined = float(response_vib.vibration)

        final_meta_state.energy_consumption += float(response_vib.original_energy) + 5.0
        self._log_response(initial_response, checkpoint, final_meta_state)

        return {
            "session_id": self.session_id,
            "timestamp": start_time.isoformat(),
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response": initial_response,
            "meta_state": final_meta_state.to_dict(),
            "sequence_position": sequence_num,
            "sequence_name": self.sequence.get_current_name(),
            "checkpoint": checkpoint.to_dict(),
            "exploration_state": self.vortex.get_state(),
            "exploration_paths": exploration_paths,
            "energy_consumption": final_meta_state.energy_consumption,
            "consistency_score": consistency_score,
            "is_consistent": is_consistent,
            "recommended_action": action.value,
            "synthetic_paths": novel_paths,
            "vibrational_signature": {
                "prompt_vibration": float(prompt_vib.vibration),
                "response_vibration": float(response_vib.vibration),
                "combined_vibration": resp_combined,
                "prompt_combined_vibration": combined,
                "frequency": float(response_vib.frequency),
                "energy": float(response_vib.original_energy),
                "residual_energy": float(response_vib.residual_energy),
                "sae_novelty": sae_novelty,
                "trigger_synth": trigger_synth,
                "codes": prompt_vib.codes.tolist() if prompt_vib.codes is not None else [],
            },
        }

    def _log_response(
        self,
        response: str,
        checkpoint: SequenceCheckpoint,
        meta_state: MetaThinkingState,
    ) -> None:
        self.response_history.append(
            {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "response_length": len(response),
                "confidence": meta_state.confidence,
                "self_doubt": meta_state.self_doubt_score,
                "sequence_num": checkpoint.sequence_num,
                "action": checkpoint.recommended_action.value,
                "energy_consumed": meta_state.energy_consumption,
            }
        )

    def get_session_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_responses": len(self.response_history),
            "meta_engine_state": self.meta_engine.get_summary(),
            "sequence_info": {
                "current_position": self.sequence.position,
                "current_name": self.sequence.get_current_name(),
                "checkpoint_count": len(self.sequence.checkpoint_history),
            },
            "vortex_state": self.vortex.get_state(),
            "zero_energy_summary": self.zero_energy.get_energy_summary(),
            "response_history": self.response_history[-10:],
            "overall_metrics": self._calculate_overall_metrics(),
        }

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        if not self.response_history:
            return {
                "avg_confidence": 0.0,
                "avg_self_doubt": 0.0,
                "avg_energy": 0.0,
                "total_actions": {},
                "avg_response_length": 0.0,
                "success_rate": 0.0,
                "total_responses": 0,
            }
        confidences = [r["confidence"] for r in self.response_history]
        doubts = [r["self_doubt"] for r in self.response_history]
        energies = [r["energy_consumed"] for r in self.response_history]
        lengths = [r["response_length"] for r in self.response_history]
        actions = [r["action"] for r in self.response_history]
        action_counts: Dict[str, int] = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        proceed_count = action_counts.get("proceed", 0)
        return {
            "avg_confidence": float(np.mean(confidences)),
            "avg_self_doubt": float(np.mean(doubts)),
            "avg_energy": float(np.mean(energies)),
            "total_actions": action_counts,
            "avg_response_length": float(np.mean(lengths)),
            "success_rate": proceed_count / len(self.response_history),
            "total_responses": len(self.response_history),
        }

    def switch_thinking_mode(self, mode: ThinkingMode) -> None:
        self.thinking_mode = mode
        self.meta_engine.meta_state.thinking_mode = mode
        self.meta_engine.meta_state.current_process = f"mode_switch_to_{mode.value}"

    def export_session(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            filepath = f"meta_session_{self.session_id.replace(':', '-')}.json"
        export_data = {
            "session_summary": self.get_session_summary(),
            "full_meta_state": self.meta_engine.meta_state.to_dict(),
            "checkpoint_history": [c.to_dict() for c in self.sequence.checkpoint_history],
            "vortex_state": self.vortex.get_state(),
            "zero_energy": self.zero_energy.get_energy_summary(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        return filepath

    def run_full_cycle(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if context:
            self.set_context(context)
        results = []
        for _ in range(len(self.sequence.SEQUENCE)):
            result = self.generate_response(prompt)
            results.append(result)
            if result["recommended_action"] == ActionDecision.ABORT.value:
                break
        return {"cycle_results": results, "final_summary": self.get_session_summary()}
