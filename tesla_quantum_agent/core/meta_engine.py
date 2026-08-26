"""Meta-thinking engine: reflection, assumption flagging, blind-spot detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ThinkingMode(Enum):
    """Available thinking modes for the agent."""

    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CREATVE = "creative"  # original-doc alias
    CRITICAL = "critical"
    META = "meta"
    HYBRID = "hybrid"


class ReflectionDepth(Enum):
    """Levels of meta-cognitive reflection."""

    SHALLOW = 1
    MODERATE = 2
    DEEP = 3
    EXHAUSTIVE = 4


class ActionDecision(Enum):
    """Actions determined by meta-reflection."""

    PROCEED = "proceed"
    REFINED = "refined"
    RETHINK = "rethink"
    SEEK_FEEDBACK = "seek_feedback"
    ABORT = "abort"


@dataclass
class MetaThinkingState:
    """Tracks the AI's thinking about its thinking."""

    current_process: str
    thinking_mode: ThinkingMode = ThinkingMode.ANALYTICAL
    confidence: float = 0.0
    self_doubt_score: float = 0.0
    reflection_depth: int = 1
    assumptions_flagged: List[str] = field(default_factory=list)
    blind_spots_detected: List[str] = field(default_factory=list)
    meta_questions: List[str] = field(default_factory=list)
    time_stamps: List[datetime] = field(default_factory=list)
    energy_consumption: float = 0.0

    def add_meta_question(self, question: str) -> None:
        self.meta_questions.append(question)
        self.time_stamps.append(datetime.now())

    def flag_assumption(self, assumption: str) -> None:
        self.assumptions_flagged.append(assumption)

    def detect_blind_spot(self, spot: str) -> None:
        self.blind_spots_detected.append(spot)

    def update_energy(self, delta: float) -> None:
        """Track computational energy for the Zero-Energy Lens."""
        self.energy_consumption += float(delta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_process": self.current_process,
            "thinking_mode": self.thinking_mode.value,
            "confidence": self.confidence,
            "self_doubt_score": self.self_doubt_score,
            "reflection_depth": self.reflection_depth,
            "assumptions_flagged": list(self.assumptions_flagged),
            "blind_spots_detected": list(self.blind_spots_detected),
            "meta_questions": list(self.meta_questions),
            "time_stamps": [t.isoformat() for t in self.time_stamps],
            "energy_consumption": self.energy_consumption,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaThinkingState":
        return cls(
            current_process=data["current_process"],
            thinking_mode=ThinkingMode(data["thinking_mode"]),
            confidence=data["confidence"],
            self_doubt_score=data["self_doubt_score"],
            reflection_depth=data["reflection_depth"],
            assumptions_flagged=list(data.get("assumptions_flagged", [])),
            blind_spots_detected=list(data.get("blind_spots_detected", [])),
            meta_questions=list(data.get("meta_questions", [])),
            time_stamps=[datetime.fromisoformat(t) for t in data.get("time_stamps", [])],
            energy_consumption=data.get("energy_consumption", 0.0),
        )


class MetaThinkingEngine:
    """Core engine for meta-cognitive processing."""

    def __init__(
        self,
        model_name: str = "Qwen3.5-9B",
        reflection_threshold: float = 0.5,
        energy_budget: float = 100.0,
    ):
        self.model_name = model_name
        self.reflection_threshold = reflection_threshold
        self.energy_budget = energy_budget
        self.meta_state = self._initialize_state()
        self.session_history: List[Dict[str, Any]] = []

    def _initialize_state(self) -> MetaThinkingState:
        return MetaThinkingState(
            current_process="initializing",
            thinking_mode=ThinkingMode.ANALYTICAL,
            confidence=0.0,
            self_doubt_score=0.0,
            reflection_depth=0,
            assumptions_flagged=[],
            blind_spots_detected=[],
            meta_questions=[],
        )

    def reset(self) -> None:
        """Reset meta state for a new task (preserves energy budget, not ledger)."""
        self.meta_state = self._initialize_state()
        self.session_history = []
        self.meta_state.current_process = "reset"

    def run_meta_reflection(self, output: str, context: Dict[str, Any]) -> MetaThinkingState:
        """Generate meta-reflection on the agent's own output."""
        self.meta_state.current_process = "reflection"
        self.meta_state.reflection_depth += 1
        self.meta_state.update_energy(1.0)

        for q in self._generate_meta_questions(output, context):
            self.meta_state.add_meta_question(q)
        assumptions = self._detect_assumptions(output)
        for a in assumptions:
            self.meta_state.flag_assumption(a)
        blind_spots = self._detect_blind_spots(output, context)
        for b in blind_spots:
            self.meta_state.detect_blind_spot(b)

        confidence = self._calculate_confidence(output, context)
        self.meta_state.confidence = confidence
        self.meta_state.self_doubt_score = 1.0 - confidence
        self._log_session(output, confidence, len(assumptions), len(blind_spots))
        return self.meta_state

    def _generate_meta_questions(self, output: str, context: Dict[str, Any]) -> List[str]:
        questions: List[str] = []
        if len(output) > 50:
            questions.extend(
                [
                    f"What assumptions underlie '{output[:50]}...'?",
                    "Is this answer too confident given the context?",
                    "What alternative perspectives might contradict this?",
                ]
            )

        task_type = context.get("task_type", "general")
        if task_type == "reasoning":
            questions.append("Did I oversimplify the causal chain?")
            questions.append("Are there unstated premises in the argument?")
        elif task_type in ("creative",):
            questions.append("Am I constrained by conventional thinking?")
            questions.append("Have I explored all unconventional angles?")
        elif task_type == "critical":
            questions.append("Am I being too critical? Is there bias in my evaluation?")
            questions.append("What evidence might contradict my conclusions?")

        if context.get("time_sensitive", False):
            questions.append("Is the information current and relevant?")
            questions.append("What might change in the near future?")

        for stakeholder in context.get("stakeholders", []) or []:
            questions.append(f"How would {stakeholder} view this conclusion?")

        return questions[:5]

    def _detect_assumptions(self, output: str) -> List[str]:
        assumptions: List[str] = []
        text_lower = output.lower()
        patterns = [
            (r"must|should|always|never", "Absolute statement"),
            (r"will|cannot|impossible", "Certainty claim"),
            (r"everyone|all|no one|nobody", "Universal generalization"),
            (r"some people|many people|most people", "Partial generalization"),
            (r"\bthis\b|\bthat\b|\bthe\b", "Implicit reference"),
            (r"\bwe\b|\bus\b|\bour\b", "Collective assumption"),
            (r"good|bad|right|wrong|better|worse", "Value judgment"),
            (r"important|trivial|significant|minor", "Importance assumption"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, text_lower):
                if label in ("Value judgment", "Importance assumption"):
                    assumptions.append(f"{label}: May indicate bias")
                else:
                    assumptions.append(f"{label}: Found in output")
        return list(dict.fromkeys(assumptions))

    def _detect_blind_spots(self, output: str, context: Dict[str, Any]) -> List[str]:
        blind_spots: List[str] = []
        text_lower = output.lower()

        confidence_markers = [
            "obviously",
            "clearly",
            "definitely",
            "certainly",
            "undoubtedly",
            "no doubt",
            "without a doubt",
        ]
        if any(marker in text_lower for marker in confidence_markers):
            blind_spots.append("Overconfidence detected - may need verification")

        hedging_markers = [
            "might",
            "could",
            "possibly",
            "may",
            "perhaps",
            "likely",
            "probably",
            "seems",
            "appears",
        ]
        if sum(1 for m in hedging_markers if m in text_lower) > 3:
            blind_spots.append("Excessive hedging - may indicate lack of confidence")

        stakeholders = context.get("stakeholders", []) or []
        if stakeholders and "missing" in text_lower:
            blind_spots.append(f"Missing stakeholder perspectives: {stakeholders}")

        domain = context.get("domain")
        if domain and str(domain).lower() not in output.lower():
            blind_spots.append("Possible domain mismatch detected")

        if context.get("time_sensitive", False) and "time" not in text_lower:
            blind_spots.append("Time sensitivity not addressed")

        if output.count(".") > 5 and len(output) < 500:
            blind_spots.append("Possible circular reasoning detected")

        return blind_spots

    def _calculate_confidence(self, output: str, context: Dict[str, Any]) -> float:
        words = output.split()
        if not words:
            return 0.0

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "this", "that",
            "these", "those", "i", "you", "we", "they", "he", "she", "it", "what",
            "which", "who", "when", "where", "why", "how", "for", "with", "about",
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "from", "up", "down", "in", "out", "on", "off",
            "over", "under", "and", "or", "but", "if", "then", "else", "as", "at",
            "by", "of", "to",
        }
        qualifiers = {
            "might", "could", "possibly", "may", "perhaps", "likely", "probably",
            "seems", "appears", "maybe", "uncertain", "unsure", "questionable",
            "debatable", "risky",
        }

        certainty_words = [
            w for w in words
            if w.lower() not in stop_words and w.lower() not in qualifiers and len(w) > 2
        ]
        if len(words) > 10:
            certainty_ratio = len(certainty_words) / len(words)
        else:
            certainty_ratio = len(words) / 10.0

        qualifier_ratio = sum(1 for w in words if w.lower() in qualifiers) / len(words)
        adjusted_ratio = certainty_ratio * (1 - 0.3 * qualifier_ratio)

        domain = str(context.get("domain", ""))
        if domain and "experimental" in domain.lower():
            adjusted_ratio *= 0.8

        return min(0.95, max(0.3, adjusted_ratio))

    def _log_session(
        self,
        output: str,
        confidence: float,
        assumptions: int,
        blind_spots: int,
    ) -> None:
        self.session_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "output_length": len(output),
                "confidence": confidence,
                "assumptions_count": assumptions,
                "blind_spots_count": blind_spots,
                "reflection_depth": self.meta_state.reflection_depth,
                "energy_consumption": self.meta_state.energy_consumption,
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "current_process": self.meta_state.current_process,
            "thinking_mode": self.meta_state.thinking_mode.value,
            "confidence": self.meta_state.confidence,
            "self_doubt": self.meta_state.self_doubt_score,
            "reflection_depth": self.meta_state.reflection_depth,
            "total_assumptions": len(self.meta_state.assumptions_flagged),
            "total_blind_spots": len(self.meta_state.blind_spots_detected),
            "total_meta_questions": len(self.meta_state.meta_questions),
            "energy_consumed": self.meta_state.energy_consumption,
            "session_count": len(self.session_history),
            "recent_sessions": self.session_history[-5:],
        }
