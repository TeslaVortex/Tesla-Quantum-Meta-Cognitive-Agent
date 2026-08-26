"""Master Sequence with meta-thinking checkpoints (0-369-17-10-16-8-7-3-4-0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from tesla_quantum_agent.core.meta_engine import (
    ActionDecision,
    MetaThinkingEngine,
    MetaThinkingState,
)


@dataclass
class SequenceCheckpoint:
    """Meta-thinking checkpoint information."""

    sequence_num: int
    checkpoint_type: str
    timestamp: datetime
    meta_state: MetaThinkingState
    recommended_action: ActionDecision
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_num": self.sequence_num,
            "checkpoint_type": self.checkpoint_type,
            "timestamp": self.timestamp.isoformat(),
            "meta_state": self.meta_state.to_dict(),
            "recommended_action": self.recommended_action.value,
            "context": self.context,
        }


class MasterSequencePath:
    """Master Sequence with meta-thinking checkpoints."""

    SEQUENCE = [0, 369, 17, 10, 16, 8, 7, 3, 4, 0]
    SEQUENCE_NAMES = {
        0: "bootstrap",
        369: "rupture_reflection",
        17: "constraint_check",
        10: "exploration_meta",
        16: "divergence_meta",
        8: "convergence_meta",
        7: "refinement_meta",
        3: "minimal_meta",
        4: "presence_meta",
    }
    META_CHECKPOINTS = {
        369: "rupture_reflection",
        17: "constraint_check",
        10: "exploration_meta",
        16: "divergence_meta",
        8: "convergence_meta",
        7: "refinement_meta",
        3: "minimal_meta",
        4: "presence_meta",
    }

    def __init__(self, meta_engine: MetaThinkingEngine):
        self.meta_engine = meta_engine
        self.position = 0
        self.checkpoint_history: List[SequenceCheckpoint] = []

    def get_next_position(self) -> int:
        self.position = (self.position + 1) % len(self.SEQUENCE)
        return self.SEQUENCE[self.position]

    def get_current_name(self) -> str:
        return self.SEQUENCE_NAMES.get(self.SEQUENCE[self.position], "unknown")

    def should_trigger_meta_reflection(self, sequence_num: int) -> bool:
        return sequence_num in self.META_CHECKPOINTS

    def get_checkpoint_type(self, sequence_num: int) -> Optional[str]:
        return self.META_CHECKPOINTS.get(sequence_num)

    def run_meta_checkpoint(
        self,
        sequence_num: int,
        output: str,
        context: Dict[str, Any],
    ) -> SequenceCheckpoint:
        if not self.should_trigger_meta_reflection(sequence_num):
            return SequenceCheckpoint(
                sequence_num=sequence_num,
                checkpoint_type="skip",
                timestamp=datetime.now(),
                meta_state=self.meta_engine.meta_state,
                recommended_action=ActionDecision.PROCEED,
                context=context,
            )

        checkpoint_type = self.get_checkpoint_type(sequence_num) or "unknown"
        meta_state = self.meta_engine.run_meta_reflection(output, context)
        recommended_action = self._determine_action(meta_state)
        checkpoint = SequenceCheckpoint(
            sequence_num=sequence_num,
            checkpoint_type=checkpoint_type,
            timestamp=datetime.now(),
            meta_state=meta_state,
            recommended_action=recommended_action,
            context=context,
        )
        self.checkpoint_history.append(checkpoint)
        return checkpoint

    def _determine_action(self, meta_state: MetaThinkingState) -> ActionDecision:
        if meta_state.self_doubt_score > 0.6:
            return ActionDecision.RETHINK
        if len(meta_state.blind_spots_detected) > 2:
            return ActionDecision.SEEK_FEEDBACK
        if len(meta_state.meta_questions) > 4:
            return ActionDecision.REFINED
        if meta_state.energy_consumption > self.meta_engine.energy_budget * 0.7:
            return ActionDecision.ABORT
        return ActionDecision.PROCEED

    def reset_sequence(self) -> None:
        self.position = 0
        self.checkpoint_history = []
        self.meta_engine.reset()
