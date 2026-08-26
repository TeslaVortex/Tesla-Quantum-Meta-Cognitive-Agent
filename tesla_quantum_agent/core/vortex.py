"""Vortex-Novelty engine: divergent exploration and path scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class VortexNoveltyState:
    """Tracks creative exploration state."""

    exploration_count: int = 0
    diversity_score: float = 0.0
    convergence_score: float = 0.0
    current_path: str = ""
    novelty_threshold: float = 0.3
    exploration_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_exploration(self, path: str, score: float) -> None:
        self.exploration_count += 1
        self.exploration_history.append(
            {
                "path": path,
                "score": float(score),
                "timestamp": datetime.now(),
            }
        )
        self.diversity_score = min(1.0, self.diversity_score + 0.1)

    def converge(self, target: str) -> None:
        self.convergence_score = max(0.0, self.convergence_score - 0.1)
        self.current_path = target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_count": self.exploration_count,
            "diversity_score": self.diversity_score,
            "convergence_score": self.convergence_score,
            "current_path": self.current_path,
            "novelty_threshold": self.novelty_threshold,
            "recent_history": [
                {
                    "path": h["path"],
                    "score": h["score"],
                    "timestamp": h["timestamp"].isoformat()
                    if hasattr(h["timestamp"], "isoformat")
                    else str(h["timestamp"]),
                }
                for h in self.exploration_history[-10:]
            ],
        }


class VortexNoveltyEngine:
    """Creative exploration engine for divergent thinking."""

    def __init__(self, novelty_threshold: float = 0.3):
        self.novelty_threshold = novelty_threshold
        self.exploration_state = VortexNoveltyState(novelty_threshold=novelty_threshold)
        self.potential_paths: List[str] = []

    def generate_exploration_paths(self, problem: str, context: Dict[str, Any]) -> List[str]:
        paths: List[str] = [
            f"Analyze {problem} from traditional perspective",
            f"Apply established frameworks to {problem}",
            f"Identify constraints in {problem}",
        ]
        if context.get("allow_creative", True):
            paths.extend(
                [
                    f"Reframe {problem} as an opportunity",
                    f"Apply metaphor from different domain to {problem}",
                    f"Question fundamental assumptions about {problem}",
                    f"Consider {problem} from opposite perspective",
                    f"Use random association with {problem}",
                ]
            )
        paths.extend(
            [
                f"Map relationships in {problem}",
                f"Identify feedback loops in {problem}",
                f"Consider long-term implications of {problem}",
            ]
        )

        scored = [(path, self._calculate_novelty(path, problem)) for path in paths]
        selected = [(path, score) for path, score in scored if score >= self.novelty_threshold]
        selected.sort(key=lambda x: x[1], reverse=True)

        self.exploration_state.novelty_threshold = self.novelty_threshold
        self.potential_paths = [path for path, _ in selected[:5]]
        for path, score in selected[:5]:
            self.exploration_state.record_exploration(path, score)
        return self.potential_paths

    def _calculate_novelty(self, path: str, problem: str) -> float:
        base_score = len(path) / 50.0
        creative_keywords = [
            "reframe", "opportunity", "metaphor", "domain", "question",
            "assumption", "opposite", "random", "associat", "systemic",
            "feedback", "relationship",
        ]
        keyword_bonus = sum(0.1 for kw in creative_keywords if kw in path.lower())
        problem_words = set(problem.lower().split())
        path_words = set(path.lower().split())
        denom = max(len(problem_words), len(path_words), 1)
        distance_bonus = 1.0 - (len(problem_words & path_words) / denom)
        return min(1.0, base_score * 0.5 + keyword_bonus * 0.5 + distance_bonus * 0.5)

    def converge_exploration(self, selected_path: str, target: str) -> float:
        self.exploration_state.converge(target)
        target_words = set(target.lower().split())
        path_words = set(selected_path.lower().split())
        denom = max(len(target_words), len(path_words), 1)
        return len(target_words & path_words) / denom

    def get_state(self) -> Dict[str, Any]:
        return self.exploration_state.to_dict()
