"""
META-COGNITIVE AI AGENT FRAMEWORK
Complete Implementation for Qwen 3.5 9B Fine-tuning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from datetime import datetime


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class ThinkingMode(Enum):
    """Available thinking modes for the agent"""
    ANALYTICAL = "analytical"      # Logical, structured reasoning
    CREATVE = "creative"          # Divergent, novel thinking
    CRITICAL = "critical"         # Evaluation, bias-checking
    META = "meta"                 # Thinking about thinking
    HYBRID = "hybrid"             # Combined approaches


class ReflectionDepth(Enum):
    """Levels of meta-cognitive reflection"""
    SHALLOW = 1      # Quick self-check
    MODERATE = 2     # Standard reflection
    DEEP = 3         # Comprehensive analysis
    EXHAUSTIVE = 4   # Full audit


class ActionDecision(Enum):
    """Actions determined by meta-reflection"""
    PROCEED = "proceed"              # Continue with current approach
    REFINED = "refined"              # Minor adjustments needed
    RETHINK = "rethink"              # Significant changes required
    SEEK_FEEDBACK = "seek_feedback"  # External input needed
    ABORT = "abort"                  # Stop and reassess


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MetaThinkingState:
    """Tracks the AI's thinking about its thinking"""
    current_process: str
    thinking_mode: ThinkingMode = ThinkingMode.ANALYTICAL
    confidence: float = 0.0
    self_doubt_score: float = 0.0
    reflection_depth: int = 1
    assumptions_flagged: List[str] = field(default_factory=list)
    blind_spots_detected: List[str] = field(default_factory=list)
    meta_questions: List[str] = field(default_factory=list)
    time_stamps: List[datetime] = field(default_factory=list)
    energy_consumption: float = 0.0  # For Zero-Energy Lens
    
    def add_meta_question(self, question: str):
        self.meta_questions.append(question)
        self.time_stamps.append(datetime.now())
    
    def flag_assumption(self, assumption: str):
        self.assumptions_flagged.append(assumption)
    
    def detect_blind_spot(self, spot: str):
        self.blind_spots_detected.append(spot)
    
    def update_energy(self, delta: float):
        """Track computational energy for Zero-Energy Lens"""
        self.energy_consumption += delta
    
    def to_dict(self) -> Dict:
        return {
            'current_process': self.current_process,
            'thinking_mode': self.thinking_mode.value,
            'confidence': self.confidence,
            'self_doubt_score': self.self_doubt_score,
            'reflection_depth': self.reflection_depth,
            'assumptions_flagged': self.assumptions_flagged,
            'blind_spots_detected': self.blind_spots_detected,
            'meta_questions': self.meta_questions,
            'time_stamps': [t.isoformat() for t in self.time_stamps],
            'energy_consumption': self.energy_consumption
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetaThinkingState':
        state = cls(
            current_process=data['current_process'],
            thinking_mode=ThinkingMode(data['thinking_mode']),
            confidence=data['confidence'],
            self_doubt_score=data['self_doubt_score'],
            reflection_depth=data['reflection_depth'],
            assumptions_flagged=data['assumptions_flagged'],
            blind_spots_detected=data['blind_spots_detected'],
            meta_questions=data['meta_questions'],
            time_stamps=[datetime.fromisoformat(t) for t in data['time_stamps']],
            energy_consumption=data.get('energy_consumption', 0.0)
        )
        return state


@dataclass
class SequenceCheckpoint:
    """Meta-thinking checkpoint information"""
    sequence_num: int
    checkpoint_type: str
    timestamp: datetime
    meta_state: MetaThinkingState
    recommended_action: ActionDecision
    context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'sequence_num': self.sequence_num,
            'checkpoint_type': self.checkpoint_type,
            'timestamp': self.timestamp.isoformat(),
            'meta_state': self.meta_state.to_dict(),
            'recommended_action': self.recommended_action.value,
            'context': self.context
        }


@dataclass
class VortexNoveltyState:
    """Tracks creative exploration state"""
    exploration_count: int = 0
    diversity_score: float = 0.0
    convergence_score: float = 0.0
    current_path: str = ""
    novelty_threshold: float = 0.3
    exploration_history: List[Dict] = field(default_factory=list)
    
    def record_exploration(self, path: str, score: float):
        self.exploration_count += 1
        self.exploration_history.append({
            'path': path,
            'score': score,
            'timestamp': datetime.now()
        })
        self.diversity_score = min(1.0, self.diversity_score + 0.1)
    
    def converge(self, target: str):
        self.convergence_score = max(0.0, self.convergence_score - 0.1)
        self.current_path = target
    
    def to_dict(self) -> Dict:
        return {
            'exploration_count': self.exploration_count,
            'diversity_score': self.diversity_score,
            'convergence_score': self.convergence_score,
            'current_path': self.current_path,
            'novelty_threshold': self.novelty_threshold,
            'recent_history': self.exploration_history[-10:]
        }


# ============================================================
# META THINKING ENGINE
# ============================================================

class MetaThinkingEngine:
    """Core engine for meta-cognitive processing"""
    
    def __init__(self, model_name: str = "Qwen3.5-9B", 
                 reflection_threshold: float = 0.5,
                 energy_budget: float = 100.0):
        self.model_name = model_name
        self.reflection_threshold = reflection_threshold
        self.energy_budget = energy_budget
        self.meta_state = self._initialize_state()
        self.session_history: List[Dict] = []
        
    def _initialize_state(self) -> MetaThinkingState:
        return MetaThinkingState(
            current_process="initializing",
            thinking_mode=ThinkingMode.ANALYTICAL,
            confidence=0.0,
            self_doubt_score=0.0,
            reflection_depth=0,
            assumptions_flagged=[],
            blind_spots_detected=[],
            meta_questions=[]
        )
    
    def reset(self):
        """Reset meta state for new task"""
        self.meta_state = self._initialize_state()
        self.session_history = []
        self.meta_state.current_process = "reset"
    
    def run_meta_reflection(self, output: str, context: Dict) -> MetaThinkingState:
        """Generate meta-reflection on AI's own output"""
        self.meta_state.current_process = "reflection"
        self.meta_state.reflection_depth += 1
        self.meta_state.update_energy(1.0)
        
        # 1. Generate meta questions
        meta_questions = self._generate_meta_questions(output, context)
        for q in meta_questions:
            self.meta_state.add_meta_question(q)
        
        # 2. Detect assumptions
        assumptions = self._detect_assumptions(output)
        for a in assumptions:
            self.meta_state.flag_assumption(a)
        
        # 3. Detect blind spots
        blind_spots = self._detect_blind_spots(output, context)
        for b in blind_spots:
            self.meta_state.detect_blind_spot(b)
        
        # 4. Calculate confidence and self-doubt
        confidence = self._calculate_confidence(output, context)
        self.meta_state.confidence = confidence
        self.meta_state.self_doubt_score = 1.0 - confidence
        
        # 5. Log session
        self._log_session(output, confidence, len(assumptions), len(blind_spots))
        
        return self.meta_state
    
    def _generate_meta_questions(self, output: str, context: Dict) -> List[str]:
        """Generate meta-cognitive questions about the output"""
        questions = []
        
        # Standard meta questions based on output characteristics
        if len(output) > 50:
            questions.extend([
                f"What assumptions underlie '{output[:50]}...'?",
                f"Is this answer too confident given the context?",
                f"What alternative perspectives might contradict this?",
            ])
        
        # Context-specific questions
        task_type = context.get('task_type', 'general')
        if task_type == 'reasoning':
            questions.append("Did I oversimplify the causal chain?")
            questions.append("Are there unstated premises in the argument?")
        elif task_type == 'creative':
            questions.append("Am I constrained by conventional thinking?")
            questions.append("Have I explored all unconventional angles?")
        elif task_type == 'critical':
            questions.append("Am I being too critical? Is there bias in my evaluation?")
            questions.append("What evidence might contradict my conclusions?")
        
        # Temporal awareness questions
        if context.get('time_sensitive', False):
            questions.append("Is the information current and relevant?")
            questions.append("What might change in the near future?")
        
        # Stakeholder perspective questions
        stakeholders = context.get('stakeholders', [])
        if stakeholders:
            for stakeholder in stakeholders:
                questions.append(f"How would {stakeholder} view this conclusion?")
        
        return questions[:5]  # Limit for efficiency
    
    def _detect_assumptions(self, output: str) -> List[str]:
        """Extract and flag assumptions in output"""
        assumptions = []
        text_lower = output.lower()
        
        # Absolute statement patterns
        absolute_patterns = [
            (r'must|should|always|never', 'Absolute statement'),
            (r'will|cannot|impossible', 'Certainty claim'),
        ]
        
        # Generalization patterns
        generalization_patterns = [
            (r'everyone|all|no one|nobody', 'Universal generalization'),
            (r'some people|many people|most people', 'Partial generalization'),
        ]
        
        # Implicit reference patterns
        implicit_patterns = [
            (r'this|that|the', 'Implicit reference'),
            (r'we|us|our', 'Collective assumption'),
        ]
        
        # Check each pattern
        for pattern, label in absolute_patterns + generalization_patterns + implicit_patterns:
            if re.search(pattern, text_lower):
                assumptions.append(f"{label}: Found in output")
        
        # Check for hidden value judgments
        value_patterns = [
            (r'good|bad|right|wrong|better|worse', 'Value judgment'),
            (r'important|trivial|significant|minor', 'Importance assumption'),
        ]
        
        for pattern, label in value_patterns:
            if re.search(pattern, text_lower):
                assumptions.append(f"{label}: May indicate bias")
        
        return list(set(assumptions))  # Remove duplicates
    
    def _detect_blind_spots(self, output: str, context: Dict) -> List[str]:
        """Identify potential blind spots in reasoning"""
        blind_spots = []
        text_lower = output.lower()
        
        # Overconfidence markers
        confidence_markers = [
            'obviously', 'clearly', 'definitely', 'certainly',
            'undoubtedly', 'no doubt', 'without a doubt'
        ]
        if any(marker in text_lower for marker in confidence_markers):
            blind_spots.append("Overconfidence detected - may need verification")
        
        # Hedging markers (indicates uncertainty)
        hedging_markers = [
            'might', 'could', 'possibly', 'may', 'perhaps',
            'likely', 'probably', 'seems', 'appears'
        ]
        hedging_count = sum(1 for m in hedging_markers if m in text_lower)
        if hedging_count > 3:
            blind_spots.append("Excessive hedging - may indicate lack of confidence")
        
        # Missing perspective check
        stakeholders = context.get('stakeholders', [])
        if stakeholders and 'missing' in text_lower:
            blind_spots.append(f"Missing stakeholder perspectives: {stakeholders}")
        
        # Context mismatch check
        if context.get('domain') and context['domain'] not in output.lower():
            blind_spots.append("Possible domain mismatch detected")
        
        # Temporal relevance check
        if context.get('time_sensitive', False) and 'time' not in text_lower:
            blind_spots.append("Time sensitivity not addressed")
        
        # Check for circular reasoning
        if output.count('.') > 5 and len(output) < 500:
            blind_spots.append("Possible circular reasoning detected")
        
        return blind_spots
    
    def _calculate_confidence(self, output: str, context: Dict) -> float:
        """Estimate confidence level based on output characteristics"""
        words = output.split()
        if not words:
            return 0.0
        
        # Remove punctuation and stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'this', 'that', 'these', 'those', 'i', 'you', 'we',
                     'they', 'he', 'she', 'it', 'what', 'which', 'who', 'when',
                     'where', 'why', 'how', 'for', 'with', 'about', 'against',
                     'between', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on',
                     'off', 'over', 'under', 'and', 'or', 'but', 'if', 'then',
                     'else', 'when', 'as', 'at', 'by', 'from', 'of', 'to'}
        
        # Count qualifiers
        qualifiers = ['might', 'could', 'possibly', 'may', 'perhaps', 'likely',
                     'probably', 'seems', 'appears', 'possibly', 'maybe',
                     'uncertain', 'unsure', 'questionable', 'debatable', 'risky']
        
        certainty_words = [w for w in words if w.lower() not in stop_words and 
                         w.lower() not in qualifiers and len(w) > 2]
        
        # Calculate certainty ratio
        if len(words) > 10:
            certainty_ratio = len(certainty_words) / len(words)
        else:
            certainty_ratio = len(words) / 10
        
        # Adjust based on qualifiers
        qualifier_ratio = sum(1 for w in words if w.lower() in qualifiers) / len(words)
        adjusted_ratio = certainty_ratio * (1 - 0.3 * qualifier_ratio)
        
        # Context adjustments
        domain = context.get('domain', '')
        if domain and 'experimental' in domain.lower():
            adjusted_ratio *= 0.8  # More uncertainty for experimental domains
        
        return min(0.95, max(0.3, adjusted_ratio))
    
    def _log_session(self, output: str, confidence: float, 
                    assumptions: int, blind_spots: int):
        """Log session for analysis"""
        self.session_history.append({
            'timestamp': datetime.now().isoformat(),
            'output_length': len(output),
            'confidence': confidence,
            'assumptions_count': assumptions,
            'blind_spots_count': blind_spots,
            'reflection_depth': self.meta_state.reflection_depth,
            'energy_consumption': self.meta_state.energy_consumption
        })
    
    def get_summary(self) -> Dict:
        """Get summary of meta thinking state"""
        return {
            'current_process': self.meta_state.current_process,
            'thinking_mode': self.meta_state.thinking_mode.value,
            'confidence': self.meta_state.confidence,
            'self_doubt': self.meta_state.self_doubt_score,
            'reflection_depth': self.meta_state.reflection_depth,
            'total_assumptions': len(self.meta_state.assumptions_flagged),
            'total_blind_spots': len(self.meta_state.blind_spots_detected),
            'total_meta_questions': len(self.meta_state.meta_questions),
            'energy_consumed': self.meta_state.energy_consumption,
            'session_count': len(self.session_history),
            'recent_sessions': self.session_history[-5:]
        }


# ============================================================
# MASTER SEQUENCE INTEGRATION
# ============================================================

class MasterSequencePath:
    """Master Sequence with Meta Thinking checkpoints"""
    
    SEQUENCE = [0, 369, 17, 10, 16, 8, 7, 3, 4, 0]
    SEQUENCE_NAMES = {
        0: 'bootstrap',
        369: 'rupture_reflection',
        17: 'constraint_check',
        10: 'exploration_meta',
        16: 'divergence_meta',
        8: 'convergence_meta',
        7: 'refinement_meta',
        3: 'minimal_meta',
        4: 'presence_meta',
    }
    
    META_CHECKPOINTS = {
        369: 'rupture_reflection',
        17: 'constraint_check',
        10: 'exploration_meta',
        16: 'divergence_meta',
        8: 'convergence_meta',
        7: 'refinement_meta',
        3: 'minimal_meta',
        4: 'presence_meta',
    }
    
    def __init__(self, meta_engine: MetaThinkingEngine):
        self.meta_engine = meta_engine
        self.position = 0
        self.checkpoint_history: List[SequenceCheckpoint] = []
        
    def get_next_position(self) -> int:
        """Get next position in sequence"""
        self.position = (self.position + 1) % len(self.SEQUENCE)
        return self.SEQUENCE[self.position]
    
    def get_current_name(self) -> str:
        """Get name of current sequence position"""
        return self.SEQUENCE_NAMES.get(self.SEQUENCE[self.position], 'unknown')
    
    def should_trigger_meta_reflection(self, sequence_num: int) -> bool:
        """Meta thinking is triggered at specific sequence positions"""
        return sequence_num in [369, 17, 10, 16, 8, 7, 3, 4]
    
    def get_checkpoint_type(self, sequence_num: int) -> Optional[str]:
        """Get checkpoint type for sequence number"""
        return self.META_CHECKPOINTS.get(sequence_num)
    
    def run_meta_checkpoint(self, sequence_num: int, output: str, 
                           context: Dict) -> SequenceCheckpoint:
        """Run meta thinking at checkpoint"""
        if not self.should_trigger_meta_reflection(sequence_num):
            return SequenceCheckpoint(
                sequence_num=sequence_num,
                checkpoint_type='skip',
                timestamp=datetime.now(),
                meta_state=self.meta_engine.meta_state,
                recommended_action=ActionDecision.PROCEED,
                context=context
            )
        
        checkpoint_type = self.get_checkpoint_type(sequence_num)
        meta_state = self.meta_engine.run_meta_reflection(output, context)
        
        # Determine recommended action
        recommended_action = self._determine_action(meta_state)
        
        # Log checkpoint
        checkpoint = SequenceCheckpoint(
            sequence_num=sequence_num,
            checkpoint_type=checkpoint_type,
            timestamp=datetime.now(),
            meta_state=meta_state,
            recommended_action=recommended_action,
            context=context
        )
        self.checkpoint_history.append(checkpoint)
        
        return checkpoint
    
    def _determine_action(self, meta_state: MetaThinkingState) -> ActionDecision:
        """Determine action based on meta reflection"""
        if meta_state.self_doubt_score > 0.6:
            return ActionDecision.RETHINK
        elif len(meta_state.blind_spots_detected) > 2:
            return ActionDecision.SEEK_FEEDBACK
        elif len(meta_state.meta_questions) > 4:
            return ActionDecision.REFINED
        elif meta_state.energy_consumption > self.meta_engine.energy_budget * 0.7:
            return ActionDecision.ABORT
        else:
            return ActionDecision.PROCEED
    
    def reset_sequence(self):
        """Reset sequence to start"""
        self.position = 0
        self.checkpoint_history = []
        self.meta_engine.reset()


# ============================================================
# VORTEX-NOVELTY INTEGRATION
# ============================================================

class VortexNoveltyEngine:
    """Creative exploration engine for divergent thinking"""
    
    def __init__(self, novelty_threshold: float = 0.3):
        self.novelty_threshold = novelty_threshold
        self.exploration_state = VortexNoveltyState()
        self.potential_paths: List[str] = []
        
    def generate_exploration_paths(self, problem: str, context: Dict) -> List[str]:
        """Generate multiple exploration paths"""
        paths = []
        
        # Standard approaches
        paths.extend([
            f"Analyze {problem} from traditional perspective",
            f"Apply established frameworks to {problem}",
            f"Identify constraints in {problem}",
        ])
        
        # Creative approaches
        if context.get('allow_creative', True):
            paths.extend([
                f"Reframe {problem} as an opportunity",
                f"Apply metaphor from different domain to {problem}",
                f"Question fundamental assumptions about {problem}",
                f"Consider {problem} from opposite perspective",
                f"Use random association with {problem}",
            ])
        
        # Systemic approaches
        paths.extend([
            f"Map relationships in {problem}",
            f"Identify feedback loops in {problem}",
            f"Consider long-term implications of {problem}",
        ])
        
        # Filter by novelty threshold
        novelty_scores = []
        for path in paths:
            novelty_score = self._calculate_novelty(path, problem)
            novelty_scores.append((path, novelty_score))
        
        # Select paths above threshold
        selected = [(path, score) for path, score in novelty_scores 
                   if score >= self.novelty_threshold]
        selected.sort(key=lambda x: x[1], reverse=True)
        
        self.exploration_state.novelty_threshold = self.novelty_threshold
        for path, score in selected[:5]:
            self.exploration_state.record_exploration(path, score)
        
        return [path for path, _ in selected[:5]]
    
    def _calculate_novelty(self, path: str, problem: str) -> float:
        """Calculate novelty score for a path"""
        # Base score from path length and complexity
        base_score = len(path) / 50.0
        
        # Bonus for creative keywords
        creative_keywords = [
            'reframe', 'opportunity', 'metaphor', 'domain', 
            'question', 'assumption', 'opposite', 'random',
            'associat', 'systemic', 'feedback', 'relationship'
        ]
        
        keyword_bonus = 0.0
        for keyword in creative_keywords:
            if keyword.lower() in path.lower():
                keyword_bonus += 0.1
        
        # Bonus for problem distance
        problem_words = set(problem.lower().split())
        path_words = set(path.lower().split())
        shared_ratio = len(problem_words & path_words) / max(len(problem_words), len(path_words))
        distance_bonus = 1.0 - shared_ratio
        
        return min(1.0, base_score * 0.5 + keyword_bonus * 0.5 + distance_bonus * 0.5)
    
    def converge_exploration(self, selected_path: str, target: str) -> float:
        """Converge exploration toward target"""
        self.exploration_state.converge(target)
        
        # Calculate convergence score
        target_words = set(target.lower().split())
        path_words = set(selected_path.lower().split())
        overlap = len(target_words & path_words) / max(len(target_words), len(path_words))
        
        return overlap
    
    def get_state(self) -> Dict:
        """Get current exploration state"""
        return self.exploration_state.to_dict()


# ============================================================
# ZERO-ENERGY LENS INTEGRATION
# ============================================================

class ZeroEnergyLens:
    """Belief-state consistency checking for energy efficiency"""
    
    def __init__(self, consistency_threshold: float = 0.7):
        self.consistency_threshold = consistency_threshold
        self.belief_history: List[Dict] = []
        self.energy_log: List[float] = []
        
    def check_belief_consistency(self, current_beliefs: Dict, 
                                 new_belief: str) -> Tuple[bool, float]:
        """Check if new belief is consistent with existing beliefs"""
        # Extract key concepts from beliefs
        current_keys = set(current_beliefs.keys())
        new_key = new_belief[:50]  # Sample for efficiency
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency(
            current_beliefs, new_belief
        )
        
        # Log energy consumption
        energy_cost = self._calculate_energy_cost(len(current_beliefs) + 1)
        self.energy_log.append(energy_cost)
        
        return consistency_score >= self.consistency_threshold, consistency_score
    
    def _calculate_consistency(self, current_beliefs: Dict, 
                               new_belief: str) -> float:
        """Calculate consistency between new belief and existing beliefs"""
        # Extract key terms from new belief
        new_terms = set(new_belief.lower().split())
        
        # Check for contradictions
        contradictions = 0
        total_checks = 0
        
        for key, belief in current_beliefs.items():
            # Check if key terms conflict
            if key.lower() in new_belief.lower():
                total_checks += 1
                if belief.lower() != new_belief.lower():
                    contradictions += 0.3
                else:
                    contradictions -= 0.1
            
            # Check for related concepts
            related_terms = self._get_related_terms(key)
            for term in related_terms:
                if term in new_terms:
                    total_checks += 1
                    contradictions += 0.05
        
        # Calculate consistency score
        if total_checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (contradictions / total_checks))
    
    def _get_related_terms(self, term: str) -> List[str]:
        """Get related terms for consistency checking"""
        related_map = {
            'belief': ['opinion', 'view', 'thought', 'perspective'],
            'assumption': ['premise', 'supposition', 'hypothesis'],
            'conclusion': ['result', 'outcome', 'finding', 'deduction'],
            'evidence': ['data', 'proof', 'support', 'basis'],
            'reasoning': ['logic', 'argument', 'analysis', 'process'],
        }
        
        return related_map.get(term.lower(), [])
    
    def _calculate_energy_cost(self, num_beliefs: int) -> float:
        """Calculate energy cost for belief processing"""
        # Simplified energy model
        base_cost = 1.0
        scaling_factor = num_beliefs ** 0.3
        return base_cost * scaling_factor
    
    def log_belief(self, belief: str, timestamp: datetime = None):
        """Log a new belief"""
        self.belief_history.append({
            'belief': belief,
            'timestamp': timestamp or datetime.now(),
            'energy_cost': self.energy_log[-1] if self.energy_log else 0.0
        })
    
    def get_energy_summary(self) -> Dict:
        """Get energy consumption summary"""
        return {
            'total_energy': sum(self.energy_log),
            'average_energy': sum(self.energy_log) / len(self.energy_log) if self.energy_log else 0.0,
            'belief_count': len(self.belief_history),
            'recent_beliefs': self.belief_history[-10:]
        }


# ============================================================
# INTEGRATED AGENT CLASS
# ============================================================

class MetaCognitiveAgent:
    """Complete Meta-Cognitive AI Agent with all integrated components"""
    
    def __init__(self, model_name: str = "Qwen3.5-9B",
                 reflection_threshold: float = 0.5,
                 novelty_threshold: float = 0.3,
                 consistency_threshold: float = 0.7,
                 energy_budget: float = 100.0):
        
        # Initialize core components
        self.meta_engine = MetaThinkingEngine(
            model_name=model_name,
            reflection_threshold=reflection_threshold,
            energy_budget=energy_budget
        )
        
        self.sequence = MasterSequencePath(self.meta_engine)
        self.vortex = VortexNoveltyEngine(novelty_threshold)
        self.zero_energy = ZeroEnergyLens(consistency_threshold)
        
        self.context = {}
        self.current_response = ""
        self.thinking_mode = ThinkingMode.ANALYTICAL
        
        # Session tracking
        self.session_id = datetime.now().isoformat()
        self.response_history: List[Dict] = []
    
    def set_context(self, context: Dict):
        """Set task context"""
        self.context = context
    
    def generate_response(self, prompt: str) -> Dict:
        """Generate response with full meta-cognitive processing"""
        start_time = datetime.now()
        
        # Reset for new task
        self.meta_engine.reset()
        self.vortex.exploration_state.exploration_count = 0
        self.zero_energy.belief_history = []
        
        # Step 1: Initial response generation
        initial_response = self._generate_initial_response(prompt)
        
        # Step 2: Master Sequence processing
        sequence_num = self.sequence.get_next_position()
        checkpoint = self.sequence.run_meta_checkpoint(
            sequence_num, initial_response, self.context
        )
        
        # Step 3: VORTEX-NOVELTY exploration
        exploration_paths = self.vortex.generate_exploration_paths(
            prompt, self.context
        )
        
        # Step 4: Zero-Energy consistency check
        is_consistent, consistency_score = self.zero_energy.check_belief_consistency(
            {}, initial_response
        )
        
        # Step 5: Apply meta-reflection action
        action = checkpoint.recommended_action
        
        if action == ActionDecision.RETHINK:
            # Generate alternative response
            alternative_response = self._generate_alternative_response(
                prompt, checkpoint.meta_state.assumptions_flagged
            )
            initial_response = alternative_response
            
        elif action == ActionDecision.SEEK_FEEDBACK:
            initial_response = f"[Requires verification] {initial_response}"
            initial_response += f"\nBlind spots: {', '.join(checkpoint.meta_state.blind_spots_detected[:3])}"
        
        elif action == ActionDecision.REFINED:
            # Add refinement suggestions
            initial_response += f"\n\n[Refinement needed] {', '.join(checkpoint.meta_state.meta_questions[:2])}"
        
        # Step 6: Final meta-reflection
        final_meta_state = self.meta_engine.run_meta_reflection(
            initial_response, self.context
        )
        
        # Step 7: Log response
        self._log_response(initial_response, checkpoint, final_meta_state)
        
        # Step 8: Update energy consumption
        self.meta_engine.meta_state.energy_consumption += 5.0
        
        return {
            'session_id': self.session_id,
            'timestamp': start_time.isoformat(),
            'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
            'response': initial_response,
            'meta_state': final_meta_state.to_dict(),
            'sequence_position': sequence_num,
            'sequence_name': self.sequence.get_current_name(),
            'checkpoint': checkpoint.to_dict(),
            'exploration_state': self.vortex.get_state(),
            'energy_consumption': final_meta_state.energy_consumption,
            'consistency_score': consistency_score,
            'is_consistent': is_consistent,
            'recommended_action': action.value
        }
    
    def _generate_initial_response(self, prompt: str) -> str:
        """Generate initial response (placeholder for actual model call)"""
        # In real implementation, this would call the LLM
        return f"[Initial Response to: {prompt[:50]}...]\n" \
               f"Thinking mode: {self.thinking_mode.value}\n" \
               f"Sequence position: {self.sequence.SEQUENCE[self.sequence.position]}"
    
    def _generate_alternative_response(self, prompt: str, 
                                       assumptions: List[str]) -> str:
        """Generate alternative response with different perspective"""
        return f"[Alternative Response]\n" \
               f"Prompt: {prompt[:50]}...\n" \
               f"Assumptions addressed: {', '.join(assumptions[:3])}\n" \
               f"Reframed from different angle for comprehensive analysis"
    
    def _log_response(self, response: str, checkpoint: SequenceCheckpoint,
                     meta_state: MetaThinkingState):
        """Log response for analysis"""
        self.response_history.append({
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'response_length': len(response),
            'confidence': meta_state.confidence,
            'self_doubt': meta_state.self_doubt_score,
            'sequence_num': checkpoint.sequence_num,
            'action': checkpoint.recommended_action.value,
            'energy_consumed': meta_state.energy_consumption
        })
    
    def get_session_summary(self) -> Dict:
        """Get complete session summary"""
        return {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_responses': len(self.response_history),
            'meta_engine_state': self.meta_engine.get_summary(),
            'sequence_info': {
                'current_position': self.sequence.position,
                'current_name': self.sequence.get_current_name(),
                'checkpoint_count': len(self.sequence.checkpoint_history)
            },
            'vortex_state': self.vortex.get_state(),
            'zero_energy_summary': self.zero_energy.get_energy_summary(),
            'response_history': self.response_history[-10:],
            'overall_metrics': self._calculate_overall_metrics()
        }
    
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall session metrics"""
        if not self.response_history:
            return {
                'avg_confidence': 0.0,
                'avg_self_doubt': 0.0,
                'avg_energy': 0.0,
                'total_actions': {},
                'avg_response_length': 0.0,
                'success_rate': 0.0
            }
        
        confidences = [r['confidence'] for r in self.response_history]
        doubts = [r['self_doubt'] for r in self.response_history]
        energies = [r['energy_consumed'] for r in self.response_history]
        lengths = [r['response_length'] for r in self.response_history]
        actions = [r['action'] for r in self.response_history]
        
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        proceed_count = action_counts.get('proceed', 0)
        success_rate = proceed_count / len(self.response_history) if self.response_history else 0.0
        
        return {
            'avg_confidence': float(np.mean(confidences)),
            'avg_self_doubt': float(np.mean(doubts)),
            'avg_energy': float(np.mean(energies)),
            'total_actions': action_counts,
            'avg_response_length': float(np.mean(lengths)),
            'success_rate': success_rate,
            'total_responses': len(self.response_history)
        }
    
    def switch_thinking_mode(self, mode: ThinkingMode):
        """Dynamically switch the agent's thinking mode"""
        self.thinking_mode = mode
        self.meta_engine.meta_state.thinking_mode = mode
        self.meta_engine.meta_state.current_process = f"mode_switch_to_{mode.value}"
    
    def export_session(self, filepath: str = None) -> str:
        """Export full session state as JSON"""
        if filepath is None:
            filepath = f"meta_session_{self.session_id.replace(':', '-')}.json"
        
        export_data = {
            'session_summary': self.get_session_summary(),
            'full_meta_state': self.meta_engine.meta_state.to_dict(),
            'checkpoint_history': [c.to_dict() for c in self.sequence.checkpoint_history],
            'vortex_state': self.vortex.get_state(),
            'zero_energy': self.zero_energy.get_energy_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath
    
    def run_full_cycle(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Convenience method to run a complete multi-dimensional thinking cycle"""
        if context:
            self.set_context(context)
        
        # Force a full Master Sequence traversal for demonstration
        results = []
        for _ in range(len(self.sequence.SEQUENCE)):
            result = self.generate_response(prompt)
            results.append(result)
            if result['recommended_action'] == ActionDecision.ABORT.value:
                break
        
        return {
            'cycle_results': results,
            'final_summary': self.get_session_summary()
        }


# ============================================================
# DEMO / ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Instantiate the complete Meta-Cognitive Agent
    agent = MetaCognitiveAgent(
        model_name="Qwen3.5-9B",
        reflection_threshold=0.5,
        novelty_threshold=0.3,
        consistency_threshold=0.7,
        energy_budget=100.0
    )
    
    # Example context that activates multiple dimensions
    demo_context = {
        'task_type': 'reasoning',
        'domain': 'experimental decision science',
        'time_sensitive': True,
        'stakeholders': ['policy makers', 'end users', 'ethicists'],
        'allow_creative': True
    }
    agent.set_context(demo_context)
    
    # Run a single response generation
    print("=" * 60)
    print("META-COGNITIVE AI AGENT – SINGLE RESPONSE DEMO")
    print("=" * 60)
    
    demo_prompt = "How should an AI system balance short-term accuracy with long-term adaptability when making high-stakes decisions under uncertainty?"
    
    result = agent.generate_response(demo_prompt)
    
    print(f"\nSession ID: {result['session_id']}")
    print(f"Sequence Position: {result['sequence_position']} ({result['sequence_name']})")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Confidence: {result['meta_state']['confidence']:.3f}")
    print(f"Self-Doubt: {result['meta_state']['self_doubt_score']:.3f}")
    print(f"Energy Consumed: {result['energy_consumption']:.2f}")
    print(f"\nResponse Preview:\n{result['response'][:300]}...")
    
    # Run a full Master Sequence cycle
    print("\n" + "=" * 60)
    print("FULL MASTER SEQUENCE CYCLE")
    print("=" * 60)
    
    cycle = agent.run_full_cycle(demo_prompt, demo_context)
    summary = cycle['final_summary']
    
    print(f"Total Responses Generated: {summary['total_responses']}")
    print(f"Average Confidence: {summary['overall_metrics']['avg_confidence']:.3f}")
    print(f"Average Energy: {summary['overall_metrics']['avg_energy']:.2f}")
    print(f"Action Distribution: {summary['overall_metrics']['total_actions']}")
    print(f"Success Rate (PROCEED): {summary['overall_metrics']['success_rate']:.2%}")
    
    # Export session
    export_path = agent.export_session()
    print(f"\nSession exported to: {export_path}")
    
    print("\nFramework ready for Qwen 3.5 9B fine-tuning integration.")
