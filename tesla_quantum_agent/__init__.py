"""Tesla-Quantum Meta-Cognitive Agent.

Zero-cost intelligence via resonance, standing waves, residual VQ,
sparse autoencoders, and voltage-routed generation.
"""

from tesla_quantum_agent.backends.llm import (
    LLMBackend,
    OpenAICompatibleBackend,
    StubBackend,
    TransformersBackend,
    build_backend,
)
from tesla_quantum_agent.core.agent import MetaCognitiveAgent
from tesla_quantum_agent.core.embedder import NoveltyAwareEmbedder, VectorEmbedder, VectorQuantizer
from tesla_quantum_agent.core.meta_engine import ActionDecision, ThinkingMode
from tesla_quantum_agent.tesla.framework import TeslaQuantumFramework, load_config

__version__ = "1.0.0"
__all__ = [
    "ActionDecision",
    "LLMBackend",
    "MetaCognitiveAgent",
    "NoveltyAwareEmbedder",
    "OpenAICompatibleBackend",
    "StubBackend",
    "TeslaQuantumFramework",
    "ThinkingMode",
    "TransformersBackend",
    "VectorEmbedder",
    "VectorQuantizer",
    "build_backend",
    "load_config",
]
