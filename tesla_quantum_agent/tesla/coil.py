"""Transformer coil: voltage / capability routing driven by vibration."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tesla_quantum_agent.backends.llm import LLMBackend

VOLTAGE_ENERGY = {
    "resonant-cache": 0.0,
    "12V": 0.01,
    "120V": 0.10,
    "480V": 0.50,
    "high_tension": 1.50,
}

VOLTAGE_GEN_PARAMS = {
    "12V": {"max_new_tokens": 256, "temperature": 0.4},
    "120V": {"max_new_tokens": 1024, "temperature": 0.7},
    "480V": {"max_new_tokens": 2048, "temperature": 0.75},
    "high_tension": {"max_new_tokens": 2048, "temperature": 0.9},
}


class TransformerCoil:
    """Dimension 3 – voltage transformation (capability routing)."""

    def __init__(self, llm_backend: Optional["LLMBackend"] = None):
        self.llm = llm_backend
        self.voltage_map = {
            "12V": "local-0.5B",
            "120V": "local-7B/9B",
            "480V": "haiku-class",
            "high_tension": "sonnet/opus-class",
        }
        self.last_voltage: str = "12V"

    def detect_required_voltage(
        self,
        query: str,
        vibration: float,
        novelty: float,
    ) -> str:
        token_est = len(query.split())
        q = query.lower()
        wants_depth = any(
            w in q.split() for w in ("reason", "reasoning", "novel", "novelty", "prove")
        )
        if token_est < 40 and vibration < 0.35 and novelty < 0.3:
            voltage = "12V"
        elif token_est < 120 and vibration < 0.55:
            voltage = "120V"
        elif novelty > 0.7 or vibration > 0.7 or wants_depth:
            voltage = "high_tension"
        else:
            voltage = "480V"
        self.last_voltage = voltage
        return voltage

    def energy_for(self, voltage: str) -> float:
        return float(VOLTAGE_ENERGY.get(voltage, 0.10))

    def transform(
        self,
        query: str,
        context: Dict[str, Any],
        voltage: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        if self.llm is None:
            return f"[{voltage}] Resonant response to: {query[:80]}..."

        params = VOLTAGE_GEN_PARAMS.get(voltage, VOLTAGE_GEN_PARAMS["120V"])
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif context.get("system"):
            messages.append({"role": "system", "content": str(context["system"])})
        messages.append({"role": "user", "content": query})
        result = self.llm.generate(
            messages,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            enable_thinking=voltage != "12V",
        )
        return result.get("content") or f"[{voltage}] empty generation"
