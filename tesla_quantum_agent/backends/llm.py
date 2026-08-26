"""Pluggable LLM backends: Transformers, OpenAI-compatible (vLLM), and stub."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBackend(ABC):
    """Abstract generation backend used by the agent and transformer coil."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Returns a dict with at least:
          - content: final response text
          - thinking: optional reasoning content
          - raw: full model output
          - usage: token counts if available
        """

    @abstractmethod
    def get_model_name(self) -> str:
        pass


class StubBackend(LLMBackend):
    """Deterministic local backend so the full pipeline runs without a GPU."""

    def __init__(self, model_name: str = "stub-tesla-resonant"):
        self.model_name = model_name

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content", "")
                break
        thinking = (
            f"Energy-frequency-vibration scan of the query. "
            f"Temperature={temperature:.2f}. Seeking standing-wave collapse."
        )
        content = (
            "Under the Tesla-Quantum lens this query is a standing wave. "
            "Short-term accuracy is a low-voltage local collapse; long-term "
            "adaptability is high-voltage routing when residual vibration and "
            "rare SAE features fire. Resonance caches repeated structure at "
            "zero energy; novelty injects synthetic paths into the vortex. "
            f"Query essence: {user[:240]}"
        )
        if "assumption" in user.lower() or "reframe" in user.lower():
            content = (
                "[Reframed] Challenge the hidden premise that accuracy and "
                "adaptability trade off linearly. Treat them as orthogonal "
                "modes on the same coil: cache the stable band, escalate voltage "
                f"only on residual novelty.\n\nOriginal: {user[:200]}"
            )
        usage = {
            "prompt_tokens": max(1, len(user.split())),
            "completion_tokens": max(1, len(content.split())),
            "total_tokens": max(1, len(user.split()) + len(content.split())),
        }
        return {
            "content": content[: max(64, max_new_tokens * 6)],
            "thinking": thinking if enable_thinking else None,
            "raw": content,
            "usage": usage,
        }

    def get_model_name(self) -> str:
        return self.model_name


class TransformersBackend(LLMBackend):
    """Local Hugging Face Transformers backend (Qwen3.5-9B and friends)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self._torch = torch

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        top_p: float = 0.8,
        top_k: int = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **template_kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                **{k: v for k, v in kwargs.items() if k != "extra_body"},
            )
        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        thinking, content = self._split_thinking(full_text)
        return {
            "content": (content or "").strip(),
            "thinking": thinking.strip() if thinking else None,
            "raw": full_text,
            "usage": {
                "prompt_tokens": int(inputs.input_ids.shape[-1]),
                "completion_tokens": int(len(generated_ids)),
            },
        }

    def _split_thinking(self, text: str) -> tuple:
        if "<think>" in text and "</think>" in text:
            start = text.find("<think>") + len("<think>")
            end = text.find("</think>")
            return text[start:end], text[end + len("</think>") :]
        return None, text

    def get_model_name(self) -> str:
        return self.model_name


class OpenAICompatibleBackend(LLMBackend):
    """vLLM / SGLang / Ollama / DashScope OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        extra_body.update(kwargs.get("extra_body", {}) or {})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        msg = response.choices[0].message
        content = msg.content or ""
        thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "thinking", None)
        usage = {}
        if response.usage:
            dump = getattr(response.usage, "model_dump", None)
            usage = dump() if callable(dump) else dict(response.usage)
        return {
            "content": content,
            "thinking": thinking,
            "raw": content,
            "usage": usage,
        }

    def get_model_name(self) -> str:
        return self.model_name


def build_backend(
    kind: str = "stub",
    model_name: str = "Qwen/Qwen3.5-9B",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> LLMBackend:
    """Construct a backend from config, falling back to stub on import/runtime errors."""
    kind = (kind or "stub").lower()
    if kind in ("stub", "none", "demo"):
        return StubBackend(model_name=f"{model_name} [stub]")
    try:
        if kind in ("transformers", "hf", "local"):
            return TransformersBackend(
                model_name=model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        if kind in ("openai", "vllm", "sglang", "ollama", "compatible"):
            return OpenAICompatibleBackend(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
    except Exception as exc:  # noqa: BLE001 — backend is optional for the demo path
        print(f"[tesla_quantum_agent] LLM backend '{kind}' unavailable ({exc}); using stub.")
    return StubBackend(model_name=f"stub({kind}:{model_name})")
