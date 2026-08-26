#!/usr/bin/env python3
"""CLI / demo entry point for the Tesla-Quantum Meta-Cognitive Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_path() -> None:
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_banner() -> None:
    print("=" * 72)
    print("  TESLA-QUANTUM META-COGNITIVE AGENT")
    print("  Resonance · Voltage routing · VQ + SAE vibration · Manifestation")
    print("=" * 72)


def _print_result(title: str, result: Dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    print(f"  Resonance hit : {result['resonance_hit']}  ({result['resonance_source']})")
    print(f"  Voltage used  : {result['voltage']}")
    print(f"  Vibration     : {result['vibration']:.4f}")
    print(f"  Combined vib. : {result['combined_vibration']:.4f}")
    print(f"  Residual E    : {result['residual_energy']:.4f}")
    print(f"  Energy cost   : {result['energy_cost']:.4f}")
    sae = result.get("sae_novelty") or {}
    if sae:
        print(
            f"  SAE           : residual={sae.get('residual_energy', 0):.4f}  "
            f"rare_hits={sae.get('rare_feature_hits', 0):.0f}  "
            f"sparsity={sae.get('sparsity', 0):.3f}"
        )
    print(f"  Trigger synth : {result['trigger_synth']}")
    print(f"  Coherence     : {result['coherence']:.4f}")
    meta = result.get("meta") or {}
    if meta:
        print(
            f"  Meta          : seq={meta.get('sequence_position')} "
            f"({meta.get('sequence_name')})  action={meta.get('recommended_action')}  "
            f"confidence={meta.get('confidence')}"
        )
    manifestation = result.get("result")
    if isinstance(manifestation, dict):
        core = manifestation.get("core", "")
        preview = core if isinstance(core, str) else json.dumps(manifestation, default=str)
    else:
        preview = str(manifestation)
    print(f"  Manifestation : {preview[:360]}{'…' if len(preview) > 360 else ''}")


def build_framework(args: argparse.Namespace):
    from tesla_quantum_agent.backends.llm import build_backend
    from tesla_quantum_agent.tesla.framework import TeslaQuantumFramework, load_config

    cfg = load_config(args.config)
    if args.backend:
        cfg.setdefault("model", {})["backend"] = args.backend
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.base_url:
        cfg.setdefault("model", {})["base_url"] = args.base_url
    if args.api_key:
        cfg.setdefault("model", {})["api_key"] = args.api_key

    backend = None
    if args.backend or args.model or args.base_url:
        m = cfg.get("model", {})
        backend = build_backend(
            kind=m.get("backend", "stub"),
            model_name=m.get("name", "Qwen/Qwen3.5-9B"),
            base_url=m.get("base_url", "http://localhost:8000/v1"),
            api_key=m.get("api_key", "EMPTY"),
        )
    return TeslaQuantumFramework(llm_backend=backend, config=cfg, config_path=args.config)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tesla-Quantum Meta-Cognitive Agent demo / CLI",
    )
    p.add_argument(
        "--query",
        default=(
            "How should an AI system balance short-term accuracy with long-term "
            "adaptability when making high-stakes decisions under uncertainty?"
        ),
        help="Query to process",
    )
    p.add_argument("--profile", default="technical", help="Field-effect profile")
    p.add_argument(
        "--output",
        default="markdown",
        choices=["json", "markdown", "api", "text"],
        help="Manifestation format",
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument(
        "--backend",
        default=None,
        choices=["stub", "transformers", "openai", "vllm", "ollama"],
        help="LLM backend (default: config or stub)",
    )
    p.add_argument("--model", default=None, help="Model name, e.g. Qwen/Qwen3.5-9B")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default=None, help="API key (EMPTY for local vLLM)")
    p.add_argument("--json", action="store_true", help="Dump full result dicts as JSON")
    p.add_argument("--skip-repeat", action="store_true", help="Do not re-run the query for cache demo")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    _ensure_path()
    args = parse_args(argv)
    _print_banner()

    framework = build_framework(args)
    print(f"\nBackend : {framework.llm.get_model_name()}")
    print(f"Profile : {args.profile}   Output : {args.output}")

    context = {
        "task_type": "reasoning",
        "domain": "experimental decision science",
        "time_sensitive": True,
        "stakeholders": ["policy makers", "end users", "ethicists"],
        "allow_creative": True,
    }

    first = framework.process(
        args.query,
        user_id="demo",
        profile=args.profile,
        output=args.output,
        context=context,
    )
    _print_result("FIRST PASS (expected cache miss)", first)

    second = None
    if not args.skip_repeat:
        second = framework.process(
            args.query,
            user_id="demo",
            profile=args.profile,
            output=args.output,
            context=context,
        )
        _print_result("REPEAT PASS (expected zero-energy resonance hit)", second)

    print("\n--- SESSION STATS ---")
    stats = framework.stats()
    print(
        f"  Queries={stats['queries']}  hits={stats['resonance_hits']}  "
        f"hit_rate={stats['hit_rate']:.0%}  avg_energy={stats['avg_energy']:.4f}"
    )

    if args.json:
        payload = {"first": first, "second": second, "stats": stats}
        print("\n" + json.dumps(payload, indent=2, default=str))

    print("\nDone. Cache hits cost 0 energy. Point --backend vllm at a local Qwen3.5-9B server to generate for real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
