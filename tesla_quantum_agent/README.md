# Tesla-Quantum Meta-Cognitive Agent

Zero-cost intelligence via **resonance**, **standing waves**, **residual vector quantization**, **sparse autoencoders**, and **voltage-routed generation**.

Computation collapses only when resonance fails. Repeated queries are standing-wave retrieval (energy = 0). Novel queries are scored by VQ residual energy + SAE rare-feature hits; that combined vibration drives 12V / 120V / 480V / high-tension routing, meta-cognitive reflection, and synthetic novelty injection.

## Layout

```
tesla_quantum_agent/
├── core/          Meta engine, Master Sequence, Vortex, Zero-Energy, embedder+VQ, SAE, synth, agent
├── tesla/         Perception, memory, coil, oscillator, field, damper, evaluator, manifest, framework
├── backends/      TransformersBackend, OpenAICompatibleBackend, StubBackend
├── config.yaml
├── main.py
└── requirements.txt
```

## Install

```bash
cd Tesla-Quantum-Meta-Cognitive-Agent
python -m venv .venv
source .venv/bin/activate
pip install -r tesla_quantum_agent/requirements.txt
```

The demo runs with **NumPy + PyYAML only**. Optional extras:

```bash
pip install torch sentence-transformers transformers openai scikit-learn lancedb
```

## Run the demo (stub backend, no GPU)

From the repository root:

```bash
python -m tesla_quantum_agent.main
```

or:

```bash
python tesla_quantum_agent/main.py
```

You should see a **cache miss** on the first query (voltage, vibration, energy, manifestation) and a **resonance hit with energy_cost = 0** on the repeat.

## Point at a local Qwen3.5-9B / vLLM server

Start vLLM (nightly if needed for full Qwen3.5 support):

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.9
```

Then:

```bash
python -m tesla_quantum_agent.main \
  --backend vllm \
  --model Qwen/Qwen3.5-9B \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY
```

Ollama (if the model is pulled):

```bash
python -m tesla_quantum_agent.main \
  --backend ollama \
  --model qwen3.5:9b \
  --base-url http://localhost:11434/v1
```

Local Transformers (needs ~19 GB BF16 or a 4-bit checkpoint):

```bash
python -m tesla_quantum_agent.main --backend transformers --model Qwen/Qwen3.5-9B
```

## Config

Edit `tesla_quantum_agent/config.yaml` or pass `--config path/to/config.yaml`.

| Key | Meaning |
|-----|---------|
| `model.backend` | `stub` / `transformers` / `openai` / `vllm` / `ollama` |
| `embedder.n_codes` / `n_residuals` | Residual VQ lattice |
| `embedder.sae_dict_size` / `sae_k` | Sparse autoencoder dictionary |
| `perception.resonance_threshold` | Cosine / LSH cache hit threshold |
| `synth.target_vibration` | Novelty gate for synthetic injection |

## Python API

```python
from tesla_quantum_agent import TeslaQuantumFramework

fw = TeslaQuantumFramework.from_config()
out = fw.process(
    "How do we achieve zero-cost high-novelty intelligence?",
    user_id="researcher",
    profile="technical",
    output="markdown",
)
print(out["resonance_hit"], out["voltage"], out["combined_vibration"], out["energy_cost"])
print(out["result"])
```

## Pipeline

```
Query
  → TeslaPerceptionLayer (exact / code / LSH / LanceDB cache)
  → miss: VectorEmbedder → Residual VQ → Sparse Autoencoder
  → combined vibration = 0.5·VQ + 0.5·SAE(rare features + residual)
  → TransformerCoil voltage routing
  → MetaCognitiveAgent (Master Sequence, Vortex, Zero-Energy Lens)
  → SyntheticDataGenerator if vibration high or rare features fire
  → StandingWaveMemory · TemporalOscillator · FieldEffect · UncertaintyDamper
  → QuantumEvaluator → ManifestationEngine
```

Cache hits skip the rest. Energy cost is **0**.
