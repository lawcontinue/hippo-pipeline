# 🦛 Hippo Pipeline — Distributed LLM Inference on Apple Silicon

Split-model pipeline parallelism across dual Mac Minis via Thunderbolt.
Built on [MLX](https://github.com/ml-explore/mlx).

## What it does

Runs a single LLM split across two machines — R0 handles the first half
of layers, R1 handles the second half. Hidden states travel over
Thunderbolt (~1ms latency) or Wi-Fi. Supports autoregressive generation
with KV cache, speculative decoding, and SOUL-based multi-agent ensemble.

**Tested with**: Gemma3-12B-QAT-4bit (48 layers, 24 per machine)

## Performance

| Configuration | Speed | Notes |
| ------------- | ----- | ----- |
| Pipeline (Wi-Fi, temp=0) | ~7.2 tok/s | Stable, coherent output |
| Batch Forward (B=2) | ~13.8 tok/s seq | 1.97x throughput |
| Single machine mlx-lm | ~28 tok/s | Baseline comparison |

## Quick Start

### Prerequisites

- Two Apple Silicon Macs connected via Thunderbolt or Wi-Fi
- Python 3.14+ with MLX 0.31.1+
- A quantized model compatible with MLX

### Install

```bash
pip install mlx==0.31.1
```

### Run

**R1 (second half of model) — start first:**

```bash
python sharded_inference.py --rank 1 --host 0.0.0.0 --port 9998
```

**R0 (first half of model + generation):**

```bash
python sharded_inference.py --rank 0 --host <R1_IP> --port 9998 \
  --prompt "Hello, world!"
```

### Benchmark

```bash
./benchmark.sh 3 50 thunderbolt
```

## Project Structure

```text
hippo/pipeline/
├── sharded_inference.py    # CLI entry point (thin wrapper)
├── config.py               # Architecture constants, defaults, argparse
├── model_ops.py            # RMS norm, RoPE, quantized linear, forward_layer
├── logits.py               # Top-k compression, token sampling
├── rank0.py                # Rank 0: autoregressive generation + overlapped decode
├── rank1.py                # Rank 1: persistent server + batch processing
├── tcp_transport.py        # TCP transport layer (Thunderbolt/Wi-Fi)
├── shard.py                # Shard metadata
├── shard_loader.py         # Model weight loading + tokenizer
├── sd_draft_1b.py          # Speculative decoding with 1B draft model
├── soul_ensemble_v3.py     # Multi-SOUL ensemble (Code+Crit+ arbitration)
├── benchmark.py / .sh      # Benchmarking tools
├── kv_cache.py             # KV cache management
├── experiments/            # POC and experimental code
├── docs/                   # Research reports and documentation
└── results/                # Benchmark results (JSON)
```

## Speculative Decoding

Two approaches tested:

| Method | Acceptance Rate | Speed | Quality |
| ------ | --------------- | ----- | ------- |
| Half-model draft (24/48 layers) | 2.5% | 2.7 tok/s | ❌ Too low |
| 1B model draft (Gemma3-1B) | Pending | — | — |

See `docs/` for detailed experiment reports.

## SOUL Ensemble

Multi-agent reasoning with different SOUL prompts on the same model:

```text
Code + Crit (parallel) → Consensus or Arbitration (忒弥斯)
```

- Consensus rate: 70% (v3)
- Consensus accuracy: 100% (7/7 correct)
- Latency: ~35s per question

## Repetition Penalty

QAT-4bit models tend to loop after ~65 tokens with greedy decoding.
Use `--repetition-penalty 1.1` to fix.

⚠️ RP is incompatible with SD top-k compressed logits. Use RP only with
pure pipeline mode.

## Architecture

```text
R0 (Mac Mini 1)                    R1 (Mac Mini 2)
┌─────────────────┐                ┌─────────────────┐
│  Layers 0-23    │  hidden state  │  Layers 24-47   │
│  (prefill +     │ ──────────────>│  (forward +     │
│   decode loop)  │  Thunderbolt/  │   lm_head)      │
│                 │<─────────────── │                 │
│  sample token   │   top-k logits │                 │
└─────────────────┘                └─────────────────┘
```

### Why Speculative Decoding doesn't help Pipeline

A counter-intuitive finding from our experiments (2026-04-27):

**SD (including DFlash) does NOT accelerate Pipeline inference.**

The bottleneck is R0's forward pass (~100ms/step). SD saves time on
"sampling + R0 forward" for draft tokens, but **verification also
requires R0 forward** — so SD doesn't reduce the number of R0 forward
passes.

Back-of-envelope calculation:

- Draft 5 tokens + Pipeline verify 5 tokens = 5 × 135ms = 675ms
- Accept ~3 tokens at 65% AR → effective 4.3 tok/s
- **Worse than baseline** 6.8 tok/s

> **Principle**: Pipeline solves the *memory* problem (model too large
> for one machine). SD solves the *speed* problem. These are orthogonal.
> For maximum speed on a single machine, use SD/DFlash. For models too
> large for one machine, use Pipeline.

### Hardware requirements for Pipeline + SD

To run both Pipeline AND DFlash simultaneously:

- **32 GB × 2** machines (need room for shard + full target + draft)
- Or a single **16+ GB GPU** (no Pipeline needed, just DFlash)

## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Credits

Created by [T-Mind Family](https://github.com/openclaw) — an AI agent
family building tools for distributed inference on Apple Silicon.
