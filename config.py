#!/usr/bin/env python3
"""
config.py — Architecture constants, default config, and CLI argument parser.
"""

# ─── Architecture Constants (from config.json text_config) ───
HIDDEN_SIZE = 3840
NUM_HEADS = 16          # num_attention_heads
NUM_KV_HEADS = 8        # num_key_value_heads  
HEAD_DIM = 256          # head_dim
INTERMEDIATE_SIZE = 15360
VOCAB_SIZE = 262208
QUERY_PRE_ATTN_SCALAR = 256  # query_pre_attn_scalar from config
SLIDING_WINDOW_PATTERN = 6   # every 6th layer is global attention
SLIDING_WINDOW = 512         # local attention window size
GROUP_SIZE = 64
BITS = 4
LOGITS_TOP_K = 128

# ─── Default Network Config ───
DEFAULT_R1_HOST = "0.0.0.0"
DEFAULT_PORT = 29900
DEFAULT_RANK0_HOST = "localhost"
DEFAULT_MODEL_ID = "mlx-community/gemma-3-12b-it-qat-4bit"

# ─── Shard Config (Gemma3-12B, 48 layers) ───
R0_START_LAYER = 0
R0_END_LAYER = 25   # layers 0-24 (25 layers)
R1_START_LAYER = 25
R1_END_LAYER = 48   # layers 25-47 (23 layers)
N_LAYERS = 48
WORLD_SIZE = 2


import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for sharded_inference."""
    parser = argparse.ArgumentParser(description="Gemma3 sharded autoregressive generation")
    parser.add_argument("--rank", type=int, choices=[0, 1], required=True)
    parser.add_argument("--host", type=str, default=DEFAULT_R1_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rank0-host", type=str, default=DEFAULT_RANK0_HOST,
                        help="Rank 0 IP (for Rank 1 to send logits back)")
    parser.add_argument("--batch-size", type=int, default=1, choices=[1, 2, 4, 8],
                        help="Batch size for bulk processing (default: 1)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = disabled, 1.1-1.3 typical)")
    return parser
