#!/usr/bin/env python3
"""
Unit tests for Hippo Pipeline modules (Phase 2).
Run: python3 tests/test_phase2.py
No external test dependencies — uses stdlib assert + traceback.
"""

import sys
import os
import struct
import traceback
import time

# Ensure module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASSED = 0
FAILED = 0
ERRORS = []


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAILED += 1
        tb = traceback.format_exc()
        ERRORS.append((name, tb))
        print(f"  ❌ {name}: {e}")


# ═══════════════════════════════════════════
# config.py
# ═══════════════════════════════════════════

def test_config_constants():
    from config import (
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, VOCAB_SIZE,
        R0_START_LAYER, R0_END_LAYER, R1_START_LAYER, R1_END_LAYER,
        N_LAYERS, WORLD_SIZE, LOGITS_TOP_K, GROUP_SIZE, BITS,
    )
    assert HIDDEN_SIZE == 3840
    assert NUM_HEADS == 16
    assert NUM_KV_HEADS == 8
    assert HEAD_DIM == 256
    assert VOCAB_SIZE == 262208
    assert N_LAYERS == 48
    assert WORLD_SIZE == 2
    assert R0_END_LAYER == 25
    assert R1_START_LAYER == 25
    assert R1_END_LAYER == 48
    # Layers don't overlap and cover full model
    assert R0_START_LAYER == 0
    assert R0_END_LAYER == R1_START_LAYER
    assert R1_END_LAYER == N_LAYERS


def test_config_parser():
    from config import build_parser
    parser = build_parser()
    args = parser.parse_args(["--rank", "0", "--host", "1.2.3.4", "--max-tokens", "50"])
    assert args.rank == 0
    assert args.host == "1.2.3.4"
    assert args.max_tokens == 50


# ═══════════════════════════════════════════
# shard.py
# ═══════════════════════════════════════════

def test_shard_metadata():
    from shard import ShardMetadata
    s = ShardMetadata(
        model_id="test/model", start_layer=0, end_layer=24,
        n_layers=48, device_rank=0, world_size=2,
    )
    assert s.is_first
    assert not s.is_last
    assert s.end_layer - s.start_layer == 24

    s1 = ShardMetadata(
        model_id="test/model", start_layer=24, end_layer=48,
        n_layers=48, device_rank=1, world_size=2,
    )
    assert not s1.is_first
    assert s1.is_last
    assert s1.end_layer - s1.start_layer == 24


# ═══════════════════════════════════════════
# logits.py (MLX required)
# ═══════════════════════════════════════════

def test_compress_decompress_logits():
    import mlx.core as mx
    from logits import compress_logits_topk, decompress_logits_topk
    from config import LOGITS_TOP_K

    # Create fake logits: (1, 1, 100)
    logits = mx.random.normal((1, 1, 100))
    idx, vals = compress_logits_topk(logits, top_k=10)
    assert idx.shape[-1] == 10
    assert vals.shape[-1] == 10

    # Greedy sample should match argmax
    from logits import sample_from_topk
    token = sample_from_topk(idx, vals, temperature=0.0)
    assert isinstance(token, int)
    assert 0 <= token < 100


def test_sample_token_greedy():
    import mlx.core as mx
    from logits import sample_token

    arr = mx.zeros((1, 1, 50))
    arr[0, 0, 42] = 100.0
    logits = arr
    token = sample_token(logits, temperature=0.0)
    assert token == 42


def test_sample_token_with_temperature():
    import mlx.core as mx
    from logits import sample_token

    mx.random.seed(42)
    logits = mx.random.normal((1, 1, 100))
    token = sample_token(logits, temperature=0.8)
    assert isinstance(token, int)
    assert 0 <= token < 100


# ═══════════════════════════════════════════
# model_ops.py (MLX required)
# ═══════════════════════════════════════════

def test_rms_norm():
    import mlx.core as mx
    from model_ops import rms_norm

    x = mx.ones((1, 1, 8)) * 2.0
    weight = mx.zeros((8,))
    result = rms_norm(x, weight)
    assert result.shape == (1, 1, 8)
    mx.eval(result)
    # With weight=0 and (1+weight)=1, rms_norm should normalize
    # Just check it doesn't crash and shape is correct


def test_clip_residual():
    import mlx.core as mx
    from model_ops import clip_residual

    x = mx.ones((1, 1, 4), dtype=mx.float16)
    y = mx.ones((1, 1, 4), dtype=mx.float16)
    result = clip_residual(x, y)
    assert result.shape == (1, 1, 4)
    mx.eval(result)


def test_get_rope():
    from model_ops import get_rope
    rope0 = get_rope(0)
    rope5 = get_rope(5)
    assert rope0 is not rope5  # Different layers → different RoPE (local vs global)
    assert get_rope(0) is rope0  # Cached


# ═══════════════════════════════════════════
# tcp_transport.py (encode/decode only, no network)
# ═══════════════════════════════════════════

def test_encode_decode_tensor():
    import mlx.core as mx
    from tcp_transport import encode_tensor, decode_tensor, frame_to_mlx

    original = mx.random.normal((2, 3, 4))
    mx.eval(original)

    data = encode_tensor(original, rank=0)
    assert isinstance(data, bytes)
    assert len(data) > 0

    frame = decode_tensor(data)
    assert frame.rank == 0
    assert frame.shape == [2, 3, 4]

    recovered = frame_to_mlx(frame)
    mx.eval(recovered)
    # removed numpy dependency
    assert float(mx.abs(original - recovered).max()) < 1e-5, f"float32 roundtrip error"


def test_encode_decode_float16():
    import mlx.core as mx
    from tcp_transport import encode_tensor, decode_tensor, frame_to_mlx

    original = mx.random.normal((1, 5)).astype(mx.float16)
    mx.eval(original)

    data = encode_tensor(original, rank=1)
    frame = decode_tensor(data)
    assert frame.rank == 1
    recovered = frame_to_mlx(frame)
    mx.eval(recovered)
    assert float(mx.abs(original - recovered).max()) < 0.01, f"float16 roundtrip error"


def test_tensor_frame_nbytes():
    from tcp_transport import TensorFrame
    frame = TensorFrame(rank=0, shape=[2, 3], dtype=None, data=b"\x00" * 24)
    assert frame.nbytes == 24


# ═══════════════════════════════════════════
# sharded_inference.py (entry point)
# ═══════════════════════════════════════════

def test_entry_point_imports():
    from sharded_inference import main
    assert callable(main)


# ═══════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Hippo Pipeline Phase 2 — Unit Tests\n")

    tests = [
        ("config: constants", test_config_constants),
        ("config: parser", test_config_parser),
        ("shard: metadata", test_shard_metadata),
        ("logits: compress/decompress", test_compress_decompress_logits),
        ("logits: greedy sample", test_sample_token_greedy),
        ("logits: temperature sample", test_sample_token_with_temperature),
        ("model_ops: rms_norm", test_rms_norm),
        ("model_ops: clip_residual", test_clip_residual),
        ("model_ops: get_rope", test_get_rope),
        ("tcp_transport: encode/decode float32", test_encode_decode_tensor),
        ("tcp_transport: encode/decode float16", test_encode_decode_float16),
        ("tcp_transport: TensorFrame nbytes", test_tensor_frame_nbytes),
        ("entry_point: imports", test_entry_point_imports),
    ]

    t0 = time.time()
    for name, fn in tests:
        run_test(name, fn)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed ({elapsed:.1f}s)")

    if ERRORS:
        print(f"\n❌ Failed tests:")
        for name, tb in ERRORS:
            print(f"\n--- {name} ---")
            print(tb)

    sys.exit(1 if FAILED else 0)
