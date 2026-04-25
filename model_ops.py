#!/usr/bin/env python3
"""
model_ops.py — Building blocks for Gemma3 transformer layers.

RMS norm, residual clipping, RoPE, weight dequantization, quantized linear, and
single-layer forward pass.
"""

import mlx.core as mx
import mlx.nn as nn

from config import (
    HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
    QUERY_PRE_ATTN_SCALAR, SLIDING_WINDOW_PATTERN,
    GROUP_SIZE, BITS,
)

__all__ = [
    "rms_norm", "clip_residual", "get_rope",
    "dequantize_weight", "quant_linear", "forward_layer",
]


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Gemma3 RMSNorm: uses mx.fast.rms_norm with (1.0 + weight) gain."""
    return mx.fast.rms_norm(x, 1.0 + weight, eps)


def clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """Clip residual addition to prevent fp16 overflow."""
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)


# ─── RoPE (Rotary Position Embeddings) ──────────────────────
# Gemma3 uses alternating local/global attention:
# - Layers 0-4: local (sliding window, base=10000)
# - Layer 5: global (base=1000000)
# - Repeat pattern every 6 layers

def _init_rope(layer_idx: int):
    """Initialize RoPE for a given layer (local vs global)."""
    is_global = (layer_idx + 1) % SLIDING_WINDOW_PATTERN == 0
    base = 1_000_000.0 if is_global else 10_000.0
    return nn.RoPE(HEAD_DIM, traditional=False, base=base)


_rope_cache: dict = {}


def get_rope(layer_idx: int):
    """Get or create cached RoPE for a layer."""
    if layer_idx not in _rope_cache:
        _rope_cache[layer_idx] = _init_rope(layer_idx)
    return _rope_cache[layer_idx]


def dequantize_weight(weight: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """
    MLX 4-bit affine quantization dequantization.
    
    weight: (out_features, in_features * bits / 8) — packed uint32
    scales: (out_features, groups) 
    biases: (out_features, groups)
    
    Returns: (out_features, in_features) float32
    """
    items_per_int = 32 // BITS  # 8
    out_features = weight.shape[0]
    packed_cols = weight.shape[1]

    w = weight.reshape(out_features, packed_cols, 1)
    shifts = mx.array([i * BITS for i in range(items_per_int)])
    w = (w >> shifts.reshape(1, 1, items_per_int)).reshape(out_features, -1)
    w = (w & 0xF).astype(mx.float32)

    num_groups = scales.shape[1]
    w = w.reshape(out_features, num_groups, -1)
    w = w * scales.reshape(out_features, num_groups, 1) + biases.reshape(out_features, num_groups, 1)
    return w.reshape(out_features, -1)


def quant_linear(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """Quantized linear layer using fused quantized_matmul."""
    return mx.quantized_matmul(
        x,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.scales"],
        weights.get(f"{prefix}.biases"),
        group_size=GROUP_SIZE, bits=BITS,
    )


def forward_layer(
    hidden_states: mx.array,
    layer_idx: int,
    weights: dict,
    kv_cache: dict = None,
    offset: int = 0,
) -> mx.array:
    """
    Forward pass through one Gemma3 transformer layer with KV cache support.
    """
    pfx = f"language_model.model.layers.{layer_idx}"
    use_cache = kv_cache is not None

    # ── Attention Block ──
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{pfx}.input_layernorm.weight"])

    q = quant_linear(hidden_states, weights, f"{pfx}.self_attn.q_proj")
    k = quant_linear(hidden_states, weights, f"{pfx}.self_attn.k_proj")
    v = quant_linear(hidden_states, weights, f"{pfx}.self_attn.v_proj")

    B, S, _ = q.shape
    q = q.reshape(B, S, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    q = rms_norm(q, weights[f"{pfx}.self_attn.q_norm.weight"])
    k = rms_norm(k, weights[f"{pfx}.self_attn.k_norm.weight"])

    rope = get_rope(layer_idx)
    q = rope(q, offset=offset)
    k = rope(k, offset=offset)

    if use_cache and layer_idx in kv_cache:
        k_prev, v_prev = kv_cache[layer_idx]
        k = mx.concatenate([k_prev, k], axis=2)
        v = mx.concatenate([v_prev, v], axis=2)

    if use_cache:
        kv_cache[layer_idx] = (k, v)

    n_rep = NUM_HEADS // NUM_KV_HEADS
    k_expanded = mx.repeat(k, n_rep, axis=1)
    v_expanded = mx.repeat(v, n_rep, axis=1)

    scale = QUERY_PRE_ATTN_SCALAR ** -0.5
    attn = (q @ k_expanded.transpose(0, 1, 3, 2)) * scale
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    attn_out = (attn @ v_expanded).transpose(0, 2, 1, 3).reshape(B, S, NUM_HEADS * HEAD_DIM)

    attn_out = quant_linear(attn_out, weights, f"{pfx}.self_attn.o_proj")

    hidden_states = clip_residual(
        residual,
        rms_norm(attn_out, weights[f"{pfx}.post_attention_layernorm.weight"]),
    )

    # ── MLP Block ──
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{pfx}.pre_feedforward_layernorm.weight"])

    gate = quant_linear(hidden_states, weights, f"{pfx}.mlp.gate_proj")
    up = quant_linear(hidden_states, weights, f"{pfx}.mlp.up_proj")
    down = quant_linear(nn.gelu_approx(gate) * up, weights, f"{pfx}.mlp.down_proj")

    hidden_states = clip_residual(
        residual,
        rms_norm(down, weights[f"{pfx}.post_feedforward_layernorm.weight"]),
    )

    return hidden_states
