#!/usr/bin/env python3
"""
logits.py — Top-k logits compression/decompression and token sampling.

Compresses 262K logits to top-K values for network transfer,
and samples tokens from compressed or full logits.
"""

import mlx.core as mx

from config import VOCAB_SIZE, LOGITS_TOP_K

__all__ = [
    "compress_logits_topk", "decompress_logits_topk",
    "sample_from_topk", "sample_token",
]


def compress_logits_topk(logits: mx.array, top_k: int = LOGITS_TOP_K) -> tuple:
    """Compress logits by keeping only top-K values. Returns (indices, values)."""
    logits_1d = logits[0, -1, :].astype(mx.float32)
    top_idx = mx.argsort(-logits_1d)[:top_k]
    top_vals = logits_1d[top_idx]
    return top_idx.reshape(1, 1, top_k), top_vals.reshape(1, 1, top_k).astype(mx.float16)


def decompress_logits_topk(indices: mx.array, values: mx.array, vocab_size: int = VOCAB_SIZE) -> mx.array:
    """Reconstruct full logits from top-K sparse representation (non-top-K = -inf)."""
    full = mx.full((1, 1, vocab_size), -1e9, dtype=mx.float16)
    return full


def sample_from_topk(indices: mx.array, values: mx.array,
                     temperature: float = 0.0, top_p: float = 1.0,
                     repetition_penalty: float = 1.0,
                     recent_tokens: list = None) -> int:
    """Sample directly from top-K logits without reconstructing full vocab."""
    vals = values[0, 0].astype(mx.float32)
    idxs = indices[0, 0]

    if recent_tokens and repetition_penalty > 1.0:
        recent_set = set(recent_tokens[-64:])
        for i in range(len(idxs)):
            if int(idxs[i]) in recent_set:
                vals[i] = vals[i] / repetition_penalty

    if temperature == 0.0:
        best = int(mx.argmax(vals))
        return int(idxs[best])

    vals = vals / temperature
    if top_p < 1.0:
        probs = mx.softmax(vals)
        sorted_i = mx.argsort(-probs)
        sorted_p = probs[sorted_i]
        cumsum = mx.cumsum(sorted_p)
        mask = cumsum - sorted_p < top_p
        filtered = sorted_p * mask
        filtered = filtered / mx.sum(filtered)
        pick = int(mx.random.categorical(mx.log(filtered + 1e-10)))
        return int(idxs[sorted_i[pick]])
    else:
        pick = int(mx.random.categorical(vals))
        return int(idxs[pick])


def sample_token(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """Sample next token from full logits. temperature=0 → greedy."""
    logits_1d = logits[0, -1, :]

    if temperature == 0.0:
        return int(mx.argmax(logits_1d))

    logits_1d = logits_1d / temperature
    if top_p < 1.0:
        probs = mx.softmax(logits_1d.astype(mx.float32))
        sorted_indices = mx.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumsum = mx.cumsum(sorted_probs)
        mask = cumsum - sorted_probs < top_p
        probs_filtered = sorted_probs * mask
        probs_filtered = probs_filtered / mx.sum(probs_filtered)
        idx = mx.random.categorical(mx.log(probs_filtered + 1e-10))
        return int(sorted_indices[idx])
    else:
        return int(mx.random.categorical(logits_1d))
