#!/usr/bin/env python3
"""
rank0.py — Rank 0 autoregressive generation pipeline.

Handles tokenization, embedding, R0 layer forward pass (layers 0-24),
network communication with Rank 1, and overlapped decode loop.
"""

import asyncio
import time

import mlx.core as mx

from config import (
    HIDDEN_SIZE, DEFAULT_MODEL_ID,
    R0_START_LAYER, R0_END_LAYER, N_LAYERS, WORLD_SIZE,
)
from model_ops import rms_norm, dequantize_weight, forward_layer
from logits import compress_logits_topk, sample_from_topk
from shard import ShardMetadata
from shard_loader import load_shard_weights, load_tokenizer
from tcp_transport import TensorSender, TensorReceiver, frame_to_mlx

__all__ = ["rank0_generate"]


async def rank0_generate(
    target_host: str,
    port: int,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    repetition_penalty: float = 1.0,
):
    """Rank 0: autoregressive generation with KV cache and overlapped decode."""
    print(f"=== Rank 0 自回归生成 ===")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}, temperature: {temperature}")
    print(f"Target: {target_host}:{port}")

    shard = ShardMetadata(
        model_id=DEFAULT_MODEL_ID,
        start_layer=R0_START_LAYER, end_layer=R0_END_LAYER,
        n_layers=N_LAYERS, device_rank=0, world_size=WORLD_SIZE,
    )

    print("\n步骤 1: 加载 Rank 0 权重...")
    t0 = time.time()
    weights = load_shard_weights(shard.model_id, shard)
    for v in weights.values():
        mx.eval(v)
    mx.synchronize()
    print(f"✅ 权重加载完成 ({time.time()-t0:.1f}s), GPU: {mx.get_active_memory()/1024**3:.2f} GB")

    tokenizer = load_tokenizer(shard.model_id)
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"   Input tokens: {len(tokens)}")

    embed_full = dequantize_weight(
        weights['language_model.model.embed_tokens.weight'],
        weights['language_model.model.embed_tokens.scales'],
        weights['language_model.model.embed_tokens.biases'],
    )

    print("\n步骤 2: 连接 Rank 1...")
    sender = TensorSender(target_host, port)
    await sender.connect()
    receiver = TensorReceiver(host='0.0.0.0', port=port+1)
    await receiver.start()
    print("✅ 双向通道建立")

    kv_cache_r0 = {}

    # ── Prefill ──
    print(f"\n步骤 3: Prefill ({len(tokens)} tokens)...")
    t_prefill_start = time.time()

    input_ids = mx.array([tokens])
    hidden_states = embed_full[input_ids] * mx.array(HIDDEN_SIZE**0.5, dtype=mx.float16)

    for i in range(shard.start_layer, shard.end_layer):
        hidden_states = forward_layer(hidden_states, i, weights, kv_cache=kv_cache_r0, offset=0)
    mx.eval(hidden_states)
    mx.synchronize()

    await sender.send(hidden_states, rank=0)

    frame = await receiver.recv(timeout=300)
    recv_data = frame_to_mlx(frame)
    mx.eval(recv_data)

    top_k = recv_data.shape[-1] // 2
    indices = recv_data[:, :, :top_k].astype(mx.int32)
    values = recv_data[:, :, top_k:]

    first_token = sample_from_topk(indices, values, temperature, top_p,
                                    repetition_penalty, None)
    generated_tokens = [first_token]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    t_prefill_end = time.time()
    print(f"✅ Prefill 完成 ({t_prefill_end - t_prefill_start:.1f}s)")
    print(f"   First token: {first_token} → {generated_text!r}")

    # ── Decode loop (Overlapped Pipeline v4) ──
    print(f"\n步骤 4: Decode loop (overlapped v4, max {max_tokens} tokens)...")
    t_decode_start = time.time()
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 1

    spec_hidden = None
    spec_input_token = -1
    spec_task = None

    def _do_forward(input_token, offset):
        inp = mx.array([[input_token]])
        h = embed_full[inp] * mx.array(HIDDEN_SIZE**0.5, dtype=mx.float16)
        for i in range(shard.start_layer, shard.end_layer):
            h = forward_layer(h, i, weights, kv_cache=kv_cache_r0, offset=offset)
        mx.eval(h)
        mx.synchronize()
        return h

    for step in range(1, max_tokens):
        t_step = time.time()

        if spec_task is not None:
            spec_hidden = await spec_task
            spec_task = None

        if spec_hidden is not None and spec_input_token == generated_tokens[-1]:
            hidden_states = spec_hidden
            spec_hidden = None
            t_r0 = 0.0
        else:
            t_r0 = time.time()
            hidden_states = _do_forward(generated_tokens[-1], len(tokens) + step - 1)
            t_r0 = time.time() - t_r0

        await sender.send(hidden_states, rank=0)
        recv_task = asyncio.create_task(receiver.recv(timeout=300))

        frame = await recv_task
        recv_data = frame_to_mlx(frame)
        mx.eval(recv_data)

        top_k_recv = recv_data.shape[-1] // 2
        indices = recv_data[:, :, :top_k_recv].astype(mx.int32)
        values = recv_data[:, :, top_k_recv:]
        token_id = sample_from_topk(indices, values, temperature, top_p,
                                     repetition_penalty, generated_tokens if repetition_penalty > 1.0 else None)
        generated_tokens.append(token_id)

        if step + 1 < max_tokens:
            spec_task = asyncio.create_task(
                asyncio.to_thread(_do_forward, token_id, len(tokens) + step)
            )
            spec_input_token = token_id

        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        elapsed = time.time() - t_decode_start
        tok_per_s = step / elapsed if elapsed > 0 else 0
        print(f"   Step {step}: {token_text!r} | r0={t_r0:.1f}s step={time.time()-t_step:.3f}s | {tok_per_s:.2f} tok/s", flush=True)

        if token_id == eos_token_id:
            print(f"   EOS at step {step}")
            break

    t_decode_end = time.time()

    for closer in [sender.close, receiver.stop]:
        try:
            await asyncio.wait_for(closer(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass

    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    total_time = t_decode_end - t0
    decode_time = t_decode_end - t_decode_start
    decode_tokens = len(generated_tokens) - 1

    print(f"\n{'='*60}")
    print(f"📝 生成完成")
    print(f"   Generated ({len(generated_tokens)} tokens): {full_text!r}")
    print(f"   Prefill: {t_prefill_end - t_prefill_start:.1f}s")
    print(f"   Decode: {decode_time:.1f}s ({decode_tokens} tokens, {decode_tokens/decode_time:.1f} tok/s)")
    print(f"   Total: {total_time:.1f}s")
    print(f"   GPU peak: {mx.get_active_memory()/1024**3:.2f} GB")
    print(f"{'='*60}")

    return full_text
