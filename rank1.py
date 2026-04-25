#!/usr/bin/env python3
"""
rank1.py — Rank 1 persistent inference server.

Receives hidden states from Rank 0, runs R1 layer forward pass (layers 25-47),
computes lm_head + top-k compression, and sends logits back.
Supports batch processing and graceful disconnect recovery.
"""

import asyncio
import time

import mlx.core as mx

from config import (
    GROUP_SIZE, BITS, DEFAULT_MODEL_ID, DEFAULT_RANK0_HOST,
    R1_START_LAYER, R1_END_LAYER, N_LAYERS, WORLD_SIZE,
)
from model_ops import rms_norm, forward_layer
from logits import compress_logits_topk
from shard import ShardMetadata
from shard_loader import load_shard_weights
from tcp_transport import TensorSender, TensorReceiver, frame_to_mlx

__all__ = ["rank1_serve"]


async def rank1_serve(bind_host: str, port: int, rank0_host: str = DEFAULT_RANK0_HOST, batch_size: int = 1):
    """Rank 1: persistent server with overlapped pipeline and batch processing."""
    print(f"=== Rank 1 推理服务 (batch_size={batch_size}) ===")
    print(f"监听: {bind_host}:{port}")

    shard = ShardMetadata(
        model_id=DEFAULT_MODEL_ID,
        start_layer=R1_START_LAYER, end_layer=R1_END_LAYER,
        n_layers=N_LAYERS, device_rank=1, world_size=WORLD_SIZE,
    )

    print("\n步骤 1: 加载 Rank 1 权重...")
    t0 = time.time()
    weights = load_shard_weights(shard.model_id, shard)
    for v in weights.values():
        mx.eval(v)
    mx.synchronize()
    print(f"✅ 权重加载完成 ({time.time()-t0:.1f}s), GPU: {mx.get_active_memory()/1024**3:.2f} GB")

    lm_w = weights['language_model.lm_head.weight']
    lm_s = weights['language_model.lm_head.scales']
    lm_b = weights['language_model.lm_head.biases']
    norm_w = weights['language_model.model.norm.weight']

    kv_cache_r1 = {}

    print(f"\n步骤 3: 启动接收器...")
    receiver = TensorReceiver(host=bind_host, port=port)
    await receiver.start()
    print(f"✅ 等待 Rank 0 的 hidden states...")

    sender = None

    # ── Internal helpers (captured in closure for speed) ──

    def _do_forward(hidden_states, offset):
        h = hidden_states
        for i in range(shard.start_layer, shard.end_layer):
            h = forward_layer(h, i, weights, kv_cache=kv_cache_r1, offset=offset)
        mx.eval(h)
        mx.synchronize()
        return h

    def _do_lm_head(hidden_states):
        last_hidden = hidden_states[:, -1:, :]
        last_hidden = rms_norm(last_hidden, norm_w)
        logits = mx.quantized_matmul(last_hidden, lm_w, lm_s, lm_b, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(logits)
        top_idx, top_val = compress_logits_topk(logits)
        packed = mx.concatenate([top_idx.astype(mx.float32), top_val.astype(mx.float32)], axis=-1)
        mx.eval(packed)
        return packed

    def _do_forward_and_lm(hidden_states, offset):
        """Fused: forward + norm + lm_head + top-k in one eval pass."""
        h = hidden_states
        for i in range(shard.start_layer, shard.end_layer):
            h = forward_layer(h, i, weights, kv_cache=kv_cache_r1, offset=offset)
        last_hidden = h[:, -1:, :]
        last_hidden = rms_norm(last_hidden, norm_w)
        logits = mx.quantized_matmul(last_hidden, lm_w, lm_s, lm_b, group_size=GROUP_SIZE, bits=BITS)
        top_idx, top_val = compress_logits_topk(logits)
        packed = mx.concatenate([top_idx.astype(mx.float32), top_val.astype(mx.float32)], axis=-1)
        mx.eval(packed)
        mx.synchronize()
        return packed

    def _do_forward_batch(batch_hidden, start_offset):
        bs = batch_hidden.shape[1]
        results = []
        for i in range(bs):
            hi = batch_hidden[:, i:i+1, :]
            for layer_idx in range(shard.start_layer, shard.end_layer):
                hi = forward_layer(hi, layer_idx, weights, kv_cache=kv_cache_r1, offset=start_offset + i)
            mx.eval(hi)
            results.append(hi)
        h = mx.concatenate(results, axis=1)
        mx.synchronize()
        return h

    def _do_lm_head_batch(batch_hidden):
        last_hidden = rms_norm(batch_hidden, norm_w)
        logits = mx.quantized_matmul(last_hidden, lm_w, lm_s, lm_b, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(logits)
        all_idx, all_val = [], []
        for i in range(batch_hidden.shape[1]):
            ti, tv = compress_logits_topk(logits[:, i:i+1, :])
            all_idx.append(ti)
            all_val.append(tv)
        packed = mx.concatenate([
            mx.concatenate(all_idx, axis=1).astype(mx.float32),
            mx.concatenate(all_val, axis=1).astype(mx.float32),
        ], axis=-1)
        mx.eval(packed)
        return packed

    # ── Main loop ──

    step = 0
    pending_recv_task = asyncio.create_task(receiver.recv(timeout=300))
    batch_buffer = []
    batch_offsets = []
    batch_count = 0

    while True:
        try:
            t_step = time.time()

            frame = await pending_recv_task
            if frame is None:
                print("⚠️ 接收超时，退出")
                break

            hidden_states = frame_to_mlx(frame)
            mx.eval(hidden_states)
            t_recv = time.time() - t_step

            if step == 0:
                current_offset = 0
            elif kv_cache_r1:
                current_offset = next(iter(kv_cache_r1.values()))[0].shape[2]
            else:
                current_offset = 0

            # ── Batch logic ──
            if batch_size > 1 and step > 0:
                batch_buffer.append(hidden_states)
                batch_offsets.append(current_offset)
                batch_count += 1

                if batch_count < batch_size:
                    pending_recv_task = asyncio.create_task(receiver.recv(timeout=30))
                    step += 1
                    continue

                t_batch_start = time.time()
                batch_hidden = mx.concatenate(batch_buffer, axis=1)

                t_fwd = time.time()
                batch_hidden = await asyncio.to_thread(_do_forward_batch, batch_hidden, batch_offsets[0])
                t_fwd = time.time() - t_fwd

                t_lm = time.time()
                last_logits = await asyncio.to_thread(_do_lm_head_batch, batch_hidden)
                t_lm = time.time() - t_lm

                batch_buffer.clear()
                batch_offsets.clear()
                batch_count = 0

                if step <= 3 or (step // batch_size) % 10 == 0:
                    t_total = time.time() - t_batch_start
                    print(f"   Batch {step // batch_size + 1}: size={batch_size} fwd={t_fwd:.3f}s lm={t_lm:.3f}s total={t_total:.3f}s", flush=True)
            else:
                t_fwd = time.time()
                last_logits = await asyncio.to_thread(_do_forward_and_lm, hidden_states, current_offset)
                t_fwd = time.time() - t_fwd
                t_lm = 0.0

            # Lazy-connect sender to Rank 0
            if sender is None:
                print(f"   连接 Rank 0 ({rank0_host}:{port+1})...", flush=True)
                for retry in range(5):
                    try:
                        sender = TensorSender(rank0_host, port+1)
                        await sender.connect()
                        print(f"✅ Rank 0 连接建立 (attempt {retry+1})")
                        break
                    except Exception as e:
                        if retry < 4:
                            await asyncio.sleep(2)
                        else:
                            raise

            t_send = time.time()
            await sender.send(last_logits, rank=1)
            t_send = time.time() - t_send

            pending_recv_task = asyncio.create_task(receiver.recv(timeout=30))

            step += 1
            if step <= 3 or step % 10 == 0:
                t_total = time.time() - t_step
                print(f"   Step {step}: step={t_total:.3f}s | recv={t_recv:.3f}s fwd={t_fwd:.3f}s send={t_send:.3f}s | {1/t_total:.2f} tok/s", flush=True)

        except (BrokenPipeError, ConnectionResetError, OSError, asyncio.TimeoutError) as e:
            print(f"⚠️ 连接断开: {e} — 等待新连接...", flush=True)
            if sender:
                try:
                    await asyncio.wait_for(sender.close(), timeout=3.0)
                except:
                    pass
                sender = None
            try:
                await asyncio.wait_for(receiver.stop(), timeout=3.0)
            except:
                pass
            receiver = TensorReceiver(host=bind_host, port=port)
            await receiver.start()
            print(f"✅ 等待新 Rank 0 连接...", flush=True)
            kv_cache_r1.clear()
            step = 0
            pending_recv_task = asyncio.create_task(receiver.recv(timeout=30))
            continue
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            break

    if sender:
        try:
            await sender.close()
        except:
            pass
    await receiver.stop()
    print(f"\n=== Rank 1 服务结束 ({step} steps) ===")
