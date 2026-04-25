"""
tcp_transport.py — 纯 TCP tensor 传输层

绕过 MPI/mx.distributed，用 asyncio + TCP 在两台设备间传输 MLX tensor。

协议格式（简单二进制帧）：
  [4 bytes] magic = 0xHPPT (Hippo Pipeline Transport)
  [4 bytes] tensor rank (发送方 rank)
  [4 bytes] ndim (维度数)
  [4 bytes] dtype code (见 DTYPE_MAP)
  [ndim * 4 bytes] shape (每个维度 uint32)
  [body bytes] tensor data (raw bytes)

设计原则：
- 最小依赖（只用 asyncio + struct + socket）
- 支持 Thunderbolt 直连（1ms 延迟）
- 与现有 ShardMetadata / PipelineLayers 解耦
"""

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

# ─── 协议常量 ───────────────────────────────────────────

MAGIC = 0x48505054  # "HPPT"
HEADER_FMT = "!IIII"  # magic, rank, ndim, dtype_code
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 16 bytes

# dtype → code 映射
DTYPE_TO_CODE: dict[mx.Dtype, int] = {
    mx.float32: 1,
    mx.float16: 2,
    mx.bfloat16: 3,
    mx.int32: 4,
    mx.int16: 5,
    mx.int8: 6,
    mx.uint8: 7,
    mx.bool_: 8,
}

CODE_TO_DTYPE: dict[int, mx.Dtype] = {v: k for k, v in DTYPE_TO_CODE.items()}

# dtype → bytes per element
DTYPE_ITEMSIZE: dict[mx.Dtype, int] = {
    mx.float32: 4,
    mx.float16: 2,
    mx.bfloat16: 2,
    mx.int32: 4,
    mx.int16: 2,
    mx.int8: 1,
    mx.uint8: 1,
    mx.bool_: 1,
}


@dataclass
class TensorFrame:
    """一个 tensor 帧"""
    rank: int
    shape: list[int]
    dtype: mx.Dtype
    data: bytes
    send_time: float = 0.0  # 发送方时间戳
    recv_time: float = 0.0  # 接收方时间戳

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def latency_ms(self) -> float:
        """单向延迟（ms），需两台机器时钟同步才准确"""
        if self.send_time > 0 and self.recv_time > 0:
            return (self.recv_time - self.send_time) * 1000
        return -1.0


# ─── 序列化 ─────────────────────────────────────────────

def encode_tensor(tensor: mx.array, rank: int) -> bytes:
    """将 MLX tensor 编码为二进制帧"""
    mx.eval(tensor)  # 确保数据就绪

    dtype = tensor.dtype
    shape = tensor.shape
    ndim = len(shape)
    dtype_code = DTYPE_TO_CODE.get(dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Header: magic + rank + ndim + dtype_code
    header = struct.pack(HEADER_FMT, MAGIC, rank, ndim, dtype_code)

    # Shape: ndim * uint32
    shape_bytes = struct.pack(f"!{ndim}I", *shape)

    # Body: raw tensor data
    import numpy as np
    if tensor.dtype == mx.bfloat16:
        # numpy 不支持 bfloat16 buffer protocol，先转 float32
        body = np.array(tensor.astype(mx.float32)).tobytes()
    else:
        body = np.array(tensor).tobytes()

    return header + shape_bytes + body


def decode_tensor(data: bytes) -> TensorFrame:
    """将二进制帧解码为 TensorFrame"""
    offset = 0

    # Header
    magic, rank, ndim, dtype_code = struct.unpack_from(HEADER_FMT, data, offset)
    offset += HEADER_SIZE

    if magic != MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")

    # Shape
    shape = list(struct.unpack_from(f"!{ndim}I", data, offset))
    offset += ndim * 4

    # Dtype
    dtype = CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    # Body
    body = data[offset:]

    # bfloat16 在 wire 上以 float32 传输（numpy 不支持 bfloat16 buffer）
    wire_itemsize = 4 if dtype == mx.bfloat16 else DTYPE_ITEMSIZE.get(dtype, 4)
    n_elements = 1
    for s in shape:
        n_elements *= s
    expected_size = n_elements * wire_itemsize

    if len(body) != expected_size:
        raise ValueError(
            f"Body size mismatch: got {len(body)}, expected {expected_size}"
        )

    return TensorFrame(rank=rank, shape=shape, dtype=dtype, data=body)


def frame_to_mlx(frame: TensorFrame) -> mx.array:
    """TensorFrame → MLX array"""
    import numpy as np

    if frame.dtype == mx.bfloat16:
        # wire 上是 float32，转回 bfloat16
        arr = np.frombuffer(frame.data, dtype=np.float32).reshape(frame.shape)
        return mx.array(arr).astype(mx.bfloat16)

    np_dtype_map = {
        mx.float32: np.float32,
        mx.float16: np.float16,
        mx.int32: np.int32,
        mx.int16: np.int16,
        mx.int8: np.int8,
        mx.uint8: np.uint8,
        mx.bool_: np.bool_,
    }
    np_dtype = np_dtype_map.get(frame.dtype, np.float32)
    arr = np.frombuffer(frame.data, dtype=np_dtype).reshape(frame.shape)
    return mx.array(arr)


# ─── TCP Server (接收方) ────────────────────────────────

class TensorReceiver:
    """
    TCP 接收服务器。
    
    在指定端口监听，接收 tensor 帧并放入 asyncio.Queue。
    
    用法:
        receiver = TensorReceiver(host="0.0.0.0", port=29500)
        await receiver.start()
        frame = await receiver.recv(timeout=30.0)
        tensor = frame_to_mlx(frame)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 29500,
                 max_queue: int = 10):
        self.host = host
        self.port = port
        self._queue: asyncio.Queue[TensorFrame] = asyncio.Queue(maxsize=max_queue)
        self._server: Optional[asyncio.AbstractServer] = None
        self._stats = {"received": 0, "bytes": 0, "errors": 0}

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle_conn, self.host, self.port
        )
        addr = self._server.sockets[0].getsockname()
        print(f"[TensorReceiver] Listening on {addr[0]}:{addr[1]}")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def recv(self, timeout: float = 30.0) -> TensorFrame:
        """等待接收一个 tensor 帧"""
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """处理一个连接（可能发送多个 tensor）"""
        # TCP_NODELAY on server side too
        sock = writer.transport.get_extra_info('socket')
        if sock is not None:
            import socket as _socket
            sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        print(f"[_handle_conn] New connection accepted at {time.time():.3f}")
        try:
            while True:
                frame = await self._read_frame(reader)
                if frame is None:
                    break
                frame.recv_time = time.time()
                print(f"[_handle_conn] Frame received, putting to queue at {time.time():.3f}")
                await self._queue.put(frame)
                self._stats["received"] += 1
                self._stats["bytes"] += frame.nbytes
                print(f"[_handle_conn] Frame put to queue at {time.time():.3f}")
        except asyncio.IncompleteReadError:
            # sender 正常关闭连接，EOF 不是错误
            pass
        except Exception as e:
            self._stats["errors"] += 1
            print(f"[TensorReceiver] Error: {e}")

    async def _read_frame(self, reader: asyncio.StreamReader) -> Optional[TensorFrame]:
        """从流中读取一个完整的 tensor 帧"""
        _t0 = time.time()
        # 1. Header
        header_data = await reader.readexactly(HEADER_SIZE)
        _t1 = time.time()
        magic, rank, ndim, dtype_code = struct.unpack(HEADER_FMT, header_data)

        if magic != MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")

        # 2. Shape
        shape_data = await reader.readexactly(ndim * 4)
        _t2 = time.time()
        shape = list(struct.unpack(f"!{ndim}I", shape_data))

        # 3. Body size
        dtype = CODE_TO_DTYPE.get(dtype_code, mx.float32)
        n_elements = 1
        for s in shape:
            n_elements *= s
        # bfloat16 在 wire 上以 float32 传输
        wire_itemsize = 4 if dtype == mx.bfloat16 else DTYPE_ITEMSIZE.get(dtype, 4)
        body_size = n_elements * wire_itemsize

        # 4. Body
        body_data = await reader.readexactly(body_size)
        _t3 = time.time()

        print(f"[_read_frame] header={_t1-_t0:.3f}s shape={_t2-_t1:.3f}s body={_t3-_t2:.3f}s total={_t3-_t0:.3f}s body_size={body_size}")

        return TensorFrame(
            rank=rank, shape=shape, dtype=dtype, data=body_data,
        )

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ─── TCP Client (发送方) ────────────────────────────────

class TensorSender:
    """
    TCP 发送客户端。
    
    连接到目标设备，发送 tensor 帧。
    
    用法:
        sender = TensorSender(host="<TARGET_IP>", port=29500)
        await sender.connect()
        await sender.send(tensor, rank=0)
    """

    def __init__(self, host: str, port: int = 29500):
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._stats = {"sent": 0, "bytes": 0, "errors": 0}

    async def connect(self, retries: int = 3, delay: float = 1.0):
        """连接到接收方（带重试）"""
        for i in range(retries):
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self.host, self.port
                )
                # TCP_NODELAY: disable Nagle's algorithm for low-latency sends
                sock = self._writer.transport.get_extra_info('socket')
                if sock is not None:
                    import socket as _socket
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                print(f"[TensorSender] Connected to {self.host}:{self.port} (TCP_NODELAY=1)")
                return
            except (ConnectionRefusedError, OSError) as e:
                if i < retries - 1:
                    print(f"[TensorSender] Retry {i+1}/{retries}: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    async def send(self, tensor: mx.array, rank: int = 0):
        """发送一个 tensor 帧"""
        if self._writer is None:
            raise RuntimeError("Not connected. Call connect() first.")

        _t0 = time.time()
        data = encode_tensor(tensor, rank)
        _t1 = time.time()
        self._writer.write(data)
        await self._writer.drain()
        _t2 = time.time()
        print(f"[_send] encode={_t1-_t0:.3f}s write+drain={_t2-_t1:.3f}s total={_t2-_t0:.3f}s size={len(data)}")
        self._stats["sent"] += 1
        self._stats["bytes"] += len(data)

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ─── Pipeline Transport (双向) ─────────────────────────

class PipelineTransport:
    """
    Pipeline 双向传输层。
    
    每个 rank 既是发送方（发给下一个 rank），也是接收方（从前一个 rank 接收）。
    
    用法:
        transport = PipelineTransport(rank=0, world_size=2, port=29500)
        await transport.start(neighbors={"prev": None, "next": "<NEXT_RANK_IP>"})
        
        # 发送 hidden states 给下一个 rank
        await transport.send_next(hidden_states)
        
        # 从前一个 rank 接收 hidden states
        hidden = await transport.recv_prev()
    """

    def __init__(self, rank: int, world_size: int, port: int = 29500):
        self.rank = rank
        self.world_size = world_size
        self.port = port
        self._receiver: Optional[TensorReceiver] = None
        self._sender_next: Optional[TensorSender] = None
        self._sender_prev: Optional[TensorSender] = None  # 用于反向传播（future）

    async def start(
        self,
        next_host: Optional[str] = None,
        prev_host: Optional[str] = None,
    ):
        """
        启动传输层。
        
        rank 0: 只发不给前一个（没有）
        rank N-1: 只收不发下一个（没有）
        中间: 又收又发
        """
        # 1. 启动接收服务器
        self._receiver = TensorReceiver(host="0.0.0.0", port=self.port + self.rank)
        await self._receiver.start()

        # 2. 连接到下一个 rank（如果存在）
        if next_host and self.rank < self.world_size - 1:
            self._sender_next = TensorSender(
                host=next_host, port=self.port + self.rank + 1
            )
            await self._sender_next.connect()

        # 3. 给一点时间让连接稳定
        await asyncio.sleep(0.1)

    async def send_next(self, tensor: mx.array):
        """发送 hidden states 给下一个 rank"""
        if self._sender_next is None:
            if self.rank == self.world_size - 1:
                return  # 最后一个 rank 不需要发
            raise RuntimeError(f"Rank {self.rank}: no sender to next rank")
        await self._sender_next.send(tensor, rank=self.rank)

    async def recv_prev(self, timeout: float = 30.0) -> mx.array:
        """从前一个 rank 接收 hidden states"""
        if self._receiver is None:
            raise RuntimeError("Receiver not started")
        frame = await self._receiver.recv(timeout=timeout)
        return frame_to_mlx(frame)

    async def stop(self):
        if self._receiver:
            await self._receiver.stop()
        if self._sender_next:
            await self._sender_next.close()

    @property
    def stats(self) -> dict:
        s = {"rank": self.rank}
        if self._receiver:
            s["recv"] = self._receiver.stats
        if self._sender_next:
            s["send"] = self._sender_next.stats
        return s
