"""
45P Ghost State Memory — TurboQuant-style vector compression.

Implements a simplified TurboQuant pipeline without third-party quantizers:

  * **Keys (first dim//2 components):** L2-normalize to the unit sphere, then
    apply **PolarQuant-style** discretization: each coordinate is a projection
    onto an orthonormal axis of the tangent box [-1, 1]^d, uniformly quantized.
    At high dimension this approximates fine angular cells on S^{d-1} with
    bounded error; full hyperspherical angle chains (theta_1..theta_{d-1}) are
    documented below for reference:

      u_1 = cos(phi_1)
      u_k = (prod_{j<k} sin(phi_j)) * cos(phi_k),  k < d
      u_d = (prod_{j<d} sin(phi_j)) * sin(phi_{d-1})

    with phi_k in [0, pi] (last latitude) / [0, 2pi] (azimuth). We avoid storing
    191 angles at 3 bits each because quantization noise accumulates; instead
    we quantize normalized coordinates directly (common in ANN codebooks).

  * **Values (remaining components):** per-vector min/max scalar quantization
    with configurable bits (default 3), linearly mapped to uint8 lanes.

Packed as raw bytes for SQLite-friendly persistence.
"""

from __future__ import annotations

import struct

import numpy as np

_MAGIC = b"GH45\x02"


def _quantize_uniform(x: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    """Map x linearly from [lo, hi] to integers 0 .. 2^bits-1."""
    levels = (1 << bits) - 1
    t = (np.clip(x.astype(np.float64), lo, hi) - lo) / (hi - lo + 1e-12)
    q = np.round(t * levels).astype(np.int32)
    return np.clip(q, 0, levels).astype(np.uint8)


def _dequantize_uniform(q: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    """Inverse of _quantize_uniform."""
    levels = (1 << bits) - 1
    t = q.astype(np.float64) / float(levels)
    return lo + t * (hi - lo)


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Pack small integers (0..2^bits-1) MSB-first per value into bytes."""
    if values.size == 0:
        return b""
    n = int(values.shape[0])
    total_bits = n * bits
    out = bytearray((total_bits + 7) // 8)
    bit_pos = 0
    for i in range(n):
        v = int(values[i]) & ((1 << bits) - 1)
        for b in range(bits - 1, -1, -1):
            if (v >> b) & 1:
                byte_i = bit_pos // 8
                bit_in_byte = 7 - (bit_pos % 8)
                out[byte_i] |= 1 << bit_in_byte
            bit_pos += 1
    return bytes(out)


def _unpack_bits(data: bytes, n: int, bits: int) -> np.ndarray:
    """Unpack n values of `bits` bits from a bit stream."""
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    out = np.zeros(n, dtype=np.uint8)
    bit_pos = 0
    for i in range(n):
        v = 0
        for _ in range(bits):
            byte_i = bit_pos // 8
            bit_in_byte = 7 - (bit_pos % 8)
            bit = (data[byte_i] >> bit_in_byte) & 1
            v = (v << 1) | bit
            bit_pos += 1
        out[i] = v
    return out


def compress_vector(
    vec: np.ndarray,
    bits: int = 3,
    *,
    key_bits: int = 5,
) -> bytes:
    """
    Compress a 1-D float32 embedding into a packed byte string.

    Splits the vector into key/value halves. Keys are L2-normalized, then each
    coordinate is uniformly quantized in [-1, 1] with `key_bits` bits. Values
    use global min/max scalar quantization with `bits` bits per coordinate.

    Parameters
    ----------
    vec:
        Full embedding (384 floats for MiniLM).
    bits:
        Bit width for the **value** half (default 3 per project spec).
    key_bits:
        Bit width for the **key** half on the unit sphere (default 5 so cosine
        retrieval stays stable while total size stays near ~6x vs float32).
    """
    v = np.asarray(vec, dtype=np.float64).ravel()
    dim = int(v.shape[0])
    half = dim // 2
    key = v[:half]
    val = v[half:]

    r_k = float(np.linalg.norm(key, ord=2))
    if r_k < 1e-12:
        u = np.zeros_like(key)
        if key.size > 0:
            u[0] = 1.0
    else:
        u = key / r_k
    u = np.clip(u, -1.0, 1.0)

    q_key = _quantize_uniform(u, key_bits, -1.0, 1.0)
    key_packed = _pack_bits(q_key, key_bits)

    if val.size:
        v_min = float(np.min(val))
        v_max = float(np.max(val))
        if v_max <= v_min:
            v_max = v_min + 1e-6
        vn = (val - v_min) / (v_max - v_min)
        q_val = _quantize_uniform(vn, bits, 0.0, 1.0)
        val_packed = _pack_bits(q_val, bits)
    else:
        v_min, v_max = 0.0, 1.0
        val_packed = b""

    header = struct.pack(
        "<4sHHHH",
        _MAGIC[:4],
        dim,
        half,
        int(key_bits),
        int(bits),
    )
    meta = struct.pack("<e", np.float16(r_k)) + struct.pack("<ff", v_min, v_max)
    nk = int(q_key.shape[0])
    nv = int(val.size)
    counts = struct.pack("<ii", nk, nv)
    return header + meta + counts + key_packed + val_packed


def decompress_vector(blob: bytes) -> np.ndarray:
    """
    Decompress a blob from :func:`compress_vector` back to float32.

    Reconstructs the key half as (unit direction from dequantized coordinates)
    times the stored norm, and the value half via inverse linear scaling.
    """
    if len(blob) < 4:
        raise ValueError("Truncated ghost memory blob")
    if blob[:4] != _MAGIC[:4]:
        raise ValueError("Invalid ghost memory magic header")
    rest = blob[4:]
    dim, half, key_bits, val_bits = struct.unpack("<HHHH", rest[:8])
    rest = rest[8:]
    r_k = float(np.frombuffer(rest[:2], dtype=np.float16)[0])
    rest = rest[2:]
    v_min, v_max = struct.unpack("<ff", rest[:8])
    rest = rest[8:]
    nk, nv = struct.unpack("<ii", rest[:8])
    rest = rest[8:]
    kb = (int(nk) * int(key_bits) + 7) // 8
    key_blob = rest[:kb]
    rest = rest[kb:]
    vb = (int(nv) * int(val_bits) + 7) // 8
    val_blob = rest[:vb]

    q_key = _unpack_bits(key_blob, int(nk), int(key_bits))
    u_hat = _dequantize_uniform(q_key, int(key_bits), -1.0, 1.0)
    un = np.linalg.norm(u_hat, ord=2)
    if un > 1e-12:
        u_hat = u_hat / un

    if int(nv) > 0:
        q_val = _unpack_bits(val_blob, int(nv), int(val_bits))
        t = _dequantize_uniform(q_val, int(val_bits), 0.0, 1.0)
        val = v_min + t * (v_max - v_min)
    else:
        val = np.zeros(0, dtype=np.float64)

    out = np.zeros(int(dim), dtype=np.float32)
    hk = min(half, u_hat.shape[0])
    out[:hk] = (u_hat[:hk] * r_k).astype(np.float32)
    hv = int(dim) - half
    if hv > 0 and val.size:
        out[half : half + min(hv, val.size)] = val[: min(hv, val.size)].astype(np.float32)
    return out


def compressed_size_bytes(blob: bytes) -> int:
    """Return payload byte length (used for benchmarks)."""
    return len(blob)


def float32_baseline_bytes(dim: int) -> int:
    """Size of a dense float32 vector of length dim."""
    return int(dim) * 4


def batch_compress(vectors: np.ndarray, bits: int = 3, *, key_bits: int = 5) -> list:
    """Compress each row of a 2-D array; returns list of byte blobs."""
    rows = []
    for i in range(vectors.shape[0]):
        rows.append(compress_vector(vectors[i], bits=bits, key_bits=key_bits))
    return rows
