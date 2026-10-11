#!/usr/bin/env python3
"""Standalone CPU PyTorch check of the compact DeepSeek-V4-Flash slice.

This reproduces the C++ semi-e2e schedule: attention 0, attention 2, MoE 1.
The mHC, gating, SwiGLU and attention formulas follow the official
DeepSeek-V4-Flash inference/model.py and inference/kernel.py:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
The cache and
golden are a *semi* reference: the golden uses a synthetic embedding, 8 reused
KV warmup positions, and omits compressed keys from layer-2 attention. For
golden parity, this script uses that same schedule and the v2 dump's input.

Only tensors for the requested stages and the six token-3 experts are opened.
Peak RSS is checked against 2.5 GiB before successful exit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


FP4_TABLE = torch.tensor(
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
EXPERTS = [30, 205, 192, 177, 242, 102]
D = 4096
HC = 4


def read_golden(path: Path) -> dict[str, torch.Tensor | int]:
    data = path.read_bytes()
    if len(data) < 48:
        raise ValueError("golden dump is shorter than its header")
    magic, version, dtype, streams, hidden, token, count, sink_n, compressed, topk = struct.unpack_from(
        "<8sIIiiqiiii", data,
    )
    if (magic, dtype, streams, hidden, token, count, sink_n) != (
        b"DSV4SE2E", 1, 4, 4096, 3, 6, 4,
    ) or version not in (1, 2, 3):
        raise ValueError("golden header is incompatible with token-3 BF16 reference")
    offset = 48
    result: dict[str, torch.Tensor | int] = {"version": version}

    def take(name: str, numel: int, dtype: torch.dtype, nbytes: int) -> None:
        nonlocal offset
        end = offset + numel * nbytes
        if end > len(data):
            raise ValueError(f"truncated golden at {name}")
        result[name] = torch.frombuffer(bytearray(data[offset:end]), dtype=dtype)
        offset = end

    take("x_final", HC * D, torch.bfloat16, 2)
    take("B0", 16, torch.float32, 4)
    take("B2", 16, torch.float32, 4)
    take("expert_ids", 6, torch.int32, 4)
    take("expert_weights", 6, torch.float32, 4)
    if version >= 2:
        for name in ("x_input", "x_after0", "x_after2"):
            take(name, HC * D, torch.bfloat16, 2)
        for name in ("h_in0", "h_in2", "h_in_moe"):
            take(name, D, torch.float32, 4)
        take("Bmoe", 16, torch.float32, 4)
        take("expert_outputs", 6 * D, torch.float32, 4)
        take("moe_routed", D, torch.float32, 4)
        take("moe_shared", D, torch.float32, 4)
    if version >= 3:
        take("compressed_tokens", compressed * 512, torch.bfloat16, 2)
    if offset != len(data):
        raise ValueError(f"golden has {len(data) - offset} trailing bytes")
    assert result["expert_ids"].tolist() == EXPERTS
    return result


def check_compact_offsets(path: Path) -> int:
    """Validate every repacked byte extent before any selective tensor read."""
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    dtypes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
              "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E8M0": 1}
    tensors = [(name, item) for name, item in header.items() if name != "__metadata__"]
    previous_end = 0
    for name, item in sorted(tensors, key=lambda entry: entry[1]["data_offsets"][0]):
        start, end = item["data_offsets"]
        expected = math.prod(item["shape"]) * dtypes[item["dtype"]]
        if start != previous_end or end - start != expected or end > path.stat().st_size - 8 - hlen:
            raise ValueError(f"invalid/shifted byte extent for {name}: {start}:{end}")
        previous_end = end
    if previous_end != path.stat().st_size - 8 - hlen:
        raise ValueError("last tensor does not end at the data segment boundary")
    metadata = header.get("__metadata__", {})
    if metadata.get("token_id") != "3" or metadata.get("routed_experts") != ",".join(map(str, EXPERTS)):
        raise ValueError("cache metadata does not match token 3 routing")
    return len(tensors)


def verify_source_bytes(cache: Path, index_path: Path) -> int:
    """Hash each compact tensor against its exact upstream shard byte range."""
    from dsv4_fetch_tensors import BASE_URL, curl_range, read_shard_header

    with cache.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        compact = json.loads(f.read(hlen))
    index = json.loads(index_path.read_text())["weight_map"]
    shard_headers: dict = {}
    names = [name for name in compact if name != "__metadata__"]

    class HashSink:
        def __init__(self):
            self.hash = hashlib.sha256()

        def write(self, block: bytes) -> None:
            self.hash.update(block)

    with cache.open("rb") as local:
        for pos, name in enumerate(names, 1):
            shard = index[name]
            upstream_header_len, upstream_header = read_shard_header(shard, shard_headers)
            upstream = upstream_header[name]
            item = compact[name]
            if upstream["shape"] != item["shape"] or upstream["dtype"] != item["dtype"]:
                raise ValueError(f"source metadata differs for {name}")
            size = item["data_offsets"][1] - item["data_offsets"][0]
            remote_start = 8 + upstream_header_len + upstream["data_offsets"][0]
            remote = HashSink()
            curl_range(f"{BASE_URL}/{shard}", remote_start,
                       remote_start + size - 1, remote)
            local.seek(8 + hlen + item["data_offsets"][0])
            nearby = hashlib.sha256()
            left = size
            while left:
                chunk = local.read(min(1 << 22, left))
                if not chunk:
                    raise ValueError(f"short compact data for {name}")
                nearby.update(chunk)
                left -= len(chunk)
            if remote.hash.digest() != nearby.digest():
                raise ValueError(f"source byte mismatch (possible shifted offset): {name}")
            if pos % 10 == 0 or pos == len(names):
                print(f"Source byte ranges verified: {pos}/{len(names)}", flush=True)
    return len(names)


class Weights:
    def __init__(self, path: Path):
        self.handle = safe_open(path, framework="pt", device="cpu")
        self.f = self.handle.__enter__()
        self.keys = set(self.f.keys())

    def close(self) -> None:
        self.handle.__exit__(None, None, None)

    def get(self, name: str) -> torch.Tensor:
        if name not in self.keys:
            raise KeyError(f"missing selective tensor: {name}")
        return self.f.get_tensor(name)

    def fp8_linear(self, stem: str, x: torch.Tensor) -> torch.Tensor:
        """Block-128 E4M3 dequantization followed by a FP32 PyTorch GEMV."""
        w = self.get(stem + ".weight")
        s = self.get(stem + ".scale")
        if w.dtype != torch.float8_e4m3fn or s.dtype != torch.float8_e8m0fnu:
            raise TypeError(f"{stem}: expected E4M3 weight and E8M0 scale")
        rows, cols = w.shape
        assert s.shape == ((rows + 127) // 128, (cols + 127) // 128)
        out = torch.empty(rows, dtype=torch.float32)
        # Whole matrix dequantization can transiently double RSS; blocks of 128
        # output rows cap the live copy at 128 * cols * 4 bytes.
        for row in range(0, rows, 128):
            sub = w[row:row + 128].float()
            scales = s[row // 128].float().repeat_interleave(128)[:cols]
            sub *= scales
            out[row:row + sub.shape[0]] = F.linear(x.float(), sub)
        return out

    def mxfp4_linear(self, stem: str, x: torch.Tensor) -> torch.Tensor:
        packed = self.get(stem + ".weight")
        scale = self.get(stem + ".scale")
        assert packed.dtype == torch.int8 and scale.dtype == torch.float8_e8m0fnu
        rows, bytes_per_row = packed.shape
        assert scale.shape == (rows, bytes_per_row // 16)
        out = torch.empty(rows, dtype=torch.float32)
        for row in range(0, rows, 128):
            raw = packed[row:row + 128].view(torch.uint8)
            lo = FP4_TABLE[(raw & 15).long()]
            hi = FP4_TABLE[((raw >> 4) & 15).long()]
            sub = torch.stack((lo, hi), -1).flatten(1)
            sub *= scale[row:row + sub.shape[0]].float().repeat_interleave(32, 1)
            out[row:row + sub.shape[0]] = F.linear(x.float(), sub)
        return out


def rms_norm(x: torch.Tensor, gain: torch.Tensor | None = None) -> torch.Tensor:
    value = x.float() * torch.rsqrt(x.float().square().mean() + 1e-6)
    return value if gain is None else value * gain.float()


def mhc_pre(weights: Weights, x: torch.Tensor, layer: int, part: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stem = f"layers.{layer}.hc_{part}"
    phi = weights.get(stem + "_fn").float()
    alpha = weights.get(stem + "_scale").float()
    bias = weights.get(stem + "_base").float()
    assert phi.shape == (24, HC * D) and alpha.shape == (3,) and bias.shape == (24,)
    flat = x.float().flatten()
    mixes = F.linear(flat, phi) * torch.rsqrt(flat.square().mean() + 1e-6)
    pre = torch.sigmoid(mixes[:4] * alpha[0] + bias[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[4:8] * alpha[1] + bias[4:8])
    B = (mixes[8:] * alpha[2] + bias[8:]).view(4, 4).softmax(-1) + 1e-6
    B = B / (B.sum(0, keepdim=True) + 1e-6)
    for _ in range(19):
        B = B / (B.sum(1, keepdim=True) + 1e-6)
        B = B / (B.sum(0, keepdim=True) + 1e-6)
    h_in = (pre[:, None] * x.float()).sum(0)
    return h_in, post, B


def mhc_post(x: torch.Tensor, out: torch.Tensor, post: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Official Block.hc_post: post * sublayer + comb^T * residual.
    # The CPU emulation casts sublayer output to BF16 before this update.
    return (B.T @ x.float() + post[:, None] * out.to(torch.bfloat16).float()).to(torch.bfloat16)


def half_rope(x: torch.Tensor, position: int, theta: float = 10000.) -> torch.Tensor:
    result = x.clone()
    freq = torch.arange(32, dtype=torch.float32)
    angle = position / (theta ** (freq / 32))
    c, s = angle.cos(), angle.sin()
    low, high = x[..., -64:-32], x[..., -32:]
    result[..., -64:-32] = c * low - s * high
    result[..., -32:] = s * low + c * high
    return result


def interleaved_rope(x: torch.Tensor, position: int, theta: float, inverse: bool = False) -> torch.Tensor:
    """Official model.py apply_rotary_emb: adjacent pairs as complex lanes."""
    result = x.clone()
    freq = torch.arange(32, dtype=torch.float32)
    angle = position / (theta ** (freq / 32))
    if inverse:
        angle = -angle
    c, s = angle.cos(), angle.sin()
    even, odd = x[..., -64::2], x[..., -63::2]
    result[..., -64::2] = c * even - s * odd
    result[..., -63::2] = s * even + c * odd
    return result


def attention(weights: Weights, h: torch.Tensor, layer: int, formula: str) -> torch.Tensor:
    stem = f"layers.{layer}.attn"
    q_lora = weights.fp8_linear(stem + ".wq_a", h)
    q_lora = rms_norm(q_lora, weights.get(stem + ".q_norm.weight"))
    q = weights.fp8_linear(stem + ".wq_b", q_lora).view(64, 512)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    theta = 160000. if layer == 2 else 10000.
    q = half_rope(q, 8) if formula == "golden" else interleaved_rope(q, 0, theta)
    kv = weights.fp8_linear(stem + ".wkv", h)
    kv = rms_norm(kv, weights.get(stem + ".kv_norm.weight"))
    # At start_pos=0, seqlen=1, official layer-2 attention has no compressed
    # KV token yet (ratio=4). The C++ golden instead reuses this activation at
    # eight warmup positions and measures position 8, without compressed keys.
    keys = (torch.stack([half_rope(kv, pos) for pos in range(9)])
            if formula == "golden" else interleaved_rope(kv, 0, theta).unsqueeze(0))
    logits = q @ keys.T / math.sqrt(512)
    sink = weights.get(stem + ".attn_sink").float()
    prob = torch.softmax(torch.cat((logits, sink[:, None]), 1), 1)[:, :keys.shape[0]]
    context_heads = prob @ keys
    if formula == "official":
        context_heads = interleaved_rope(context_heads, 0, theta, inverse=True)
    context = context_heads.flatten()
    o_lora = torch.empty(8192, dtype=torch.float32)
    for group in range(8):
        block = context[group * 4096:(group + 1) * 4096]
        # wo_a's row groups are contiguous in the checkpoint. Select only the
        # group's rows, even though safe_open yields the mapped tensor view.
        w = weights.get(stem + ".wo_a.weight")[group * 1024:(group + 1) * 1024]
        s = weights.get(stem + ".wo_a.scale")[group * 8:(group + 1) * 8]
        for row in range(0, 1024, 128):
            sub = w[row:row + 128].float()
            sub *= s[row // 128].float().repeat_interleave(128)
            o_lora[group * 1024 + row:group * 1024 + row + 128] = F.linear(block, sub)
    return weights.fp8_linear(stem + ".wo_b", o_lora)


def expert(weights: Weights, stem: str, x: torch.Tensor, routed: bool) -> torch.Tensor:
    linear = weights.mxfp4_linear if routed else weights.fp8_linear
    gate = linear(stem + ".w1", x)
    up = linear(stem + ".w3", x)
    hidden = F.silu(torch.clamp(gate, max=10.)) * torch.clamp(up, -10., 10.)
    return linear(stem + ".w2", hidden)


def compressor_c4a(weights: Weights, h: torch.Tensor) -> torch.Tensor:
    """Four-step layer-2 compressed-KV side path used by the C++ benchmark.

    Official Compressor.forward projects KV and gate, adds ape, softmaxes the
    ratio axis, reduces, RMS normalizes, then rotates. At the first group the
    four overlap rows have -inf gate scores, leaving four current b-half rows.
    """
    stem = "layers.2.attn.compressor"
    token = torch.stack([(h * (1. + .01 * step)).to(torch.bfloat16).float()
                         for step in range(4)])
    kv = F.linear(token, weights.get(stem + ".wkv.weight").float())
    gate = F.linear(token, weights.get(stem + ".wgate.weight").float())
    gate += weights.get(stem + ".ape").float()
    kv_b = kv[:, 512:]
    gate_b = gate[:, 512:]
    compressed = (kv_b * torch.softmax(gate_b, dim=0)).sum(0)
    normalized = rms_norm(compressed, weights.get(stem + ".norm.weight"))
    return half_rope(normalized, 0, 160000.).to(torch.bfloat16)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, threshold: float = .9995) -> bool:
    a, b = actual.float().flatten().double(), expected.float().flatten().double()
    if a.shape != b.shape:
        raise ValueError(f"{name} shape mismatch: {a.shape} vs {b.shape}")
    rel = torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)
    cosine = F.cosine_similarity(a, b, dim=0)
    good = cosine.item() >= threshold
    print(f"{name:22} {rel.item():12.6g} {cosine.item():12.8f} {'PASS' if good else 'FAIL'}")
    return good


def peak_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmHWM unavailable")


def fetch_embedding_row(data_dir: Path, token_id: int) -> torch.Tensor:
    """HTTP-range read of one BF16 row; never materialize the 1.06 GB table."""
    from dsv4_fetch_tensors import BASE_URL, fetch_bytes, read_shard_header

    index = json.loads((data_dir / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["embed.weight"]
    header_len, header = read_shard_header(shard, {})
    entry = header["embed.weight"]
    if entry["dtype"] != "BF16" or entry["shape"][1] != D:
        raise ValueError("source embedding has unexpected dtype or shape")
    if not 0 <= token_id < entry["shape"][0]:
        raise ValueError("token ID outside embedding table")
    start = 8 + header_len + entry["data_offsets"][0] + token_id * D * 2
    raw = fetch_bytes(f"{BASE_URL}/{shard}", start, start + D * 2 - 1)
    if len(raw) != D * 2:
        raise ValueError("short embedding-row range response")
    print(f"Fetched embed.weight[{token_id}] from {shard}: {len(raw)} bytes")
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone().repeat(HC, 1)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dsv4_layer0"))
    parser.add_argument("--input", choices=("golden-synthetic", "source-embedding"),
                        default="golden-synthetic")
    parser.add_argument("--attention-formula", choices=("golden", "official"),
                        default="golden",
                        help="golden uses reused warmup KV; official uses one-token start_pos=0")
    parser.add_argument("--verify-source-offsets", action="store_true",
                        help="range-fetch and hash every cached tensor against the source shards")
    parser.add_argument("--source-only", action="store_true",
                        help="run the source-byte audit without opening the golden or weights")
    args = parser.parse_args()
    if args.attention_formula == "official" and args.input != "source-embedding":
        parser.error("--attention-formula official requires --input source-embedding")
    torch.set_num_threads(4)
    cache = args.data_dir / "dsv4_semi_e2e.safetensors"
    count = check_compact_offsets(cache)
    if args.verify_source_offsets or args.source_only:
        verified = verify_source_bytes(cache, args.data_dir / "model.safetensors.index.json")
        print(f"Source extraction verified byte-for-byte: {verified} tensors")
    if args.source_only:
        peak = peak_rss_mb()
        limit = 2_500_000_000 / 1048576
        print(f"Peak process RSS: {peak:.1f} MiB / {limit:.1f} MiB (2.5 GB) limit")
        return 0 if peak < limit else 1
    gold = read_golden(args.data_dir / "golden_semi_e2e_ref.bin")
    if gold["version"] != 3:
        raise RuntimeError("golden lacks stage and compressor tensors; rebuild and rerun the host-only dsv4_semi_e2e_bench to write v3")
    print(f"Validated compact offsets: {count} tensors; token 3 routes to {EXPERTS}")
    weights = Weights(cache)
    try:
        x = (gold["x_input"].view(HC, D).clone() if args.input == "golden-synthetic"
             else fetch_embedding_row(args.data_dir, 3))
        actual: dict[str, torch.Tensor] = {"x_input": x}
        layer2_normed = None
        for layer in (0, 2):
            h, post, B = mhc_pre(weights, x, layer, "attn")
            actual[f"h_in{layer}"] = h
            actual[f"B{layer}"] = B
            normed = rms_norm(h, weights.get(f"layers.{layer}.attn_norm.weight"))
            if layer == 2:
                layer2_normed = normed
            out = attention(weights, normed, layer, args.attention_formula)
            x = mhc_post(x, out, post, B)
            actual[f"x_after{layer}"] = x
            gc.collect()

        actual["compressed_tokens"] = compressor_c4a(weights, layer2_normed)

        h, post, B = mhc_pre(weights, x, 1, "ffn")
        actual["h_in_moe"] = h
        actual["Bmoe"] = B
        normed = rms_norm(h, weights.get("layers.1.ffn_norm.weight"))
        gate_w = weights.get("layers.1.ffn.gate.weight").float()
        scores = F.softplus(F.linear(normed, gate_w)).sqrt()
        row = weights.get("layers.1.ffn.gate.tid2eid")[3].long().tolist()
        if row != EXPERTS:
            raise ValueError(f"tid2eid[3] = {row}, expected {EXPERTS}")
        route_w = scores[row] / scores[row].sum()
        actual["expert_weights"] = route_w
        routed = torch.zeros(D, dtype=torch.float32)
        for i, eid in enumerate(row):
            stem = f"layers.1.ffn.experts.{eid}"
            output = expert(weights, stem, normed, True)
            actual[f"expert_{eid}"] = output
            routed += route_w[i] * output
            gc.collect()
        actual["moe_routed"] = routed
        shared = expert(weights, "layers.1.ffn.shared_experts", normed, False)
        actual["moe_shared"] = shared
        out = routed * 1.5 + shared
        actual["x_final"] = mhc_post(x, out, post, B)
    finally:
        weights.close()

    if args.input == "source-embedding":
        print("Source-embedding run completed through attention 0, attention 2, MoE 1 "
              f"with {args.attention_formula} attention formula.")
        print("Quantized weights were dequantized for FP32 PyTorch GEMVs; this "
              "does not reproduce the fused FP8 activation quantizer.")
        print("Golden uses a different, synthetic input, so numerical parity "
              "metrics do not apply to this run.")
        print(f"Input L2: {torch.linalg.vector_norm(actual['x_input'].float()):.6f}; "
              f"final L2: {torch.linalg.vector_norm(actual['x_final'].float()):.6f}")
        peak = peak_rss_mb()
        limit = 2_500_000_000 / 1048576
        print(f"Peak process RSS: {peak:.1f} MiB / {limit:.1f} MiB (2.5 GB) limit")
        return 0 if peak < limit else 1

    print(f"{'tensor':22} {'relative L2':>12} {'cosine':>12} verdict")
    print("-" * 58)
    checks = []
    for name in ("x_input", "h_in0", "B0", "x_after0", "h_in2", "B2", "x_after2",
                 "h_in_moe", "Bmoe", "expert_weights"):
        checks.append(compare(name, actual[name], gold[name]))
    for i, eid in enumerate(EXPERTS):
        checks.append(compare(f"expert_{eid}", actual[f"expert_{eid}"],
                              gold["expert_outputs"][i * D:(i + 1) * D]))
    checks.append(compare("compressed_tokens", actual["compressed_tokens"],
                          gold["compressed_tokens"]))
    for name in ("moe_routed", "moe_shared", "x_final"):
        checks.append(compare(name, actual[name], gold[name]))
    peak = peak_rss_mb()
    limit = 2_500_000_000 / 1048576
    print(f"Peak process RSS: {peak:.1f} MiB / {limit:.1f} MiB (2.5 GB) limit")
    return 0 if all(checks) and peak < limit else 1


if __name__ == "__main__":
    sys.exit(main())
