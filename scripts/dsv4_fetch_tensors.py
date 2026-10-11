#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
"""Fetch individual DeepSeek-V4-Flash tensors by HTTP range and repack them.

WHY NOT A LEADING WINDOW
    scripts/download_dsv4_layer0.sh takes a leading slice of a shard, which
    works because every tensor the Layer 0 mHC pass needs sits in the first
    ~11.5 MB. That breaks down completely for the MoE experts: in shard 3,
    expert 0's weights start at byte 344,923,480, expert 2's at 1,754,209,624
    and expert 7's at 3,150,912,856. The ~100 MB we actually want is smeared
    across 3.4 GB, so the only way to avoid the whole shard is a RANGE REQUEST
    PER TENSOR.

    Scattered ranges cannot be written back into a shard-shaped file -- the
    result would be a sparse file whose holes read as zeros, and zeros are
    indistinguishable from real weights. Instead this script writes a NEW,
    compact, fully valid safetensors file with a regenerated header whose
    data_offsets describe the repacked layout. include/moe/emu/
    safetensors_view.hpp then reads it as an ordinary checkpoint, with every
    tensor `available`.

WHICH EXPERTS
    Layers 0-2 are DeepSeek-V4's hash layers (`num_hash_layers: 3`), where the
    routed experts are a pure function of the token id:
    `indices = tid2eid[token_id]` (vllm-ascend
    tools/dsv4_moe_runtime/routing/hash_router.py, matching Gate.forward). So
    "the first 8 experts" are NOT the ones a token activates. This script
    fetches `layers.<moe>.ffn.gate.tid2eid`, reads the row for the chosen token
    id, and then downloads exactly those `num_experts_per_tok` experts -- the
    ones the MoE block will really run. The chosen id and its expert row are
    written into the output's __metadata__ so the bench can assert it routed to
    the experts that are actually present.

This is a build-time tool. Nothing in the engine or the emulation library
depends on Python.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

REPO = "deepseek-ai/DeepSeek-V4-Flash"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"

# Element size per safetensors dtype spelling. Must agree with
# SafetensorsDtypeSize() in include/moe/emu/safetensors_view.hpp.
DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}


def curl_range(url: str, start: int, end_inclusive: int, dest_fh, retries: int = 5) -> int:
    """Append bytes [start, end_inclusive] of `url` to `dest_fh`. Returns count."""
    span = end_inclusive - start + 1
    for attempt in range(retries):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--retry", "3", "--retry-delay", "2",
                 "-H", f"Range: bytes={start}-{end_inclusive}",
                 "-o", tmp_path, url],
                capture_output=True, timeout=1800,
            )
            size = os.path.getsize(tmp_path)
            if result.returncode == 0 and size == span:
                with open(tmp_path, "rb") as src:
                    while True:
                        chunk = src.read(1 << 22)
                        if not chunk:
                            break
                        dest_fh.write(chunk)
                return size
            sys.stderr.write(
                f"  range attempt {attempt + 1}/{retries} failed "
                f"(rc={result.returncode}, got {size} of {span} bytes)\n")
        finally:
            os.unlink(tmp_path)
    raise RuntimeError(f"could not fetch bytes {start}-{end_inclusive} of {url}")


def fetch_bytes(url: str, start: int, end_inclusive: int) -> bytes:
    """Small ranged read straight into memory."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as fh:
            curl_range(url, start, end_inclusive, fh)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(tmp_path)


def read_shard_header(shard: str, cache: dict) -> tuple[int, dict]:
    """(header_len, header_dict) for a shard, fetched once and memoized."""
    if shard in cache:
        return cache[shard]
    url = f"{BASE_URL}/{shard}"
    header_len = struct.unpack("<Q", fetch_bytes(url, 0, 7))[0]
    raw = fetch_bytes(url, 8, 8 + header_len - 1)
    header = json.loads(raw.decode("utf-8"))
    cache[shard] = (header_len, header)
    return cache[shard]


def expected_bytes(entry: dict) -> int:
    size = DTYPE_SIZE.get(entry["dtype"])
    if size is None:
        raise RuntimeError(f"unknown dtype {entry['dtype']}")
    count = 1
    for extent in entry["shape"]:
        count *= extent
    return count * size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .safetensors path")
    parser.add_argument("--index", required=True, help="model.safetensors.index.json")
    parser.add_argument("--config", required=True, help="config.json")
    parser.add_argument("--attn-layers", default="0,2",
                        help="layers whose attention stack to fetch (default 0,2)")
    parser.add_argument("--moe-layer", type=int, default=1,
                        help="layer whose MoE block to fetch (default 1)")
    parser.add_argument("--token-id", type=int, default=3,
                        help="token id whose routed experts to fetch (default 3)")
    parser.add_argument("--max-experts", type=int, default=8,
                        help="cap on routed experts fetched (default 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and its size, download nothing")
    args = parser.parse_args()

    with open(args.index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)

    top_k = int(config["num_experts_per_tok"])
    num_hash_layers = int(config.get("num_hash_layers", 0))
    vocab_size = int(config["vocab_size"])
    attn_layers = [int(v) for v in args.attn_layers.split(",") if v != ""]
    header_cache: dict = {}

    # ---- the per-layer tensor sets -------------------------------------
    wanted: list[str] = []

    def want(name: str, required: bool = True) -> None:
        if name in weight_map:
            if name not in wanted:
                wanted.append(name)
        elif required:
            raise RuntimeError(f"{name} is not in the index")

    for layer in attn_layers:
        prefix = f"layers.{layer}"
        # mHC triples for both rounds, plus the two activation norms.
        for stem in ("hc_attn", "hc_ffn"):
            for suffix in ("fn", "scale", "base"):
                want(f"{prefix}.{stem}_{suffix}")
        want(f"{prefix}.attn_norm.weight")
        want(f"{prefix}.ffn_norm.weight")
        # Shared-KV MLA attention: wq_a -> q_norm -> wq_b, wkv -> kv_norm,
        # grouped wo_a -> wo_b, plus the per-head attention sink.
        want(f"{prefix}.attn.attn_sink")
        want(f"{prefix}.attn.q_norm.weight")
        want(f"{prefix}.attn.kv_norm.weight")
        for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            want(f"{prefix}.attn.{projection}.weight")
            want(f"{prefix}.attn.{projection}.scale")
        # Compressed-attention layers only (compress_ratios[layer] == 4).
        for suffix in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
            want(f"{prefix}.attn.compressor.{suffix}", required=False)
        for suffix in ("wq_b.weight", "wq_b.scale", "weights_proj.weight"):
            want(f"{prefix}.attn.indexer.{suffix}", required=False)
        for suffix in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
            want(f"{prefix}.attn.indexer.compressor.{suffix}", required=False)

    # ---- the MoE block --------------------------------------------------
    moe = args.moe_layer
    moe_prefix = f"layers.{moe}.ffn"
    want(f"layers.{moe}.ffn_norm.weight")
    for suffix in ("fn", "scale", "base"):
        want(f"layers.{moe}.hc_ffn_{suffix}")
    want(f"{moe_prefix}.gate.weight")
    for projection in ("w1", "w2", "w3"):
        want(f"{moe_prefix}.shared_experts.{projection}.weight")
        want(f"{moe_prefix}.shared_experts.{projection}.scale")

    # Which experts? For a hash layer, exactly the ones tid2eid names.
    routed: list[int] = []
    tid2eid_name = f"{moe_prefix}.gate.tid2eid"
    is_hash_layer = moe < num_hash_layers and tid2eid_name in weight_map
    if is_hash_layer:
        want(tid2eid_name)
        if not args.dry_run:
            shard = weight_map[tid2eid_name]
            _, header = read_shard_header(shard, header_cache)
            entry = header[tid2eid_name]
            rows, columns = entry["shape"]
            if columns != top_k:
                raise RuntimeError(
                    f"{tid2eid_name} has {columns} columns but "
                    f"num_experts_per_tok is {top_k}")
            if not 0 <= args.token_id < rows:
                raise RuntimeError(
                    f"--token-id {args.token_id} is outside [0, {rows})")
            element = DTYPE_SIZE[entry["dtype"]]
            base = 8 + read_shard_header(shard, header_cache)[0] + entry["data_offsets"][0]
            row_start = base + args.token_id * columns * element
            row_bytes = fetch_bytes(f"{BASE_URL}/{shard}", row_start,
                                    row_start + columns * element - 1)
            code = {8: "<q", 4: "<i"}[element]
            routed = [struct.unpack_from(code, row_bytes, i * element)[0]
                      for i in range(columns)]
            print(f"  tid2eid[{args.token_id}] -> experts {routed}")
            # Deduplicate while preserving rank order; a table may repeat an id.
            seen: set[int] = set()
            unique = [e for e in routed if not (e in seen or seen.add(e))]
            for expert in unique[:args.max_experts]:
                for projection in ("w1", "w2", "w3"):
                    want(f"{moe_prefix}.experts.{expert}.{projection}.weight")
                    want(f"{moe_prefix}.experts.{expert}.{projection}.scale")
    else:
        want(f"{moe_prefix}.gate.bias")
        for expert in range(min(args.max_experts, int(config["n_routed_experts"]))):
            for projection in ("w1", "w2", "w3"):
                want(f"{moe_prefix}.experts.{expert}.{projection}.weight")
                want(f"{moe_prefix}.experts.{expert}.{projection}.scale")

    # ---- resolve every wanted tensor to a (shard, offset, size) --------
    plan = []
    total = 0
    for name in wanted:
        shard = weight_map[name]
        header_len, header = read_shard_header(shard, header_cache)
        entry = header[name]
        begin, end = entry["data_offsets"]
        declared = expected_bytes(entry)
        if end - begin != declared:
            raise RuntimeError(
                f"{name}: data_offsets span {end - begin} bytes but "
                f"{entry['dtype']} {entry['shape']} needs {declared}")
        plan.append({
            "name": name, "shard": shard, "dtype": entry["dtype"],
            "shape": entry["shape"],
            "file_begin": 8 + header_len + begin,
            "size": end - begin,
        })
        total += end - begin

    print(f"  {len(plan)} tensors, {total / 1e6:.1f} MB of weight bytes")
    if args.dry_run:
        by_shard: dict[str, int] = {}
        for item in plan:
            by_shard[item["shard"]] = by_shard.get(item["shard"], 0) + item["size"]
        for shard in sorted(by_shard):
            print(f"    {shard}  {by_shard[shard] / 1e6:9.1f} MB")
        return 0

    # ---- build the compact header, then stream the data ----------------
    # Grouping the ranges by shard and ascending offset keeps each shard's
    # reads monotonic, which is what lets the CDN serve them from one place
    # instead of seeking back and forth.
    plan.sort(key=lambda item: (item["shard"], item["file_begin"]))

    header_out: dict = {}
    cursor = 0
    for item in plan:
        header_out[item["name"]] = {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "data_offsets": [cursor, cursor + item["size"]],
        }
        item["out_begin"] = cursor
        cursor += item["size"]

    header_out["__metadata__"] = {
        "produced_by": "scripts/dsv4_fetch_tensors.py",
        "source_repo": REPO,
        "attn_layers": ",".join(str(v) for v in attn_layers),
        "moe_layer": str(moe),
        "moe_is_hash_layer": "true" if is_hash_layer else "false",
        "token_id": str(args.token_id),
        "routed_experts": ",".join(str(v) for v in routed),
        "num_experts_per_tok": str(top_k),
        "vocab_size": str(vocab_size),
    }

    blob = json.dumps(header_out, separators=(",", ":")).encode("utf-8")
    # safetensors wants the data segment 8-byte aligned in practice; pad the
    # header with spaces (legal JSON whitespace) rather than shifting offsets.
    padding = (-(8 + len(blob))) % 8
    blob += b" " * padding

    out_path = args.out
    part_path = out_path + ".part"
    written = 0
    with open(part_path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for index, item in enumerate(plan, start=1):
            print(f"  [{index:3d}/{len(plan)}] {item['name']}  "
                  f"{item['size'] / 1e6:.2f} MB", flush=True)
            got = curl_range(f"{BASE_URL}/{item['shard']}", item["file_begin"],
                             item["file_begin"] + item["size"] - 1, out)
            if got != item["size"]:
                raise RuntimeError(f"{item['name']}: short read")
            written += got
    if written != cursor:
        raise RuntimeError(f"wrote {written} data bytes, expected {cursor}")
    os.replace(part_path, out_path)
    print(f"  wrote {out_path}: {os.path.getsize(out_path) / 1e6:.1f} MB "
          f"({len(plan)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
