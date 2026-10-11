#!/usr/bin/env python3
"""Check expert-30 MXFP4 byte order against DeepSeek's inference/convert.py.

Only w1.weight and w1.scale are read from the compact safetensors cache.
The official decoder is at
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/convert.py
(cast_e2m1fn_to_e4m3fn, low/high stacking before flattening).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


FP4_TABLE = torch.tensor(
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def unpack(packed: torch.Tensor, low_first: bool) -> torch.Tensor:
    """Return logical E2M1 values in [row, column] order, without scale."""
    raw = packed.view(torch.uint8)
    low, high = raw & 0x0F, (raw >> 4) & 0x0F
    first, second = (low, high) if low_first else (high, low)
    return torch.stack((FP4_TABLE[first.long()], FP4_TABLE[second.long()]), -1).flatten(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path,
        default=Path("data/dsv4_layer0/dsv4_semi_e2e.safetensors"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    stem = "layers.1.ffn.experts.30.w1"
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        packed = f.get_tensor(stem + ".weight")
        scale = f.get_tensor(stem + ".scale")
    assert packed.shape == (2048, 2048) and scale.shape == (2048, 128)
    assert packed.dtype == torch.int8 and scale.dtype == torch.float8_e8m0fnu

    # Literal order from official DeepSeek convert.py lines 27-30.
    raw = packed.view(torch.uint8)
    official = torch.stack(
        (FP4_TABLE[(raw & 0x0F).long()], FP4_TABLE[((raw >> 4) & 0x0F).long()]),
        dim=-1,
    ).flatten(1)
    block_scales = scale.float().repeat_interleave(32, dim=1)
    official *= block_scales
    assert torch.isfinite(official).all(), "expert 30 contains non-finite values"

    outputs = {}
    x_ones = torch.ones(official.shape[1], dtype=torch.float32)
    # Ones cannot distinguish a swap within every adjacent pair. A nonuniform
    # input is required to test the GEMV convention, so report both probes.
    x_probe = torch.linspace(-1.0, 1.0, official.shape[1])
    official_ones = official @ x_ones
    official_probe = official @ x_probe
    for label, low_first in (("A low-first", True), ("B high-first", False)):
        decoded = unpack(packed, low_first) * block_scales
        outputs[label] = (decoded @ x_ones, decoded @ x_probe)
        max_weight_diff = (decoded - official).abs().max().item()
        ones_error = torch.linalg.vector_norm(outputs[label][0] - official_ones).item()
        probe_error = torch.linalg.vector_norm(outputs[label][1] - official_probe).item()
        print(f"{label:14} max|W-official|={max_weight_diff:.6g} "
              f"ones L2={ones_error:.6g} nonuniform L2={probe_error:.6g}")
        del decoded

    assert torch.equal(unpack(packed, True) * block_scales, official)
    assert not torch.equal(unpack(packed, False) * block_scales, official)
    print("Result: kMxfp4LowNibbleFirst = true (official low, then high).")
    print("The all-ones GEMV is invariant to swapping adjacent nibbles; "
          "the nonuniform probe and elementwise comparison resolve the order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
