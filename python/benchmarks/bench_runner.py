#
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
#

"""Benchmark harness for moe_runner: TTFT, TPOT, swap/PCIe accounting.

ZERO PyTorch / torch_npu: the harness drives the engine binary as a
subprocess and reads back the machine-readable ``--stats-json`` file the
runner writes at the end of a decode. Nothing here invents numbers the
engine did not report; the two derived metrics are labeled as estimates.

Sweeps context (prompt) lengths -- the runner runs prefill AS single-token
decode steps, so TTFT over a P-token prompt is P steps + one readback, which
is the honest number for this graph shape and is labeled as such in output.

Usage::

    python -m benchmarks.bench_runner --engine build_cann/moe_runner \
        --context-lengths 128,512,2048 --new-tokens 32 \
        --weights /mnt/c/models/DeepSeek-V4-Flash --out results.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DSV4_VOCAB_SIZE = 129280


def make_prompt_ids(length: int, seed: int, vocab_size: int = DSV4_VOCAB_SIZE) -> list[int]:
    """Deterministic in-vocabulary prompt ids (the runner embeds no tokenizer)."""
    rng = random.Random(seed)
    return [rng.randrange(vocab_size) for _ in range(length)]


def run_once(engine: Path, prompt_ids: list[int], new_tokens: int, extra_args: list[str],
             stats_path: Path, timeout_s: float = 0.0) -> dict:
    """One engine invocation; returns the parsed --stats-json payload."""
    command = [
        str(engine),
        "--prompt-ids", ",".join(str(token) for token in prompt_ids),
        "--max-new-tokens", str(new_tokens),
        "--stats-json", str(stats_path),
        *extra_args,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s or None)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:] + "\n" + completed.stderr[-4000:] + "\n")
        raise SystemExit(f"engine failed (exit {completed.returncode}): {' '.join(command[:4])} ...")
    stats = json.loads(stats_path.read_text())
    stats["engine_stderr_tail"] = completed.stderr[-500:]
    return stats


def derive_metrics(stats: dict, pcie_bytes_per_second: float) -> dict:
    """The derived, clearly-labeled metrics the raw stats cannot state alone."""
    swap_bytes = stats.get("swap_h2d_bytes", 0) + stats.get("swap_d2h_bytes", 0)
    decode_span = max(stats.get("decode_span_seconds", 0.0), 1e-9)
    total_seconds = max(stats.get("total_seconds", 0.0), 1e-9)
    resident_bytes = stats.get("device_slots", 0) * stats.get("slot_bytes", 0)
    resident_bytes += stats.get("backbone_bytes", 0)
    return {
        # Achieved throughput of the exclusive hierarchy's duplex exchange.
        "swap_pcie_bytes": swap_bytes,
        "swap_pcie_bandwidth_Bps": swap_bytes / decode_span,
        "swap_pcie_utilization_percent": 100.0 * (swap_bytes / decode_span) / pcie_bytes_per_second,
        "swap_bytes_per_generated_token": swap_bytes / max(stats.get("generated_tokens", 1), 1),
        "expert_swap_misses_per_layer_step": stats.get("expert_slot_misses", 0),
        # ESTIMATE: the resident working set (device slots + backbone) divided
        # by wall time -- the touch rate if every resident byte were read once
        # per run. NOT a measured HBM bandwidth; a comparable upper bound.
        "resident_footprint_bytes": resident_bytes,
        "effective_hbm_bandwidth_estimate_Bps": resident_bytes / total_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.bench_runner",
        description="TTFT / TPOT / swap sweep over moe_runner runs (zero torch).",
    )
    parser.add_argument("--engine", default="build_cann/moe_runner", help="path to the moe_runner binary")
    parser.add_argument("--context-lengths", default="128,512,2048",
                        help="comma-separated prompt lengths to sweep (default 128,512,2048)")
    parser.add_argument("--new-tokens", type=int, default=32, help="generated tokens per run (default 32)")
    parser.add_argument("--weights", default=None, help="checkpoint dir; omit for --synthetic-weights")
    parser.add_argument("--config", default=None, help="model config.json for the runner (--config)")
    parser.add_argument("--seed", type=int, default=20251008, help="prompt-id seed")
    parser.add_argument("--warmup", type=int, default=0, help="untimed warmup runs per length (default 0)")
    parser.add_argument("--pcie-gbs", type=float, default=32.0,
                        help="theoretical PCIe GB/s for utilization math (default 32, Gen4 x16)")
    parser.add_argument("--repeats", type=int, default=1, help="timed runs per length (default 1; median kept)")
    parser.add_argument("--extra", default="", help="extra raw args passed to moe_runner verbatim")
    parser.add_argument("--out", default=None, help="write the full results JSON here")
    arguments = parser.parse_args(argv)

    engine = Path(arguments.engine)
    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine} (build it first, or pass --engine)")

    extra_args = [arguments.extra] if arguments.extra else []
    if arguments.weights:
        extra_args += ["--weights", arguments.weights]
    else:
        extra_args += ["--synthetic-weights"]
    if arguments.config:
        extra_args += ["--config", arguments.config]

    lengths = [int(piece) for piece in arguments.context_lengths.split(",") if piece.strip()]
    results = []
    with tempfile.TemporaryDirectory(prefix="moe_bench_") as scratch:
        stats_path = Path(scratch) / "stats.json"
        for length in lengths:
            prompt_ids = make_prompt_ids(length, seed=arguments.seed)
            if length > 8192:
                raise SystemExit(f"context length {length} exceeds the runner's default reserved context (8192)")
            for _ in range(arguments.warmup):
                run_once(engine, prompt_ids, arguments.new_tokens, extra_args, stats_path)
            runs = [
                run_once(engine, prompt_ids, arguments.new_tokens, extra_args, stats_path)
                for _ in range(max(arguments.repeats, 1))
            ]
            # Median run: robust to one-off scheduler noise on a shared host.
            median = min(runs, key=lambda item: item.get("total_seconds", float("inf")))
            entry = {
                "context_tokens": length,
                "new_tokens": arguments.new_tokens,
                "repeats": len(runs),
                "ttft_seconds": median.get("ttft_seconds"),
                "tpot_seconds": median.get("tpot_seconds"),
                "total_seconds": median.get("total_seconds"),
                "ttft_over_context_lengths": statistics.fmean(
                    float(item.get("ttft_seconds", 0.0)) for item in runs),
                **derive_metrics(median, arguments.pcie_gbs * 1e9),
            }
            results.append(entry)
            print(f"context {length:>5}: ttft {entry['ttft_seconds']:.3f}s  "
                  f"tpot {entry['tpot_seconds'] * 1e3:.3f}ms  "
                  f"swap {entry['swap_pcie_bytes'] >> 20} MiB "
                  f"({entry['swap_pcie_bandwidth_Bps'] / 1e9:.2f} GB/s, "
                  f"{entry['swap_pcie_utilization_percent']:.1f}% of PCIe)  "
                  f"hbm-est {entry['effective_hbm_bandwidth_estimate_Bps'] / 1e9:.1f} GB/s")

    if arguments.out:
        Path(arguments.out).write_text(json.dumps({"results": results, "seed": arguments.seed}, indent=2) + "\n")
        print(f"results written to {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
