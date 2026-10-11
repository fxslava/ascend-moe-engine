#!/usr/bin/env python3
"""Generate the byte-exact static UB layout for the arch35 Stage-1 kernels.

Two artefacts come out of one source of truth:

1. ``op_kernel/arch35_ub_layout.h`` -- constexpr byte offsets the device
   kernels and host tilers both use.
2. ``// @ascend-layout:`` annotation blocks injected into each kernel source
   between generated markers.

Why the annotations are needed
------------------------------
``ascend_analyzer`` does not model CANN 9.2's static-mode constructor
``LocalTensor<T>(TPosition, addr, tileSize)``, so without help it resolves no
on-core tensor for these kernels and silently skips the UB capacity (AKA1001),
alignment (AKA1002), aliasing (AKA1003) and bank-conflict (AKA3003) checks --
an audit that passes while verifying nothing. The annotations restore them.
See ISSUES_AND_GAPS.md.

Why offsets are generated rather than hand-written
--------------------------------------------------
Ascend 950PR's Unified Buffer has 8 banks on a 32-byte stride, so two operands
of one vector instruction collide (AKA3003) exactly when their base offsets
differ by a multiple of 256 B. Every buffer is therefore placed at a
32-byte-aligned offset whose bank index equals its declaration index mod 8:
two buffers share a bank only if their declaration indices differ by a
multiple of 8, and the order below keeps co-operands closer than that.

Usage:
  gen_arch35_ub_layout.py --header <path> [--inject <op_kernel dir>] [--check]
"""
import argparse
import re
import sys

ALIGN = 32
BANKS = 8
BANK_STRIDE = 32
USABLE_UB = 216 * 1024

DTYPE_SIZES = {"float": 4, "uint16_t": 2, "bfloat16_t": 2, "uint8_t": 1,
               "int32_t": 4, "uint32_t": 4, "half": 2, "fp8_e4m3fn_t": 1}

BEGIN = "// >>> BEGIN GENERATED UB LAYOUT ANNOTATIONS -- scripts/gen_arch35_ub_layout.py"
END = "// <<< END GENERATED UB LAYOUT ANNOTATIONS"

# (const stem, kernel variable name, dtype, element count, comment)
KERNELS = {
    # KER-01. x is widened once (the RMS, the projection and the h_raw
    # reduction all read it), phi streams in 4096-element chunks. Every piece
    # of small gate/Sinkhorn state gets its own slot so the generator's
    # bank-distinct placement covers it too; co-operands are listed adjacently
    # because the bank index is the declaration index mod 8.
    "Ker01": ("ker01_mhc_pre_norm_sinkhorn", [
        ("XF32",     "xf32",      "float",    16384, "widened flat x [4, 4096]"),
        ("PHI_PING", "phi_ping",  "float",     4096, "phi chunk, ping"),
        ("PROD",     "prod",      "float",     4096, "phi*x products, gamma staging"),
        ("PHI_PONG", "phi_pong",  "float",     4096, "phi chunk, pong"),
        ("HRAW",     "h_raw",     "float",     4096, "h_raw accumulator"),
        ("XBF16",    "x_bf16",    "bfloat16_t", 4096, "BF16 x / gamma load tile"),
        ("HNORM_BF", "h_norm_bf", "bfloat16_t", 4096, "BF16 h_norm store tile"),
        ("REDTMP",   "red_tmp",   "float",      512, "ReduceSum shared scratch"),
        ("RED_DST",  "red_dst",   "float",        8, "ReduceSum destination"),
        # The gate and Sinkhorn state lives in SIMT registers, not UB. Only the
        # 24 projection results going in and the 4 pre gates coming out need a
        # UB handoff; post and B are written straight to GM by the SIMT VF.
        ("U_ALL",    "u_all",     "float",       32, "the 24 projection results, SIMT input"),
        ("PRE",      "pre",       "float",        8, "h_pre gates, SIMT output"),
    ]),
    # KER-02. 4 AIVs, one output stream each, 512-element feature tiles.
    "Ker02": ("ker02_mhc_post_residual", [
        ("XW",       "xw",       "float",     2048, "widened source tiles, 4 streams"),
        ("HO_W",     "ho_w",     "float",      512, "widened h_out tile"),
        ("ACC",      "acc",      "float",      512, "FP32 accumulator"),
        ("XS_BF",    "xs_bf",    "bfloat16_t", 2048, "BF16 source tiles, 4 streams"),
        ("HO_BF",    "ho_bf",    "bfloat16_t",  512, "BF16 h_out tile"),
        ("OUT_BF",   "out_bf",   "bfloat16_t",  512, "BF16 y tile"),
        ("COEF",     "coef",     "float",       64, "reserved for staged coefficients"),
    ]),
    # KER-03. One BF16 weight row at a time against a resident FP32 x; the
    # emit phase holds the [8, 512] KV and score windows plus the softmax
    # weights. Disjoint rather than overlaid, which is why the footprint is
    # above the spec's 84,992 B estimate (see ISSUES_AND_GAPS.md).
    "Ker03": ("ker03_paged_compressor_rope", [
        ("XF32",      "xf32",       "float",    4096, "widened x"),
        ("WROW_PING", "wrow_ping",  "bfloat16_t", 4096, "BF16 weight row, ping"),
        ("WROW_F32",  "wrow_f32",   "float",    4096, "widened weight row, scaled in place"),
        ("WROW_PONG", "wrow_pong",  "bfloat16_t", 4096, "BF16 weight row, pong"),
        ("PROJ",      "proj",       "float",    1024, "one projection row [coff*D]"),
        ("XBF16",     "x_bf16",     "bfloat16_t", 4096, "BF16 x load tile"),
        # The [8, 512] windows use a SKEWED row stride of 520 floats (2080 B).
        # A natural 512-float row is 2048 B, which is 64 bank steps and so
        # bank-neutral (64 % 8 == 0): every row of a window would land on one
        # bank and folding two rows together would collide (AKA3003). 2080 B is
        # 65 steps, so consecutive rows differ by exactly one bank.
        ("WIN_KV",    "win_kv",     "float",    4160, "[8, 512] KV window, stride 520"),
        ("WIN_SC",    "win_sc",     "float",    4160, "[8, 512] score window, stride 520"),
        ("WEIGHTS",   "weights",    "float",    4160, "column-softmax weights, stride 520"),
        ("COL_MAX",   "col_max",    "float",     512, "per-column softmax maximum"),
        ("COL_SUM",   "col_sum",    "float",     512, "per-column softmax denominator"),
        ("COMP",      "comp",       "float",     512, "reduced/normalized/roped token"),
        ("COMP_BF",   "comp_bf",    "bfloat16_t", 512, "BF16 emitted token"),
        ("NORMW_F32", "norm_w_f32", "float",     512, "widened RMSNorm gain"),
        ("NORMW_BF",  "norm_w_bf",  "bfloat16_t", 512, "RMSNorm gain"),
        ("MASK",      "mask",       "uint8_t",   256, "flush-to-zero select mask"),
        ("ROPE_LO",   "rope_lo",    "float",      64, "RoPE low-half workspace"),
        ("ROPE_HI",   "rope_hi",    "float",      64, "RoPE high-half workspace"),
        ("ROPE_COS",  "rope_cos",   "float",      64, "RoPE cosine table"),
        ("ROPE_SIN",  "rope_sin",   "float",      64, "RoPE sine table"),
        ("REDTMP",    "red_tmp",    "float",     512, "ReduceSum shared scratch"),
        ("RED_DST",   "red_dst",    "float",       8, "ReduceSum destination"),
    ]),
    # KER-04. The 64x128 query is dequantized once; each key row is replicated
    # across the 64 head slots so one Mul covers the whole block, then three
    # reduction stages collapse it to 64 per-head dot products.
    "Ker04": ("ker04_fp8_indexer_topk", [
        ("Q_F32",       "q_f32",       "float",    8192, "dequantized query [64, 128]"),
        ("KEY_REP",     "key_rep",     "float",    8192, "key replicated 64x, products in place"),
        ("K_F32",       "k_f32",       "float",     128, "dequantized key row"),
        ("Q_FP8",       "q_fp8",       "uint8_t",  8192, "raw FP8 query"),
        ("CHUNK_HALF",  "chunk_half",  "half",     1024, "query dequant staging, 1024 at a time"),
        ("CHUNK_I32",   "chunk_i32",   "int32_t",  1024, "query LUT byte offsets"),
        ("FP8_LUT",     "fp8_lut",     "float",     256, "E4M3 code -> FP32, built on device"),
        ("K_FP8",       "k_fp8",       "uint8_t",   256, "raw FP8 key row, padded"),
        ("K_HALF",      "k_half",      "half",      256, "key dequant staging"),
        ("K_I32",       "k_i32",       "int32_t",   256, "key LUT byte offsets"),
        ("RED1",        "red1",        "float",    1152, "per-8-lane partials (+1 repeat of reduce slack)"),
        ("RED2",        "red2",        "float",     256, "per-64-lane partials (+1 repeat of reduce slack)"),
        ("HEAD_DOT",    "head_dot",    "float",     256, "per-head dots; ReducePairElem dst slack, see Ker05 GROUPSUM"),
        ("HEAD_SCALE",  "head_scale",  "float",      64, "folded query dequant and head gate"),
        ("SCORE",       "score",       "float",    2048, "FP32 scores, reloaded from workspace"),
        ("SCORE_BF",    "score_bf",    "bfloat16_t", 2048, "scores rounded through BF16"),
        ("COMPOSITE",   "composite",   "float",    2048, "BF16 score bits + index tie-break"),
        ("TIE",         "tie",         "int32_t",  2048, "seq_len-1-j, the tie-break term"),
        ("SORT_IDX",    "sort_idx",    "int32_t",  2048, "0..seq_len-1 companion for Sort"),
        ("SORTED",      "sorted",      "float",    4096, "Sort output, value+index pairs"),
        ("SORT_TMP",    "sort_tmp",    "float",    4096, "Sort scratch"),
        ("SVAL",        "sval",        "float",    2048, "Extract values (unused, required)"),
        ("SIDX",        "sidx",        "int32_t",  2048, "Extract indices, the kernel output"),
        ("REDTMP",      "red_tmp",     "float",     512, "ReduceSum shared scratch"),
        ("RED_DST",     "red_dst",     "float",       8, "ReduceSum destination"),
        ("NANBOX",      "nan_box",     "int32_t",     8, "quiet-NaN bit pattern for E4M3 0x7F/0xFF"),
    ]),
    # KER-05. One row at a time per core; nibbles are decoded into logical
    # column order through an on-device E2M1 lookup, then folded against x and
    # reduced per 32-element scale group.
    "Ker05": ("ker05_mxfp4_expert_gemv", [
        # The nibble unpack, the E2M1 decode and the E8M0 scale all happen in
        # SIMT registers straight off the Data Cache, so none of the former
        # staging buffers (packed bytes, half/int32 conversion, nibble streams,
        # even/odd x copies, LUT) exist any more.
        ("XF32",     "xf32",     "float", 4096, "resident FP32 x[cols]"),
        ("PROD",     "prod",     "float", 4096, "per-column products from the SIMT VF"),
        ("REDTMP",   "red_tmp",  "float",  512, "ReduceSum shared scratch"),
        ("RED_DST",  "red_dst",  "float",    8, "ReduceSum destination"),
    ]),
}


# SIMT VFs take raw `__ubuf__ T*` pointers, which a kernel obtains from an
# existing LocalTensor's GetPhyAddr(). The analyzer sees those as separate,
# unresolvable tensors (AKA3002), so each alias is declared at its source's
# offset and the pair is put in a reuse group to say the overlap is deliberate
# rather than an aliasing bug (AKA1003).
#   kernel -> [(alias variable name, source const stem)]
ALIASES = {
    "Ker01": [],
    "Ker02": [],
    "Ker03": [],
    "Ker04": [("fp8_lut_ub", "FP8_LUT")],
    "Ker05": [("x_ub", "XF32"), ("prod_ub", "PROD")],
}


def align_up(value, to=ALIGN):
    return (value + to - 1) // to * to


def bank(off):
    return (off // BANK_STRIDE) % BANKS


def place(buffers):
    out, cursor = [], 0
    for index, (stem, var, dtype, count, note) in enumerate(buffers):
        nbytes = count * DTYPE_SIZES[dtype]
        want = index % BANKS
        off = align_up(cursor)
        while (off // BANK_STRIDE) % BANKS != want:
            off += BANK_STRIDE
        out.append((stem, var, dtype, count, note, off, align_up(nbytes)))
        cursor = off + align_up(nbytes)
    return out, cursor


def const_name(stem):
    return "k" + "".join(p.title() for p in stem.split("_")) + "Off"


def emit_header(placed_all, totals):
    L = []
    L.append("// GENERATED by scripts/gen_arch35_ub_layout.py -- do not edit by hand.")
    L.append("//")
    L.append("// Byte-exact static Unified Buffer layout for the arch35 Stage-1 kernels.")
    L.append("// Every offset is 32-byte aligned and carries bank index (offset/32)%8 equal")
    L.append("// to its declaration index mod 8, so two buffers share a UB bank only when")
    L.append("// their declaration indices differ by a multiple of 8 (AKA3003). 950PR UB has")
    L.append("// 8 banks on a 32-byte stride, so a collision means a 256 B offset multiple.")
    L.append("")
    L.append("#ifndef MOE_KERNELS_ARCH35_UB_LAYOUT_H")
    L.append("#define MOE_KERNELS_ARCH35_UB_LAYOUT_H")
    L.append("")
    L.append("#include <cstdint>")
    L.append("")
    L.append("namespace moe::kernels::arch35 {")
    for kernel, (entry, _) in KERNELS.items():
        placed, total = placed_all[kernel], totals[kernel]
        L.append("")
        L.append(f"// --- {kernel}: {entry} {'-' * max(0, 40 - len(entry))}")
        L.append(f"namespace {kernel.lower()}_ub {{")
        width = max(len(const_name(p[0])) for p in placed) + 1
        for stem, var, dtype, count, note, off, size in placed:
            L.append(f"constexpr std::uint32_t {const_name(stem):<{width}}= {off:>7}U;"
                     f"  // {size:>6} B  bank {bank(off)}  {note}")
            L.append(f"constexpr std::uint32_t {('k' + ''.join(p.title() for p in stem.split('_')) + 'Count'):<{width}}= {count:>7}U;")
        L.append(f"constexpr std::uint32_t {'kTotalBytes':<{width}}= {total:>7}U;")
        L.append(f"}}  // namespace {kernel.lower()}_ub")
    L.append("")
    L.append("// Every layout must fit the 950PR usable tensor budget: 256 KiB physical UB")
    L.append("// less the 32 KiB SIMT DataCache floor and the 8 KiB compiler reserve.")
    for kernel in KERNELS:
        L.append(f"static_assert({kernel.lower()}_ub::kTotalBytes <= {USABLE_UB}U,")
        L.append(f"              \"{kernel} static UB layout exceeds the 216 KiB arch35 ceiling\");")
    L.append("")
    L.append("}  // namespace moe::kernels::arch35")
    L.append("")
    L.append("#endif  // MOE_KERNELS_ARCH35_UB_LAYOUT_H")
    return "\n".join(L) + "\n"


def emit_annotations(kernel, placed, total):
    L = [BEGIN,
         f"// {kernel} static UB layout: {total} B of {USABLE_UB} B usable "
         f"({100.0 * total / USABLE_UB:.1f}%).",
         "// These restore the analyzer's UB capacity, alignment, aliasing and bank",
         "// checks, which it otherwise skips for static-mode LocalTensor offsets."]
    by_stem = {stem: (var, dtype, count, off) for stem, var, dtype, count, _n, off, _s in placed}
    for stem, var, dtype, count, note, off, size in placed:
        L.append(f"// @ascend-layout: name={var} pos=VECCALC offset={off} "
                 f"count={count} dtype={dtype}")
    aliases = ALIASES.get(kernel, [])
    if aliases:
        L.append("// SIMT raw-pointer aliases of the tensors above, declared so the")
        L.append("// analyzer can still verify their extents, and grouped so the")
        L.append("// deliberate overlap is not reported as accidental aliasing.")
        for alias_var, stem in aliases:
            src_var, dtype, count, off = by_stem[stem]
            L.append(f"// @ascend-layout: name={alias_var} pos=VECCALC offset={off} "
                     f"count={count} dtype={dtype}")
            L.append(f"// @ascend-reuse-group: group=simt_{alias_var} "
                     f"names={src_var},{alias_var}")
    L.append(END)
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", required=True)
    ap.add_argument("--inject", help="op_kernel directory holding the kernel sources")
    ap.add_argument("--check", action="store_true",
                    help="fail if an artefact is out of date instead of writing it")
    args = ap.parse_args()

    placed_all, totals = {}, {}
    for kernel, (entry, buffers) in KERNELS.items():
        placed, total = place(buffers)
        placed_all[kernel], totals[kernel] = placed, total

    header = emit_header(placed_all, totals)
    stale = []
    if args.check:
        try:
            if open(args.header).read() != header:
                stale.append(args.header)
        except FileNotFoundError:
            stale.append(args.header)
    else:
        with open(args.header, "w") as fh:
            fh.write(header)

    if args.inject:
        for kernel, (entry, _) in KERNELS.items():
            path = f"{args.inject}/{entry}.cpp"
            try:
                src = open(path).read()
            except FileNotFoundError:
                sys.stderr.write(f"{kernel:<7} SKIP (no {path} yet)\n")
                continue
            block = emit_annotations(kernel, placed_all[kernel], totals[kernel])
            pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n",
                                 re.DOTALL)
            if not pattern.search(src):
                sys.stderr.write(f"{kernel:<7} ERROR: no generated marker block in {path}\n")
                return 2
            updated = pattern.sub(lambda _m: block, src)
            if args.check:
                if updated != src:
                    stale.append(path)
            elif updated != src:
                with open(path, "w") as fh:
                    fh.write(updated)

    if stale:
        sys.stderr.write("STALE GENERATED ARTEFACTS (re-run without --check):\n")
        for path in stale:
            sys.stderr.write(f"  {path}\n")
        return 1

    for kernel, total in totals.items():
        pct = 100.0 * total / USABLE_UB
        flag = "OK" if total <= USABLE_UB else "OVERFLOW"
        sys.stderr.write(f"{kernel:<7} {total:>7} B  ({total / 1024:6.1f} KiB, "
                         f"{pct:5.1f}% of 216 KiB)  {flag}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
