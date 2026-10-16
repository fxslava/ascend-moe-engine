/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The W4A16 golden for src/ops/dsv4_moe_expert/op_kernel/
// dsv4_moe_prefill_cube_kernel.cpp -- MXFP4 weights dequantized to BF16 and
// activations ROUNDED to BF16, which is the function the Cube's BF16 tensor
// path actually computes.
//
// WHY THIS IS A SECOND ORACLE AND NOT A FLAG ON THE FIRST
// moe_prefill_batched_mxfp4_cpu is W4A32: it reduces FP32 activations against
// dequantized weights, which is what the AIV kernels do because they decode
// nibbles into FP32 registers. The Cube cannot be driven that way -- its BF16
// path takes BF16 on both sides -- so `dsv4_moe_prefill_cube` computes a
// different function and needs its own reference. Collapsing the two behind a
// boolean would hide which one a given test is asserting, which is the
// distinction the whole numerics argument rests on.
//
// THE ARITHMETIC THIS MIRRORS, step for step, from the kernel:
//
//   xb [m, c] = bf16_rne(x[m, c])                              Vector, Cast
//   gate[m,r] = sum_c xb[m,c] * W1b[r,c]      FP32 accumulate  Cube
//   up  [m,r] = sum_c xb[m,c] * W3b[r,c]      FP32 accumulate  Cube
//   gate      = min(gate, L)                                   Vector, Mins
//   up        = min(L, max(-L, up))                            Vector, Maxs/Mins
//   a   [m,r] = bf16_rne(silu(gate) * up)                      Vector, Silu/Mul/Cast
//   y   [m,h] = sum_r a[m,r] * W2b[h,r]       FP32 accumulate  Cube
//
// Two places where this file is deliberately literal about the hardware:
//
//   * the clamp is ASYMMETRIC -- the gate is bounded ABOVE only, the up
//     projection on both sides. Shared with ExpertSwiglu in moe_block.cpp by
//     construction rather than by a second transcription: this file calls the
//     same arithmetic in the same order.
//   * the BF16 rounding of `activated` happens BEFORE the down GEMM, not
//     after the SwiGLU's multiply as a presentation detail. It is the second
//     of the path's two lossy steps and dropping it would make this oracle
//     stricter than the kernel can be.
//
// WHAT THIS ORACLE DOES NOT MODEL, stated because a tolerance that pretends
// otherwise is how an oracle stops being one:
//
//   * the Cube's K-reduction ORDER. L0C accumulates in FP32 over baseK
//     chunks; this file sums in natural column order. The difference is FP32
//     reassociation, bounded by cols * 2^-24 relative (~3e-5 at cols = 512)
//     and far inside any tolerance a BF16 path can carry.
//   * the DEVICE's silu. `AscendC::Silu` is a vector intrinsic over the
//     hardware's exp/reciprocal, not std::exp, and its error is not derivable
//     from the format. It is the one term in the comparison bound that is
//     measured rather than derived, and the CAModel driver says so where it
//     sets its tolerance.

#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("moe_prefill_w4a16: " + message);
  }
}

// silu(z) = z * sigmoid(z). Byte-for-byte the guard in moe_block.cpp's Silu:
// the two oracles must not disagree about the activation, only about where
// the BF16 rounding sits.
float Silu(float z) noexcept {
  if (z < -80.0F) {
    return 0.0F;
  }
  return z / (1.0F + std::exp(-z));
}

}  // namespace

// ---------------------------------------------------------------------------
// BF16 rounding
// ---------------------------------------------------------------------------

float bf16_round_cpu(float value) noexcept {
  return Bf16ToFloat(FloatToBf16(value));
}

bool Mxfp4DequantIsExactInBf16(std::uint8_t nibble,
                               std::uint8_t scale_code) noexcept {
  // The NaN scale code has no value to be exact about, and the device helper
  // turns it into an infinity rather than a NaN (see the note on DecodeE8M0 in
  // include/moe/kernels/arch35/arch35_common.hpp). Outside the domain either
  // way.
  if (scale_code == 0xFFU) {
    return false;
  }
  const float value = DecodeFp4E2M1(nibble) * DecodeFp8E8M0(scale_code);
  if (value == 0.0F) {
    return true;  // a zero nibble is exact at every scale
  }
  if (!std::isfinite(value)) {
    return false;  // overflowed FP32 on the way, let alone BF16
  }
  // Below BF16's smallest NORMAL (2^-126) the bit layouts still align, so the
  // value IS representable -- but the device flushes subnormals to zero, so
  // this is outside the domain of agreement even though it is inside the
  // domain of representability. Excluded for that reason, not for a rounding
  // one.
  if (std::fabs(value) < 1.17549435e-38F) {
    return false;
  }
  // BF16 shares binary32's 8-bit exponent and keeps 8 significand bits, so
  // what remains to check is that an RNE round to BF16 precision is the
  // identity here. For the E2M1 set (three significant bits at most) times a
  // power of two it must be, and this is where that is established rather
  // than argued.
  return bf16_round_cpu(value) == value;
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

void mxfp4_dequant_row_bf16_cpu(const Mxfp4Matrix& weight, int row, float* out) {
  Require(weight.packed != nullptr && weight.scale != nullptr,
          "fp4 matrix is null");
  Require(out != nullptr, "dequant destination is null");
  Require(row >= 0 && row < weight.rows, "fp4 row out of range");
  Require(weight.cols % kMxfp4GroupSize == 0,
          "fp4 logical columns must be a multiple of the 32-element group");

  const Fp8Table& scales = Fp8E8M0Table();
  const int groups = weight.cols / kMxfp4GroupSize;
  const std::uint8_t* packed_row =
      weight.packed + static_cast<std::size_t>(row) * (weight.cols / 2);
  const std::uint8_t* scale_row =
      weight.scale + static_cast<std::size_t>(row) * groups;

  for (int group = 0; group < groups; ++group) {
    const float scale = scales[scale_row[group]];
    const int first = group * kMxfp4GroupSize;
    for (int col = first; col < first + kMxfp4GroupSize; col += 2) {
      const std::uint8_t byte = packed_row[col / 2];
      // kMxfp4LowNibbleFirst: the low nibble is the EVEN logical column. The
      // convention is source-backed in dsv4_layers.h and spelled here rather
      // than taken from a table so this file carries no second copy of it.
      const std::uint8_t low = static_cast<std::uint8_t>(byte & 0x0FU);
      const std::uint8_t high = static_cast<std::uint8_t>(byte >> 4U);
      const std::uint8_t even = kMxfp4LowNibbleFirst ? low : high;
      const std::uint8_t odd = kMxfp4LowNibbleFirst ? high : low;
      // The BF16 round is what the hardware's Cast-then-Muls produces. Inside
      // the exact domain it changes nothing, which is the claim the kernel
      // rests on; outside it, this oracle rounds the same way the device does
      // rather than keeping FP32 precision the device does not have.
      out[col] = bf16_round_cpu(DecodeFp4E2M1(even) * scale);
      out[col + 1] = bf16_round_cpu(DecodeFp4E2M1(odd) * scale);
    }
  }
}

// ---------------------------------------------------------------------------
// The W4A16 GEMV
// ---------------------------------------------------------------------------

void mxfp4_gemv_w4a16_cpu(const Mxfp4Matrix& weight, const float* x_bf16,
                          float* y) {
  Require(weight.packed != nullptr && weight.scale != nullptr,
          "fp4 matrix is null");
  Require(x_bf16 != nullptr && y != nullptr, "w4a16 gemv operand is null");
  Require(weight.rows > 0 && weight.cols > 0, "fp4 matrix has an empty extent");
  Require(weight.cols % kMxfp4GroupSize == 0,
          "fp4 logical columns must be a multiple of the 32-element group");

  const Fp8Table& scales = Fp8E8M0Table();
  const int groups = weight.cols / kMxfp4GroupSize;

  for (int row = 0; row < weight.rows; ++row) {
    const std::uint8_t* packed_row =
        weight.packed + static_cast<std::size_t>(row) * (weight.cols / 2);
    const std::uint8_t* scale_row =
        weight.scale + static_cast<std::size_t>(row) * groups;
    // ONE running FP32 accumulator over the whole reduction axis, with the
    // scale DISTRIBUTED into every product. That is the Cube's only option:
    // the GEMM sees dequantized BF16 weights and has no notion of the group
    // the scale came from. moe_block.cpp's W4A32 gemv instead scales each
    // group's subtotal, and the difference between the two is exactly FP32
    // reassociation -- see the header note.
    float total = 0.0F;
    for (int group = 0; group < groups; ++group) {
      const float scale = scales[scale_row[group]];
      const int first = group * kMxfp4GroupSize;
      for (int col = first; col < first + kMxfp4GroupSize; col += 2) {
        const std::uint8_t byte = packed_row[col / 2];
        const std::uint8_t low = static_cast<std::uint8_t>(byte & 0x0FU);
        const std::uint8_t high = static_cast<std::uint8_t>(byte >> 4U);
        const std::uint8_t even = kMxfp4LowNibbleFirst ? low : high;
        const std::uint8_t odd = kMxfp4LowNibbleFirst ? high : low;
        total += bf16_round_cpu(DecodeFp4E2M1(even) * scale) * x_bf16[col];
        total += bf16_round_cpu(DecodeFp4E2M1(odd) * scale) * x_bf16[col + 1];
      }
    }
    y[row] = total;
  }
}

// ---------------------------------------------------------------------------
// The expert, and the batch over it
// ---------------------------------------------------------------------------

void moe_expert_swiglu_mxfp4_w4a16_cpu(const Mxfp4Matrix& w1,
                                       const Mxfp4Matrix& w3,
                                       const Mxfp4Matrix& w2, const float* x,
                                       float* out, float limit,
                                       std::vector<float>* scratch) {
  Require(w1.rows == w3.rows, "w1 and w3 must agree on the intermediate size");
  Require(w1.cols == w3.cols, "w1 and w3 must agree on the hidden size");
  Require(w2.cols == w1.rows, "w2's input must be the intermediate size");
  Require(x != nullptr && out != nullptr, "expert operand is null");
  Require(scratch != nullptr, "expert scratch is null");

  const int hidden = w1.cols;
  const int inter = w1.rows;
  Require(hidden > 0 && inter > 0, "expert has an empty extent");

  // One owned buffer: the BF16 copy of x, then gate, then up.
  scratch->resize(static_cast<std::size_t>(hidden) +
                  static_cast<std::size_t>(inter) * 2);
  float* xb = scratch->data();
  float* gate = scratch->data() + hidden;
  float* up = scratch->data() + hidden + inter;

  // LOSSY STEP 1 of 2. The kernel casts the whole row once into its workspace
  // plane and both GEMMs read that plane, so ONE rounding feeds both -- not
  // one per GEMM, which would be a different (and slightly worse) function.
  for (int c = 0; c < hidden; ++c) {
    xb[c] = bf16_round_cpu(x[c]);
  }

  mxfp4_gemv_w4a16_cpu(w1, xb, gate);
  mxfp4_gemv_w4a16_cpu(w3, xb, up);

  for (int i = 0; i < inter; ++i) {
    // AscendSiluAndMulWithClamp: gate bounded ABOVE only, up on both sides.
    const float clamped_gate = std::min(gate[i], limit);
    const float clamped_up = std::min(limit, std::max(-limit, up[i]));
    // LOSSY STEP 2 of 2: `activated` is stored BF16 because it is the down
    // GEMM's A matrix and the Cube reads BF16.
    gate[i] = bf16_round_cpu(Silu(clamped_gate) * clamped_up);
  }

  mxfp4_gemv_w4a16_cpu(w2, gate, out);
}

void moe_prefill_batched_mxfp4_w4a16_cpu(const Mxfp4Matrix* w1,
                                         const Mxfp4Matrix* w3,
                                         const Mxfp4Matrix* w2,
                                         const float* x_tokens,
                                         const std::int32_t* token_to_expert,
                                         int total_tokens, int num_experts,
                                         float* y_tokens, float limit,
                                         std::vector<float>* scratch) {
  Require(w1 != nullptr && w3 != nullptr && w2 != nullptr,
          "prefill expert weight array is null");
  Require(x_tokens != nullptr && y_tokens != nullptr, "prefill operand is null");
  Require(token_to_expert != nullptr, "prefill routing operand is null");
  Require(scratch != nullptr, "prefill scratch is null");
  Require(total_tokens > 0 && num_experts > 0, "prefill has an empty extent");

  const int hidden = w2[0].rows;
  const int inter = w1[0].rows;
  // Every expert must share one geometry: the device addresses expert e at a
  // FIXED stride, so a ragged expert set reads the wrong bytes rather than
  // merely running slowly.
  for (int e = 0; e < num_experts; ++e) {
    Require(w1[e].rows == inter && w3[e].rows == inter,
            "every expert must share the intermediate size");
    Require(w2[e].rows == hidden, "every expert must share the hidden size");
    Require(w1[e].cols == hidden && w3[e].cols == hidden,
            "w1/w3 must reduce over the hidden size");
    Require(w2[e].cols == inter, "w2 must reduce over the intermediate size");
  }

  // Rows are independent -- no router weighting and no cross-token state --
  // so the token tiling the kernel uses cannot change the answer. That is the
  // property the collapsed CAModel grid in tests/sim relies on: a sampled
  // tile's rows are a function of those rows alone.
  for (int t = 0; t < total_tokens; ++t) {
    const std::int32_t e = token_to_expert[t];
    Require(e >= 0 && e < num_experts, "token's expert id is out of range");
    const std::size_t row =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(hidden);
    moe_expert_swiglu_mxfp4_w4a16_cpu(w1[e], w3[e], w2[e], x_tokens + row,
                                      y_tokens + row, limit, scratch);
  }
}


// ---------------------------------------------------------------------------
// The FLASH-STYLE TILED expert: the same function, computed without ever
// materializing the intermediate
// ---------------------------------------------------------------------------
//
// WHY A SECOND SHAPE OF THE SAME ARITHMETIC
// The monolithic form above computes all `inter` gate/up values, activates
// them, and only then runs the down GEMM. On device that costs a resident
// intermediate of M * inter FP32 words and -- in the whole-weight staging
// variant -- a GM workspace holding W1/W3/W2 dequantized to BF16, which for
// one expert at hidden 4096 / inter 2048 is tens of megabytes of traffic for
// weights that are read exactly once.
//
// The tiled form slices the CONTRACTION dimension `inter` into chunks of R and
// keeps nothing but one chunk alive:
//
//   Y[.] = 0
//   for r0 in 0, R, 2R, ... < inter:
//     R' = min(R, inter - r0)                     the final chunk may be ragged
//     gate[0:R'] = W1b[r0:r0+R', :] @ xb          dequantized on the fly
//     up  [0:R'] = W3b[r0:r0+R', :] @ xb          dequantized on the fly
//     act [0:R'] = bf16(silu(min(gate, L)) * clamp(up, -L, L))
//     Y[.]      += W2b[:, r0:r0+R'] @ act[0:R']   dequantized on the fly
//
// WHAT IS AND IS NOT THE SAME FUNCTION AS THE MONOLITHIC ORACLE -- the whole
// reason this file carries both:
//
//   * gate, up and act are BIT-IDENTICAL at every chunk size. Each gate[r] is
//     a full reduction over the hidden axis of ROW r alone, and the rows are
//     independent, so slicing the OUTPUT rows of W1/W3 cannot reorder
//     anything. Nothing about R enters.
//   * the down GEMM's accumulation order DOES change. The monolithic form
//     sums `inter` products into one running FP32 accumulator; the tiled form
//     sums each chunk into its own accumulator and adds that subtotal to Y.
//     That is a two-level tree against a flat sum -- pure FP32
//     reassociation, and the ONLY difference between the two functions.
//   * the two extremes are therefore bit-exact, for opposite reasons:
//     R >= inter is a single chunk whose accumulator IS the flat sum, and
//     R == 1 adds every product to Y separately, which is also the flat sum.
//     Any 1 < R < inter is the genuinely reassociated case. tests/emu asserts
//     all three, which is how this claim is checked rather than asserted.
//
// WHAT THIS MODELS ABOUT THE DEVICE, AND WHAT IT STILL DOES NOT
// It models the chunk structure -- the loop order, where the activation
// happens, and that Y accumulates subtotals. It does NOT model the Cube's
// reduction order WITHIN a chunk (L0C accumulates over baseK tiles, this file
// sums in natural column order), exactly as the monolithic oracle does not.
// The difference is FP32 reassociation of the same products; the host cannot
// narrow it and must not pretend to.
//
// ON-THE-FLY DEQUANTIZATION IS THE POINT, so this path decodes one element at
// the moment it is multiplied. There is no dequantized weight buffer of any
// size -- not a row, not a tile -- and the scratch footprint is
// `hidden + 3 * min(R, inter)` floats, which does not grow with `inter`.
// `moe_prefill_flash_scratch_floats()` is that number, exposed so a test can
// hold the no-materialization claim to an arithmetic statement instead of a
// comment.

// One logical element of an MXFP4 matrix, dequantized the way the Cube path
// does it. Deliberately the SAME three operations in the same order as the
// inner loop of mxfp4_gemv_w4a16_cpu -- decode the nibble, multiply by the
// group's power-of-two scale, hold the product at BF16 -- so that a tiled
// path cannot drift from the monolithic one in the weight decode. The only
// thing that may differ between the two is the order of the ADDITIONS.
//
// PUBLIC so that the 1D-strip reference below and the 2D block-tiled Cube
// model in moe_prefill_cube_2d.cpp share one decode rather than carrying two
// transcriptions of the nibble order and the group-scale indexing. Unchecked
// on purpose -- it is an inner-loop primitive, and every caller in this
// library validates the matrix and the indices before entering its loops.
float mxfp4_dequant_element_bf16_cpu(const Mxfp4Matrix& w, int row,
                                     int col) noexcept {
  const std::size_t byte_stride = static_cast<std::size_t>(w.cols / 2);
  const std::size_t group_stride =
      static_cast<std::size_t>(w.cols / kMxfp4GroupSize);
  const std::uint8_t byte =
      w.packed[static_cast<std::size_t>(row) * byte_stride +
               static_cast<std::size_t>(col / 2)];
  const std::uint8_t low = static_cast<std::uint8_t>(byte & 0x0FU);
  const std::uint8_t high = static_cast<std::uint8_t>(byte >> 4U);
  const std::uint8_t even_nibble = kMxfp4LowNibbleFirst ? low : high;
  const std::uint8_t odd_nibble = kMxfp4LowNibbleFirst ? high : low;
  const std::uint8_t nibble = (col % 2 == 0) ? even_nibble : odd_nibble;
  const float scale =
      Fp8E8M0Table()[w.scale[static_cast<std::size_t>(row) * group_stride +
                             static_cast<std::size_t>(col / kMxfp4GroupSize)]];
  return bf16_round_cpu(DecodeFp4E2M1(nibble) * scale);
}

std::size_t moe_prefill_flash_scratch_floats(int hidden, int inter,
                                             int chunk) noexcept {
  if (hidden <= 0 || inter <= 0 || chunk <= 0) {
    return 0;
  }
  // xb[hidden] + gate[R'] + up[R'] + act[R'], with R' the effective chunk.
  // `inter` appears only inside the min, which is the claim: the resident set
  // is set by the CHUNK, not by the intermediate size.
  const std::size_t effective =
      static_cast<std::size_t>(chunk < inter ? chunk : inter);
  return static_cast<std::size_t>(hidden) + 3U * effective;
}

void moe_expert_swiglu_mxfp4_w4a16_flash_cpu(const Mxfp4Matrix& w1,
                                             const Mxfp4Matrix& w3,
                                             const Mxfp4Matrix& w2,
                                             const float* x, float* out,
                                             float limit, int chunk,
                                             std::vector<float>* scratch) {
  Require(w1.packed != nullptr && w1.scale != nullptr, "w1 is null");
  Require(w3.packed != nullptr && w3.scale != nullptr, "w3 is null");
  Require(w2.packed != nullptr && w2.scale != nullptr, "w2 is null");
  Require(w1.rows == w3.rows, "w1 and w3 must agree on the intermediate size");
  Require(w1.cols == w3.cols, "w1 and w3 must agree on the hidden size");
  Require(w2.cols == w1.rows, "w2's input must be the intermediate size");
  // And w2's OUTPUT extent, which the tiled form needs stated explicitly: Y
  // is indexed over `hidden` rows of w2 inside the chunk loop, so a w2 with
  // fewer rows would be read past its end rather than merely producing a
  // short answer.
  Require(w2.rows == w1.cols, "w2's output must be the hidden size");
  Require(x != nullptr && out != nullptr, "expert operand is null");
  Require(scratch != nullptr, "expert scratch is null");
  // A nonpositive chunk is a CALLER MISTAKE, not a geometry this reference
  // declines: it would make the loop below either spin forever or skip the
  // reduction and return a zeroed output, which is the one failure mode an
  // oracle must never have.
  Require(chunk > 0, "the intermediate chunk R must be positive");

  const int hidden = w1.cols;
  const int inter = w1.rows;
  Require(hidden > 0 && inter > 0, "expert has an empty extent");
  Require(hidden % kMxfp4GroupSize == 0 && inter % kMxfp4GroupSize == 0,
          "both reduction axes must be whole 32-element MXFP4 groups");

  const int span = chunk < inter ? chunk : inter;
  scratch->resize(moe_prefill_flash_scratch_floats(hidden, inter, chunk));
  float* xb = scratch->data();
  float* gate = xb + hidden;
  float* up = gate + span;
  float* act = up + span;

  // LOSSY STEP 1 of 2, and it happens ONCE for the whole token however many
  // chunks follow: the kernel casts the row into UB and every chunk's gate/up
  // GEMM reads that one copy. Rounding per chunk would be a different (and
  // worse) function.
  for (int c = 0; c < hidden; ++c) {
    xb[c] = bf16_round_cpu(x[c]);
  }

  // Y is the accumulator. It starts at zero and is only ever ADDED to, which
  // is what lets the intermediate never exist in full.
  for (int h = 0; h < hidden; ++h) {
    out[h] = 0.0F;
  }

  for (int r0 = 0; r0 < inter; r0 += span) {
    const int rows = (inter - r0) < span ? (inter - r0) : span;

    // gate / up: W1 and W3 row slices against the one BF16 copy of x. Rows
    // are independent, so this is the monolithic GEMV restricted to a row
    // range -- same products, same column order, same single accumulator.
    for (int i = 0; i < rows; ++i) {
      const int row = r0 + i;
      float gate_total = 0.0F;
      float up_total = 0.0F;
      for (int c = 0; c < hidden; ++c) {
        gate_total += mxfp4_dequant_element_bf16_cpu(w1, row, c) * xb[c];
        up_total += mxfp4_dequant_element_bf16_cpu(w3, row, c) * xb[c];
      }
      gate[i] = gate_total;
      up[i] = up_total;
    }

    // The clamped SwiGLU, on the chunk, in registers. Asymmetric exactly as
    // in the monolithic expert: the gate is bounded ABOVE only, the up
    // projection on both sides. LOSSY STEP 2 of 2 -- `act` is the down GEMM's
    // A matrix and the Cube reads BF16.
    for (int i = 0; i < rows; ++i) {
      const float clamped_gate = std::min(gate[i], limit);
      const float clamped_up = std::min(limit, std::max(-limit, up[i]));
      act[i] = bf16_round_cpu(Silu(clamped_gate) * clamped_up);
    }

    // The down GEMM, accumulated into Y. Each chunk's contribution is summed
    // in its OWN FP32 accumulator and then added -- the two-level tree the
    // header note describes, and the only place the tiled form can differ
    // from the monolithic one.
    for (int h = 0; h < hidden; ++h) {
      float partial = 0.0F;
      for (int i = 0; i < rows; ++i) {
        partial += act[i] * mxfp4_dequant_element_bf16_cpu(w2, h, r0 + i);
      }
      out[h] += partial;
    }
  }
}

void moe_prefill_flash_ffn_w4a16_cpu(const Mxfp4Matrix* w1,
                                     const Mxfp4Matrix* w3,
                                     const Mxfp4Matrix* w2,
                                     const float* x_tokens,
                                     const std::int32_t* token_to_expert,
                                     int total_tokens, int num_experts,
                                     float* y_tokens, float limit, int chunk,
                                     std::vector<float>* scratch) {
  Require(w1 != nullptr && w3 != nullptr && w2 != nullptr,
          "prefill expert weight array is null");
  Require(x_tokens != nullptr && y_tokens != nullptr, "prefill operand is null");
  Require(token_to_expert != nullptr, "prefill routing operand is null");
  Require(scratch != nullptr, "prefill scratch is null");
  Require(total_tokens > 0 && num_experts > 0, "prefill has an empty extent");
  Require(chunk > 0, "the intermediate chunk R must be positive");

  const int hidden = w2[0].rows;
  const int inter = w1[0].rows;
  // Same fixed-stride argument as the monolithic batch: the device addresses
  // expert e at a fixed offset, so a ragged expert set reads the wrong bytes
  // rather than merely running slowly.
  for (int e = 0; e < num_experts; ++e) {
    Require(w1[e].rows == inter && w3[e].rows == inter,
            "every expert must share the intermediate size");
    Require(w2[e].rows == hidden, "every expert must share the hidden size");
    Require(w1[e].cols == hidden && w3[e].cols == hidden,
            "w1/w3 must reduce over the hidden size");
    Require(w2[e].cols == inter, "w2 must reduce over the intermediate size");
  }

  // Rows stay independent under tiling -- the chunk loop is inside the token
  // loop and carries no state across tokens -- so the token tiling the kernel
  // chooses still cannot change the answer.
  for (int t = 0; t < total_tokens; ++t) {
    const std::int32_t e = token_to_expert[t];
    Require(e >= 0 && e < num_experts, "token's expert id is out of range");
    const std::size_t row =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(hidden);
    moe_expert_swiglu_mxfp4_w4a16_flash_cpu(w1[e], w3[e], w2[e], x_tokens + row,
                                            y_tokens + row, limit, chunk,
                                            scratch);
  }
}

}  // namespace moe::emu
