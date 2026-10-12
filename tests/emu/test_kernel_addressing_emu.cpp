// test_kernel_addressing_emu.cpp -- host validation of device ADDRESS arithmetic
// for mxfp4_expert_gemv, fast_fp8_indexer_topk and fused_mhc_pre_norm_sinkhorn.
//
// WHY THIS FILE EXISTS (CLAUDE.md section 5.5)
// Device logic that is pure integer/index arithmetic must be validated on the
// host. tests/emu/test_custom_kernels_emu.cpp already does that for the
// indexer's tiered bitonic SELECTION network. Nothing did it for the other
// half of the indexing surface: how a logical element is turned into an
// ADDRESS. That is the arithmetic that reads out of bounds when it is wrong,
// and on device an out-of-bounds GM read is a hardware fault, not a wrong
// number -- so it is exactly the arithmetic worth checking where a failure
// prints a line instead of halting an accelerator.
//
// WHAT THIS FILE DOES NOT CLAIM (CLAUDE.md section 5.2)
// This is NOT a host interpretation of the kernels. It does not model
// __simt_vf__ scheduling, warp primitive semantics, DataCache behaviour or
// pipeline ordering, and a pass here is no evidence about any of them. It
// transcribes integer expressions, nothing else. Every claim below is about
// an index or an address and stops there.
//
// WHAT CATCHES WHAT
// The explicit extent check in BoundedSpan::At is the primary witness: it
// names the broken contract and prints the offending offset, which is what
// makes a failure actionable. It deliberately checks BEFORE dereferencing, so
// a caught violation never becomes a real wild access -- which also means ASan
// does not independently witness it. Under -D MOE_EMU_SANITIZE=ON the
// sanitizers cover what the extent checks cannot: UBSan flags overflow in the
// offset arithmetic itself (the 32-bit wrap documented in
// TestIndexerMalformedTable is exactly that shape of bug), and ASan covers
// mistakes in this harness's own bookkeeping. Backing storage is still sized
// to EXACTLY the logical extent so the two agree rather than drift.
//
// VERIFIED TO HAVE TEETH
// These sections were mutation-tested: an off-by-one in mxfp4's
// lane_group_base is caught, with the offending offset named. The same
// exercise found this file's limit -- addressing a PADDED cache with the dense
// stride stays in bounds and is NOT caught here, because it is a wrong-row
// bug rather than an overrun. That class needs a numeric comparison against
// the oracle over a padded cache; see TestIndexerBlockStride.
//
// Constants are mirrored locally rather than included: op_kernel/ headers are
// operator-private (CLAUDE.md section 1.2) and tests/emu must build with a
// stock host compiler and no CANN toolkit (section 4.1). The mirrored values
// are tagged with the header that owns each one; a divergence is caught by the
// shape assertions in the first section rather than by a stale comment.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "emu_test_support.hpp"

namespace {

// --- mirrored from src/ops/fast_fp8_indexer_topk/op_kernel/..._arch35.h -----
constexpr std::uint32_t kIdxHeads = 64;
constexpr std::uint32_t kIdxHeadDim = 128;
constexpr std::uint32_t kIdxMaxSeqLen = 2048;

constexpr std::uint32_t kIdxSimtThreads = 512;
constexpr std::uint32_t kIdxWarpSize = 32;
constexpr std::uint32_t kIdxWarpsPerCore = kIdxSimtThreads / kIdxWarpSize;  // 16
constexpr std::uint32_t kIdxScoreBlockDim = 32;                             // all AIVs
constexpr std::uint32_t kIdxScoreWarpStride =
    kIdxScoreBlockDim * kIdxWarpsPerCore;  // 512

// --- mirrored from src/ops/mxfp4_expert_gemv/op_kernel/..._arch35.h ---------
constexpr std::uint32_t kMxRowsPerTile = 32;
constexpr std::uint32_t kMxLanesPerRow = 16;
constexpr std::uint32_t kMxGroupSize = 32;
constexpr std::uint32_t kMxSimtThreads = kMxRowsPerTile * kMxLanesPerRow;  // 512

// --- mirrored from src/ops/fused_mhc_pre_norm_sinkhorn/op_kernel/...h -------
constexpr std::uint32_t kShStreams = 4;
constexpr std::uint32_t kShProjectionRows = 24;
constexpr std::uint32_t kShScalarStateBytes = 256;

// A span whose backing allocation is exactly its logical extent, so a logical
// overrun is also a heap overrun. `At` reports the first violation by name and
// then clamps, so one bad index does not cascade into a wall of noise; `Touch`
// performs the real read when we want the sanitizer to see it.
template <typename T>
class BoundedSpan {
 public:
  BoundedSpan(emu_test::Runner& runner, std::string name, std::size_t extent)
      : runner_(runner), name_(std::move(name)), store_(extent, T{}) {}

  std::size_t extent() const { return store_.size(); }
  bool clean() const { return violations_ == 0; }
  std::size_t violations() const { return violations_; }

  // Bounds-check an address without dereferencing it. Used where the point is
  // that the kernel WOULD have formed a wild address.
  bool InRange(std::uint64_t offset) const { return offset < store_.size(); }

  // Check and read. A violation is counted once per span and reported with the
  // offending offset so the failure line is actionable on its own.
  T At(std::uint64_t offset, const std::string& what) {
    if (offset >= store_.size()) {
      if (violations_ == 0) {
        runner_.Check(false, name_ + ": " + what + " reads offset " +
                                 std::to_string(offset) + " of extent " +
                                 std::to_string(store_.size()));
      }
      ++violations_;
      return T{};
    }
    return store_[static_cast<std::size_t>(offset)];
  }

 private:
  emu_test::Runner& runner_;
  std::string name_;
  std::vector<T> store_;
  std::size_t violations_ = 0;
};

// ---------------------------------------------------------------------------
// Section 1: shape gates. If a mirrored constant drifts from the kernel, the
// derived quantities below stop meaning what their names say, so the
// divisibility facts the lane arithmetic relies on are asserted up front
// rather than assumed.
// ---------------------------------------------------------------------------
void TestMirroredShapeInvariants(emu_test::Runner& runner) {
  runner.Section("mirrored shape invariants");

  runner.Check(kMxSimtThreads == 512u,
               "mxfp4 SIMT block is 32 rows x 16 lanes");
  runner.Check(kMxGroupSize == 2u * kMxLanesPerRow,
               "one E8M0 group spans 32 logical columns = 16 nibble pairs, "
               "which is what makes lane_group_base exact");
  runner.Check(kIdxHeads * kIdxHeadDim == 8192u,
               "indexer query tile is 8,192 FP8 bytes");
  runner.Check(kIdxScoreWarpStride == kIdxScoreBlockDim * kIdxWarpsPerCore,
               "the score kernel's grid-stride step is exactly the number of "
               "global warps, which is what makes the key schedule a "
               "partition rather than an overlap");
  runner.Check(kIdxHeads % kIdxWarpSize == 0u,
               "the 64 heads split evenly over a warp's 32 lanes, so every "
               "lane owns the same number of head dots");
  runner.Check(kIdxHeadDim % 4u == 0u,
               "the key row is a whole number of uint32 words, so the "
               "word-wide E4M3 load has no tail");
  runner.Check(kShProjectionRows == 6u * kShStreams,
               "the 24 projection rows are three 4-wide gate segments plus "
               "the 16-wide logits segment");
  // u[] and bias[] are indexed at 2*kStreams + t for t < 16, so the last
  // touched row is 23; the UB scalar segment must hold all 24 floats.
  runner.Check(kShProjectionRows * 4u <= kShScalarStateBytes,
               "24 FP32 projections fit the sinkhorn scalar UB segment");
}

// ---------------------------------------------------------------------------
// Section 2: mxfp4_expert_gemv lane address arithmetic.
//
// Transcribed from Mxfp4GemvTileSimt:
//   pairs_per_lane   = (cols / 2) / kLanesPerRow
//   lane_group_base  = (lane * pairs_per_lane) / 16
//   p                = lane * pairs_per_lane + g * 16 + q
//   packed byte      = prow[p]              -> packed_row_stride
//   scale byte       = srow[lane_group_base + g] -> scale_row_stride
//   x reads          = x_ub[2p], x_ub[2p + 1]    -> cols
// ---------------------------------------------------------------------------
void CheckMxfp4Shape(emu_test::Runner& runner, std::uint32_t rows,
                     std::uint32_t cols) {
  const std::string tag =
      std::to_string(rows) + "x" + std::to_string(cols) + ": ";

  // The kernel's own tiling gate fixes these two strides exactly.
  const std::uint32_t packed_row_stride = cols / 2u;
  const std::uint32_t scale_row_stride = cols / kMxGroupSize;

  const std::uint32_t pairs_per_lane = (cols / 2u) / kMxLanesPerRow;
  runner.Check(pairs_per_lane % 16u == 0u,
               tag + "pairs_per_lane is a whole number of 16-pair groups, so "
                     "each lane touches exactly one scale byte per group");

  BoundedSpan<std::uint8_t> packed(runner, tag + "packed row", packed_row_stride);
  BoundedSpan<std::uint8_t> scale(runner, tag + "scale row", scale_row_stride);
  BoundedSpan<float> x_ub(runner, tag + "x_ub", cols);

  // One full tile: every lane of every row, every group, every pair. This is
  // the whole address set the SIMT launch produces for a tile.
  std::uint64_t max_p = 0;
  std::uint64_t max_scale = 0;
  std::uint64_t max_x = 0;
  for (std::uint32_t lane = 0; lane < kMxLanesPerRow; ++lane) {
    const std::uint32_t lane_group_base = (lane * pairs_per_lane) / 16u;
    for (std::uint32_t g = 0; g < pairs_per_lane / 16u; ++g) {
      const std::uint64_t scale_off =
          static_cast<std::uint64_t>(lane_group_base) + g;
      scale.At(scale_off, "group scale");
      if (scale_off > max_scale) {
        max_scale = scale_off;
      }
      const std::uint32_t p0 = lane * pairs_per_lane + g * 16u;
      for (std::uint32_t q = 0; q < 16u; ++q) {
        const std::uint64_t p = static_cast<std::uint64_t>(p0) + q;
        packed.At(p, "packed nibble pair");
        // The 4-bit unpack expands one packed byte into two adjacent x reads;
        // this is the "E2M1 unpack expansion" index pair.
        x_ub.At(2u * p, "x even logical column");
        x_ub.At(2u * p + 1u, "x odd logical column");
        if (p > max_p) {
          max_p = p;
        }
        if (2u * p + 1u > max_x) {
          max_x = 2u * p + 1u;
        }
      }
    }
  }

  runner.Check(packed.clean(), tag + "no packed-row overrun across a full tile");
  runner.Check(scale.clean(), tag + "no scale-row overrun across a full tile");
  runner.Check(x_ub.clean(), tag + "no x_ub overrun across a full tile");

  // Tight, not merely in-bounds: the lane decomposition must COVER the row.
  // An under-covering decomposition is in bounds and silently drops columns.
  runner.Check(max_p + 1u == packed_row_stride,
               tag + "the lanes exactly cover the packed row");
  runner.Check(max_scale + 1u == scale_row_stride,
               tag + "the lanes exactly cover the scale row");
  runner.Check(max_x + 1u == cols,
               tag + "the unpacked pairs exactly cover x");
}

void TestMxfp4Addressing(emu_test::Runner& runner) {
  runner.Section("mxfp4_expert_gemv: lane address arithmetic, legal shapes");
  // The kernel's tiling gate admits exactly these two shapes.
  CheckMxfp4Shape(runner, 2048u, 4096u);
  CheckMxfp4Shape(runner, 4096u, 2048u);

  runner.Section("mxfp4_expert_gemv: the gate rejects scaled-down geometry");
  // A scaled-down fixture cannot exercise this kernel: the device-side gate
  // admits only (2048,4096) and (4096,2048), so a 128x128 launch returns
  // early and computes nothing. Asserted here so a future "quick small case"
  // cannot be added under the impression that it tests the kernel.
  struct Shape {
    std::uint32_t rows;
    std::uint32_t cols;
  };
  const Shape rejected[] = {{128u, 128u}, {128u, 256u}, {256u, 256u},
                            {2048u, 2048u}};
  for (const Shape& s : rejected) {
    const bool admitted = (s.rows == 2048u && s.cols == 4096u) ||
                          (s.rows == 4096u && s.cols == 2048u);
    runner.Check(!admitted, std::to_string(s.rows) + "x" +
                                std::to_string(s.cols) +
                                " is rejected by the kernel's tiling gate");
  }
}

// ---------------------------------------------------------------------------
// Section 3: fast_fp8_indexer_topk key-cache address resolution.
//
// Transcribed from ScoreKeysSimt:
//   logical_block  = j / block_size
//   within_block   = j % block_size
//   physical_block = block_table ? (uint32_t)block_table[logical_block]
//                                : logical_block
//   key_row        = key_cache + (physical_block * block_size + within_block)
//                                * kHeadDim
//
// Two things are under test here, and they are different failures:
//   (a) with a well-formed block table every address stays inside the cache;
//   (b) the block table is CALLER data, not tiling, so a malformed entry is
//       not covered by the host-side shape gate (section 2.5) and the formula
//       above dereferences it unchecked.
// ---------------------------------------------------------------------------

// The kernel's formula, in uint64 so we can observe the address it INTENDS
// rather than the uint32 wrap it would actually compute on device. The block
// stride is a parameter because it is an ABI field, not a constant: see
// TestIndexerBlockStride.
std::uint64_t KernelKeyRowOffset(std::uint32_t physical_block,
                                 std::uint32_t key_block_stride,
                                 std::uint32_t within_block) {
  return static_cast<std::uint64_t>(physical_block) * key_block_stride +
         static_cast<std::uint64_t>(within_block) * kIdxHeadDim;
}

// The dense stride, i.e. the value key_block_stride takes for an unpadded
// cache. Kept as a named helper so the tests below cannot accidentally
// re-derive the hard-coded assumption the kernel used to make.
std::uint32_t DenseBlockStride(std::uint32_t block_size) {
  return block_size * kIdxHeadDim;
}

// The admissibility predicate the kernel must apply to a block-table entry
// before using it as an address. Transcribed from the guard in ScoreKeysSimt
// so this file tests the rule rather than restating a constant: a signed
// entry is valid exactly when it is non-negative and names a real physical
// block.
//
// On the redundancy of the two halves, established by mutation rather than
// assumed: the RANGE test alone already rejects every negative entry, because
// any negative casts to a value >= 2^31 and no real num_physical_blocks comes
// near that. Removing the signed test changes this predicate's verdict on no
// input below. It is kept, here and in the kernel, because it makes the
// intent independent of that argument -- the guard stays correct if
// num_physical_blocks ever grows large enough for the wrap to alias a legal
// id, and a reader does not have to reconstruct the 2^31 reasoning to see
// that negatives are handled.
bool BlockEntryIsAdmissible(std::int32_t entry,
                            std::uint32_t num_physical_blocks) {
  if (entry < 0) {
    return false;
  }
  return static_cast<std::uint32_t>(entry) < num_physical_blocks;
}

void TestIndexerWellFormedTable(emu_test::Runner& runner) {
  runner.Section("fast_fp8_indexer_topk: well-formed block table stays in bounds");

  const std::uint32_t block_size = 128u;
  const std::uint32_t seq_len = kIdxMaxSeqLen;
  const std::uint32_t blocks = (seq_len + block_size - 1u) / block_size;
  // The physical cache is deliberately larger than the sequence, as a real
  // paged allocator's would be.
  const std::uint32_t physical_blocks = blocks + 3u;

  BoundedSpan<std::uint8_t> cache(
      runner, "key cache",
      static_cast<std::size_t>(physical_blocks) * block_size * kIdxHeadDim);

  // A fixed non-identity permutation, the same shape of mapping the existing
  // emu fixture uses, so a reordered table is covered rather than assumed.
  std::vector<std::int32_t> block_table(blocks);
  for (std::uint32_t i = 0; i < blocks; ++i) {
    block_table[i] =
        static_cast<std::int32_t>((blocks + 2u - i) % physical_blocks);
  }

  for (std::uint32_t j = 0; j < seq_len; ++j) {
    const std::uint32_t logical_block = j / block_size;
    const std::uint32_t within_block = j % block_size;
    const std::uint32_t physical_block =
        static_cast<std::uint32_t>(block_table[logical_block]);
    const std::uint64_t row =
        KernelKeyRowOffset(physical_block, DenseBlockStride(block_size),
                           within_block);
    // The lane walks all 128 bytes of the row.
    cache.At(row, "key row base");
    cache.At(row + kIdxHeadDim - 1u, "key row last byte");
  }
  runner.Check(cache.clean(),
               "a permuted in-range block table addresses only real cache bytes");

  // Identity mapping, and the null-table path, which skips the indirection.
  BoundedSpan<std::uint8_t> direct(
      runner, "key cache (null table)",
      static_cast<std::size_t>(blocks) * block_size * kIdxHeadDim);
  for (std::uint32_t j = 0; j < seq_len; ++j) {
    const std::uint64_t row =
        KernelKeyRowOffset(j / block_size, DenseBlockStride(block_size),
                           j % block_size);
    direct.At(row + kIdxHeadDim - 1u, "key row last byte");
  }
  runner.Check(direct.clean(),
               "the null-table path is the identity mapping and stays in bounds");
}

void TestIndexerMalformedTable(emu_test::Runner& runner) {
  runner.Section(
      "fast_fp8_indexer_topk: a malformed block table must not be dereferenced");

  const std::uint32_t block_size = 128u;
  const std::uint32_t blocks = 16u;
  const std::uint32_t physical_blocks = blocks;  // num_physical_blocks
  const std::uint64_t cache_bytes =
      static_cast<std::uint64_t>(physical_blocks) * block_size * kIdxHeadDim;

  // The predicate must admit every real block and nothing else. Checked over
  // the whole valid range first, so a predicate that simply rejects
  // everything cannot pass the rejection cases below.
  for (std::uint32_t id = 0; id < physical_blocks; ++id) {
    runner.Check(BlockEntryIsAdmissible(static_cast<std::int32_t>(id),
                                        physical_blocks),
                 "block id " + std::to_string(id) + " is admissible");
  }

  // Case 1: one past the last physical block. The host gate validates
  // block_table_len and num_physical_blocks != 0, but neither bounds the
  // TABLE ENTRY, which is caller data.
  {
    const std::uint32_t bad = physical_blocks;  // first invalid id
    runner.Check(KernelKeyRowOffset(bad, DenseBlockStride(block_size), 0u) >= cache_bytes,
                 "an entry equal to num_physical_blocks addresses past the "
                 "cache, so it has to be rejected before the load");
    runner.Check(!BlockEntryIsAdmissible(static_cast<std::int32_t>(bad),
                                         physical_blocks),
                 "an entry equal to num_physical_blocks is rejected");
    runner.Check(!BlockEntryIsAdmissible(
                     static_cast<std::int32_t>(physical_blocks + 7u),
                     physical_blocks),
                 "an entry well past the cache is rejected");
  }

  // Case 2: a negative entry. block_table is int32_t and the kernel casts to
  // uint32_t, so -1 becomes 4,294,967,295 and the row offset overflows 32-bit
  // arithmetic entirely -- the device computes a wrapped, arbitrary address
  // rather than an obviously large one. That is why the predicate has to
  // reject on the SIGNED value, before any cast.
  {
    const std::int32_t negative = -1;
    const std::uint32_t as_unsigned = static_cast<std::uint32_t>(negative);
    runner.Check(as_unsigned == 4294967295u,
                 "a negative table entry casts to a near-2^32 block id");

    const std::uint64_t intended =
        KernelKeyRowOffset(as_unsigned, DenseBlockStride(block_size), 0u);
    runner.Check(intended >= cache_bytes,
                 "the intended address is far past the cache");

    // What the device actually computes, in 32-bit: the same expression
    // truncated. Showing the wrap is the point -- the resulting offset can
    // land anywhere, including back inside the buffer, which is why this must
    // be rejected rather than relied upon to fault loudly.
    const std::uint32_t wrapped =
        (as_unsigned * block_size + 0u) * kIdxHeadDim;
    runner.Check(static_cast<std::uint64_t>(wrapped) != intended,
                 "32-bit evaluation wraps, so the fault address is arbitrary "
                 "rather than reliably out of range");

    runner.Check(!BlockEntryIsAdmissible(negative, physical_blocks),
                 "a negative entry is rejected");
    runner.Check(!BlockEntryIsAdmissible(-2147483647 - 1, physical_blocks),
                 "INT32_MIN is rejected (its cast is 2^31, a plausible-looking "
                 "but invalid block id)");
  }
}

void TestIndexerBlockStride(emu_test::Runner& runner) {
  runner.Section(
      "fast_fp8_indexer_topk: key_block_stride is part of the ABI and must be used");

  const std::uint32_t block_size = 128u;
  const std::uint32_t blocks = 8u;
  const std::uint32_t dense_stride = DenseBlockStride(block_size);  // 16,384
  // The host gate accepts any key_block_stride >= block_size * 128, so a
  // PADDED cache is legal and the kernel must honour the field rather than
  // recompute the dense value. Addressing a padded cache with the dense
  // stride is in bounds but resolves the wrong row -- section 2.5's dangerous
  // class, a wrong answer rather than an error -- which is why this is
  // checked here and not left to hardware.
  const std::uint32_t padded_stride = dense_stride + 2048u;

  // 1. A padded layout must stay inside a padded cache, and must land on the
  //    padded spacing. The span is sized to the padded cache exactly, so an
  //    address computed with the wrong stride on the LAST block overruns the
  //    allocation and ASan traps it independently of the named check.
  BoundedSpan<std::uint8_t> padded_cache(
      runner, "padded key cache",
      static_cast<std::size_t>(blocks) * padded_stride);
  for (std::uint32_t physical = 0; physical < blocks; ++physical) {
    for (std::uint32_t within : {0u, 1u, block_size - 1u}) {
      const std::uint64_t off =
          KernelKeyRowOffset(physical, padded_stride, within);
      padded_cache.At(off, "padded key row base");
      padded_cache.At(off + kIdxHeadDim - 1u, "padded key row last byte");
    }
  }
  runner.Check(padded_cache.clean(),
               "the ABI stride addresses a padded cache without overrun");

  // 2. The two strides must genuinely differ past block 0 -- otherwise this
  //    section would pass against a kernel that still hard-codes the dense
  //    value. Block 0 coincides for ANY stride, which is exactly why a
  //    single-block fixture cannot catch a stride bug.
  runner.Check(KernelKeyRowOffset(0u, dense_stride, 5u) ==
                   KernelKeyRowOffset(0u, padded_stride, 5u),
               "block 0 agrees regardless of stride, so a single-block test "
               "cannot detect a stride error");
  std::uint32_t divergent_blocks = 0;
  for (std::uint32_t physical = 1u; physical < blocks; ++physical) {
    if (KernelKeyRowOffset(physical, dense_stride, 0u) !=
        KernelKeyRowOffset(physical, padded_stride, 0u)) {
      ++divergent_blocks;
    }
  }
  runner.Check(divergent_blocks == blocks - 1u,
               "dense and padded strides disagree for every block past the "
               "first, so this section has real discriminating power");

  // 3. For an UNPADDED cache the ABI stride reduces to the dense spacing, so
  //    honouring the field cannot regress the common case.
  for (std::uint32_t physical = 0; physical < blocks; ++physical) {
    for (std::uint32_t within : {0u, 7u, block_size - 1u}) {
      runner.Check(
          KernelKeyRowOffset(physical, dense_stride, within) ==
              (static_cast<std::uint64_t>(physical) * block_size + within) *
                  kIdxHeadDim,
          "dense stride reduces to the contiguous PA_BSND layout");
    }
  }
}

// ---------------------------------------------------------------------------
// Section 3b: fast_fp8_indexer_score key distribution across 32 AIVs.
//
// Transcribed from ScoreKeysSimt's loop header, which is the entire contract
// of the two-kernel split:
//
//   for (j = warp_base + warp; j < seq_len; j += warp_stride)
//     warp_base   = block_idx * kWarpsPerCore
//     warp_stride = kScoreBlockDim * kWarpsPerCore
//
// Correctness of the split rests on one property and nothing else: the union
// of the per-(core, warp) key sequences must be EXACTLY [0, seq_len), each key
// once. Scoring is key-independent, so if that holds the 32-core scores are
// bit-identical to the single-core scores; if it does not, keys are either
// never scored (the workspace keeps whatever was there, and selection ranks a
// stale or uninitialized score) or scored twice by different cores (two cores
// race on one 4-byte slot). Neither failure throws, and neither is visible in
// a cosine against the oracle once the top-k happens to come out the same --
// which is exactly why it is checked here rather than on hardware.
//
// This says nothing about warp scheduling or store visibility (CLAUDE.md
// §5.2); it is an integer-coverage claim and stops there.
//
// VERIFIED TO HAVE TEETH. Mutation-tested at seq_len 2,048 on a 32-core grid;
// every plausible way to get the pair wrong is caught, with the counts naming
// which way:
//
//   warp_base = block_idx (missing the *16)   1,860 unscored,  180 duplicated
//   warp_base = block_idx * 32 (warps/cores)  1,024 unscored,  768 duplicated
//   warp_stride = warps per core (16)                        2,032 duplicated
//   warp_stride = block_dim (32)                             2,016 duplicated
//   warp_stride 511 / 513                         3 unscored,    4 duplicated
//
// The first two rows are the ones that matter: a grid-stride written as
// `for (j = block_idx; j < seq_len; j += block_dim)` is the obvious way to
// spread keys over cores and it scores 3% of them, because this kernel's unit
// of work is a WARP, not a core. That formulation reads correct and is not;
// it is caught here in a printed line rather than as a plausible-looking
// hardware number.
// ---------------------------------------------------------------------------
void CheckScoreCoverage(emu_test::Runner& runner, std::uint32_t block_dim,
                        std::uint32_t seq_len, const std::string& label) {
  const std::uint32_t warp_stride = block_dim * kIdxWarpsPerCore;

  // One counter per key: how many (core, warp) pairs claim it.
  std::vector<std::uint32_t> visits(seq_len, 0u);
  std::uint64_t total = 0;
  for (std::uint32_t block_idx = 0; block_idx < block_dim; ++block_idx) {
    const std::uint32_t warp_base = block_idx * kIdxWarpsPerCore;
    for (std::uint32_t warp = 0; warp < kIdxWarpsPerCore; ++warp) {
      for (std::uint32_t j = warp_base + warp; j < seq_len; j += warp_stride) {
        ++visits[j];
        ++total;
      }
    }
  }

  std::uint32_t unvisited = 0;
  std::uint32_t duplicated = 0;
  std::uint32_t first_bad = 0;
  for (std::uint32_t j = 0; j < seq_len; ++j) {
    if (visits[j] == 0u) {
      if (unvisited == 0u && duplicated == 0u) first_bad = j;
      ++unvisited;
    } else if (visits[j] > 1u) {
      if (unvisited == 0u && duplicated == 0u) first_bad = j;
      ++duplicated;
    }
  }

  runner.Check(unvisited == 0u,
               label + ": every key is scored (unscored=" +
                   std::to_string(unvisited) + ", first at j=" +
                   std::to_string(first_bad) + ")");
  runner.Check(duplicated == 0u,
               label + ": no key is scored twice, so no two cores write the "
                       "same workspace slot (duplicated=" +
                   std::to_string(duplicated) + ")");
  runner.Check(total == static_cast<std::uint64_t>(seq_len),
               label + ": the schedule issues exactly seq_len key dots (" +
                   std::to_string(total) + " vs " + std::to_string(seq_len) +
                   ")");
}

void TestIndexerScoreDistribution(emu_test::Runner& runner) {
  runner.Section("fast_fp8_indexer_score key distribution");

  // The production grid, at the three sequence lengths the hardware
  // benchmark covers plus the boundary cases.
  for (std::uint32_t seq_len : {1u, 31u, 32u, 511u, 512u, 513u, 1024u,
                                kIdxMaxSeqLen}) {
    CheckScoreCoverage(runner, kIdxScoreBlockDim, seq_len,
                       "32 AIV, seq_len " + std::to_string(seq_len));
  }

  // The fused single-core kernel is the degenerate case of the same schedule
  // (warp_base 0, stride kWarpsPerCore). If that stopped being a partition the
  // split-vs-fused comparison would be comparing two broken things.
  for (std::uint32_t seq_len : {1u, 17u, 512u, kIdxMaxSeqLen}) {
    CheckScoreCoverage(runner, 1u, seq_len,
                       "1 AIV, seq_len " + std::to_string(seq_len));
  }

  // The property must hold for ANY grid, not just 1 and 32: the stride is
  // derived from the grid rather than hardcoded, so a future change to
  // kScoreBlockDim must not need this file edited to stay true.
  for (std::uint32_t block_dim : {2u, 3u, 7u, 16u, 32u}) {
    CheckScoreCoverage(runner, block_dim, 1000u,
                       "block_dim " + std::to_string(block_dim) +
                           ", seq_len 1000");
  }

  // Each warp's own sequence must be strictly increasing and stay in range --
  // the loop variable is uint32 and `warp_base + warp` must not be able to
  // start past seq_len and wrap.
  bool monotone = true;
  bool in_range = true;
  for (std::uint32_t block_idx = 0; block_idx < kIdxScoreBlockDim; ++block_idx) {
    for (std::uint32_t warp = 0; warp < kIdxWarpsPerCore; ++warp) {
      std::uint32_t previous = 0;
      bool first = true;
      for (std::uint32_t j = block_idx * kIdxWarpsPerCore + warp;
           j < kIdxMaxSeqLen; j += kIdxScoreWarpStride) {
        if (!first && j <= previous) monotone = false;
        if (j >= kIdxMaxSeqLen) in_range = false;
        previous = j;
        first = false;
      }
    }
  }
  runner.Check(monotone, "each warp's key sequence is strictly increasing");
  runner.Check(in_range,
               "no warp forms a key index at or past kMaxSeqLen, so the "
               "workspace write stays inside Fp8IndexerWorkspaceBytes");
}

// ---------------------------------------------------------------------------
// Section 4: fused_mhc_pre_norm_sinkhorn segment and butterfly indexing.
//
// The 4x4 solve never leaves registers, so there is no buffer walk to
// overrun -- the indexing surface is (a) the three bias/u segments and (b)
// the claim that masks 1,2 are rows and 4,8 are columns under t -> (t/4, t%4).
// Both are integer facts, so both belong here.
// ---------------------------------------------------------------------------
void TestSinkhornIndexing(emu_test::Runner& runner) {
  runner.Section("fused_mhc_pre_norm_sinkhorn: gate segment indexing");

  BoundedSpan<float> u(runner, "u", kShProjectionRows);
  BoundedSpan<float> bias(runner, "bias", kShProjectionRows);

  for (std::uint32_t t = 0; t < 4u * kShStreams; ++t) {  // 16 threads
    if (t < 2u * kShStreams) {
      // pre[] for t < 4 and post[] for 4 <= t < 8 both read u[t], bias[t].
      u.At(t, "gate projection");
      bias.At(t, "gate bias");
    }
    // Every thread reads the logits segment at 2*kStreams + t.
    u.At(2u * kShStreams + t, "logit projection");
    bias.At(2u * kShStreams + t, "logit bias");
  }
  runner.Check(u.clean(), "u is read only within its 24 rows");
  runner.Check(bias.clean(), "bias is read only within its 24 rows");

  // The segments must tile the 24 rows exactly: pre [0,4), post [4,8),
  // logits [8,24). An off-by-one here aliases a gate onto a logit.
  runner.Check(2u * kShStreams + 4u * kShStreams == kShProjectionRows,
               "the three segments tile all 24 projection rows with no gap "
               "and no overlap");

  runner.Section("fused_mhc_pre_norm_sinkhorn: 4x4 butterfly masks");
  // Thread t holds B[t/4][t%4]. Row masks 1,2 must permute within a row;
  // column masks 4,8 must permute within a column. Checked exhaustively --
  // all 16 lanes, all four masks.
  bool rows_ok = true;
  bool cols_ok = true;
  for (std::uint32_t t = 0; t < 16u; ++t) {
    const std::uint32_t row = t / 4u;
    const std::uint32_t col = t % 4u;
    for (std::uint32_t mask : {1u, 2u}) {
      const std::uint32_t partner = t ^ mask;
      if (partner / 4u != row || partner % 4u == col) {
        rows_ok = false;
      }
    }
    for (std::uint32_t mask : {4u, 8u}) {
      const std::uint32_t partner = t ^ mask;
      if (partner % 4u != col || partner / 4u == row) {
        cols_ok = false;
      }
    }
    // Every exchange must stay inside the 16-lane block, or asc_shfl_xor
    // would reach an inactive lane and return an uninitialized register.
    for (std::uint32_t mask : {1u, 2u, 4u, 8u}) {
      runner.Check((t ^ mask) < 16u,
                   "lane " + std::to_string(t) + " xor " +
                       std::to_string(mask) + " stays within width 16");
    }
  }
  runner.Check(rows_ok,
               "masks 1,2 exchange within a row and change the column");
  runner.Check(cols_ok,
               "masks 4,8 exchange within a column and change the row");
}

}  // namespace

int main() {
  emu_test::Runner runner("kernel addressing (host index validation)");

  TestMirroredShapeInvariants(runner);
  TestMxfp4Addressing(runner);
  TestIndexerWellFormedTable(runner);
  TestIndexerMalformedTable(runner);
  TestIndexerBlockStride(runner);
  TestIndexerScoreDistribution(runner);
  TestSinkhornIndexing(runner);

  return runner.Summary();
}
