/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// The hybrid compressed-KV cache entry, and the ONE way this engine is allowed
// to learn a stride.
//
// WHY A STRUCT AT ALL
// -------------------
// `aclnnCompressor` emits one pooled row per window; `aclnnKvCompressEpilog`
// quantizes that row and scatters it into the paged cache; and
// `aclnnKvQuantSparseAttnSharedkv` reads the cache back. All three address the
// cache through an axis-0 stride passed as a plain int64 attribute, with no
// cross-check anywhere in the vendored wrappers (dsv4_operator_hypotheses_test
// H3 records the attribute as UNGUARDED on the CANN backend: a stale value
// pools or scatters into the wrong block with no diagnostic). So the row width
// is load-bearing twice over -- once to size the reservation, once as the
// number the kernels address with -- and it is written down exactly once, here.
//
// THE 604-BYTE ENTRY, AND WHAT IS AND IS NOT SETTLED ABOUT IT
// -----------------------------------------------------------
// The layout below is the DSV4-Flash compressed entry as the runtime brief
// states it: of the `kv_lora_rank` = 512 compressed channels, 448 are the
// no-position (nope) half in FP8 E4M3 with one UE8M0 block scale per 64
// elements, and 64 are the RoPE half kept at full BF16 precision.
//
//   448 nope bytes + 7 scale bytes + 21 pad = 476, then 64 * 2 = 128 rope
//   bytes -> 604 bytes, and 448 + 64 = 512 = kv_lora_rank.
//
// That is internally consistent and `static_assert` holds it to the byte. What
// is NOT settled is whether the DEPLOYED arch35 kernel writes this packing:
// H3 measured the compressor's own `cmpKvOut` as D x sizeof(dtype) (1024 bytes
// at BF16 x 512) and recorded the 604-byte claim as UNRESOLVED, because no
// (width, dtype) pair in this geometry produces it. The entry is therefore the
// HOST-SIDE authority for the cache row -- it sizes the reservation and the
// descriptor -- while every stride an operator is TOLD comes from
// `DeriveDimensionStrideElements` reading the descriptor that was actually
// bound. If the deployed kernel disagrees, the mismatch surfaces as a refused
// plan or a failed stride assertion, never as silent corruption.
//
// HOW TO GET A STRIDE
// -------------------
// Never from arithmetic at a call site. `DeriveDimensionStrideElements` and
// `DeriveDimensionStrideBytes` interrogate the live `aclTensor` through
// `aclGetViewStrides` / `aclGetDataType`, so the value handed to the operator
// is by construction the stride of the view the operator received. Element
// strides are what the operators take (`stateCacheStrideDim0`, `blockStride`,
// `oriKvStride0`, `cmpKvStride0`, the indexer's `stride` / `scaleStride`); the
// byte form exists for reservation and host-pointer arithmetic.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

typedef struct aclTensor aclTensor;

namespace ascend_moe {

// ---------------------------------------------------------------------------
// The packed hybrid entry
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct Dsv4CompressedKvEntry {
  uint8_t nope_fp8[448];     // 448 channels, FP8 E4M3
  uint8_t scales_ue8m0[7];   // one OCP UE8M0 block scale per 64 nope channels
  uint8_t padding[21];       // pads the quantized half to 476 bytes
  uint16_t rope_bf16[64];    // 64 RoPE channels kept at BF16 (128 bytes)
};
#pragma pack(pop)

static_assert(sizeof(Dsv4CompressedKvEntry) == 604, "Invalid KV cache stride size");
// The field arithmetic, not just the total: a compiler that honoured the pack
// but reordered or re-padded a member would still hit 604.
static_assert(offsetof(Dsv4CompressedKvEntry, nope_fp8) == 0, "nope half must lead the entry");
static_assert(offsetof(Dsv4CompressedKvEntry, scales_ue8m0) == 448, "block scales follow the nope half");
static_assert(offsetof(Dsv4CompressedKvEntry, padding) == 455, "padding follows the block scales");
static_assert(offsetof(Dsv4CompressedKvEntry, rope_bf16) == 476, "the rope half starts at 476");
static_assert(sizeof(Dsv4CompressedKvEntry::nope_fp8) / 64 == sizeof(Dsv4CompressedKvEntry::scales_ue8m0),
              "one UE8M0 block scale per 64 nope channels");

// The row width every compressed-KV reservation and descriptor is sized from.
inline constexpr int64_t kCompressedKvEntryBytes = static_cast<int64_t>(sizeof(Dsv4CompressedKvEntry));

// The two halves, as channel counts, so a geometry check can compare them
// against the checkpoint's kv_lora_rank + qk_rope_head_dim instead of
// re-spelling 448 and 64 at a call site.
inline constexpr int64_t kCompressedKvNopeChannels =
    static_cast<int64_t>(sizeof(Dsv4CompressedKvEntry::nope_fp8));
inline constexpr int64_t kCompressedKvRopeChannels =
    static_cast<int64_t>(sizeof(Dsv4CompressedKvEntry::rope_bf16) / sizeof(uint16_t));
inline constexpr int64_t kCompressedKvScaleCount =
    static_cast<int64_t>(sizeof(Dsv4CompressedKvEntry::scales_ue8m0));
inline constexpr int64_t kCompressedKvScaleBlock = kCompressedKvNopeChannels / kCompressedKvScaleCount;
static_assert(kCompressedKvScaleBlock == 64, "the nope half is quantized in blocks of 64 channels");

// ---------------------------------------------------------------------------
// Stride derivation -- the only sanctioned source of a stride attribute
// ---------------------------------------------------------------------------

// Bytes per element of an `aclDataType` code (the `kAcl*` constants in
// config.hpp). FP4 E2M1 reports 1: two elements share a byte and the caller
// halves the count, matching StaticMemoryArena::CreateFp4Tensor.
size_t AclDataTypeBytes(int32_t acl_dtype);

// The view geometry of a live descriptor, read back from the handle rather
// than recomputed. `strides` are in ELEMENTS, as aclCreateTensor received them.
struct TensorSliceLayout {
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
  int32_t dtype = 0;
  size_t element_bytes = 0;

  size_t rank() const { return dims.size(); }
  std::string Describe() const;
};

// Interrogates `tensor` through aclGetViewShape / aclGetViewStrides /
// aclGetDataType. Throws (Dsv4Error) on a null handle or a refusing runtime.
TensorSliceLayout ReadTensorSliceLayout(const aclTensor* tensor);

// The element stride of `axis`, which is what every stride ATTRIBUTE in the
// vendored arch35 surface takes. Throws when `axis` is outside the rank, so a
// mis-derived axis is a message rather than a silently wrong number.
int64_t DeriveDimensionStrideElements(const aclTensor* tensor, size_t axis);

// The same stride in bytes. Used for reservation arithmetic and for the
// host-side pointer walks that carve a per-layer slice out of one reservation.
size_t DeriveDimensionStrideBytes(const aclTensor* tensor, size_t axis);

// Throws unless `tensor`'s axis-0 element stride equals `expected`. The guard
// exists because the wrappers do not have one: it turns "the attribute drifted
// from the view" into a build-time-shaped failure at plan time, named.
void AssertDimensionStride(const aclTensor* tensor, size_t axis, int64_t expected, const char* what);

}  // namespace ascend_moe
