#ifndef DSV4_PREFILL_CUBE_UB_CENSUS_H
#define DSV4_PREFILL_CUBE_UB_CENSUS_H

// SIM ONLY -- the wire format of the UB physical-address census the staged
// bisection probe writes, shared by the device writer
// (kernel/test_subgemm_stage_kernel.cpp) and the host decoder
// (test_subgemm_stage_sim.cpp) so the two cannot drift.
//
// ACL-free and device-header-free on purpose: the host TU includes it with a
// stock g++ and the device TU includes it after kernel_operator.h.
//
// ---------------------------------------------------------------------------
// WHY THE CENSUS EXISTS
// ---------------------------------------------------------------------------
// `rvec_dec.cc judge_valid_addr ub_addr_overflow` names an ADDRESS, not a
// buffer. Mapping one to the other by adding up kCube*Bytes takes two things
// on trust -- that the allocations start at 0, and that the kCube*Bytes are
// all of them -- and guessing wrong about either sends the search after the
// wrong buffer. Both had in fact been guessed, in opposite directions:
//
//   * the pool was believed to start at ~131,616, past a SIMT DataCache
//     reservation, on the strength of TPipe::ResetPool() seeding it with
//     GetDynamicMemStartPos<Hardware::UB>() = `&dynamicStartUB[0]`. It
//     MEASURES AS 0: that linker symbol resolves to zero in this build.
//   * the four TPosition::TSCM weight ring slots were believed to be in L1,
//     which is what GetPhyType(TSCM) == Hardware::L1 and TscmInitBuffer's
//     top-of-L1 allocation say. They MEASURE AS UB, contiguous with the
//     sixteen UB buffers, and they cost 131,072 B.
//
// LocalTensor::GetPhyAddr() returns the real address, so the census replaces
// both guesses with a reading.
//
// ---------------------------------------------------------------------------
// WIRE FORMAT -- uint32 words, in the buffer the probe takes as `status_out`
// ---------------------------------------------------------------------------
//   0                 liveness word: the stage that completed (pre-existing
//                     contract; the census does not touch it)
//   1                 kUbCensusMagic, written once the census is valid
//   2                 entry count
//   3                 reserved, zero
//   4 + 2i, 5 + 2i    entry i: GetPhyAddr() low 32 bits, then high 32 bits
//
// Entries are in InitBuffer order, which is also TPipe's allocation order, so
// consecutive addresses bound each buffer without the sizes being transmitted.

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kUbCensusMagic       = 0x55424330u;  // "UBC0"
inline constexpr std::uint32_t kUbCensusHeaderWords = 4u;

// Sixteen UB buffers, then all four TPosition::TSCM weight-ring slots.
//
// THE RING SLOTS ARE IN THE CENSUS BECAUSE THEIR POOL IS IN DOUBT.
// GetPhyType maps TSCM to Hardware::L1, and TPipe::TscmInitBuffer allocates
// it from the TOP of L1 downward -- so by the headers these four should come
// back near 524,288 and have nothing to do with UB. The first measurement did
// not agree: slot 0 of the phase 1 ring reported 127,520, the exact byte the
// sixteen UB allocations end on. Four 32 KiB slots is 131,072 B, which is
// also exactly the gap between the UB total that overflowed (176,672) and the
// one that does not (127,520), so "the ring is in UB" and "something else
// 128 KiB wide caps the pool" both fit that one reading. All four addresses
// separate them: contiguous from the UB end means the ring IS the pool's
// other tenant.
inline constexpr std::uint32_t kUbCensusUbEntries    = 16u;
inline constexpr std::uint32_t kUbCensusTotalEntries = 20u;
inline constexpr std::uint32_t kUbCensusWords =
    kUbCensusHeaderWords + 2u * kUbCensusTotalEntries;

// The probe's status buffer must hold the census and stay 32 B aligned in
// length, so the host allocates this many words and the device declares the
// same.
inline constexpr std::uint32_t kUbCensusStatusWords = 48u;
static_assert(kUbCensusStatusWords >= kUbCensusWords,
              "the status buffer must hold the whole census");

// The four ring slots, in InitBuffer order: q_weight_fwd[0..1] then
// q_weight_down[0..1].
inline constexpr std::uint32_t kUbCensusTscmEntry  = 16u;
inline constexpr std::uint32_t kUbCensusTscmSlots  = 4u;
static_assert(kUbCensusTscmEntry + kUbCensusTscmSlots == kUbCensusTotalEntries,
              "the census is the UB buffers followed by the ring slots");

// Physical UB on dav-3510. The census is read against this, not against
// kUsableUbBytes: a buffer past this address is what the simulator reports.
inline constexpr std::uint32_t kUbCensusPhysicalUbBytes = 256u * 1024u;

#ifndef __CCE__
// Host-side only: the names, in the same order the device writes them.
inline constexpr const char* const kUbCensusNames[kUbCensusTotalEntries] = {
    "activation_resident",  "gate_accumulator",     "up_accumulator",
    "subgemm_fwd",          "tokens_f32_ping",      "tokens_f32_pong",
    "tokens_sub_bf16_ping", "tokens_sub_bf16_pong", "down_accumulator",
    "subgemm_down",         "activation_sub_bf16",  "weight_nibbles",
    "weight_expanded",      "weight_scales",        "scale_plane",
    "scale_temporary",      "weight_fwd[0] (TSCM)", "weight_fwd[1] (TSCM)",
    "weight_down[0] (TSCM)", "weight_down[1] (TSCM)",
};
#endif

}  // namespace moe::kernels::arch35

#endif  // DSV4_PREFILL_CUBE_UB_CENSUS_H
