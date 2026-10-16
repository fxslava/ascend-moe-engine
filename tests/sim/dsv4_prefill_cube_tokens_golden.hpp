/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

// =================================================================================================
// dsv4_prefill_cube_tokens_golden.hpp -- audited CPU oracle for the token prefetch ring
// Target Architecture : Huawei Ascend 950PR (arch35 / dav-3510 / CANN 9.2)
// =================================================================================================
//
// 1. TWO ORACLES, BECAUSE THERE ARE TWO THINGS TO BE WRONG
// -------------------------------------------------------------------------------------------------
// `AsyncLoadTokensChunk` + `PrepareTokensChunk` are a two-slot software pipeline, so they can fail
// in two unrelated ways and each needs its own oracle:
//
//     VALUE   -- `CpuPrepareTokensChunk` says what one chunk's BF16 should be. It catches a wrong
//                `DataCopyParams` stride, a wrong `hidden_base`, a transposed gather, a cast in the
//                wrong rounding mode.
//     SCHEDULE -- `CpuTokenChunkPlan` says which slot and which event id each chunk uses, and how
//                many times each event is set and waited. It catches a ring that does not rotate,
//                a prologue that primes the wrong slot, an epilogue that drops or leaks a token.
//
// The schedule oracle runs in microseconds on the host and needs no simulator, which is the whole
// point: a marked-graph token imbalance presents on CAModel as a HANG, and diagnosing a hang costs
// a 290 s run plus a watchdog trip. Counting the tokens in a host loop first is free.
//
// 2. THE MARKED-GRAPH OBLIGATIONS, STATED SO THEY CAN BE CHECKED
// -------------------------------------------------------------------------------------------------
// Model the loop as a graph whose edges carry tokens (set flags). With `kTokenSlots` = 2:
//
//     LIVENESS    -- no `WaitFlag` may run with zero outstanding tokens on its event. Violated, the
//                    simulator hangs rather than failing.
//     BALANCE     -- per event id, sets == waits over the whole launch. A leaked token survives the
//                    launch and corrupts the NEXT one, which is why this class shows up as an
//                    unrelated kernel failing second.
//     ROTATION    -- consecutive chunks must occupy DIFFERENT slots, or the prefetch of chunk k+1
//                    lands on the buffer chunk k is still being read out of.
//     PRIMING     -- exactly one chunk is loaded before the loop, and the loop must not consume a
//                    token the prologue did not produce.
//
// `AuditTokenChunkPlan` discharges all four for a given (hidden, k_sub), over every chunk count the
// gate runs and a sweep besides.
//
// 3. WHAT THIS ORACLE DELIBERATELY DOES *NOT* CLAIM
// -------------------------------------------------------------------------------------------------
// It does not claim the loads and the casts OVERLAP IN TIME. Overlap is a scheduling outcome, and
// the only honest evidence for it on this workstation would be hardware. What a correct plan plus a
// correct per-chunk value does establish is the CORRECTNESS CONTENT of double buffering: the two
// slots are independent, the ring rotates, and no chunk is overwritten before it is consumed. That
// is a precondition for overlap and it is what a wrong ring breaks -- so it is worth gating, under
// its own name rather than under "overlap demonstrated".
//
// 4. THE BUFFER-FREE EDGE, AND THE BUG THIS ORACLE WAS WRITTEN TO CATCH
// -------------------------------------------------------------------------------------------------
// A two-slot ring needs edges in BOTH directions: `MTE2_V` says "the slot is full, the Vector may
// read it", and a matching `V_MTE2` says "the Vector is done, MTE2 may refill it". Until
// 2026-10-16 the production loop issued only the first, and the consequence was OBSERVED on the
// dav-3510 CAModel at hidden = 512: chunk 0 came back holding chunk 2's values, because the MTE2
// burst for chunk 2 reached the ping slot before the Cast of chunk 0 had read it. Silent, no fault,
// plausible numbers from the wrong chunk -- and invisible below four chunks, since a shorter run
// reuses no slot.
//
// So this oracle models BOTH edges and holds the fixed design to its accounting: every refill of a
// slot that has already been converted out of must be guarded by a `V_MTE2` wait, the first fill of
// each slot must NOT be (the edge starts empty and waiting there would hang), and the tokens the
// final chunks leave outstanding must be drained -- one per occupied slot, independent of parity --
// so none survives into the next `inter_base` pass or the next launch.

#ifndef TESTS_SIM_DSV4_PREFILL_CUBE_TOKENS_GOLDEN_HPP
#define TESTS_SIM_DSV4_PREFILL_CUBE_TOKENS_GOLDEN_HPP

#include "moe/emu/cpu_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::sim::prefill_cube {

// The ring the production loop implements: ping and pong, with event ids 2 and 3. Restated here
// rather than included, for the same reason the dequant golden restates its layout constants -- the
// operator's definitions live in op_kernel/ and the micro-kernel TU static_asserts the two copies
// against each other, so a drift is a compile error and not a test that agrees with itself.
inline constexpr std::uint32_t kTokenSlots       = 2;
inline constexpr std::uint32_t kTokenEventPing   = 2;
inline constexpr std::uint32_t kTokenEventPong   = 3;
inline constexpr std::uint32_t kTokenMaxChunks   = 32;

// The minimal inductive grid for a kSlots-slot ring: one chunk primes, one flips, one wraps back to
// slot 0 with a real reuse, one drains. Below this the ring never wraps and a rotation bug cannot
// be observed, so a gate that runs fewer chunks than this has not tested the ring.
inline constexpr std::uint32_t kTokenMinChunks = kTokenSlots + 2;

[[noreturn]] inline void TokensFail(const std::string& message) {
    throw std::runtime_error("tokens golden: " + message);
}

inline void TokensRequire(bool condition, const std::string& message) {
    if (!condition) TokensFail(message);
}

// ---------------------------------------------------------------------------------------------
// VALUE oracle
// ---------------------------------------------------------------------------------------------
// One chunk of the token plane, gathered and cast exactly as the device does it: rows are
// contiguous within the chunk, the source row stride is the FULL hidden extent, and the cast is
// round-to-nearest-even. Returns BF16 bit patterns so the device compare needs no float round trip.
inline std::vector<std::uint16_t> CpuPrepareTokensChunk(const float* tokens, std::uint32_t rows,
                                                        std::uint32_t hidden,
                                                        std::uint32_t hidden_base,
                                                        std::uint32_t chunk_width) {
    TokensRequire(tokens != nullptr, "null token plane");
    TokensRequire(rows > 0 && chunk_width > 0, "empty token chunk");
    TokensRequire(hidden_base + chunk_width <= hidden, "token chunk runs past the hidden extent");

    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * chunk_width, 0U);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < chunk_width; ++col) {
            out[static_cast<std::size_t>(row) * chunk_width + col] = moe::emu::FloatToBf16(
                tokens[static_cast<std::size_t>(row) * hidden + hidden_base + col]);
        }
    }
    return out;
}

// A token plane generated in memory from a printed seed. The magnitudes are deliberately NOT
// BF16-representable: a plane of exact values would let a cast that silently truncated instead of
// rounding to nearest pass, which is precisely the distinction `CAST_RINT` names.
//
// Ties are excluded here on purpose and probed separately. A tie is the one input where a converter
// may legitimately differ from this oracle by one ULP depending on its tie-breaking convention, so
// including them would conflate "the ring is wrong" with "the rounding convention differs" -- and
// this gate exists to tell those apart.
inline std::vector<float> BuildTokenPlane(std::uint32_t rows, std::uint32_t hidden,
                                          std::uint64_t seed) {
    std::vector<float>              plane(static_cast<std::size_t>(rows) * hidden, 0.0F);
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> source(0.0F, 2.5F);

    for (std::size_t i = 0; i < plane.size(); ++i) {
        float value = source(rng);
        // Reject the exactly-representable and the exactly-tied. A BF16 tie is a value whose
        // discarded low 16 bits are exactly 0x8000; an exact value has them all zero.
        for (int attempt = 0; attempt < 64; ++attempt) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint32_t residue = bits & 0xFFFFU;
            if (residue != 0U && residue != 0x8000U) break;
            value = source(rng);
        }
        plane[i] = value;
    }
    return plane;
}

// ---------------------------------------------------------------------------------------------
// SCHEDULE oracle
// ---------------------------------------------------------------------------------------------
struct TokenChunkStep {
    std::uint32_t chunk          = 0;      // which hidden chunk this step converts
    std::uint32_t hidden_base    = 0;
    std::uint32_t slot           = 0;      // 0 = ping, 1 = pong
    std::uint32_t wait_event     = 0;      // the event PrepareTokensChunk waits on
    bool          is_epilogue    = false;  // converted outside the loop, with no successor load
    bool          successor_load_guarded = false;  // the load this step issues, for chunk+1,
                                                   // waited for its slot to be free
};

struct TokenChunkPlan {
    std::uint32_t               chunks = 0;
    std::vector<TokenChunkStep> steps;
    // MTE2_V, the "slot is full" edge, per slot (which is also per event id).
    std::uint32_t full_sets[kTokenSlots]  = {0, 0};
    std::uint32_t full_waits[kTokenSlots] = {0, 0};
    // V_MTE2, the "slot is free" edge.
    std::uint32_t free_sets[kTokenSlots]  = {0, 0};
    std::uint32_t free_waits[kTokenSlots] = {0, 0};
    std::uint32_t drained                 = 0;
};

// Transcribes the production prologue / loop / epilogue / drain, which is what makes it an oracle
// for the schedule rather than a restatement of it: the loop bound, the flip, the refill guard and
// the drain are copied from the kernel, and the token accounting is derived from them here, where
// an imbalance costs microseconds to find instead of a 350 s simulator hang.
inline TokenChunkPlan CpuTokenChunkPlan(std::uint32_t hidden, std::uint32_t chunk_width) {
    TokensRequire(chunk_width > 0, "zero chunk width");
    TokensRequire(hidden % chunk_width == 0,
                  "this gate only runs chunk-aligned hidden extents; a ragged tail is a separate "
                  "obligation and the production loop does not implement one");

    TokenChunkPlan plan;
    plan.chunks = hidden / chunk_width;
    TokensRequire(plan.chunks >= 1, "no chunks");

    // Per slot: whether it holds data nobody has converted yet, and how many "this slot is free"
    // tokens are outstanding on it.
    bool          holds_unconverted[kTokenSlots] = {false, false};
    std::uint32_t free_outstanding[kTokenSlots]  = {0, 0};
    bool          ever_converted[kTokenSlots]    = {false, false};
    std::uint32_t slot                           = 0;

    // Prologue: chunk 0 into ping, unguarded because the free edge starts empty.
    ++plan.full_sets[slot];
    holds_unconverted[slot] = true;

    std::uint32_t hidden_base = 0;
    for (; hidden_base + chunk_width < hidden; hidden_base += chunk_width) {
        const std::uint32_t next_slot  = 1U - slot;
        const std::uint32_t next_chunk = (hidden_base / chunk_width) + 1U;

        TokenChunkStep step;
        step.chunk       = hidden_base / chunk_width;
        step.hidden_base = hidden_base;
        step.slot        = slot;
        step.wait_event  = slot == 0 ? kTokenEventPing : kTokenEventPong;

        // The loop body issues the NEXT load BEFORE converting the CURRENT chunk -- the overlap the
        // double buffer exists for, and the reason the refill has to be guarded.
        const bool guarded = next_chunk >= kTokenSlots;
        if (guarded) {
            TokensRequire(free_outstanding[next_slot] > 0,
                          "LIVENESS: the load for chunk " + std::to_string(next_chunk) +
                              " waits for slot " + std::to_string(next_slot) +
                              " to be free with no outstanding V_MTE2 token");
            --free_outstanding[next_slot];
            ++plan.free_waits[next_slot];
        } else {
            TokensRequire(!ever_converted[next_slot],
                          "SAFETY: the load for chunk " + std::to_string(next_chunk) +
                              " refills slot " + std::to_string(next_slot) +
                              " unguarded, but that slot has already been converted out of once -- "
                              "this is the race that returned chunk 2's data as chunk 0");
        }
        TokensRequire(!holds_unconverted[next_slot],
                      "BOUNDEDNESS: the load for chunk " + std::to_string(next_chunk) +
                          " overwrites slot " + std::to_string(next_slot) +
                          ", which still holds a chunk nobody has converted");
        ++plan.full_sets[next_slot];
        holds_unconverted[next_slot] = true;

        TokensRequire(holds_unconverted[slot],
                      "LIVENESS: the loop waits on MTE2_V event " + std::to_string(step.wait_event) +
                          " for chunk " + std::to_string(step.chunk) +
                          " with no outstanding token on it");
        ++plan.full_waits[slot];
        holds_unconverted[slot] = false;
        ever_converted[slot]    = true;
        ++free_outstanding[slot];
        ++plan.free_sets[slot];

        step.successor_load_guarded = guarded;
        plan.steps.push_back(step);
        slot = next_slot;
    }

    TokenChunkStep tail;
    tail.chunk       = hidden_base / chunk_width;
    tail.hidden_base = hidden_base;
    tail.slot        = slot;
    tail.wait_event  = slot == 0 ? kTokenEventPing : kTokenEventPong;
    tail.is_epilogue = true;
    TokensRequire(holds_unconverted[slot],
                  "LIVENESS: the epilogue waits on MTE2_V event " + std::to_string(tail.wait_event) +
                      " with no outstanding token on it");
    ++plan.full_waits[slot];
    holds_unconverted[slot] = false;
    ever_converted[slot]    = true;
    ++free_outstanding[slot];
    ++plan.free_sets[slot];
    plan.steps.push_back(tail);

    // The drain, exactly as DrainTokenRing issues it: ping always, pong once the ring has two
    // chunks. Parity-independent -- every occupied slot ends one free token up, because a slot is
    // refilled for every chunk it holds except the first.
    for (std::uint32_t s = 0; s < kTokenSlots; ++s) {
        if (plan.chunks < s + 1U) continue;
        TokensRequire(free_outstanding[s] > 0,
                      "LIVENESS: the drain waits on V_MTE2 for slot " + std::to_string(s) +
                          " with no outstanding token; DrainTokenRing would hang");
        --free_outstanding[s];
        ++plan.free_waits[s];
        ++plan.drained;
    }

    for (std::uint32_t s = 0; s < kTokenSlots; ++s) {
        TokensRequire(!holds_unconverted[s],
                      "BALANCE: slot " + std::to_string(s) +
                          " still holds an unconsumed MTE2_V token at the end of the pass");
        TokensRequire(free_outstanding[s] == 0,
                      "BALANCE: slot " + std::to_string(s) + " leaks " +
                          std::to_string(free_outstanding[s]) +
                          " V_MTE2 token(s) past the drain; a leaked flag outlives the launch and "
                          "corrupts the next one");
    }
    return plan;
}

inline void AuditTokenChunkPlan(std::uint32_t hidden, std::uint32_t chunk_width) {
    const TokenChunkPlan plan = CpuTokenChunkPlan(hidden, chunk_width);

    // BALANCE, on BOTH edges and per event id. An imbalance on the full edge hangs the Vector; an
    // imbalance on the free edge either hangs MTE2 or leaks a flag into the next launch.
    for (std::uint32_t s = 0; s < kTokenSlots; ++s) {
        const std::uint32_t event = s == 0 ? kTokenEventPing : kTokenEventPong;
        if (plan.full_sets[s] != plan.full_waits[s]) {
            TokensFail("BALANCE: MTE2_V event " + std::to_string(event) + " is set " +
                       std::to_string(plan.full_sets[s]) + " times and waited " +
                       std::to_string(plan.full_waits[s]) + " times over " +
                       std::to_string(plan.chunks) + " chunks");
        }
        if (plan.free_sets[s] != plan.free_waits[s]) {
            TokensFail("BALANCE: V_MTE2 event " + std::to_string(event) + " is set " +
                       std::to_string(plan.free_sets[s]) + " times and waited " +
                       std::to_string(plan.free_waits[s]) + " times over " +
                       std::to_string(plan.chunks) + " chunks");
        }
    }

    // SAFETY: every REUSE of a slot is guarded, and only the first fill of each is not. Step i
    // issues the load for chunk i+1, so that load must be guarded exactly when chunk i+1 is at or
    // past the slot count. This is the invariant whose absence returned chunk 2's data as chunk 0.
    for (std::uint32_t i = 0; i < plan.chunks; ++i) {
        if (plan.steps[i].is_epilogue) continue;  // the epilogue step issues no load
        const std::uint32_t loaded_chunk   = i + 1U;
        const bool          expect_guarded = loaded_chunk >= kTokenSlots;
        if (plan.steps[i].successor_load_guarded != expect_guarded) {
            TokensFail("SAFETY: the load for chunk " + std::to_string(loaded_chunk) + " is " +
                       (plan.steps[i].successor_load_guarded ? "guarded" : "unguarded") +
                       ", want " + (expect_guarded ? "guarded" : "unguarded"));
        }
    }

    // COVERAGE: every chunk converted exactly once, in order.
    TokensRequire(plan.steps.size() == plan.chunks, "the plan does not convert every chunk once");
    for (std::uint32_t i = 0; i < plan.chunks; ++i) {
        TokensRequire(plan.steps[i].chunk == i, "chunk " + std::to_string(i) + " out of order");
        TokensRequire(plan.steps[i].hidden_base == i * chunk_width,
                      "chunk " + std::to_string(i) + " reads the wrong hidden_base");
    }

    // ROTATION: consecutive chunks must not share a slot, or a prefetch lands on a live buffer.
    for (std::uint32_t i = 1; i < plan.chunks; ++i) {
        TokensRequire(plan.steps[i].slot != plan.steps[i - 1].slot,
                      "ROTATION: chunks " + std::to_string(i - 1) + " and " + std::to_string(i) +
                          " share slot " + std::to_string(plan.steps[i].slot) +
                          "; the ring is not rotating");
    }

    // PRIMING: exactly one chunk is in flight before the first conversion, so the first `sets` is
    // the prologue's and nothing waits ahead of it.
    TokensRequire(plan.steps[0].slot == 0, "PRIMING: the first chunk must be converted out of ping");
    TokensRequire(plan.steps[plan.chunks - 1U].is_epilogue,
                  "the last chunk must be the epilogue conversion");

    std::printf("[PASS] token ring plan hidden=%u k_sub=%u: %u chunks; MTE2_V set %u/%u waited "
                "%u/%u; V_MTE2 set %u/%u waited %u/%u, %u drained; rotation, balance and refill "
                "guarding all hold\n",
                hidden, chunk_width, plan.chunks, plan.full_sets[0], plan.full_sets[1],
                plan.full_waits[0], plan.full_waits[1], plan.free_sets[0], plan.free_sets[1],
                plan.free_waits[0], plan.free_waits[1], plan.drained);
}

}  // namespace moe::sim::prefill_cube

#endif  // TESTS_SIM_DSV4_PREFILL_CUBE_TOKENS_GOLDEN_HPP
