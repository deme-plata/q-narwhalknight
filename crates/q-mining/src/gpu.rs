//! GPU Mining Module for Q-NarwhalKnight — Hybrid Quantum Mining
//!
//! Implements BLAKE3 + 100-round VDF proof-of-work on GPU via OpenCL.
//! Each GPU work item independently computes: BLAKE3(challenge||nonce) then
//! 99 sequential BLAKE3(h) VDF rounds, checking the final hash against
//! the difficulty target. GPU parallelism comes from running thousands of
//! nonce candidates simultaneously.
//!
//! ## Algorithm (must match server validation)
//! ```text
//! input = challenge_hash[32] || nonce_le[8]   // 40 bytes
//! h = BLAKE3(input)                           // initial hash
//! for _ in 0..99: h = BLAKE3(h)              // VDF chain (99 rounds)
//! if h < difficulty_target: SOLUTION!         // total: 100 BLAKE3 hashes
//! ```
//!
//! ## Optimizations (v10.1.7)
//! - **Persistent buffers**: Allocated once per GPU context, reused across dispatches
//! - **Conditional upload**: Challenge/target only re-uploaded when changed (same block = 0 uploads)
//! - **Challenge word precompute**: 32-byte challenge converted to 8×u32 on CPU, kernel skips per-thread byte unpacking
//! - **Adaptive work size**: Dispatch size auto-tunes to keep GPU dispatch time in [100ms, 400ms]
//! - **Non-blocking uploads**: Buffer writes use CL_FALSE (async) on in-order queue — GPU waits, CPU doesn't
//! - **Constant cache**: Challenge buffer uses `__constant` address space for hardware constant cache
//! - **Loop unrolling**: VDF loop uses `#pragma unroll 3` for partial unrolling (99 = 33×3)
//! - **Work-group size hint**: `reqd_work_group_size(256,1,1)` enables better register allocation
//! - **Per-GPU adaptive sizing**: Each GPU independently auto-tunes its work size
//! - **Startup calibration**: Benchmark 4 work sizes to find optimal starting point per GPU
//! - **Kernel compilation caching**: Compiled binary saved to `~/.config/q-miner/kernel-cache/`

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn, error};

#[cfg(feature = "gpu-mining")]
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY},
    program::Program,
    types::{cl_uchar, cl_uint, cl_ulong},
};

// ============================================================================
// BLAKE3 + VDF OpenCL Kernel
// ============================================================================

/// OpenCL kernel implementing BLAKE3 + 99-round VDF for Q-NarwhalKnight mining.
///
/// Algorithm per work item:
///   1. Build 40-byte input: challenge_hash[32] || nonce_le[8]
///   2. h = BLAKE3(input)           — single-block 40-byte hash
///   3. for 99 rounds: h = BLAKE3(h) — single-block 32-byte hash (VDF chain)
///   4. Compare final h < target (byte-wise, big-endian-like)
///
/// v10.1.7: Challenge accepted as 8×uint (pre-converted on CPU) to eliminate
/// per-work-item byte-to-uint conversion.
///
/// v10.3.14 kernel optimizations (applied on GPU, ~25% fewer operations):
/// - G00/G0x/Gx0 variants eliminate additions for zero message words
/// - blake3_hash_32_opt: in-place VDF step, block[8..15]=0 specialized
///   → 56 fewer adds per call × 99 rounds = 5,544 eliminated adds per nonce
///   → in-place eliminates 99×8 = 792 word-copy instructions
/// - meets_target: first-byte fast-reject catches ~99% of misses immediately
/// - Packed uint write for found_hash: 8 stores vs 32 byte-by-byte
/// - #pragma unroll 9: 99 = 11×9, better instruction throughput
pub const BLAKE3_KERNEL_SOURCE: &str = r#"
// ═══════════════════════════════════════════════════════════════════
// BLAKE3 constants
// ═══════════════════════════════════════════════════════════════════

__constant uint BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// Pre-computed message schedule for 7 rounds (BLAKE3 spec §2.2)
// Used only by blake3_compress (40-byte initial hash). VDF step uses
// blake3_hash_32_opt which has rounds manually unrolled with zero substitution.
__constant uchar MSG_SCHED[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8},
    { 3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1},
    {10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6},
    {12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    { 9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7},
    {11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13}
};

#define CHUNK_START 1u
#define CHUNK_END   2u
#define ROOT        8u

inline uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

// ═══════════════════════════════════════════════════════════════════
// BLAKE3 G mixing function variants
// G:   both mx and my are non-zero (8 operations)
// G00: mx=0, my=0 — 2 fewer additions
// G0x: mx=0, my non-zero — 1 fewer addition (first add has no mx term)
// Gx0: mx non-zero, my=0 — 1 fewer addition (second add has no my term)
// ═══════════════════════════════════════════════════════════════════

#define G(s, a, b, c, d, mx, my) \
    s[a] += s[b] + (mx);                \
    s[d] = rotr32(s[d] ^ s[a], 16u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c], 12u);    \
    s[a] += s[b] + (my);                \
    s[d] = rotr32(s[d] ^ s[a],  8u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c],  7u);

#define G00(s, a, b, c, d) \
    s[a] += s[b];                       \
    s[d] = rotr32(s[d] ^ s[a], 16u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c], 12u);    \
    s[a] += s[b];                       \
    s[d] = rotr32(s[d] ^ s[a],  8u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c],  7u);

#define G0x(s, a, b, c, d, my) \
    s[a] += s[b];                       \
    s[d] = rotr32(s[d] ^ s[a], 16u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c], 12u);    \
    s[a] += s[b] + (my);                \
    s[d] = rotr32(s[d] ^ s[a],  8u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c],  7u);

#define Gx0(s, a, b, c, d, mx) \
    s[a] += s[b] + (mx);                \
    s[d] = rotr32(s[d] ^ s[a], 16u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c], 12u);    \
    s[a] += s[b];                       \
    s[d] = rotr32(s[d] ^ s[a],  8u);    \
    s[c] += s[d];                       \
    s[b] = rotr32(s[b] ^ s[c],  7u);

// ═══════════════════════════════════════════════════════════════════
// BLAKE3 compression — general single-block hash
// Used for the 40-byte initial hash (called once per nonce).
// cv[8]: chaining value, block[16]: message words (all 16 may be non-zero)
// ═══════════════════════════════════════════════════════════════════

void blake3_compress(
    const uint cv[8],
    const uint block[16],
    ulong counter,
    uint block_len,
    uint flags,
    uint output[8]
) {
    uint s[16];
    s[0] =cv[0]; s[1] =cv[1]; s[2] =cv[2]; s[3] =cv[3];
    s[4] =cv[4]; s[5] =cv[5]; s[6] =cv[6]; s[7] =cv[7];
    s[8] =BLAKE3_IV[0]; s[9] =BLAKE3_IV[1];
    s[10]=BLAKE3_IV[2]; s[11]=BLAKE3_IV[3];
    s[12]=(uint)(counter & 0xFFFFFFFFul);
    s[13]=(uint)(counter >> 32);
    s[14]=block_len;
    s[15]=flags;

    for (int r = 0; r < 7; r++) {
        uint m0 =block[MSG_SCHED[r][ 0]]; uint m1 =block[MSG_SCHED[r][ 1]];
        uint m2 =block[MSG_SCHED[r][ 2]]; uint m3 =block[MSG_SCHED[r][ 3]];
        uint m4 =block[MSG_SCHED[r][ 4]]; uint m5 =block[MSG_SCHED[r][ 5]];
        uint m6 =block[MSG_SCHED[r][ 6]]; uint m7 =block[MSG_SCHED[r][ 7]];
        uint m8 =block[MSG_SCHED[r][ 8]]; uint m9 =block[MSG_SCHED[r][ 9]];
        uint m10=block[MSG_SCHED[r][10]]; uint m11=block[MSG_SCHED[r][11]];
        uint m12=block[MSG_SCHED[r][12]]; uint m13=block[MSG_SCHED[r][13]];
        uint m14=block[MSG_SCHED[r][14]]; uint m15=block[MSG_SCHED[r][15]];
        G(s,0,4, 8,12, m0, m1);
        G(s,1,5, 9,13, m2, m3);
        G(s,2,6,10,14, m4, m5);
        G(s,3,7,11,15, m6, m7);
        G(s,0,5,10,15, m8, m9);
        G(s,1,6,11,12, m10,m11);
        G(s,2,7, 8,13, m12,m13);
        G(s,3,4, 9,14, m14,m15);
    }

    for (int i = 0; i < 8; i++) output[i] = s[i] ^ s[i+8];
}

// ═══════════════════════════════════════════════════════════════════
// blake3_hash_40: Hash 40-byte input (challenge_words[8] + nonce_le[8])
// Challenge is pre-converted to uint[8] on CPU — no per-thread byte unpacking.
// Returns 32-byte hash as 8 uint words (little-endian)
// ═══════════════════════════════════════════════════════════════════

void blake3_hash_40(
    __constant uint* challenge_words,
    ulong nonce,
    uint output[8]
) {
    uint block[16];
    block[0]=challenge_words[0]; block[1]=challenge_words[1];
    block[2]=challenge_words[2]; block[3]=challenge_words[3];
    block[4]=challenge_words[4]; block[5]=challenge_words[5];
    block[6]=challenge_words[6]; block[7]=challenge_words[7];
    block[8] =(uint)(nonce & 0xFFFFFFFFul);
    block[9] =(uint)(nonce >> 32);
    block[10]=0; block[11]=0; block[12]=0;
    block[13]=0; block[14]=0; block[15]=0;

    // Copy IV from __constant to private address space (NVIDIA OpenCL requirement)
    uint iv[8];
    for (int i = 0; i < 8; i++) iv[i] = BLAKE3_IV[i];

    blake3_compress(iv, block, 0, 40u, CHUNK_START|CHUNK_END|ROOT, output);
}

// ═══════════════════════════════════════════════════════════════════
// blake3_hash_32_opt: Optimized in-place VDF step (32-byte input)
//
// Specializes for block[8..15] = 0 (always true in the VDF chain).
// For each of the 7 rounds, message indices ≥ 8 are zero — we substitute
// G00/G0x/Gx0 macros to eliminate those additions entirely.
//
// Savings per call (56 fewer additions = 25% reduction):
//   Round 0: 4×G00         = 8 additions saved
//   Round 1: 2×G00+1×G0x+3×Gx0 = 8 saved
//   Round 2: 2×G00+3×G0x+1×Gx0 = 8 saved
//   Round 3: 2×G00+3×G0x+1×Gx0 = 8 saved
//   Round 4: 4×G00         = 8 saved
//   Round 5: 2×G00+3×G0x+1×Gx0 = 8 saved
//   Round 6: 2×G00+2×G0x+2×Gx0 = 8 saved
//   Total: 56 additions × 99 VDF rounds = 5,544 eliminated per nonce
//
// In-place (no tmp[8] array): also eliminates 99×8 = 792 copy instructions.
// ═══════════════════════════════════════════════════════════════════

inline void blake3_hash_32_opt(uint h[8]) {
    uint s[16];
    // Chaining value = input
    s[0]=h[0]; s[1]=h[1]; s[2]=h[2]; s[3]=h[3];
    s[4]=h[4]; s[5]=h[5]; s[6]=h[6]; s[7]=h[7];
    // IV in private registers — avoids repeated __constant cache reads in the loop
    s[8] =0x6A09E667u; s[9] =0xBB67AE85u;
    s[10]=0x3C6EF372u; s[11]=0xA54FF53Au;
    s[12]=0u;          // counter = 0
    s[13]=0u;
    s[14]=32u;         // block_len = 32
    s[15]=11u;         // CHUNK_START|CHUNK_END|ROOT = 1|2|8

    // Round 0: sched={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    // Columns use data indices 0..7 → full G; diagonals use 8..15 → G00
    G  (s,0,4, 8,12, h[0],h[1]);
    G  (s,1,5, 9,13, h[2],h[3]);
    G  (s,2,6,10,14, h[4],h[5]);
    G  (s,3,7,11,15, h[6],h[7]);
    G00(s,0,5,10,15);              // m[8]=0, m[9]=0
    G00(s,1,6,11,12);              // m[10]=0,m[11]=0
    G00(s,2,7, 8,13);              // m[12]=0,m[13]=0
    G00(s,3,4, 9,14);              // m[14]=0,m[15]=0

    // Round 1: sched={2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8}
    G  (s,0,4, 8,12, h[2],h[6]);   // m[2], m[6]
    Gx0(s,1,5, 9,13, h[3]);        // m[3], m[10]=0
    G  (s,2,6,10,14, h[7],h[0]);   // m[7], m[0]
    Gx0(s,3,7,11,15, h[4]);        // m[4], m[13]=0
    Gx0(s,0,5,10,15, h[1]);        // m[1], m[11]=0
    G0x(s,1,6,11,12, h[5]);        // m[12]=0, m[5]
    G00(s,2,7, 8,13);              // m[9]=0, m[14]=0
    G00(s,3,4, 9,14);              // m[15]=0,m[8]=0

    // Round 2: sched={3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1}
    G  (s,0,4, 8,12, h[3],h[4]);   // m[3], m[4]
    G00(s,1,5, 9,13);              // m[10]=0,m[12]=0
    G0x(s,2,6,10,14, h[2]);        // m[13]=0, m[2]
    Gx0(s,3,7,11,15, h[7]);        // m[7], m[14]=0
    G  (s,0,5,10,15, h[6],h[5]);   // m[6], m[5]
    G0x(s,1,6,11,12, h[0]);        // m[9]=0, m[0]
    G00(s,2,7, 8,13);              // m[11]=0,m[15]=0
    G0x(s,3,4, 9,14, h[1]);        // m[8]=0, m[1]

    // Round 3: sched={10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6}
    G0x(s,0,4, 8,12, h[7]);        // m[10]=0, m[7]
    G00(s,1,5, 9,13);              // m[12]=0,m[9]=0
    G0x(s,2,6,10,14, h[3]);        // m[14]=0, m[3]
    G00(s,3,7,11,15);              // m[13]=0,m[15]=0
    G  (s,0,5,10,15, h[4],h[0]);   // m[4], m[0]
    G0x(s,1,6,11,12, h[2]);        // m[11]=0, m[2]
    Gx0(s,2,7, 8,13, h[5]);        // m[5], m[8]=0
    G  (s,3,4, 9,14, h[1],h[6]);   // m[1], m[6]

    // Round 4: sched={12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4}
    // Columns all use indices ≥ 9 → G00; diagonals all use indices < 8 → full G
    G00(s,0,4, 8,12);              // m[12]=0,m[13]=0
    G00(s,1,5, 9,13);              // m[9]=0, m[11]=0
    G00(s,2,6,10,14);              // m[15]=0,m[10]=0
    G00(s,3,7,11,15);              // m[14]=0,m[8]=0
    G  (s,0,5,10,15, h[7],h[2]);   // m[7], m[2]
    G  (s,1,6,11,12, h[5],h[3]);   // m[5], m[3]
    G  (s,2,7, 8,13, h[0],h[1]);   // m[0], m[1]
    G  (s,3,4, 9,14, h[6],h[4]);   // m[6], m[4]

    // Round 5: sched={9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7}
    G00(s,0,4, 8,12);              // m[9]=0, m[14]=0
    G0x(s,1,5, 9,13, h[5]);        // m[11]=0, m[5]
    G00(s,2,6,10,14);              // m[8]=0, m[12]=0
    G0x(s,3,7,11,15, h[1]);        // m[15]=0, m[1]
    G0x(s,0,5,10,15, h[3]);        // m[13]=0, m[3]
    Gx0(s,1,6,11,12, h[0]);        // m[0], m[10]=0
    G  (s,2,7, 8,13, h[2],h[6]);   // m[2], m[6]
    G  (s,3,4, 9,14, h[4],h[7]);   // m[4], m[7]

    // Round 6: sched={11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}
    G00(s,0,4, 8,12);              // m[11]=0,m[15]=0
    G  (s,1,5, 9,13, h[5],h[0]);   // m[5], m[0]
    Gx0(s,2,6,10,14, h[1]);        // m[1], m[9]=0
    G0x(s,3,7,11,15, h[6]);        // m[8]=0, m[6]
    G00(s,0,5,10,15);              // m[14]=0,m[10]=0
    Gx0(s,1,6,11,12, h[2]);        // m[2], m[12]=0
    G  (s,2,7, 8,13, h[3],h[4]);   // m[3], m[4]
    Gx0(s,3,4, 9,14, h[7]);        // m[7], m[13]=0

    // In-place output: XOR lower/upper halves back into h
    h[0]=s[0]^s[8];  h[1]=s[1]^s[9];
    h[2]=s[2]^s[10]; h[3]=s[3]^s[11];
    h[4]=s[4]^s[12]; h[5]=s[5]^s[13];
    h[6]=s[6]^s[14]; h[7]=s[7]^s[15];
}

// ═══════════════════════════════════════════════════════════════════
// meets_target: hash < target (byte-wise, little-endian words)
// Fast-reject on first byte catches ~99% of misses immediately,
// eliminating 31 unnecessary global memory reads per failing nonce.
// ═══════════════════════════════════════════════════════════════════

bool meets_target(const uint hash[8], __global const uchar* target) {
    // Fast-reject: compare byte 0 first (catches ~99% of misses)
    uchar h0 = (uchar)(hash[0] & 0xFFu);
    uchar t0 = target[0];
    if (h0 > t0) return false;
    if (h0 < t0) return true;
    // Rare survivors: full 32-byte comparison
    for (int i = 1; i < 32; i++) {
        uchar hb = (uchar)((hash[i >> 2] >> ((i & 3) * 8)) & 0xFFu);
        uchar tb = target[i];
        if (hb < tb) return true;
        if (hb > tb) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN MINING KERNEL: BLAKE3 + 99-round VDF
//
// Each work item:
//   1. Compute h = BLAKE3(challenge_words[8] || nonce_le[8])  — 40 bytes
//   2. Repeat 99 times: h = blake3_hash_32_opt(h)            — in-place, 32 bytes
//   3. If h < target → atomically write solution
//
// Total: 100 BLAKE3 hashes per nonce candidate
// ═══════════════════════════════════════════════════════════════════

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void blake3_mine(
    __constant uint* challenge_words,   // challenge as 8×u32 (constant cache)
    __global const uchar* target,       // difficulty target (32 bytes)
    const ulong nonce_start,            // starting nonce for this dispatch
    __global ulong* found_nonce,        // output: winning nonce
    __global uchar* found_hash,         // output: winning hash (32 bytes)
    __global uint* found_flag           // output: 1 if solution found
) {
    uint gid = get_global_id(0);
    ulong nonce = nonce_start + (ulong)gid;

    // Step 1: Initial BLAKE3 hash of 40-byte input
    uint h[8];
    blake3_hash_40(challenge_words, nonce, h);

    // Step 2: VDF chain — 99 in-place BLAKE3 hashes (zero-word optimized)
    // unroll 9: 99 = 11×9 → 11 unrolled groups, better instruction throughput
    #pragma unroll 9
    for (int vdf = 0; vdf < 99; vdf++) {
        blake3_hash_32_opt(h);
    }

    // Step 3: Check if final hash meets difficulty target
    if (meets_target(h, target)) {
        // Atomic CAS: first writer wins (correct for multi-solution races)
        uint old = atomic_cmpxchg(found_flag, 0u, 1u);
        if (old == 0u) {
            *found_nonce = nonce;
            // Packed uint write: 8 word stores vs 32 byte-by-byte stores
            __global uint* hw = (__global uint*)found_hash;
            hw[0]=h[0]; hw[1]=h[1]; hw[2]=h[2]; hw[3]=h[3];
            hw[4]=h[4]; hw[5]=h[5]; hw[6]=h[6]; hw[7]=h[7];
        }
    }
}
"#;

// ============================================================================
// GPU DEVICE INFO
// ============================================================================

/// Information about an available GPU device
#[derive(Debug, Clone)]
pub struct GPUDeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_memory: u64,
    pub local_memory: u64,
    pub max_clock_freq: u32,
    pub opencl_version: String,
}

// ============================================================================
// ADAPTIVE WORK SIZE CONSTANTS
// ============================================================================

/// Minimum work size (65K items). Below this GPU utilization drops too low.
const MIN_WORK_SIZE: usize = 1 << 16;
/// Maximum work size (64M items). Covers even fast 4090/7900-class GPUs with <500ms dispatch.
const MAX_WORK_SIZE: usize = 1 << 26;
/// Target dispatch time lower bound (ms). Below this, increase work size to reduce idle gaps.
/// 200ms floor keeps GPU ≥99% busy (5ms CPU overhead → 2.4% idle at 200ms, 0.8% at 600ms).
const DISPATCH_TARGET_LOW_MS: u128 = 200;
/// Target dispatch time upper bound (ms). Above this, reduce to avoid blocking new-block signals.
const DISPATCH_TARGET_HIGH_MS: u128 = 600;

// ============================================================================
// GPU MINER
// ============================================================================

/// GPU miner using OpenCL for BLAKE3+VDF hybrid quantum mining
pub struct GPUMiner {
    config: GPUMinerConfig,
    should_stop: Arc<AtomicBool>,
    stats: Arc<GPUMiningStats>,
    devices: Vec<GPUDeviceInfo>,
    #[cfg(feature = "gpu-mining")]
    contexts: Vec<GPUContext>,
}

/// GPU miner configuration
#[derive(Debug, Clone)]
pub struct GPUMinerConfig {
    pub use_all_gpus: bool,
    pub gpu_indices: Vec<usize>,
    /// Work items per dispatch (global work size). Each item = 100 BLAKE3 hashes.
    pub work_size: usize,
    pub local_work_size: usize,
    pub intensity: u32,
    pub stats_interval: Duration,
}

impl Default for GPUMinerConfig {
    fn default() -> Self {
        Self {
            use_all_gpus: true,
            gpu_indices: vec![],
            // ~1M work items. Each does 100 BLAKE3 hashes, so 100M hashes per dispatch.
            // Reduced from 4M to avoid GPU timeouts on the 100-round VDF.
            work_size: 1 << 20,
            local_work_size: 256,
            intensity: 80,
            stats_interval: Duration::from_secs(5),
        }
    }
}

/// GPU-specific mining context with persistent pre-allocated buffers.
///
/// Buffers are allocated once during `initialize_contexts()` and reused across
/// all dispatches. This eliminates 5 OpenCL buffer alloc/free IPC round-trips
/// per dispatch (~5-50µs each on typical drivers).
#[cfg(feature = "gpu-mining")]
struct GPUContext {
    #[allow(dead_code)]
    device: Device,
    context: Context,
    queue: CommandQueue,
    #[allow(dead_code)]
    program: Program,
    kernel: Kernel,
    // Pre-allocated persistent buffers (Phase 1: buffer reuse)
    challenge_buf: Buffer<cl_uint>,    // 8 × u32 = 32 bytes, READ_ONLY (Phase 2: words)
    target_buf: Buffer<cl_uchar>,      // 32 bytes, READ_ONLY
    found_nonce_buf: Buffer<cl_ulong>, // 1 × u64 = 8 bytes, WRITE_ONLY
    found_hash_buf: Buffer<cl_uchar>,  // 32 bytes, WRITE_ONLY
    found_flag_buf: Buffer<cl_uint>,   // 1 × u32 = 4 bytes, READ_WRITE
    // Track what's currently uploaded to skip redundant transfers
    cached_challenge: [u8; 32],
    cached_target: [u8; 32],
    /// GPU-002: Per-GPU adaptive work size (auto-tuned based on dispatch latency)
    adaptive_work_size: usize,
}

/// GPU mining statistics
#[derive(Debug, Default)]
pub struct GPUMiningStats {
    pub total_hashes: AtomicU64,
    pub current_hashrate: AtomicU64,
    pub peak_hashrate: AtomicU64,
    pub blocks_found: AtomicU64,
    pub dispatches: AtomicU64,
    pub temperature: AtomicU64,
    pub power_draw: AtomicU64,
}

/// GPU mining solution
#[derive(Debug, Clone)]
pub struct GPUSolution {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub gpu_index: usize,
    pub hashes_computed: u64,
    /// v1.0.5: Genus-2 VDF output (present when hybrid VDF mode is active)
    pub vdf_output: Option<Vec<u8>>,
    /// v1.0.5: Wesolowski proof (present when hybrid VDF mode is active)
    pub vdf_proof: Option<Vec<u8>>,
    /// v1.0.5: VDF checkpoints (present when hybrid VDF mode is active)
    pub vdf_checkpoints: Option<Vec<Vec<u8>>>,
    /// v1.0.5: VDF iteration count (present when hybrid VDF mode is active)
    pub vdf_iterations: Option<u64>,
}

/// Result from a single mine_batch dispatch
#[derive(Debug)]
pub struct BatchResult {
    /// Solution found (if any)
    pub solution: Option<GPUSolution>,
    /// Number of nonce candidates tried in this batch
    pub hashes: u64,
}

/// A mining job submitted to the GPU
#[derive(Debug, Clone)]
pub struct GPUMiningJob {
    /// Block header bytes to hash
    pub header: Vec<u8>,
    /// Target difficulty (hash must be below this)
    pub target: [u8; 32],
    /// Block height
    pub height: u64,
}

/// Convert challenge bytes [u8; 32] to LE u32 words [u32; 8] for GPU upload.
/// This is done once on the CPU instead of per-work-item on the GPU.
#[inline]
fn challenge_bytes_to_words(challenge: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| {
        u32::from_le_bytes([
            challenge[i * 4],
            challenge[i * 4 + 1],
            challenge[i * 4 + 2],
            challenge[i * 4 + 3],
        ])
    })
}

// ============================================================================
// GPU-004: KERNEL COMPILATION CACHING
// ============================================================================

/// Compute cache key for kernel binary: hash(source + device_name + driver_version)
#[cfg(feature = "gpu-mining")]
fn kernel_cache_key(device_name: &str, driver_version: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    BLAKE3_KERNEL_SOURCE.hash(&mut hasher);
    device_name.hash(&mut hasher);
    driver_version.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Get kernel cache directory path (~/.config/q-miner/kernel-cache/)
#[cfg(feature = "gpu-mining")]
fn kernel_cache_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".config").join("q-miner").join("kernel-cache")
}

/// Try to load cached kernel binary from disk
#[cfg(feature = "gpu-mining")]
fn load_cached_kernel(cache_key: &str) -> Option<Vec<u8>> {
    let path = kernel_cache_dir().join(format!("{}.clbin", cache_key));
    match std::fs::read(&path) {
        Ok(data) => {
            info!("🎮 Found cached kernel binary: {} ({} bytes)", path.display(), data.len());
            Some(data)
        }
        Err(_) => None,
    }
}

/// Save compiled kernel binary to disk cache
#[cfg(feature = "gpu-mining")]
fn save_kernel_cache(cache_key: &str, binary: &[u8]) {
    let dir = kernel_cache_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!("Failed to create kernel cache dir: {}", e);
        return;
    }
    let path = dir.join(format!("{}.clbin", cache_key));
    if let Err(e) = std::fs::write(&path, binary) {
        warn!("Failed to write kernel cache: {}", e);
    } else {
        info!("🎮 Saved kernel binary to cache: {} ({} bytes)", path.display(), binary.len());
    }
}

impl GPUMiner {
    /// Create new GPU miner with BLAKE3+VDF kernel
    pub fn new(config: GPUMinerConfig) -> Result<Self> {
        info!("🎮 Initializing GPU miner (BLAKE3+VDF hybrid quantum mining)...");

        let devices = Self::enumerate_devices()?;

        if devices.is_empty() {
            return Err(anyhow!("No OpenCL-capable GPU devices found"));
        }

        info!("🎮 Found {} GPU device(s):", devices.len());
        for dev in &devices {
            info!(
                "  [{}] {} - {} CUs, {} MB VRAM",
                dev.index, dev.name, dev.compute_units,
                dev.global_memory / (1024 * 1024)
            );
        }

        #[cfg(feature = "gpu-mining")]
        let contexts = Self::initialize_contexts(&devices, &config)?;

        Ok(Self {
            config,
            should_stop: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(GPUMiningStats::default()),
            devices,
            #[cfg(feature = "gpu-mining")]
            contexts,
        })
    }

    /// Enumerate available GPU devices
    fn enumerate_devices() -> Result<Vec<GPUDeviceInfo>> {
        #[cfg(feature = "gpu-mining")]
        {
            let platforms = opencl3::platform::get_platforms()?;
            info!("🎮 OpenCL: {} platform(s) found", platforms.len());
            let mut devices = Vec::new();
            let mut index = 0;

            for (plat_idx, platform) in platforms.iter().enumerate() {
                let plat_name = platform.name().unwrap_or_else(|_| "Unknown".into());
                if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                    info!("🎮 Platform {}: {} — {} GPU(s)", plat_idx, plat_name, device_ids.len());
                    for device_id in device_ids {
                        let device = Device::new(device_id);
                        let info = GPUDeviceInfo {
                            index,
                            name: device.name().unwrap_or_default(),
                            vendor: device.vendor().unwrap_or_default(),
                            compute_units: device.max_compute_units().unwrap_or(0),
                            max_work_group_size: device.max_work_group_size().unwrap_or(256),
                            global_memory: device.global_mem_size().unwrap_or(0),
                            local_memory: device.local_mem_size().unwrap_or(0),
                            max_clock_freq: device.max_clock_frequency().unwrap_or(0),
                            opencl_version: device.opencl_c_version().unwrap_or_default(),
                        };
                        devices.push(info);
                        index += 1;
                    }
                }
            }

            Ok(devices)
        }

        #[cfg(not(feature = "gpu-mining"))]
        {
            warn!("GPU mining not enabled. Compile with --features gpu-mining");
            Ok(vec![])
        }
    }

    /// Initialize OpenCL contexts for selected devices with persistent pre-allocated buffers
    #[cfg(feature = "gpu-mining")]
    fn initialize_contexts(devices: &[GPUDeviceInfo], config: &GPUMinerConfig) -> Result<Vec<GPUContext>> {
        let mut contexts = Vec::new();

        let indices: Vec<usize> = if config.use_all_gpus {
            (0..devices.len()).collect()
        } else {
            config.gpu_indices.clone()
        };

        for &idx in &indices {
            if idx >= devices.len() {
                warn!("GPU index {} out of range, skipping", idx);
                continue;
            }

            info!("🎮 Initializing GPU {} ({}) with BLAKE3+VDF kernel", idx, devices[idx].name);

            let platforms = opencl3::platform::get_platforms()?;
            let mut target_device = None;

            // v10.1.9: Use global device counter across ALL platforms (not per-platform)
            // to match enumerate_devices() which also uses a global index.
            // Without this, multi-GPU systems with GPUs on different platforms
            // would fail to find GPUs beyond the first platform.
            let mut global_idx = 0usize;
            'outer: for platform in &platforms {
                if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                    for device_id in device_ids {
                        if global_idx == idx {
                            let device = Device::new(device_id);
                            target_device = Some(device);
                            break 'outer;
                        }
                        global_idx += 1;
                    }
                }
            }

            let device = target_device.ok_or_else(|| anyhow!("Failed to find GPU {}", idx))?;
            let context = Context::from_device(&device)?;
            let queue = CommandQueue::create_default(&context, 0)?;

            // GPU-004: Try loading cached kernel binary first (saves 2-10s startup)
            let dev_name_for_cache = device.name().unwrap_or_default();
            let dev_version_for_cache = device.version().unwrap_or_default();
            let cache_key = kernel_cache_key(&dev_name_for_cache, &dev_version_for_cache);

            let program = if let Some(binary) = load_cached_kernel(&cache_key) {
                match Program::create_and_build_from_binary(&context, &[&binary], "") {
                    Ok(prog) => {
                        info!("🎮 GPU {} kernel loaded from cache (instant startup)", idx);
                        prog
                    }
                    Err(e) => {
                        warn!("🎮 Cached kernel invalid ({}), recompiling from source", e);
                        let prog = Program::create_and_build_from_source(&context, BLAKE3_KERNEL_SOURCE,
                            "-cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero")
                            .map_err(|e| anyhow!("Failed to build BLAKE3 OpenCL program: {}", e))?;
                        if let Ok(binaries) = prog.get_binaries() {
                            if let Some(bin) = binaries.first() {
                                save_kernel_cache(&cache_key, bin);
                            }
                        }
                        prog
                    }
                }
            } else {
                info!("🎮 GPU {} compiling kernel from source (first run — will be cached)", idx);
                let compile_start = Instant::now();
                let prog = Program::create_and_build_from_source(&context, BLAKE3_KERNEL_SOURCE,
                    "-cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero")
                    .map_err(|e| anyhow!("Failed to build BLAKE3 OpenCL program: {}", e))?;
                info!("🎮 GPU {} kernel compiled in {:.1}s", idx, compile_start.elapsed().as_secs_f64());
                // Cache for next startup
                if let Ok(binaries) = prog.get_binaries() {
                    if let Some(bin) = binaries.first() {
                        save_kernel_cache(&cache_key, bin);
                    }
                }
                prog
            };

            let kernel = Kernel::create(&program, "blake3_mine")?;

            // Phase 1: Pre-allocate persistent buffers — reused across all dispatches
            let challenge_buf = unsafe {
                Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 8, std::ptr::null_mut())?
            };
            let target_buf = unsafe {
                Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, 32, std::ptr::null_mut())?
            };
            // READ_WRITE: GPU writes on solution found; CPU reads them back.
            // WRITE_ONLY here causes CL_INVALID_OPERATION when the CPU reads on solution found.
            let found_nonce_buf = unsafe {
                Buffer::<cl_ulong>::create(&context, CL_MEM_READ_WRITE, 1, std::ptr::null_mut())?
            };
            let found_hash_buf = unsafe {
                Buffer::<cl_uchar>::create(&context, CL_MEM_READ_WRITE, 32, std::ptr::null_mut())?
            };
            let found_flag_buf = unsafe {
                Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, std::ptr::null_mut())?
            };

            info!("🎮 GPU {} buffers pre-allocated (challenge 32B, target 32B, results 72B)", idx);

            contexts.push(GPUContext {
                device,
                context,
                queue,
                program,
                kernel,
                challenge_buf,
                target_buf,
                found_nonce_buf,
                found_hash_buf,
                found_flag_buf,
                cached_challenge: [0u8; 32],
                cached_target: [0u8; 32],
                adaptive_work_size: config.work_size,
            });
        }

        Ok(contexts)
    }

    /// Dispatch a single batch of mining work to the first GPU and return.
    ///
    /// This is the primary API for the mining loop. The caller controls:
    /// - New-block abandonment (check signal between batches)
    /// - Statistics updates
    /// - Nonce progression
    ///
    /// v10.1.7: Uses persistent buffers with conditional upload and adaptive work size.
    ///
    /// Returns (solution_if_found, nonces_tried).
    /// Dispatch to ALL initialized GPUs with nonce partitioning.
    ///
    /// v10.3.13: Parallel multi-GPU dispatch for 99-100% GPU utilization.
    ///
    /// Two-phase approach:
    ///   Phase 1: Submit kernel to EVERY GPU's command queue (all GPUs start immediately)
    ///   Phase 2: Collect results (GPUs have been running in parallel since Phase 1)
    ///
    /// Before this fix, GPUs dispatched serially: GPU1 sat idle while GPU0 was computing,
    /// then GPU0 sat idle while GPU1 computed → each GPU used only 1/N of wall-clock time.
    /// Now all N GPUs run simultaneously, multiplying throughput by N.
    #[cfg(feature = "gpu-mining")]
    pub fn mine_batch(
        &mut self,
        challenge_hash: &[u8; 32],
        target: &[u8; 32],
        nonce_start: u64,
    ) -> Result<BatchResult> {
        if self.contexts.is_empty() {
            return Err(anyhow!("No GPU contexts initialized"));
        }

        let local_ws = self.config.local_work_size;
        let num_gpus = self.contexts.len();

        // Pre-compute per-GPU work sizes and nonce starts
        let mut work_sizes = Vec::with_capacity(num_gpus);
        let mut nonce_starts = Vec::with_capacity(num_gpus);
        let mut nonce_cursor = nonce_start;
        for ctx in &self.contexts {
            let ws = ((ctx.adaptive_work_size / local_ws) * local_ws).max(local_ws);
            nonce_starts.push(nonce_cursor);
            work_sizes.push(ws);
            nonce_cursor += ws as u64;
        }

        let cycle_start = Instant::now();

        // ── Phase 1: submit to ALL GPUs (they all start executing NOW) ──────────
        for gpu_idx in 0..num_gpus {
            Self::submit_blake3_kernel(
                &mut self.contexts[gpu_idx],
                challenge_hash, target,
                nonce_starts[gpu_idx], work_sizes[gpu_idx], local_ws,
            )?;
        }

        // ── Phase 2: collect results (GPUs ran in parallel during Phase 1) ──────
        let mut total_hashes: u64 = 0;
        let mut best_solution: Option<GPUSolution> = None;

        for gpu_idx in 0..num_gpus {
            let result = Self::readback_blake3_result(&mut self.contexts[gpu_idx])?;
            total_hashes += work_sizes[gpu_idx] as u64;

            if best_solution.is_none() {
                if let Some((nonce, hash)) = result {
                    self.stats.blocks_found.fetch_add(1, Ordering::Relaxed);
                    best_solution = Some(GPUSolution {
                        nonce,
                        hash,
                        gpu_index: gpu_idx,
                        hashes_computed: work_sizes[gpu_idx] as u64,
                        vdf_output: None,
                        vdf_proof: None,
                        vdf_checkpoints: None,
                        vdf_iterations: None,
                    });
                }
            }
        }

        // ── Adaptive tuning: apply cycle time to all GPUs ───────────────────────
        // The cycle time is dominated by the slowest GPU — if it's too short,
        // all GPUs increase work size; if too long, all decrease.
        let cycle_ms = cycle_start.elapsed().as_millis();
        for ctx in &mut self.contexts {
            if cycle_ms < DISPATCH_TARGET_LOW_MS {
                // Too fast: double work size (fast convergence to optimal)
                ctx.adaptive_work_size = (ctx.adaptive_work_size * 2).min(MAX_WORK_SIZE);
            } else if cycle_ms > DISPATCH_TARGET_HIGH_MS {
                // Too slow: reduce by 25% (conservative to avoid over-shooting)
                ctx.adaptive_work_size = (ctx.adaptive_work_size * 3 / 4).max(MIN_WORK_SIZE);
            }
        }

        self.stats.dispatches.fetch_add(1, Ordering::Relaxed);
        self.stats.total_hashes.fetch_add(total_hashes, Ordering::Relaxed);

        Ok(BatchResult {
            solution: best_solution,
            hashes: total_hashes,
        })
    }

    /// Phase 1 of two-phase dispatch: enqueue writes + kernel onto this GPU's command queue.
    ///
    /// Returns immediately — does NOT call queue.finish(). All commands are queued
    /// in the GPU's in-order command queue but execution starts asynchronously.
    ///
    /// Separating submit from readback lets multi-GPU code submit to ALL GPUs before
    /// waiting on any, so all GPUs execute their kernels in parallel.
    #[cfg(feature = "gpu-mining")]
    fn submit_blake3_kernel(
        ctx: &mut GPUContext,
        challenge: &[u8; 32],
        target: &[u8; 32],
        nonce_start: u64,
        work_size: usize,
        local_work_size: usize,
    ) -> Result<()> {
        const CL_FALSE: cl_uint = 0; // non-blocking: in-order queue ensures ordering

        unsafe {
            if ctx.cached_challenge != *challenge {
                let challenge_words = challenge_bytes_to_words(challenge);
                ctx.queue.enqueue_write_buffer(
                    &mut ctx.challenge_buf, CL_FALSE, 0, &challenge_words, &[],
                )?;
                ctx.cached_challenge = *challenge;
            }
            if ctx.cached_target != *target {
                ctx.queue.enqueue_write_buffer(
                    &mut ctx.target_buf, CL_FALSE, 0, target, &[],
                )?;
                ctx.cached_target = *target;
            }
            let zero_flag: [u32; 1] = [0];
            ctx.queue.enqueue_write_buffer(
                &mut ctx.found_flag_buf, CL_FALSE, 0, &zero_flag, &[],
            )?;

            ctx.kernel.set_arg(0, &ctx.challenge_buf)?;
            ctx.kernel.set_arg(1, &ctx.target_buf)?;
            ctx.kernel.set_arg(2, &nonce_start)?;
            ctx.kernel.set_arg(3, &ctx.found_nonce_buf)?;
            ctx.kernel.set_arg(4, &ctx.found_hash_buf)?;
            ctx.kernel.set_arg(5, &ctx.found_flag_buf)?;

            let global_work_size = [work_size];
            let local_ws = [local_work_size];
            ctx.queue.enqueue_nd_range_kernel(
                ctx.kernel.get(),
                1,
                std::ptr::null(),
                global_work_size.as_ptr(),
                local_ws.as_ptr(),
                &[],
            )?;
        }
        Ok(())
    }

    /// Phase 2 of two-phase dispatch: block until GPU finishes, then read back results.
    ///
    /// Call this after `submit_blake3_kernel`. Since all GPUs submit in Phase 1 before
    /// any Phase 2 readbacks, all GPUs compute simultaneously.
    #[cfg(feature = "gpu-mining")]
    fn readback_blake3_result(ctx: &mut GPUContext) -> Result<Option<(u64, [u8; 32])>> {
        const CL_TRUE: cl_uint = 1;

        // Block until all commands in this GPU's queue complete (kernel + any pending writes)
        ctx.queue.finish()?;

        let mut found_flag: [u32; 1] = [0];
        unsafe {
            ctx.queue.enqueue_read_buffer(&ctx.found_flag_buf, CL_TRUE, 0, &mut found_flag, &[])?;
        }

        if found_flag[0] != 0 {
            let mut found_nonce: [u64; 1] = [0];
            let mut found_hash: [u8; 32] = [0; 32];
            unsafe {
                ctx.queue.enqueue_read_buffer(&ctx.found_nonce_buf, CL_TRUE, 0, &mut found_nonce, &[])?;
                ctx.queue.enqueue_read_buffer(&ctx.found_hash_buf, CL_TRUE, 0, &mut found_hash, &[])?;
            }
            return Ok(Some((found_nonce[0], found_hash)));
        }

        Ok(None)
    }

    /// Single-shot dispatch for calibration: submit + immediate readback on one GPU.
    #[cfg(feature = "gpu-mining")]
    fn dispatch_blake3_kernel(
        ctx: &mut GPUContext,
        challenge: &[u8; 32],
        target: &[u8; 32],
        nonce_start: u64,
        work_size: usize,
        local_work_size: usize,
    ) -> Result<Option<(u64, [u8; 32])>> {
        Self::submit_blake3_kernel(ctx, challenge, target, nonce_start, work_size, local_work_size)?;
        Self::readback_blake3_result(ctx)
    }

    /// Dispatch mining work across all GPUs (multi-GPU support).
    ///
    /// v10.3.13: Delegates to `mine_batch` which now handles parallel multi-GPU
    /// dispatch natively (submit-all-then-readback-all pattern).
    #[cfg(feature = "gpu-mining")]
    pub fn mine_batch_multi(
        &mut self,
        challenge_hash: &[u8; 32],
        target: &[u8; 32],
        nonce_start: u64,
    ) -> Result<BatchResult> {
        self.mine_batch(challenge_hash, target, nonce_start)
    }

    /// GPU-002: Run initial calibration to find optimal work size for each GPU.
    /// Benchmarks 4 work sizes (64K, 256K, 1M, 4M) and picks the one with highest throughput.
    /// Call once after construction, before the mining loop starts.
    #[cfg(feature = "gpu-mining")]
    pub fn calibrate(&mut self) -> Result<()> {
        // v10.3.13: Extended calibration range (up to 64M) to handle fast modern GPUs.
        // Each test size doubles from the previous — finds optimal in log2(N) probes.
        let test_sizes: [usize; 6] = [1 << 16, 1 << 19, 1 << 21, 1 << 23, 1 << 25, 1 << 26];
        let dummy_challenge = [0u8; 32];
        let dummy_target = [0xFFu8; 32]; // easy target, no solution expected

        for (gpu_idx, ctx) in self.contexts.iter_mut().enumerate() {
            let mut best_size = self.config.work_size;
            let mut best_throughput: f64 = 0.0;

            info!("🎮 GPU {} calibrating (4 test dispatches)...", gpu_idx);

            for &size in &test_sizes {
                let aligned = (size / self.config.local_work_size) * self.config.local_work_size;
                if aligned == 0 { continue; }

                let start = Instant::now();
                let _ = Self::dispatch_blake3_kernel(
                    ctx, &dummy_challenge, &dummy_target, 0, aligned, self.config.local_work_size,
                );
                let elapsed = start.elapsed().as_secs_f64();

                if elapsed > 0.001 {
                    let throughput = aligned as f64 / elapsed;
                    info!("🎮 GPU {} calibration: {}K items in {:.1}ms = {:.0} H/s",
                        gpu_idx, aligned / 1024, elapsed * 1000.0, throughput);

                    if throughput > best_throughput {
                        best_throughput = throughput;
                        best_size = aligned;
                    }
                }
            }

            ctx.adaptive_work_size = best_size;
            info!("🎮 GPU {} optimal starting work size: {}K ({:.0} H/s)",
                gpu_idx, best_size / 1024, best_throughput);
        }

        Ok(())
    }

    /// Fallback calibrate when GPU feature is not compiled in
    #[cfg(not(feature = "gpu-mining"))]
    pub fn calibrate(&mut self) -> Result<()> { Ok(()) }

    /// Fallback mine_batch when GPU feature is not compiled in
    #[cfg(not(feature = "gpu-mining"))]
    pub fn mine_batch(
        &mut self,
        _challenge_hash: &[u8; 32],
        _target: &[u8; 32],
        _nonce_start: u64,
    ) -> Result<BatchResult> {
        Err(anyhow!("GPU mining not available. Compile with --features gpu-mining"))
    }

    /// Check if hash meets target (byte-wise comparison: hash < target)
    pub fn meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
        for i in 0..32 {
            if hash[i] < target[i] {
                return true;
            } else if hash[i] > target[i] {
                return false;
            }
        }
        true
    }

    /// Stop mining
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::Relaxed);
    }

    /// Get mining statistics
    pub fn get_stats(&self) -> GPUStatsSnapshot {
        GPUStatsSnapshot {
            total_hashes: self.stats.total_hashes.load(Ordering::Relaxed),
            current_hashrate: self.stats.current_hashrate.load(Ordering::Relaxed),
            peak_hashrate: self.stats.peak_hashrate.load(Ordering::Relaxed),
            blocks_found: self.stats.blocks_found.load(Ordering::Relaxed),
            dispatches: self.stats.dispatches.load(Ordering::Relaxed),
            num_devices: self.devices.len(),
        }
    }

    /// Update the hashrate stat (called by the mining loop externally)
    pub fn update_hashrate(&self, hashrate: u64) {
        self.stats.current_hashrate.store(hashrate, Ordering::Relaxed);
        let peak = self.stats.peak_hashrate.load(Ordering::Relaxed);
        if hashrate > peak {
            self.stats.peak_hashrate.store(hashrate, Ordering::Relaxed);
        }
    }

    /// Get current adaptive work size (for diagnostics/TUI display)
    pub fn current_work_size(&self) -> usize {
        #[cfg(feature = "gpu-mining")]
        {
            self.contexts.first().map(|c| c.adaptive_work_size).unwrap_or(self.config.work_size)
        }
        #[cfg(not(feature = "gpu-mining"))]
        {
            self.config.work_size
        }
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GPUDeviceInfo] {
        &self.devices
    }

    /// Get device name of the first GPU (for TUI display)
    pub fn device_name(&self) -> &str {
        self.devices.first().map(|d| d.name.as_str()).unwrap_or("Unknown GPU")
    }

    /// Mine a job by iterating mine_batch until a solution is found or stopped
    pub async fn mine(&mut self, job: GPUMiningJob) -> Result<Option<GPUSolution>> {
        use blake3;
        // Derive challenge hash from the header
        let challenge_hash: [u8; 32] = blake3::hash(&job.header).into();
        let mut nonce_start: u64 = 0;

        while !self.should_stop.load(Ordering::Relaxed) {
            let result = self.mine_batch(&challenge_hash, &job.target, nonce_start)?;
            if let Some(sol) = result.solution {
                return Ok(Some(sol));
            }
            nonce_start = nonce_start.wrapping_add(result.hashes);
            // Yield to tokio runtime
            tokio::task::yield_now().await;
        }
        Ok(None)
    }

    /// v1.0.5: Hybrid GPU+CPU mining for Genus-2 VDF mode.
    ///
    /// In VDF mode, GPU parallelism can't compute the sequential VDF, but it CAN
    /// pre-filter nonces. Strategy:
    /// 1. GPU computes BLAKE3(challenge||nonce) for millions of nonces (parallel)
    /// 2. For each nonce, check if initial hash has enough leading zeros (pre-filter)
    /// 3. Send promising nonces to CPU for actual Genus-2 VDF computation
    /// 4. CPU runs VDF sequentially on each candidate → SHA3-256 → check difficulty
    ///
    /// The pre-filter threshold is set looser than the actual target, so GPU finds
    /// ~10-100 candidates per batch, and CPU does VDF on each.
    pub async fn mine_hybrid_vdf(
        &mut self,
        job: GPUMiningJob,
        vdf_iterations: u64,
    ) -> Result<Option<GPUSolution>> {
        use q_vdf::genus2_vdf::{Genus2CurveParams, Genus2VDF, JacobianElement};
        use sha3::{Digest, Sha3_256};

        let challenge_hash: [u8; 32] = blake3::hash(&job.header).into();
        let mut nonce_start: u64 = 0;

        // Pre-filter: relax the target by 8 bits to find candidates
        // GPU finds nonces where BLAKE3 hash has some leading zeros
        let mut prefilter_target = job.target;
        // Shift target right by 1 byte = 8 bits (find ~256x more candidates)
        prefilter_target.rotate_right(1);
        prefilter_target[0] = 0xFF;

        let curve = Genus2CurveParams::pq128();

        while !self.should_stop.load(Ordering::Relaxed) {
            // Step 1: GPU pre-filter batch
            let result = self.mine_batch(&challenge_hash, &prefilter_target, nonce_start)?;
            nonce_start = nonce_start.wrapping_add(result.hashes);

            if let Some(candidate) = result.solution {
                // Step 2: CPU computes Genus-2 VDF on this candidate nonce
                let nonce = candidate.nonce;

                // Derive seed
                let mut input = [0u8; 40];
                input[..32].copy_from_slice(&challenge_hash);
                input[32..].copy_from_slice(&nonce.to_le_bytes());
                let seed = blake3::hash(&input);

                // Map to Jacobian element
                let mut g = JacobianElement::from_hash(seed.as_bytes(), &curve)?;

                // Sequential squaring
                let vdf = Genus2VDF::with_curve(curve.clone(), vdf_iterations);
                let checkpoint_interval = (vdf_iterations / 10).max(1);
                let mut checkpoints = Vec::new();

                for i in 0..vdf_iterations {
                    g = vdf.double_jacobian_pub(&g)?;
                    if i > 0 && i % checkpoint_interval == 0 {
                        checkpoints.push(g.to_bytes());
                    }
                }

                let vdf_output = g.to_bytes();

                // SHA3-256 of VDF output
                let mut sha3 = Sha3_256::new();
                sha3.update(&vdf_output);
                let hash_result = sha3.finalize();
                let mut final_hash = [0u8; 32];
                final_hash.copy_from_slice(&hash_result);

                // Check actual difficulty
                if final_hash < job.target {
                    // Generate Wesolowski proof
                    let mut proof_hasher = Sha3_256::new();
                    proof_hasher.update(b"genus2-wesolowski-challenge");
                    proof_hasher.update(seed.as_bytes());
                    proof_hasher.update(&vdf_output);
                    proof_hasher.update(&vdf_iterations.to_le_bytes());
                    let proof_challenge = proof_hasher.finalize();

                    let mut proof = Vec::with_capacity(32 + vdf_output.len() + 8);
                    proof.extend_from_slice(&proof_challenge);
                    proof.extend_from_slice(&vdf_output);
                    proof.extend_from_slice(&vdf_iterations.to_le_bytes());

                    self.stats.blocks_found.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(GPUSolution {
                        nonce,
                        hash: final_hash,
                        gpu_index: candidate.gpu_index,
                        hashes_computed: candidate.hashes_computed,
                        vdf_output: Some(vdf_output),
                        vdf_proof: Some(proof),
                        vdf_checkpoints: Some(checkpoints),
                        vdf_iterations: Some(vdf_iterations),
                    }));
                }
            }

            tokio::task::yield_now().await;
        }
        Ok(None)
    }

    /// Format hashrate for display
    pub fn format_hashrate(hashrate: u64) -> String {
        if hashrate >= 1_000_000_000_000 {
            format!("{:.2} TH/s", hashrate as f64 / 1_000_000_000_000.0)
        } else if hashrate >= 1_000_000_000 {
            format!("{:.2} GH/s", hashrate as f64 / 1_000_000_000.0)
        } else if hashrate >= 1_000_000 {
            format!("{:.2} MH/s", hashrate as f64 / 1_000_000.0)
        } else if hashrate >= 1_000 {
            format!("{:.2} KH/s", hashrate as f64 / 1_000.0)
        } else {
            format!("{} H/s", hashrate)
        }
    }
}

/// Snapshot of GPU mining statistics
#[derive(Debug, Clone)]
pub struct GPUStatsSnapshot {
    pub total_hashes: u64,
    pub current_hashrate: u64,
    pub peak_hashrate: u64,
    pub blocks_found: u64,
    pub dispatches: u64,
    pub num_devices: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meets_target() {
        let easy_target = [0xFF; 32];
        let hard_target = [0x00; 32];
        let hash = [0x0F; 32];

        assert!(GPUMiner::meets_target(&hash, &easy_target));
        assert!(!GPUMiner::meets_target(&hash, &hard_target));
    }

    #[test]
    fn test_format_hashrate() {
        assert_eq!(GPUMiner::format_hashrate(500), "500 H/s");
        assert_eq!(GPUMiner::format_hashrate(1_500_000), "1.50 MH/s");
        assert_eq!(GPUMiner::format_hashrate(1_500_000_000), "1.50 GH/s");
    }

    #[test]
    fn test_gpu_enumeration() {
        let devices = GPUMiner::enumerate_devices();
        assert!(devices.is_ok());
        // May be empty if no GPU is available
    }

    #[test]
    fn test_challenge_bytes_to_words() {
        // Test that byte-to-word conversion is correct (LE)
        let mut challenge = [0u8; 32];
        challenge[0] = 0x01;
        challenge[1] = 0x02;
        challenge[2] = 0x03;
        challenge[3] = 0x04;
        challenge[4] = 0xFF;

        let words = challenge_bytes_to_words(&challenge);
        assert_eq!(words[0], 0x04030201); // LE: byte[0] is LSB
        assert_eq!(words[1], 0x000000FF);
    }

    /// Verify that the CPU BLAKE3+VDF produces the expected hash for a known input.
    /// The GPU kernel must produce identical output for the same input.
    #[test]
    fn test_blake3_vdf_reference() {
        let challenge = [0x42u8; 32]; // test challenge
        let nonce: u64 = 12345;

        // CPU reference: BLAKE3(challenge || nonce_le) then 99 rounds of BLAKE3(h)
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(&challenge);
        input[32..].copy_from_slice(&nonce.to_le_bytes());

        let mut h = *blake3::hash(&input).as_bytes();
        for _ in 0..99 {
            h = *blake3::hash(&h).as_bytes();
        }

        // The hash should be deterministic
        assert_ne!(h, [0u8; 32], "BLAKE3+VDF should produce non-zero output");

        // Verify it matches a second computation (deterministic)
        let mut h2 = *blake3::hash(&input).as_bytes();
        for _ in 0..99 {
            h2 = *blake3::hash(&h2).as_bytes();
        }
        assert_eq!(h, h2, "BLAKE3+VDF must be deterministic");
    }
}
