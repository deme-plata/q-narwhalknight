// Keccak-f[1600] / SHA3-256 in WGSL — proof-of-concept WebGPU miner kernel
// for Quillon Graph.
//
// Caveat: production Quillon mining uses a Genus-2 hyperelliptic-curve VDF,
// not SHA3 PoW. This kernel demonstrates the WebGPU pattern and provides a
// useful TPS test harness. The Genus-2 port is a separate ~6-week effort.
//
// WGSL does not have native u64. Each Keccak lane (64 bits) is stored as
// two u32 (low, high). All Keccak operations are unrolled in u32 pairs.
//
// Memory layout per invocation:
//   * header (32 bytes, vec<u32>) — block header committed to PoW
//   * target (32 bytes) — the SHA3-256 must be lexicographically <= target
//   * nonce_base (u32) + invocation_id (u32) = the trial nonce
//   * If hash <= target: atomically claim a slot in `results` buffer
//
// One workgroup is 256 invocations. Typical dispatch: 1024 workgroups =
// 262144 trial nonces per dispatch, single-pass.

// ───────────────────────────────────────────────────────────────────────────
// Round constants (24 × u64, stored as 48 × u32 pairs in lo/hi order)
// ───────────────────────────────────────────────────────────────────────────

const RC_LO: array<u32, 24> = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
    0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
    0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u,
);
const RC_HI: array<u32, 24> = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
);

// Rotation offsets for Rho step (5×5 matrix, packed row-major)
const RHO: array<u32, 25> = array<u32, 25>(
     0u,  1u, 62u, 28u, 27u,
    36u, 44u,  6u, 55u, 20u,
     3u, 10u, 43u, 25u, 39u,
    41u, 45u, 15u, 21u,  8u,
    18u,  2u, 61u, 56u, 14u,
);

// Pi permutation: Pi[i] is the SOURCE lane that fills lane i.
const PI: array<u32, 25> = array<u32, 25>(
     0u,  6u, 12u, 18u, 24u,
     3u,  9u, 10u, 16u, 22u,
     1u,  7u, 13u, 19u, 20u,
     4u,  5u, 11u, 17u, 23u,
     2u,  8u, 14u, 15u, 21u,
);

// ───────────────────────────────────────────────────────────────────────────
// 64-bit operations expressed as pairs of u32
// ───────────────────────────────────────────────────────────────────────────

// Rotate left by N bits (0 <= N < 64). N is dynamic so we branch on N<32.
fn rotl64(lo: u32, hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) {
        return vec2<u32>(lo, hi);
    }
    if (n < 32u) {
        let nlo = (lo << n) | (hi >> (32u - n));
        let nhi = (hi << n) | (lo >> (32u - n));
        return vec2<u32>(nlo, nhi);
    }
    // n >= 32: swap and rotate by n-32
    let m = n - 32u;
    if (m == 0u) {
        return vec2<u32>(hi, lo);
    }
    let nlo = (hi << m) | (lo >> (32u - m));
    let nhi = (lo << m) | (hi >> (32u - m));
    return vec2<u32>(nlo, nhi);
}

// ───────────────────────────────────────────────────────────────────────────
// State: 25 lanes × 2 u32 = 50 u32 per invocation.
// We use private storage (one state per invocation; cheap on GPU).
// ───────────────────────────────────────────────────────────────────────────

struct KeccakState {
    lo: array<u32, 25>,
    hi: array<u32, 25>,
}

fn keccak_f1600(state: ptr<function, KeccakState>) {
    for (var round = 0u; round < 24u; round = round + 1u) {
        // ── Theta ──
        var c_lo: array<u32, 5>;
        var c_hi: array<u32, 5>;
        for (var x = 0u; x < 5u; x = x + 1u) {
            c_lo[x] = (*state).lo[x] ^ (*state).lo[x + 5u] ^ (*state).lo[x + 10u] ^ (*state).lo[x + 15u] ^ (*state).lo[x + 20u];
            c_hi[x] = (*state).hi[x] ^ (*state).hi[x + 5u] ^ (*state).hi[x + 10u] ^ (*state).hi[x + 15u] ^ (*state).hi[x + 20u];
        }
        var d_lo: array<u32, 5>;
        var d_hi: array<u32, 5>;
        for (var x = 0u; x < 5u; x = x + 1u) {
            let rotated = rotl64(c_lo[(x + 1u) % 5u], c_hi[(x + 1u) % 5u], 1u);
            d_lo[x] = c_lo[(x + 4u) % 5u] ^ rotated.x;
            d_hi[x] = c_hi[(x + 4u) % 5u] ^ rotated.y;
        }
        for (var y = 0u; y < 25u; y = y + 5u) {
            for (var x = 0u; x < 5u; x = x + 1u) {
                (*state).lo[y + x] = (*state).lo[y + x] ^ d_lo[x];
                (*state).hi[y + x] = (*state).hi[y + x] ^ d_hi[x];
            }
        }

        // ── Rho + Pi (combined; new lane i comes from source lane PI[i] rotated by RHO[PI[i]]) ──
        var b_lo: array<u32, 25>;
        var b_hi: array<u32, 25>;
        for (var i = 0u; i < 25u; i = i + 1u) {
            let src = PI[i];
            let rot = rotl64((*state).lo[src], (*state).hi[src], RHO[src]);
            b_lo[i] = rot.x;
            b_hi[i] = rot.y;
        }

        // ── Chi ──
        for (var y = 0u; y < 25u; y = y + 5u) {
            for (var x = 0u; x < 5u; x = x + 1u) {
                let a = y + x;
                let b = y + ((x + 1u) % 5u);
                let c = y + ((x + 2u) % 5u);
                (*state).lo[a] = b_lo[a] ^ ((~b_lo[b]) & b_lo[c]);
                (*state).hi[a] = b_hi[a] ^ ((~b_hi[b]) & b_hi[c]);
            }
        }

        // ── Iota ──
        (*state).lo[0] = (*state).lo[0] ^ RC_LO[round];
        (*state).hi[0] = (*state).hi[0] ^ RC_HI[round];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Bindings
// ───────────────────────────────────────────────────────────────────────────

struct MiningInput {
    header: array<u32, 8>,    // 32-byte block header
    target: array<u32, 8>,    // 32-byte target (hash must be <= target)
    nonce_base: u32,          // base nonce for this dispatch
    _pad: array<u32, 3>,      // align to 16
}

struct Solution {
    nonce: u32,
    hash: array<u32, 8>,
}

@group(0) @binding(0) var<uniform> input: MiningInput;
@group(0) @binding(1) var<storage, read_write> result_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<Solution, 256>;

// ───────────────────────────────────────────────────────────────────────────
// Compute kernel: one invocation = one trial nonce
// ───────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn mine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let invocation = gid.x;
    let nonce = input.nonce_base + invocation;

    // Initialize state to zeros, then absorb the input.
    // SHA3-256 has rate=1088 bits = 136 bytes. Our input is 32 (header) + 4 (nonce) = 36 bytes,
    // which fits in a single block. We pad with 0x06 || 0x00... || 0x80 per FIPS 202.
    var state: KeccakState;
    for (var i = 0u; i < 25u; i = i + 1u) {
        state.lo[i] = 0u;
        state.hi[i] = 0u;
    }

    // Absorb 8 u32 header words into lanes 0-3 (each lane = 2 u32; little-endian on each u32).
    // SHA3 absorbs lanes in little-endian byte order; our u32s are already little-endian on host side.
    state.lo[0] = input.header[0];
    state.hi[0] = input.header[1];
    state.lo[1] = input.header[2];
    state.hi[1] = input.header[3];
    state.lo[2] = input.header[4];
    state.hi[2] = input.header[5];
    state.lo[3] = input.header[6];
    state.hi[3] = input.header[7];

    // Absorb nonce (4 bytes) at offset 32 — bytes 32..36 of the rate.
    // That's the lower 32 bits of lane 4.
    state.lo[4] = nonce;
    state.hi[4] = 0u;

    // Domain separation byte for SHA3: 0x06 right after the message bytes,
    // then zeros, then 0x80 at the end of the rate (byte 135).
    // 0x06 at byte 36 = byte 0 of lane[4]'s upper 32 bits' byte 0… actually
    // byte 36 lands in lane 4 high u32 at the LOW byte position.
    state.hi[4] = state.hi[4] | 0x00000006u;
    // 0x80 at byte 135 = lane 16's high u32, top byte
    state.hi[16] = state.hi[16] | 0x80000000u;

    // Apply Keccak-f
    keccak_f1600(&state);

    // Squeeze first 256 bits = 32 bytes = lanes 0..3 (lo, hi, lo, hi, ...)
    var digest: array<u32, 8>;
    digest[0] = state.lo[0];
    digest[1] = state.hi[0];
    digest[2] = state.lo[1];
    digest[3] = state.hi[1];
    digest[4] = state.lo[2];
    digest[5] = state.hi[2];
    digest[6] = state.lo[3];
    digest[7] = state.hi[3];

    // Compare hash <= target, big-endian by-word from most-significant first
    // (digest[7] is the most-significant per SHA3 spec).
    var is_solution = false;
    var decided = false;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (decided) { continue; }
        let idx = 7u - i;
        if (digest[idx] < input.target[idx]) {
            is_solution = true;
            decided = true;
        } else if (digest[idx] > input.target[idx]) {
            is_solution = false;
            decided = true;
        }
    }
    if (!decided) {
        is_solution = true; // equal counts as a solution
    }

    if (is_solution) {
        let slot = atomicAdd(&result_count, 1u);
        if (slot < 256u) {
            results[slot].nonce = nonce;
            results[slot].hash[0] = digest[0];
            results[slot].hash[1] = digest[1];
            results[slot].hash[2] = digest[2];
            results[slot].hash[3] = digest[3];
            results[slot].hash[4] = digest[4];
            results[slot].hash[5] = digest[5];
            results[slot].hash[6] = digest[6];
            results[slot].hash[7] = digest[7];
        }
    }
}
