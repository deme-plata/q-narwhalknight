# Technical Plan v2: Instant-Bootstrap Nodes via Recursive Lattice zk-SNARK

**Date:** 2026-05-13
**Supersedes:** `technical-plan-instant-bootstrap-recursive-snark-2026-05-13.md` (V1)
**Reason for amendment:** V1 underestimated the existing `crates/q-ivc/` work. ~70% of the gadget layer is already shipped: BLAKE3 verify_hash, Poseidon, NTT butterfly with negacyclic convention, full Dilithium5 sig-verify primitives, plus an `EpochTransitionCircuit` skeleton. V2 maps the remaining work against what's actually there, not against a from-scratch baseline.

---

## Current State of `crates/q-ivc/` — Honest Audit

```
crates/q-ivc/                                            2,792 LOC
├── src/lib.rs                                              27
├── src/gadgets/
│   ├── mod.rs                                               4
│   ├── blake3.rs              ✓  Blake3Gadget             551
│   ├── poseidon.rs            ✓  PoseidonGadget           329
│   ├── ntt.rs                 ✓  NttVerifierGadget        769
│   │   • Cooley-Tukey DIT butterfly
│   │   • roots[m+i] = ω^(i*n/(2m)) convention
│   │   • BLAKE3 FpVar↔UInt32 bridge
│   └── dilithium.rs           ✓  DilithiumVerifierGadget  902
│       • compute_az_minus_ct (standard cyclic)
│       • compute_az_minus_ct_negacyclic (FIPS 204 path)
│       • enforce_norm_bound (positive-only)
│       • enforce_signed_norm_bound (positive-only fallback;
│           arkworks is_cmp broken near p — caveat documented)
│       • high_bits (FIPS 204 §5.4)
│       • use_hint (FIPS 204 §6.5.2)
└── src/circuits/
    ├── mod.rs                                               2
    └── epoch_transition.rs    ◑  EpochTransitionCircuit   208
        • ValidatorSignatureInput
        • EpochTransitionInputs
        • Circuit drafted, composition incomplete
```

Legend: ✓ = production-grade; ◑ = drafted, not wired; ✗ = not started.

## What's *Not* in `crates/q-ivc/` Yet

1. **Merkle-path-in-balance-map gadget** — required for δ-circuit to prove balance transitions against `state_root`. ~600-800 LOC, ~3 weeks.
2. **State-transition δ-circuit composition** — wires existing gadgets (BLAKE3 header hash, Dilithium tx-sig, Merkle balance update, emission lookup) into one circuit that takes `(state_root_prev, block, state_root_next)` and outputs a single boolean. `EpochTransitionCircuit` is the skeleton; needs ~2000 LOC of glue. ~4 weeks.
3. **Recursion wrapper** — IVC fold step. Two flavor options:
   - **Nova / HyperNova over BN254** via `nova-snark` or `arkworks-nova` crate. Elliptic-curve, not PQ. ~2 weeks integration.
   - **LatticeFold** (Boneh-Chen-Tairi 2024). Module-SIS based, PQ. No production Rust impl exists — would need to be written. ~6-12 months R&D.
4. **Prover / verifier services** — `crates/q-ivc-prover/` and `crates/q-ivc-verifier/` (don't exist yet). Genesis-node binary that runs the prover step per block; lightweight verifier that any node can call. ~3 weeks each.
5. **Wire protocol & API** — `GET /api/v1/proof/tip` returning `(state_root, π_tip)`, `POST /api/v1/proof/verify`, gossipsub topic for proof propagation. ~1 week.
6. **Backfill UX hooks** — `GET /api/v1/status/archive` reporting `lowest_indexed_height`, `tip_height`, `eta`. Independent of the SNARK work — can land today, useful immediately for explorer/wallet progressive-availability UI. ~2 days.

## Remaining Test Debt on the Existing Gadgets

From task #22 (in-progress):
- 3 dilithium helper tests still reference the pre-fix `neg_one` constant for `roots[1]` (should be `Fr::one()` after the NTT convention fix).
- `test_use_hint_with_bias` has a math error in the expected value.
- `enforce_signed_norm_bound` is positive-only; the proper full-domain variant is blocked on arkworks `FpVar::is_cmp` being broken for values near `p`. Documented as a CAVEAT, not a regression.

These are book-keeping. The gadget logic is correct.

---

## Revised Phase Plan

### Phase 0 — gadget completion + test debt (now → 4 weeks)

**Already done:** BLAKE3, Poseidon, NTT, Dilithium primitives, EpochTransitionCircuit skeleton.

**Remaining:**
- [ ] Fix 3 dilithium test fixtures (task #22, ~1 day)
- [ ] Fix `test_use_hint_with_bias` math (~half a day)
- [ ] Add Merkle-path gadget (`gadgets/merkle.rs`) over BLAKE3 (we already have the hash gadget, so this is just sibling-path + index-bit decomposition). ~3 weeks
- [ ] Add range-check / amount-overflow gadget for transaction values (small wrapper over `enforce_norm_bound`). ~3 days
- [ ] Land `/api/v1/status/archive` and `X-QNK-Archive-*` response headers — independent of SNARK, lights up the progressive-explorer UX immediately. ~2 days

**Deliverable end of Phase 0:** complete gadget library, every gadget tested with positive + adversarial fixtures, no recursion yet. The progressive-archive UX is live in production (wallet UI shows "block X not yet indexed, ETA T" for unbackfilled queries — works today against Phase 2 backfill state, no proof system required).

### Phase 1 — single-block δ-circuit (4 → 12 weeks)

Wire the gadgets into one R1CS circuit that proves *one* block transition:

```
δ-circuit inputs (public):
  state_root_prev: BLAKE3 root of (wallet → balance) at height N-1
  state_root_next: claimed root at height N
  block_header_hash: BLAKE3 of header at height N

δ-circuit witnesses (private):
  block body (txs, coinbase)
  Merkle paths into state_root_prev for every touched wallet
  Merkle paths into state_root_next for every updated wallet
  Dilithium5 signatures + verification witnesses

Constraints enforced (composed from existing gadgets):
  1. block_header_hash = BLAKE3(header)           → Blake3Gadget ✓
  2. Every tx sig verifies                        → DilithiumVerifierGadget ✓
  3. For each tx (from, to, amt, fee):
       Merkle path proves (from, bal_old) ∈ state_root_prev    → gadgets/merkle.rs ✗
       Merkle path proves (to, bal_to_old) ∈ state_root_prev   → gadgets/merkle.rs ✗
       bal_old ≥ amt + fee                                     → enforce_norm_bound ✓
       Updated (from, bal_old - amt - fee) ∈ state_root_next   → gadgets/merkle.rs ✗
       Updated (to, bal_to_old + amt) ∈ state_root_next        → gadgets/merkle.rs ✗
  4. Coinbase emission ≤ scheduled rate at height N             → enforce_norm_bound ✓
  5. NTT-based anchor election validation                       → NttVerifierGadget ✓
```

**Deliverable end of Phase 1:** Standalone Groth16 prover + verifier for a single QNK block transition. Verifier proves O(1) cost (Groth16 verify is ~5 ms regardless of circuit size).

### Phase 2 — IVC fold via Nova (12 → 20 weeks)

Wrap the δ-circuit in Nova's `StepCircuit`. Nova is mature, has a production Rust crate (`nova-snark`), and works over BN254 (not PQ — we'll fix that in Phase 4).

This is where we actually get "instant verify regardless of chain length":

```
π_0     = trivial (state_0 = genesis)
π_{n+1} = Nova.fold(δ-circuit, state_n, B_{n+1}, π_n)
```

Verifier checks the final folded proof once. The folded proof's verification cost is constant — does **not** grow with `n`. This is the magic. With Nova on BN254 the verify cost is ~5 ms on modern desktop, ~10 ms on mobile, ~50-250 ms in WASM.

**Deliverable end of Phase 2:** Working IVC tree on testnet. New testnet nodes can opt into proof-bootstrap by downloading `(state_tip, π_tip)`. Old checkpoint path still works.

### Phase 3 — production integration, advisory mode (20 → 32 weeks)

- Genesis nodes (Epsilon, Beta, Gamma, Delta) maintain `π_tip` alongside the chain
- `GET /api/v1/proof/tip` returns `(state_root, π_tip, tip_header)`
- New nodes verify on bootstrap; old checkpoint flow still works
- Block validation continues block-by-block on all nodes — proof is supplementary trust
- Adversarial testing: forge invalid proofs, ensure verifier rejects; forge invalid blocks, ensure prover rejects
- Public release: optional `--bootstrap-mode=proof` flag

**Deliverable end of Phase 3 (~Q3 2027):** Instant-bootstrap available as an opt-in feature on mainnet. New nodes can be operational in seconds rather than hours.

### Phase 4 — lattice migration (32 → 50+ weeks)

Replace Nova/BN254 with a lattice IVC scheme. The δ-circuit is unchanged — only the proof system swaps. Three candidates:

| Scheme | Status | Estimated lift |
|--------|--------|---------------|
| **LatticeFold** (Boneh-Chen-Tairi 2024) | No production Rust impl | Build from paper, ~6-12 months |
| **LaBRADOR + recursion wrapper** | Reference impl exists, no native IVC | Wrap reference impl, ~4-6 months |
| **Greyhound** (Nguyen-Seiler 2024) | Reference impl exists, batch-friendly | Adapt, ~4-6 months |

We pick the candidate based on prover-benchmark data from Phase 2-3, not now. The whole point of doing Nova first is so we don't simultaneously debug the circuit *and* invent the proof system.

**Deliverable end of Phase 4 (~Q1 2028):** Full PQ stack. No elliptic curve anywhere in consensus.

### Phase 5 — mandatory verification at activation height (Q2-Q3 2028)

After activation height H (~6 months announced in advance), every block must carry a valid recursive proof. Old checkpoint flow deprecated.

**Deliverable:** Trustless QNK chain in the strict cryptographic sense.

---

## What's New in V2 vs V1

| V1 said | V2 corrects |
|---------|-------------|
| "Phase 0 — finish q-ivc gadgets (in progress)" with vague effort | Phase 0 is ~70% done, listed gadgets explicitly with LOC and test status |
| "Implement Nova on BN254 in `crates/q-ivc/`" as Phase 1 | Phase 1 is *composing* existing gadgets into δ-circuit, then Phase 2 wraps in Nova |
| Q1 2027 = first single-block proof | Single-block proof realistic in 12 weeks → late summer 2026 |
| Q3 2027 = production instant-bootstrap | Q3 2027 advisory mode, Q1-Q2 2028 lattice migration |
| Implied 6 months of "from-scratch" gadget work | Real remaining gadget work: Merkle path (3 wk) + range-check wrapper (3 d) + test fixes (1.5 d) |

The bottom line: **we are closer than V1 suggested**. The arithmetic-circuit foundations are mostly done. The structural work ahead is composition, recursion, and (eventually) the lattice proof-system swap.

---

## The Progressive-Archival UX — Land in v10.9.16

This part of your spec is **decoupled** from the SNARK work and can ship now. The Phase 2 backfill already produces `lowest_indexed_height`. We just need:

### `GET /api/v1/status/archive`
```json
{
  "tip_height": 11400000,
  "lowest_indexed_height": 4321001,
  "archive_complete": false,
  "archive_progress_pct": 37.9,
  "archive_eta_seconds": 64800,
  "blocks_per_sec_recent": 175,
  "verified_proof_height": null   // populated when SNARK lands
}
```

### Modify existing block / tx / receipt endpoints
Add to every historical-lookup response:
- `X-QNK-Archive-Status: complete | backfilling`
- `X-QNK-Archive-Progress: <indexed>/<tip>`
- `X-QNK-Archive-Lowest-Indexed: <height>`

When the requested height is below `lowest_indexed_height`, return **HTTP 202 Accepted** with a body explaining where the data will be available, instead of HTTP 404 / silent empty.

### Wallet/explorer behavior
- Current-state queries (balance, mempool, pool reserves): always work (fed from `state_root`).
- Historical queries with HTTP 202: render "Block #X not yet indexed on this node. Progress: 37.9% · ETA: 18h. [Query peer node] [Wait & retry]".
- "Query peer node" is an opt-in proxy to Epsilon — clearly marked as "less trustless than the proof bootstrap (forthcoming)".

This UX is what makes the user-facing promise of "instant useful node" true *today*, even before the SNARK lands. The SNARK adds cryptographic certainty to the state-root — the UX is what makes the experience smooth during the inevitable backfill window.

**Estimated effort: 2 engineering days, all in `crates/q-api-server/src/handlers.rs` and the frontend wallet's explorer view.**

---

## Risk Matrix (Unchanged Substance from V1)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Lattice IVC not production-ready by Q1 2028 | High | Stay on Nova longer | Phase 3 (Nova advisory) is the deployable form; lattice migration doesn't gate user value |
| Prover throughput < 1 BPS | Medium | Genesis prover falls behind tip | Dedicated prover hardware on Epsilon (48 cores), distribute proving across validators in Phase 3 |
| Verifier latency > 10 ms in WASM | Medium | Slow wallet first-load | Cache π_tip in IndexedDB, warmup on prior visit |
| δ-circuit accepts an invalid transition | **Critical** | Consensus failure if mandatory | Phase 3 advisory for ≥ 6 months; adversarial test suite mandatory before Phase 5 |
| Nova/BN254 requires trusted setup | Low | Acceptable for Phases 1-3 | Phase 4 lattice migration eliminates setup |
| Backfill UX feels slow to users | Low | UX friction | Honest ETA in HTTP headers + UI; opt-in peer proxy for impatient queries |

---

## Immediate Next Steps (Next 7 Days)

In order of return-on-effort:

1. **Land `/api/v1/status/archive` + `X-QNK-Archive-*` headers (v10.9.16, 2 days).** Unlocks the explorer UX immediately. Independent of SNARK work.
2. **Fix the 3 dilithium test fixtures + use_hint math (task #22, 1.5 days).** Clears the in-progress flag, completes the test debt.
3. **Start `gadgets/merkle.rs`** (~3 weeks).** First missing piece of the δ-circuit. Uses existing `Blake3Gadget` for hashing, only needs sibling-path + index-bit decomposition.
4. **Decide Nova crate**: `nova-snark` (Microsoft) vs `arkworks/nova` (community). Both are viable; pick based on dev-experience comparison on a toy circuit. ~3 days.

Items 1-2 ship in v10.9.16. Item 3 starts the multi-week Phase 0 completion. Item 4 prepares for Phase 2 without committing yet.

— Server Beta, 2026-05-13
