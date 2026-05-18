# ZK / Recursive-Proof Phase Roadmap

**Status**: Living document. v10.9.56 candidate. Opened for external (Codex) review.
**Owner**: Quillon Foundation
**Last updated**: 2026-05-18

---

## 0. Why this document exists

The Quillon Graph codebase has a substantial ZK/recursive-proof subsystem already on disk — roughly **20,000 lines** across five crates. An earlier internal review described parts of it as "stubs". On closer reading that framing is wrong. The system is an *announced, staged build*: Phase 1 is live scaffolding with explicit version strings and warning gates; Phase 2 swaps in real Nova recursion; Phase 4 swaps to a lattice scheme. This document makes that phase model legible — to internal reviewers, to external auditors, and to the deployed code itself.

It also pins down the **storage scheme**: every artefact the proof system needs to persist lives under new keys in the existing `CF_MANIFEST` column family. Zero schema change. Zero migration. Existing nodes' wallet-balance state is untouched.

---

## 1. What's in the tree today

| Crate | Lines | Role | Maturity |
|---|---|---|---|
| `crates/q-zk-stark` | 7,765 | STARK prover + verifier, FRI, AIR | **Phase 2 production cryptography** — see §2.1 |
| `crates/q-recursive-proofs` | 5,933 | Recursive proof protocol scaffold | **Phase 1 scaffold + Phase 2 in progress** |
| `crates/q-lattice-guard` | 3,398 | Lattice commitment scheme | **Phase 4 scaffold** |
| `crates/q-zk-snark` | 2,846 | PLONK + Groth16 wrappers | Mostly **Phase 2 work-in-progress** |
| `crates/q-ivc-verifier-wasm` | 165 | Browser-side verifier (WASM) | **Phase 1 — placeholder-v0** |
| `crates/q-tip-proof-stir` | (separate tree) | Tip-proof via STIR (FRI) | **Active**, v10.9.51 persisted step_count to RocksDB |
| `crates/q-ivc` | (separate tree) | IVC / Nova-style folding | **Phase 2 active** (DeltaBlockCircuit, ExpandA, synthesize_step) |
| `crates/q-sync-optimizers` | (separate tree) | Sync-path proof-aware optimisations | Active |

Recent commits (last 30 days) demonstrate the work is ongoing, not abandoned:

```
049b2ba1b feat(ivc): Phase 2 wire-up — synthesize_step invokes generate_constraints_inner
e43a404d4 feat(ivc): split DeltaBlockCircuit into outer (publics) + inner (Phase 1-5)
ebbdc14e0 docs(ivc): DS-2 honest Nova-crate ADR + DS-3 DeepSeek feedback
9fc5a9432 test(ivc): DS-1 — ExpandA conformance scaffolding (gated until external reference)
4d81443dc fix(ivc): repair q-ivc lib compile — UInt32/UInt8 byte<->bit roundtrip + R1CSVar import
e1e757efc feat(ivc-recursion): synthesize_step real impl — Phase 2 boundary unblock
4f725aa04 docs: DeepSeek handoff — Phase 2 Nova fold (2 coding tasks)
0e0b227a4 fix(ivc-host): ExpandA byte order ρ∥j∥i → ρ∥i∥j (DeepSeek peer review catch)
075e19599 v10.9.51 tip-proof: bind step_count + persist to RocksDB
7fdcaf00a draft(tip-proof): bootstrap verification hook scaffold (Q_REQUIRE_TIP_PROOF tri-state)
```

---

## 2. The Phase Model — explicit

### 2.1 q-zk-stark: where the real cryptography already lives

`crates/q-zk-stark/src/air.rs:178` exposes `AirConstraints::verify_constraints(&trace) → bool`. This is the real constraint verifier. It walks every constraint in the AIR, calls `constraint.evaluate(trace)`, and returns `evaluations.all_satisfied()`. Callers run it BEFORE entering `StarkVerifier::verify`.

`crates/q-zk-stark/src/stark_prover.rs::evaluate_constraints_cpu` returns `Vec::new()` — that is **intentional delegation**, not a placeholder. The function's doc comment is explicit:

```rust
// PLACEHOLDER prover: does not actually evaluate the caller's AIR. The
// verifier's `verify_constraints` requires ALL evaluations to be zero,
// and synthesizing fictional non-zero values here used to reject every
// legitimate proof (including `test_basic_stark_proof` and the Nova
// SRS attestation in `nova_srs_generator_air`).
//
// Empty == "no constraints to violate" per `verify_constraints`. Real
// AIR-driven evaluation lives in callers that do their own check
// (e.g. `AirConstraints::verify_constraints(&trace)`) before calling
// through to `StarkVerifier::verify`.
```

The FRI proof generation in the same file builds real Merkle trees with SHA3-256, runs folding rounds, and emits authentic Reed-Solomon commitments. That's STARK, not scaffolding.

**Codex review ask #1**: confirm that the delegation pattern at `stark_prover.rs:74-86` + `air.rs:178` is sound — i.e. that every code path which calls `StarkVerifier::verify` has already run `AirConstraints::verify_constraints` against the same trace. If any path doesn't, that's a real defect.

### 2.2 q-ivc-verifier-wasm: announced Phase 1

The WASM verifier loaded by the browser wallet is *deliberately* a placeholder right now. The crate's own doc comment states it plainly:

```rust
//! ## Phase 1 — scaffolding (current)
//!
//! The WASM build pipeline, the JS API, and IndexedDB caching all work end-to-end.
//! The verifier function `verify_proof_bytes` is a placeholder that returns true
//! for any non-empty proof. This is intentional — the JS-side wallet plumbing
//! can be wired and tested without depending on Nova being ready. **Phase 1
//! provides NO cryptographic security on its own.** Browser UI must show a
//! prominent warning while `verifier_version()` returns "placeholder-v0".
```

The function `verifier_version()` exists for runtime detection. Three valid return values:

| Version | Meaning |
|---|---|
| `placeholder-v0` | Phase 1. Always returns true on non-empty proofs. Wallet MUST display a warning banner. |
| `nova-bn254-v1` | Phase 2. Real Nova recursive verification over BN254. |
| `latticefold-modulesis-v1` | Phase 4. Post-quantum lattice scheme. |

**Codex review ask #2**: confirm the wallet front-end actually reads `verifier_version()` and displays the warning banner when it returns `placeholder-v0`. If the banner isn't wired, the front-end is implicitly claiming cryptographic security it doesn't have.

### 2.3 q-recursive-proofs: protocol scaffold (Phase 1 → Phase 2)

`crates/q-recursive-proofs/src/protocol/prover_node.rs::fetch_epoch_data` currently returns empty blocks and signatures. `generate_witness` returns a zero vector. The proof protocol's outer shape is in place — round structure, transcript binding, public-input encoding — but the actual *witness* it commits to is empty. So a proof generated today commits to "I claim epoch N is valid, witness = (nothing)".

This is fine *during Phase 1* because the verifier is also `placeholder-v0`. It's NOT fine the moment we flip the verifier to `nova-bn254-v1` — at that point the witness must be real epoch data (real blocks, real validator signatures, real previous proof) or every generated proof fails verification.

The Phase 2 wire-up is what `synthesize_step` (commit `049b2ba1b`) is for: it's the function that connects `generate_constraints_inner` to the real epoch payload. As `synthesize_step` lands fully, `fetch_epoch_data` gets its real implementation behind it, and the proof becomes non-trivial.

**Codex review ask #3**: review the synthesize_step → DeltaBlockCircuit (inner) → fetch_epoch_data wiring and assess: when this lands end-to-end, what's the minimum set of files that have to change AND deploy together to flip `verifier_version` to `nova-bn254-v1` safely? List the diff.

---

## 3. Storage scheme: CF_MANIFEST, no schema change

Existing consensus-critical state already lives in `CF_MANIFEST`. The comment at `crates/q-storage/src/balance_consensus.rs:275` makes the precedent explicit:

```
// Storage: CF_MANIFEST (same as wallet_balance_, zero schema change)
```

Wallet balances are `wallet_balance_<hex_address>` keys in CF_MANIFEST. Adding the proof system's keys to the **same** column family means:

- No new column family creation (zero RocksDB migration risk)
- No checkpoint replay needed (`is_checkpoint_applied()` semantics unchanged)
- Old binaries ignore new keys (forward compat)
- New binaries read+write new keys without disturbing balance state

### 3.1 Proposed CF_MANIFEST key namespace

| Key prefix | Value | Writer | Reader | Notes |
|---|---|---|---|---|
| `tip_proof:latest` | Serialised tip-proof bytes incl. `step_count` | Producer (Phase 1+) | Bootstrap clients | Already lives here per v10.9.51 |
| `tip_proof:<height>` | Per-height tip proof (sparse — every Nth) | Producer | Bootstrap clients | Future use; not required for Phase 2 |
| `recursive_proof:epoch:<N>` | Nova/IVC recursive proof for epoch N | Phase 2+ prover | Phase 2+ verifier | Empty until Phase 2 activates |
| `state_root:epoch:<N>` | 32-byte BLAKE3 SMT root v2 at epoch end | Producer | Verifier + audits | Anchored by recursive proof |
| `verifier_version_required` | String (`placeholder-v0` / `nova-bn254-v1` / …) | Operator config | Verifier gating | Network-wide minimum |
| `proof_audit_log:<unix_ts>` | Verification attempt outcomes (success/failure/version) | Verifier paths | Operators | Replaces ad-hoc warn! logging; allows post-hoc audit |

### 3.2 What stays out of the proof CF namespace

These DO NOT go into the proof namespace, and the proof system MUST NOT touch them:

- `wallet_balance_<hex>` — authoritative balance state
- `qblock:<height>` / `qblock:hash:<hex>` — block storage  
- `qblock:latest`, `qblock:tip_height`, `qblock:contiguous_verified`, `qblock:synced_through` — chain pointers
- `block_height_index:*`, `tx_hash:*` — block + tx indexing

A proof system that needs balance/block data to construct a witness READS those keys; it never writes them.

---

## 4. Safety guards — existing nodes' DBs are off-limits

The user has been explicit (CLAUDE.md Rule 3): Epsilon's wallet balances are authoritative; no code path should ever overwrite them with proof-derived data. This is non-negotiable.

The proof system honours this in three ways:

### 4.1 Default-off service

`Q_ENABLE_RECURSIVE_PROOFS` defaults to `0`. Even the service that *could* consume a proof isn't instantiated unless an operator opts in. So existing production nodes (Epsilon, Beta, Gamma, Delta) cannot consume a proof today — the code path is never reached.

### 4.2 `is_genesis_db()` gate (Phase 2 prerequisite)

Before Phase 2 activates, the proof-consumption path will gain a fail-closed guard:

```rust
if !state.storage.is_genesis_db().await {
    warn!("🛡️ [PROOF-CONSUME] Refusing on non-genesis DB — existing state is authoritative");
    return Err(BootstrapError::ExistingDbAuthoritative);
}
```

`is_genesis_db()` returns true iff the wallet-balance column is empty AND `qblock:tip_height < N_GENESIS_TIP_BOOTSTRAP` (e.g. 100). This is belt-and-braces with the default-off setting.

### 4.3 Proof verification verifies what the proof CLAIMS, not what local state is

A proof asserts "the state root at epoch N is X, witnessed by these signatures, derived recursively from epoch N-1's proof". A new node accepts X. An existing node IGNORES X — it already has its own authoritative state. The proof system never has a code path that takes a proof and overwrites existing balances.

**Codex review ask #4**: trace every code path from `recursive_proofs_api.rs::request_bootstrap` through to anywhere that writes to CF_MANIFEST. Confirm none of those writes target `wallet_balance_*` keys.

---

## 5. The minimum viable v2 activation

For Phase 2 (`nova-bn254-v1`) to ship safely, the following must land **together**:

1. **q-ivc** Nova IVC fully wired: `DeltaBlockCircuit` outer+inner, `synthesize_step` real impl, `ExpandA` byte order verified against an external reference (DeepSeek DS-1 conformance test now in tree).
2. **q-recursive-proofs::fetch_epoch_data** returns real blocks + validator signatures + previous proof.
3. **q-recursive-proofs::generate_witness** returns a real witness vector binding the trace to the public inputs.
4. **q-zk-stark verifier** — confirm the delegation pattern (§2.1) holds across every caller, OR move `AirConstraints::verify_constraints` *inside* `StarkVerifier::verify` so the verifier is self-contained.
5. **q-ivc-verifier-wasm::verify_proof_bytes** — replace placeholder body with `nova_snark::RecursiveSNARK::verify`. `verifier_version()` returns `nova-bn254-v1`. JS API does not change. Old wallets keep working because they detect via `verifier_version()`.
6. **Wallet UI banner** — when `verifier_version()` is `placeholder-v0`, show "Light-client verification scaffolding active — full cryptographic verification pending Phase 2". Banner removes itself once `verifier_version()` ≠ placeholder-v0.
7. **`verifier_version_required` key in CF_MANIFEST** — node refuses to consume a proof from a peer running an older verifier than the network-wide minimum.

Items 1-3 are the work that's actively underway. Items 4-7 are documentation + glue + UX. None of this changes the consensus path. None of it touches existing wallet-balance state.

---

## 6. Open questions for external review (Codex)

1. **Soundness of the delegation pattern at `q-zk-stark/src/stark_prover.rs:74-86`** — does every call site that reaches `StarkVerifier::verify` first invoke `AirConstraints::verify_constraints` against the same trace? If not, list the leak.
2. **Wallet UI placeholder-v0 banner** — is `verifier_version()` actually queried + rendered by `gui/quantum-wallet/src/`? Find the call site (or its absence) and report.
3. **Phase 2 minimum-deploy-together set** — for `verifier_version` to flip to `nova-bn254-v1` safely, what's the minimum set of files+features that must ship in one binary?
4. **CF_MANIFEST key namespace collisions** — review §3.1 against existing CF_MANIFEST keyspace usage; flag any conflict.
5. **`is_genesis_db()`** — does this predicate exist in `crates/q-storage`? If yes, cite it. If no, propose its implementation (presumably checks the wallet-balance iterator is empty + `qblock:tip_height < N`).
6. **Existing tip-proof bootstrap path** — `crates/q-api-server/src/recursive_proofs_api.rs::request_bootstrap` plus `bootstrap_verification_hook` (drafted in `7fdcaf00a`). Trace every read/write target; confirm only `tip_proof:*` and `state_root:*` keys are written, no `wallet_balance_*` ever.
7. **`Q_REQUIRE_TIP_PROOF` tri-state** — `disabled` / `optional` / `required`. Is the gating in `bootstrap_verification_hook` complete? When `required`, what fail-closed behaviour kicks in?
8. **Future**: should the network-wide `verifier_version_required` be a height-gated upgrade (per `q-consensus-guard::Upgrade`) so all nodes flip atomically at a future block?

---

## 7. Glossary

| Term | Meaning |
|---|---|
| AIR | Algebraic Intermediate Representation — the constraint system for a STARK |
| FRI | Fast Reed-Solomon Interactive oracle proof — the low-degree-testing kernel of STARKs |
| STIR | A FRI variant with better verifier complexity — used in `q-tip-proof-stir` |
| IVC | Incrementally Verifiable Computation — Nova-style folding |
| BLAKE3 SMT v2 | Quillon's Sparse Merkle Tree commitment with BLAKE3 as the underlying hash |
| Tip proof | A proof commitment to a specific block height + state root |
| `placeholder-v0` | The Phase 1 verifier version string — wallet must warn when active |
| `nova-bn254-v1` | The Phase 2 verifier version — real recursive verification over BN254 |
| `latticefold-modulesis-v1` | The Phase 4 post-quantum verifier |
| CF_MANIFEST | The RocksDB column family that stores consensus-critical key→value state |

---

## 8. Reading list (in tree)

- `docs/spec-10ms-verification-2026-05-16.tex` — the original whitepaper draft for 10ms verification
- `docs/deepseek-handoff-phase3-stir-fri.md` — DeepSeek handoff for STIR/FRI work
- `docs/tip-proof-v1-technical-review.md` — internal review of v10.9.51's tip-proof commit
- `crates/q-ivc-verifier-wasm/src/lib.rs` — read the doc comment in full
- `crates/q-zk-stark/src/air.rs:170-220` — the real constraint verifier
- `crates/q-storage/src/balance_consensus.rs:275-300` — the CF_MANIFEST precedent

---

*Opened on `docs/zk-phase-roadmap` for Codex review. The aim is not to ship Phase 2 in this PR — only to make Phase 1's honesty model legible and to confirm the storage scheme + safety guards before Phase 2 code lands.*
