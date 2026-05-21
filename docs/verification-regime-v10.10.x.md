# Verification Regime — what the tip-proof verifier actually does (v10.10.x)

**Status**: Path C measurement layer, week 1 deliverable. Replaces the
soundness-claims paragraphs in `spec-10ms-verification-2026-05-16.tex`
that no longer reflect what the wasm verifier actually runs.

## Why this doc exists

The wasm verifier exposes two separate facts that have been silently
conflated until v10.10.x:

- **Proof format version** (`verifier_version()`, e.g.
  `"latticeguard-rlwe-v1"`) — the *wire shape* of the bytes the verifier
  knows how to deserialize.
- **Soundness regime** (`verification_regime()`, new in v10.10.x) —
  *what guarantee a wallet gets* if `verify_proof_bytes()` returns true.

Pre-v10.10.x wallets read `verifier_version()` and assumed "lattice" in
the name meant cryptographic guarantees were running. The Phase B1
implementation ships chain-integrity-only checks. The new
`verification_regime()` separates these so wallets render an honest banner.

## The three regimes

| Regime | Wire format | What's verified | What's NOT verified | When advisory? |
|---|---|---|---|---|
| **`"chain-integrity-advisory"`** | `LatticeTipProofV2` (`latticeguard-rlwe-v1`) | Anchor binds to expected genesis (DeepSeek §0 defeated). Heights monotone. Per-step `z_in` chains from prior `z_out`. Final `z_out` matches claimed tip. | Per-step `LatticeGuardProof` is **not cryptographically verified** — `LatticeGuardProof::verify` is never called from the default wallet path. | **YES — wallet MUST show a warning banner** |
| **`"full-cryptographic"`** | `LatticeTipProofV3` (`latticefold-modulesis-v1`) | Everything above + folded Module-SIS commitment over the entire chain is cryptographically verified end-to-end in a single ~10ms operation. | — (modulo external crypto-review status). | NO once Phase C ships and is externally audited |
| **`"legacy-blake3-hash"`** | `LatticeTipProof` v1 (`tip-blake3-fs-v1.1`) | BLAKE3 Fiat-Shamir hash-chain over (anchor, tip, transcript). DeepSeek §0 forgery rejected post-v10.9.41. | Bound to BLAKE3 collision-resistance and Grover. ~128-bit quantum security, NOT lattice-based. | Available for fallback; not the default in v10.10.x |

## Wallet UX requirements

While `verification_regime() == "chain-integrity-advisory"`:

1. Render a visible advisory banner on any UI surface that displays
   "trustless verification" or similar trust claims. Suggested copy:

   > **Chain integrity verified** — full cryptographic per-block verify
   > lands in v3. This proof confirms the chain hasn't been spliced,
   > rolled back, or anchored at the wrong genesis. Per-block lattice
   > proofs are downloaded but not yet cryptographically replayed
   > client-side.

2. Use a distinct icon/color (e.g. yellow info badge) rather than the
   green "verified" check that future regimes will use.

3. Read the regime **once at startup**, not per-proof. The regime is
   build-time; the verify RESULT is per-proof.

When `verification_regime() == "full-cryptographic"`:

- Drop the advisory banner.
- If the build is freshly upgraded but external crypto review is
  outstanding, the wallet may surface a separate "unaudited" disclaimer
  — see the Path C plan in `/root/.claude/plans/precious-watching-clock.md`.

## Why we don't run per-step crypto in the default Phase B1 wallet

Two reasons captured at `crates/q-recursive-proofs/src/tip_proof_client.rs:38-50`
and `crates/q-recursive-proofs/src/tip_proof_v2.rs:41`:

1. **Bandwidth.** Each step's `LatticeGuardProof` is 10–50 KB. A
   1000-block chain → 10–50 MB serialized. Wallets on flaky networks
   can't afford a multi-second proof fetch.
2. **Verify time.** `LatticeGuardProof::verify` is roughly 10–100 ms per
   step at pq128 parameters. A 100-block chain = 1–10 seconds verify.
   That's 100–1000× over the 10 ms target.

Phase C (Module-SIS folding) collapses both: ~8 KB constant-size proof
regardless of chain length, ~10 ms verify regardless of chain length.
Phase C scaffold lives at `crates/q-lattice-guard/src/folding.rs` with
every cryptographic method body currently `todo!()`. The work is
research-tier (LatticeFold ePrint 2024/257, LaBRADOR ePrint 2022/1341)
and gated on the measurement deliverables described in the Path C plan.

## How to read measured times

Bench harness lives at
`crates/q-recursive-proofs/benches/tip_proof_verify_bench.rs`. Run:

```bash
cargo bench -p q-recursive-proofs --bench tip_proof_verify_bench --features benchmarks
```

Output: criterion HTML under `target/criterion/`. Four bench groups:

- `tip_proof_v2_deserialize` — bincode → `LatticeTipProofV2` cost.
- `tip_proof_v2_serialize` — symmetric serialize (producer side).
- `tip_verify_v2_structural` — what wallets pay today.
- `tip_proof_v2_e2e_wallet_path` — bytes → deser → verify (full wallet path).

Each runs at chain lengths 10, 100, 1000.

Measured numbers should be pinned in
`docs/measured-verify-times-v10.10.x.md` (this doc tracks the regime
contract; that doc tracks the empirical numbers).

## Migration path

- **v10.10.x** ships `verification_regime() == "chain-integrity-advisory"`.
- Wallets that already shipped reading only `verifier_version()` keep
  working (no breaking change).
- New wallet builds opt into `verification_regime()` and render the
  advisory banner.
- When Phase C cryptography lands and is externally reviewed,
  `verification_regime()` flips to `"full-cryptographic"`. Wallets that
  already render based on this string need no code change to drop the
  banner.

## Related files

- `crates/q-ivc-verifier-wasm/src/lib.rs` — exports
  `verifier_version()` and the new `verification_regime()`.
- `crates/q-recursive-proofs/src/tip_proof_v2.rs` — implements
  `tip_verify_v2` (the structural check).
- `crates/q-recursive-proofs/src/tip_proof_client.rs` — the path between
  "bytes received" and "verify called". The crypto-verify path is here
  but blocked behind the `ingest_with_folder` API which wallets don't use
  by default.
- `crates/q-lattice-guard/src/folding.rs` — Phase C scaffold; `todo!()`
  bodies until cryptography lands.
- `docs/spec-10ms-verification-2026-05-16.tex` — the original 10ms spec.
  Numbers in §3 of that doc are v1-BLAKE3 measurements; do not extrapolate
  to v2 or v3.
- `/root/.claude/plans/precious-watching-clock.md` — Path C plan, includes
  the week 2 re-evaluation checkpoint that decides whether to commit to
  Phase C cryptography based on measured v2 numbers.
