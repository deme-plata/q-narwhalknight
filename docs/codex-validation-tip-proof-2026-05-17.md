# Codex Validation Prompt 2 — 10ms BLAKE3 Tip Proof (`tip-blake3-fs-v1`)

**Repo:** `/opt/orobit/shared/q-narwhalknight`
**Mode:** READ-ONLY cryptographic validation. Find soundness gaps, forgery vectors, and edge cases the existing DeepSeek reviews may have missed. Do NOT modify code. Produce a punch list ranked CRITICAL / HIGH / MEDIUM / LOW.

## Background

`tip-blake3-fs-v1` is a Fiat-Shamir hash-chain commitment served at `/api/v1/proof/tip`. A fresh node verifies it in <10ms before beginning sync, gaining cheap (not zero-knowledge, not SNARK) assurance that the tip claim is consistent with a hash-chain extending from a known anchor.

**Construction summary** (from `crates/q-recursive-proofs/src/tip_proof_v1.rs`):

- 64-byte transcript, BLAKE3 XOF-extended each `extend()`
- `commitment = BLAKE3-keyed(SIS_KEY, 0x01 || anchor_height || anchor_state || tip_height || folded_state || step_count || transcript)`
- Wire size: 184 bytes (was 176; v10.9.51 added `step_count`)
- `step_count` (v10.9.51) closes DeepSeek §8 length-extension; serializes with `#[serde(default)]` for v10.9.41 wire-compat
- Public inputs (anchor + tip claim) are folded into the commitment (v10.9.41 fixed DeepSeek §0 forgery)

DeepSeek has already reviewed v10.9.41 (§0 fix) and v10.9.51 (§8 fix). Your job: **find what they missed**, plus validate the existing fixes are complete.

## Files to audit (in priority order)

1. **`crates/q-recursive-proofs/src/tip_proof_v1.rs`** (536 lines) — Full module. Pay deep attention to:
   - `anchor(anchor_height, anchor_state)` lines 143–174 — transcript seeding, version byte, domain string
   - `extend(prev, new_height, new_state_root, new_prev_block_hash, new_tx_root)` lines 176–234 — the recursive step; check `debug_assert_eq!` is sufficient (not panic in release)
   - `verify(proof, expected_anchor_height, expected_anchor_state)` lines 236–290 — verification logic; look for: missing checks, early returns, branches that leak via timing, accept of malformed proofs
   - `commit(...)` lines 327–365 — keyed-hash commitment; check input absorption order is injective
   - `write_transcript`, `transcript_bytes_of` lines 73–88 — transcript byte layout
   - Test coverage lines 367+: which attack scenarios are NOT covered?

2. **`crates/q-recursive-proofs/src/lib.rs`** — Module exports, any wrapper that could bypass `verify()`.

3. **`crates/q-recursive-proofs/src/light_client/mod.rs`** — `/api/v1/proof/tip` consumer path. Confirm verify() result is actually checked and a failed verify causes the bootstrap to abort, not silently downgrade.

4. **`crates/q-api-server/src/handlers.rs`** — Producer endpoint serving the tip proof. Confirm it ships the same `LatticeTipProof` struct that `verify()` consumes, and that the anchor used is genuinely the network anchor (not arbitrary attacker-chosen).

5. **`docs/tip-proof-v1-technical-review.md`** — DeepSeek's existing review. Don't repeat their findings; look for what's NOT addressed.

6. **`docs/spec-10ms-verification-2026-05-16.tex`** — Spec document; check whether the code matches the spec.

## Specific questions to answer

### Forgery resistance (CRITICAL — proof bypass = sync attack)
- F1. **Cross-anchor forgery**: Can a malicious prover construct a proof that verifies under TWO different `(expected_anchor_height, expected_anchor_state)` pairs? (After v10.9.41 §0 fix this should be infeasible — verify the fix actually works.)
- F2. **Step-count downgrade attack**: `#[serde(default)]` on `step_count` means a v10.9.41 proof (no field) deserializes with `step_count=0`. The verifier asserts `step_count == tip_height - anchor_height`. So a v10.9.41 proof is only accepted when `tip_height == anchor_height` — i.e., anchor-only proofs. Confirm:
   - (a) Producer doesn't accidentally serve a v10.9.41-format proof for a non-anchor tip
   - (b) Verifier cannot be tricked into accepting a `step_count=0` proof for a non-anchor tip via any code path
   - (c) Is there a `min_step_count` floor that prevents replaying old anchor proofs to claim "we're at genesis"?
- F3. **Transcript injection**: `extend()` absorbs `(new_height, new_state_root, new_prev_block_hash, new_tx_root)` without length prefixes. BLAKE3 XOF is collision-resistant on input bytes — but if two different (height, state, prev, tx_root) tuples produce the same concatenated bytes, transcripts collide. Check: are all four fields fixed-width? (u64 + [u8;32] + [u8;32] + [u8;32] = 104 bytes, all fixed — likely safe, but verify.)
- F4. **Commit input absorption**: `commit()` absorbs 6 fields. Are they all fixed-width and absorbed in a fixed order? Could a malicious prover produce two distinct `(anchor_h, anchor_s, tip_h, folded_s, step, transcript)` tuples with the same commitment? (BLAKE3-keyed preimage resistance, but worth checking the absorption is injective.)

### Domain separation (HIGH)
- D1. `BLAKE3("qnk-tip-commit-v1")` is used as the keyed-hash key. Is this sufficient, or should we use `BLAKE3::derive_key(context, ikm)` (BLAKE3's purpose-built domain separator)? Document the trade-off.
- D2. Transcript seed string `b"qnk-tip-blake3-fs-v1"` — is this unique vs every other BLAKE3 absorption in the codebase? (Grep for collisions.)
- D3. Version byte `0x01` is absorbed in both `anchor()` transcript seed AND `commit()`. If we ever ship v2, is there a clean upgrade path? Does the wire format include a version field the verifier can dispatch on?

### Wire format and downgrade (HIGH)
- W1. **Serde compatibility**: `#[serde(default)]` on `step_count`. Is there a similar fallback on any OTHER field that could enable a downgrade attack? Check every field.
- W2. The doc comment says "Wire size = 184 bytes (was 176)". Does bincode actually produce exactly 184 bytes for the current struct? (Test: serialize and `.len()`.) Mismatch between doc and reality is a smell.
- W3. Is `LatticeTipProof::wire_size()` used anywhere as a size assertion (e.g., to reject obviously-malformed payloads)? If not, an attacker can send arbitrarily large payloads to DoS the verifier.

### Timing and DoS (MEDIUM)
- T1. The doc claims "0.3ms on Epsilon's xeon-gold". Is verify() actually constant-time, or are there early returns that leak (e.g., AnchorHeightMismatch returns before AnchorStateMismatch, leaking which check failed)? Timing leak is harmless here (no secret inputs) but worth noting.
- T2. Could a malformed proof cause verify() to panic? (e.g., `tip_height < anchor_height` underflow if computed as `tip_height - anchor_height`?) `step_count` verification is `claimed vs derived`; check the derivation for underflow.
- T3. Bincode deserialization can be expensive for malformed input. Is there a `max_size` cap on incoming proof bytes before deserialization?

### API/integration (MEDIUM)
- I1. The `/api/v1/proof/tip` endpoint: does it cache the current proof or recompute on every request? A 10K req/s flood could be expensive if recomputed.
- I2. Does the API surface the proof's `anchor_height` so the client knows which anchor to verify against, or does it require the client to know the anchor out-of-band? Out-of-band trust is the right model, but verify the client actually checks anchor match.
- I3. If verify() fails, does the bootstrap actually refuse to sync, or just warn? A warning-only path defeats the purpose.

### Future-compat (LOW — informational)
- FC1. When v10.10.0 ships a Module-SIS commitment alongside v1, will both proofs be served from the same endpoint with version negotiation? Document the migration path.

## Output format

Markdown punch list:

```
## CRITICAL — proof can be forged or bypassed
- [issue] — file:line — exploit description — proposed fix

## HIGH — soundness or wire-format hazard
...

## MEDIUM — robustness/DoS

## LOW — informational / spec drift
```

If DeepSeek already addressed an issue, note that and skip. Focus on what's still open.

## What I am NOT asking you to do
- Don't replace BLAKE3 with another hash.
- Don't introduce a real zk-SNARK / FRI — that's the v10.10.0 roadmap, scoped separately.
- Don't audit `q-tip-proof-stir` crate — that's the FRI work, also scoped separately.
- Don't propose performance optimizations unless they fix a DoS.
