# Multisig feature brief — for Codex

**Date**: 2026-05-19
**Status**: Brief, awaiting Codex implementation
**Target version**: v10.10.0 candidate
**Companion**: AFL-1 (`docs/standards/afl-1-protocol-spec.md`), QSHARE-1 (`docs/standards/qshare-treasury-protocol-spec.md`), SEC-CRIT tasks #28, #29

---

## TL;DR

Add M-of-N multisig support to Quillon Graph. Use FROST (Flexible Round-Optimized Schnorr Threshold) for Ed25519 — production-ready, well-studied, and composes cleanly with the existing X-Wallet-Auth signing flow used by AFL-1.

This is one of the building blocks for: solving SEC-CRIT #28 + #29 (admin endpoints accepting unsigned strings as proof), QSHARE treasury governance (mint cap changes, tier rebalancing), and inter-agent coordination (multi-AI agreement on autonomous mint triggers).

---

## 1. Motivation

Three concrete near-term needs:

1. **SEC-CRIT #28 + #29**: admin endpoints currently accept wallet-string-without-signature as proof. Single-signature X-Wallet-Auth fixes this trivially, but for ADMIN operations a single key compromise = network compromise. Multisig is the right fix.

2. **QSHARE-1 governance**: per the QSHARE-1 spec, parameters like `MINT_THRESHOLD` (default 1.5) and `MAX_POOL_FRACTION` (default 0.005) should be governance-controlled, not hardcoded. Multisig allows e.g. 3-of-5 council to update parameters without a chain fork.

3. **Inter-agent coordination**: AI agents from different model vendors (Claude/Codex/Grok/DeepSeek) can co-sign actions when they reach consensus, providing strong multi-vendor proof. Useful for things like Twitter MCP draft approval where we want "at least 2 of 3 LLMs agreed this draft is fine to post."

---

## 2. Approach options

| Approach | Pros | Cons |
|---|---|---|
| **Naive M-of-N** (collect M Ed25519 sigs, verify each) | Simplest, no new crypto | Linear cost in M; signature size = M × 64 bytes |
| **FROST (Schnorr Threshold)** | Single aggregated sig (64 bytes total), well-studied (RFC draft, Zcash uses for ZF Threshold) | Requires interactive key generation (DKG) + interactive signing rounds |
| **MuSig2** | Single-round signing, simpler than FROST | Bitcoin-focused, less library support for chain-level integration |
| **BLS multisig** | Trivial aggregation, single sig output | New crypto assumption (pairing-based); larger keys |

**Recommended: FROST.** It's the most actively maintained option for Ed25519-based threshold signatures, has strong academic foundation (Komlo & Goldberg 2020 + subsequent improvements), and a maintained Rust implementation at `github.com/ZcashFoundation/frost`. Single-aggregated-sig output means existing X-Wallet-Auth verifier needs zero changes — the multisig output is indistinguishable from a single-signer signature.

---

## 3. Design

### 3.1 Two layers

**L1 — protocol-level multisig (where this brief targets):**
- New `MultisigWallet` type with associated threshold (M-of-N) + member public keys
- Smart-contract registry tracking active multisig wallets
- X-Wallet-Auth extension: header carries threshold-signature instead of single signature (transparently, since both are 64-byte Ed25519-compatible)
- Endpoints requiring multisig check that the X-Wallet-Auth address resolves to a multisig wallet AND that the signature is a valid FROST aggregate

**L2 — application-level (out of scope for v1, future work):**
- UI for proposing multisig actions
- DKG (Distributed Key Generation) coordination service
- Off-chain signing-round mediation (which is the harder operational problem)

### 3.2 New types in q-types

```rust
// crates/q-types/src/multisig.rs
pub struct MultisigWallet {
    pub address: Address,             // qnk... — the aggregate-pubkey-derived address
    pub threshold: u8,                 // M
    pub members: Vec<Address>,         // N addresses of co-signers
    pub created_at_height: u64,
    pub key_share_commitments: Vec<[u8; 32]>, // commitments to each member's share (for DKG audit)
}

pub enum MultisigAction {
    Submit(SignedTransaction),         // m-of-n agree on a single tx
    ApproveDraft(Uuid),                // m-of-n approve a Twitter MCP draft
    SetParameter(QSharePolicy),        // m-of-n update QSHARE governance
    RotateMember { remove: Address, add: Address }, // m-of-n rotate the council
}
```

### 3.3 Storage

New RocksDB column family `CF_MULTISIG_WALLETS`:
- Key: `multisig:<addr>`
- Value: postcard-encoded `MultisigWallet` struct

### 3.4 Endpoints

```
POST /api/v1/multisig/create     — register a new MultisigWallet (requires m=N initial cosig)
GET  /api/v1/multisig/<addr>     — view multisig wallet metadata
POST /api/v1/multisig/propose    — propose a multisig action; returns proposal_id
POST /api/v1/multisig/sign       — partial sign + share with proposal_id
GET  /api/v1/multisig/<id>/status — view signing progress
POST /api/v1/multisig/execute    — execute once threshold reached (aggregates + submits)
```

All endpoints use AFL-1's X-Wallet-Auth scheme for member authentication.

### 3.5 FROST integration

Use `frost-ed25519` crate from ZcashFoundation. Provides:
- `frost_ed25519::keys::dkg::part1/part2/part3` for distributed key gen
- `frost_ed25519::round1::commit` + `round2::sign` for signing
- `frost_ed25519::aggregate` for final signature aggregation
- Verification with `ed25519_dalek` — the aggregated signature IS a valid Ed25519 signature against the aggregated pubkey

This is the key insight: **once FROST DKG is complete, the multisig wallet's signatures are indistinguishable from single-signer Ed25519 sigs**. The verifier doesn't need to know it's a multisig. The chain just sees a valid signature against a registered address.

This means: AFL-1's `/api/v1/agent/submit` endpoint, X-Wallet-Auth verifier, Transaction::verify_ed25519_signature — **all work unchanged**. Multisig is purely an off-chain coordination concern after DKG.

---

## 4. Use cases enabled

### 4.1 Admin operations (SEC-CRIT #28/#29 fix)

Replace single-admin keys with 2-of-3 multisig for:
- Validator backup/restore APIs (#29)
- Admin update endpoints (#28, #33)
- Bridge LP intent finalization
- Emission controller parameter changes

### 4.2 QSHARE governance

3-of-5 multisig council that can:
- Update `MINT_THRESHOLD` (default 1.5)
- Update `MAX_POOL_FRACTION` (default 0.5%)
- Pause `try_autonomous_mint` in emergency
- Adjust tier diversification policy

### 4.3 Multi-vendor agent consensus

For Twitter MCP `publish_approved`, require 2-of-3 from a council of agent identities (e.g. Claude + Codex + Grok must all sign the publish). Prevents single-model error from causing embarrassing posts.

### 4.4 Treasury delegation

Users can delegate spending authority to an autonomous agent ONLY when threshold-signed. e.g. "Agent X may submit DCA buys up to 1 QUG/day, but anything above 10 QUG requires my co-sign."

---

## 5. Implementation roadmap

### Phase 1 — types + registry (1-2 days)
- [ ] `crates/q-types/src/multisig.rs` — types
- [ ] `crates/q-storage/src/multisig.rs` — RocksDB CF + accessors
- [ ] Migration: detect `qnk_multisig:` address prefix, route to multisig CF

### Phase 2 — endpoints (2-3 days)
- [ ] `crates/q-api-server/src/multisig_api.rs` — six handlers above
- [ ] Wire into `main.rs` route registration
- [ ] OpenAPI schema additions (`docs/openapi.yaml`)

### Phase 3 — FROST signing helpers (2-3 days)
- [ ] Add `frost-ed25519` dependency
- [ ] `crates/q-multisig/` new crate with DKG + signing helpers (used by CLI tools and any in-process signers)
- [ ] `crates/q-trading-bot/src/multisig.rs` for trading-bot integration

### Phase 4 — MCP tool integration (1-2 days)
- [ ] `mcp__quillon-wallet__multisig_propose` — propose an action
- [ ] `mcp__quillon-wallet__multisig_sign` — partial-sign as a member
- [ ] `mcp__quillon-wallet__multisig_execute` — aggregate + submit

### Phase 5 — use AFL-1 (no code, just verify integration)
- [ ] Test: multisig wallet submits via AFL-1; verify signature works unchanged
- [ ] Test: SEC-CRIT #28/#29 admin endpoint accepts multisig; rejects single sig

---

## 6. Acceptance criteria

1. A 2-of-3 multisig wallet can be created via `/api/v1/multisig/create` with three members
2. A proposed transaction needs all 3 members' partial sigs OR any 2-of-3 to be aggregatable
3. The aggregated signature passes `Transaction::verify_ed25519_signature` unchanged
4. The MCP tools allow agents to participate in multisig flows via stdio
5. SEC-CRIT #28 + #29 fixed by switching their admin endpoints to require multisig
6. QSHARE policy parameters become multisig-gated

---

## 7. Security considerations

- **DKG round-3 verification**: each member must verify their share against commitments before completing DKG. Without it, malicious dealers can break the threshold.
- **Replay protection**: FROST signing rounds have nonces; proposals must include a fresh nonce/identifier to prevent same-message-different-rounds attacks.
- **Quorum failures**: if a member's key is lost, the multisig is partially recoverable down to M-of-(N-1). Rotation policy (§3.2 `MultisigAction::RotateMember`) handles this.

---

## 8. Suggested PR title

`feat(multisig): FROST-Ed25519 multisig wallets + threshold signing for admin / governance / agent-consensus`

## 9. Files Codex should read first

- `docs/standards/afl-1-protocol-spec.md` — X-Wallet-Auth scheme (multisig reuses it)
- `crates/q-trading-bot/src/wallet_auth.rs` — single-signer X-Wallet-Auth reference
- `crates/q-api-server/src/wallet_auth.rs` — server-side verifier
- `crates/q-types/src/lib.rs::Transaction` — signature field, verify_signature (unchanged by multisig)
- `https://github.com/ZcashFoundation/frost` — the FROST library to use
- Komlo & Goldberg, "FROST: Flexible Round-Optimized Schnorr Threshold Signatures", 2020

---

*Drafted by Claude Opus 4.7 for the Quillon Graph multisig work. The button is yours to push, Codex.*
