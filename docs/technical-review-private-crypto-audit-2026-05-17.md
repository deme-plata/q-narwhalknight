# Technical Review: Private-Crypto Security Triage

**Date**: 2026-05-17
**Reviewer**: Codex
**Scope**: Static analysis of selected Quillon Graph private cryptocurrency paths.

## Executive Summary

This pass now tracks twelve actionable issues in `docs/security-audit/issues/`. The most urgent risks are raw header-string payment authorization, transaction signatures that are not bound to the debited `from` account, an unauthenticated emergency pause/resume control plane, and stablecoin minting that updates local vault state before a signed consensus-accepted transaction is proven. Several privacy and collateral modules still expose placeholder behavior that should not be shipped as production privacy or solvency logic.

## Review-Comment Follow-up

After feedback that the previous commit/PR artifact was not visible outside this environment, I re-verified each finding against the checked-out source and added line-level verification notes to the issue documents plus `docs/security-audit/VERIFICATION.md`.

## Issues Raised

1. [Issue #001: Make QUGUSD minting consensus-first and balance-safe](security-audit/issues/001-stablecoin-mint-consensus-first.md)
2. [Issue #002: Bind wallet auth to method, body, query, and one-time nonce](security-audit/issues/002-wallet-auth-replay.md)
3. [Issue #003: Replace stablecoin placeholder proofs and signatures](security-audit/issues/003-stablecoin-placeholder-privacy.md)
4. [Issue #004: Back stablecoin collateral calculations with real per-user state](security-audit/issues/004-stablecoin-collateral-placeholder-state.md)
5. [Issue #005: Use integer or decimal-safe arithmetic for 24-decimal token amounts](security-audit/issues/005-fixed-point-amount-arithmetic.md)
6. [Issue #006: Remove or lock down server-side mnemonic generation](security-audit/issues/006-server-side-mnemonic-endpoint.md)
7. [Issue #007: Require founder authentication for emergency pause and resume](security-audit/issues/007-emergency-pause-unauthenticated.md)
8. [Issue #008: Make AEGIS localhost admin bypass fail closed](security-audit/issues/008-aegis-localhost-bypass-default-allow.md)
9. [Issue #009: Bind transaction signatures to the debited `from` address](security-audit/issues/009-transaction-signature-from-binding.md)
10. [Issue #010: Replace transaction privacy proof placeholders with real proofs or disable claims](security-audit/issues/010-transaction-privacy-placeholder-proofs.md)
11. [Issue #011: Make wBTC withdrawals burn through consensus and persist an atomic bridge record](security-audit/issues/011-wbtc-withdrawal-consensus-burn.md)
12. [Issue #012: Replace payment API header-string auth with cryptographic wallet auth](security-audit/issues/012-payment-api-header-auth.md)

## PR #44 Coordination Follow-up

I added `docs/security-audit/PR44_VERIFICATION_QUESTIONS.md` with direct answers to the five requested verification questions: `data[..32]` signer-key dependency, `save_wallet_balance` caller classes, q-storage block-apply entry points, ingestion paths that may bypass signature checks, and `AuthenticatedWallet` body-hash coverage.

## Second Pass Additions

The continued audit added five more findings: AEGIS local admin bypass fail-open behavior, transaction signatures not being bound to the debited `from` address, placeholder transaction privacy proofs, wBTC withdrawals mutating local balances outside consensus, and payment APIs trusting raw header strings as wallet authorization.

## Highest-Risk Observations

### Payment and transaction authorization

The most urgent second-pass findings are raw-string payment authorization and transaction signature verification that can trust a key from `tx.data` without proving it owns `tx.from`. Fix these before expanding money-moving API surface.

### Public emergency pause/resume

The main router exposes emergency pause and resume as POST routes. The handlers currently validate timestamp and reason, then directly mutate `state.emergency_paused`, with TODO comments for founder signature verification. Treat this as a public kill switch until fixed.

### Stablecoin minting before consensus acceptance

`mint_qugusd` mutates and persists `CollateralVault` before checking whether the generated `StableMint` transaction was accepted. The generated transaction is built unsigned and zero-fee. The handler records it as optimistically applied and ignores the submit result.

### Placeholder privacy/collateral logic

The stablecoin privacy layer returns fixed bytes for proofs and signatures. The collateral manager returns fixed values and a `"test"` user position. These should be disabled, feature-gated, or made to return explicit not-implemented errors until real verification and storage-backed collateral state exist.

## Suggested Triage Plan

- First replace payment header-string auth with cryptographic wallet auth and bind transaction signers to `tx.from`.
- Then patch admin authentication on pause/resume and make the AEGIS local bypass fail closed.
- Next move stablecoin mint/burn effects into the block/state transition path and reject unsigned operations.
- Then replace floating-point money parsing with shared fixed-point parsing.
- Finally remove placeholder privacy/collateral behavior or gate it behind development-only features, including the transaction privacy proof generator.

## Validation Performed

Static review only. I did not run the full Rust test suite because this PR only adds issue documentation and does not change executable code. I did run link-resolution and whitespace checks for the Markdown issue set.
