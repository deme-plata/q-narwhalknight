# Quillon Graph Private-Crypto Security Issues

**Created**: 2026-05-17
**Scope**: Static review of cryptocurrency-critical API, wallet-auth, stablecoin, privacy, and emergency-control paths.

This tracker records high-impact findings raised for follow-up PRs. Each item is intentionally scoped so it can be fixed independently.

## Verification

These findings were re-checked against the current workspace after review feedback noted that the previous PR artifact was not visible outside the sandbox. See [VERIFICATION.md](VERIFICATION.md) for source anchors, commands, and repository transport notes.

| Issue | Priority | Area | Summary |
| --- | --- | --- | --- |
| [#001](issues/001-stablecoin-mint-consensus-first.md) | CRITICAL | Stablecoin / consensus | QUGUSD minting mutates and persists vault state before signed consensus acceptance. |
| [#002](issues/002-wallet-auth-replay.md) | HIGH | Wallet auth | Wallet auth is replayable and does not bind method/query/body. |
| [#003](issues/003-stablecoin-placeholder-privacy.md) | HIGH | Privacy / cryptography | Stablecoin privacy layer returns fixed placeholder proofs/signatures. |
| [#004](issues/004-stablecoin-collateral-placeholder-state.md) | HIGH | Stablecoin collateral | Collateral manager returns fixed fake per-user state. |
| [#005](issues/005-fixed-point-amount-arithmetic.md) | HIGH | Accounting | 24-decimal amounts are converted through `f64`. |
| [#006](issues/006-server-side-mnemonic-endpoint.md) | HIGH | Wallet key management | Server-side mnemonic endpoint returns raw seed material and entropy. |
| [#007](issues/007-emergency-pause-unauthenticated.md) | CRITICAL | Admin / availability | Emergency pause/resume routes lack founder authentication. |
| [#008](issues/008-aegis-localhost-bypass-default-allow.md) | HIGH | Admin auth | AEGIS localhost bypass fails open when connection metadata is missing. |
| [#009](issues/009-transaction-signature-from-binding.md) | CRITICAL | Transaction signatures | Signature verification can use a public key from `tx.data` without proving it owns `tx.from`. |
| [#010](issues/010-transaction-privacy-placeholder-proofs.md) | HIGH | Privacy / ZK | Transaction privacy proof generator creates placeholder/format-only proof bytes and can leak amount material. |
| [#011](issues/011-wbtc-withdrawal-consensus-burn.md) | HIGH | Bitcoin bridge | wBTC withdrawal deducts local balances outside a consensus burn/withdrawal state machine. |
| [#012](issues/012-payment-api-header-auth.md) | CRITICAL | Payments auth | Payment API trusts raw header strings as wallet authorization and has a stubbed rate limiter. |

## Recommended Fix Order

1. **#012 payment API cryptographic auth** — raw header-string auth can authorize money movement.
2. **#009 signer/from binding** — signatures must prove authority over the debited account.
3. **#007 emergency pause auth** — public kill switch creates immediate liveness risk.
4. **#001 stablecoin mint consensus-first** — prevents unbacked/divergent QUGUSD state.
5. **#005 fixed-point arithmetic** — removes precision bugs before accounting fixes grow.
6. **#002 wallet-auth replay protection** — reduces API replay risk across mutating routes.
7. **#006 mnemonic endpoint** — eliminates server custody of wallet recovery material.
8. **#003/#004/#010 privacy/collateral placeholders** — remove false security claims and fake collateral state.
9. **#008/#011 admin bypass and bridge withdrawal hardening** — close deployment-sensitive auth and bridge accounting gaps.

## Notes

These are documentation-only issue records. No production code was changed in this PR so the repo owner can triage and implement the fixes in separate, focused branches.
