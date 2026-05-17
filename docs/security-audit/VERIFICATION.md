# Security-Audit Finding Verification

**Date**: 2026-05-17
**Workspace**: `/workspace/q-narwhalknight`
**Git branch at verification time**: `work`

## Why this file exists

A review comment noted that the previously reported commit/PR artifact was not visible outside the execution sandbox and asked to verify the findings against the live code before filing them. This file records the local verification pass and the exact source anchors used for each issue.

## Repository / PR Transport Notes

- The local workspace contains commit `b017e23` (`docs: raise private crypto security issues`) on branch `work`.
- `git remote -v` produced no configured remotes in this environment, and `gh` was not installed, so the issues cannot be pushed or opened against GitHub directly from this container.
- The PR is therefore represented through the provided `make_pr` tool plus committed repository changes. If you want these as GitHub issues, copy the files under `docs/security-audit/issues/` into your external tracker.

## Verification Commands

```bash
rg -n "mint_qugusd|create_stable_mint_transaction|generate_mint_proof|calculate_quantum_collateral_value|generate_mnemonic|activate_emergency_pause|resume_from_pause|Message format|SHA3-256\\(address" crates/q-api-server/src crates/q-stablecoin/src -g '*.rs'

nl -ba crates/q-api-server/src/stablecoin_api.rs | sed -n '610,685p'
nl -ba crates/q-api-server/src/transaction_utils.rs | sed -n '138,150p;395,414p'
nl -ba crates/q-api-server/src/wallet_auth.rs | sed -n '205,230p'
nl -ba crates/q-stablecoin/src/privacy.rs | sed -n '20,50p'
nl -ba crates/q-stablecoin/src/collateral.rs | sed -n '19,35p'
nl -ba crates/q-api-server/src/handlers.rs | sed -n '7109,7168p;17228,17316p'
nl -ba crates/q-api-server/src/main.rs | sed -n '24675,24682p;24793,24799p;25310,25317p;25432,25433p'

nl -ba crates/q-api-server/src/aegis_auth_middleware.rs | sed -n '180,245p'
nl -ba crates/q-types/src/lib.rs | sed -n '2828,2908p'
nl -ba crates/q-api-server/src/handlers.rs | sed -n '2590,2630p;4350,4470p;4594,4610p'
nl -ba crates/q-storage/src/balance_consensus.rs | sed -n '834,918p'
nl -ba crates/q-api-server/src/privacy_proof_generator.rs | sed -n '1,220p;219,394p'
nl -ba crates/q-api-server/src/bitcoin_deposit_api.rs | sed -n '330,485p'
nl -ba crates/q-api-server/src/payment_api.rs | sed -n '1,45p;80,125p;523,540p;650,820p;860,940p'
```

## Verified Findings

| Issue | Current-source anchors | Verification result |
| --- | --- | --- |
| [#001 Stablecoin mint consensus-first](issues/001-stablecoin-mint-consensus-first.md) | `stablecoin_api.rs:624-685`, `transaction_utils.rs:144`, `transaction_utils.rs:405-414` | Confirmed. Vault mutation/persistence happens before submit result handling, and helper transaction is unsigned/zero-fee. |
| [#002 Wallet-auth replay](issues/002-wallet-auth-replay.md) | `wallet_auth.rs:209`, `wallet_auth.rs:213-222` | Confirmed. Signed message covers address, timestamp, and path only; query/method/body/nonce are absent. |
| [#003 Stablecoin placeholder privacy](issues/003-stablecoin-placeholder-privacy.md) | `privacy.rs:24-47` | Confirmed. Mint proof, burn proof, and quantum signature return fixed bytes. |
| [#004 Placeholder collateral state](issues/004-stablecoin-collateral-placeholder-state.md) | `collateral.rs:21-34` | Confirmed. Request/user are ignored and fixed placeholder position data is returned. |
| [#005 Floating-point amount arithmetic](issues/005-fixed-point-amount-arithmetic.md) | `stablecoin_api.rs:617-628`, `stablecoin_api.rs:640-651`, `stablecoin_api.rs:698-710` | Confirmed. Stablecoin mint/redeem paths convert 24-decimal amounts through `f64`. |
| [#006 Server-side mnemonic endpoint](issues/006-server-side-mnemonic-endpoint.md) | `main.rs:24679`, `handlers.rs:7109-7165` | Confirmed. Public route calls server-side mnemonic generation and returns mnemonic/entropy fields. |
| [#007 Emergency pause unauthenticated](issues/007-emergency-pause-unauthenticated.md) | `main.rs:24795-24797`, `handlers.rs:17228-17314` | Confirmed. Pause/resume are public routes and signature verification remains TODO in handlers. |
| [#008 AEGIS localhost bypass](issues/008-aegis-localhost-bypass-default-allow.md) | `aegis_auth_middleware.rs:185-197`, `main.rs:25310-25317` | Confirmed. `X-Admin-Local: true` bypass treats missing `ConnectInfo` as local. |
| [#009 Signer/from binding](issues/009-transaction-signature-from-binding.md) | `lib.rs:2870-2900`, `handlers.rs:2593-2614`, `balance_consensus.rs:868-912` | Confirmed. Signature verifier can use `tx.data` public key while balance application debits `tx.from`. |
| [#010 Transaction privacy placeholders](issues/010-transaction-privacy-placeholder-proofs.md) | `privacy_proof_generator.rs:5-12`, `privacy_proof_generator.rs:128-161`, `privacy_proof_generator.rs:219-394` | Confirmed. Proof bytes are generated locally from hashes/format scaffolding and failures are non-fatal. |
| [#011 wBTC withdrawal consensus burn](issues/011-wbtc-withdrawal-consensus-burn.md) | `main.rs:25432-25433`, `bitcoin_deposit_api.rs:354-485` | Confirmed. Handler directly edits/persists local token balance before Bitcoin broadcast. |
| [#012 Payment header auth](issues/012-payment-api-header-auth.md) | `payment_api.rs:25-43`, `payment_api.rs:97-116`, `payment_api.rs:523-540`, `payment_api.rs:654-766`, `payment_api.rs:863-880` | Confirmed. Money-moving payment handlers trust raw header text and rate limiting is stubbed. |

## Scope Caveat

This verification is static and line-number based. It does not prove exploitability under every deployment configuration, but it confirms that the documented code paths and TODOs exist in the checked-out workspace.
