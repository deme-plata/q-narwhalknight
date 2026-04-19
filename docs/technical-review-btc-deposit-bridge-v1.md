# Technical Security Review: Bitcoin Deposit Bridge (Phase 1)

**Project:** Q-NarwhalKnight Bitcoin Deposit Bridge  
**Version:** v10.2.11 (Phase 1 — Receive-Only)  
**Date:** 2026-04-11  
**Reviewers:** Claude Code (plan agent), Gemma4 27B (Ollama, security audit), Claude Opus (code audit)  
**Status:** Pre-deployment — requesting peer review from DeepSeek and ChatGPT  

---

## 1. Executive Summary

This document reviews the Phase 1 Bitcoin deposit bridge for Q-NarwhalKnight. The bridge enables one-way BTC deposits: users send BTC on-chain to a generated address, and after 6 confirmations, the bridge mints wBTC (wrapped Bitcoin) tokens on the QUG DEX.

**Architecture:** Bitcoin Knots v28.1 (fully synced, 944K blocks, 835GB) running in Docker on Server Delta (5.79.79.158). Bridge code on Server Beta talks to Delta via JSON-RPC with HTTP basic auth. wBTC is an existing DEX token with working trading pairs.

**Scope:** This review covers the deposit bridge module (`deposit_bridge.rs`), the API endpoints (`bitcoin_deposit_api.rs`), and the RPC client (`real_bitcoin_client.rs`). Atomic swap code (separate, all mocks) is explicitly out of scope.

**Risk Profile:** Low TVL cap (1 BTC max), 0.1 BTC max per deposit, receive-only (no withdrawals in Phase 1).

---

## 2. Architecture Overview

```
User Wallet                    QUG Network                     Bitcoin Network
    |                              |                                |
    |  1. POST /deposit/address    |                                |
    |----------------------------->|                                |
    |  <-- bc1q... address + QR    |                                |
    |                              |                                |
    |  2. Send BTC to bc1q...      |                                |
    |------------------------------------------------------------->|
    |                              |                                |
    |                              |  3. Poll listtransactions()    |
    |                              |------------------------------->|
    |                              |  <-- txid, confirmations       |
    |                              |                                |
    |  4. SSE: "detected 1/6"      |                                |
    |<-----------------------------|                                |
    |  5. SSE: "confirmed 6/6"     |                                |
    |<-----------------------------|                                |
    |  6. SSE: "minted"            |                                |
    |<-----------------------------|                                |
    |                              |                                |
    |  wBTC now in DEX wallet      |                                |
    |  Trade wBTC/QUG on DEX       |                                |
```

### Key Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| BridgeWalletClient | `deposit_bridge.rs` | 131-280 | Bitcoin Knots wallet RPC (getnewaddress, listtransactions) |
| DepositBridge | `deposit_bridge.rs` | 336-810 | Main manager: address gen, polling, status tracking, dedup |
| API Handlers | `bitcoin_deposit_api.rs` | 1-310 | REST endpoints for frontend integration |
| Bridge Tokens | `bridge_tokens.rs` | 130-166 | wBTC mint function (production-ready, existing) |
| Bridge Safety | `bridge_safety.rs` | - | Kill switch, amount limits, deposit verification |

### Bitcoin Knots Node (Delta)

| Property | Value |
|----------|-------|
| Version | Satoshi:28.1.0/Knots:20250305 |
| Chain | mainnet, fully synced (block 944,476) |
| Wallet | `qug-bridge` (descriptor, BIP84, SQLite) |
| txindex | Enabled |
| ZMQ | Enabled (28332/28333/28334) |
| Connections | 39 peers (29 in, 10 out) |
| Runtime | Docker container on Delta |
| RPC auth | HTTP basic auth (rpcauth HMAC) |
| RPC allowed IPs | 127.0.0.1, 172.17.0.0/16, Beta, Gamma, Epsilon |

---

## 3. Security Model

### Threat Model

| Threat | Likelihood | Impact | Mitigation |
|--------|-----------|--------|------------|
| Double-mint (same UTXO minted twice) | Medium | Critical | txid:vout dedup set + single-threaded mint |
| Reorg after mint (BTC deposit orphaned) | Low | High | 6 confirmations; reorg checker planned |
| RPC credential theft | Low | Critical | Env vars only, no hardcoded creds |
| Address generation flooding | Medium | Medium | 3 addr/wallet/hour rate limit |
| TVL limit bypass via restart | Medium | High | TVL rebuilt from persisted deposits on startup |
| Hot wallet compromise | Low | High | 0.1 BTC max deposit cap, 1 BTC TVL cap |
| IDOR (viewing other users' deposits) | Medium | Medium | Auth required, wallet ownership verified |

### Safety Limits

| Limit | Value | Enforcement |
|-------|-------|-------------|
| Min confirmations | 6 (hardcoded floor: 3) | `config.min_confirmations.max(3)` |
| Max deposit | 0.1 BTC (10,000,000 sats) | `BTC_MAX_DEPOSIT_SATS` |
| Max bridge TVL | 1 BTC (100,000,000 sats) | Kill switch triggers above |
| Min deposit | 0.0001 BTC (10,000 sats) | `BTC_MIN_DEPOSIT_SATS` |
| Address rate limit | 3 per wallet per hour | `addr_gen_timestamps` HashMap |
| Deposit expiry | 48 hours | Auto-expire in polling loop |
| Kill switch | Manual + auto (TVL breach) | `AtomicBool`, checked every operation |

---

## 4. Findings from Multi-Model Review

### 4.1 Critical Findings (Fixed)

#### C-1: Hardcoded RPC Credentials (FIXED)

**Before:**
```rust
impl Default for DepositBridgeConfig {
    fn default() -> Self {
        Self {
            btc_rpc_url: "http://5.79.79.158:8332".to_string(),
            btc_rpc_user: "qnk".to_string(),
            btc_rpc_pass: "QnkBtcBridge2026".to_string(),
            // ...
        }
    }
}
```

**After:**
```rust
impl Default for DepositBridgeConfig {
    fn default() -> Self {
        Self {
            btc_rpc_url: String::new(),  // SECURITY: require env vars
            btc_rpc_user: String::new(),
            btc_rpc_pass: String::new(),
            // ...
        }
    }
}

impl DepositBridgeConfig {
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("BTC_RPC_URL").ok()?;  // Returns None if not set
        let user = std::env::var("BTC_RPC_USER").unwrap_or_default();
        let pass = std::env::var("BTC_RPC_PASS").unwrap_or_default();
        if user.is_empty() || pass.is_empty() { return None; }
        // ...
    }
}
```

**Risk:** Anyone with repo access could drain the Bitcoin Knots wallet via `sendtoaddress`.  
**Fix:** Credentials only from env vars. Bridge disabled if not configured.

#### C-2: Missing Double-Mint Protection (FIXED)

**Before:** No dedup mechanism. The polling loop could re-detect the same confirmed deposit and trigger multiple mints.

**After:**
```rust
pub struct DepositBridge {
    minted_txids: Arc<Mutex<HashMap<String, String>>>,  // "txid:vout" -> deposit_id
    // ...
}

// In check_deposit_status():
let dedup_key = format!("{}:{}", txid, vout);
{
    let minted = self.minted_txids.lock().await;
    if minted.contains_key(&dedup_key) {
        continue;  // Skip already-minted UTXO
    }
}

// In mark_minted():
let mut minted = self.minted_txids.lock().await;
if minted.contains_key(&dedup_key) {
    return Err(anyhow!("DOUBLE-MINT BLOCKED: UTXO {} already minted", dedup_key));
}
minted.insert(dedup_key, deposit_id.to_string());
```

**Risk:** Same Bitcoin deposit minted multiple times, creating unbacked wBTC.  
**Fix:** In-memory dedup set keyed by txid:vout, checked before mint, written atomically with mint. Rebuilt from persisted deposits on startup.

#### C-3: Missing Reorg Protection (PARTIALLY ADDRESSED)

**Status:** 6 confirmations significantly reduces risk. Background reorg checker is planned but not yet implemented.

**Risk:** If Bitcoin undergoes a 6+ block reorg (extremely rare, requires >50% hashpower), a deposited transaction could be reversed while wBTC remains minted.

**Mitigation:** 
- 6 confirmations = ~$50K+ attack cost at current hashrate, for max 0.1 BTC ($6,500) gain — economically irrational
- 1 BTC TVL cap limits total exposure
- Kill switch available for manual response

**Recommendation for peer review:** Is 6 confirmations sufficient given the 0.1 BTC cap? Should we add a post-mint reorg monitoring window?

### 4.2 High Findings (Fixed)

#### H-1: In-Memory State Loss on Restart (MITIGATED)

**Before:** All deposit state (pending deposits, TVL counter, minted dedup set) was in-memory only. Restart lost everything, allowing TVL bypass and re-minting.

**After:** 
- `load_pending_deposits()` rebuilds dedup set and TVL from persisted data
- `get_all_deposits()` returns all deposits for periodic persistence
- TVL reconstructed: `self.total_minted_sats.store(tvl, ...)` from minted deposit amounts

**Remaining work:** RocksDB persistence layer not yet wired (Step 4 in implementation plan). Currently depends on the API server persisting via existing storage infrastructure.

#### H-2: IDOR on Deposit Status Endpoint (FIXED)

**Before:** `GET /api/v1/bitcoin/deposit/:id` required no authentication.  
**After:** Requires `AuthenticatedWallet`, uses `get_deposit_for_wallet()` which verifies wallet ownership:

```rust
pub async fn get_deposit_for_wallet(
    &self, deposit_id: &str, qug_wallet: &[u8; 32],
) -> Option<DepositAddress> {
    pending.get(deposit_id)
        .filter(|d| d.qug_wallet == *qug_wallet)  // Ownership check
        .cloned()
}
```

#### H-3: IEEE 754 Float-to-Integer Satoshi Conversion (FIXED)

**Before:** `(tx.amount.abs() * 100_000_000.0) as u64` — truncation errors.  
**After:** `(tx.amount.abs() * 100_000_000.0).round() as u64` — correct rounding.

**Note:** The `.abs()` is correct for Bitcoin Knots `listtransactions` which returns positive amounts for `receive` category. The `category != "receive"` filter runs before this conversion.

### 4.3 Medium Findings (Fixed)

#### M-1: Rate Limit Not Enforced (FIXED)

**Before:** `MAX_ADDRS_PER_WALLET_PER_HOUR = 3` was defined but never checked.

**After:**
```rust
let mut timestamps = self.addr_gen_timestamps.lock().await;
let entries = timestamps.entry(wallet_hex).or_default();
entries.retain(|&t| now - t < 3600);  // Remove >1hr old entries
if entries.len() >= MAX_ADDRS_PER_WALLET_PER_HOUR as usize {
    return Err(anyhow!("Rate limit: max {} addresses per wallet per hour", ...));
}
entries.push(now);
```

#### M-2: Min Confirmations Overridable to 0 (FIXED)

**Before:** Env var parsed without floor.  
**After:** `min_confirmations = min_confirmations.max(3)` — minimum 3 confirmations enforced.

#### M-3: Error Messages Leak Internal Details (FIXED)

**Before:** Raw RPC errors (including connection URLs) returned to client.  
**After:** Sanitized error messages. Only rate limit errors pass through; all others return generic messages. Full errors logged server-side.

### 4.4 Low Findings

#### L-1: Hot Wallet Balance Exposed (MITIGATED)

**Before:** `bridge-status` endpoint showed wallet balance to anyone.  
**After:** Wallet balance shown only to authenticated users. Unauthenticated callers see `-1.0` (redacted).

---

## 5. Questions for Peer Review

### For DeepSeek / ChatGPT to evaluate:

1. **Reorg Risk Assessment:** With 6 confirmations and 0.1 BTC max deposit, is the reorg attack vector sufficiently mitigated for a bootstrap bridge? Should we implement a post-mint monitoring window that freezes wBTC if the deposit tx disappears?

2. **Single-Threaded Mint Processing:** The design relies on a single-threaded mint processor (tokio::mpsc channel → sequential processing) to avoid race conditions in the dedup check. Is this sufficient, or should we use a database-level lock (RocksDB WriteBatch with CAS semantics)?

3. **Float Precision:** Bitcoin Knots RPC returns amounts as JSON floats. We use `.round()` to convert to satoshis. Are there edge cases where `round()` still produces incorrect values for legitimate amounts (e.g., 0.00000001 BTC = 1 sat)?

4. **Label-Based Tracking:** We use Bitcoin Knots wallet labels (`deposit:<wallet_hex>`) to scope `listtransactions` queries. If a user sends BTC to a deposit address without using the bridge UI, the label won't match. Is this a problem? Should we also scan by address?

5. **TVL Reconstruction:** On restart, TVL is rebuilt by summing `amount_sats` from all deposits with `Minted` status. If a deposit was minted but the persistence layer hadn't flushed yet, the TVL will be lower than actual, potentially allowing over-minting. How critical is atomic persistence here?

6. **Rate Limiting by Wallet vs IP:** We rate-limit by QUG wallet address (3 addrs/wallet/hour). Should we also rate-limit by IP? (The API server doesn't currently track client IPs for bridge endpoints.)

7. **Deposit Expiry:** Addresses expire after 48 hours. If a user sends BTC to an expired address, the BTC is received by the wallet but no wBTC is minted. The user's BTC is in our custody with no automated recovery path. Is this acceptable for Phase 1, or should expired addresses still be monitored?

---

## 6. Code References

### Key Files (read these for full context)

| File | Description | Lines |
|------|-------------|-------|
| `crates/q-bitcoin-bridge/src/deposit_bridge.rs` | Core bridge logic | ~850 |
| `crates/q-api-server/src/bitcoin_deposit_api.rs` | REST API endpoints | ~310 |
| `crates/q-bitcoin-bridge/src/real_bitcoin_client.rs` | Base Bitcoin RPC client | ~300 |
| `crates/q-api-server/src/bridge_tokens.rs` | wBTC mint/burn (existing, production) | ~220 |
| `crates/q-api-server/src/bridge_safety.rs` | Kill switch + amount limits (existing) | ~400 |

### Configuration (Environment Variables)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BTC_RPC_URL` | Yes | (none) | Bitcoin Knots RPC URL (e.g., `http://5.79.79.158:8332`) |
| `BTC_RPC_USER` | Yes | (none) | RPC username |
| `BTC_RPC_PASS` | Yes | (none) | RPC password |
| `BTC_MIN_CONFIRMATIONS` | No | 6 | Min confirmations (floor: 3) |
| `BTC_BRIDGE_ENABLED` | No | true | Enable/disable bridge |

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/bitcoin/deposit/address` | Required | Generate deposit address |
| GET | `/api/v1/bitcoin/deposit/:id` | Required | Get deposit status (owner only) |
| GET | `/api/v1/bitcoin/deposits` | Required | List deposits for wallet |
| GET | `/api/v1/bitcoin/deposit/bridge-status` | Optional | Bridge health/TVL (balance redacted if no auth) |

### Deposit Status State Machine

```
Awaiting → Detected (0-5 confs) → Confirming (6+ confs) → Minted
    ↓                                                         
  Expired (48h timeout)                                      
    ↓                                                         
  Failed (amount too large, TVL exceeded, RPC error)         
```

---

## 7. Test Plan

### Unit Tests (in `deposit_bridge.rs`)

- [x] Status transition logic
- [x] Config defaults (no hardcoded creds)
- [x] Satoshi conversion (0.1 BTC = 10,000,000 sats)
- [x] Serialization roundtrip

### Integration Tests (planned)

- [ ] Generate address via Bitcoin Knots RPC (regtest)
- [ ] Detect deposit after N confirmations
- [ ] Verify dedup blocks double-mint
- [ ] Verify rate limit (4th address rejected)
- [ ] Verify min confirmations floor (env var set to 0, enforced at 3)
- [ ] Verify TVL kill switch triggers
- [ ] Verify expired deposit not minted
- [ ] Verify wallet ownership check on status endpoint
- [ ] Verify error messages don't leak RPC details
- [ ] Verify persistence rebuild restores dedup set and TVL

### Manual Testing Checklist

- [ ] Deploy to Server Alpha (Docker canary)
- [ ] Generate deposit address and verify on blockchair.com
- [ ] Send testnet BTC and watch confirmation progress in SSE
- [ ] Verify wBTC appears in DEX wallet after 6 confirmations
- [ ] Try to query another wallet's deposit (should fail)
- [ ] Try to generate >3 addresses in 1 hour (should fail)
- [ ] Kill bridge and verify all operations fail

---

## 8. Deployment Plan

Phase 1 deployment uses the existing HA rolling deploy pipeline:

1. Add env vars to Beta service file: `BTC_RPC_URL`, `BTC_RPC_USER`, `BTC_RPC_PASS`
2. Build: `cargo build --release --package q-api-server`
3. Deploy: `echo "y" | ./scripts/ha-deploy.sh full`
4. Verify: Check `journalctl -u q-api-server | grep "Bridge wallet"` for successful init
5. Test: Generate a deposit address via API and verify it appears on Bitcoin blockchain explorers

---

## 9. Future Work (Not in Phase 1)

| Feature | Phase | Description |
|---------|-------|-------------|
| RocksDB persistence | 1.1 | Save deposits to RocksDB for crash recovery |
| Reorg monitoring | 1.1 | Background job re-checks minted deposits for reorgs |
| ZMQ notifications | 2 | Replace polling with instant push from Bitcoin Knots |
| BTC withdrawals | 2 | Burn wBTC → send BTC from bridge wallet |
| LND + LNBits | 3 | Lightning Network for instant deposits/withdrawals |
| Multisig custody | 3 | Move from hot wallet to multisig |

---

*This document was generated for cross-model peer review. Please evaluate the security model, identify missed attack vectors, and answer the questions in Section 5.*
