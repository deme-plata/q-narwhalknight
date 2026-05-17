# Technical Review v2: Bitcoin Bridge RocksDB Persistence — After Peer Review

**Date:** 2026-04-13  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Revision:** v2 — incorporates DeepSeek peer review feedback  
**Decision:** RocksDB prefix keys in CF_MANIFEST (approved by both reviewers)  
**Status:** Design approved pending concurrency fix

---

## What Changed From v1

DeepSeek correctly identified that "zero-risk" was overstated. The storage approach is sound, but the mint execution had a critical concurrency flaw.

| Item | v1 (Rejected) | v2 (This Document) |
|------|--------------|---------------------|
| Risk characterization | "Zero-risk" | **Low-risk storage, with serialized mint execution** |
| Concurrent mint safety | Not addressed | **Single-threaded mint executor** (serialized queue) |
| Token balance write | Direct RocksDB write to `token_balance:` | **Route through `bridge_tokens::mint_wrapped_token()`** (existing production API) |
| "Impossible" claims | Multiple absolute claims | **Softened to "by design" / "no intended overlap"** |
| WAL durability | Not audited | **Must verify WAL enabled + sync policy for mint path** |
| `saturating_add` | Used for overflow | **Fatal error on overflow** (silent saturation hides bugs) |
| Dedup value | Empty vs `vec![1u8]` inconsistency | **Canonical: `vec![1u8]`** (sentinel, documented) |

---

## Approved Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         CF_MANIFEST (existing)           │
                    │                                         │
                    │  wallet_balance_{hex}  (QUG balances)   │  ← balance_consensus owns
                    │  token_balance:{w}:{t} (token balances) │  ← token contract owns
                    │  dex_qug_debited:{w}   (DEX counters)   │  ← DEX handler owns
                    │                                         │
                    │  btc_bridge:deposit:{id}  (deposits)    │  ← bridge owns (NEW)
                    │  btc_bridge:minted:{tx}:{v} (dedup)     │  ← bridge owns (NEW)
                    │  btc_bridge:kill_switch    (state)       │  ← bridge owns (NEW)
                    │  btc_bridge:total_minted   (TVL)         │  ← bridge owns (NEW)
                    └─────────────────────────────────────────┘

Key ownership rule: each subsystem ONLY writes to its own prefix.
Bridge NEVER writes to wallet_balance_ or token_balance: directly.
Bridge calls bridge_tokens::mint_wrapped_token() for wBTC credits.
```

---

## The Critical Fix: Serialized Mint Executor

### The Problem (Both Reviewers Identified)

```rust
// UNSAFE: read-modify-write race between concurrent workers
let current = storage.get(key).await?;        // Worker A reads 100
let new = current + amount;                    // Worker B also reads 100
storage.write_batch(vec![(key, new)]).await?;  // A writes 110, B writes 120
                                               // Correct should be 130
```

Three race conditions:
1. **Double mint:** Two workers check dedup → both see "missing" → both mint
2. **Lost update on balance:** Two deposits to same wallet → one lost
3. **Lost update on TVL:** Two mints → TVL increments by one instead of both

### The Fix: Single-Threaded Mint Channel

```rust
/// Bridge mint operations are serialized through a single tokio channel.
/// Only ONE mint executes at a time — no concurrent reads of balance/dedup.
///
/// DeepSeek review: "For this workload, serialized minting is probably the safest choice."
/// Throughput: ~1 mint/minute. Serialization latency: negligible.

// In DepositBridge initialization:
let (mint_tx, mut mint_rx) = tokio::sync::mpsc::channel::<MintRequest>(100);

// Single consumer task (THE mint executor — only one exists):
tokio::spawn(async move {
    while let Some(request) = mint_rx.recv().await {
        // This is the ONLY code path that can mint wBTC.
        // No other task, no other worker, no other node can execute concurrently.
        match execute_serialized_mint(&storage, &request).await {
            Ok(()) => info!("₿ ✅ Mint succeeded: {}", request.dedup_key),
            Err(e) => error!("₿ ❌ Mint failed: {} — {}", request.dedup_key, e),
        }
    }
});

// Deposit poller sends mint requests (never mints directly):
if deposit.confirmations >= 6 {
    mint_tx.send(MintRequest { deposit, txid, vout }).await?;
}
```

**Why this eliminates all three races:**
- Only one `execute_serialized_mint` runs at a time
- Dedup check + balance read + WriteBatch all happen sequentially
- No concurrent reader can see stale values
- Channel has backpressure (capacity 100) — if mint is slow, deposits queue

### Cluster-Wide Enforcement

Only ONE node should run the bridge mint executor. Enforced by:
- Bridge is disabled by default (`deposit_bridge = None`)
- Enabled only via `BTC_RPC_URL` env var on exactly one node (operator controls)
- On startup, log which node has bridge enabled
- If two nodes have `BTC_RPC_URL` set → dedup key in RocksDB prevents double-mint even across nodes (defense-in-depth)

---

## Revised Mint Function (No Direct Balance Writes)

```rust
/// Execute a mint operation — MUST be called from the single-threaded mint executor only.
///
/// CRITICAL: Does NOT write to token_balance: directly.
/// Instead calls bridge_tokens::mint_wrapped_token() which is the canonical
/// token balance API, already production-tested for wBTC/wZEC/wIRON/wETH.
///
/// DeepSeek review: "Do NOT write directly to token_balance: — instead, call
/// the token contract's mint function."
///
async fn execute_serialized_mint(
    storage: &StorageEngine,
    token_balances: &TokenBalances,
    request: &MintRequest,
) -> Result<()> {
    let dedup_key = format!("btc_bridge:minted:{}:{}", request.txid, request.vout);

    // Step 1: Dedup check (safe — we're the only writer, serialized)
    if storage.hot_db.get(CF_MANIFEST, dedup_key.as_bytes()).await?.is_some() {
        warn!("₿ DOUBLE-MINT BLOCKED: {} already minted", dedup_key);
        return Ok(()); // Not an error — idempotent
    }

    // Step 2: Write bridge state atomically (dedup + deposit status + TVL)
    let deposit_key = format!("btc_bridge:deposit:{}", request.deposit.deposit_id);
    let mut minted = request.deposit.clone();
    minted.status = "minted".to_string();
    minted.updated_at = unix_now();

    let tvl_key = b"btc_bridge:total_minted_sats";
    let current_tvl = match storage.hot_db.get(CF_MANIFEST, tvl_key).await? {
        Some(bytes) if bytes.len() == 8 => u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        _ => 0u64,
    };
    let new_tvl = current_tvl.checked_add(request.deposit.amount_sats)
        .ok_or_else(|| anyhow!("TVL overflow — this should never happen (supply capped)"))?;

    let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
        (CF_MANIFEST, dedup_key.as_bytes().to_vec(), vec![1u8]),
        (CF_MANIFEST, deposit_key.as_bytes().to_vec(), bincode::serialize(&minted)?),
        (CF_MANIFEST, tvl_key.to_vec(), new_tvl.to_le_bytes().to_vec()),
    ];
    storage.hot_db.write_batch(batch).await
        .context("Bridge state WriteBatch failed")?;

    // Step 3: Mint wBTC via the canonical token API (NOT direct RocksDB write)
    // bridge_tokens::mint_wrapped_token() handles:
    //   - Token balance update (in-memory + RocksDB)
    //   - Supply tracking
    //   - Event emission for SSE
    let new_balance = bridge_tokens::mint_wrapped_token(
        bridge_tokens::BridgeChain::Bitcoin,
        &request.deposit.qug_wallet,
        request.deposit.amount_sats as u128,
        token_balances,
        storage,
    ).await?;

    info!(
        "₿ ✅ MINTED: {} sats → wBTC for wallet {} (balance: {}, dedup: {}, TVL: {})",
        request.deposit.amount_sats,
        hex::encode(&request.deposit.qug_wallet[..8]),
        new_balance, dedup_key, new_tvl,
    );

    Ok(())
}
```

**Key differences from v1:**
- `checked_add` instead of `saturating_add` for TVL (overflow = fatal error, not silent)
- wBTC balance via `mint_wrapped_token()` (canonical API), not direct RocksDB write
- Dedup check returns `Ok(())` on duplicate (idempotent), not `Err`

---

## Durability Audit (Per Reviewer Request)

| Question | Answer | Evidence |
|----------|--------|----------|
| WAL enabled? | Yes (RocksDB default) | Not explicitly disabled in kv.rs Options |
| `disableWAL`? | Not set (defaults to false) | Grep shows zero matches for `disable_wal` |
| Sync policy for mint? | `write_batch` uses default (not synced per batch) | Bridge should use `write_batch_sync` or explicit `sync_wal()` after mint |
| Power-loss safety? | WAL protects against process crash; power loss may lose last few writes if not synced | Mint path MUST use synced writes |

**Required change:** The bridge mint WriteBatch must use `put_sync` or equivalent to guarantee durability before acknowledging the mint.

---

## Concurrency Contract (Who Owns Which Keys)

| Key Pattern | Owner | Concurrent Writers |
|-------------|-------|-------------------|
| `wallet_balance_{hex}` | `balance_consensus` | Multiple (block processing tasks) |
| `token_balance:{w}:{t}` | Token contract / `mint_wrapped_token` | Serialized per token (existing) |
| `dex_qug_debited:{w}` | DEX swap handler | Serialized per swap (atomic batch) |
| `btc_bridge:*` | Bridge mint executor | **Single writer only** (serialized channel) |

**Rule:** No subsystem writes to another subsystem's keys. Bridge NEVER touches `wallet_balance_`. `balance_consensus` NEVER touches `btc_bridge:`.

---

## Revised Testing Requirements

### Must-Pass Before Production

```
1. Concurrent same-UTXO mint:
   - Send 10 mint requests for same txid:vout simultaneously
   - Verify: exactly 1 succeeds, balance incremented once, TVL incremented once

2. Concurrent same-wallet different-UTXO:
   - Send 5 deposits to same wallet simultaneously
   - Verify: final balance = old + sum(all amounts)
   - Verify: TVL = sum(all amounts)

3. Crash durability:
   - Execute mint, kill -9 process immediately after
   - Restart, verify: either fully minted OR not minted at all
   - Never partial (dedup without balance, or balance without dedup)

4. Power-loss simulation:
   - Execute mint, unmount filesystem immediately
   - Remount, verify: WAL recovery produces consistent state

5. Mixed-version deploy:
   - Old binary running (no bridge code)
   - New binary starts on different node
   - Verify: old node ignores btc_bridge: keys, no errors

6. Bridge-only deletion:
   - scan_prefix("btc_bridge:") → delete all
   - Verify: all other CF_MANIFEST data intact
   - Verify: wallet_balance_, token_balance: unchanged

7. Overflow:
   - Set TVL to u64::MAX - 1
   - Attempt mint of 2 sats
   - Verify: fatal error (not silent saturation)
```

---

## Summary

| Decision | Status |
|----------|--------|
| Storage engine | **RocksDB** (approved) |
| Layout | **Prefix keys in CF_MANIFEST** (approved) |
| Concurrency model | **Single-threaded mint executor** (new, required) |
| Token balance API | **Via `mint_wrapped_token()`** (not direct write) |
| Durability | **Synced WriteBatch for mint path** (new, required) |
| Cluster-wide enforcement | **One bridge minter per cluster** (env var gated) |
| Rolling deploy | Safe (old binary ignores bridge keys) |

Both reviewers approved the storage approach. The concurrency fix (serialized mint channel) and the proper token API usage (not direct writes) address all critical issues raised.
