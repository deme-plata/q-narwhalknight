# AI Treasury Wallet System - Master Wallet Revenue Collection

**Date**: October 29, 2025
**Status**: Design Phase
**Goal**: Route all AI inference profits to master treasury wallet with node operator revenue sharing planned for future

---

## Current Implementation: Centralized Treasury

### Master Wallet Configuration

**Treasury Wallet Address**: `MASTER_AI_TREASURY_WALLET`
- All AI inference payments flow to this single address
- Revenue accumulates for project funding
- Future: Will be distributed to node operators based on contribution

**Environment Variable**:
```bash
AI_TREASURY_WALLET=0x1234567890abcdef1234567890abcdef12345678  # Master treasury address
AI_TREASURY_ENABLED=true
```

---

## Payment Flow (Current)

```
User Payment (QNK/QUGUSD)
         ↓
  Payment Consensus
         ↓
   Payment Locked
         ↓
   AI Generation
         ↓
  Actual Cost Calculated
         ↓
┌─────────────────────────┐
│  ALL PROFITS → TREASURY │
│                         │
│  Treasury Balance:      │
│  + Actual Cost          │
│                         │
│  User Refund:           │
│  - (Locked - Actual)    │
└─────────────────────────┘
         ↓
  100% Revenue to Master Wallet
```

---

## Future: Node Operator Revenue Sharing

### Planned Distribution Model (Phase 2)

```
AI Inference Revenue
         ↓
    Split Distribution:
         ↓
    ┌─────────────────┐
    │  70% Treasury   │ ──► Master project wallet
    └─────────────────┘
         │
    ┌─────────────────┐
    │  30% Operators  │ ──► Distributed to AI-enabled nodes
    └─────────────────┘
         ↓
  Based on Contribution:
  - Model hosting weight: 40%
  - Inference compute: 40%
  - Network uptime: 20%
```

**Node Operator Eligibility**:
- Must have `distributed_ai_enabled = true`
- Must serve AI inference requests
- Must maintain >95% uptime
- Revenue calculated per epoch (daily/weekly)

---

## Database Schema for Treasury & Future Revenue Sharing

### Treasury Balance Tracking

```sql
-- Master treasury balance (current)
CREATE TABLE ai_treasury (
    wallet_address TEXT PRIMARY KEY DEFAULT 'MASTER_AI_TREASURY_WALLET',
    total_revenue_qnk INTEGER NOT NULL DEFAULT 0,
    total_revenue_qugusd INTEGER NOT NULL DEFAULT 0,
    total_requests_served INTEGER NOT NULL DEFAULT 0,
    total_tokens_generated INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

### Node Operator Earnings (Future)

```sql
-- Track individual node operator contributions (for future distribution)
CREATE TABLE node_operator_earnings (
    node_id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL,
    total_requests_served INTEGER NOT NULL DEFAULT 0,
    total_tokens_generated INTEGER NOT NULL DEFAULT 0,
    total_compute_time_ms INTEGER NOT NULL DEFAULT 0,
    earned_qnk INTEGER NOT NULL DEFAULT 0,  -- Future: accumulated earnings
    earned_qugusd INTEGER NOT NULL DEFAULT 0,
    last_payout_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Revenue distribution epochs (future)
CREATE TABLE revenue_epochs (
    epoch_id INTEGER PRIMARY KEY,
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    total_revenue_qnk INTEGER NOT NULL,
    treasury_share_qnk INTEGER NOT NULL,  -- 70%
    operators_share_qnk INTEGER NOT NULL,  -- 30%
    num_operators INTEGER NOT NULL,
    status TEXT NOT NULL,  -- pending, distributed, finalized
    created_at INTEGER NOT NULL
);
```

---

## Implementation Plan

### Phase 1: Master Treasury (CURRENT - Implement Now)

**Storage Structures**:
```rust
/// Master AI Treasury
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITreasury {
    pub wallet_address: String,
    pub total_revenue_qnk: u64,
    pub total_revenue_qugusd: u64,
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Transaction settlement - ALL profits to treasury
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentSettlement {
    pub request_id: String,
    pub wallet_address: String,  // User wallet
    pub actual_tokens_generated: u32,
    pub actual_cost_qnk: u64,
    pub refund_amount_qnk: u64,
    pub treasury_payment_qnk: u64,  // = actual_cost_qnk (100% to treasury)
    pub treasury_wallet: String,  // MASTER_AI_TREASURY_WALLET
    pub generation_node_id: String,
    pub validator_signatures: Vec<ValidatorSignature>,
    pub timestamp: u64,
}
```

**Storage Methods**:
```rust
impl QStorage {
    /// Get treasury balance
    pub async fn get_treasury_balance(&self) -> Result<AITreasury> {
        let key = b"treasury:master";
        match self.hot_db.get(CF_AI_TREASURY, key).await? {
            Some(data) => {
                let treasury: AITreasury = bincode::deserialize(&data)?;
                Ok(treasury)
            }
            None => {
                // Initialize treasury
                let treasury = AITreasury {
                    wallet_address: env::var("AI_TREASURY_WALLET")
                        .unwrap_or_else(|_| "MASTER_AI_TREASURY_WALLET".to_string()),
                    total_revenue_qnk: 0,
                    total_revenue_qugusd: 0,
                    total_requests_served: 0,
                    total_tokens_generated: 0,
                    created_at: current_timestamp(),
                    updated_at: current_timestamp(),
                };
                self.save_treasury_balance(&treasury).await?;
                Ok(treasury)
            }
        }
    }

    /// Credit treasury with AI payment
    pub async fn credit_treasury(
        &self,
        amount_qnk: u64,
        amount_qugusd: u64,
        tokens_generated: u32,
    ) -> Result<()> {
        let mut treasury = self.get_treasury_balance().await?;

        treasury.total_revenue_qnk += amount_qnk;
        treasury.total_revenue_qugusd += amount_qugusd;
        treasury.total_requests_served += 1;
        treasury.total_tokens_generated += tokens_generated as u64;
        treasury.updated_at = current_timestamp();

        self.save_treasury_balance(&treasury).await?;

        info!("💰 Treasury credited: {} QNK, {} QUGUSD | Total revenue: {} QNK",
            amount_qnk, amount_qugusd, treasury.total_revenue_qnk);

        Ok(())
    }

    async fn save_treasury_balance(&self, treasury: &AITreasury) -> Result<()> {
        let key = b"treasury:master";
        let value = bincode::serialize(treasury)?;
        self.hot_db.put(CF_AI_TREASURY, key, &value).await?;
        Ok(())
    }
}
```

### Phase 2: Node Operator Revenue Sharing (FUTURE)

**Implementation Steps** (when ready):
1. Add `node_operator_earnings` column family
2. Track compute time per node during inference
3. Implement epoch-based revenue distribution
4. Add operator payout mechanism
5. Create dashboard for operator earnings

**Revenue Split Calculation**:
```rust
// Future implementation
async fn distribute_epoch_revenue(epoch_id: u64) -> Result<()> {
    let epoch = get_epoch(epoch_id).await?;
    let total_revenue = epoch.total_revenue_qnk;

    // 70% to treasury
    let treasury_share = (total_revenue * 70) / 100;

    // 30% split among operators
    let operators_share = (total_revenue * 30) / 100;

    let operators = get_active_operators(epoch_id).await?;

    for operator in operators {
        // Calculate operator's contribution weight
        let weight = calculate_operator_weight(&operator);
        let payout = (operators_share * weight) / 100;

        // Credit operator wallet
        credit_operator_earnings(&operator.node_id, payout).await?;
    }

    Ok(())
}

fn calculate_operator_weight(operator: &NodeOperator) -> u64 {
    let total_compute = operator.total_compute_time_ms;
    let total_tokens = operator.total_tokens_generated;
    let uptime_percent = operator.uptime_percentage;

    // Weighted score
    let compute_score = (total_compute * 40) / 100;
    let token_score = (total_tokens * 40) / 100;
    let uptime_score = (uptime_percent * 20) / 100;

    compute_score + token_score + uptime_score
}
```

---

## Data Persistence Requirements

### Critical Persistence Points

1. **Treasury Balance** (ACID properties)
   - Every payment MUST update treasury atomically
   - Use RocksDB write batches for atomicity
   - Persist before responding to user

2. **Transaction History** (Append-only log)
   - All AI transactions logged permanently
   - Include: user, cost, tokens, treasury credit, timestamp
   - Never delete (audit trail)

3. **Payment Locks** (Temporary state)
   - Persist during AI generation
   - Clean up after settlement
   - Prevent double-spending

4. **Consensus Votes** (Ephemeral with replay log)
   - Store votes during consensus window
   - Archive after finalization
   - Keep for dispute resolution

### Atomic Operations

**Settlement Transaction** (All-or-nothing):
```rust
async fn settle_payment_atomic(
    settlement: &PaymentSettlement,
    storage: &QStorage,
) -> Result<()> {
    // Atomic batch: refund user + credit treasury + log transaction
    let mut batch = Vec::new();

    // 1. Refund user
    if settlement.refund_amount_qnk > 0 {
        let refund_key = format!("credits:{}", settlement.wallet_address);
        let mut user_credits = storage.get_wallet_credits(&settlement.wallet_address).await?
            .ok_or_else(|| anyhow!("User credits not found"))?;
        user_credits.balance_qnk += settlement.refund_amount_qnk;
        batch.push((
            CF_AI_CREDITS,
            refund_key.into_bytes(),
            bincode::serialize(&user_credits)?,
        ));
    }

    // 2. Credit treasury (100% of actual cost)
    let treasury_key = b"treasury:master".to_vec();
    let mut treasury = storage.get_treasury_balance().await?;
    treasury.total_revenue_qnk += settlement.actual_cost_qnk;
    treasury.total_requests_served += 1;
    treasury.total_tokens_generated += settlement.actual_tokens_generated as u64;
    batch.push((
        CF_AI_TREASURY,
        treasury_key,
        bincode::serialize(&treasury)?,
    ));

    // 3. Log transaction
    let tx = AITransaction {
        tx_id: settlement.request_id.clone(),
        wallet_address: settlement.wallet_address.clone(),
        chat_id: "...".to_string(),
        input_tokens: 0,
        output_tokens: settlement.actual_tokens_generated,
        cost_usd_cents: 0, // calculated from oracle
        cost_qnk: settlement.actual_cost_qnk,
        payment_token: PaymentToken::QNK,
        oracle_price_usd_cents: 0,
        timestamp: settlement.timestamp,
        status: PaymentStatus::Completed,
    };
    let tx_key = format!("aitx:{}", tx.tx_id);
    batch.push((
        CF_AI_TRANSACTIONS,
        tx_key.into_bytes(),
        bincode::serialize(&tx)?,
    ));

    // 4. Remove payment lock
    let lock_key = format!("lock:{}", settlement.request_id);
    batch.push((
        CF_PAYMENT_LOCKS,
        lock_key.into_bytes(),
        vec![], // Delete operation
    ));

    // Execute atomic batch
    storage.hot_db.write_batch(batch).await?;

    info!("✅ Payment settled atomically: {} QNK to treasury, {} QNK refunded",
        settlement.actual_cost_qnk, settlement.refund_amount_qnk);

    Ok(())
}
```

### Crash Recovery

**On Node Restart**:
```rust
async fn recover_payment_state(storage: &QStorage) -> Result<()> {
    info!("🔄 Recovering payment state from disk...");

    // 1. Check for orphaned locks (AI generation crashed)
    let locks = storage.get_all_payment_locks().await?;
    for lock in locks {
        let age = current_timestamp() - lock.locked_at;
        if age > 3600 {  // 1 hour timeout
            warn!("⚠️ Found stale payment lock: {}, issuing refund", lock.request_id);
            // Refund user
            storage.update_wallet_balance(
                &lock.wallet_address,
                lock.locked_amount_qnk as i64,
                0,
            ).await?;
            storage.remove_payment_lock(&lock.request_id).await?;
        }
    }

    // 2. Verify treasury balance integrity
    let treasury = storage.get_treasury_balance().await?;
    let total_tx_revenue: u64 = storage.sum_all_transaction_revenue().await?;

    if treasury.total_revenue_qnk != total_tx_revenue {
        error!("❌ Treasury balance mismatch! Recorded: {}, Actual: {}",
            treasury.total_revenue_qnk, total_tx_revenue);
        // Trigger manual reconciliation
    }

    info!("✅ Payment state recovered successfully");
    Ok(())
}
```

---

## API Response Format

**After AI Generation**:
```json
{
  "success": true,
  "data": {
    "message": "AI response text...",
    "tokens_generated": 150,
    "cost_qnk": 15000,
    "locked_qnk": 20000,
    "refund_qnk": 5000,
    "treasury_credited_qnk": 15000,
    "treasury_wallet": "MASTER_AI_TREASURY_WALLET",
    "user_balance_remaining_qnk": 985000,
    "timestamp": 1730239200
  }
}
```

**Treasury Stats Endpoint** (Admin):
```
GET /api/treasury/stats
{
  "success": true,
  "data": {
    "wallet_address": "MASTER_AI_TREASURY_WALLET",
    "total_revenue_qnk": 1500000000,
    "total_revenue_usd": 7500.00,
    "total_requests_served": 10000,
    "total_tokens_generated": 5000000,
    "average_cost_per_request_qnk": 150000,
    "created_at": 1730000000,
    "updated_at": 1730239200
  }
}
```

---

## Security Considerations

1. **Atomic Settlements**: Use RocksDB write batches to ensure refunds and treasury credits happen together
2. **Double-Spend Prevention**: Payment locks prevent concurrent requests from same wallet
3. **Audit Trail**: All transactions logged permanently for accountability
4. **Crash Recovery**: Automatic refund of stale locks on restart
5. **Treasury Protection**: Treasury balance verified against transaction sum on startup

---

## Migration Path

**Current → Future**:
1. **Now**: Implement master treasury (100% revenue collection)
2. **Phase 2**: Add node operator tracking (silent background collection)
3. **Phase 3**: Enable operator revenue sharing (configurable split %)
4. **Phase 4**: Operator dashboard & automated payouts

---

*All AI profits currently flow to master treasury. Node operator revenue sharing will be enabled in a future update.*
