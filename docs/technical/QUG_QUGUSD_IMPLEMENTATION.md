# QUG/QUGUSD Dual-Token Implementation Plan

**Date**: October 16, 2025
**Status**: 🚧 Phase 1 Implementation Starting
**Goal**: Implement dual-token economics with QUG mining and QUGUSD stablecoin

---

## System Overview

### Token Design

**Quillon (QUG)** - Deflationary Mining Token
- Fixed supply: 21,000,000 QUG
- Base unit: 1 QUG = 100,000,000 base units (8 decimals, like Bitcoin satoshis)
- Mining rewards: Bitcoin-style halving schedule
- Initial reward: 500 QUG/block
- Halving interval: Every 210,000 blocks (~4 years)
- Use case: Store of value, collateral, governance

**Quillon USD (QUGUSD)** - Algorithmic Stablecoin
- Pegged to: $1.00 USD
- Backing: Over-collateralized by QUG
- Minting ratio: 150% collateral (e.g., $150 QUG → 100 QUGUSD)
- Base unit: 1 QUGUSD = 100,000,000 base units (8 decimals)
- Use case: Transactions, smart contracts, stable medium of exchange

---

## Architecture

### Phase 1: Core Token Infrastructure (Current)

#### 1.1 Token Registry System
**File**: `crates/q-types/src/tokens.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    QUG,        // Native mining token
    QUGUSD,     // USD-pegged stablecoin
}

pub struct TokenInfo {
    pub token_type: TokenType,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub max_supply: Option<u64>,  // None for QUGUSD (unlimited if backed)
}

pub const QUG_INFO: TokenInfo = TokenInfo {
    token_type: TokenType::QUG,
    name: "Quillon".to_string(),
    symbol: "QUG".to_string(),
    decimals: 8,
    max_supply: Some(2_100_000_000_000_000), // 21M * 10^8
};

pub const QUGUSD_INFO: TokenInfo = TokenInfo {
    token_type: TokenType::QUGUSD,
    name: "Quillon USD".to_string(),
    symbol: "QUGUSD".to_string(),
    decimals: 8,
    max_supply: None, // Unlimited if properly collateralized
};
```

#### 1.2 Multi-Token Wallet Support
**File**: `crates/q-wallet/src/multi_token_wallet.rs`

```rust
pub struct TokenBalance {
    pub qug_balance: u64,      // In base units
    pub qugusd_balance: u64,   // In base units
}

impl HybridWallet {
    /// Get balance for specific token type
    pub fn get_token_balance(&self, token: TokenType) -> u64 {
        match token {
            TokenType::QUG => self.qug_balance,
            TokenType::QUGUSD => self.qugusd_balance,
        }
    }

    /// Transfer tokens (multi-token aware)
    pub fn transfer_token(
        &mut self,
        token: TokenType,
        to: &str,
        amount: u64,
    ) -> Result<Transaction> {
        // Multi-token transaction logic
    }
}
```

#### 1.3 QUGUSD Collateral Vault
**File**: `crates/q-vm/src/contracts/collateral_vault.rs`

```rust
pub struct CollateralVault {
    /// User address -> Locked QUG amount
    pub locked_qug: HashMap<String, u64>,

    /// User address -> Minted QUGUSD amount
    pub minted_qugusd: HashMap<String, u64>,

    /// Current QUG price in USD (from oracle)
    pub qug_price_usd: f64,

    /// Minimum collateralization ratio (150%)
    pub min_collateral_ratio: f64,
}

impl CollateralVault {
    /// Mint QUGUSD by locking QUG as collateral
    pub fn mint_qugusd(
        &mut self,
        user: String,
        qug_amount: u64,
        qug_price: f64,
    ) -> Result<u64> {
        let qug_value_usd = (qug_amount as f64 / 1e8) * qug_price;
        let max_qugusd = (qug_value_usd / 1.5) * 1e8; // 150% collateral

        self.locked_qug.entry(user.clone())
            .and_modify(|e| *e += qug_amount)
            .or_insert(qug_amount);

        let qugusd_minted = max_qugusd as u64;
        self.minted_qugusd.entry(user.clone())
            .and_modify(|e| *e += qugusd_minted)
            .or_insert(qugusd_minted);

        Ok(qugusd_minted)
    }

    /// Redeem QUG by burning QUGUSD
    pub fn redeem_qug(
        &mut self,
        user: String,
        qugusd_amount: u64,
        qug_price: f64,
    ) -> Result<u64> {
        let qugusd_value = (qugusd_amount as f64 / 1e8);
        let qug_to_unlock = (qugusd_value / qug_price) * 1e8;

        // Burn QUGUSD, unlock QUG
        // Check collateral ratio remains healthy

        Ok(qug_to_unlock as u64)
    }

    /// Liquidate undercollateralized positions
    pub fn liquidate(
        &mut self,
        user: String,
        qug_price: f64,
    ) -> Result<LiquidationResult> {
        // If collateral ratio < 110%, allow liquidation
        // Liquidator gets 5% bonus
    }
}
```

#### 1.4 Mining Reward Distribution
**File**: `crates/q-api-server/src/handlers.rs` (update existing)

```rust
pub async fn submit_mining_solution(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MiningSolutionRequest>,
) -> Result<Json<ApiResponse<MiningSolutionResponse>>, StatusCode> {
    // ... existing VDF verification ...

    // Calculate QUG mining reward (halving schedule)
    let block_reward_qug = calculate_block_reward(block_height);

    // Award QUG to miner
    storage.update_balance(
        &miner_address,
        TokenType::QUG,
        block_reward_qug as i64,
    ).await?;

    // Emit mining reward event
    sse_manager.emit_event(SSEEvent::MiningReward {
        miner_address: miner_address.clone(),
        reward_qug: block_reward_qug as f64 / 1e8,
        block_height,
        token: TokenType::QUG,
    });

    Ok(Json(ApiResponse::success(MiningSolutionResponse {
        accepted: true,
        reward_qug: block_reward_qug as f64 / 1e8,
        reward_qugusd: 0.0, // No QUGUSD mining rewards
        block_height,
    })))
}
```

#### 1.5 Transaction Fee Distribution
**File**: `crates/q-api-server/src/fee_distribution.rs`

```rust
pub struct FeeDistribution {
    /// Bank master account address
    bank_master_account: String,
}

impl FeeDistribution {
    /// Process transaction fee
    pub async fn collect_fee(
        &self,
        storage: &Arc<RwLock<Storage>>,
        fee_amount: u64,
        fee_token: TokenType,
    ) -> Result<()> {
        match fee_token {
            TokenType::QUGUSD => {
                // Collect QUGUSD fees
                let fee_qugusd = fee_amount;

                // Split fee distribution:
                // - 40% to Bank master account (operations)
                let bank_share = (fee_qugusd as f64 * 0.40) as u64;

                // - 30% used to buyback and burn QUG (deflationary)
                let buyback_share = (fee_qugusd as f64 * 0.30) as u64;

                // - 30% distributed to active miners (incentive)
                let miner_share = (fee_qugusd as f64 * 0.30) as u64;

                // Transfer to Bank
                storage.write().await.update_balance(
                    &self.bank_master_account,
                    TokenType::QUGUSD,
                    bank_share as i64,
                ).await?;

                // Queue QUG buyback
                self.queue_qug_buyback(buyback_share).await?;

                // Distribute to miners
                self.distribute_to_miners(miner_share).await?;
            },
            TokenType::QUG => {
                // All QUG fees go to Bank
                storage.write().await.update_balance(
                    &self.bank_master_account,
                    TokenType::QUG,
                    fee_amount as i64,
                ).await?;
            }
        }

        Ok(())
    }

    /// Buyback QUG from DEX using QUGUSD and burn it
    async fn queue_qug_buyback(&self, qugusd_amount: u64) -> Result<()> {
        // 1. Use QUGUSD to market buy QUG from DEX
        // 2. Burn purchased QUG (send to 0x000... address)
        // 3. Emit burn event for transparency
        Ok(())
    }

    /// Distribute QUGUSD to recent miners
    async fn distribute_to_miners(&self, qugusd_amount: u64) -> Result<()> {
        // Distribute proportionally to last 100 blocks of miners
        Ok(())
    }
}
```

---

### Phase 2: API Endpoints

#### 2.1 Token Balance API
**Endpoint**: `GET /api/v1/wallet/{address}/tokens`

```json
{
  "success": true,
  "data": {
    "address": "qnk...",
    "tokens": {
      "QUG": {
        "balance": "1234.56789012",
        "balance_base_units": 123456789012,
        "usd_value": 12345.67
      },
      "QUGUSD": {
        "balance": "5000.00000000",
        "balance_base_units": 500000000000,
        "usd_value": 5000.00
      }
    },
    "total_usd_value": 17345.67
  }
}
```

#### 2.2 Mint QUGUSD API
**Endpoint**: `POST /api/v1/stablecoin/mint`

```json
{
  "qug_amount": "1000.00000000",
  "slippage_tolerance": 0.01
}

// Response:
{
  "success": true,
  "data": {
    "qug_locked": "1000.00000000",
    "qugusd_minted": "666.66666666",
    "collateral_ratio": 1.50,
    "liquidation_price": 6.67
  }
}
```

#### 2.3 Redeem QUG API
**Endpoint**: `POST /api/v1/stablecoin/redeem`

```json
{
  "qugusd_amount": "100.00000000"
}

// Response:
{
  "success": true,
  "data": {
    "qugusd_burned": "100.00000000",
    "qug_unlocked": "10.00000000",
    "remaining_collateral_ratio": 1.65
  }
}
```

#### 2.4 Fee Statistics API
**Endpoint**: `GET /api/v1/stats/fees`

```json
{
  "success": true,
  "data": {
    "last_24h": {
      "total_fees_qugusd": "1234.56",
      "bank_share": "493.82",
      "qug_buyback_amount": "370.37",
      "miner_distribution": "370.37",
      "qug_burned": "37.04"
    },
    "all_time": {
      "total_fees_collected": "1234567.89",
      "total_qug_burned": "12345.67"
    }
  }
}
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure ✅ (In Progress)
- [ ] Create `TokenType` enum and `TokenInfo` structs
- [ ] Update `Storage` to support multi-token balances
- [ ] Implement `CollateralVault` smart contract
- [ ] Add fee distribution logic with Bank master account
- [ ] Update mining rewards to pay in QUG only

### Phase 2: API Layer
- [ ] Multi-token balance endpoints
- [ ] QUGUSD mint/redeem endpoints
- [ ] Collateral health monitoring
- [ ] Fee statistics endpoints
- [ ] Bank account management APIs

### Phase 3: Frontend Integration
- [ ] Update wallet UI to show QUG and QUGUSD separately
- [ ] Add "Mint QUGUSD" interface
- [ ] Add "Redeem QUG" interface
- [ ] Show collateral ratio and health
- [ ] Display fee statistics

### Phase 4: Oracle & Price Feeds
- [ ] Implement QUG/USD price oracle
- [ ] Add Chainlink or decentralized oracle integration
- [ ] Price feed failsafes and manipulation resistance

### Phase 5: DEX Integration
- [ ] QUG/QUGUSD trading pair
- [ ] Automated buyback mechanism
- [ ] Burn transaction verification

---

## Economic Parameters

### Collateralization Ratios
- **Minimum**: 150% (user can mint up to 66.67% of collateral value)
- **Warning**: 120% (user gets warning to add collateral)
- **Liquidation**: 110% (position can be liquidated)
- **Liquidation Bonus**: 5% (incentive for liquidators)

### Fee Distribution
- **Bank Master Account**: 40%
- **QUG Buyback & Burn**: 30%
- **Miner Rewards**: 30%

### Supply Caps
- **QUG**: 21,000,000 (hard cap, deflationary via burning)
- **QUGUSD**: Unlimited (but fully collateralized at 150%+)

---

## Bank Master Account

**Purpose**: Central operations account for network sustainability

**Address**: `qnkBANK_MASTER_ACCOUNT_RESERVED_ADDRESS_00000000000000000`

**Receives**:
- 40% of all QUGUSD transaction fees
- 100% of QUG transaction fees
- Revenue from liquidation penalties

**Uses**:
- Development funding
- Marketing and adoption
- Emergency liquidity provision
- Network operations

---

## Security Considerations

1. **Collateral Safety**: Over-collateralization prevents bank runs
2. **Oracle Manipulation**: Use decentralized price feeds with median
3. **Liquidation Cascades**: Circuit breakers at 20% price drops
4. **Smart Contract Audits**: Full audit before mainnet launch

---

## Next Steps

1. Implement `TokenType` and multi-token balance system
2. Create `CollateralVault` smart contract
3. Update APIs for QUG/QUGUSD support
4. Test minting/redemption logic
5. Frontend integration

**Target**: Phase 1 complete by November 2025
