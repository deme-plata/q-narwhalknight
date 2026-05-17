# QNK-INDEX: Decentralized Top Token Index Fund

## Technical Review & Implementation Specification

**Version**: 1.0.0
**Date**: 2025-12-25
**Status**: Design Phase

---

## 1. Executive Summary

QNK-INDEX is a smart contract-based index token representing a weighted basket of the top tokens on Q-NarwhalKnight's decentralized exchange (DEX). Similar to how the S&P 500 tracks the top 500 US companies, QNK-INDEX tracks and automatically rebalances holdings of the top tokens by market capitalization.

### Key Features

- **Automated Rebalancing**: Weekly rebalancing based on market cap
- **Transparent Holdings**: On-chain verification of underlying assets
- **Decentralized Governance**: Token holders vote on index methodology
- **Gas-Efficient**: Batch operations for cost-effective management
- **Post-Quantum Secure**: All signatures use Dilithium5

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QNK-INDEX Smart Contract                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Oracle    │  │  Rebalancer │  │   Vault     │             │
│  │   Module    │  │   Module    │  │   Module    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Index State Management                      │   │
│  │  - Component tokens & weights                           │   │
│  │  - NAV calculation                                      │   │
│  │  - Share tracking                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Q-NarwhalKnight DEX                          │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Token A │  │  Token B │  │  Token C │  │  Token D │  ...  │
│  │  (25%)   │  │  (20%)   │  │  (15%)   │  │  (10%)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Smart Contract Specification

### 3.1 Data Structures

```rust
/// Index component representing a single token in the basket
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexComponent {
    /// Token contract address (32 bytes)
    pub token_address: [u8; 32],

    /// Token symbol (e.g., "WBTC", "WETH")
    pub symbol: String,

    /// Target weight in basis points (e.g., 2500 = 25%)
    pub target_weight_bps: u16,

    /// Current actual weight based on holdings
    pub actual_weight_bps: u16,

    /// Amount of this token held by the index
    pub holdings: u64,

    /// Latest price in QUG (8 decimals)
    pub price_qug: u64,

    /// Market cap rank (1 = highest)
    pub rank: u8,
}

/// Main index fund state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QnkIndex {
    /// Index identifier
    pub index_id: [u8; 32],

    /// Human-readable name
    pub name: String,  // "QNK Top 10 Index"

    /// Symbol for the index token
    pub symbol: String,  // "QNK10"

    /// Total supply of index tokens (8 decimals)
    pub total_supply: u64,

    /// Components in the index (max 50)
    pub components: Vec<IndexComponent>,

    /// Net Asset Value per share in QUG (8 decimals)
    pub nav_per_share: u64,

    /// Last rebalance block height
    pub last_rebalance_height: u64,

    /// Rebalance interval in blocks (~weekly)
    pub rebalance_interval: u64,

    /// Management fee in basis points (e.g., 50 = 0.5%)
    pub management_fee_bps: u16,

    /// Minimum market cap for inclusion (in QUG)
    pub min_market_cap: u64,

    /// Creator/manager address
    pub manager: [u8; 32],

    /// Governance token holders can vote on methodology
    pub governance_enabled: bool,
}

/// Index share holdings per wallet
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexShareHolder {
    pub wallet: [u8; 32],
    pub shares: u64,
    pub entry_nav: u64,  // NAV when shares were acquired
    pub entry_block: u64,
}
```

### 3.2 Core Operations

```rust
/// Create a new index fund
pub async fn create_index(
    name: String,
    symbol: String,
    initial_components: Vec<IndexComponent>,
    management_fee_bps: u16,
    min_market_cap: u64,
) -> Result<[u8; 32], IndexError>;

/// Mint index shares by depositing QUG
/// The contract automatically buys underlying tokens
pub async fn mint_shares(
    index_id: [u8; 32],
    qug_amount: u64,
    min_shares_out: u64,  // Slippage protection
) -> Result<u64, IndexError>;

/// Redeem index shares for underlying tokens or QUG
pub async fn redeem_shares(
    index_id: [u8; 32],
    shares_amount: u64,
    redeem_as_qug: bool,  // true = sell tokens, return QUG
    min_qug_out: u64,     // Slippage protection
) -> Result<u64, IndexError>;

/// Trigger rebalancing (called by keeper or anyone after interval)
pub async fn rebalance(
    index_id: [u8; 32],
) -> Result<RebalanceResult, IndexError>;

/// Update component weights based on market cap
pub async fn update_weights(
    index_id: [u8; 32],
) -> Result<(), IndexError>;
```

---

## 4. Weighting Methodology

### 4.1 Market Cap Weighted

Default methodology mirrors traditional index funds:

```rust
/// Calculate target weights based on market capitalization
pub fn calculate_market_cap_weights(
    components: &[IndexComponent],
) -> Vec<u16> {
    let total_market_cap: u128 = components.iter()
        .map(|c| c.price_qug as u128 * c.circulating_supply as u128)
        .sum();

    components.iter().map(|c| {
        let component_cap = c.price_qug as u128 * c.circulating_supply as u128;
        // Weight in basis points (10000 = 100%)
        ((component_cap * 10000) / total_market_cap) as u16
    }).collect()
}
```

### 4.2 Weight Caps

To prevent concentration risk:

```rust
/// Maximum weight for any single component
const MAX_COMPONENT_WEIGHT_BPS: u16 = 2500;  // 25%

/// Minimum weight to remain in index
const MIN_COMPONENT_WEIGHT_BPS: u16 = 100;   // 1%

/// Apply weight caps and redistribute excess
pub fn apply_weight_caps(weights: &mut Vec<u16>) {
    let mut excess: u16 = 0;
    let mut capped_count: usize = 0;

    // First pass: cap weights and accumulate excess
    for weight in weights.iter_mut() {
        if *weight > MAX_COMPONENT_WEIGHT_BPS {
            excess += *weight - MAX_COMPONENT_WEIGHT_BPS;
            *weight = MAX_COMPONENT_WEIGHT_BPS;
            capped_count += 1;
        }
    }

    // Second pass: redistribute excess to uncapped components
    if excess > 0 && capped_count < weights.len() {
        let redistribution = excess / (weights.len() - capped_count) as u16;
        for weight in weights.iter_mut() {
            if *weight < MAX_COMPONENT_WEIGHT_BPS {
                *weight += redistribution;
            }
        }
    }
}
```

---

## 5. Rebalancing Mechanism

### 5.1 Trigger Conditions

```rust
pub struct RebalanceTrigger {
    /// Minimum blocks between rebalances
    pub min_interval: u64,        // ~10,080 blocks = 1 week

    /// Maximum drift before forced rebalance
    pub max_drift_bps: u16,       // 500 = 5% drift triggers rebalance

    /// Market cap change threshold for component swap
    pub rank_change_threshold: u8, // Component drops 5+ ranks = swap
}

/// Check if rebalancing is needed
pub fn should_rebalance(index: &QnkIndex, current_height: u64) -> bool {
    // Time-based trigger
    if current_height >= index.last_rebalance_height + index.rebalance_interval {
        return true;
    }

    // Drift-based trigger
    for component in &index.components {
        let drift = (component.actual_weight_bps as i32 -
                    component.target_weight_bps as i32).abs() as u16;
        if drift > MAX_DRIFT_BPS {
            return true;
        }
    }

    false
}
```

### 5.2 Rebalancing Process

```rust
pub async fn execute_rebalance(
    index: &mut QnkIndex,
    dex: &DexState,
) -> Result<RebalanceResult, IndexError> {
    let mut trades: Vec<Trade> = Vec::new();

    // 1. Fetch current prices from oracle
    let prices = oracle::get_batch_prices(&index.components).await?;

    // 2. Calculate current NAV
    let current_nav = calculate_nav(index, &prices);

    // 3. Determine target holdings for each component
    for component in &mut index.components {
        let target_value = (current_nav as u128 *
                          component.target_weight_bps as u128 / 10000) as u64;
        let target_holdings = target_value / component.price_qug;

        let diff = target_holdings as i64 - component.holdings as i64;

        if diff > 0 {
            // Need to buy
            trades.push(Trade::Buy {
                token: component.token_address,
                amount: diff as u64,
                max_price: component.price_qug * 102 / 100, // 2% slippage
            });
        } else if diff < 0 {
            // Need to sell
            trades.push(Trade::Sell {
                token: component.token_address,
                amount: (-diff) as u64,
                min_price: component.price_qug * 98 / 100, // 2% slippage
            });
        }
    }

    // 4. Execute trades via DEX (batched for efficiency)
    let execution_result = dex.execute_batch_trades(trades).await?;

    // 5. Update holdings
    for (component, fill) in index.components.iter_mut().zip(execution_result.fills) {
        component.holdings = fill.new_balance;
        component.actual_weight_bps = calculate_component_weight(component, current_nav);
    }

    // 6. Update metadata
    index.last_rebalance_height = current_height();
    index.nav_per_share = current_nav / index.total_supply;

    Ok(RebalanceResult {
        trades_executed: execution_result.fills.len(),
        total_volume: execution_result.total_volume,
        new_nav: index.nav_per_share,
        gas_used: execution_result.gas_used,
    })
}
```

---

## 6. NAV Calculation

```rust
/// Calculate Net Asset Value of the entire index fund
pub fn calculate_nav(index: &QnkIndex, prices: &HashMap<[u8; 32], u64>) -> u64 {
    let mut total_value: u128 = 0;

    for component in &index.components {
        let price = prices.get(&component.token_address)
            .copied()
            .unwrap_or(component.price_qug);

        total_value += component.holdings as u128 * price as u128;
    }

    // Subtract accrued management fees
    let blocks_since_rebalance = current_height() - index.last_rebalance_height;
    let annualized_blocks = 365 * 24 * 60 * 60 / 6;  // ~5.25M blocks/year
    let fee_fraction = index.management_fee_bps as u128 *
                       blocks_since_rebalance as u128 /
                       (annualized_blocks * 10000) as u128;

    let fees_accrued = total_value * fee_fraction / 10000;

    (total_value - fees_accrued) as u64
}

/// Calculate NAV per share
pub fn nav_per_share(index: &QnkIndex, prices: &HashMap<[u8; 32], u64>) -> u64 {
    if index.total_supply == 0 {
        return 100_000_000; // 1.0 QUG initial price
    }

    let total_nav = calculate_nav(index, prices);
    total_nav / index.total_supply
}
```

---

## 7. Index Composition Rules

### 7.1 Inclusion Criteria

1. **Minimum Market Cap**: Token must have market cap > 1,000,000 QUG
2. **Minimum Liquidity**: Daily volume > 10,000 QUG (7-day average)
3. **Age Requirement**: Token must be active for > 30 days
4. **Security Audit**: Token contract must pass automated vulnerability scan
5. **No Concentration**: Single holder cannot own > 50% of supply

### 7.2 Exclusion Criteria

1. **Stablecoins**: QUGUSD and other pegged tokens excluded
2. **Wrapped Native**: WQUG excluded (circular reference)
3. **Governance Tokens**: Pure governance tokens excluded
4. **Low Activity**: < 100 unique holders
5. **Security Issues**: Tokens with known vulnerabilities

### 7.3 Review Schedule

```rust
/// Component review happens every rebalance
pub struct ComponentReview {
    /// Current components being evaluated
    pub current: Vec<IndexComponent>,

    /// Candidates for inclusion
    pub candidates: Vec<TokenCandidate>,

    /// Components marked for removal
    pub removals: Vec<[u8; 32]>,
}

pub async fn review_components(index: &QnkIndex) -> ComponentReview {
    // Fetch all tokens meeting minimum criteria
    let eligible_tokens = dex::get_eligible_tokens(
        index.min_market_cap,
        MIN_DAILY_VOLUME,
        MIN_TOKEN_AGE,
    ).await;

    // Rank by market cap
    let mut ranked: Vec<_> = eligible_tokens.into_iter()
        .filter(|t| passes_exclusion_criteria(t))
        .collect();
    ranked.sort_by(|a, b| b.market_cap.cmp(&a.market_cap));

    // Top N become the new index
    let top_n = ranked.into_iter().take(index.max_components).collect();

    ComponentReview {
        current: index.components.clone(),
        candidates: top_n,
        removals: find_removals(&index.components, &top_n),
    }
}
```

---

## 8. Fee Structure

| Fee Type | Amount | Recipient |
|----------|--------|-----------|
| Management Fee | 0.50% annually | Index Manager |
| Mint Fee | 0.10% | Protocol Treasury |
| Redeem Fee | 0.10% | Protocol Treasury |
| Rebalance Gas | Variable | Keeper |

### Fee Distribution

```rust
pub fn distribute_fees(index: &QnkIndex, fees_collected: u64) {
    // 80% to manager
    let manager_share = fees_collected * 80 / 100;

    // 20% to protocol treasury
    let protocol_share = fees_collected - manager_share;

    transfer(index.manager, manager_share);
    transfer(PROTOCOL_TREASURY, protocol_share);
}
```

---

## 9. Security Considerations

### 9.1 Oracle Manipulation Protection

```rust
/// Use time-weighted average price (TWAP) for rebalancing
pub fn get_twap_price(
    token: [u8; 32],
    window_blocks: u64,  // e.g., 100 blocks = ~10 minutes
) -> u64 {
    let observations = oracle::get_price_observations(token, window_blocks);

    let sum: u128 = observations.iter()
        .map(|o| o.price as u128 * o.weight as u128)
        .sum();
    let total_weight: u128 = observations.iter()
        .map(|o| o.weight as u128)
        .sum();

    (sum / total_weight) as u64
}
```

### 9.2 Flash Loan Protection

```rust
/// Prevent flash loan attacks on mint/redeem
pub fn validate_no_flashloan(caller: [u8; 32], block_height: u64) -> bool {
    let last_action = get_last_action(caller);

    // Must wait at least 1 block between actions
    block_height > last_action.block + 1
}
```

### 9.3 Slippage Protection

```rust
/// All trades have maximum slippage limits
const MAX_SINGLE_TRADE_SLIPPAGE_BPS: u16 = 200;  // 2%
const MAX_REBALANCE_SLIPPAGE_BPS: u16 = 500;     // 5% total
```

---

## 10. Governance

Token holders can vote on:

1. **Component Selection**: Add/remove specific tokens
2. **Weight Methodology**: Market cap vs. equal weight vs. custom
3. **Fee Changes**: Adjust management fees (capped at 2%)
4. **Rebalance Frequency**: Weekly, bi-weekly, monthly
5. **Risk Parameters**: Max single-component weight

```rust
pub struct GovernanceProposal {
    pub proposal_id: u64,
    pub proposal_type: ProposalType,
    pub description: String,
    pub votes_for: u64,
    pub votes_against: u64,
    pub voting_ends_block: u64,
    pub executed: bool,
}

pub enum ProposalType {
    AddComponent([u8; 32]),
    RemoveComponent([u8; 32]),
    ChangeMethodology(WeightMethodology),
    ChangeFee(u16),
    ChangeRebalanceInterval(u64),
}
```

---

## 11. Example Index Configurations

### QNK Top 10 (QNK10)
- Components: Top 10 tokens by market cap
- Weighting: Market cap weighted, 25% max
- Rebalance: Weekly
- Fee: 0.50% annually

### QNK DeFi Index (QNKDEFI)
- Components: Top 10 DeFi tokens
- Weighting: Equal weight (10% each)
- Rebalance: Bi-weekly
- Fee: 0.75% annually

### QNK Stablecoin Yield (QNKSTY)
- Components: Top 5 yield-bearing stablecoins
- Weighting: Risk-adjusted (based on protocol TVL)
- Rebalance: Monthly
- Fee: 0.25% annually

---

## 12. Implementation Roadmap

### Phase 1: Core Contract (Week 1-2)
- [ ] Index state management
- [ ] Mint/redeem functions
- [ ] NAV calculation

### Phase 2: Rebalancing (Week 3-4)
- [ ] Oracle integration
- [ ] Weight calculation
- [ ] Trade execution

### Phase 3: Governance (Week 5-6)
- [ ] Proposal system
- [ ] Voting mechanism
- [ ] Timelock execution

### Phase 4: Testing & Audit (Week 7-8)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Security audit

---

## 13. Conclusion

QNK-INDEX provides a decentralized, transparent, and automated way to gain diversified exposure to the Q-NarwhalKnight ecosystem. By leveraging smart contracts, TWAP oracles, and on-chain governance, it offers the benefits of traditional index funds while maintaining the trustless properties of decentralized finance.

The post-quantum security of the underlying blockchain ensures these index tokens remain secure even as quantum computing advances.

---

**Document Hash**: `sha3-256:` (computed at deployment)
**Author**: Q-NarwhalKnight Development Team
**License**: MIT
