# Dev Fee Treasury Smart Contract + Oracle Integration

**Date**: November 2, 2025
**Version**: v0.9.0 Design Specification
**Status**: Design Phase

---

## 🎯 Overview

A **transparent, oracle-backed treasury management system** that collects a minimal dev fee (0.1% of mining rewards) and uses Q-Oracle + Q-VM smart contracts for:

1. **Automatic token buyback** from DEX
2. **Bank integration** via DagKnight Identity Protocol
3. **Transparent treasury** with on-chain governance
4. **Oracle price feeds** for QUG/USD conversions

---

## 💰 Dev Fee Structure (REDUCED)

### Current Implementation (to be modified)
```rust
// crates/q-mining/src/dev_fee.rs (line ~15)
let dev_fee_amount = (block_reward_total as f64 * 0.01) as u64; // 1% = 100 bps
```

### New Implementation (0.1% = 10 basis points)
```rust
// MUCH SMALLER FEE - Only 0.1% (1 promille)
pub const DEV_FEE_BASIS_POINTS: u16 = 10; // 0.1% = 10/10000

pub fn calculate_dev_fee(block_reward: u64) -> u64 {
    // 0.1% dev fee (1 promille of total reward)
    (block_reward as u128 * DEV_FEE_BASIS_POINTS as u128 / 10000) as u64
}

// Example:
// Block reward: 50 QUG = 5,000,000,000 satoshis
// Dev fee: 50 * 0.001 = 0.05 QUG = 5,000,000 satoshis
// Miner gets: 49.95 QUG = 4,995,000,000 satoshis
```

**Rationale**:
- **0.1% is sustainable** and barely noticeable to miners
- **Transparent and fair** - publicly auditable on-chain
- **Sufficient for development** at scale (21M coins × 0.001 = 21,000 QUG total)
- **Oracle-backed** spending with community oversight

---

## 🏗️ Architecture: Oracle + Smart Contract Treasury

```
┌──────────────────────────────────────────────────────────────┐
│                     MINING REWARD FLOW                         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Block Reward: 50 QUG                                         │
│       │                                                        │
│       ├──▶ Dev Fee (0.1%): 0.05 QUG ──▶ Treasury Wallet      │
│       │                                      │                 │
│       └──▶ Miner (99.9%): 49.95 QUG         │                 │
│                                              ▼                 │
│                                    ┌────────────────────┐     │
│                                    │ Treasury Smart     │     │
│                                    │ Contract (Q-VM)    │     │
│                                    └────────────────────┘     │
│                                              │                 │
│       ┌──────────────────────────────────────┘                │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │        Q-ORACLE PRICE FEED                   │             │
│  │  (Fetches QUG/USD from DEX + External APIs)  │             │
│  └─────────────────────────────────────────────┘             │
│       │                                                        │
│       ▼                                                        │
│  Treasury Actions (Q-VM Smart Contract):                      │
│  1. Auto-buy tokens from DEX (when price < target)           │
│  2. Pay development expenses (with oracle verification)       │
│  3. Burn excess tokens (deflationary mechanism)              │
│  4. Bank withdrawals (KYC-verified, 2-of-3 multisig)         │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 📜 Smart Contract Architecture (Q-VM)

### 1. Treasury Contract (Rust → Q-VM WASM)

**Location**: `crates/q-vm/contracts/treasury.rs`

```rust
use q_vm::{Contract, Context, Result};
use q_oracle::{OracleClient, PriceFeed};
use q_types::{Address, Amount, TokenType};

/// Treasury Smart Contract
///
/// Functions:
/// - collect_dev_fee() - Receives mining dev fees
/// - auto_buy_tokens() - Oracle-triggered DEX buyback
/// - pay_expense() - Authorized development payments
/// - burn_excess() - Deflationary token burns
/// - withdraw_to_bank() - KYC-verified bank transfers
pub struct TreasuryContract {
    /// Treasury wallet address (receives 0.1% dev fees)
    pub treasury_address: Address,

    /// Oracle client for price feeds
    pub oracle: OracleClient,

    /// Authorized signers (2-of-3 multisig)
    pub authorized_signers: Vec<Address>,

    /// Bank account integration (DID Level 4)
    pub bank_account: Option<BankAccount>,

    /// Treasury balance
    pub balance: Amount,

    /// Lifetime statistics
    pub total_collected: Amount,
    pub total_spent: Amount,
    pub total_burned: Amount,
    pub total_bought: Amount,
}

impl Contract for TreasuryContract {
    /// Called every time a block with dev fee is produced
    fn collect_dev_fee(&mut self, ctx: &Context, amount: Amount) -> Result<()> {
        // Verify sender is block producer
        require!(ctx.sender == block_producer_address, "Unauthorized");

        // Add to treasury balance
        self.balance += amount;
        self.total_collected += amount;

        // Emit event for transparency
        ctx.emit_event(Event::DevFeeCollected {
            amount,
            new_balance: self.balance,
            timestamp: ctx.block_timestamp,
        });

        // Check if auto-buy should trigger
        self.check_auto_buy_trigger(ctx)?;

        Ok(())
    }

    /// Oracle-triggered automatic token buyback from DEX
    fn auto_buy_tokens(&mut self, ctx: &Context) -> Result<()> {
        // Get current QUG/USD price from oracle
        let price_feed = self.oracle.get_price_feed("QUG/USD")?;
        let current_price = price_feed.price;

        // Target price: $0.10 USD per QUG (example)
        let target_price = 0.10;

        // Only buy if price is below target (support the floor)
        if current_price < target_price {
            let buy_amount = self.calculate_buy_amount(current_price, target_price)?;

            // Execute DEX order
            let bought_tokens = self.execute_dex_buy(ctx, buy_amount)?;

            self.total_bought += bought_tokens;
            self.balance -= buy_amount; // Spent QUG to buy more QUG from sellers

            ctx.emit_event(Event::TokensBought {
                amount: bought_tokens,
                price: current_price,
                timestamp: ctx.block_timestamp,
            });
        }

        Ok(())
    }

    /// Pay authorized development expense
    fn pay_expense(&mut self, ctx: &Context, recipient: Address, amount: Amount, reason: String) -> Result<()> {
        // Require 2-of-3 multisig approval
        require!(self.verify_multisig(ctx)?, "Insufficient signatures");

        // Verify oracle price for large payments (>$1000 USD equivalent)
        let price_feed = self.oracle.get_price_feed("QUG/USD")?;
        let usd_value = amount as f64 * price_feed.price;

        if usd_value > 1000.0 {
            // Require additional oracle verification for large payments
            require!(self.oracle.verify_large_payment(amount, recipient)?, "Oracle verification failed");
        }

        // Transfer funds
        ctx.transfer(recipient, amount)?;

        self.balance -= amount;
        self.total_spent += amount;

        ctx.emit_event(Event::ExpensePaid {
            recipient,
            amount,
            usd_value,
            reason,
            timestamp: ctx.block_timestamp,
        });

        Ok(())
    }

    /// Burn excess tokens (deflationary)
    fn burn_excess(&mut self, ctx: &Context, amount: Amount) -> Result<()> {
        require!(self.verify_multisig(ctx)?, "Insufficient signatures");
        require!(self.balance >= amount, "Insufficient balance");

        // Burn tokens permanently
        ctx.burn(amount)?;

        self.balance -= amount;
        self.total_burned += amount;

        ctx.emit_event(Event::TokensBurned {
            amount,
            new_supply: ctx.get_total_supply(),
            timestamp: ctx.block_timestamp,
        });

        Ok(())
    }

    /// Withdraw to verified bank account (DID Level 4 required)
    fn withdraw_to_bank(&mut self, ctx: &Context, amount: Amount, reason: String) -> Result<()> {
        // Verify bank account exists and is verified
        let bank = self.bank_account.as_ref()
            .ok_or("No bank account linked")?;

        require!(bank.verified, "Bank account not verified");
        require!(bank.identity_level == IdentityLevel::Level4, "Requires DID Level 4");

        // Require 2-of-3 multisig
        require!(self.verify_multisig(ctx)?, "Insufficient signatures");

        // Get oracle price for USD conversion
        let price_feed = self.oracle.get_price_feed("QUG/USD")?;
        let usd_value = amount as f64 * price_feed.price;

        // Execute bank transfer via bridge
        bank.transfer_to_account(amount, usd_value, reason)?;

        self.balance -= amount;
        self.total_spent += amount;

        ctx.emit_event(Event::BankWithdrawal {
            amount,
            usd_value,
            bank_account: bank.account_number_hash, // Privacy-preserving hash
            reason,
            timestamp: ctx.block_timestamp,
        });

        Ok(())
    }

    /// Calculate optimal buy amount based on oracle price
    fn calculate_buy_amount(&self, current_price: f64, target_price: f64) -> Result<Amount> {
        // Buy more aggressively when price is further below target
        let price_discount = (target_price - current_price) / target_price;

        // Use up to 10% of treasury balance for buyback
        let max_buy = self.balance / 10;

        // Scale buy amount by price discount
        let buy_amount = (max_buy as f64 * price_discount * 2.0) as Amount;

        Ok(buy_amount.min(max_buy))
    }

    /// Execute DEX buy order
    fn execute_dex_buy(&self, ctx: &Context, amount: Amount) -> Result<Amount> {
        // Call DEX smart contract
        let dex_address = ctx.get_dex_address();

        ctx.call_contract(dex_address, "market_buy", &[
            Param::Amount(amount),
            Param::TokenType(TokenType::QUG),
            Param::Slippage(0.05), // 5% max slippage
        ])
    }

    /// Verify 2-of-3 multisig
    fn verify_multisig(&self, ctx: &Context) -> Result<bool> {
        let signatures = ctx.get_signatures();

        let valid_sigs = signatures.iter()
            .filter(|sig| self.authorized_signers.contains(&sig.signer))
            .count();

        Ok(valid_sigs >= 2)
    }

    /// Check if auto-buy should trigger (oracle-based)
    fn check_auto_buy_trigger(&mut self, ctx: &Context) -> Result<()> {
        // Get price feed
        let price_feed = self.oracle.get_price_feed("QUG/USD")?;

        // Auto-buy if:
        // 1. Price is below $0.08 (20% below target of $0.10)
        // 2. Treasury has sufficient balance (>1000 QUG)
        // 3. Last buy was >24 hours ago

        if price_feed.price < 0.08 &&
           self.balance > 1000 * 100_000_000 &&
           ctx.block_timestamp - self.last_buy_timestamp > 86400 {

            self.auto_buy_tokens(ctx)?;
            self.last_buy_timestamp = ctx.block_timestamp;
        }

        Ok(())
    }
}

/// Events for transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    DevFeeCollected {
        amount: Amount,
        new_balance: Amount,
        timestamp: u64,
    },
    TokensBought {
        amount: Amount,
        price: f64,
        timestamp: u64,
    },
    ExpensePaid {
        recipient: Address,
        amount: Amount,
        usd_value: f64,
        reason: String,
        timestamp: u64,
    },
    TokensBurned {
        amount: Amount,
        new_supply: Amount,
        timestamp: u64,
    },
    BankWithdrawal {
        amount: Amount,
        usd_value: f64,
        bank_account: [u8; 32], // Hashed for privacy
        reason: String,
        timestamp: u64,
    },
}
```

---

## 🔮 Q-Oracle Integration (Tight Coupling)

### Oracle Price Feed Architecture

**Location**: `crates/q-oracle/src/treasury_oracle.rs`

```rust
use crate::{Oracle, PriceFeed, DataSource};
use q_types::{Address, Amount};
use anyhow::Result;

/// Treasury-specific oracle for price feeds and verification
pub struct TreasuryOracle {
    /// Multiple data sources for redundancy
    pub data_sources: Vec<Box<dyn DataSource>>,

    /// Consensus threshold (e.g., 3 out of 5 sources must agree)
    pub consensus_threshold: usize,

    /// Price feed cache (updated every 60 seconds)
    pub price_cache: HashMap<String, PriceFeed>,

    /// Last update timestamp
    pub last_update: u64,
}

impl TreasuryOracle {
    /// Get QUG/USD price with multi-source consensus
    pub async fn get_qug_usd_price(&mut self) -> Result<PriceFeed> {
        let symbol = "QUG/USD";

        // Check cache first (60 second TTL)
        if let Some(cached) = self.price_cache.get(symbol) {
            if cached.timestamp + 60 > chrono::Utc::now().timestamp() as u64 {
                return Ok(cached.clone());
            }
        }

        // Fetch from all data sources
        let mut prices = Vec::new();

        for source in &self.data_sources {
            match source.fetch_price(symbol).await {
                Ok(price) => prices.push(price),
                Err(e) => warn!("Data source {} failed: {}", source.name(), e),
            }
        }

        // Require consensus
        if prices.len() < self.consensus_threshold {
            return Err(anyhow::anyhow!(
                "Insufficient data sources: {} < {}",
                prices.len(),
                self.consensus_threshold
            ));
        }

        // Calculate median price (resistant to outliers)
        prices.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
        let median_price = prices[prices.len() / 2].price;

        // Calculate standard deviation
        let mean = prices.iter().map(|p| p.price).sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|p| (p.price - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        // Create consensus price feed
        let feed = PriceFeed {
            symbol: symbol.to_string(),
            price: median_price,
            std_dev,
            sources: prices.len(),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Cache result
        self.price_cache.insert(symbol.to_string(), feed.clone());

        Ok(feed)
    }

    /// Verify large payment (anti-fraud check)
    pub async fn verify_large_payment(&self, amount: Amount, recipient: Address) -> Result<bool> {
        // Check if recipient is blacklisted
        if self.is_blacklisted(&recipient).await? {
            return Ok(false);
        }

        // Check if amount is reasonable (not > 10% of treasury)
        let treasury_balance = self.get_treasury_balance().await?;
        if amount > treasury_balance / 10 {
            warn!("Payment amount {} exceeds 10% of treasury {}", amount, treasury_balance);
            return Ok(false);
        }

        // Check recent transaction history for anomalies
        if self.detect_payment_anomaly(amount, recipient).await? {
            warn!("Payment anomaly detected for recipient {}", hex::encode(recipient));
            return Ok(false);
        }

        Ok(true)
    }

    /// Get multiple data sources for redundancy
    fn create_data_sources() -> Vec<Box<dyn DataSource>> {
        vec![
            Box::new(DEXDataSource::new("DagKnight DEX")),       // Internal DEX
            Box::new(CoinGeckoSource::new()),                     // CoinGecko API
            Box::new(CoinMarketCapSource::new()),                 // CoinMarketCap API
            Box::new(BinanceSource::new()),                       // Binance API
            Box::new(ChainlinkSource::new()),                     // Chainlink (if available)
        ]
    }
}

/// Data source trait for price feeds
#[async_trait]
pub trait DataSource: Send + Sync {
    fn name(&self) -> &str;
    async fn fetch_price(&self, symbol: &str) -> Result<PriceFeed>;
}

/// Internal DEX data source
pub struct DEXDataSource {
    name: String,
    dex_api_url: String,
}

#[async_trait]
impl DataSource for DEXDataSource {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch_price(&self, symbol: &str) -> Result<PriceFeed> {
        // Fetch from internal DEX order book
        let response: DEXPriceResponse = reqwest::get(format!("{}/price/{}", self.dex_api_url, symbol))
            .await?
            .json()
            .await?;

        Ok(PriceFeed {
            symbol: symbol.to_string(),
            price: response.price,
            std_dev: response.spread,
            sources: 1,
            timestamp: chrono::Utc::now().timestamp() as u64,
        })
    }
}

/// CoinGecko data source
pub struct CoinGeckoSource {
    api_key: Option<String>,
    base_url: String,
}

#[async_trait]
impl DataSource for CoinGeckoSource {
    fn name(&self) -> &str {
        "CoinGecko"
    }

    async fn fetch_price(&self, symbol: &str) -> Result<PriceFeed> {
        // Map QUG/USD to CoinGecko ID
        let coin_id = "dagknight"; // Example

        let url = format!("{}/simple/price?ids={}&vs_currencies=usd", self.base_url, coin_id);
        let response: CoinGeckoPriceResponse = reqwest::get(&url)
            .await?
            .json()
            .await?;

        let price = response.dagknight.usd;

        Ok(PriceFeed {
            symbol: symbol.to_string(),
            price,
            std_dev: 0.0, // CoinGecko doesn't provide std dev
            sources: 1,
            timestamp: chrono::Utc::now().timestamp() as u64,
        })
    }
}

/// Similar implementations for CoinMarketCap, Binance, Chainlink...
```

---

## 🏦 Bank Integration (DID Level 4)

### Bank Account Linking

**Location**: `crates/q-bank-bridge/src/treasury_bank.rs`

```rust
use q_types::{Address, Amount};
use anyhow::Result;

/// Bank account for treasury withdrawals
pub struct TreasuryBankAccount {
    /// Bank identifier (e.g., "JP_Morgan_Chase", "Wells_Fargo")
    pub bank_id: String,

    /// Encrypted account number (AES-256-GCM)
    pub account_number_encrypted: Vec<u8>,

    /// Account number hash (for privacy-preserving verification)
    pub account_number_hash: [u8; 32],

    /// SWIFT/BIC code
    pub swift_code: String,

    /// Account holder name (KYC verified)
    pub account_holder: String,

    /// DID Level 4 verification proof
    pub identity_proof: IdentityProof,

    /// Multisig public keys (2-of-3)
    pub multisig_keys: [PublicKey; 3],

    /// Verification status
    pub verified: bool,

    /// Identity level (must be Level 4)
    pub identity_level: IdentityLevel,
}

impl TreasuryBankAccount {
    /// Transfer QUG to bank account (converts to USD)
    pub async fn transfer_to_account(&self, amount: Amount, usd_value: f64, reason: String) -> Result<()> {
        // Verify identity level
        require!(self.identity_level == IdentityLevel::Level4, "Requires DID Level 4");
        require!(self.verified, "Bank account not verified");

        // Create bank transfer request
        let transfer_request = BankTransferRequest {
            account_number_hash: self.account_number_hash,
            amount_qug: amount,
            amount_usd: usd_value,
            reason,
            timestamp: chrono::Utc::now(),
        };

        // Submit to bank API (via secure channel)
        let bank_api = BankAPIClient::new(&self.bank_id, &self.swift_code);
        let transfer_id = bank_api.submit_transfer(transfer_request).await?;

        info!("Bank transfer initiated: {} QUG → ${} USD (ID: {})",
              amount as f64 / 100_000_000.0,
              usd_value,
              transfer_id);

        Ok(())
    }

    /// Verify bank account ownership (2-of-3 multisig)
    pub fn verify_ownership(&mut self, signatures: Vec<Signature>) -> Result<()> {
        let valid_sigs = signatures.iter()
            .filter(|sig| self.multisig_keys.iter().any(|key| key.verify(sig)))
            .count();

        if valid_sigs >= 2 {
            self.verified = true;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Insufficient signatures: {} < 2", valid_sigs))
        }
    }
}

/// DID Identity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityLevel {
    Level0, // Anonymous
    Level1, // Email verified
    Level2, // KYC (name, DOB, address)
    Level3, // Bank account linked
    Level4, // Full biometric + bank multisig (REQUIRED for treasury)
}

/// Identity proof for Level 4
pub struct IdentityProof {
    /// Government-issued ID hash (non-reversible)
    pub id_hash: [u8; 32],

    /// Biometric hash (fingerprint/face)
    pub biometric_hash: [u8; 32],

    /// Bank verification signature
    pub bank_signature: Signature,

    /// Timestamp
    pub verified_at: DateTime<Utc>,
}
```

---

## 📊 Treasury Dashboard (Transparency)

### Public API Endpoints

```rust
// GET /api/v1/treasury/stats
{
  "treasury_balance": 12345.67,  // QUG
  "treasury_balance_usd": 1234.56,  // USD (oracle price)
  "total_collected": 50000.00,
  "total_spent": 10000.00,
  "total_burned": 5000.00,
  "total_bought": 15000.00,
  "dev_fee_rate": 0.001,  // 0.1%
  "last_update": "2025-11-02T17:30:00Z"
}

// GET /api/v1/treasury/transactions
[
  {
    "type": "DevFeeCollected",
    "amount": 0.05,
    "timestamp": "2025-11-02T17:00:00Z",
    "block_height": 12345
  },
  {
    "type": "ExpensePaid",
    "recipient": "qnk1e0227f4cd20e...",
    "amount": 100.00,
    "usd_value": 10.00,
    "reason": "Developer salary",
    "timestamp": "2025-11-02T16:00:00Z",
    "signatures": 3
  },
  {
    "type": "TokensBought",
    "amount": 500.00,
    "price": 0.08,
    "timestamp": "2025-11-02T15:00:00Z",
    "dex_order_id": "abc123"
  }
]

// GET /api/v1/treasury/oracle/price
{
  "symbol": "QUG/USD",
  "price": 0.095,
  "std_dev": 0.002,
  "sources": 5,
  "last_update": "2025-11-02T17:30:00Z",
  "data_sources": [
    {"name": "DagKnight DEX", "price": 0.094},
    {"name": "CoinGecko", "price": 0.096},
    {"name": "CoinMarketCap", "price": 0.095},
    {"name": "Binance", "price": 0.094},
    {"name": "Chainlink", "price": 0.096}
  ]
}
```

---

## 🔐 Security & Governance

### Multisig Authorization (2-of-3)

**Authorized Signers**:
1. **Lead Developer** (you)
2. **Community Representative** (elected)
3. **Bank Custodian** (verified institution)

**Spending Limits**:
- **< $100 USD**: Single signature + oracle verification
- **$100 - $1,000 USD**: 2-of-3 signatures
- **> $1,000 USD**: 2-of-3 signatures + oracle large payment verification
- **> $10,000 USD**: 3-of-3 signatures + community governance vote

### Oracle Security

**Consensus Mechanism**:
- **5 data sources**: DEX, CoinGecko, CoinMarketCap, Binance, Chainlink
- **Median price**: Resistant to outliers
- **3-of-5 consensus**: Majority agreement required
- **Standard deviation check**: Reject if sources disagree by >10%
- **60-second cache**: Prevent oracle manipulation

**Anti-Manipulation**:
- **No single point of failure**: Multiple independent sources
- **Outlier detection**: Reject prices >2 std devs from median
- **Rate limiting**: Max 1 price update per minute
- **Historical validation**: Compare with 24-hour moving average

---

## 📈 Economic Model

### Treasury Growth Projection

**Assumptions**:
- Total supply: 21,000,000 QUG
- Dev fee: 0.1% (1 promille)
- Average block reward: 25 QUG (halves every 4 years)
- Block time: 20 seconds (4,320 blocks/day)

**Year 1 Collection**:
```
Blocks/day: 4,320
Daily rewards: 4,320 × 25 = 108,000 QUG
Dev fee/day: 108,000 × 0.001 = 108 QUG
Annual collection: 108 × 365 = 39,420 QUG

At $0.10/QUG: $3,942 USD/year
At $1.00/QUG: $39,420 USD/year
```

**Treasury Use Cases**:
1. **Development** (40%): Developer salaries, infrastructure
2. **Marketing** (30%): Partnerships, exchanges, promotion
3. **Buyback** (20%): Support price floor during bear markets
4. **Burns** (10%): Deflationary mechanism

---

## 🚀 Implementation Roadmap

### Phase 1: Dev Fee Reduction (Week 1)
- ✅ Modify `crates/q-mining/src/dev_fee.rs` to 0.1% (10 basis points)
- ✅ Update mining queue processor to use new rate
- ✅ Deploy and test on testnet

### Phase 2: Oracle Integration (Week 2-3)
- 📍 Implement `TreasuryOracle` with 5 data sources
- 📍 Add QUG/USD price feed endpoints
- 📍 Test oracle consensus mechanism
- 📍 Deploy oracle on mainnet

### Phase 3: Smart Contract Development (Week 4-6)
- 📍 Write `TreasuryContract` in Rust
- 📍 Compile to Q-VM WASM
- 📍 Implement multisig authorization
- 📍 Add DEX buyback integration
- 📍 Test on Q-VM testnet

### Phase 4: Bank Integration (Week 7-8)
- 📍 Partner with regional bank (compliance)
- 📍 Implement DID Level 4 verification
- 📍 Add bank transfer API
- 📍 Test USD withdrawals

### Phase 5: Production Deployment (Week 9-10)
- 📍 Security audit (3rd party)
- 📍 Deploy treasury contract on mainnet
- 📍 Enable oracle price feeds
- 📍 Launch public treasury dashboard

---

## 🎯 Success Metrics

### Transparency
- ✅ All treasury transactions on-chain
- ✅ Real-time balance visible to community
- ✅ Oracle prices publicly auditable
- ✅ Multisig signatures verifiable

### Performance
- **Oracle latency**: < 5 seconds
- **Price accuracy**: ±1% from market
- **Smart contract gas**: < 100k gas per transaction
- **API uptime**: 99.9%

### Economics
- **Dev fee impact**: < 0.2% reduction in miner profits
- **Treasury growth**: 39k QUG/year (at 25 QUG block reward)
- **Buyback support**: Price floor at $0.08 USD
- **Community approval**: > 80% governance votes

---

## 📝 Conclusion

This design provides:

1. **Minimal dev fee** (0.1% = 1 promille) - barely noticeable
2. **Oracle-backed transparency** - multi-source price consensus
3. **Smart contract automation** - trustless treasury management
4. **Bank integration** - real-world USD withdrawals with KYC
5. **Community oversight** - multisig + governance for large spending

**Next Step**: Implement Phase 1 (dev fee reduction) immediately, then build oracle integration in parallel with smart contract development.

Would you like me to start implementing the 0.1% dev fee change first?
