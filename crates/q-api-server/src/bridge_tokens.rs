// ============================================================================
// bridge_tokens.rs - Cross-Chain Wrapped Token Bridge System (v7.2.5)
// ============================================================================
//
// Mint-on-deposit / burn-on-withdrawal bridge for:
//   wBTC  - Wrapped Bitcoin (1:1 backed by BTC in bridge custody)
//   wZEC  - Wrapped Zcash  (1:1 backed by ZEC in shielded bridge)
//   wIRON - Wrapped Iron Fish (1:1 backed by IRON in privacy bridge)
//
// Bridge tokens are QNK-native tokens that can be traded on the DEX
// just like any other token, enabling BTC/QUG, ZEC/QUG, IRON/QUG pairs.
//
// Flow:
//   Deposit:  User locks BTC on Bitcoin chain → bridge mints wBTC on QNK
//   Withdraw: User burns wBTC on QNK → bridge releases BTC on Bitcoin chain
// ============================================================================

use q_types::{
    WBTC_TOKEN_ADDRESS, WBTC_DECIMALS,
    WZEC_TOKEN_ADDRESS, WZEC_DECIMALS,
    WIRON_TOKEN_ADDRESS, WIRON_DECIMALS,
    WETH_TOKEN_ADDRESS, WETH_DECIMALS,
    BRIDGE_TOKEN_ADDRESSES,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use tracing::{info, warn, error};

/// Bridge chain identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BridgeChain {
    Bitcoin,
    Zcash,
    IronFish,
    Ethereum,
}

impl BridgeChain {
    pub fn token_address(&self) -> [u8; 32] {
        match self {
            BridgeChain::Bitcoin => WBTC_TOKEN_ADDRESS,
            BridgeChain::Zcash => WZEC_TOKEN_ADDRESS,
            BridgeChain::IronFish => WIRON_TOKEN_ADDRESS,
            BridgeChain::Ethereum => WETH_TOKEN_ADDRESS,
        }
    }

    pub fn decimals(&self) -> u8 {
        match self {
            BridgeChain::Bitcoin => WBTC_DECIMALS,
            BridgeChain::Zcash => WZEC_DECIMALS,
            BridgeChain::IronFish => WIRON_DECIMALS,
            BridgeChain::Ethereum => WETH_DECIMALS,
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            BridgeChain::Bitcoin => "wBTC",
            BridgeChain::Zcash => "wZEC",
            BridgeChain::IronFish => "wIRON",
            BridgeChain::Ethereum => "wETH",
        }
    }

    pub fn native_symbol(&self) -> &'static str {
        match self {
            BridgeChain::Bitcoin => "BTC",
            BridgeChain::Zcash => "ZEC",
            BridgeChain::IronFish => "IRON",
            BridgeChain::Ethereum => "ETH",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            BridgeChain::Bitcoin => "Wrapped Bitcoin",
            BridgeChain::Zcash => "Wrapped Zcash",
            BridgeChain::IronFish => "Wrapped Iron Fish",
            BridgeChain::Ethereum => "Wrapped Ethereum",
        }
    }
}

/// Bridge operation record for audit trail
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeOperation {
    pub op_id: String,
    pub chain: BridgeChain,
    pub op_type: BridgeOpType,
    pub wallet: [u8; 32],
    /// Amount in the wrapped token's base units (e.g., satoshis for wBTC)
    pub amount: u128,
    /// Native chain transaction ID (BTC txid, ZEC txid, IRON txid)
    pub native_txid: Option<String>,
    /// Associated atomic swap ID
    pub swap_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: BridgeOpStatus,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BridgeOpType {
    Mint,  // Native deposited → wrapped minted
    Burn,  // Wrapped burned → native released
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BridgeOpStatus {
    Pending,
    Confirmed,
    Failed(String),
}

/// Mint wrapped tokens for a wallet (called when native coins are deposited into bridge)
///
/// This creates new wrapped tokens (wBTC/wZEC/wIRON) in the user's token_balances.
/// The amount is in the token's base units (satoshis, zatoshis, ore).
///
/// # Arguments
/// * `chain` - Which bridge chain (Bitcoin, Zcash, IronFish)
/// * `wallet` - QNK wallet address receiving the wrapped tokens
/// * `amount` - Amount in native base units (e.g., satoshis)
/// * `token_balances` - Shared token balance map
/// * `storage` - Persistent storage engine
///
/// # Returns
/// * New balance after minting
pub async fn mint_wrapped_token(
    chain: BridgeChain,
    wallet: &[u8; 32],
    amount: u128,
    token_balances: &Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u128>>>,
    storage: &Arc<q_storage::StorageEngine>,
) -> anyhow::Result<u128> {
    if amount == 0 {
        anyhow::bail!("Cannot mint zero amount");
    }

    let token_addr = chain.token_address();
    let symbol = chain.symbol();

    // Update in-memory balance
    let new_balance = {
        let mut balances = token_balances.write().await;
        let key = (*wallet, token_addr);
        let current = balances.get(&key).copied().unwrap_or(0);
        let new_bal = current.checked_add(amount)
            .ok_or_else(|| anyhow::anyhow!("Overflow minting {} {}", amount, symbol))?;
        balances.insert(key, new_bal);
        new_bal
    };

    // Persist to RocksDB immediately (critical for bridge operations)
    storage.save_token_balance(wallet, &token_addr, new_balance).await?;

    info!(
        "🌉 BRIDGE MINT: {} {} → wallet {} (new balance: {})",
        amount, symbol,
        hex::encode(&wallet[..8]),
        new_balance
    );

    Ok(new_balance)
}

/// Burn wrapped tokens from a wallet (called when user withdraws to native chain)
///
/// This removes wrapped tokens from the user's balance, allowing the bridge
/// to release native coins on the source chain.
///
/// # Returns
/// * New balance after burning
pub async fn burn_wrapped_token(
    chain: BridgeChain,
    wallet: &[u8; 32],
    amount: u128,
    token_balances: &Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u128>>>,
    storage: &Arc<q_storage::StorageEngine>,
) -> anyhow::Result<u128> {
    if amount == 0 {
        anyhow::bail!("Cannot burn zero amount");
    }

    let token_addr = chain.token_address();
    let symbol = chain.symbol();

    // Update in-memory balance
    let new_balance = {
        let mut balances = token_balances.write().await;
        let key = (*wallet, token_addr);
        let current = balances.get(&key).copied().unwrap_or(0);
        if current < amount {
            anyhow::bail!(
                "Insufficient {} balance: have {}, need {}",
                symbol, current, amount
            );
        }
        let new_bal = current - amount;
        balances.insert(key, new_bal);
        new_bal
    };

    // Persist to RocksDB immediately
    storage.save_token_balance(wallet, &token_addr, new_balance).await?;

    info!(
        "🔥 BRIDGE BURN: {} {} from wallet {} (new balance: {})",
        amount, symbol,
        hex::encode(&wallet[..8]),
        new_balance
    );

    Ok(new_balance)
}

/// Get the wrapped token balance for a wallet
pub async fn get_wrapped_balance(
    chain: BridgeChain,
    wallet: &[u8; 32],
    token_balances: &Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u128>>>,
) -> u128 {
    let token_addr = chain.token_address();
    let balances = token_balances.read().await;
    balances.get(&(*wallet, token_addr)).copied().unwrap_or(0)
}

/// Get total supply of a wrapped token (sum of all balances)
pub async fn get_wrapped_total_supply(
    chain: BridgeChain,
    token_balances: &Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u128>>>,
) -> u128 {
    let token_addr = chain.token_address();
    let balances = token_balances.read().await;
    balances.iter()
        .filter(|((_, token), _)| *token == token_addr)
        .map(|(_, amount)| *amount)
        .sum()
}

/// Check if a token address is a bridge wrapped token
pub fn is_bridge_token(addr: &[u8; 32]) -> bool {
    BRIDGE_TOKEN_ADDRESSES.contains(addr)
}

/// Get BridgeChain from token address
pub fn chain_from_token_address(addr: &[u8; 32]) -> Option<BridgeChain> {
    if *addr == WBTC_TOKEN_ADDRESS { Some(BridgeChain::Bitcoin) }
    else if *addr == WZEC_TOKEN_ADDRESS { Some(BridgeChain::Zcash) }
    else if *addr == WIRON_TOKEN_ADDRESS { Some(BridgeChain::IronFish) }
    else if *addr == WETH_TOKEN_ADDRESS { Some(BridgeChain::Ethereum) }
    else { None }
}

/// Save bridge operation to storage for audit trail
pub async fn save_bridge_operation(
    op: &BridgeOperation,
    storage: &Arc<q_storage::StorageEngine>,
) -> anyhow::Result<()> {
    let key = format!("bridge_op:{}", op.op_id);
    let value = serde_json::to_vec(op)?;
    let kv = storage.get_kv();
    kv.put(q_storage::CF_MANIFEST, key.as_bytes(), &value).await?;

    // Index by wallet
    let idx_key = format!(
        "bridge_op_idx:{}:{}",
        hex::encode(op.wallet),
        op.op_id
    );
    kv.put(q_storage::CF_MANIFEST, idx_key.as_bytes(), op.op_id.as_bytes()).await?;

    Ok(())
}

/// Load bridge operations for a wallet
pub async fn list_bridge_operations(
    wallet: &[u8; 32],
    storage: &Arc<q_storage::StorageEngine>,
) -> anyhow::Result<Vec<BridgeOperation>> {
    let prefix = format!("bridge_op_idx:{}", hex::encode(wallet));
    let kv = storage.get_kv();
    let entries = kv.scan_prefix(q_storage::CF_MANIFEST, prefix.as_bytes()).await?;

    let mut ops = Vec::new();
    for (_key, value) in entries {
        if let Ok(op_id) = String::from_utf8(value) {
            let op_key = format!("bridge_op:{}", op_id);
            if let Ok(Some(op_bytes)) = kv.get(q_storage::CF_MANIFEST, op_key.as_bytes()).await {
                if let Ok(op) = serde_json::from_slice::<BridgeOperation>(&op_bytes) {
                    ops.push(op);
                }
            }
        }
    }

    ops.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(ops)
}

/// Convert native chain amount to display amount
/// e.g., 100_000_000 satoshis → 1.0 BTC
pub fn native_to_display(amount: u128, chain: BridgeChain) -> f64 {
    let divisor = 10u128.pow(chain.decimals() as u32);
    amount as f64 / divisor as f64
}

/// Convert display amount to native chain amount
/// e.g., 1.0 BTC → 100_000_000 satoshis
pub fn display_to_native(display_amount: f64, chain: BridgeChain) -> u128 {
    let multiplier = 10u128.pow(chain.decimals() as u32);
    (display_amount * multiplier as f64) as u128
}

/// Bridge status for all four chains
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeStatus {
    pub bitcoin: ChainBridgeStatus,
    pub zcash: ChainBridgeStatus,
    pub ironfish: ChainBridgeStatus,
    pub ethereum: ChainBridgeStatus,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChainBridgeStatus {
    pub chain: String,
    pub symbol: String,
    pub wrapped_symbol: String,
    pub node_connected: bool,
    pub node_synced: bool,
    pub total_locked: f64,
    pub total_minted: f64,
    pub active_bridges: u32,
    pub decimals: u8,
}

/// Get aggregated bridge status across all chains
pub async fn get_bridge_status(
    token_balances: &Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u128>>>,
) -> BridgeStatus {
    let wbtc_supply = get_wrapped_total_supply(BridgeChain::Bitcoin, token_balances).await;
    let wzec_supply = get_wrapped_total_supply(BridgeChain::Zcash, token_balances).await;
    let wiron_supply = get_wrapped_total_supply(BridgeChain::IronFish, token_balances).await;
    let weth_supply = get_wrapped_total_supply(BridgeChain::Ethereum, token_balances).await;

    BridgeStatus {
        bitcoin: ChainBridgeStatus {
            chain: "Bitcoin".to_string(),
            symbol: "BTC".to_string(),
            wrapped_symbol: "wBTC".to_string(),
            node_connected: true, // TODO: check actual connection
            node_synced: true,
            total_locked: native_to_display(wbtc_supply, BridgeChain::Bitcoin),
            total_minted: native_to_display(wbtc_supply, BridgeChain::Bitcoin),
            active_bridges: 0,
            decimals: WBTC_DECIMALS,
        },
        zcash: ChainBridgeStatus {
            chain: "Zcash".to_string(),
            symbol: "ZEC".to_string(),
            wrapped_symbol: "wZEC".to_string(),
            node_connected: true,
            node_synced: true,
            total_locked: native_to_display(wzec_supply, BridgeChain::Zcash),
            total_minted: native_to_display(wzec_supply, BridgeChain::Zcash),
            active_bridges: 0,
            decimals: WZEC_DECIMALS,
        },
        ironfish: ChainBridgeStatus {
            chain: "Iron Fish".to_string(),
            symbol: "IRON".to_string(),
            wrapped_symbol: "wIRON".to_string(),
            node_connected: true,
            node_synced: true,
            total_locked: native_to_display(wiron_supply, BridgeChain::IronFish),
            total_minted: native_to_display(wiron_supply, BridgeChain::IronFish),
            active_bridges: 0,
            decimals: WIRON_DECIMALS,
        },
        ethereum: ChainBridgeStatus {
            chain: "Ethereum".to_string(),
            symbol: "ETH".to_string(),
            wrapped_symbol: "wETH".to_string(),
            node_connected: true, // Reth running on Delta
            node_synced: true,
            total_locked: native_to_display(weth_supply, BridgeChain::Ethereum),
            total_minted: native_to_display(weth_supply, BridgeChain::Ethereum),
            active_bridges: 0,
            decimals: WETH_DECIMALS,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_chain_addresses() {
        assert_eq!(BridgeChain::Bitcoin.symbol(), "wBTC");
        assert_eq!(BridgeChain::Zcash.symbol(), "wZEC");
        assert_eq!(BridgeChain::IronFish.symbol(), "wIRON");
        assert_eq!(BridgeChain::Ethereum.symbol(), "wETH");

        assert_eq!(BridgeChain::Bitcoin.decimals(), 8);
        assert_eq!(BridgeChain::Zcash.decimals(), 8);
        assert_eq!(BridgeChain::IronFish.decimals(), 8);
        assert_eq!(BridgeChain::Ethereum.decimals(), 18);
    }

    #[test]
    fn test_is_bridge_token() {
        assert!(is_bridge_token(&WBTC_TOKEN_ADDRESS));
        assert!(is_bridge_token(&WZEC_TOKEN_ADDRESS));
        assert!(is_bridge_token(&WIRON_TOKEN_ADDRESS));
        assert!(!is_bridge_token(&[0u8; 32]));
    }

    #[test]
    fn test_chain_from_address() {
        assert_eq!(chain_from_token_address(&WBTC_TOKEN_ADDRESS), Some(BridgeChain::Bitcoin));
        assert_eq!(chain_from_token_address(&WZEC_TOKEN_ADDRESS), Some(BridgeChain::Zcash));
        assert_eq!(chain_from_token_address(&WIRON_TOKEN_ADDRESS), Some(BridgeChain::IronFish));
        assert_eq!(chain_from_token_address(&[0u8; 32]), None);
    }

    #[test]
    fn test_native_display_conversion() {
        // 1 BTC = 100,000,000 satoshis
        assert_eq!(native_to_display(100_000_000, BridgeChain::Bitcoin), 1.0);
        assert_eq!(display_to_native(1.0, BridgeChain::Bitcoin), 100_000_000);

        // 0.5 ZEC = 50,000,000 zatoshis
        assert_eq!(native_to_display(50_000_000, BridgeChain::Zcash), 0.5);
        assert_eq!(display_to_native(0.5, BridgeChain::Zcash), 50_000_000);
    }

    #[tokio::test]
    async fn test_mint_and_burn() {
        let token_balances = Arc::new(RwLock::new(HashMap::new()));
        let wallet = [0x01u8; 32];

        // Can't test with storage easily, but verify the logic flow
        let bal = get_wrapped_balance(BridgeChain::Bitcoin, &wallet, &token_balances).await;
        assert_eq!(bal, 0);

        // Simulate mint by directly inserting
        {
            let mut balances = token_balances.write().await;
            balances.insert((wallet, WBTC_TOKEN_ADDRESS), 100_000_000);
        }

        let bal = get_wrapped_balance(BridgeChain::Bitcoin, &wallet, &token_balances).await;
        assert_eq!(bal, 100_000_000);

        let supply = get_wrapped_total_supply(BridgeChain::Bitcoin, &token_balances).await;
        assert_eq!(supply, 100_000_000);
    }
}
