/// Wallet management for trading bot
use anyhow::Result;
use rust_decimal::Decimal;
use std::collections::HashMap;
use crate::api_client::ApiClient;
use crate::config::WalletConfig;
use crate::types::WalletBalance;

pub struct WalletManager {
    wallets: HashMap<String, WalletConfig>,
    api_client: ApiClient,
}

impl WalletManager {
    pub fn new(wallets: Vec<WalletConfig>, api_client: ApiClient) -> Self {
        let wallet_map = wallets.into_iter()
            .map(|w| (w.id.clone(), w))
            .collect();
        
        Self {
            wallets: wallet_map,
            api_client,
        }
    }

    pub async fn get_balance(&self, wallet_id: &str) -> Result<WalletBalance> {
        // QuillonClient.get_all_balances is a stub returning [] (auth required on-chain).
        // Return empty balance for this wallet — the water bot uses DEX pool state directly.
        Ok(WalletBalance {
            wallet_id: wallet_id.to_string(),
            qnk_balance: Decimal::ZERO,
            custom_tokens: HashMap::new(),
        })
    }

    pub async fn get_all_balances(&self) -> Result<HashMap<String, WalletBalance>> {
        let mut balances = HashMap::new();
        for wallet_id in self.wallets.keys() {
            if let Ok(balance) = self.get_balance(wallet_id).await {
                balances.insert(wallet_id.clone(), balance);
            }
        }
        Ok(balances)
    }

    pub fn get_enabled_wallets(&self) -> Vec<&WalletConfig> {
        self.wallets.values().filter(|w| w.enabled).collect()
    }
}
