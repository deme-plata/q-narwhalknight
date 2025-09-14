//! Q-Wallet: Placeholder wallet module

pub struct QWallet;

impl QWallet {
    pub fn new() -> Self {
        Self
    }
}

// Placeholder exports for API server compatibility
pub struct WalletManager;

impl WalletManager {
    pub fn new() -> Self {
        Self
    }

    pub async fn create_wallet(&self, _name: &str, _password: &str) -> anyhow::Result<String> {
        Ok("wallet-id-placeholder".to_string())
    }

    pub async fn get_balance(&self, _wallet_id: &str) -> anyhow::Result<u64> {
        Ok(0)
    }

    pub async fn sign_transaction(
        &self,
        _wallet_id: &str,
        _transaction: serde_json::Value,
        _password: Option<&str>,
    ) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({"signed": true}))
    }

    pub async fn get_wallet(&self, _wallet_id: &str) -> anyhow::Result<Option<serde_json::Value>> {
        Ok(Some(serde_json::json!({"id": _wallet_id, "balance": 0})))
    }

    pub async fn list_wallets(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        Ok(vec![serde_json::json!({"id": "wallet1", "balance": 0})])
    }

    pub async fn create_transaction(
        &self,
        _request: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({"tx_id": "tx123", "status": "created"}))
    }
}

pub struct MemoryWalletStore;

impl MemoryWalletStore {
    pub fn new() -> Self {
        Self
    }
}
