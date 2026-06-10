//! File-based persistence for multisig wallets + proposals.
//!
//! Same pattern as `agent_panel::score_history` v10.10.10 used: JSON files
//! under `$Q_DB_PATH/multisig/`, atomic-rename writes, load-on-startup.
//! Avoids widening the q-storage hot_db API for v0; v10.10.11+ can move to a
//! dedicated `CF_MULTISIG` column family once we're sure of the wire format.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use crate::proposal::{MultisigProposal, ProposalId};
use crate::wallet::{Address, MultisigWallet};

#[derive(Default, Serialize, Deserialize)]
struct SerializedStore {
    version: u32,
    wallets: HashMap<String, MultisigWallet>, // key: hex(address)
    proposals: HashMap<String, MultisigProposal>, // key: uuid
}

/// In-memory wallet + proposal index. Thread-safe; persisted to JSON on
/// every meaningful mutation.
pub struct MultisigStore {
    by_addr: RwLock<HashMap<Address, MultisigWallet>>,
    by_proposal: RwLock<HashMap<ProposalId, MultisigProposal>>,
}

impl MultisigStore {
    pub fn new() -> Self {
        Self {
            by_addr: RwLock::new(HashMap::new()),
            by_proposal: RwLock::new(HashMap::new()),
        }
    }

    pub fn persist_dir() -> PathBuf {
        let base = std::env::var("Q_DB_PATH")
            .unwrap_or_else(|_| "./data-mainnet-genesis".to_string());
        PathBuf::from(base).join("multisig")
    }

    pub fn wallet_file() -> PathBuf {
        Self::persist_dir().join("store.json")
    }

    // --- Wallets ---

    pub fn insert_wallet(&self, wallet: MultisigWallet) {
        self.by_addr.write().insert(wallet.address, wallet);
    }

    pub fn get_wallet(&self, addr: &Address) -> Option<MultisigWallet> {
        self.by_addr.read().get(addr).cloned()
    }

    pub fn wallet_count(&self) -> usize {
        self.by_addr.read().len()
    }

    pub fn list_wallets(&self) -> Vec<MultisigWallet> {
        self.by_addr.read().values().cloned().collect()
    }

    // --- Proposals ---

    pub fn insert_proposal(&self, proposal: MultisigProposal) {
        self.by_proposal.write().insert(proposal.id, proposal);
    }

    pub fn get_proposal(&self, id: &ProposalId) -> Option<MultisigProposal> {
        self.by_proposal.read().get(id).cloned()
    }

    pub fn update_proposal<F: FnOnce(&mut MultisigProposal)>(
        &self,
        id: &ProposalId,
        f: F,
    ) -> Option<MultisigProposal> {
        let mut map = self.by_proposal.write();
        let p = map.get_mut(id)?;
        f(p);
        Some(p.clone())
    }

    pub fn proposals_for_wallet(&self, addr: &Address) -> Vec<MultisigProposal> {
        self.by_proposal
            .read()
            .values()
            .filter(|p| &p.wallet_addr == addr)
            .cloned()
            .collect()
    }

    // --- Persistence ---

    fn snapshot(&self) -> SerializedStore {
        let wallets: HashMap<String, MultisigWallet> = self
            .by_addr
            .read()
            .iter()
            .map(|(addr, w)| (hex::encode(addr), w.clone()))
            .collect();
        let proposals: HashMap<String, MultisigProposal> = self
            .by_proposal
            .read()
            .iter()
            .map(|(id, p)| (id.to_string(), p.clone()))
            .collect();
        SerializedStore {
            version: 1,
            wallets,
            proposals,
        }
    }

    /// Atomically write the full state to `$Q_DB_PATH/multisig/store.json`.
    pub async fn persist_to_file(&self) -> Result<(), String> {
        let path = Self::wallet_file();
        let tmp = path.with_extension("json.tmp");
        let payload = self.snapshot();
        let bytes = serde_json::to_vec_pretty(&payload).map_err(|e| format!("encode: {}", e))?;
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("create_dir_all: {}", e))?;
            }
        }
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|e| format!("write tmp: {}", e))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .map_err(|e| format!("rename: {}", e))?;
        tracing::debug!(
            wallets = payload.wallets.len(),
            proposals = payload.proposals.len(),
            path = %path.display(),
            "multisig store persisted",
        );
        Ok(())
    }

    pub fn spawn_persist(self: Arc<Self>) {
        tokio::spawn(async move {
            if let Err(e) = self.persist_to_file().await {
                tracing::warn!(error = %e, "multisig store persist_to_file failed");
            }
        });
    }

    pub async fn load_from_file(&self) -> usize {
        let path = Self::wallet_file();
        if !path.exists() {
            return 0;
        }
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "multisig store read failed");
                return 0;
            }
        };
        let payload: SerializedStore = match serde_json::from_slice(&bytes) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "multisig store parse failed (schema?)");
                return 0;
            }
        };
        if payload.version != 1 {
            tracing::warn!(version = payload.version, "multisig store version unknown");
            return 0;
        }
        let mut by_addr = self.by_addr.write();
        for (_hex, wallet) in payload.wallets {
            by_addr.insert(wallet.address, wallet);
        }
        let mut by_proposal = self.by_proposal.write();
        for (_uuid, proposal) in payload.proposals {
            by_proposal.insert(proposal.id, proposal);
        }
        let total = by_addr.len() + by_proposal.len();
        tracing::info!(
            wallets = by_addr.len(),
            proposals = by_proposal.len(),
            "multisig store loaded from disk",
        );
        total
    }
}

impl Default for MultisigStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global store so the API handlers + mempool path see the same
/// state without threading through AppState.
static GLOBAL_STORE: OnceLock<Arc<MultisigStore>> = OnceLock::new();

pub fn global() -> Arc<MultisigStore> {
    GLOBAL_STORE.get_or_init(|| Arc::new(MultisigStore::new())).clone()
}
