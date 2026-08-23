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
use crate::wallet::{Address, HybridPublicKey, MultisigWallet};

#[derive(Default, Serialize, Deserialize)]
struct SerializedStore {
    version: u32,
    wallets: HashMap<String, MultisigWallet>, // key: hex(address)
    proposals: HashMap<String, MultisigProposal>, // key: uuid
    #[serde(default)]
    member_keys: HashMap<String, RegisteredKey>, // key: hex(regular qnk address)
    #[serde(default)]
    org_policies: HashMap<String, OrgPolicy>, // key: hex(wallet address)
}

/// Declared (NOT chain-enforced) per-member spending policy for an org.
/// Mirrors the frontend's `OrgDraft`/`OrgMember` shape. This is persisted
/// server-side — unlike the original localStorage-only draft — purely so
/// every member (not just the CEO's own browser) can read their own role
/// and limits, e.g. to show a "you're in, here's what you can spend"
/// welcome moment. Enforcement is a separate, not-yet-built, transaction-
/// validation-path feature; a member holding their own key can still sign
/// a plain transfer that ignores these numbers.
#[derive(Clone, Serialize, Deserialize)]
pub struct OrgPolicy {
    pub org_name: String,
    pub ceo_address_hex: String,
    pub members: Vec<OrgPolicyMember>,
    pub updated_at_unix: i64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OrgPolicyMember {
    pub address_hex: String, // ordinary qnk address (hex, no prefix)
    pub name: String,
    pub role: String,
    pub per_tx_limit: u64,
    pub daily_limit: u64,
    pub approval_threshold: u64,
    pub approvals_required: u64,
}

/// A member's hybrid pubkey bundle, registered once by that member's own
/// wallet (via a Hybrid-scheme signed call) so a CEO can later reference
/// them by their ordinary `qnk` address when building a `MultisigWallet`.
/// `label` is optional operator-facing context ("associate's display name
/// at registration time"); the org's own member label (set at wallet
/// creation) is what actually gets stored on the `MultisigWallet` itself.
#[derive(Clone, Serialize, Deserialize)]
pub struct RegisteredKey {
    pub pubkey: HybridPublicKey,
    pub registered_at_unix: i64,
}

/// In-memory wallet + proposal index. Thread-safe; persisted to JSON on
/// every meaningful mutation.
pub struct MultisigStore {
    by_addr: RwLock<HashMap<Address, MultisigWallet>>,
    by_proposal: RwLock<HashMap<ProposalId, MultisigProposal>>,
    /// Keyed by the member's ORDINARY qnk wallet address (Ed25519-derived) —
    /// NOT `HybridPublicKey::member_address()`, which is a different value
    /// derived from the full hybrid bundle. This registry is the bridge: a
    /// CEO types a familiar qnk address, this resolves it to the pubkey
    /// bundle `MultisigWallet::new` needs.
    member_keys: RwLock<HashMap<Address, RegisteredKey>>,
    /// Keyed by the multisig wallet's own derived address.
    org_policies: RwLock<HashMap<Address, OrgPolicy>>,
}

impl MultisigStore {
    pub fn new() -> Self {
        Self {
            by_addr: RwLock::new(HashMap::new()),
            by_proposal: RwLock::new(HashMap::new()),
            member_keys: RwLock::new(HashMap::new()),
            org_policies: RwLock::new(HashMap::new()),
        }
    }

    // --- Member key registry ---

    pub fn register_member_key(&self, address: Address, pubkey: HybridPublicKey) {
        self.member_keys.write().insert(
            address,
            RegisteredKey {
                pubkey,
                registered_at_unix: chrono::Utc::now().timestamp(),
            },
        );
    }

    pub fn get_member_key(&self, address: &Address) -> Option<RegisteredKey> {
        self.member_keys.read().get(address).cloned()
    }

    pub fn is_registered(&self, address: &Address) -> bool {
        self.member_keys.read().contains_key(address)
    }

    // --- Org policy (declared, not enforced) ---

    pub fn set_org_policy(&self, wallet_addr: Address, policy: OrgPolicy) {
        self.org_policies.write().insert(wallet_addr, policy);
    }

    pub fn get_org_policy(&self, wallet_addr: &Address) -> Option<OrgPolicy> {
        self.org_policies.read().get(wallet_addr).cloned()
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
        let member_keys: HashMap<String, RegisteredKey> = self
            .member_keys
            .read()
            .iter()
            .map(|(addr, k)| (hex::encode(addr), k.clone()))
            .collect();
        let org_policies: HashMap<String, OrgPolicy> = self
            .org_policies
            .read()
            .iter()
            .map(|(addr, p)| (hex::encode(addr), p.clone()))
            .collect();
        SerializedStore {
            version: 1,
            wallets,
            proposals,
            member_keys,
            org_policies,
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
        let mut member_keys = self.member_keys.write();
        for (addr_hex, key) in payload.member_keys {
            if let Ok(bytes) = hex::decode(&addr_hex) {
                if bytes.len() == 32 {
                    let mut addr = [0u8; 32];
                    addr.copy_from_slice(&bytes);
                    member_keys.insert(addr, key);
                }
            }
        }
        let mut org_policies = self.org_policies.write();
        for (addr_hex, policy) in payload.org_policies {
            if let Ok(bytes) = hex::decode(&addr_hex) {
                if bytes.len() == 32 {
                    let mut addr = [0u8; 32];
                    addr.copy_from_slice(&bytes);
                    org_policies.insert(addr, policy);
                }
            }
        }
        let total = by_addr.len() + by_proposal.len() + member_keys.len() + org_policies.len();
        tracing::info!(
            wallets = by_addr.len(),
            proposals = by_proposal.len(),
            member_keys = member_keys.len(),
            org_policies = org_policies.len(),
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
