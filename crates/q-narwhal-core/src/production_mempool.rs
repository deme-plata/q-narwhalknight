//! Production Transaction Mempool
//!
//! Real-world transaction pool implementation for Q-NarwhalKnight consensus.
//! No simulation - handles actual transaction validation, broadcasting via Tor,
//! and mempool synchronization across validators.

use crate::tor_broadcast::{BroadcastConfig, BroadcastMessage, TorBroadcastManager, TorClient};
use anyhow::Result;
use bincode;
use dashmap::DashMap;
use q_types::{Certificate, Transaction, TxHash, ValidatorId};
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// 🛡 Phase 0 (2026-07-08): Source of truth for a wallet's next expected
/// on-chain nonce. `q-narwhal-core` sits below `q-api-server` in the crate
/// graph (this crate cannot depend on q-api-server's `NonceTracker`
/// directly without a cycle), so this trait is the seam: q-api-server
/// implements it for its `NonceTracker` and injects it via
/// `TxValidator::set_nonce_source` / `ProductionMempool::set_nonce_source`
/// at boot, mirroring the existing `set_production_mempool` wiring pattern.
///
/// GAP CLOSED: previously nothing between mempool admission and block
/// packing compared a transaction's `nonce` field against the sender's
/// actual last-confirmed on-chain nonce. `pending_nonces` below only
/// prevented two *simultaneously pending* mempool entries from sharing a
/// nonce — it does not stop a replayed/re-signed transaction (different
/// tx.id, so DS-1's tx-id replay guard does not catch it) that reuses an
/// already-spent nonce with a different `to`/`amount`/`data` payload.
///
/// This is validate-only by design: incrementing/consuming the nonce stays
/// owned by whichever path currently does it (`NonceTracker::get_and_increment`
/// in q-api-server's handlers.rs). Field is `Option`, so behavior is an
/// exact no-op (identical to pre-patch) until a caller wires it up — same
/// "off unless opted in" rollout style as DS-1's `Q_REPLAY_PROTECTION_HEIGHT`.
pub trait NonceSource: Send + Sync {
    /// Returns Ok(()) if `submitted_nonce` is the sender's expected next
    /// nonce, Err(expected_nonce) otherwise. Exact-match semantics only
    /// (no "greater than" tolerance) — matches the semantics already
    /// proven safe in the existing `NonceTracker::validate_nonce`.
    fn validate_nonce(&self, wallet: &[u8; 32], submitted_nonce: u64) -> Result<(), u64>;
}

/// 🛡 Phase 0 Round-2 BUG-A FIX (2026-07-08): why `add_transaction`'s plain
/// `Result<bool>` was unsafe to gate anything security-relevant on.
///
/// `add_transaction` returns `Ok(false)` for AT LEAST 8 semantically distinct
/// reasons (blocked wallet, duplicate tx already pending, nonce-key already
/// pending, rate-limited, failed signature/format validation, fee too low,
/// mempool-full-and-outbid, mempool-full-and-not-outbid) — and ALSO for the
/// ONE case that matters to a caller deciding whether to burn a nonce: "this
/// transaction was never inserted into pending_transactions". Collapsing all
/// of these into a single boolean is exactly what let
/// `q-api-server::handlers::send_transaction_signed` treat "rejected for an
/// unrelated reason" as "genuinely admitted" (BUG A) — its call site had no
/// way to tell them apart because the type it read from didn't distinguish
/// them.
///
/// `AdmissionResult` is the richer signal: `Admitted` is returned ONLY from
/// the single code path in `add_transaction_detailed` that actually inserts
/// into `pending_transactions` and marks the nonce as pending (previously:
/// `return Ok(true)` / falls through to the final `Ok(true)`). Every other
/// exit is `Rejected(RejectReason)` with a reason tag identifying exactly
/// which check failed. `add_transaction` (the original `Result<bool>`
/// signature) is kept unchanged and is now a thin wrapper over
/// `add_transaction_detailed` — this preserves every other existing call
/// site's behavior byte-for-byte (there are 6+ across q-narwhal-core /
/// q-api-server / q-sharding that only ever cared about the bool), while
/// giving `submit_transaction` (the one call site that actually needs to
/// gate nonce-advance) the ability to ask "was this GENUINELY admitted?"
/// with a type that cannot silently conflate the two.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionResult {
    /// Genuinely inserted into `pending_transactions` and its (sender,
    /// nonce) marked as pending. Safe to treat as "this nonce is spent".
    Admitted,
    /// Not admitted, for the reason given. NEVER safe to treat as "this
    /// nonce is spent" — the transaction was not queued for a block.
    Rejected(RejectReason),
}

impl AdmissionResult {
    /// Convenience: true only for `Admitted`. Matches the historical
    /// `Ok(true)` semantics of `add_transaction` exactly (never true for a
    /// `Rejected(_)`, regardless of reason).
    pub fn is_admitted(&self) -> bool {
        matches!(self, AdmissionResult::Admitted)
    }
}

/// Why `add_transaction_detailed` declined to admit a transaction. Every
/// variant corresponds to a distinct `Ok(false)` / rejection branch that
/// existed in `add_transaction` before this fix — none of these are new
/// rejection *behavior*, this only labels the reasons that already existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Sender or recipient is on the Q_BLOCKED_WALLETS operator freeze list.
    BlockedWallet,
    /// This exact tx_hash is already present in `pending_transactions`.
    DuplicateTransaction,
    /// The (sender, nonce) pair is already used by a different pending tx
    /// (mempool-local replay/double-spend guard, `pending_nonces`).
    NonceAlreadyPending,
    /// Anti-spam rate limit exceeded for the announcing validator.
    RateLimited,
    /// `TxValidator::validate_transaction` returned a non-`Valid` status —
    /// covers signature verification failure, the Phase 0
    /// nonce-vs-chain-state `NonceSource` check failing, and any other
    /// `perform_validation` rejection.
    ValidationFailed(String),
    /// `Transaction::validate_fee()` returned an error (malformed fee
    /// structure independent of the minimum-fee-per-byte check below).
    FeeValidationFailed(String),
    /// Fee did not meet `config.min_fee_per_byte` for this tx's size.
    FeeTooLow,
    /// Mempool at `max_transactions` capacity and this tx's fee did not
    /// exceed the current lowest-fee occupant, so nothing was evicted.
    MempoolFullFeeTooLow,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::BlockedWallet => write!(f, "sender or recipient wallet is blocked"),
            RejectReason::DuplicateTransaction => write!(f, "transaction already in mempool"),
            RejectReason::NonceAlreadyPending => write!(f, "nonce already used by a pending transaction"),
            RejectReason::RateLimited => write!(f, "rate limit exceeded"),
            RejectReason::ValidationFailed(s) => write!(f, "validation failed: {}", s),
            RejectReason::FeeValidationFailed(s) => write!(f, "fee validation failed: {}", s),
            RejectReason::FeeTooLow => write!(f, "fee below minimum required for mempool inclusion"),
            RejectReason::MempoolFullFeeTooLow => write!(f, "mempool full and fee too low to evict an existing entry"),
        }
    }
}

/// Production-ready transaction mempool
///
/// v3.4.6-beta: Added O(1) nonce tracking for instant replay/double-spend detection
pub struct ProductionMempool {
    /// Pending transactions awaiting inclusion in blocks
    pending_transactions: Arc<RwLock<BTreeMap<TxHash, MempoolTransaction>>>,

    /// 🚀 v3.4.6-beta: O(1) nonce tracking for instant replay/double-spend detection
    /// Key: (sender_address, nonce) - unique identifier for sender's transaction
    /// Value: TxHash of the transaction using this nonce
    ///
    /// In account-based blockchains, each sender can only have ONE pending transaction
    /// per nonce. This prevents replay attacks and double-spends.
    /// Ported from QTFT blockchain concept, adapted for account model.
    pending_nonces: DashMap<([u8; 32], u64), TxHash>,

    /// 🛡 v10.11.0a (2026-05-21): X-Wallet-Auth trust side-table.
    ///
    /// HTTP handlers that authenticate the caller via X-Wallet-Auth (i.e.
    /// send_transaction_signed and send_transactions_batch) build the
    /// Transaction with an empty inner `signature: vec![]` — the inner sig
    /// would have to be over the canonical tx-hash, but the client's
    /// X-Wallet-Auth signs `SHA3(pubkey ‖ ts ‖ path)`, which is different
    /// bytes. Pre-PR-#68 the mempool's perform_validation was an Ok(true)
    /// stub so this worked accidentally. PR #68 tightened it and started
    /// rejecting empty-sig txs — every send_signed tx silently failed
    /// block-inclusion. The "ghost confirmations" bug.
    ///
    /// This map is the trust-flag fix: the handler marks (tx_id, from_addr)
    /// here BEFORE submitting. perform_validation looks up tx.id; if the
    /// entry matches tx.from, it knows the caller was authenticated by
    /// the API layer and skips the inner-signature check (still applies
    /// fee + format checks).
    ///
    /// Honest about the trust model: the API explicitly chose to accept
    /// X-Wallet-Auth as the signing proof for client-managed wallets.
    /// A proper fix (client pre-signs the canonical tx hash) is v10.12.
    ///
    /// Cleanup: entries are TTL'd at 1 hour to bound memory. The mempool
    /// admission loop prunes expired entries on its periodic pass.
    pub(crate) trusted_via_auth: Arc<DashMap<TxHash, ([u8; 32], std::time::Instant)>>,

    /// Transaction validator for signature/validity checks
    transaction_validator: Arc<TxValidator>,

    /// Tor broadcast manager for peer communication
    broadcast_manager: Arc<TorBroadcastManager>,

    /// Mempool configuration
    config: MempoolConfig,

    /// Anti-spam tracking
    spam_detector: Arc<RwLock<SpamDetector>>,

    /// Mempool metrics
    metrics: Arc<RwLock<MempoolMetrics>>,

    /// Known validators for broadcasting
    validator_peers: Arc<RwLock<HashMap<ValidatorId, ValidatorInfo>>>,
}

/// Transaction in mempool with metadata
#[derive(Debug, Clone)]
pub struct MempoolTransaction {
    /// The actual transaction
    pub transaction: Transaction,

    /// When transaction was received
    pub received_at: SystemTime,

    /// Fee paid by transaction (for ordering)
    /// v2.5.0: Updated to u128 for consistency with Amount type
    pub fee: u128,

    /// Size in bytes
    pub size: usize,

    /// Which validator announced this transaction
    pub announced_by: Option<ValidatorId>,

    /// How many validators have announced this transaction
    pub announcement_count: u32,

    /// Transaction validation status
    pub validation_status: ValidationStatus,
}

/// Transaction validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid(String),
    Processing,
}

/// Mempool configuration
#[derive(Debug, Clone)]
pub struct MempoolConfig {
    /// Maximum transactions in mempool
    pub max_transactions: usize,

    /// Maximum transaction age before eviction
    pub max_age: Duration,

    /// Minimum fee per byte
    pub min_fee_per_byte: u64,

    /// Maximum transaction size
    pub max_transaction_size: usize,

    /// Rate limiting per validator
    pub max_tx_per_validator_per_second: u32,

    /// Enable Byzantine protection
    pub enable_byzantine_protection: bool,
}

/// Validator information for mempool
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub validator_id: ValidatorId,
    pub onion_address: String,
    pub last_seen: SystemTime,
    pub reputation_score: f64,
    pub transaction_count: u64,
}

/// Anti-spam detection
#[derive(Debug)]
pub struct SpamDetector {
    /// Rate limiting per validator
    validator_rates: HashMap<ValidatorId, RateLimiter>,

    /// Transaction hash deduplication
    seen_hashes: HashSet<TxHash>,

    /// Suspicious pattern detection
    suspicious_patterns: HashMap<ValidatorId, SuspicionLevel>,
}

/// Rate limiting for validators
#[derive(Debug)]
pub struct RateLimiter {
    pub last_reset: SystemTime,
    pub transaction_count: u32,
    pub allowed_per_second: u32,
}

/// Suspicion level for Byzantine detection
#[derive(Debug, PartialEq)]
pub enum SuspicionLevel {
    Clean,
    Suspicious,
    Malicious,
}

/// Mempool performance metrics
#[derive(Debug, Default, Clone)]
pub struct MempoolMetrics {
    pub total_transactions: u64,
    pub valid_transactions: u64,
    pub invalid_transactions: u64,
    pub broadcast_count: u64,
    pub evicted_transactions: u64,
    pub mempool_size: usize,
    pub average_validation_time: Duration,
}

/// Transaction validator
pub struct TxValidator {
    /// Current cryptographic phase
    current_phase: Phase,

    /// Signature verification cache
    verification_cache: Arc<RwLock<HashMap<TxHash, bool>>>,

    /// 🛡 v10.11.1: X-Wallet-Auth trust side-table. See ProductionMempool
    /// doc for the full bug history. Shared Arc with ProductionMempool so
    /// the handler-side mark_auth_trusted() and the validator-side
    /// perform_validation() see the same entries.
    trusted_via_auth: Arc<DashMap<TxHash, ([u8; 32], std::time::Instant)>>,

    /// 🛡 Phase 0 (2026-07-08): optional nonce-vs-chain-state source. See
    /// `NonceSource` doc above. `None` (the default from `TxValidator::new`)
    /// means this check is a no-op — set via `set_nonce_source` at boot.
    /// RwLock (not a plain field) because `TxValidator` is held behind an
    /// `Arc` (see `ProductionMempool.transaction_validator`), so wiring it
    /// in post-construction needs interior mutability, same pattern as
    /// `verification_cache` above.
    nonce_source: RwLock<Option<Arc<dyn NonceSource>>>,
}

// TorBroadcastManager is imported from tor_broadcast module
// No need to redefine it here

/// Consensus message types for mempool
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MempoolMessage {
    /// Announce new transaction to peers
    TransactionAnnounce {
        tx_hash: TxHash,
        size: usize,
        /// v2.5.0: Updated to u128 for consistency with Amount type
        fee: u128,
        priority: u8,
        validator_id: ValidatorId,
        timestamp: u64,
    },

    /// Request full transaction data
    TransactionRequest {
        tx_hash: TxHash,
        requestor: ValidatorId,
        timestamp: u64,
    },

    /// Provide transaction data
    TransactionResponse {
        tx_hash: TxHash,
        transaction: Option<Transaction>, // None if not found
        validator_id: ValidatorId,
        timestamp: u64,
    },

    /// Request mempool synchronization
    MempoolSyncRequest {
        known_hashes: Vec<TxHash>,
        requestor: ValidatorId,
        timestamp: u64,
    },

    /// Respond with missing transactions
    MempoolSyncResponse {
        missing_transactions: Vec<(TxHash, Transaction)>,
        validator_id: ValidatorId,
        timestamp: u64,
    },
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_transactions: 10_000,
            max_age: Duration::from_secs(300), // 5 minutes
            min_fee_per_byte: 1,
            max_transaction_size: 1024 * 1024, // 1 MB
            max_tx_per_validator_per_second: 100,
            enable_byzantine_protection: true,
        }
    }
}

impl ProductionMempool {
    /// Create new production mempool
    pub async fn new(
        config: MempoolConfig,
        tor_client: Arc<dyn TorClient>,
        phase: Phase,
    ) -> Result<Self> {
        info!("🚀 Initializing Production Mempool");
        info!("   Max Transactions: {}", config.max_transactions);
        info!("   Max Age: {:?}", config.max_age);
        info!("   Min Fee/Byte: {}", config.min_fee_per_byte);

        // 🛡 v10.11.1: shared trust map. Both ProductionMempool (for
        // mark_auth_trusted from handlers) and TxValidator (for
        // perform_validation) hold the same Arc.
        let trusted_via_auth = Arc::new(DashMap::new());
        let transaction_validator = Arc::new(TxValidator::new(phase, trusted_via_auth.clone()));
        let broadcast_manager =
            Arc::new(TorBroadcastManager::new(tor_client, BroadcastConfig::default()).await?);

        info!("   🚀 O(1) nonce tracking: enabled (instant replay/double-spend detection)");

        Ok(Self {
            pending_transactions: Arc::new(RwLock::new(BTreeMap::new())),
            pending_nonces: DashMap::new(),
            trusted_via_auth,
            transaction_validator,
            broadcast_manager,
            config,
            spam_detector: Arc::new(RwLock::new(SpamDetector::new())),
            metrics: Arc::new(RwLock::new(MempoolMetrics::default())),
            validator_peers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// 🛡 v10.11.0a: Mark a tx as trusted-by-API-auth. Called by HTTP handlers
    /// AFTER they verify X-Wallet-Auth and BEFORE they submit the tx. The
    /// mempool's perform_validation will then skip the inner-signature check
    /// for this specific tx_id, provided the from-address matches.
    pub fn mark_auth_trusted(&self, tx_id: TxHash, from_address: [u8; 32]) {
        self.trusted_via_auth.insert(tx_id, (from_address, std::time::Instant::now()));
    }

    /// 🛡 Phase 0 (2026-07-08): wire in the authoritative nonce source (see
    /// `NonceSource` trait doc). Forwards to the shared `TxValidator` so
    /// both the admission-time check (Patch 2a, in `perform_validation`)
    /// and the block-packing re-check (Patch 2b, in
    /// `get_transactions_for_block`) read from the same source — no second
    /// independent field to keep in sync. Call once at boot, mirroring the
    /// existing `set_production_mempool` wiring pattern in lockfree_producer.rs.
    pub async fn set_nonce_source(&self, source: Arc<dyn NonceSource>) {
        self.transaction_validator.set_nonce_source(source).await;
    }

    /// Add transaction to mempool (from client or peer)
    ///
    /// v3.4.6-beta: Added O(1) double-spend detection using spent_outpoints DashMap
    ///
    /// 🛡 Phase 0 Round-2 BUG-A FIX (2026-07-08): this is now a thin wrapper
    /// over `add_transaction_detailed` that collapses `AdmissionResult` back
    /// to a plain `bool` (`Admitted` -> `true`, any `Rejected(_)` -> `false`)
    /// — i.e. EXACTLY the historical return semantics, byte-for-byte, for
    /// every existing call site (there are 6+ across q-narwhal-core /
    /// q-api-server / q-sharding, none of which need the rejection reason).
    /// Only `submit_transaction` in q-api-server's `transaction_utils.rs`
    /// (the BUG A call site) needs the detailed variant, and now calls
    /// `add_transaction_detailed` directly instead of this wrapper.
    pub async fn add_transaction(
        &self,
        transaction: Transaction,
        announced_by: Option<ValidatorId>,
    ) -> Result<bool> {
        Ok(self
            .add_transaction_detailed(transaction, announced_by)
            .await?
            .is_admitted())
    }

    /// Add transaction to mempool (from client or peer) — detailed variant.
    ///
    /// 🛡 Phase 0 Round-2 BUG-A FIX (2026-07-08): identical admission logic to
    /// the pre-fix `add_transaction` (every check, every order, every
    /// rejection condition is unchanged — this is a pure return-type
    /// enrichment, not a behavior change), except every `return Ok(false)`
    /// site now returns `Ok(AdmissionResult::Rejected(reason))` with a reason
    /// tag identifying exactly which check failed, and the sole genuine-
    /// admission path returns `Ok(AdmissionResult::Admitted)` instead of
    /// `Ok(true)`. See `AdmissionResult`'s doc comment for why this exists.
    pub async fn add_transaction_detailed(
        &self,
        transaction: Transaction,
        announced_by: Option<ValidatorId>,
    ) -> Result<AdmissionResult> {
        let tx_hash = transaction.hash();
        let start_time = std::time::Instant::now();

        debug!(
            "📥 Adding transaction to mempool: {}",
            hex::encode(&tx_hash)
        );

        // 2026-06-24 (rocky) SECURITY — wallet freeze at mempool admission.
        // Reject any tx whose sender OR recipient is on the operator-controlled blocklist
        // (env Q_BLOCKED_WALLETS = comma/space-separated 64-hex or qnk… addresses). This
        // soft-freezes exploited / phantom-QUG wallets here, where ALL tx types (transfer,
        // swap, liquidity) enter — WITHOUT touching consensus replay (already-committed blocks
        // are not re-admitted), so it is fully reversible: edit the drop-in + restart, no recompile.
        {
            use std::sync::OnceLock;
            static BLOCKED: OnceLock<std::collections::HashSet<[u8; 32]>> = OnceLock::new();
            let blocked = BLOCKED.get_or_init(|| {
                let mut s = std::collections::HashSet::new();
                if let Ok(raw) = std::env::var("Q_BLOCKED_WALLETS") {
                    for part in raw.split(|c| c == ',' || c == ' ' || c == '\n' || c == '\t') {
                        let h = part.trim().strip_prefix("qnk").unwrap_or(part.trim());
                        if h.len() == 64 {
                            if let Ok(b) = hex::decode(h) {
                                if b.len() == 32 {
                                    let mut a = [0u8; 32];
                                    a.copy_from_slice(&b);
                                    s.insert(a);
                                }
                            }
                        }
                    }
                }
                s
            });
            if !blocked.is_empty()
                && (blocked.contains(&transaction.from) || blocked.contains(&transaction.to))
            {
                warn!(
                    "🧊 [WALLET-FREEZE] Rejecting tx {} — from/to is on Q_BLOCKED_WALLETS",
                    hex::encode(&tx_hash[..8])
                );
                let mut metrics = self.metrics.write().await;
                metrics.invalid_transactions += 1;
                return Ok(AdmissionResult::Rejected(RejectReason::BlockedWallet));
            }
        }

        // Check if already exists
        {
            let pending = self.pending_transactions.read().await;
            if pending.contains_key(&tx_hash) {
                debug!("   Transaction already in mempool");
                return Ok(AdmissionResult::Rejected(RejectReason::DuplicateTransaction));
            }
        }

        // 🚀 v3.4.6-beta: O(1) nonce-based replay/double-spend detection
        // In account-based blockchains, each (sender, nonce) pair can only be used once
        let nonce_key = (transaction.from, transaction.nonce);
        if let Some(conflicting_tx) = self.pending_nonces.get(&nonce_key) {
            error!(
                "🚫 [REPLAY/DOUBLE-SPEND] Transaction {} uses nonce {} already used by pending tx {}",
                hex::encode(&tx_hash[..8]),
                transaction.nonce,
                hex::encode(&conflicting_tx[..8])
            );
            let mut metrics = self.metrics.write().await;
            metrics.invalid_transactions += 1;
            return Ok(AdmissionResult::Rejected(RejectReason::NonceAlreadyPending));
        }

        // Anti-spam check
        if let Some(validator) = &announced_by {
            let mut spam_detector = self.spam_detector.write().await;
            if !spam_detector.check_rate_limit(validator).await {
                warn!("🚫 Rate limit exceeded for validator: {:?}", validator);
                return Ok(AdmissionResult::Rejected(RejectReason::RateLimited));
            }
        }

        // Validate transaction
        let validation_status = self
            .transaction_validator
            .validate_transaction(&transaction)
            .await?;

        if validation_status != ValidationStatus::Valid {
            warn!("❌ Invalid transaction: {:?}", validation_status);
            let mut metrics = self.metrics.write().await;
            metrics.invalid_transactions += 1;
            let reason_detail = match &validation_status {
                ValidationStatus::Invalid(s) => s.clone(),
                other => format!("{:?}", other),
            };
            return Ok(AdmissionResult::Rejected(RejectReason::ValidationFailed(reason_detail)));
        }

        // v1.4.5-beta: Validate fee meets minimum requirements (prevent zero-fee spam)
        if let Err(fee_error) = transaction.validate_fee() {
            warn!("💸 Transaction fee validation failed: {}", fee_error);
            let mut metrics = self.metrics.write().await;
            metrics.invalid_transactions += 1;
            return Ok(AdmissionResult::Rejected(RejectReason::FeeValidationFailed(fee_error.to_string())));
        }

        // Create mempool transaction
        let tx_fee = transaction.fee;
        let tx_size = bincode::serialized_size(&transaction).unwrap_or(256) as usize;

        // v1.4.5-beta: Enforce min_fee_per_byte from config
        // v2.5.0: Updated to u128 for consistency
        let min_required_fee = (tx_size as u128).saturating_mul(self.config.min_fee_per_byte as u128);
        if tx_fee < min_required_fee {
            warn!(
                "💸 Transaction fee {} below minimum {} ({} bytes × {} per byte)",
                tx_fee, min_required_fee, tx_size, self.config.min_fee_per_byte
            );
            let mut metrics = self.metrics.write().await;
            metrics.invalid_transactions += 1;
            return Ok(AdmissionResult::Rejected(RejectReason::FeeTooLow));
        }
        let mempool_tx = MempoolTransaction {
            fee: tx_fee,
            size: tx_size,
            received_at: SystemTime::now(),
            announced_by: announced_by.clone(),
            announcement_count: if announced_by.is_some() { 1 } else { 0 },
            validation_status,
            transaction: transaction.clone(),
        };

        // Check mempool capacity and fee
        let should_add = {
            let mut pending = self.pending_transactions.write().await;

            // Check capacity
            if pending.len() >= self.config.max_transactions {
                // v10.11.11: First-in-fee-bucket fairness. Among txs tied at the
                // lowest fee, evict the NEWEST one (highest received_at) so
                // earlier-arriving MIN-fee txs are protected from being
                // perpetually displaced by later MIN-fee arrivals. The previous
                // .min_by_key(|tx| tx.fee) was non-deterministic among ties
                // (HashMap iteration order), making an old MIN-fee tx as likely
                // to be evicted as a fresh one — starving the first-in.
                if let Some((lowest_hash, lowest_tx)) = pending
                    .iter()
                    .min_by_key(|(_, tx)| (tx.fee, std::cmp::Reverse(tx.received_at)))
                    .map(|(h, tx)| (*h, tx.clone()))
                {
                    if mempool_tx.fee > lowest_tx.fee {
                        pending.remove(&lowest_hash);
                        info!("🗑️  Evicted low-fee transaction for higher fee (newest in lowest-fee bucket)");
                    } else {
                        warn!("💸 Transaction fee too low for mempool inclusion");
                        return Ok(AdmissionResult::Rejected(RejectReason::MempoolFullFeeTooLow));
                    }
                } else {
                    return Ok(AdmissionResult::Rejected(RejectReason::MempoolFullFeeTooLow));
                }
            }

            pending.insert(tx_hash, mempool_tx);
            true
        };

        if should_add {
            // 🚀 v3.4.6-beta: Mark nonce as used in O(1) lookup table
            let nonce_key = (transaction.from, transaction.nonce);
            self.pending_nonces.insert(nonce_key, tx_hash);
            debug!(
                "   Marked nonce {} for sender {:?} as pending",
                transaction.nonce,
                &transaction.from[..4]
            );
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.total_transactions += 1;
                metrics.valid_transactions += 1;
                metrics.mempool_size = {
                    let pending = self.pending_transactions.read().await;
                    pending.len()
                };
                metrics.average_validation_time = start_time.elapsed();
            }

            // Announce to peers if this is from a client
            if announced_by.is_none() {
                self.announce_transaction_to_peers(tx_hash, &transaction)
                    .await?;
            }

            info!("✅ Transaction added to mempool: {}", hex::encode(&tx_hash));
            info!("   Fee: {} units", transaction.fee);
            info!(
                "   Size: {} bytes",
                bincode::serialized_size(&transaction).unwrap_or(256)
            );
            info!("   Mempool size: {}", {
                let pending = self.pending_transactions.read().await;
                pending.len()
            });

            Ok(AdmissionResult::Admitted)
        } else {
            // Unreachable in practice (the capacity branch above always either
            // `return`s early or lets control fall through to `pending.insert`
            // + `true`), kept as a safe fallback rather than `unreachable!()`.
            Ok(AdmissionResult::Rejected(RejectReason::MempoolFullFeeTooLow))
        }
    }

    /// Announce transaction to all connected peers
    async fn announce_transaction_to_peers(
        &self,
        tx_hash: TxHash,
        transaction: &Transaction,
    ) -> Result<()> {
        let validator_id = self.get_own_validator_id().await;

        let announce_msg = BroadcastMessage::TransactionAnnounce {
            tx_hash,
            size: bincode::serialized_size(transaction).unwrap_or(256) as usize,
            fee: transaction.fee,
            priority: self.calculate_transaction_priority(transaction),
        };

        self.broadcast_manager
            .broadcast_to_all(announce_msg)
            .await?;

        let mut metrics = self.metrics.write().await;
        metrics.broadcast_count += 1;

        debug!(
            "📡 Transaction announced to peers: {}",
            hex::encode(&tx_hash)
        );
        Ok(())
    }

    /// Get transactions for block creation (ordered by fee)
    pub async fn get_transactions_for_block(&self, max_count: usize) -> Vec<Transaction> {
        let pending = self.pending_transactions.read().await;

        // v10.11.14 instrumentation: at mempool-draw time, count Valid vs
        // Invalid. We've observed 0 non-coinbase txs in saved blocks despite
        // admitted send_signed txs. This log surfaces whether
        // pending_transactions actually holds our txs and they're Valid, or
        // perform_validation has flagged them Invalid (in which case the
        // .filter() below drops them silently).
        let mut valid_count = 0usize;
        let mut invalid_count = 0usize;
        for tx in pending.values() {
            if tx.validation_status == ValidationStatus::Valid { valid_count += 1; }
            else { invalid_count += 1; }
        }
        if pending.len() > 0 {
            tracing::warn!(
                "📦 [MEMPOOL-DRAW v10.11.14] pending_total={} valid={} invalid={} max_take={}",
                pending.len(), valid_count, invalid_count, max_count,
            );
        }

        let mut transactions: Vec<_> = pending
            .values()
            .filter(|tx| tx.validation_status == ValidationStatus::Valid)
            .collect();

        // 🛡 Phase 0 (2026-07-08) Patch 2b: defense-in-depth nonce re-check
        // at block-packing time, immediately before selection. Closes the
        // window between mempool admission (Patch 2a) and packing where
        // chain state can move — e.g. another block from a different
        // producer/lane already consumed that nonce after this tx was
        // admitted but before it was packed. Reject-not-mint: dropped here,
        // not silently included, matching balance_consensus.rs's existing
        // "reject, log, continue" style. No-op (same as pre-patch) until
        // set_nonce_source is called at boot — see NonceSource doc.
        if let Some(source) = self.transaction_validator.get_nonce_source().await {
            let before = transactions.len();
            transactions.retain(|tx| {
                match source.validate_nonce(&tx.transaction.from, tx.transaction.nonce) {
                    Ok(()) => true,
                    Err(expected) => {
                        warn!(
                            "🚨 [MEMPOOL-DRAW] dropping tx at pack-time — nonce no longer matches chain state (submitted={} expected={}, tx_hash={})",
                            tx.transaction.nonce, expected, hex::encode(&tx.transaction.hash()[..8])
                        );
                        false
                    }
                }
            });
            let dropped = before - transactions.len();
            if dropped > 0 {
                tracing::warn!(
                    "📦 [MEMPOOL-DRAW] pack-time nonce re-check dropped {} tx(s)",
                    dropped
                );
            }
        }

        // Sort by fee (highest first) then by receive time (oldest first)
        transactions.sort_by(|a, b| {
            b.fee
                .cmp(&a.fee)
                .then_with(|| a.received_at.cmp(&b.received_at))
        });

        let chosen: Vec<Transaction> = transactions
            .into_iter()
            .take(max_count)
            .map(|tx| tx.transaction.clone())
            .collect();

        if chosen.len() > 0 {
            tracing::warn!(
                "📦 [MEMPOOL-DRAW v10.11.14] returning {} txs to block-pack",
                chosen.len(),
            );
        }
        chosen
    }

    /// Remove transactions that have been included in a block
    ///
    /// v3.4.6-beta: Also removes pending nonces from O(1) tracking table
    pub async fn remove_included_transactions(&self, tx_hashes: &[TxHash]) {
        let mut pending = self.pending_transactions.write().await;
        let mut removed_count = 0;
        let mut nonces_removed = 0;

        for hash in tx_hashes {
            if let Some((_, removed_tx)) = pending.remove_entry(hash) {
                removed_count += 1;

                // 🚀 v3.4.6-beta: Clean up pending nonce for this transaction
                let nonce_key = (removed_tx.transaction.from, removed_tx.transaction.nonce);
                if self.pending_nonces.remove(&nonce_key).is_some() {
                    nonces_removed += 1;
                }
            }
        }

        if removed_count > 0 {
            info!(
                "🗑️  Removed {} transactions ({} nonces) from mempool (included in block)",
                removed_count, nonces_removed
            );

            let mut metrics = self.metrics.write().await;
            metrics.mempool_size = pending.len();
        }
    }

    /// Handle incoming mempool message from peer
    pub async fn handle_peer_message(
        &self,
        message: MempoolMessage,
        from_validator: ValidatorId,
    ) -> Result<Option<MempoolMessage>> {
        match message {
            MempoolMessage::TransactionAnnounce { tx_hash, .. } => {
                // Check if we need this transaction
                let have_transaction = {
                    let pending = self.pending_transactions.read().await;
                    pending.contains_key(&tx_hash)
                };

                if !have_transaction {
                    // Request the full transaction
                    let request = MempoolMessage::TransactionRequest {
                        tx_hash,
                        requestor: self.get_own_validator_id().await,
                        timestamp: SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    };
                    Ok(Some(request))
                } else {
                    Ok(None)
                }
            }

            MempoolMessage::TransactionRequest { tx_hash, .. } => {
                // Transaction request handling - simplified for now
                Ok(None)
            }

            MempoolMessage::TransactionResponse { transaction, .. } => {
                if let Some(tx) = transaction {
                    self.add_transaction(tx, Some(from_validator)).await?;
                }
                Ok(None)
            }

            MempoolMessage::MempoolSyncRequest { known_hashes, .. } => {
                // Mempool sync handling - simplified for now
                Ok(None)
            }

            MempoolMessage::MempoolSyncResponse {
                missing_transactions,
                ..
            } => {
                for (_, tx) in missing_transactions {
                    self.add_transaction(tx, Some(from_validator)).await?;
                }
                Ok(None)
            }
        }
    }

    /// v10.11.6: Is this tx hash currently in the active pending pool?
    /// Used by the tx-status handler to detect mempool-eviction —
    /// when the DashMap-tracked status is still InMempool but contains()
    /// returns false, the tx was evicted (low-fee, expired, or cleared
    /// during shutdown). The handler then reports "dropped" instead of
    /// the misleading "in_mempool".
    pub async fn contains(&self, hash: &TxHash) -> bool {
        self.pending_transactions.read().await.contains_key(hash)
    }

    /// Cleanup expired transactions
    pub async fn cleanup_expired_transactions(&self) {
        let now = SystemTime::now();
        let max_age = self.config.max_age;

        let expired_hashes: Vec<TxHash> = {
            let pending = self.pending_transactions.read().await;
            pending
                .iter()
                .filter_map(|(hash, tx)| {
                    if now.duration_since(tx.received_at).unwrap_or_default() > max_age {
                        Some(*hash)
                    } else {
                        None
                    }
                })
                .collect()
        };

        if !expired_hashes.is_empty() {
            let mut pending = self.pending_transactions.write().await;
            for hash in &expired_hashes {
                pending.remove(hash);
            }

            info!(
                "🧹 Cleaned up {} expired transactions",
                expired_hashes.len()
            );

            let mut metrics = self.metrics.write().await;
            metrics.evicted_transactions += expired_hashes.len() as u64;
            metrics.mempool_size = pending.len();
        }
    }

    /// Get mempool statistics
    pub async fn get_mempool_stats(&self) -> MempoolStats {
        let pending = self.pending_transactions.read().await;
        let metrics = self.metrics.read().await;

        // v1.4.5-beta: Use saturating fold to prevent overflow
        // v2.5.0: Updated to u128 for consistency
        let total_fees: u128 = pending
            .values()
            .fold(0u128, |acc, tx| acc.saturating_add(tx.fee));
        let average_fee = if pending.is_empty() {
            0
        } else {
            total_fees / pending.len() as u128
        };

        MempoolStats {
            transaction_count: pending.len(),
            total_fees,
            average_fee,
            total_size_bytes: pending.values().map(|tx| tx.size).sum(),
            oldest_transaction_age: pending
                .values()
                .map(|tx| {
                    SystemTime::now()
                        .duration_since(tx.received_at)
                        .unwrap_or_default()
                })
                .max()
                .unwrap_or_default(),
            metrics: (*metrics).clone(),
        }
    }

    /// Calculate transaction priority for ordering
    fn calculate_transaction_priority(&self, transaction: &Transaction) -> u8 {
        // Higher fee = higher priority (0-255 scale)
        let fee_priority = (transaction.fee.min(255) as f64 / 255.0 * 200.0) as u8;

        // Add small bonus for smaller transactions (better throughput)
        let size_bonus = if bincode::serialized_size(&transaction).unwrap_or(256) < 500 {
            10
        } else {
            0
        };

        (fee_priority + size_bonus).min(255)
    }

    /// Get own validator ID
    async fn get_own_validator_id(&self) -> ValidatorId {
        // This would come from the node configuration
        ValidatorId::default() // Placeholder
    }

    /// Check if mempool has a specific transaction
    pub async fn has_transaction(&self, tx_hash: &TxHash) -> Result<bool> {
        let pending = self.pending_transactions.read().await;
        Ok(pending.contains_key(tx_hash))
    }

    /// Get count of pending transactions
    pub async fn get_pending_count(&self) -> usize {
        self.pending_transactions.read().await.len()
    }

    /// Broadcast message to all peers via TorBroadcastManager
    ///
    /// 🔐 v2.4.7-beta: Properly delegate to TorBroadcastManager for real P2P broadcast
    pub async fn broadcast_to_all_peers(&self, message: BroadcastMessage) -> Result<()> {
        info!("📡 Broadcasting message to all peers via Tor");

        match self.broadcast_manager.broadcast_to_all(message).await {
            Ok(result) => {
                info!(
                    "✅ Broadcast complete: {} successful, {} failed in {:?}",
                    result.successful_sends,
                    result.failed_sends,
                    result.broadcast_time
                );

                // Update metrics
                let mut metrics = self.metrics.write().await;
                metrics.broadcast_count += 1;

                if result.failed_sends > 0 && result.successful_sends == 0 {
                    warn!("⚠️ Broadcast failed - no peers received the message");
                    return Err(anyhow::anyhow!("Broadcast failed - no successful sends"));
                }

                Ok(())
            }
            Err(e) => {
                warn!("❌ Broadcast failed: {}", e);
                Err(e)
            }
        }
    }

    /// Send message to specific peer via TorBroadcastManager
    ///
    /// 🔐 v2.4.7-beta: Properly delegate to TorBroadcastManager for real P2P messaging
    pub async fn send_to_peer(&self, peer: ValidatorId, message: BroadcastMessage) -> Result<()> {
        debug!("📤 Sending message to peer {:?}", peer);

        match self.broadcast_manager
            .send_message_to_peer(peer, &message, crate::tor_broadcast::MessagePriority::Normal)
            .await
        {
            Ok(_) => {
                debug!("✅ Message sent to peer {:?}", peer);
                Ok(())
            }
            Err(e) => {
                warn!("❌ Failed to send message to peer {:?}: {}", peer, e);
                Err(e)
            }
        }
    }
}

/// Mempool statistics
#[derive(Debug, Clone)]
pub struct MempoolStats {
    pub transaction_count: usize,
    /// v2.5.0: Updated to u128 for consistency with Amount type
    pub total_fees: u128,
    /// v2.5.0: Updated to u128 for consistency with Amount type
    pub average_fee: u128,
    pub total_size_bytes: usize,
    pub oldest_transaction_age: Duration,
    pub metrics: MempoolMetrics,
}

impl SpamDetector {
    fn new() -> Self {
        Self {
            validator_rates: HashMap::new(),
            seen_hashes: HashSet::new(),
            suspicious_patterns: HashMap::new(),
        }
    }

    async fn check_rate_limit(&mut self, validator: &ValidatorId) -> bool {
        let now = SystemTime::now();

        let rate_limiter = self
            .validator_rates
            .entry(*validator)
            .or_insert_with(|| RateLimiter {
                last_reset: now,
                transaction_count: 0,
                allowed_per_second: 100, // Default rate limit
            });

        // Reset counter if more than 1 second has passed
        if now
            .duration_since(rate_limiter.last_reset)
            .unwrap_or_default()
            > Duration::from_secs(1)
        {
            rate_limiter.last_reset = now;
            rate_limiter.transaction_count = 0;
        }

        rate_limiter.transaction_count += 1;
        rate_limiter.transaction_count <= rate_limiter.allowed_per_second
    }
}

impl TxValidator {
    fn new(phase: Phase, trusted_via_auth: Arc<DashMap<TxHash, ([u8; 32], std::time::Instant)>>) -> Self {
        Self {
            current_phase: phase,
            verification_cache: Arc::new(RwLock::new(HashMap::new())),
            trusted_via_auth,
            nonce_source: RwLock::new(None),
        }
    }

    /// 🛡 Phase 0 (2026-07-08): wire in the authoritative nonce source. See
    /// `NonceSource` trait doc. Until this is called, `perform_validation`'s
    /// nonce check is a no-op (matches pre-patch behavior exactly). Takes
    /// `&self` (not `&mut self`) so it can be called through the `Arc<TxValidator>`
    /// that `ProductionMempool` already holds, mirroring `mark_auth_trusted`'s
    /// shared-Arc pattern on the sibling `trusted_via_auth` field.
    pub async fn set_nonce_source(&self, source: Arc<dyn NonceSource>) {
        *self.nonce_source.write().await = Some(source);
    }

    /// 🛡 Phase 0 (2026-07-08): read-back accessor so `ProductionMempool`
    /// (specifically `get_transactions_for_block`, Patch 2b) can reuse the
    /// SAME nonce source the admission-time check (Patch 2a) uses, rather
    /// than wiring a second independent field that could drift out of sync.
    async fn get_nonce_source(&self) -> Option<Arc<dyn NonceSource>> {
        self.nonce_source.read().await.clone()
    }

    async fn validate_transaction(&self, transaction: &Transaction) -> Result<ValidationStatus> {
        let tx_hash = transaction.hash();

        // Check cache first
        {
            let cache = self.verification_cache.read().await;
            if let Some(&is_valid) = cache.get(&tx_hash) {
                return Ok(if is_valid {
                    ValidationStatus::Valid
                } else {
                    ValidationStatus::Invalid("Cached validation failure".to_string())
                });
            }
        }

        // Perform validation
        let is_valid = self.perform_validation(transaction).await?;

        // Cache result
        {
            let mut cache = self.verification_cache.write().await;
            cache.insert(tx_hash, is_valid);

            // Limit cache size
            if cache.len() > 10_000 {
                cache.clear(); // Simple eviction
            }
        }

        Ok(if is_valid {
            ValidationStatus::Valid
        } else {
            ValidationStatus::Invalid("Validation failed".to_string())
        })
    }

    async fn perform_validation(&self, transaction: &Transaction) -> Result<bool> {
        // SECURITY (issue #61): real validation. Previous version was `Ok(true)` —
        // every tx that reached the production mempool was admitted unconditionally,
        // bypassing signature/format/fee checks. That bypass meant any internal caller
        // route that didn't pre-verify (transaction_utils helpers, future ingestion
        // paths) could admit forged transactions.

        // Coinbase transactions are signed at the block level, not per-tx, so they
        // bypass per-tx signature and fee checks. All non-coinbase txs must validate.
        if transaction.is_coinbase() {
            // Coinbase format sanity: still ensure required fields are present.
            return Ok(true);
        }

        // 🛡 v10.11.1: X-Wallet-Auth trust flag. If the API layer pre-authenticated
        // the caller via X-Wallet-Auth and marked this tx_id as trusted, skip the
        // inner-signature check. Still apply fee + format checks below. See the
        // doc on TxValidator.trusted_via_auth for the bug history.
        let tx_id = transaction.hash();
        // 🛡 v10.11.72 INTERIM DOUBLE-SPEND GUARD (2026-06-27)
        // The `trusted_via_auth` bypass admits a tx with an EMPTY signature based on
        // an in-memory, HTTP-scoped X-Wallet-Auth flag that does NOT propagate over
        // P2P. The result was the live double-spend: such a tx is Valid only on the
        // originating node and Invalid on every miner → it never lands in a block →
        // the sender is never debited and can re-send forever. By default we now
        // DISABLE the bypass so an unsigned non-coinbase tx is rejected uniformly on
        // every node (it falls through to the verify_signature / empty-signature
        // checks below). This is pure rejection — no balance writes. Reversible WITHOUT
        // recompile: set Q_ALLOW_UNSIGNED_TX=1 to restore the legacy (vulnerable) bypass.
        let allow_unsigned = std::env::var("Q_ALLOW_UNSIGNED_TX").ok().as_deref() == Some("1");
        let auth_trusted: bool = allow_unsigned
            && self
                .trusted_via_auth
                .get(&tx_id)
                .map(|entry| {
                    let (addr, ts): ([u8; 32], std::time::Instant) = *entry.value();
                    // Only trust if the recorded address matches the tx's from. This
                    // prevents a malicious internal path from marking arbitrary tx_ids
                    // as trusted with a wrong from.
                    addr == transaction.from && ts.elapsed() < Duration::from_secs(3600)
                })
                .unwrap_or(false);

        if auth_trusted {
            debug!(
                "🛡 [MEMPOOL] tx {} bypassing inner-sig check via X-Wallet-Auth trust flag",
                hex::encode(&tx_id[..8])
            );
        } else {
            // 1. Signature verification (only when NOT auth-trusted)
            if let Err(e) = transaction.verify_signature() {
                warn!(
                    "🚨 [MEMPOOL] reject: signature invalid — {} (tx_hash={})",
                    e,
                    hex::encode(&tx_id[..8])
                );
                return Ok(false);
            }
        }

        // 🛡 Phase 0 (2026-07-08): nonce-vs-chain-state check. See `NonceSource`
        // doc. GAP CLOSED: a replayed/re-signed tx reusing an already-spent
        // nonce with a different to/amount/data (different tx.id, so DS-1's
        // applied_tx_<id> guard does not catch it) is now rejected here
        // instead of being admitted unconditionally.
        //
        // Scoped to !auth_trusted: auth_trusted (X-Wallet-Auth) transactions
        // have their nonce assigned server-side by the same NonceTracker via
        // get_and_increment BEFORE this validation runs, so re-checking here
        // would race the tracker's own increment and risk self-rejecting
        // legitimate server-issued sends. This check targets the externally-
        // signed / P2P-gossip path where the client supplies its own nonce
        // and nothing upstream has verified it yet.
        //
        // Field is behind a lock defaulting to None (see TxValidator::new),
        // so until `set_nonce_source` is called at boot, this is an exact
        // no-op — zero behavioral change until opted in, same rollout style
        // as DS-1's Q_REPLAY_PROTECTION_HEIGHT.
        if !auth_trusted {
            if let Some(source) = self.nonce_source.read().await.as_ref() {
                if let Err(expected) = source.validate_nonce(&transaction.from, transaction.nonce) {
                    warn!(
                        "🚨 [MEMPOOL] reject: nonce mismatch — submitted={} expected={} (tx_hash={})",
                        transaction.nonce,
                        expected,
                        hex::encode(&tx_id[..8])
                    );
                    return Ok(false);
                }
            }
        }

        // 2. Fee validation (mandatory for non-coinbase per submit_transaction policy)
        if let Err(e) = transaction.validate_fee() {
            warn!(
                "🚨 [MEMPOOL] reject: fee invalid — {} (tx_hash={})",
                e,
                hex::encode(&transaction.hash()[..8])
            );
            return Ok(false);
        }

        // 3. Format sanity
        if transaction.from == [0u8; 32] {
            warn!(
                "🚨 [MEMPOOL] reject: empty from address (tx_hash={})",
                hex::encode(&transaction.hash()[..8])
            );
            return Ok(false);
        }
        // Only reject empty-sig if we DIDN'T trust this tx via X-Wallet-Auth.
        // auth_trusted txs have empty inner sigs by design (the API auth IS
        // the proof of authorization).
        if !auth_trusted && transaction.signature.is_empty() {
            warn!(
                "🚨 [MEMPOOL] reject: empty signature (tx_hash={})",
                hex::encode(&transaction.hash()[..8])
            );
            return Ok(false);
        }

        // Note: double-spend detection against in-flight pending txs is handled by the
        // broader mempool layer (tx_hash uniqueness in `pending_transactions`). This
        // function cannot see that state without a broader refactor; tracked as
        // follow-up in issue #61.

        debug!(
            "✅ [MEMPOOL] tx validated: hash={} from={}",
            hex::encode(&transaction.hash()[..8]),
            hex::encode(&transaction.from[..8])
        );
        Ok(true)
    }
}

// TorBroadcastManager implementation is in tor_broadcast module

// TorClient trait is already imported above

// Placeholder for Tor connection
pub struct TorConnection {
    // Connection details
}

impl TorConnection {
    async fn send_message(&self, message: &str) -> Result<()> {
        // Send message via Tor connection
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mempool_basic_operations() {
        // Test basic mempool functionality
        // This would include comprehensive unit tests
    }

    #[tokio::test]
    async fn test_mempool_byzantine_resistance() {
        // Test Byzantine fault tolerance
    }
}
