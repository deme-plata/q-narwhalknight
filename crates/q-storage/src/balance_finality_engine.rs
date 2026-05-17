//! Bracha Reliable Broadcast engine for BFT-safe balance finalization.
//!
//! Implements the three-phase Bracha RB protocol over the DAG-Knight round clock:
//!   SEND → ECHO (2f+1 identical echoes) → READY (f+1 amplify / 2f+1 deliver)
//!
//! Phase 1 shadow mode: f=0 (all-honest), quorums = 1. Promotion to f=1 when
//! the validator set reaches 4+ nodes is done by bumping `f` without a hard fork.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use q_types::{
    balance_finality::{
        BalanceFinalityRecord, BrachaBalanceMsg, BrachaInstance, BrachaPhase,
        ValidatorBitmask, ANCHOR_FLUSH_SECS, BRACHA_PROPOSAL_TIMEOUT_ROUNDS, MAX_ANCHOR_BATCH,
    },
    balance_update::P2PBalanceUpdate,
};

use crate::StorageEngine;

/// Returned to callers of `handle_bracha_msg` describing what happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandleResult {
    /// Message accepted; no threshold reached yet.
    Accepted,
    /// We relayed an ECHO (because we just received a SEND).
    EchoSent,
    /// We relayed a READY (because 2f+1 ECHOs or f+1 READYs arrived).
    ReadySent,
    /// 2f+1 READYs received → balance update delivered and written to DB.
    Delivered { broadcast_id: [u8; 32] },
    /// Message rejected (duplicate, tampered value, bad sig, out of round window, etc.)
    Rejected(String),
}

/// Owns the per-broadcast Bracha state and the per-node validator-index map.
///
/// The engine is cloned cheaply (all heavy data behind Arc) and injected into
/// `AppState`. The gossipsub handler calls `handle_bracha_msg` on every incoming
/// `/consensus/balance-rb` message; the balance-path entry points call
/// `propose_balance_update` instead of writing directly to the DB.
pub struct BalanceFinalityEngine {
    /// f in the Bracha protocol: max Byzantine faults tolerated.
    /// Phase 1: f=0 (shadow mode — no Byzantines assumed, quorums = 1).
    /// Phase 2: f=1 (4-node network, 2f+1=3, f+1=2).
    pub f: usize,

    /// Per-broadcast state keyed by broadcast_id.
    instances: Arc<Mutex<HashMap<[u8; 32], BrachaInstance>>>,

    /// Validator public key → bitmask index.
    validator_index: Arc<RwLock<HashMap<[u8; 32], u8>>>,

    /// Our own validator index (255 = not in set / observer mode).
    our_index: Arc<std::sync::atomic::AtomicU8>,

    /// Our own Ed25519 signing key (32-byte seed). `None` when running as non-validator observer.
    our_signing_key: Option<Arc<ed25519_dalek::SigningKey>>,

    /// Finalized-but-not-yet-anchored records. Drained into a DAG vertex every ANCHOR_FLUSH_SECS.
    pub pending_anchor: Arc<Mutex<Vec<BalanceFinalityRecord>>>,

    /// Storage engine for writing finality proofs.
    storage: Arc<StorageEngine>,

    /// Channel for sending Bracha messages to gossipsub.
    /// Payload: CBOR-serialized BrachaBalanceMsg.
    gossip_tx: Option<tokio::sync::mpsc::UnboundedSender<(String, Vec<u8>)>>,

    /// Network topic string (e.g. `/qnk/mainnet-genesis/consensus/balance-rb`).
    rb_topic: String,
}

impl BalanceFinalityEngine {
    pub fn new(
        f: usize,
        storage: Arc<StorageEngine>,
        our_signing_key: Option<Arc<ed25519_dalek::SigningKey>>,
        gossip_tx: Option<tokio::sync::mpsc::UnboundedSender<(String, Vec<u8>)>>,
        rb_topic: String,
    ) -> Self {
        Self {
            f,
            instances: Arc::new(Mutex::new(HashMap::new())),
            validator_index: Arc::new(RwLock::new(HashMap::new())),
            our_index: Arc::new(std::sync::atomic::AtomicU8::new(255)),
            our_signing_key,
            pending_anchor: Arc::new(Mutex::new(Vec::new())),
            storage,
            gossip_tx,
            rb_topic,
        }
    }

    // ── Threshold helpers ─────────────────────────────────────────────────────

    /// 2f+1 (ECHO threshold to trigger READY, READY threshold to deliver).
    fn echo_quorum(&self) -> u32 {
        (2 * self.f + 1) as u32
    }

    /// f+1 (READY amplification threshold).
    fn ready_amplify(&self) -> u32 {
        (self.f + 1) as u32
    }

    // ── Validator index management ────────────────────────────────────────────

    /// Update the validator set. Called at startup and whenever the on-chain set changes.
    /// Assigns each known validator pubkey a stable bitmask index.
    pub async fn update_validator_set(&self, validators: Vec<[u8; 32]>, our_pubkey: &[u8; 32]) {
        let mut map = self.validator_index.write().await;
        map.clear();
        for (i, pk) in validators.iter().enumerate() {
            if i < q_types::balance_finality::MAX_VALIDATORS {
                map.insert(*pk, i as u8);
            }
        }
        let idx = map.get(our_pubkey).copied().unwrap_or(255);
        self.our_index.store(idx, std::sync::atomic::Ordering::Relaxed);
        info!(
            "BalanceFinalityEngine: validator set updated ({} validators, our_index={})",
            map.len(),
            idx
        );
    }

    fn our_index(&self) -> u8 {
        self.our_index.load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn sender_index_for(&self, pubkey: &[u8; 32]) -> Option<u8> {
        self.validator_index.read().await.get(pubkey).copied()
    }

    // ── Broadcast ID computation ──────────────────────────────────────────────

    /// BLAKE3(wallet_addr || amount_le || dag_round || nonce)
    pub fn compute_broadcast_id(update: &P2PBalanceUpdate, dag_round: u64) -> [u8; 32] {
        use blake3::Hasher;
        let mut h = Hasher::new();
        if let Ok(addr_bytes) = hex::decode(&update.wallet_address) {
            h.update(&addr_bytes);
        } else {
            h.update(update.wallet_address.as_bytes());
        }
        h.update(&update.amount.to_le_bytes());
        h.update(&dag_round.to_le_bytes());
        h.update(&update.nonce.to_le_bytes());
        *h.finalize().as_bytes()
    }

    // ── Entry point: proposing a balance update ───────────────────────────────

    /// Called by every non-block balance path (mining rewards, DEX credits, etc.)
    /// instead of writing directly to the DB.
    ///
    /// Constructs and gossips a SEND message. Returns the broadcast_id so callers
    /// can correlate the eventual delivery event.
    pub async fn propose_balance_update(
        &self,
        update: P2PBalanceUpdate,
        current_dag_round: u64,
    ) -> anyhow::Result<[u8; 32]> {
        let broadcast_id = Self::compute_broadcast_id(&update, current_dag_round);
        let our_idx = self.our_index();

        let signing_key = match &self.our_signing_key {
            Some(sk) => sk.clone(),
            None => {
                return Err(anyhow::anyhow!(
                    "BalanceFinalityEngine: cannot propose — not a validator (no signing key)"
                ));
            }
        };

        let sender_pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();

        let mut msg = BrachaBalanceMsg {
            start_round: current_dag_round,
            broadcast_id,
            update,
            phase: BrachaPhase::Send,
            sender_index: our_idx,
            sender_pubkey,
            signature: [0u8; 64],
        };
        msg.signature = {
            use ed25519_dalek::Signer;
            let payload = msg.signing_message();
            signing_key.sign(&payload).to_bytes()
        };

        // Record locally as if we received our own SEND.
        {
            let mut instances = self.instances.lock().await;
            let inst = instances
                .entry(broadcast_id)
                .or_insert_with(|| BrachaInstance::new(current_dag_round));
            inst.send_msg = Some(msg.clone());
        }

        // Gossip the SEND.
        self.gossip(&msg).await;

        info!(
            "BalanceFinalityEngine: SEND broadcast_id={} wallet={} amount={}",
            hex::encode(&broadcast_id[..8]),
            &msg.update.wallet_address[..16.min(msg.update.wallet_address.len())],
            msg.update.amount,
        );

        Ok(broadcast_id)
    }

    // ── Core Bracha message handler ───────────────────────────────────────────

    /// Called by the gossipsub handler for every message on the balance-rb topic.
    /// Returns what action was taken (for logging/metrics).
    pub async fn handle_bracha_msg(
        &self,
        msg: BrachaBalanceMsg,
        current_dag_round: u64,
    ) -> HandleResult {
        // 1. Round-window check.
        if !msg.in_round_window(current_dag_round) {
            return HandleResult::Rejected(format!(
                "outside round window: start_round={} current={}",
                msg.start_round, current_dag_round
            ));
        }

        // 2. Signature check.
        if !msg.verify_signature() {
            return HandleResult::Rejected("invalid signature".into());
        }

        // 3. Validate sender_index matches claimed pubkey.
        match self.sender_index_for(&msg.sender_pubkey).await {
            Some(idx) if idx == msg.sender_index => {}
            Some(idx) => {
                return HandleResult::Rejected(format!(
                    "sender_index mismatch: claimed {} actual {}",
                    msg.sender_index, idx
                ));
            }
            None => {
                // Unknown sender — could be a non-validator gossip relay.
                // Accept in f=0 shadow mode; reject in f≥1.
                if self.f > 0 {
                    return HandleResult::Rejected("sender not in validator set".into());
                }
            }
        }

        let bid = msg.broadcast_id;

        match msg.phase {
            BrachaPhase::Send => self.handle_send(msg, current_dag_round).await,
            BrachaPhase::Echo => self.handle_echo(msg, bid, current_dag_round).await,
            BrachaPhase::Ready => self.handle_ready(msg, bid, current_dag_round).await,
        }
    }

    async fn handle_send(&self, msg: BrachaBalanceMsg, current_dag_round: u64) -> HandleResult {
        let bid = msg.broadcast_id;
        let mut instances = self.instances.lock().await;
        let inst = instances
            .entry(bid)
            .or_insert_with(|| BrachaInstance::new(current_dag_round));

        if inst.send_msg.is_some() {
            return HandleResult::Rejected("duplicate SEND".into());
        }
        inst.send_msg = Some(msg.clone());

        if inst.echoed {
            return HandleResult::Accepted;
        }
        inst.echoed = true;

        // Build and gossip our ECHO.
        let echo = match self.build_echo(&msg) {
            Ok(e) => e,
            Err(e) => return HandleResult::Rejected(format!("cannot build ECHO: {e}")),
        };
        drop(instances);

        self.gossip(&echo).await;
        // Process our own ECHO immediately.
        let _ = Box::pin(self.handle_echo(echo, bid, current_dag_round)).await;

        HandleResult::EchoSent
    }

    async fn handle_echo(
        &self,
        msg: BrachaBalanceMsg,
        bid: [u8; 32],
        current_dag_round: u64,
    ) -> HandleResult {
        let mut instances = self.instances.lock().await;
        let inst = instances
            .entry(bid)
            .or_insert_with(|| BrachaInstance::new(current_dag_round));

        // Cross-phase value validation: if we already have the SEND, the ECHO
        // must carry the same update (prevents Byzantine SEND echo-stuffing).
        if let Some(ref send) = inst.send_msg.clone() {
            if !msg.update_matches(send) {
                return HandleResult::Rejected("ECHO value does not match SEND".into());
            }
        }

        // Deduplicate by sender_index.
        if inst.echo_mask.has(msg.sender_index) {
            return HandleResult::Accepted;
        }
        inst.echo_mask.set(msg.sender_index);

        let echo_count = inst.echo_mask.count();
        debug!(
            "ECHO bid={} count={}/{}",
            hex::encode(&bid[..4]),
            echo_count,
            self.echo_quorum()
        );

        if echo_count >= self.echo_quorum() && !inst.ready_sent {
            inst.ready_sent = true;
            let send_msg = inst.send_msg.clone();
            drop(instances);

            // Build and gossip READY.
            if let Some(ref sm) = send_msg {
                let ready = match self.build_ready(sm) {
                    Ok(r) => r,
                    Err(e) => return HandleResult::Rejected(format!("cannot build READY: {e}")),
                };
                self.gossip(&ready).await;
                let _ = Box::pin(self.handle_ready(ready, bid, current_dag_round)).await;
                return HandleResult::ReadySent;
            }
        }

        HandleResult::Accepted
    }

    async fn handle_ready(
        &self,
        msg: BrachaBalanceMsg,
        bid: [u8; 32],
        current_dag_round: u64,
    ) -> HandleResult {
        let mut instances = self.instances.lock().await;
        let inst = instances
            .entry(bid)
            .or_insert_with(|| BrachaInstance::new(current_dag_round));

        if inst.delivered {
            return HandleResult::Accepted; // Already delivered, ignore.
        }

        if inst.ready_mask.has(msg.sender_index) {
            return HandleResult::Accepted;
        }
        inst.ready_mask.set(msg.sender_index);

        let ready_count = inst.ready_mask.count();
        debug!(
            "READY bid={} count={}/{} (amplify={}, deliver={})",
            hex::encode(&bid[..4]),
            ready_count,
            self.echo_quorum(),
            self.ready_amplify(),
            self.echo_quorum(),
        );

        // Amplification: f+1 READYs → we also send READY (if we haven't already).
        if ready_count >= self.ready_amplify() && !inst.ready_sent {
            inst.ready_sent = true;
            let send_msg = inst.send_msg.clone();
            drop(instances);

            if let Some(ref sm) = send_msg {
                let ready = match self.build_ready(sm) {
                    Ok(r) => r,
                    Err(e) => return HandleResult::Rejected(format!("amplify READY build: {e}")),
                };
                self.gossip(&ready).await;
                // Re-enter to process our own READY.
                return Box::pin(self.handle_ready(ready, bid, current_dag_round)).await;
            }
            return HandleResult::ReadySent;
        }

        // Delivery: 2f+1 READYs → finalize.
        if ready_count >= self.echo_quorum() {
            inst.delivered = true;
            let update = msg.update.clone();
            let ready_mask = inst.ready_mask;
            let send_msg = inst.send_msg.clone();
            let created_round = inst.created_round;
            drop(instances);

            let wallet_addr = if let Ok(b) = hex::decode(&update.wallet_address) {
                if b.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&b);
                    arr
                } else {
                    return HandleResult::Rejected("invalid wallet address length".into());
                }
            } else {
                return HandleResult::Rejected("invalid wallet address hex".into());
            };

            let record = BalanceFinalityRecord {
                wallet_address: wallet_addr,
                new_balance: update.new_balance,
                dag_round: created_round,
                dag_vertex_hash: None,
                broadcast_id: bid,
                finalized_at_height: update.block_height,
                ready_witness_mask: ready_mask,
                finalized_ts: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            // Write finality proof to DB.
            if let Err(e) = self.write_finality_record(&record).await {
                error!("BalanceFinalityEngine: DB write failed: {e}");
            }

            // Update the wallet balance in DB.
            if let Err(e) = self.storage.save_wallet_balance(&wallet_addr, update.new_balance).await {
                error!("BalanceFinalityEngine: balance write failed: {e}");
            }

            // Park in pending_anchor for DAG vertex inclusion.
            {
                let mut pa = self.pending_anchor.lock().await;
                pa.push(record);
            }

            info!(
                "BalanceFinalityEngine: DELIVERED bid={} wallet={} new_balance={}",
                hex::encode(&bid[..8]),
                &update.wallet_address[..16.min(update.wallet_address.len())],
                update.new_balance,
            );

            return HandleResult::Delivered { broadcast_id: bid };
        }

        HandleResult::Accepted
    }

    // ── Background tasks ──────────────────────────────────────────────────────

    /// Spawn the anchor-flush and timeout-cleanup background tasks.
    /// Must be called once after construction.
    pub fn spawn_background_tasks(self: Arc<Self>) {
        let engine = self.clone();
        tokio::spawn(async move {
            engine.anchor_flush_loop().await;
        });

        let engine2 = self.clone();
        tokio::spawn(async move {
            engine2.timeout_cleanup_loop().await;
        });
    }

    /// Every ANCHOR_FLUSH_SECS, if pending_anchor is non-empty, produce an
    /// anchor-only DAG vertex (stub — DAG-Knight integration fills this in).
    /// Bounded anchor flush: after ANCHOR_FLUSH_SECS, any delivered-but-not-anchored records
    /// that the DAG-Knight producer hasn't drained get a synthetic anchor hash stamped and
    /// persisted. This prevents unbounded memory growth during low-mining-activity periods.
    async fn anchor_flush_loop(&self) {
        let interval = Duration::from_secs(ANCHOR_FLUSH_SECS);
        loop {
            tokio::time::sleep(interval).await;

            let batch = self.drain_pending_anchor().await;
            if batch.is_empty() {
                continue;
            }

            info!(
                "BalanceFinalityEngine: anchor flush — {} records not drained by DAG-Knight, stamping synthetic anchor",
                batch.len()
            );

            // Synthetic anchor hash = BLAKE3(timestamp || broadcast_ids)
            let synthetic_hash: [u8; 32] = {
                use blake3::Hasher;
                let mut h = Hasher::new();
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                h.update(&ts.to_le_bytes());
                for rec in &batch {
                    h.update(&rec.broadcast_id);
                }
                *h.finalize().as_bytes()
            };

            self.stamp_anchor_hash(&batch, synthetic_hash).await;
            info!(
                "BalanceFinalityEngine: stamped {} records with synthetic anchor {}",
                batch.len(),
                hex::encode(&synthetic_hash[..8])
            );
        }
    }

    /// Every ~2 seconds, drop stalled Bracha instances that have exceeded the
    /// proposal timeout (50 rounds × 100ms ≈ 5 seconds).
    async fn timeout_cleanup_loop(&self) {
        let interval = Duration::from_millis(2_000);
        loop {
            tokio::time::sleep(interval).await;
            // We need a proxy for current_round — use wall clock approximation
            // (100ms per round → elapsed_secs × 10).
            let approx_round = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
                / 100;

            let mut instances = self.instances.lock().await;
            let before = instances.len();
            instances.retain(|_, inst| {
                !inst.is_timed_out(approx_round) || inst.delivered
            });
            let dropped = before.saturating_sub(instances.len());
            if dropped > 0 {
                debug!("BalanceFinalityEngine: timed out {} stalled proposals", dropped);
            }
        }
    }

    // ── DAG-Knight integration ────────────────────────────────────────────────

    /// Drain up to MAX_ANCHOR_BATCH pending records for inclusion in a DAG vertex.
    /// Called by the DAG-Knight block producer before signing a vertex.
    pub async fn drain_pending_anchor(&self) -> Vec<BalanceFinalityRecord> {
        let mut pa = self.pending_anchor.lock().await;
        let n = pa.len().min(MAX_ANCHOR_BATCH);
        if n == 0 {
            return Vec::new();
        }
        pa.drain(..n).collect()
    }

    /// Called after a DAG vertex is produced and its hash is known.
    /// Stamps each included finality record with the vertex hash in the DB.
    pub async fn stamp_anchor_hash(
        &self,
        records: &[BalanceFinalityRecord],
        vertex_hash: [u8; 32],
    ) {
        for rec in records {
            let mut stamped = rec.clone();
            stamped.dag_vertex_hash = Some(vertex_hash);
            if let Err(e) = self.write_finality_record(&stamped).await {
                error!("BalanceFinalityEngine: stamp_anchor_hash DB error: {e}");
            }
        }
    }

    // ── State sync API ────────────────────────────────────────────────────────

    /// Returns all anchored finality records from the DB for the state-sync endpoint.
    pub async fn load_anchored_records(&self) -> anyhow::Result<Vec<BalanceFinalityRecord>> {
        let prefix = b"balance_finality_proof:";
        let entries = self
            .storage
            .scan_manifest_prefix(prefix)
            .await?;

        let mut out = Vec::new();
        for (_k, v) in entries {
            match BalanceFinalityRecord::from_cbor(&v) {
                Ok(r) => out.push(r),
                Err(e) => warn!("BalanceFinalityEngine: corrupt finality record: {e}"),
            }
        }
        Ok(out)
    }

    /// Returns records delivered but not yet anchored into a DAG vertex.
    pub async fn pending_anchor_snapshot(&self) -> Vec<BalanceFinalityRecord> {
        self.pending_anchor.lock().await.clone()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn build_echo(&self, send: &BrachaBalanceMsg) -> anyhow::Result<BrachaBalanceMsg> {
        self.build_phase_msg(send, BrachaPhase::Echo)
    }

    fn build_ready(&self, send: &BrachaBalanceMsg) -> anyhow::Result<BrachaBalanceMsg> {
        self.build_phase_msg(send, BrachaPhase::Ready)
    }

    fn build_phase_msg(
        &self,
        template: &BrachaBalanceMsg,
        phase: BrachaPhase,
    ) -> anyhow::Result<BrachaBalanceMsg> {
        let sk = self.our_signing_key.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no signing key — cannot build Bracha phase message")
        })?;
        let sender_pubkey: [u8; 32] = sk.verifying_key().to_bytes();
        let mut msg = BrachaBalanceMsg {
            start_round: template.start_round,
            broadcast_id: template.broadcast_id,
            update: template.update.clone(),
            phase,
            sender_index: self.our_index(),
            sender_pubkey,
            signature: [0u8; 64],
        };
        use ed25519_dalek::Signer;
        let payload = msg.signing_message();
        msg.signature = sk.sign(&payload).to_bytes();
        Ok(msg)
    }

    async fn gossip(&self, msg: &BrachaBalanceMsg) {
        let tx = match &self.gossip_tx {
            Some(t) => t,
            None => return,
        };
        match msg.to_cbor() {
            Ok(bytes) => {
                let _ = tx.send((self.rb_topic.clone(), bytes));
            }
            Err(e) => error!("BalanceFinalityEngine: CBOR serialization failed: {e}"),
        }
    }

    async fn write_finality_record(&self, record: &BalanceFinalityRecord) -> anyhow::Result<()> {
        let key = BalanceFinalityRecord::db_key(&record.wallet_address);
        let value = record.to_cbor()
            .map_err(|e| anyhow::anyhow!("CBOR encode: {e}"))?;
        self.storage
            .put_manifest_sync(key.as_bytes(), &value)
            .await
    }
}
