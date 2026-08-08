//! # Atomic, double-spend-safe settlement for mixed transfers
//!
//! This module is the missing bridge between the *real* mixing crypto in this
//! crate (CLSAG ring signatures + Pedersen commitments, both over Ristretto255)
//! and an account-based ledger. Everything else in `q-quantum-mixing` produces
//! signatures, commitments and stealth addresses but never moves a coin — and the
//! API layer historically fabricated a "privacy_score" and returned a UUID while
//! the actual transfer happened on a separate, consensus-bypassing, unverified
//! balance write. That is the code path implicated in past unauthorized issuance.
//!
//! [`Settler::verify_and_settle`] closes that gap. It refuses to move any balance
//! until, in order:
//!   1. the amount passes sanity bounds,
//!   2. the CLSAG signature verifies over a message that *binds* the exact
//!      `(sender, recipient, amount, nonce)` tuple (real dalek EC math — no stub),
//!   3. the signed Pedersen commitment re-opens to that exact amount, so a signer
//!      cannot authorize a commitment to X and then settle Y,
//!   4. the sender actually holds `amount + fee`,
//!   5. and the key image is provably unspent.
//! Only then is a single, conservation-checked, all-or-nothing settlement applied
//! that records the key image *in the same atomic step* — so a replay of the exact
//! transfer is rejected as a double-spend rather than minting a second credit.
//!
//! ## What this layer does and does not guarantee (be honest)
//! Guaranteed: no double-spend of a key image; no inflation (u64 amounts, checked
//! arithmetic, deltas that must sum to zero, sufficient-balance precondition);
//! authorization by a valid ring signature; and amount↔commitment↔signature
//! binding. NOT yet guaranteed here: that the ring members are one-time outputs
//! actually owned by `sender` (the stealth-address membership proof), and hidden-
//! from-the-settler amounts (would need a Bulletproofs++ range proof whose
//! generator/blinding is bound to this commitment — `bulletproofs_pp` is real but
//! its `prove()` picks its own blinding, so binding needs a `prove_with_blinding`
//! entry point; tracked as the next step). The settler is trusted with cleartext
//! amounts because it must debit/credit known balances; the commitment still hides
//! amounts from third-party observers of the gossiped mix.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use curve25519_dalek::scalar::Scalar;
use sha3::{Digest, Sha3_512};
use thiserror::Error;

use crate::clsag::{create_pedersen_commitment, CLSAGSignature};
use crate::error::MixingError;

/// Domain separator so a settlement message can never collide with any other
/// signed payload in the system.
const SETTLEMENT_DOMAIN: &[u8] = b"Q-NarwhalKnight.Settlement.v1";

/// Defensive upper bound on a single transfer (atomic units). Prevents a single
/// request from probing overflow edges; well above any legitimate transfer.
pub const MAX_SETTLEMENT_AMOUNT: u64 = 1u64 << 62;

pub type SettlementResult<T> = std::result::Result<T, SettlementError>;

#[derive(Debug, Error)]
pub enum SettlementError {
    #[error("amount out of range: {0}")]
    AmountOutOfRange(u64),

    #[error("ring signature did not verify for the settlement message")]
    InvalidSignature,

    #[error("commitment does not open to the stated amount (binding check failed)")]
    CommitmentMismatch,

    #[error("blinding factor is not a canonical scalar")]
    InvalidBlinding,

    #[error("insufficient funds: account holds {have}, needs {need}")]
    InsufficientFunds { have: u64, need: u64 },

    #[error("double-spend: key image already recorded")]
    DoubleSpend,

    #[error("settlement deltas do not conserve value (sum != 0)")]
    ConservationViolation,

    #[error("balance arithmetic overflow/underflow for account")]
    BalanceArithmetic,

    #[error("crypto layer error: {0}")]
    Crypto(#[from] MixingError),

    #[error("ledger persistence error: {0}")]
    Persistence(String),
}

/// A signed balance change against one account. `delta` is applied as an `i128`
/// so debits and credits share one type; the resulting balance must stay within
/// `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BalanceDelta {
    pub account: [u8; 32],
    pub delta: i128,
}

/// A request to settle exactly one mixed transfer.
///
/// The client is responsible for deriving a *per-transfer one-time key* (e.g.
/// `H(account_secret || nonce)`) so that each distinct transfer has a distinct
/// key image, while a replay of the identical transfer collides on the key image
/// and is rejected. The `nonce` is bound into the signed message, so a signature
/// for one nonce cannot be replayed under another.
#[derive(Debug, Clone)]
pub struct SettlementRequest {
    /// Account to debit (`amount + fee`).
    pub sender: [u8; 32],
    /// Account/stealth address to credit (`amount`).
    pub recipient: [u8; 32],
    /// Transfer amount in atomic units.
    pub amount: u64,
    /// Per-transfer uniqueness; also the derivation input for the one-time key.
    pub nonce: [u8; 32],
    /// Pedersen blinding used to form the signed commitment, revealed to the
    /// settler so it can re-open the commitment against `amount`.
    pub mask: [u8; 32],
    /// CLSAG ring signature authorizing the spend. Carries the ring, the amount
    /// commitment and the key image.
    pub signature: CLSAGSignature,
}

impl SettlementRequest {
    /// Canonical, domain-separated preimage that the CLSAG signature must be over.
    /// Binding all of `(sender, recipient, amount, nonce)` here is what stops a
    /// valid signature from being re-aimed at a different recipient or amount.
    pub fn message(&self) -> Vec<u8> {
        settlement_message(&self.sender, &self.recipient, self.amount, &self.nonce)
    }
}

/// Deterministic settlement message. Exposed so signer and verifier derive the
/// exact same bytes.
pub fn settlement_message(
    sender: &[u8; 32],
    recipient: &[u8; 32],
    amount: u64,
    nonce: &[u8; 32],
) -> Vec<u8> {
    let mut h = Sha3_512::new();
    h.update(SETTLEMENT_DOMAIN);
    h.update(sender);
    h.update(recipient);
    h.update(amount.to_le_bytes());
    h.update(nonce);
    h.finalize().to_vec()
}

/// A record of a successfully applied settlement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementReceipt {
    pub key_image: [u8; 32],
    pub sender: [u8; 32],
    pub recipient: [u8; 32],
    pub amount: u64,
    pub fee: u64,
}

/// The cryptographically verified core of a settlement request — everything that
/// must hold *before* any ledger state is touched. Produced by [`verify_spend`],
/// which performs no ledger I/O, so a caller with its own balance store (e.g. the
/// API server's async, `u128`, RocksDB-backed ledger) can run the exact same crypto
/// checks and then apply the transfer atomically itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedSpend {
    /// Must be recorded as spent atomically with the balance move to prevent
    /// double-spend / replay.
    pub key_image: [u8; 32],
    pub sender: [u8; 32],
    pub recipient: [u8; 32],
    pub amount: u64,
}

/// Verify the cryptography of a settlement request WITHOUT touching any ledger:
/// amount sanity, the LSAG ring signature over the bound message, and the Pedersen
/// commitment re-opening to the stated amount. Returns the validated spend (incl.
/// the key image) on success. This is the single reusable check shared by the
/// in-crate [`Settler`] and any external ledger, so every settlement path — the
/// reference here and the API server's real one — is guarded identically.
pub fn verify_spend(req: &SettlementRequest) -> SettlementResult<VerifiedSpend> {
    // 1. Amount sanity — reject zero and anything near the overflow edge.
    if req.amount == 0 || req.amount > MAX_SETTLEMENT_AMOUNT {
        return Err(SettlementError::AmountOutOfRange(req.amount));
    }

    // 2. Authorization: the ring signature must verify over a message that binds
    //    this exact (sender, recipient, amount, nonce). Real dalek EC ring-closure.
    let message = req.message();
    if !req.signature.verify(&message)? {
        return Err(SettlementError::InvalidSignature);
    }

    // 3. Amount<->commitment binding: re-open the signed Pedersen commitment with
    //    the revealed mask; it must equal the commitment the ring signed.
    let mask = Scalar::from_canonical_bytes(req.mask)
        .into_option()
        .ok_or(SettlementError::InvalidBlinding)?;
    let (expected_commitment, _point) = create_pedersen_commitment(req.amount, &mask);
    if expected_commitment != req.signature.commitment {
        return Err(SettlementError::CommitmentMismatch);
    }

    Ok(VerifiedSpend {
        key_image: *req.signature.get_key_image(),
        sender: req.sender,
        recipient: req.recipient,
        amount: req.amount,
    })
}

/// The ledger a settlement writes to.
///
/// Implementations MUST make [`SettlementLedger::commit_spend`] atomic and
/// crash-consistent: recording the key image and applying every delta is a single
/// all-or-nothing operation, and a key image that already exists must be rejected
/// with [`SettlementError::DoubleSpend`] *without* applying any delta. In the API
/// server this is a single RocksDB `WriteBatch`; the in-memory default below uses
/// one mutex-guarded critical section.
pub trait SettlementLedger: Send + Sync {
    /// Current balance of `account` (0 if unknown).
    fn balance(&self, account: &[u8; 32]) -> SettlementResult<u64>;

    /// Atomically: reject if `key_image` is already spent; otherwise verify the
    /// deltas conserve value and keep every balance within `u64`, then apply them
    /// and record the key image — all-or-nothing.
    fn commit_spend(&self, key_image: [u8; 32], deltas: &[BalanceDelta]) -> SettlementResult<()>;
}

/// The verifier/settler. Generic over the ledger so the crate stays free of any
/// database dependency; the API server supplies a RocksDB-backed implementation.
pub struct Settler<L: SettlementLedger> {
    ledger: L,
    fee_account: [u8; 32],
    mixing_fee: u64,
}

impl<L: SettlementLedger> Settler<L> {
    pub fn new(ledger: L, fee_account: [u8; 32], mixing_fee: u64) -> Self {
        Self { ledger, fee_account, mixing_fee }
    }

    pub fn ledger(&self) -> &L {
        &self.ledger
    }

    pub fn mixing_fee(&self) -> u64 {
        self.mixing_fee
    }

    /// The whole point of the module: verify all crypto and preconditions, then
    /// settle atomically, or change nothing at all. Synchronous by design — CLSAG
    /// verification is CPU-bound and synchronous, and keeping the critical section
    /// lock-free of `.await` removes a whole class of TOCTOU races.
    pub fn verify_and_settle(&self, req: &SettlementRequest) -> SettlementResult<SettlementReceipt> {
        // Steps 1-3: all cryptography (amount sanity, ring signature, commitment
        // binding). No ledger I/O — identical to what the API server's real ledger
        // runs, so both paths are guarded the same way.
        let verified = verify_spend(req)?;

        // 4. Funding precondition. `total` is checked so `amount + fee` cannot wrap.
        let total = verified
            .amount
            .checked_add(self.mixing_fee)
            .ok_or(SettlementError::BalanceArithmetic)?;
        let have = self.ledger.balance(&verified.sender)?;
        if have < total {
            return Err(SettlementError::InsufficientFunds { have, need: total });
        }

        // 5. Build the conservation-balanced delta set: debit sender total,
        //    credit recipient the amount, credit the fee account the fee.
        let deltas = [
            BalanceDelta { account: verified.sender, delta: -(total as i128) },
            BalanceDelta { account: verified.recipient, delta: verified.amount as i128 },
            BalanceDelta { account: self.fee_account, delta: self.mixing_fee as i128 },
        ];
        // Defense in depth: the ledger re-checks this, but never hand it a set
        // that does not conserve.
        if deltas.iter().map(|d| d.delta).sum::<i128>() != 0 {
            return Err(SettlementError::ConservationViolation);
        }

        // 6. Atomic commit. The key image is recorded in the same step, so a
        //    concurrent or replayed identical transfer fails here as DoubleSpend
        //    rather than applying a second credit.
        self.ledger.commit_spend(verified.key_image, &deltas)?;

        Ok(SettlementReceipt {
            key_image: verified.key_image,
            sender: verified.sender,
            recipient: verified.recipient,
            amount: verified.amount,
            fee: self.mixing_fee,
        })
    }
}

// ---------------------------------------------------------------------------
// Default in-memory ledger (also the reference semantics for any real backend).
// ---------------------------------------------------------------------------

#[derive(Default)]
struct LedgerState {
    balances: HashMap<[u8; 32], u64>,
    spent: HashSet<[u8; 32]>,
}

/// A mutex-guarded, optionally file-backed ledger. Correct and crash-consistent
/// for a single process; the API server should instead implement
/// [`SettlementLedger`] over its RocksDB store with a `WriteBatch` for durability
/// across restarts. Persistence here is a bincode snapshot written after each
/// successful commit while the lock is held.
pub struct InMemoryLedger {
    state: Mutex<LedgerState>,
    wal: Option<PathBuf>,
}

impl Default for InMemoryLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryLedger {
    pub fn new() -> Self {
        Self { state: Mutex::new(LedgerState::default()), wal: None }
    }

    /// Snapshot-persisting variant. Best-effort restore from `path` if present.
    pub fn with_persistence(path: PathBuf) -> Self {
        let state = std::fs::read(&path)
            .ok()
            .and_then(|bytes| bincode::deserialize::<PersistShape>(&bytes).ok())
            .map(|p| LedgerState {
                balances: p.balances.into_iter().collect(),
                spent: p.spent.into_iter().collect(),
            })
            .unwrap_or_default();
        Self { state: Mutex::new(state), wal: Some(path) }
    }

    /// Seed an initial balance (setup/testing).
    pub fn seed(&self, account: [u8; 32], amount: u64) {
        let mut st = self.state.lock().expect("ledger mutex poisoned");
        st.balances.insert(account, amount);
    }

    /// Number of recorded (spent) key images — for assertions/metrics.
    pub fn spent_count(&self) -> usize {
        self.state.lock().expect("ledger mutex poisoned").spent.len()
    }

    fn persist_locked(&self, st: &LedgerState) -> SettlementResult<()> {
        let Some(path) = &self.wal else { return Ok(()) };
        let shape = PersistShape {
            balances: st.balances.iter().map(|(k, v)| (*k, *v)).collect(),
            spent: st.spent.iter().copied().collect(),
        };
        let bytes = bincode::serialize(&shape)
            .map_err(|e| SettlementError::Persistence(e.to_string()))?;
        // Write to a temp sibling then rename for atomic-ish replacement.
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| SettlementError::Persistence(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| SettlementError::Persistence(e.to_string()))?;
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PersistShape {
    balances: Vec<([u8; 32], u64)>,
    spent: Vec<[u8; 32]>,
}

impl SettlementLedger for InMemoryLedger {
    fn balance(&self, account: &[u8; 32]) -> SettlementResult<u64> {
        let st = self.state.lock().expect("ledger mutex poisoned");
        Ok(st.balances.get(account).copied().unwrap_or(0))
    }

    fn commit_spend(&self, key_image: [u8; 32], deltas: &[BalanceDelta]) -> SettlementResult<()> {
        let mut st = self.state.lock().expect("ledger mutex poisoned");

        // Double-spend guard — the atomic reject.
        if st.spent.contains(&key_image) {
            return Err(SettlementError::DoubleSpend);
        }

        // Conservation.
        if deltas.iter().map(|d| d.delta).sum::<i128>() != 0 {
            return Err(SettlementError::ConservationViolation);
        }

        // Compute every resulting balance first; only apply if ALL are valid, so a
        // failure leaves state untouched (all-or-nothing).
        let mut next: HashMap<[u8; 32], u64> = HashMap::new();
        for d in deltas {
            let current = *next
                .get(&d.account)
                .or_else(|| st.balances.get(&d.account))
                .unwrap_or(&0);
            let updated = (current as i128)
                .checked_add(d.delta)
                .ok_or(SettlementError::BalanceArithmetic)?;
            if updated < 0 || updated > u64::MAX as i128 {
                return Err(SettlementError::BalanceArithmetic);
            }
            next.insert(d.account, updated as u64);
        }

        // Commit: apply balances + record key image, all under the same lock.
        for (account, value) in next {
            st.balances.insert(account, value);
        }
        st.spent.insert(key_image);

        self.persist_locked(&st)?;
        Ok(())
    }
}

