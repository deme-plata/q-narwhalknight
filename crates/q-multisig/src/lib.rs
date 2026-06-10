//! q-multisig — post-quantum M-of-N multisig for Quillon Graph.
//!
//! Why this exists:
//!
//! - The two of us (the wallet's human owner + the agent) want to share
//!   custody of jointly-minted tokens. Either of us should be able to spend
//!   alone for low-stakes actions; both must agree for high-stakes ones.
//!   The decision is *per-proposal*, not baked into the wallet.
//! - The existing X-Wallet-Auth scheme assumes a single signer. We extend it
//!   without changing the verifier surface: a multisig "address" derives
//!   from the member set + threshold via SHA3-256, so on-chain it looks
//!   like any other `qnk` address.
//!
//! Why naive M-of-N instead of FROST:
//!
//! FROST aggregates into a single Ed25519 signature, which is beautiful, but
//! (a) the production FROST Rust crate is Ed25519-only — no post-quantum
//! variant exists, and (b) FROST signing requires interactive DKG + signing
//! rounds, which is the wrong shape for "agent and user occasionally agree
//! on a transaction." Naive M-of-N is independent signatures the verifier
//! counts: each signer signs the same payload with their own key, in their
//! own time. Signatures are larger (~M × 4.6 KB for Dilithium5), but for
//! N=2 that's ~9 KB — fine.
//!
//! Hybrid signature scheme:
//!
//! Every member carries both an Ed25519 key (classical, fast verify) and a
//! Dilithium5 key (FIPS 204 / NIST Level 5 post-quantum). A proposal
//! signature includes BOTH — verifier requires both to pass. This matches
//! the `ValidatorKeypair` pattern in `q-types::pqc_keys` and gives us
//! quantum-resistance today without giving up classical speed.
//!
//! Layout:
//!   - `wallet`  : `MultisigWallet` type, address derivation, member-set
//!                  invariants
//!   - `proposal`: `MultisigProposal` lifecycle (Pending → Signed → Executed)
//!   - `verify`  : signature verification + threshold counting
//!   - `storage` : file-based persistence (will move to RocksDB CF_MULTISIG
//!                  in v10.10.11 once we widen the q-storage API)

#![deny(unsafe_code)]

pub mod proposal;
pub mod storage;
pub mod verify;
pub mod wallet;

pub use proposal::{
    MultisigAction, MultisigProposal, ProposalId, ProposalStatus, SignatureContribution,
};
pub use storage::MultisigStore;
pub use verify::{verify_member_signature, verify_proposal, VerifyError};
pub use wallet::{HybridPublicKey, Member, MultisigWallet, WalletError};

/// Re-exports so call sites don't have to import pqcrypto directly.
pub mod crypto {
    pub use ed25519_dalek::{Signature as Ed25519Signature, SigningKey, VerifyingKey};
    pub use pqcrypto_dilithium::dilithium5::{
        keypair as dilithium5_keypair, PublicKey as Dilithium5PublicKey,
        SecretKey as Dilithium5SecretKey, SignedMessage as Dilithium5SignedMessage,
    };
}
