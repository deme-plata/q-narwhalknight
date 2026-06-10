#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! ed25519-dalek = { version = "2", features = ["rand_core"] }
//! rand = "0.8"
//! bincode = "1"
//! rmp-serde = "1"
//! serde = { version = "1", features = ["derive"] }
//! chrono = { version = "0.4", features = ["serde"] }
//! sha3 = "0.10"
//! ```

//! Q-NarwhalKnight Realistic TPS Benchmark
//!
//! Measures REAL transaction throughput with actual Ed25519 cryptography,
//! serialization, and validation. No simulated delays, no multipliers,
//! no inflated numbers.
//!
//! Reports honest, measured metrics:
//!   - Ed25519 signature creation throughput
//!   - Ed25519 signature verification throughput
//!   - Bincode serialization round-trip throughput
//!   - MessagePack serialization round-trip throughput
//!   - Combined full-pipeline throughput (sign + serialize + deserialize + verify)

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey, Signature};
use rand::rngs::OsRng;
use serde::{Serialize, Deserialize};
use sha3::{Sha3_256, Digest};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Transaction type that mirrors q_types::Transaction field layout.
//
// We define it here because rust-script cannot reference path-based workspace
// crates. The struct has the same fields and serde attributes as the real
// Transaction in crates/q-types/src/lib.rs so the serialization cost is
// representative.
// ---------------------------------------------------------------------------

type TxHash = [u8; 32];
type Address = [u8; 32];
type Amount = u128;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum TokenType {
    QUG,
    QUGUSD,
    Custom([u8; 32]),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
enum TransactionType {
    Transfer = 0x00,
    Coinbase = 0x01,
    Burn = 0x02,
    Fee = 0x03,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum TxSignaturePhase {
    Phase0Ed25519,
    Phase2SQIsign,
    HybridEd25519SQIsign,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
enum TransactionPrivacyLevel {
    #[default]
    Transparent,
    ConfidentialAmount,
    FullPrivacy,
}

/// Mirrors q_types::Transaction with identical fields and serde layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    id: TxHash,
    from: Address,
    to: Address,
    amount: Amount,
    fee: Amount,
    nonce: u64,
    signature: Vec<u8>,
    timestamp: chrono::DateTime<chrono::Utc>,
    data: Vec<u8>,
    token_type: TokenType,
    fee_token_type: TokenType,
    tx_type: TransactionType,
    pqc_signature: Option<Vec<u8>>,
    signature_phase: TxSignaturePhase,
    pqc_public_key: Option<Vec<u8>>,
    zk_proof_bundle: Option<Vec<u8>>,
    privacy_level: TransactionPrivacyLevel,
    bulletproof: Option<Vec<u8>>,
    nullifier: Option<[u8; 32]>,
    memo: Option<String>,
}

impl Transaction {
    /// Compute the SHA3-256 signing payload, same algorithm as q_types.
    fn signing_payload(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.from);
        hasher.update(&self.to);
        hasher.update(&self.amount.to_le_bytes());
        hasher.update(&self.fee.to_le_bytes());
        hasher.update(&self.nonce.to_le_bytes());
        hasher.update(&self.timestamp.timestamp_millis().to_le_bytes());
        hasher.update(&self.data);
        // token_type discriminant
        let token_disc: u8 = match self.token_type {
            TokenType::QUG => 0,
            TokenType::QUGUSD => 1,
            TokenType::Custom(_) => 2,
        };
        hasher.update(&[token_disc]);
        let fee_disc: u8 = match self.fee_token_type {
            TokenType::QUG => 0,
            TokenType::QUGUSD => 1,
            TokenType::Custom(_) => 2,
        };
        hasher.update(&[fee_disc]);
        hasher.update(&[self.tx_type as u8]);
        hasher.finalize().into()
    }

    /// Sign this transaction with a real Ed25519 key. Returns the payload
    /// that was signed (for later verification without re-hashing).
    fn sign(&mut self, signing_key: &SigningKey) -> [u8; 32] {
        let payload = self.signing_payload();
        let sig = signing_key.sign(&payload);
        self.signature = sig.to_bytes().to_vec();
        payload
    }

    /// Verify the Ed25519 signature against the signing payload.
    fn verify(&self, verifying_key: &VerifyingKey) -> bool {
        if self.signature.len() != 64 {
            return false;
        }
        let payload = self.signing_payload();
        let sig_bytes: [u8; 64] = match self.signature.clone().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let sig = Signature::from_bytes(&sig_bytes);
        verifying_key.verify(&payload, &sig).is_ok()
    }
}

/// Create a realistic transaction with random but valid-looking fields.
fn make_unsigned_tx(nonce: u64, from: Address, to: Address) -> Transaction {
    let mut id = [0u8; 32];
    id[..8].copy_from_slice(&nonce.to_le_bytes());
    // Mix in from/to to make each id unique
    for i in 0..8 {
        id[8 + i] = from[i] ^ to[i];
    }

    Transaction {
        id,
        from,
        to,
        amount: 1_000_000_000, // 1 QUG (in smallest unit)
        fee: 100_000,
        nonce,
        signature: Vec::new(),
        timestamp: chrono::Utc::now(),
        data: Vec::new(), // empty payload, typical transfer
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

// ---------------------------------------------------------------------------
// Benchmark stages
// ---------------------------------------------------------------------------

struct StageResult {
    name: &'static str,
    count: usize,
    elapsed: std::time::Duration,
}

impl StageResult {
    fn tps(&self) -> f64 {
        self.count as f64 / self.elapsed.as_secs_f64()
    }

    fn per_op_us(&self) -> f64 {
        self.elapsed.as_micros() as f64 / self.count as f64
    }
}

fn bench_signature_creation(
    txs: &mut [Transaction],
    signing_key: &SigningKey,
) -> StageResult {
    let start = Instant::now();
    for tx in txs.iter_mut() {
        tx.sign(signing_key);
    }
    let elapsed = start.elapsed();
    StageResult {
        name: "Ed25519 sign",
        count: txs.len(),
        elapsed,
    }
}

fn bench_signature_verification(
    txs: &[Transaction],
    verifying_key: &VerifyingKey,
) -> StageResult {
    let mut verified = 0usize;
    let start = Instant::now();
    for tx in txs.iter() {
        if tx.verify(verifying_key) {
            verified += 1;
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(verified, txs.len(), "Some signatures failed verification!");
    StageResult {
        name: "Ed25519 verify",
        count: txs.len(),
        elapsed,
    }
}

fn bench_bincode_roundtrip(txs: &[Transaction]) -> (StageResult, usize) {
    let mut total_bytes = 0usize;
    let start = Instant::now();
    for tx in txs.iter() {
        let encoded = bincode::serialize(tx).expect("bincode serialize");
        total_bytes += encoded.len();
        let _decoded: Transaction = bincode::deserialize(&encoded).expect("bincode deserialize");
    }
    let elapsed = start.elapsed();
    (
        StageResult {
            name: "Bincode round-trip",
            count: txs.len(),
            elapsed,
        },
        total_bytes / txs.len(), // average bytes per tx
    )
}

fn bench_msgpack_roundtrip(txs: &[Transaction]) -> (StageResult, usize) {
    let mut total_bytes = 0usize;
    let start = Instant::now();
    for tx in txs.iter() {
        let encoded = rmp_serde::to_vec(tx).expect("msgpack serialize");
        total_bytes += encoded.len();
        let _decoded: Transaction = rmp_serde::from_slice(&encoded).expect("msgpack deserialize");
    }
    let elapsed = start.elapsed();
    (
        StageResult {
            name: "MsgPack round-trip",
            count: txs.len(),
            elapsed,
        },
        total_bytes / txs.len(),
    )
}

fn bench_full_pipeline(
    count: usize,
    from: Address,
    to: Address,
    signing_key: &SigningKey,
    verifying_key: &VerifyingKey,
) -> StageResult {
    // Full pipeline: create + sign + bincode serialize + deserialize + verify
    let start = Instant::now();
    for i in 0..count {
        let mut tx = make_unsigned_tx(i as u64, from, to);
        tx.sign(signing_key);
        let encoded = bincode::serialize(&tx).expect("serialize");
        let decoded: Transaction = bincode::deserialize(&encoded).expect("deserialize");
        assert!(decoded.verify(verifying_key), "pipeline verify failed at tx {}", i);
    }
    let elapsed = start.elapsed();
    StageResult {
        name: "Full pipeline",
        count,
        elapsed,
    }
}

fn bench_hashing_only(txs: &[Transaction]) -> StageResult {
    let start = Instant::now();
    for tx in txs.iter() {
        let _payload = tx.signing_payload();
    }
    let elapsed = start.elapsed();
    StageResult {
        name: "SHA3-256 payload hash",
        count: txs.len(),
        elapsed,
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn print_header() {
    println!("========================================================================");
    println!("  Q-NarwhalKnight TPS Benchmark -- Honest Measurement");
    println!("========================================================================");
    println!();
    println!("  Methodology:");
    println!("    - Real Ed25519 key generation and signing (ed25519-dalek 2.x)");
    println!("    - Real SHA3-256 payload hashing");
    println!("    - Real serialization (bincode 1.x, rmp-serde 1.x)");
    println!("    - Wall-clock timing with std::time::Instant");
    println!("    - Single-threaded, sequential processing");
    println!("    - No simulated delays, no multipliers");
    println!();
}

fn print_stage(r: &StageResult) {
    println!(
        "  {:<25} {:>10} tx   {:>10.2} ms   {:>10.1} us/tx   {:>10.0} tx/s",
        r.name,
        r.count,
        r.elapsed.as_secs_f64() * 1000.0,
        r.per_op_us(),
        r.tps(),
    );
}

fn main() {
    print_header();

    let n = 10_000;
    println!("  Transaction count: {}", n);
    println!();

    // Generate a real Ed25519 keypair
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    let from: Address = verifying_key.to_bytes();
    let to: Address = SigningKey::generate(&mut OsRng).verifying_key().to_bytes();

    println!("  Sender pubkey:  {}...", hex(&from[..8]));
    println!("  Receiver pubkey: {}...", hex(&to[..8]));
    println!();

    // -- Stage 1: Create unsigned transactions --
    let create_start = Instant::now();
    let mut txs: Vec<Transaction> = (0..n)
        .map(|i| make_unsigned_tx(i as u64, from, to))
        .collect();
    let create_elapsed = create_start.elapsed();
    println!("  Created {} unsigned transactions in {:.2} ms", n, create_elapsed.as_secs_f64() * 1000.0);
    println!();

    // -- Stage 2: SHA3-256 hashing --
    let hash_result = bench_hashing_only(&txs);

    // -- Stage 3: Ed25519 signing --
    let sign_result = bench_signature_creation(&mut txs, &signing_key);

    // -- Stage 4: Ed25519 verification --
    let verify_result = bench_signature_verification(&txs, &verifying_key);

    // -- Stage 5: Bincode round-trip --
    let (bincode_result, bincode_avg_bytes) = bench_bincode_roundtrip(&txs);

    // -- Stage 6: MessagePack round-trip --
    let (msgpack_result, msgpack_avg_bytes) = bench_msgpack_roundtrip(&txs);

    // -- Stage 7: Full pipeline (1000 tx, since it's slow) --
    let pipeline_n = 1_000;
    let pipeline_result = bench_full_pipeline(pipeline_n, from, to, &signing_key, &verifying_key);

    // -- Print results --
    println!("------------------------------------------------------------------------");
    println!("  STAGE BREAKDOWN (single-threaded)");
    println!("------------------------------------------------------------------------");
    println!(
        "  {:<25} {:>10}      {:>10}      {:>10}       {:>10}",
        "Stage", "Count", "Total ms", "Per-tx us", "Throughput"
    );
    println!("  {}", "-".repeat(70));

    print_stage(&hash_result);
    print_stage(&sign_result);
    print_stage(&verify_result);
    print_stage(&bincode_result);
    print_stage(&msgpack_result);
    print_stage(&pipeline_result);

    println!();
    println!("------------------------------------------------------------------------");
    println!("  SERIALIZATION COMPARISON");
    println!("------------------------------------------------------------------------");
    println!("  {:<20} {:>10} bytes/tx   {:>10.0} tx/s", "Bincode", bincode_avg_bytes, bincode_result.tps());
    println!("  {:<20} {:>10} bytes/tx   {:>10.0} tx/s", "MessagePack", msgpack_avg_bytes, msgpack_result.tps());
    let size_ratio = msgpack_avg_bytes as f64 / bincode_avg_bytes as f64;
    let speed_ratio = bincode_result.tps() / msgpack_result.tps();
    println!();
    println!("  MsgPack is {:.1}x the size of Bincode", size_ratio);
    println!("  Bincode round-trip is {:.1}x faster than MsgPack", speed_ratio);

    println!();
    println!("------------------------------------------------------------------------");
    println!("  SUMMARY");
    println!("------------------------------------------------------------------------");
    println!();
    println!("  Signature verification ceiling:  {:>10.0} tx/s  (single thread)", verify_result.tps());
    println!("  Full pipeline ceiling:           {:>10.0} tx/s  (single thread)", pipeline_result.tps());
    println!();
    println!("  The signature verification step is the bottleneck.");
    println!("  Multi-threaded verification on N cores would yield ~{:.0} tx/s",
        verify_result.tps() * estimated_cores() as f64);
    println!("  assuming linear scaling (real scaling will be lower).");
    println!();
    println!("  These numbers represent what this machine can actually do.");
    println!("  Network latency, disk I/O, consensus, and mempool contention");
    println!("  will reduce real-world throughput further.");
    println!("========================================================================");
}

fn estimated_cores() -> usize {
    // Read from /proc/cpuinfo or fall back
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.matches("processor\t").count())
        .unwrap_or(4)
        .max(1)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
