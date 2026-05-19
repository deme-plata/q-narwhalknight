//! One-shot signed-transaction sender for agentic commons participation.
//!
//! Usage:
//!   TRADING_SEED=<hex-seed> cargo run --release --bin send_with_memo -- \
//!       --to <qnk-recipient> --amount-qug 0.05 --memo "..."
//!
//! Derives the wallet from `TRADING_SEED` (or `~/.claude/quillon-agent-seed`),
//! builds a signed Transaction, and submits it to `/api/v1/transactions` on
//! https://quillon.xyz (or whatever Q_API_URL is set to).

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use clap::Parser;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use q_types::{Transaction, TransactionType, TxHash, TokenType, TxSignaturePhase, TransactionPrivacyLevel};
use serde::Serialize;
use sha3::{Digest, Sha3_256};

#[derive(Parser, Debug)]
struct Args {
    /// Recipient address (qnk + 64 hex chars)
    #[arg(long)]
    to: String,
    /// Amount in display QUG (e.g. 0.05)
    #[arg(long)]
    amount_qug: f64,
    /// Memo string attached to the transaction (Option<String>)
    #[arg(long)]
    memo: String,
    /// API root
    #[arg(long, env = "Q_API_URL", default_value = "https://quillon.xyz")]
    api_url: String,
    /// Dry-run: build, sign, print, but don't POST
    #[arg(long)]
    dry_run: bool,
}

fn load_seed() -> Result<String> {
    if let Ok(v) = std::env::var("TRADING_SEED") {
        return Ok(v.trim().to_string());
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let path = format!("{home}/.claude/quillon-agent-seed");
    let s = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    Ok(s.trim().to_string())
}

fn parse_qnk(addr: &str) -> Result<[u8; 32]> {
    if !addr.starts_with("qnk") || addr.len() != 67 {
        return Err(anyhow!("invalid qnk address: {addr}"));
    }
    let hex_part = &addr[3..];
    let bytes = hex::decode(hex_part).context("decode hex")?;
    if bytes.len() != 32 {
        return Err(anyhow!("address hex must decode to 32 bytes"));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Match the field-by-field layout of `Transaction::signing_payload` in
/// `crates/q-types/src/lib.rs:2795` so the server accepts our signature.
fn signing_payload(
    from: &[u8; 32],
    to: &[u8; 32],
    amount: u128,
    fee: u128,
    nonce: u64,
    timestamp_millis: i64,
    data: &[u8],
    token_disc: u8,
    token_addr: &[u8; 32],
    fee_token_disc: u8,
    fee_token_addr: &[u8; 32],
    tx_type_byte: u8,
) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(from);
    h.update(to);
    h.update(&amount.to_le_bytes());
    h.update(&fee.to_le_bytes());
    h.update(&nonce.to_le_bytes());
    h.update(&timestamp_millis.to_le_bytes());
    h.update(data);
    h.update(&[token_disc]);
    h.update(token_addr);
    h.update(&[fee_token_disc]);
    h.update(fee_token_addr);
    h.update(&[tx_type_byte]);
    h.finalize().into()
}

const QUG_TOKEN_ADDR: [u8; 32] = [
    0x51, 0x55, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];
const QUGUSD_TOKEN_ADDR: [u8; 32] = [
    0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let seed = load_seed()?;
    let mut h = Sha3_256::new();
    h.update(seed.as_bytes());
    let priv_bytes: [u8; 32] = h.finalize().into();
    let sk = SigningKey::from_bytes(&priv_bytes);
    let pk: VerifyingKey = sk.verifying_key();
    let from: [u8; 32] = pk.to_bytes();
    let from_addr = format!("qnk{}", hex::encode(from));
    println!("👛 derived address: {from_addr}");

    let to = parse_qnk(&args.to)?;
    println!("➡️  recipient:       {}", args.to);

    let amount_u128: u128 = (args.amount_qug * 1e24).round() as u128;
    println!("💸 amount:          {} QUG ({} raw)", args.amount_qug, amount_u128);

    // Fetch nonce. Try the simple endpoint first; if 401/404, fall back to 0.
    let client = reqwest::Client::new();
    let nonce_url = format!("{}/api/v1/wallets/{}/nonce", args.api_url, from_addr);
    let nonce: u64 = match client.get(&nonce_url).send().await {
        Ok(r) if r.status().is_success() => {
            let v: serde_json::Value = r.json().await?;
            v.get("data").and_then(|d| d.get("nonce")).and_then(|n| n.as_u64()).unwrap_or(0)
        }
        _ => 0, // will get rejected if nonce is too low; we retry with +1 then
    };
    println!("🔢 nonce (initial guess): {nonce}");

    let now = Utc::now();

    // Use the q-types MIN_TRANSACTION_FEE (21000) like the production handler does.
    let fee_u128: u128 = q_types::MIN_TRANSACTION_FEE;

    // Match production handler recipe exactly:
    //   1. Build tx with signature=vec![], id=zero, data=vec![]
    //   2. tx_hash = transaction.hash()  (postcard-encoded SHA3-256)
    //   3. id = tx_hash; sign(&tx_hash); signature = sig; data = pub_key
    let mut tx = Transaction {
        id: TxHash::default(),
        from,
        to,
        amount: amount_u128,
        fee: fee_u128,
        nonce,
        signature: vec![],
        timestamp: now,
        data: vec![],
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
        memo: Some(args.memo.clone()),
    };

    // Roundtrip through JSON to apply whatever precision-loss serde-json
    // imposes (chrono nano-precision, etc.), so the hash we sign matches
    // what the server will compute after parsing our submission.
    let pre_json = serde_json::to_string(&tx)?;
    tx = serde_json::from_str(&pre_json)?;

    // v10.10.0: use signable_payload() which zeros signature+id before hashing.
    // This matches the new canonical verifier in q-types::verify_ed25519_signature
    // (no more chicken-and-egg between sign-time hash and verify-time hash).
    let canonical = tx.signable_payload();
    let tx_hash = tx.hash();
    tx.id = tx_hash;
    let signature = sk.sign(&canonical);
    tx.signature = signature.to_bytes().to_vec();
    tx.data = from.to_vec(); // store pub_key in data[..32] (production convention)

    // Roundtrip ONCE MORE to ensure the JSON we submit is the canonical form
    // (the id, signature, and data fields we just set must survive parse).
    let post_json = serde_json::to_string(&tx)?;
    tx = serde_json::from_str(&post_json)?;

    println!("✍️  signature:       {}...", hex::encode(&tx.signature[..8]));
    println!("🆔 tx hash (signed): {}", hex::encode(tx_hash));
    println!("🆔 tx hash (final):  {}", hex::encode(tx.hash()));

    let payload_json = serde_json::json!({ "transaction": tx });

    if args.dry_run {
        println!("\n--- DRY RUN ---");
        println!("{}", serde_json::to_string_pretty(&payload_json)?);
        return Ok(());
    }

    let submit_url = format!("{}/api/v1/transactions", args.api_url);
    println!("📤 POST {submit_url}");
    let res = client.post(&submit_url).json(&payload_json).send().await?;
    let status = res.status();
    let text = res.text().await?;
    println!("📨 response status: {status}");
    println!("📨 response body:   {text}");

    if !status.is_success() {
        return Err(anyhow!("submit failed: {status} — {text}"));
    }
    Ok(())
}

#[derive(Serialize)]
struct _Unused; // keep clippy happy if features change
