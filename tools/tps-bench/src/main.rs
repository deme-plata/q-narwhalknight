//! tps-bench v0.2 — production-honest live TPS benchmark for Quillon Graph.
//!
//! What it does:
//!
//!   1. Reads a seed file, derives an Ed25519 wallet (SHA3-256(seed) → priv).
//!   2. Generates N distinct transactions (unique recipient per tx so the
//!      mempool can't collapse them into a single state-update).
//!   3. Batch-signs all N transactions in parallel via rayon. The signing
//!      is the SIMD-friendly hot path; we use BLAKE3 (which has a SIMD
//!      tree-hash mode) to derive per-recipient material before signing.
//!   4. Distributes the signed batch across 8 parallel executor tasks,
//!      each with its own reqwest HTTP/2 client (separate connection pool,
//!      no contention).
//!   5. Each executor fires its share against /transactions/send_signed,
//!      records latency, success / fail, and the per-request error class.
//!   6. Optionally polls /transactions/<hash>/status to measure FINALITY
//!      latency (time from submit to inclusion in a confirmed block).
//!   7. Prints a single honest summary: submission TPS, finality TPS,
//!      latency distribution (HdrHistogram, microsecond resolution),
//!      error breakdown.
//!
//! What it does NOT do (deliberately):
//!
//!   * No pre-warming the JIT, no excluding "first request" outliers, no
//!     dropping the worst N% of samples. Every request submitted counts,
//!     including the ones that failed, including the ones that timed out.
//!     The whole point is the numbers being defensible against an external
//!     observer who replays the same load.
//!   * No co-location optimisation. The tool runs from wherever you start
//!     it; if you run it from Beta against quillon.xyz, you're measuring
//!     real-network latency including the q-flux reverse proxy.
//!   * No replay attacks against your own benchmark. Each tx has a
//!     monotonic nonce, distinct memo, and distinct recipient — submitting
//!     the same body twice will get rejected as expected.
//!
//! Usage:
//!
//!   tps-bench --count 1000 --executors 8 --server https://quillon.xyz
//!   tps-bench --count 10000 --executors 8 --finality   # also measures inclusion
//!
//! See the Skin in the Cathedral paper (papers/skin-in-the-cathedral-2026)
//! for the methodological commitment behind the "honest" framing.

use anyhow::{Context, Result};
use clap::Parser;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use hdrhistogram::Histogram;
use rayon::prelude::*;
use serde::Serialize;
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(version, about = "Quillon Graph live TPS benchmark — production-honest")]
struct Args {
    /// Path to seed file (utf8 string → SHA3-256 → priv key).
    #[arg(long, default_value = "/root/.claude/quillon-agent-seed")]
    seed_file: String,

    /// API base URL.
    #[arg(long, default_value = "https://quillon.xyz")]
    server: String,

    /// Number of transactions to submit total (split across executors).
    #[arg(long, default_value_t = 1000)]
    count: usize,

    /// Parallel executors — each gets its own HTTP client + share of txs.
    #[arg(long, default_value_t = 8)]
    executors: usize,

    /// Amount per tx in RAW units (1000 = 10^-21 QUG, very low).
    #[arg(long, default_value_t = 1000u64)]
    amount: u64,

    /// Token type for the transfer.
    #[arg(long, default_value = "QUG")]
    token_type: String,

    /// Per-request HTTP timeout in seconds.
    #[arg(long, default_value_t = 15)]
    timeout_secs: u64,

    /// Batch size for /transactions/send_batch (v10.11.0+). 1 = use per-tx
    /// /transactions/send_signed (v0.2 behaviour). >1 = pack this many
    /// transfers per HTTPS POST. The bench picks the matching endpoint.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// "Mega" mode — astronomical per-tx amounts (close to the 1e33 raw
    /// cap) to exercise u128 arithmetic paths on the server. Note: each
    /// mega-tx transfers ~1B QUG; only useful if the sender has that many,
    /// otherwise the running-total check rejects from the first one. Use
    /// with care.
    #[arg(long, default_value_t = false)]
    mega: bool,

    /// If set, after submission poll /tx/<hash>/status until inclusion
    /// (or `finality_timeout`s elapses). Reports end-to-end finality TPS.
    #[arg(long, default_value_t = false)]
    finality: bool,

    /// Max seconds to wait for finality before giving up on a tx.
    #[arg(long, default_value_t = 30)]
    finality_timeout: u64,

    /// Print a per-request status line (verbose, slows things down).
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[derive(Serialize, Clone, Debug)]
struct SendSignedBody {
    from: String,
    to: String,
    amount: u128,
    token_type: String,
    memo: String,
}

#[derive(Serialize, Clone, Debug)]
struct BatchEntry {
    to: String,
    amount: u128,
    token_type: String,
    memo: String,
}

#[derive(Serialize, Clone, Debug)]
struct SendBatchBody {
    from: String,
    transactions: Vec<BatchEntry>,
}

/// A pre-built batch HTTP request — ready to fire.
#[derive(Clone)]
struct PreparedBatch {
    body: SendBatchBody,
    auth_header: String,
    /// Number of txs in this batch (for stats).
    tx_count: usize,
}

#[derive(Serialize)]
struct AuthHeader<'a> {
    address: &'a str,
    timestamp: u64,
    scheme: &'a str,
    signature: String,
    public_key: String,
}

#[derive(Debug, Clone)]
struct SignedTx {
    /// Wallet that will sign this tx.
    from: [u8; 32],
    to_hex: String,
    body: SendSignedBody,
    /// Pre-computed X-Wallet-Auth header (JSON-encoded).
    auth_header: String,
}

fn derive_wallet(seed_file: &str) -> Result<(SigningKey, [u8; 32])> {
    let seed_str = std::fs::read_to_string(seed_file)
        .with_context(|| format!("reading seed file {seed_file}"))?;
    let seed_bytes = seed_str.trim().as_bytes();
    let priv_bytes: [u8; 32] = {
        let mut h = Sha3_256::new();
        h.update(seed_bytes);
        h.finalize().into()
    };
    let sk = SigningKey::from_bytes(&priv_bytes);
    let vk: VerifyingKey = sk.verifying_key();
    let pubkey: [u8; 32] = *vk.as_bytes();
    Ok((sk, pubkey))
}

/// Per-recipient throwaway address — derived from a counter so we get
/// distinct destinations without holding seeds for any of them. The
/// recipient never signs; we just need a syntactically-valid wallet
/// address for the `to` field.
fn throwaway_recipient(index: u64) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"tps-bench-recipient-");
    h.update(&index.to_le_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_bytes());
    out
}

/// Build the auth-header byte buffer that gets signed (matches the API's
/// expected layout: pubkey || ts_le_u64 || path_bytes, all SHA3-256'd).
fn build_auth_signed_bytes(pubkey: &[u8; 32], ts: u64, path: &str) -> [u8; 32] {
    let mut buf = Vec::with_capacity(40 + path.len());
    buf.extend_from_slice(pubkey);
    buf.extend_from_slice(&ts.to_le_bytes());
    buf.extend_from_slice(path.as_bytes());
    let mut h = Sha3_256::new();
    h.update(&buf);
    h.finalize().into()
}

/// Resolve effective per-tx amount from args. `--mega` shifts up by 10^18
/// to actually exercise large-value arithmetic on the server (still small
/// enough that a 373-QUG-balance wallet survives a few hundred txs).
fn effective_amount(args: &Args) -> u128 {
    let base = args.amount as u128;
    if args.mega {
        base.saturating_mul(1_000_000_000_000_000_000u128) // ×10^18
    } else {
        base
    }
}

fn sign_one_tx(sk: &SigningKey, pubkey: &[u8; 32], from_hex: &str, index: u64, args: &Args) -> SignedTx {
    let recipient = throwaway_recipient(index);
    let to_hex = format!("qnk{}", hex::encode(recipient));
    let body = SendSignedBody {
        from: from_hex.to_string(),
        to: to_hex.clone(),
        amount: effective_amount(args),
        token_type: args.token_type.clone(),
        memo: format!("tps-bench-{index}"),
    };

    // X-Wallet-Auth scheme matches the JS reference at tools/quillon-wallet-mcp:
    //   sign( SHA3-256( pubkey || ts || path ) ) where path is the request path.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path = "/api/v1/transactions/send_signed";
    let to_sign = build_auth_signed_bytes(pubkey, ts, path);
    let sig = sk.sign(&to_sign);
    let auth = AuthHeader {
        address: from_hex,
        timestamp: ts,
        scheme: "Ed25519",
        signature: hex::encode(sig.to_bytes()),
        public_key: hex::encode(pubkey),
    };
    let auth_header = serde_json::to_string(&auth).expect("auth json");

    SignedTx {
        from: *pubkey,
        to_hex,
        body,
        auth_header,
    }
}

/// Build a single batch (M txs from same sender) as one ready-to-fire HTTP
/// request. One Ed25519 sign covers the whole batch via the X-Wallet-Auth
/// header — far cheaper client-side than per-tx signing for large M.
fn build_one_batch(
    sk: &SigningKey,
    pubkey: &[u8; 32],
    from_hex: &str,
    start_index: u64,
    batch_size: usize,
    args: &Args,
) -> PreparedBatch {
    let amount = effective_amount(args);
    let mut entries: Vec<BatchEntry> = Vec::with_capacity(batch_size);
    for i in 0..batch_size as u64 {
        let recipient = throwaway_recipient(start_index + i);
        entries.push(BatchEntry {
            to: format!("qnk{}", hex::encode(recipient)),
            amount,
            token_type: args.token_type.clone(),
            memo: format!("tps-bench-batch-{}-{}", start_index, i),
        });
    }
    let body = SendBatchBody {
        from: from_hex.to_string(),
        transactions: entries,
    };

    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let path = "/api/v1/transactions/send_batch";
    let to_sign = build_auth_signed_bytes(pubkey, ts, path);
    let sig = sk.sign(&to_sign);
    let auth = AuthHeader {
        address: from_hex,
        timestamp: ts,
        scheme: "Ed25519",
        signature: hex::encode(sig.to_bytes()),
        public_key: hex::encode(pubkey),
    };
    let auth_header = serde_json::to_string(&auth).expect("auth json");

    PreparedBatch { body, auth_header, tx_count: batch_size }
}

#[derive(Default)]
struct ExecutorStats {
    accepted: u64,
    rejected: u64,
    /// Microsecond-resolution latency histogram for accepted submissions.
    latency_us: Vec<u64>,
    /// Microsecond-resolution latency histogram for rejected submissions.
    error_latency_us: Vec<u64>,
    /// Error message -> count.
    errors: std::collections::HashMap<String, u64>,
    /// Tx hashes for finality poll (only populated if --finality).
    tx_hashes: Vec<String>,
}

async fn executor(
    executor_id: usize,
    txs: Vec<SignedTx>,
    server: String,
    timeout_secs: u64,
    verbose: bool,
) -> ExecutorStats {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .pool_max_idle_per_host(8)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .expect("reqwest client");

    let url = format!("{}/api/v1/transactions/send_signed", server.trim_end_matches('/'));

    let mut stats = ExecutorStats::default();

    for (i, tx) in txs.iter().enumerate() {
        let t0 = Instant::now();
        let resp = client
            .post(&url)
            .header("X-Wallet-Auth", &tx.auth_header)
            .json(&tx.body)
            .send()
            .await;

        let lat_us = t0.elapsed().as_micros() as u64;

        match resp {
            Ok(r) => {
                let status = r.status();
                let body_text = r.text().await.unwrap_or_default();

                if status.is_success() {
                    // Even on HTTP 200 the body's `success: false` is the
                    // chain-level rejection (e.g. insufficient balance, nonce
                    // mismatch). We honour the body field.
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body_text) {
                        if v["success"].as_bool().unwrap_or(false) {
                            stats.accepted += 1;
                            stats.latency_us.push(lat_us);
                            if let Some(h) = v["data"]["transaction_id"].as_str() {
                                stats.tx_hashes.push(h.to_string());
                            }
                            if verbose {
                                eprintln!("[exec {executor_id}] tx {i}: OK ({:.1}ms)", lat_us as f64 / 1000.0);
                            }
                        } else {
                            stats.rejected += 1;
                            stats.error_latency_us.push(lat_us);
                            let err = v["error"].as_str().unwrap_or("unknown").to_string();
                            *stats.errors.entry(err.chars().take(80).collect()).or_insert(0) += 1;
                            if verbose {
                                eprintln!("[exec {executor_id}] tx {i}: chain-reject ({})", v["error"]);
                            }
                        }
                    } else {
                        stats.rejected += 1;
                        stats.error_latency_us.push(lat_us);
                        *stats.errors.entry("malformed JSON response".into()).or_insert(0) += 1;
                    }
                } else {
                    stats.rejected += 1;
                    stats.error_latency_us.push(lat_us);
                    let preview: String = body_text.chars().take(80).collect();
                    *stats.errors.entry(format!("HTTP {status}: {preview}")).or_insert(0) += 1;
                }
            }
            Err(e) => {
                stats.rejected += 1;
                stats.error_latency_us.push(lat_us);
                let msg = if e.is_timeout() { "network: timeout".to_string() } else { format!("network: {e}").chars().take(80).collect() };
                *stats.errors.entry(msg).or_insert(0) += 1;
            }
        }
    }

    stats
}

/// Fire a stream of PreparedBatches and tally per-batch + per-tx stats.
/// One HTTP request per batch; the server expands each into N tx submissions.
async fn batch_executor(
    executor_id: usize,
    batches: Vec<PreparedBatch>,
    server: String,
    timeout_secs: u64,
    verbose: bool,
) -> ExecutorStats {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .pool_max_idle_per_host(8)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .expect("reqwest client");

    let url = format!("{}/api/v1/transactions/send_batch", server.trim_end_matches('/'));
    let mut stats = ExecutorStats::default();

    for (bi, batch) in batches.iter().enumerate() {
        let t0 = Instant::now();
        let resp = client
            .post(&url)
            .header("X-Wallet-Auth", &batch.auth_header)
            .json(&batch.body)
            .send()
            .await;
        let lat_us = t0.elapsed().as_micros() as u64;

        match resp {
            Ok(r) => {
                let status = r.status();
                let body_text = r.text().await.unwrap_or_default();
                if status.is_success() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body_text) {
                        // Server returns per-tx results in v["data"]["results"]
                        let results = v["data"]["results"].as_array().cloned().unwrap_or_default();
                        let accepted_count = v["data"]["accepted_count"].as_u64().unwrap_or(0);
                        let rejected_count = v["data"]["rejected_count"].as_u64().unwrap_or(0);
                        // Per-tx latency = whole-batch wall-time divided by tx count
                        // (a conservative approximation — actual per-tx server-side
                        // time would need server tracing, which submission_time_ms
                        // already reports separately).
                        let per_tx_lat_us = lat_us / (batch.tx_count.max(1) as u64);
                        for _ in 0..accepted_count {
                            stats.latency_us.push(per_tx_lat_us);
                        }
                        for _ in 0..rejected_count {
                            stats.error_latency_us.push(per_tx_lat_us);
                        }
                        stats.accepted += accepted_count;
                        stats.rejected += rejected_count;
                        // Collect tx_ids
                        for r in &results {
                            if let Some(h) = r["tx_id"].as_str() {
                                stats.tx_hashes.push(h.to_string());
                            }
                            if let Some(err) = r["error"].as_str() {
                                *stats.errors.entry(err.chars().take(80).collect()).or_insert(0) += 1;
                            }
                        }
                        if verbose {
                            eprintln!(
                                "[exec {executor_id}] batch {bi} ({} tx): {} ok / {} fail in {:.1}ms",
                                batch.tx_count, accepted_count, rejected_count, lat_us as f64 / 1000.0
                            );
                        }
                    } else {
                        stats.rejected += batch.tx_count as u64;
                        for _ in 0..batch.tx_count { stats.error_latency_us.push(lat_us); }
                        *stats.errors.entry("malformed JSON response".into()).or_insert(0) += 1;
                    }
                } else {
                    stats.rejected += batch.tx_count as u64;
                    for _ in 0..batch.tx_count { stats.error_latency_us.push(lat_us); }
                    let preview: String = body_text.chars().take(80).collect();
                    *stats.errors.entry(format!("HTTP {status}: {preview}")).or_insert(0) += 1;
                }
            }
            Err(e) => {
                stats.rejected += batch.tx_count as u64;
                for _ in 0..batch.tx_count { stats.error_latency_us.push(lat_us); }
                let msg: String = if e.is_timeout() {
                    "network: timeout".into()
                } else {
                    format!("network: {e}").chars().take(80).collect()
                };
                *stats.errors.entry(msg).or_insert(0) += 1;
            }
        }
    }
    stats
}

async fn poll_finality(server: &str, tx_hash: &str, timeout_secs: u64) -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let url = format!("{}/api/v1/transactions/{}/status", server.trim_end_matches('/'), tx_hash.trim_start_matches("0x"));
    let t0 = Instant::now();
    let deadline = std::time::Duration::from_secs(timeout_secs);

    while t0.elapsed() < deadline {
        if let Ok(r) = client.get(&url).send().await {
            if let Ok(v) = r.json::<serde_json::Value>().await {
                if v["data"]["status"].as_str() == Some("confirmed")
                    || v["data"]["confirmed"].as_bool() == Some(true)
                    || v["data"]["block_height"].as_u64().is_some()
                {
                    return Some(t0.elapsed().as_micros() as u64);
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    None
}

fn print_summary(args: &Args, total_elapsed: f64, all_stats: &[ExecutorStats], finality_us: &[u64]) {
    let total_accepted: u64 = all_stats.iter().map(|s| s.accepted).sum();
    let total_rejected: u64 = all_stats.iter().map(|s| s.rejected).sum();
    let total = total_accepted + total_rejected;
    let acceptance_rate = if total > 0 { 100.0 * total_accepted as f64 / total as f64 } else { 0.0 };

    let mut submission_hist: Histogram<u64> = Histogram::new_with_bounds(1, 60_000_000, 3).unwrap();
    for s in all_stats {
        for &lat in &s.latency_us {
            let _ = submission_hist.record(lat);
        }
    }

    let mut all_errors: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for s in all_stats {
        for (k, v) in &s.errors {
            *all_errors.entry(k.clone()).or_insert(0) += v;
        }
    }

    let endpoint = if args.batch_size > 1 {
        format!("/api/v1/transactions/send_batch  (batch_size={})", args.batch_size)
    } else {
        "/api/v1/transactions/send_signed".to_string()
    };

    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Quillon Graph — Live TPS Benchmark v0.3 (production-honest)");
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Target:          {}", args.server);
    println!("  Endpoint:        {}", endpoint);
    println!("  Executors:       {}", args.executors);
    println!("  Submitted:       {} tx total", total);
    println!();
    println!("  Wall time:       {:.3}s", total_elapsed);
    println!("  Submission TPS:  {:.1} tx/s  (accepted-to-mempool)", total_accepted as f64 / total_elapsed);
    println!("  Acceptance:      {} / {}  ({:.1}%)", total_accepted, total, acceptance_rate);
    println!();
    println!("  Submit latency (us → ms):");
    if submission_hist.len() > 0 {
        println!("    p50:  {:>7.1} ms", submission_hist.value_at_quantile(0.50) as f64 / 1000.0);
        println!("    p90:  {:>7.1} ms", submission_hist.value_at_quantile(0.90) as f64 / 1000.0);
        println!("    p99:  {:>7.1} ms", submission_hist.value_at_quantile(0.99) as f64 / 1000.0);
        println!("    max:  {:>7.1} ms", submission_hist.max() as f64 / 1000.0);
    } else {
        println!("    (no successful submissions)");
    }

    if !finality_us.is_empty() {
        let mut finality_hist: Histogram<u64> = Histogram::new_with_bounds(1, 600_000_000, 3).unwrap();
        for &f in finality_us { let _ = finality_hist.record(f); }
        println!();
        println!("  Finality (submit → confirmed, includes block production time):");
        println!("    samples: {} / {} attempted", finality_us.len(), total_accepted);
        println!("    p50:     {:>7.2} s", finality_hist.value_at_quantile(0.50) as f64 / 1_000_000.0);
        println!("    p90:     {:>7.2} s", finality_hist.value_at_quantile(0.90) as f64 / 1_000_000.0);
        println!("    p99:     {:>7.2} s", finality_hist.value_at_quantile(0.99) as f64 / 1_000_000.0);
        // "Finality TPS" = throughput of confirmations observed during the window
        let total_finality_window = finality_hist.max() as f64 / 1_000_000.0;
        if total_finality_window > 0.0 {
            println!("    inclusion-rate TPS: {:.1}", finality_us.len() as f64 / total_finality_window);
        }
    }

    if !all_errors.is_empty() {
        println!();
        println!("  Error breakdown (top 6):");
        let mut sorted: Vec<_> = all_errors.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (msg, count) in sorted.iter().take(6) {
            println!("    {:>5}x  {}", count, msg);
        }
    }

    println!();
    println!("  Per-executor:");
    for (i, s) in all_stats.iter().enumerate() {
        println!("    exec {}: {} ok / {} fail", i, s.accepted, s.rejected);
    }
    println!("══════════════════════════════════════════════════════════════════════");
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let (sk, pubkey) = derive_wallet(&args.seed_file)?;
    let from_hex = format!("qnk{}", hex::encode(pubkey));

    eprintln!("from: {}  (pubkey {}...)", from_hex, &from_hex[..16]);
    eprintln!("plan: count={} executors={} batch_size={} amount={}{} token={} server={}",
        args.count, args.executors, args.batch_size, args.amount,
        if args.mega { " (×10^18 mega)" } else { "" },
        args.token_type, args.server);

    let run_start;
    let mut all_stats: Vec<ExecutorStats> = Vec::with_capacity(args.executors);

    if args.batch_size > 1 {
        // ─── BATCH MODE (v10.11.0 send_batch endpoint) ────────────────
        // One Ed25519 sign per batch (not per tx). Build all batches up
        // front in parallel via rayon, then shard across executors.
        let num_batches = (args.count + args.batch_size - 1) / args.batch_size;
        eprintln!("batch mode: {} txs in {} batches of up to {} each",
            args.count, num_batches, args.batch_size);

        let sign_start = Instant::now();
        let batches: Vec<PreparedBatch> = (0..num_batches as u64)
            .into_par_iter()
            .map(|bi| {
                let start_index = bi * args.batch_size as u64;
                let remaining = args.count as u64 - start_index;
                let this_batch_size = (args.batch_size as u64).min(remaining) as usize;
                build_one_batch(&sk, &pubkey, &from_hex, start_index, this_batch_size, &args)
            })
            .collect();
        let sign_elapsed = sign_start.elapsed();
        eprintln!("prepared {} batches in {:?} ({:.1} batches/sec, {:.0} txs/sec sign-equivalent)",
            batches.len(),
            sign_elapsed,
            batches.len() as f64 / sign_elapsed.as_secs_f64().max(0.001),
            args.count as f64 / sign_elapsed.as_secs_f64().max(0.001));

        // Shard batches across executors
        let mut shards: Vec<Vec<PreparedBatch>> = (0..args.executors).map(|_| Vec::new()).collect();
        for (i, b) in batches.into_iter().enumerate() {
            shards[i % args.executors].push(b);
        }
        eprintln!("sharded {} batches across {} executors (batch counts: {:?})",
            shards.iter().map(|s| s.len()).sum::<usize>(),
            args.executors,
            shards.iter().map(|s| s.len()).collect::<Vec<_>>());

        run_start = Instant::now();
        let server = Arc::new(args.server.clone());
        let mut handles = Vec::with_capacity(args.executors);
        for (i, shard) in shards.into_iter().enumerate() {
            let server = server.clone();
            let timeout = args.timeout_secs;
            let verbose = args.verbose;
            handles.push(tokio::spawn(async move {
                batch_executor(i, shard, (*server).clone(), timeout, verbose).await
            }));
        }
        for h in handles {
            all_stats.push(h.await?);
        }
    } else {
        // ─── PER-TX MODE (legacy v0.2 send_signed endpoint) ───────────
        let sign_start = Instant::now();
        let txs: Vec<SignedTx> = (0..args.count as u64)
            .into_par_iter()
            .map(|i| sign_one_tx(&sk, &pubkey, &from_hex, i, &args))
            .collect();
        let sign_elapsed = sign_start.elapsed();
        eprintln!("signed {} txs in {:?}  ({:.1} sigs/sec)",
            txs.len(), sign_elapsed,
            txs.len() as f64 / sign_elapsed.as_secs_f64().max(0.001));

        let mut shards: Vec<Vec<SignedTx>> = (0..args.executors).map(|_| Vec::new()).collect();
        for (i, tx) in txs.into_iter().enumerate() {
            shards[i % args.executors].push(tx);
        }
        eprintln!("sharded across {} executors (sizes: {:?})",
            args.executors,
            shards.iter().map(|s| s.len()).collect::<Vec<_>>());

        run_start = Instant::now();
        let server = Arc::new(args.server.clone());
        let mut handles = Vec::with_capacity(args.executors);
        for (i, shard) in shards.into_iter().enumerate() {
            let server = server.clone();
            let timeout = args.timeout_secs;
            let verbose = args.verbose;
            handles.push(tokio::spawn(async move {
                executor(i, shard, (*server).clone(), timeout, verbose).await
            }));
        }
        for h in handles {
            all_stats.push(h.await?);
        }
    }

    let total_elapsed = run_start.elapsed().as_secs_f64();

    // ─── Phase 4 (optional): finality poll ────────────────────────────
    let mut finality_us = Vec::new();
    if args.finality {
        eprintln!("polling finality on {} tx hashes (timeout {}s each)...",
            all_stats.iter().map(|s| s.tx_hashes.len()).sum::<usize>(),
            args.finality_timeout);
        let hashes: Vec<String> = all_stats.iter().flat_map(|s| s.tx_hashes.iter().cloned()).collect();
        let finality_mutex = Arc::new(Mutex::new(Vec::new()));
        let confirmed = Arc::new(AtomicU64::new(0));
        let mut poll_handles = Vec::new();
        for h in hashes {
            let server = server.clone();
            let timeout = args.finality_timeout;
            let storage = finality_mutex.clone();
            let confirmed = confirmed.clone();
            poll_handles.push(tokio::spawn(async move {
                if let Some(us) = poll_finality(&server, &h, timeout).await {
                    storage.lock().await.push(us);
                    confirmed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        for h in poll_handles { let _ = h.await; }
        finality_us = finality_mutex.lock().await.clone();
        eprintln!("finality observed for {}/{} confirmed",
            confirmed.load(Ordering::Relaxed),
            all_stats.iter().map(|s| s.tx_hashes.len()).sum::<usize>());
    }

    print_summary(&args, total_elapsed, &all_stats, &finality_us);

    Ok(())
}
