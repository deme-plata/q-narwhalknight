//! READ-ONLY forensic scanner for phantom QUGUSD holders (2026-07-02).
//! Opens the LIVE prod DB in RocksDB SECONDARY mode (no lock, no writes, safe
//! alongside the running node) and dumps `token_balance_*_<QUGUSD>` holders,
//! sorted by amount desc. Excludes an operator-provided legit wallet.
//!
//! Usage: hunt_qugusd <primary_db_path> <secondary_scratch_dir> [min_qugusd] [exclude_hex]
//!   e.g. hunt_qugusd /home/orobit/data-mainnet-genesis/hot /home/storage/hunt-sec 1000 1ca3a232...

use rocksdb::{Options, DB};
use anyhow::{anyhow, Result};

const CF_MANIFEST: &str = "manifest";
// QUGUSD token address = "QUGUSD" ascii + zeros, hex:
const QUGUSD_HEX: &str = "5155475553440000000000000000000000000000000000000000000000000000";
const DEC: u128 = 1_000_000_000_000_000_000_000_000; // 10^24 (QUGUSD decimals=24)

fn fmt(base: u128) -> String { format!("{}.{:024}", base / DEC, base % DEC) }

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: hunt_qugusd <primary_db> <secondary_dir> [min_qugusd] [exclude_hex]"); std::process::exit(2); }
    let primary = &a[1];
    let secondary = &a[2];
    let min_qugusd: u128 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let exclude = a.get(4).map(|s| s.trim().trim_start_matches("qnk").to_lowercase()).unwrap_or_default();

    let cf_names = DB::list_cf(&Options::default(), primary)?;
    let mut opts = Options::default();
    opts.create_if_missing(false);
    // SECONDARY: read-only replica of the live primary, no lock contention.
    let db = DB::open_cf_as_secondary(&opts, primary, secondary, &cf_names)
        .map_err(|e| anyhow!("open secondary failed: {e}"))?;
    let _ = db.try_catch_up_with_primary();
    let cf = db.cf_handle(CF_MANIFEST).ok_or_else(|| anyhow!("manifest CF missing"))?;

    let suffix = format!("_{}", QUGUSD_HEX);
    let mut holders: Vec<(String, u128)> = Vec::new();
    let mut total: u128 = 0;
    let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
    for item in iter {
        let (k, v) = item?;
        let key = String::from_utf8_lossy(&k);
        if !key.starts_with("token_balance_") || !key.ends_with(&suffix) { continue; }
        // wallet hex = between "token_balance_" and "_<qugusd>"
        let mid = &key["token_balance_".len()..];
        let wallet = mid.trim_end_matches(&suffix).to_string();
        let bal: u128 = if v.len() == 16 { u128::from_le_bytes(v[..16].try_into().unwrap()) }
                        else if v.len() == 8 { (u64::from_le_bytes(v[..8].try_into().unwrap()) as u128) * 10u128.pow(16) }
                        else { continue };
        total = total.saturating_add(bal);
        if bal >= min_qugusd.saturating_mul(DEC) { holders.push((wallet, bal)); }
    }
    holders.sort_by(|x, y| y.1.cmp(&x.1));

    println!("🔎 QUGUSD phantom-holder scan (READ-ONLY, secondary) — {}", primary);
    println!("   total QUGUSD across all holders: {} QUGUSD", fmt(total));
    println!("   holders >= {} QUGUSD: {}", min_qugusd, holders.len());
    if !exclude.is_empty() { println!("   EXCLUDING legit wallet: {}…", &exclude[..exclude.len().min(16)]); }
    println!("   ───────────────────────────────────────────────────────────");
    for (w, b) in holders.iter().take(40) {
        let tag = if w.to_lowercase() == exclude { "  ← LEGIT (excluded from clawback)" } else { "" };
        println!("   {}  {} QUGUSD{}", w, fmt(*b), tag);
    }
    Ok(())
}
