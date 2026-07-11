//! READ-ONLY forensic: classify each QUGUSD holder by ORIGIN — legit CDP mint vs
//! swap/transfer (phantom-class). Opens the LIVE DB in RocksDB SECONDARY mode.
//!
//! Sources cross-referenced:
//!   - CollateralVault (key "collateral_vault", bincode) → per-wallet minted_qugusd
//!     (legit CDP mint, backed by locked_qug).
//!   - token_balance_<wallet>_<QUGUSD> → current QUGUSD holding.
//! Classification: legit_backed = min(holding, cdp_minted); the excess came from
//! DEX swaps / transfers (the swap-mint bug credited token_balances w/o CDP).
//!
//! Usage: trace_qugusd_origin <primary_db> <secondary_scratch> [min_qugusd_whole]

use rocksdb::{Options, DB};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use serde::Deserialize;

const CF: &str = "manifest";
const QUGUSD: &str = "5155475553440000000000000000000000000000000000000000000000000000";
const DEC: u128 = 1_000_000_000_000_000_000_000_000; // 24-dec

fn fmt(v: u128) -> String {
    let w = v / DEC;
    // thousands separator on the whole part
    let s = w.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}

// Mirror of q_vm::contracts::CollateralVault — SAME field order/types for bincode.
#[derive(Deserialize)]
struct Vault {
    locked_qug: HashMap<[u8; 32], u128>,
    minted_qugusd: HashMap<[u8; 32], u128>,
    #[allow(dead_code)] qug_price_usd: f64,
    total_qug_locked: u128,
    total_qugusd_minted: u128,
    #[allow(dead_code)] last_price_update: i64,
}

fn read_bal(v: &[u8]) -> u128 {
    if v.len() >= 16 { u128::from_le_bytes(v[..16].try_into().unwrap()) }
    else if v.len() >= 8 { (u64::from_le_bytes(v[..8].try_into().unwrap()) as u128) * 10u128.pow(16) }
    else { 0 }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: trace_qugusd_origin <primary_db> <secondary> [min_whole]"); std::process::exit(2); }
    let (primary, secondary) = (&a[1], &a[2]);
    let min_whole: u128 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);

    let cfs = DB::list_cf(&Options::default(), primary)?;
    let mut o = Options::default(); o.create_if_missing(false);
    let db = DB::open_cf_as_secondary(&o, primary, secondary, &cfs)
        .map_err(|e| anyhow!("open secondary failed: {e}"))?;
    let _ = db.try_catch_up_with_primary();
    let cf = db.cf_handle(CF).ok_or_else(|| anyhow!("manifest CF missing"))?;

    // 1) Vault CDP data
    let vault: Option<Vault> = db.get_cf(&cf, b"collateral_vault")?
        .and_then(|b| bincode::deserialize::<Vault>(&b).ok());
    println!("═══ CDP VAULT (legit mint ground-truth) ═══");
    if let Some(ref v) = vault {
        println!("  total_qug_locked   = {} QUG", fmt(v.total_qug_locked));
        println!("  total_qugusd_minted= {} QUGUSD", fmt(v.total_qugusd_minted));
        println!("  CDP positions (minted_qugusd > 0):");
        let mut mints: Vec<(&[u8;32], &u128)> = v.minted_qugusd.iter().filter(|(_,m)| **m>0).collect();
        mints.sort_by(|a,b| b.1.cmp(a.1));
        for (w, m) in &mints {
            let locked = v.locked_qug.get(*w).copied().unwrap_or(0);
            println!("    {}…  minted={} QUGUSD  locked={} QUG", &hex::encode(w)[..16], fmt(**m), fmt(locked));
        }
        if mints.is_empty() { println!("    (none)"); }
    } else {
        println!("  ⚠️ no collateral_vault key / deserialize failed");
    }
    let minted_of = |w: &[u8;32]| vault.as_ref().and_then(|v| v.minted_qugusd.get(w).copied()).unwrap_or(0);

    // 2) QUGUSD holders → classify by origin
    let qsuffix = format!("_{}", QUGUSD);
    let prefix = b"token_balance_";
    let mut holders: Vec<([u8;32], u128)> = Vec::new();
    let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward));
    for item in iter {
        let (k, val) = item?;
        let ks = match std::str::from_utf8(&k) { Ok(s) => s, Err(_) => continue };
        if !ks.starts_with("token_balance_") { break; }
        if !ks.ends_with(&qsuffix) { continue; }
        let addr_hex = &ks["token_balance_".len()..ks.len()-qsuffix.len()];
        if addr_hex.len() != 64 { continue; }
        let mut addr = [0u8;32];
        if hex::decode_to_slice(addr_hex, &mut addr).is_err() { continue; }
        let bal = read_bal(&val);
        if bal / DEC >= min_whole { holders.push((addr, bal)); }
    }
    holders.sort_by(|a,b| b.1.cmp(&a.1));

    println!("\n═══ QUGUSD HOLDERS ≥ {} — ORIGIN CLASSIFICATION ═══", fmt(min_whole*DEC));
    println!("  {:<18} {:>18} {:>18} {:>18}  ORIGIN", "wallet", "holding", "cdp_minted", "non-cdp(swap/xfer)");
    let (mut sum_hold, mut sum_cdp, mut sum_phantom) = (0u128,0u128,0u128);
    for (addr, bal) in &holders {
        let cdp = minted_of(addr);
        let backed = (*bal).min(cdp);
        let excess = bal.saturating_sub(cdp);
        sum_hold += *bal; sum_cdp += backed; sum_phantom += excess;
        let origin = if cdp == 0 { "100% swap/transfer (NO CDP)" }
            else if excess == 0 { "fully CDP-backed" }
            else { "partial CDP + swap/transfer excess" };
        println!("  {}…  {:>18} {:>18} {:>18}  {}",
            &hex::encode(addr)[..16], fmt(*bal), fmt(cdp), fmt(excess), origin);
    }
    println!("  ───────────────────────────────────────────────");
    println!("  TOTALS: holding={}  cdp_backed={}  non-cdp={}", fmt(sum_hold), fmt(sum_cdp), fmt(sum_phantom));
    println!("  → {} QUGUSD of the {} shown originated OUTSIDE the CDP (swap-mint/transfer)", fmt(sum_phantom), fmt(sum_hold));
    Ok(())
}
