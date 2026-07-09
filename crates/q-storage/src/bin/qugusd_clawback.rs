//! Phantom-QUGUSD Clawback (2026-07-02) — token_balance analog of clawback_phantom_supply.
//! Zeroes/reduces phantom QUGUSD in NAMED exploiter wallets ONLY. Same safety model:
//!   allowlist-only, pre-image match, lower-only, floor, max-targets, DRY-RUN by default.
//! DRY-RUN opens the LIVE DB in READ-ONLY secondary mode (safe alongside the running node).
//! --confirm opens R/W (EXCLUSIVE — the node MUST be stopped + DB backed up first).
//!
//! Usage:
//!   qugusd_clawback <primary_db> <secondary_scratch> \
//!       --target <hex> --expect-qugusd <whole> --set-qugusd <whole> [more targets] \
//!       [--min-inflated 1000000] [--tolerance-pct 10] [--max-targets 5] [--confirm]

use rocksdb::{Options, DB, WriteOptions};
use anyhow::{anyhow, bail, Result};

const CF: &str = "manifest";
const QUGUSD_HEX: &str = "5155475553440000000000000000000000000000000000000000000000000000";
const DEC: u128 = 1_000_000_000_000_000_000_000_000; // 10^24

fn key_for(addr_hex: &str) -> String { format!("token_balance_{}_{}", addr_hex, QUGUSD_HEX) }
fn fmt(base: u128) -> String { format!("{}.{:024}", base / DEC, base % DEC) }

fn read_bal(v: &[u8]) -> u128 {
    if v.len() == 16 { u128::from_le_bytes(v[..16].try_into().unwrap()) }
    else if v.len() == 8 { (u64::from_le_bytes(v[..8].try_into().unwrap()) as u128) * 10u128.pow(16) }
    else { 0 }
}

struct T { hex: String, expect: u128, set: u128 }

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: qugusd_clawback <db> <secondary> --target <hex> --expect-qugusd <n> --set-qugusd <n> [...] [--confirm]"); std::process::exit(2); }
    let (primary, secondary) = (a[1].clone(), a[2].clone());
    let (mut confirm, mut floor, mut tol, mut maxt) = (false, 1_000_000u128, 10u128, 5usize);
    let mut ts: Vec<T> = Vec::new();
    let (mut ca, mut ce, mut cs): (Option<String>, Option<u128>, Option<u128>) = (None, None, None);
    let flush = |ts: &mut Vec<T>, ca: &mut Option<String>, ce: &mut Option<u128>, cs: &mut Option<u128>| -> Result<()> {
        if ca.is_some() {
            let h = ca.take().unwrap().trim().trim_start_matches("qnk").to_lowercase();
            if h.len()!=64 { bail!("bad addr len"); }
            ts.push(T { hex: h, expect: ce.take().ok_or_else(||anyhow!("missing --expect-qugusd"))?, set: cs.take().ok_or_else(||anyhow!("missing --set-qugusd"))? });
        } Ok(())
    };
    let mut i = 3;
    while i < a.len() {
        match a[i].as_str() {
            "--target" => { flush(&mut ts,&mut ca,&mut ce,&mut cs)?; i+=1; ca=Some(a[i].clone()); }
            "--expect-qugusd" => { i+=1; ce=Some(a[i].parse()?); }
            "--set-qugusd" => { i+=1; cs=Some(a[i].parse()?); }
            "--min-inflated" => { i+=1; floor=a[i].parse()?; }
            "--tolerance-pct" => { i+=1; tol=a[i].parse()?; }
            "--max-targets" => { i+=1; maxt=a[i].parse()?; }
            "--confirm" => confirm=true,
            o => bail!("unknown arg {o}"),
        } i+=1;
    }
    flush(&mut ts,&mut ca,&mut ce,&mut cs)?;
    if ts.is_empty() { bail!("no --target"); }
    if ts.len() > maxt { bail!("{} targets > max {}", ts.len(), maxt); }

    println!("🧮 QUGUSD Clawback — {}", if confirm {"CONFIRM (WILL WRITE — node must be STOPPED)"} else {"DRY-RUN (read-only)"});
    println!("📂 {}  floor≥{} tol±{}% targets={}", primary, floor, tol, ts.len());

    let cf_names = DB::list_cf(&Options::default(), &primary)?;
    let mut opts = Options::default(); opts.create_if_missing(false);
    let db = if confirm {
        // EXCLUSIVE R/W — fails if the node still holds the lock (that's the safety: node must be stopped).
        DB::open_cf(&opts, &primary, &cf_names).map_err(|e| anyhow!("open R/W failed (is the node stopped?): {e}"))?
    } else {
        DB::open_cf_as_secondary(&opts, &primary, &secondary, &cf_names).map_err(|e| anyhow!("open secondary failed: {e}"))?
    };
    if !confirm { let _ = db.try_catch_up_with_primary(); }
    let cfh = db.cf_handle(CF).ok_or_else(|| anyhow!("manifest CF missing"))?;

    let mut planned: Vec<(String,u128,u128)> = Vec::new();
    for t in &ts {
        let cur = db.get_cf(&cfh, key_for(&t.hex).as_bytes())?.map(|v| read_bal(&v)).unwrap_or(0);
        let exp = t.expect.saturating_mul(DEC); let set = t.set.saturating_mul(DEC);
        println!("── {}…{}\n   current: {} QUGUSD\n   expect : {}\n   set-to : {}", &t.hex[..12], &t.hex[52..], fmt(cur), t.expect, t.set);
        if cur < floor.saturating_mul(DEC) { println!("   ⛔ REFUSED: below floor — not inflated"); continue; }
        let lo = exp.saturating_sub(exp/100*tol); let hi = exp.saturating_add(exp/100*tol);
        if cur < lo || cur > hi { println!("   ⛔ REFUSED: current outside expect ±{}% — wrong wallet/stale", tol); continue; }
        if set >= cur { println!("   ⛔ REFUSED: set >= current — clawback only reduces"); continue; }
        println!("   ✅ PLAN: -{} QUGUSD → {}", fmt(cur-set), fmt(set));
        planned.push((t.hex.clone(), cur, set));
    }
    println!("Summary: {} planned.", planned.len());
    if planned.is_empty() || !confirm {
        if !confirm && !planned.is_empty() { println!("DRY-RUN — no writes. Re-run with --confirm after STOPPING the node + backing up the DB."); }
        return Ok(());
    }
    println!("✍️  WRITING (node must be stopped)…");
    let mut wo = WriteOptions::default(); wo.set_sync(true);
    for (h, old, new) in &planned {
        db.put_cf_opt(&cfh, key_for(h).as_bytes(), new.to_le_bytes(), &wo)?;
        println!("   ✅ {}…  {} → {} QUGUSD", &h[..12], fmt(*old), fmt(*new));
    }
    db.flush()?;
    println!("Done. Restart node; verify /stablecoin/transparency supply is corrected.");
    Ok(())
}
