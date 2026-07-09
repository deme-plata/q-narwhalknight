//! Phantom-Supply Clawback Utility (2026-06-25)
//!
//! ONE-SHOT, SURGICAL, OFFLINE balance correction for the exploiter wallets that
//! hold the phantom QUG minted by the DEX/repeat-send bugs. It exists to restore the
//! 21,000,000 QUG cap by debiting ONLY the named, provably-inflated wallets back to an
//! operator-chosen legitimate residual — and to be incapable of touching anything else.
//!
//! ════════════════════════════════════════════════════════════════════════════════
//!  SAFETY MODEL — why this cannot harm a correct wallet (CLAUDE.md Rule 1 / Rule 4)
//! ════════════════════════════════════════════════════════════════════════════════
//!  1. ALLOWLIST ONLY        — writes go to ONLY the addresses you pass with --target.
//!                             The only full-table scan is READ-ONLY (final supply report).
//!  2. PRE-IMAGE MATCH        — each --target carries --expect-qug; the on-disk balance
//!                             must match it (within --tolerance-pct, default 10%) or that
//!                             wallet is SKIPPED. A typo'd address lands on a different
//!                             wallet whose balance won't match → skipped. No blind writes.
//!  3. INFLATION FLOOR        — refuses any wallet holding < --min-inflated-qug (default
//!                             1,000,000 QUG). Honest wallets are orders of magnitude below
//!                             the exploiter's ~29.67e9 QUG, so they are structurally excluded.
//!  4. LOWER-ONLY             — refuses if new >= current (a clawback can only REDUCE).
//!  5. MAX TARGETS            — aborts if more than --max-targets (default 5) are given.
//!  6. DRY-RUN BY DEFAULT     — does nothing without --confirm; prints the exact plan.
//!  7. NARROW WRITE SURFACE   — touches ONLY `wallet_balance_<addr>` keys for named targets.
//!                             It does NOT modify total_minted_supply, emission state, roots,
//!                             or any other wallet. (The integrity endpoint recomputes
//!                             total_supply from balances, so it self-corrects on restart.)
//!
//! ════════════════════════════════════════════════════════════════════════════════
//!  OPERATING PROCEDURE (mandatory)
//! ════════════════════════════════════════════════════════════════════════════════
//!  A. TEST ON ALPHA FIRST against a COPY of the DB (CLAUDE.md Rule 4 + May-2026 incident).
//!  B. STOP the node (offline write; avoids in-memory/disk race).
//!  C. BACK UP the DB dir before --confirm.
//!  D. Dry-run (no --confirm), eyeball every line, THEN re-run with --confirm.
//!  E. Restart the node; it reloads balances from RocksDB and rebuilds balance roots.
//!     Verify GET /api/v1/integrity/balance-root shows total within 21,000,000 and
//!     roots_agree=true. The v10.11.69 conservation invariant prevents recurrence.
//!
//! Usage:
//!   clawback_phantom_supply <db_path> \
//!       --target <addr64hex> --expect-qug <current_whole_QUG> --set-qug <residual_whole_QUG> \
//!       [--target ... --expect-qug ... --set-qug ...] \
//!       [--min-inflated-qug 1000000] [--tolerance-pct 10] [--max-targets 5] [--confirm]
//!
//! Example (dry-run for the known exploiter + accomplice; residual 0):
//!   clawback_phantom_supply /home/orobit/data-mainnet-genesis/hot \
//!     --target 4cf3b55ebeb28c97ce5c7a5369169cf1db933a4cd3d120d7bef3a79ed8c4f2fb --expect-qug 29672179227 --set-qug 0 \
//!     --target 0817277ff9f9ebb7a096dc171a34094e2c33ab74b6cd4478fce9e9de12182ba2 --expect-qug 1552639 --set-qug 0

use anyhow::{anyhow, bail, Result};
use rocksdb::{ColumnFamilyDescriptor, Options, DB};
use std::sync::Arc;

const CF_MANIFEST: &str = "manifest";
const QUG: u128 = 1_000_000_000_000_000_000_000_000; // 10^24 base units per QUG
const CAP_QUG: u128 = 21_000_000;

#[derive(Debug)]
struct Target {
    addr_hex: String,
    addr: [u8; 32],
    expect_qug: u128,
    set_qug: u128,
}

fn parse_addr(s: &str) -> Result<[u8; 32]> {
    let s = s.trim().trim_start_matches("qnk").trim_start_matches("0x");
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("address must be 64 hex chars, got {:?}", s);
    }
    let bytes = hex::decode(s)?;
    let mut a = [0u8; 32];
    a.copy_from_slice(&bytes);
    Ok(a)
}

fn load_balance<C: rocksdb::AsColumnFamilyRef>(db: &DB, cf: &C, addr: &[u8; 32]) -> Result<u128> {
    let key = format!("wallet_balance_{}", hex::encode(addr));
    match db.get_cf(cf, key.as_bytes())? {
        Some(bytes) if bytes.len() == 16 => Ok(u128::from_le_bytes(bytes[..16].try_into().unwrap())),
        Some(bytes) if bytes.len() == 8 => {
            // legacy u64 (8 decimals) → u128 (24 decimals)
            Ok((u64::from_le_bytes(bytes[..8].try_into().unwrap()) as u128) * 10u128.pow(16))
        }
        Some(b) => bail!("wallet_balance has unexpected length {}", b.len()),
        None => Ok(0),
    }
}

fn fmt_qug(base: u128) -> String {
    format!("{}.{:024}", base / QUG, base % QUG)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: clawback_phantom_supply <db_path> --target <hex> --expect-qug <n> --set-qug <n> [...] [--min-inflated-qug N] [--tolerance-pct N] [--max-targets N] [--confirm]");
        std::process::exit(2);
    }
    let db_path = args[1].clone();

    let mut confirm = false;
    let mut min_inflated_qug: u128 = 1_000_000; // floor: never touch a wallet below this
    let mut tolerance_pct: u128 = 10; // pre-image match window
    let mut max_targets: usize = 5;
    let mut targets: Vec<Target> = Vec::new();

    // pending target fields
    let mut cur_addr: Option<String> = None;
    let mut cur_expect: Option<u128> = None;
    let mut cur_set: Option<u128> = None;

    let mut i = 2;
    let flush = |targets: &mut Vec<Target>,
                 a: &mut Option<String>,
                 e: &mut Option<u128>,
                 s: &mut Option<u128>|
     -> Result<()> {
        if a.is_some() || e.is_some() || s.is_some() {
            let addr_hex = a.take().ok_or_else(|| anyhow!("--target without address"))?;
            let expect = e.take().ok_or_else(|| anyhow!("--target {} missing --expect-qug", addr_hex))?;
            let set = s.take().ok_or_else(|| anyhow!("--target {} missing --set-qug", addr_hex))?;
            let addr = parse_addr(&addr_hex)?;
            targets.push(Target { addr_hex: hex::encode(addr), addr, expect_qug: expect, set_qug: set });
        }
        Ok(())
    };

    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                flush(&mut targets, &mut cur_addr, &mut cur_expect, &mut cur_set)?;
                i += 1;
                cur_addr = Some(args.get(i).ok_or_else(|| anyhow!("--target needs value"))?.clone());
            }
            "--expect-qug" => {
                i += 1;
                cur_expect = Some(args.get(i).ok_or_else(|| anyhow!("--expect-qug needs value"))?.parse()?);
            }
            "--set-qug" => {
                i += 1;
                cur_set = Some(args.get(i).ok_or_else(|| anyhow!("--set-qug needs value"))?.parse()?);
            }
            "--min-inflated-qug" => { i += 1; min_inflated_qug = args[i].parse()?; }
            "--tolerance-pct" => { i += 1; tolerance_pct = args[i].parse()?; }
            "--max-targets" => { i += 1; max_targets = args[i].parse()?; }
            "--confirm" => confirm = true,
            other => bail!("unknown arg: {}", other),
        }
        i += 1;
    }
    flush(&mut targets, &mut cur_addr, &mut cur_expect, &mut cur_set)?;

    if targets.is_empty() {
        bail!("no --target given; refusing to run");
    }
    if targets.len() > max_targets {
        bail!("{} targets exceeds --max-targets {} — refusing (clawback must be surgical)", targets.len(), max_targets);
    }

    println!("🧮 Phantom-Supply Clawback — {}", if confirm { "CONFIRM (WILL WRITE)" } else { "DRY-RUN (no writes)" });
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📂 DB: {}", db_path);
    println!("🛡️  floor=≥{} QUG  tolerance=±{}%  max_targets={}  targets={}", min_inflated_qug, tolerance_pct, max_targets, targets.len());
    println!();

    // Open DB read/write, all existing CFs.
    let cf_names = DB::list_cf(&Options::default(), &db_path)?;
    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);
    let cfs: Vec<_> = cf_names.iter().map(|n| ColumnFamilyDescriptor::new(n.as_str(), Options::default())).collect();
    let db = Arc::new(DB::open_cf_descriptors(&db_opts, &db_path, cfs)?);
    let cf = db.cf_handle(CF_MANIFEST).ok_or_else(|| anyhow!("manifest CF not found"))?;

    let mut planned: Vec<(String, u128, u128)> = Vec::new();
    let mut refused = 0u32;

    for t in &targets {
        let cur = load_balance(&db, &cf, &t.addr)?;
        let expect_base = t.expect_qug.saturating_mul(QUG);
        let set_base = t.set_qug.saturating_mul(QUG);
        let floor_base = min_inflated_qug.saturating_mul(QUG);

        println!("── target {}…{}", &t.addr_hex[..12], &t.addr_hex[52..]);
        println!("   on-disk current : {} QUG", fmt_qug(cur));
        println!("   expected current: {} QUG", t.expect_qug);
        println!("   new residual    : {} QUG", t.set_qug);

        // GUARD 3: inflation floor
        if cur < floor_base {
            println!("   ⛔ REFUSED: current {} < floor {} QUG — not an inflated wallet, left untouched", fmt_qug(cur), min_inflated_qug);
            refused += 1;
            continue;
        }
        // GUARD 2: pre-image match (typo/wrong-wallet guard)
        let lo = expect_base.saturating_sub(expect_base / 100 * tolerance_pct);
        let hi = expect_base.saturating_add(expect_base / 100 * tolerance_pct);
        if cur < lo || cur > hi {
            println!("   ⛔ REFUSED: current {} outside expected ±{}% [{}..{}] — wrong wallet or stale expect; left untouched",
                fmt_qug(cur), tolerance_pct, fmt_qug(lo), fmt_qug(hi));
            refused += 1;
            continue;
        }
        // GUARD 4: lower-only
        if set_base >= cur {
            println!("   ⛔ REFUSED: new {} >= current {} — clawback only reduces; left untouched", fmt_qug(set_base), fmt_qug(cur));
            refused += 1;
            continue;
        }
        let delta = cur - set_base;
        println!("   ✅ PLAN: reduce by {} QUG  →  new balance {} QUG", fmt_qug(delta), fmt_qug(set_base));
        planned.push((t.addr_hex.clone(), cur, set_base));
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Summary: {} planned, {} refused.", planned.len(), refused);

    if planned.is_empty() {
        println!("Nothing to do.");
        return Ok(());
    }

    if !confirm {
        println!();
        println!("DRY-RUN — no writes performed. Re-run with --confirm after backing up the DB.");
        return Ok(());
    }

    // WRITE — narrow surface: ONLY the named wallet_balance_<addr> keys.
    println!();
    println!("✍️  WRITING (synced)…");
    for (addr_hex, old, new) in &planned {
        let key = format!("wallet_balance_{}", addr_hex);
        let mut wo = rocksdb::WriteOptions::default();
        wo.set_sync(true);
        db.put_cf_opt(&cf, key.as_bytes(), new.to_le_bytes(), &wo)?;
        println!("   ✅ {}…  {} → {} QUG (CLAWBACK)", &addr_hex[..12], fmt_qug(*old), fmt_qug(*new));
    }
    db.flush()?;

    // READ-ONLY rescan: report the resulting total native supply vs cap.
    println!();
    println!("🔍 Read-only rescan of wallet_balance_* for new total native supply…");
    let mut total: u128 = 0;
    let mut count: u64 = 0;
    let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
    for item in iter {
        let (k, v) = item?;
        if k.starts_with(b"wallet_balance_") {
            let bal = if v.len() == 16 {
                u128::from_le_bytes(v[..16].try_into().unwrap())
            } else if v.len() == 8 {
                (u64::from_le_bytes(v[..8].try_into().unwrap()) as u128) * 10u128.pow(16)
            } else { 0 };
            total = total.saturating_add(bal);
            if bal > 0 { count += 1; }
        }
    }
    println!("   wallets with balance: {}", count);
    println!("   new total native supply: {} QUG  (cap {})", fmt_qug(total), CAP_QUG);
    if total / QUG <= CAP_QUG {
        println!("   ✅ within the 21,000,000 cap.");
    } else {
        println!("   ⚠️ STILL over cap by {} QUG — more inflated wallets remain; rerun with the next target(s).", (total / QUG).saturating_sub(CAP_QUG));
    }
    println!();
    println!("NEXT: restart the node (reloads balances + rebuilds balance roots). Verify");
    println!("      /api/v1/integrity/balance-root shows total within cap and roots_agree=true.");
    println!("NOTE: this bin did NOT touch total_minted_supply/emission/roots — only the named");
    println!("      wallet balances. If the emission counter needs aligning, do it deliberately.");
    Ok(())
}
