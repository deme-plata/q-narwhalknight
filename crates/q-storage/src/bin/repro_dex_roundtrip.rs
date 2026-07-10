//! 2026-07-09 REPRO: does a zero-sum QUG<->QUGUSD DEX round trip mint QUG?
//!
//! Drives the EXACT QStorage functions the /api/v1/dex/swap handler calls, in
//! the same order, against a fresh temp DB — no network/auth/pool/mining needed.
//!   SELL (QUG->QUGUSD): atomic_subtract_and_record_dex_debit  (debit + applied_net, atomic)
//!   BUY  (QUGUSD->QUG): record_dex_qug_credit  then  add_balance
//!                       add_balance(a,amt) == save_wallet_balance(a, load(a)+amt)  (lib.rs:9881)
//!   RECONCILER: apply_dex_qug_adjustments()  (the 5s periodic safety-net task)
//!
//! A correct system nets ~0 across a sell-1 / buy-1-back cycle. If the balance
//! grows, the reconciler is double-applying the buy-back credit = money printer.

use q_storage::QStorage;

const QUG: u128 = 1_000_000_000_000_000_000_000_000; // 1e24 base units

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(run());
}

async fn run() {
    let dir = std::env::temp_dir().join(format!("repro-dex-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let storage = QStorage::open(&dir, [7u8; 32]).await.expect("open storage");

    let wallet = [0xABu8; 32];
    let wallet_hex = hex::encode(wallet);
    let start = 100 * QUG;
    storage.save_wallet_balance(&wallet, start).await.unwrap();

    let sell = 1 * QUG; // sell 1 QUG
    let back = 1 * QUG; // buy ~1 QUG back (exact AMM value irrelevant to counter/reconciler logic)

    println!("start balance = {:.4} QUG", start as f64 / 1e24);
    for round in 0..5u32 {
        // ---- SELL 1 QUG -> QUGUSD (native QUG debit) ----
        storage
            .atomic_subtract_and_record_dex_debit(&wallet_hex, sell)
            .await
            .unwrap();
        storage.apply_dex_qug_adjustments().await.unwrap(); // reconciler tick
        let after_sell = storage.load_wallet_balance(&wallet).await.unwrap().unwrap_or(0);

        // ---- BUY 1 QUG back with QUGUSD (native QUG credit) ----
        storage.record_dex_qug_credit(&wallet_hex, back).await.unwrap();
        // add_balance emulation (identical to lib.rs:9881 — load, saturating add, max-wins save)
        let cur = storage.load_wallet_balance(&wallet).await.unwrap().unwrap_or(0);
        storage
            .save_wallet_balance(&wallet, cur.saturating_add(back))
            .await
            .unwrap();
        storage.apply_dex_qug_adjustments().await.unwrap(); // reconciler tick
        let after_buy = storage.load_wallet_balance(&wallet).await.unwrap().unwrap_or(0);

        println!(
            "ROUND {}: after_sell={:.4}  after_buy={:.4}  net_vs_start={:+.4} QUG",
            round,
            after_sell as f64 / 1e24,
            after_buy as f64 / 1e24,
            (after_buy as i128 - start as i128) as f64 / 1e24,
        );
    }

    let final_bal = storage.load_wallet_balance(&wallet).await.unwrap().unwrap_or(0);
    let net = (final_bal as i128 - start as i128) as f64 / 1e24;
    println!(
        "OLD-PATH FINAL: {:.4} QUG (start 100.0000). Net over 5 zero-sum round trips = {:+.4} QUG",
        final_bal as f64 / 1e24,
        net
    );
    println!(
        "OLD-PATH VERDICT: {}",
        if net.abs() < 0.5 {
            "NO MINT — balance conserved (bug NOT reproduced)"
        } else {
            "MINT REPRODUCED — QUG created from thin air by the reconciler"
        }
    );

    // ─────────────────────────────────────────────────────────────────────
    // v10.11.82 FIX PROOF: same 5 zero-sum round trips, but the buy-back uses the
    // ATOMIC credit (balance + counter + applied_net in one batch) instead of
    // record_dex_qug_credit + add_balance. Balance must stay flat.
    // ─────────────────────────────────────────────────────────────────────
    let w2 = [0xCDu8; 32];
    let w2_hex = hex::encode(w2);
    storage.save_wallet_balance(&w2, start).await.unwrap();
    for round in 0..5u32 {
        storage
            .atomic_subtract_and_record_dex_debit(&w2_hex, sell)
            .await
            .unwrap();
        storage.apply_dex_qug_adjustments().await.unwrap();
        // FIXED buy-back: single atomic credit (what handlers.rs:13929 now calls)
        storage
            .atomic_add_and_record_dex_credit(&w2_hex, back)
            .await
            .unwrap();
        storage.apply_dex_qug_adjustments().await.unwrap();
        let after_buy = storage.load_wallet_balance(&w2).await.unwrap().unwrap_or(0);
        println!(
            "FIX ROUND {}: after_buy={:.4}  net_vs_start={:+.4} QUG",
            round,
            after_buy as f64 / 1e24,
            (after_buy as i128 - start as i128) as f64 / 1e24,
        );
    }
    let fixed_final = storage.load_wallet_balance(&w2).await.unwrap().unwrap_or(0);
    let fixed_net = (fixed_final as i128 - start as i128) as f64 / 1e24;
    println!(
        "FIX FINAL: {:.4} QUG. Net over 5 zero-sum round trips = {:+.4} QUG",
        fixed_final as f64 / 1e24,
        fixed_net
    );
    println!(
        "FIX VERDICT: {}",
        if fixed_net.abs() < 0.5 {
            "CONSERVED — no mint (fix works)"
        } else {
            "STILL MINTING — fix failed"
        }
    );
    let _ = std::fs::remove_dir_all(&dir);
}
