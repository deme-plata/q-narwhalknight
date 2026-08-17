//! Cross-node determinism test for balance_root_v2.
//!
//! The property that actually matters for two independently-synced nodes to
//! agree on a root is NOT "did they insert data in the same order" (they
//! never will — different peers, different chunk arrival timing, different
//! sync paths) but "does insertion order affect the final tree." If it does,
//! two perfectly correct nodes with byte-identical final balances could still
//! disagree, which would break the entire cross-node verification story.
//!
//! This builds the SAME real (address, balance) data set into THREE separate,
//! throwaway SMT instances, inserted in three genuinely different orders
//! (as-loaded, reversed, randomly shuffled), and confirms all three produce
//! the bit-identical root. Read-only against the source DB; all three test
//! trees live in fresh temp directories, never touching the real SMT.
//!
//! Usage: smt_determinism_test <data_dir>

use anyhow::Result;
use q_storage::StorageEngine;
use q_storage::balance_smt::BalanceSmt;
use std::sync::Arc;

fn deterministic_shuffle(items: &mut Vec<([u8; 32], u128)>) {
    // Fixed-seed xorshift — no external RNG dependency, but genuinely
    // reorders (not just a fixed rotation) so it's a real different order.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = items.len();
    for i in (1..n).rev() {
        let j = (next() as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn build_fresh_smt(tmp_path: &std::path::Path, data: &[([u8; 32], u128)]) -> Result<[u8; 32]> {
    let mut db_opts = rocksdb::Options::default();
    db_opts.create_if_missing(true);
    db_opts.create_missing_column_families(true);
    let db = Arc::new(rocksdb::DB::open_cf(
        &db_opts,
        tmp_path,
        [q_storage::balance_smt::CF_BALANCE_SMT],
    )?);
    let smt = BalanceSmt::open(db)?;
    smt.update_batch(data)?;
    Ok(smt.root())
}

fn main() -> Result<()> {
    let data_dir = std::env::args().nth(1).unwrap();

    println!("🔬 CROSS-NODE DETERMINISM TEST");
    println!("   source data_dir = {}", data_dir);
    println!("═══════════════════════════════════════════════════════════");

    let rt = tokio::runtime::Runtime::new()?;
    let balances: Vec<([u8; 32], u128)> = rt.block_on(async {
        let node_id: [u8; 32] = [0u8; 32];
        let engine = StorageEngine::open(&data_dir, node_id).await?;
        let map = engine.load_wallet_balances().await?;
        anyhow::Ok(map.into_iter().filter(|(_, amt)| *amt > 0).collect())
    })?;

    println!("   loaded {} real wallets with nonzero balance", balances.len());

    // Order A: as-loaded (effectively HashMap iteration order — unspecified/arbitrary)
    let order_a = balances.clone();

    // Order B: fully reversed
    let mut order_b = balances.clone();
    order_b.reverse();

    // Order C: deterministic pseudo-random shuffle (genuinely different permutation)
    let mut order_c = balances.clone();
    deterministic_shuffle(&mut order_c);

    // Order D: sorted by address ascending (the "canonical" order the v1 flat-hash uses)
    let mut order_d = balances.clone();
    order_d.sort_by_key(|(addr, _)| *addr);

    let tmp_a = std::env::temp_dir().join(format!("smt-det-a-{}", std::process::id()));
    let tmp_b = std::env::temp_dir().join(format!("smt-det-b-{}", std::process::id()));
    let tmp_c = std::env::temp_dir().join(format!("smt-det-c-{}", std::process::id()));
    let tmp_d = std::env::temp_dir().join(format!("smt-det-d-{}", std::process::id()));

    println!("\n   Building 4 independent trees, 4 different insertion orders...");
    let root_a = build_fresh_smt(&tmp_a, &order_a)?;
    println!("   order A (as-loaded):        {}", hex::encode(root_a));
    let root_b = build_fresh_smt(&tmp_b, &order_b)?;
    println!("   order B (reversed):         {}", hex::encode(root_b));
    let root_c = build_fresh_smt(&tmp_c, &order_c)?;
    println!("   order C (shuffled):         {}", hex::encode(root_c));
    let root_d = build_fresh_smt(&tmp_d, &order_d)?;
    println!("   order D (sorted by addr):   {}", hex::encode(root_d));

    let _ = std::fs::remove_dir_all(&tmp_a);
    let _ = std::fs::remove_dir_all(&tmp_b);
    let _ = std::fs::remove_dir_all(&tmp_c);
    let _ = std::fs::remove_dir_all(&tmp_d);

    println!("\n═══════════════════════════════════════════════════════════");
    if root_a == root_b && root_b == root_c && root_c == root_d {
        println!("✅ DETERMINISTIC: all 4 insertion orders produced the IDENTICAL root.");
        println!("   This is the property two independently-synced nodes depend on to agree.");
    } else {
        println!("❌ NON-DETERMINISTIC: insertion order changed the final root!");
        println!("   This WOULD break cross-node agreement even with identical balances.");
        std::process::exit(1);
    }

    Ok(())
}
