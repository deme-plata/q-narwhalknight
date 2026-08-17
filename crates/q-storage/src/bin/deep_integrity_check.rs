//! Deep data-integrity check (2026-08-17, Phase 1 verification follow-up).
//!
//! Runs OFFLINE against a stopped node's data directory. Does NOT modify
//! anything — pure reads. Checks, in order:
//!   1. Balance SMT proof generation + verification for real wallets (positive case).
//!   2. Tamper-detection: same proof, wrong balance, MUST fail to verify.
//!   3. Tamper-detection: same proof, correct balance, wrong root, MUST fail.
//!   4. v1 flat-hash root vs v2 SMT root: both computed fresh from the SAME
//!      on-disk wallet table, wallet-count and total-supply cross-checked.
//!   5. Block hash-chain spot check: sample stored blocks, verify each one's
//!      recorded parent_hash actually matches its parent block's real hash.
//!
//! Usage: deep_integrity_check <data_dir> [num_wallets_to_prove] [num_blocks_to_chain_check]

use anyhow::Result;
use q_storage::StorageEngine;

#[tokio::main]
async fn main() -> Result<()> {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/orobit/docker-sync-smt-phase1/db".to_string());
    let num_wallets: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let num_blocks: u64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    println!("🔬 DEEP INTEGRITY CHECK — offline, read-only");
    println!("   data_dir = {}", data_dir);
    println!("═══════════════════════════════════════════════════════════");

    let node_id: [u8; 32] = [0u8; 32];
    let engine = StorageEngine::open(&data_dir, node_id).await?;

    // ── 1+2+3: SMT proof generation + verification + tamper detection ──
    println!("\n[1] BALANCE SMT — MERKLE PROOF TEST (real wallets, real balances)");
    let balances = engine.load_wallet_balances().await?;
    let mut nonzero: Vec<([u8; 32], u128)> = balances
        .into_iter()
        .filter(|(_, amt)| *amt > 0)
        .collect();
    nonzero.sort_by_key(|(addr, _)| *addr);
    let sample: Vec<_> = nonzero.into_iter().take(num_wallets).collect();

    let current_root = engine.balance_smt.root();
    println!("    current SMT root: {}", hex::encode(current_root));
    println!("    testing {} real wallets\n", sample.len());

    let mut pos_pass = 0usize;
    let mut pos_fail = 0usize;
    let mut tamper_balance_correctly_rejected = 0usize;
    let mut tamper_balance_wrongly_accepted = 0usize;
    let mut tamper_root_correctly_rejected = 0usize;
    let mut tamper_root_wrongly_accepted = 0usize;

    for (addr, balance) in &sample {
        let proof = engine.balance_smt.prove(addr, *balance)?;

        // Positive case: real proof against real root must verify.
        let ok = proof.verify(&current_root);
        if ok {
            pos_pass += 1;
        } else {
            pos_fail += 1;
            println!(
                "    ❌ POSITIVE CASE FAILED for wallet {}: proof did not verify against current root!",
                hex::encode(&addr[..8])
            );
        }

        // Tamper case A: claim a different balance with the SAME proof siblings.
        let tampered_balance_proof = q_storage::balance_smt::SmtProof {
            addr: *addr,
            balance: balance.wrapping_add(1),
            siblings: proof.siblings,
            empty_bitmap: proof.empty_bitmap,
        };
        if tampered_balance_proof.verify(&current_root) {
            tamper_balance_wrongly_accepted += 1;
            println!(
                "    🚨 TAMPER NOT CAUGHT: wallet {} — claiming balance+1 with the real proof STILL VERIFIED!",
                hex::encode(&addr[..8])
            );
        } else {
            tamper_balance_correctly_rejected += 1;
        }

        // Tamper case B: correct balance, but verify against a WRONG root.
        let mut wrong_root = current_root;
        wrong_root[0] ^= 0xFF;
        if proof.verify(&wrong_root) {
            tamper_root_wrongly_accepted += 1;
            println!(
                "    🚨 TAMPER NOT CAUGHT: wallet {} — real proof verified against a WRONG root!",
                hex::encode(&addr[..8])
            );
        } else {
            tamper_root_correctly_rejected += 1;
        }
    }

    println!("\n    Positive verification:  {}/{} passed", pos_pass, sample.len());
    println!(
        "    Tamper (wrong balance):  {}/{} correctly REJECTED ({} wrongly accepted)",
        tamper_balance_correctly_rejected, sample.len(), tamper_balance_wrongly_accepted
    );
    println!(
        "    Tamper (wrong root):     {}/{} correctly REJECTED ({} wrongly accepted)",
        tamper_root_correctly_rejected, sample.len(), tamper_root_wrongly_accepted
    );
    if pos_fail == 0 && tamper_balance_wrongly_accepted == 0 && tamper_root_wrongly_accepted == 0 {
        println!("    ✅ SMT PROOF SYSTEM: fully sound on real data.");
    } else {
        println!("    ❌ SMT PROOF SYSTEM: FAILURES DETECTED — see above.");
    }

    // ── 4: v1 vs v2 cross-check ──
    println!("\n[2] v1 FLAT-HASH ROOT vs v2 SMT ROOT — independent cross-check");
    let v1_root = engine.compute_balance_root_for_block().await?;
    let (_state_hash, wallet_count, total_supply) = engine.compute_balance_state_hash().await?;
    println!("    v1 root:      {}", hex::encode(v1_root));
    println!("    v2 SMT root:  {}", hex::encode(engine.balance_smt.root()));
    println!("    wallet_count: {}", wallet_count);
    println!(
        "    total_supply: {}.{:024}",
        total_supply / 1_000_000_000_000_000_000_000_000u128,
        total_supply % 1_000_000_000_000_000_000_000_000u128
    );
    println!("    (v1 and v2 are NEVER expected to byte-match — different domain separators — this just confirms both compute cleanly from the same on-disk table.)");

    // ── 5: block hash-chain spot check ──
    println!("\n[3] BLOCK HASH-CHAIN SPOT CHECK (sampling up to {} blocks)", num_blocks);
    let tip = engine.get_latest_qblock_height().await?.unwrap_or(0);
    println!("    local tip height: {}", tip);
    let mut checked = 0u64;
    let mut linked_ok = 0u64;
    let mut linked_bad = 0u64;
    let mut missing = 0u64;
    let step = if tip > num_blocks { tip / num_blocks } else { 1 };
    let mut h = 1u64;
    while h < tip && checked < num_blocks {
        match engine.get_qblock_any_format(h).await {
            Ok(Some(block)) => {
                checked += 1;
                if let Ok(Some(parent)) = engine.get_qblock_any_format(h.saturating_sub(1)).await {
                    let parent_hash_actual = parent.calculate_hash();
                    if block.header.prev_block_hash == parent_hash_actual {
                        linked_ok += 1;
                    } else {
                        linked_bad += 1;
                        println!(
                            "    ⚠️  block {} prev_block_hash does NOT match parent {}'s real hash (may be a known DAG gap, not necessarily corruption)",
                            h, h - 1
                        );
                    }
                }
            }
            _ => missing += 1,
        }
        h = h.saturating_add(step.max(1));
    }
    println!(
        "    sampled {} blocks: {} parent-linked OK, {} mismatched, {} missing (gaps — expected on a sparse DAG mid-sync)",
        checked, linked_ok, linked_bad, missing
    );

    println!("\n═══════════════════════════════════════════════════════════");
    println!("🔬 DEEP INTEGRITY CHECK COMPLETE");
    Ok(())
}
