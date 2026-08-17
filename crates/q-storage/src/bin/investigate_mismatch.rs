//! Targeted investigation of a specific block's prev_block_hash mismatch.
//! Usage: investigate_mismatch <data_dir> <height1> [<height2> ...]

use anyhow::Result;
use q_storage::StorageEngine;

#[tokio::main]
async fn main() -> Result<()> {
    let data_dir = std::env::args().nth(1).unwrap();
    let heights: Vec<u64> = std::env::args()
        .skip(2)
        .filter_map(|s| s.parse().ok())
        .collect();

    let node_id: [u8; 32] = [0u8; 32];
    let engine = StorageEngine::open(&data_dir, node_id).await?;

    for h in heights {
        println!("═══════════════════════════════════════════════");
        println!("Investigating height {}", h);
        match engine.get_qblock_any_format(h).await {
            Ok(Some(block)) => {
                println!("  block {} EXISTS. header:", h);
                println!("    height:           {}", block.header.height);
                println!("    prev_block_hash:  {}", hex::encode(block.header.prev_block_hash));
                println!("    timestamp:        {}", block.header.timestamp);
                println!("    solutions_root:   {}", hex::encode(block.header.solutions_root));
                println!("    dag_round:        {:?}", block.header.dag_round);

                match engine.get_qblock_any_format(h.saturating_sub(1)).await {
                    Ok(Some(parent)) => {
                        let real_parent_hash = parent.calculate_hash();
                        println!("  parent {} EXISTS. header:", h - 1);
                        println!("    height:           {}", parent.header.height);
                        println!("    real hash:        {}", hex::encode(real_parent_hash));
                        println!("    parent's own prev_block_hash: {}", hex::encode(parent.header.prev_block_hash));
                        println!("    MATCH? {}", block.header.prev_block_hash == real_parent_hash);

                        // Check if block's prev_block_hash matches ANY nearby block's hash instead
                        // (would indicate a DAG multi-parent / reorg situation rather than corruption)
                        for probe in (h.saturating_sub(5))..=(h.saturating_sub(1)) {
                            if let Ok(Some(candidate)) = engine.get_qblock_any_format(probe).await {
                                let cand_hash = candidate.calculate_hash();
                                if cand_hash == block.header.prev_block_hash {
                                    println!("    ✅ FOUND: block {}'s prev_block_hash actually matches block {}'s real hash (not a strict h-1 linear chain here — consistent with DAG structure, NOT corruption)", h, probe);
                                }
                            }
                        }
                    }
                    Ok(None) => println!("  parent {} is MISSING from local storage (gap)", h - 1),
                    Err(e) => println!("  parent {} read ERROR: {}", h - 1, e),
                }
            }
            Ok(None) => println!("  block {} is MISSING from local storage", h),
            Err(e) => println!("  block {} read ERROR: {}", h, e),
        }
    }

    Ok(())
}
