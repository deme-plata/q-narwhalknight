#!/usr/bin/env -S cargo +nightly run --bin migrate_to_phase2 --

//! Phase 2 Migration Utility - Populate Block-Vertex Mappings
//!
//! This utility scans the existing blockchain and populates the block-vertex
//! mappings required for Phase 2 DAG-aware sync.
//!
//! Usage:
//!   cargo run --bin migrate_to_phase2 -- [OPTIONS]
//!
//! Options:
//!   --db-path <PATH>      Path to blockchain database (default: ./data)
//!   --batch-size <SIZE>   Number of blocks to process per batch (default: 1000)
//!   --verify              Verify mappings after migration
//!   --dry-run             Simulate migration without writing to database
//!
//! Example:
//!   cargo run --bin migrate_to_phase2 -- --db-path ./data --batch-size 500 --verify

use anyhow::{Context, Result};
use clap::Parser;
use q_storage::{RocksDBKV, KVStore};
use q_types::Block;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn, error, debug};

#[derive(Parser, Debug)]
#[command(name = "migrate_to_phase2")]
#[command(about = "Migrate blockchain to Phase 2 with block-vertex mappings")]
struct Args {
    /// Path to blockchain database
    #[arg(long, default_value = "./data")]
    db_path: PathBuf,

    /// Batch size for processing blocks
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// Verify mappings after migration
    #[arg(long, default_value = "false")]
    verify: bool,

    /// Dry run (don't write to database)
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Start from specific block height
    #[arg(long, default_value = "0")]
    start_height: u64,

    /// End at specific block height (0 = latest)
    #[arg(long, default_value = "0")]
    end_height: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MigrationStats {
    total_blocks: u64,
    blocks_processed: u64,
    mappings_created: u64,
    batches_written: u64,
    duration_secs: f64,
    blocks_per_sec: f64,
    errors: Vec<String>,
}

impl Default for MigrationStats {
    fn default() -> Self {
        Self {
            total_blocks: 0,
            blocks_processed: 0,
            mappings_created: 0,
            batches_written: 0,
            duration_secs: 0.0,
            blocks_per_sec: 0.0,
            errors: Vec::new(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .init();

    let args = Args::parse();

    info!("🚀 Phase 2 Migration Utility");
    info!("📂 Database path: {}", args.db_path.display());
    info!("📦 Batch size: {}", args.batch_size);
    info!("🔍 Verify after migration: {}", args.verify);
    info!("🧪 Dry run mode: {}", args.dry_run);
    info!("");

    // Open database
    info!("🔓 Opening blockchain database...");
    let kv = Arc::new(
        RocksDBKV::open_hot_db(&args.db_path)
            .await
            .context("Failed to open database")?
    );
    info!("✅ Database opened successfully");

    // Get blockchain height
    let blockchain_height = kv.get_blockchain_height().await?;
    info!("📊 Current blockchain height: {}", blockchain_height);

    if blockchain_height == 0 {
        warn!("⚠️  No blocks found in database. Nothing to migrate.");
        return Ok(());
    }

    // Determine migration range
    let start_height = args.start_height;
    let end_height = if args.end_height == 0 {
        blockchain_height
    } else {
        args.end_height.min(blockchain_height)
    };

    if start_height > end_height {
        error!("❌ Invalid range: start_height ({}) > end_height ({})", start_height, end_height);
        return Err(anyhow::anyhow!("Invalid migration range"));
    }

    let total_blocks = end_height - start_height + 1;
    info!("🎯 Migration range: {} → {} ({} blocks)", start_height, end_height, total_blocks);
    info!("");

    // Perform migration
    let stats = migrate_blocks(
        Arc::clone(&kv),
        start_height,
        end_height,
        args.batch_size,
        args.dry_run,
    )
    .await?;

    // Print statistics
    print_migration_stats(&stats);

    // Verify if requested
    if args.verify && !args.dry_run {
        info!("");
        info!("🔍 Verifying block-vertex mappings...");
        verify_mappings(Arc::clone(&kv), start_height, end_height).await?;
    }

    info!("");
    info!("✅ Phase 2 migration complete!");

    Ok(())
}

async fn migrate_blocks(
    kv: Arc<RocksDBKV>,
    start_height: u64,
    end_height: u64,
    batch_size: usize,
    dry_run: bool,
) -> Result<MigrationStats> {
    let start_time = Instant::now();
    let mut stats = MigrationStats {
        total_blocks: end_height - start_height + 1,
        ..Default::default()
    };

    let mut current_height = start_height;

    while current_height <= end_height {
        let batch_end = (current_height + batch_size as u64 - 1).min(end_height);
        let batch_count = batch_end - current_height + 1;

        debug!("📦 Processing batch: {} → {} ({} blocks)", current_height, batch_end, batch_count);

        // Fetch blocks in batch
        let blocks = fetch_blocks_batch(&kv, current_height, batch_end).await?;

        if blocks.is_empty() {
            warn!("⚠️  No blocks found in range {} → {}", current_height, batch_end);
            current_height = batch_end + 1;
            continue;
        }

        // Extract block-vertex mappings from vertices list
        // Phase 2: Block.vertices contains VertexId ([u8; 32])
        // We convert to u64 by taking first 8 bytes for database storage
        let mut mappings: Vec<(String, u64)> = Vec::new();
        for block in &blocks {
            // Get vertex ID from block's vertices list (Phase 2 DAG-aware blocks)
            if !block.vertices.is_empty() {
                // Use first vertex as primary mapping
                // Convert VertexId ([u8; 32]) to u64 using first 8 bytes
                let vertex_bytes = &block.vertices[0][..8];
                let vertex_id = u64::from_le_bytes(vertex_bytes.try_into().unwrap_or([0u8; 8]));
                let block_hash_hex = hex::encode(&block.hash);
                mappings.push((block_hash_hex, vertex_id));
                stats.mappings_created += 1;
            } else {
                // Block doesn't have vertices (pre-Phase 2 block)
                debug!("Block {:?} (height {}) has no vertices", hex::encode(&block.hash[..8]), block.height);
            }
        }

        // Write mappings to database (unless dry run)
        if !dry_run && !mappings.is_empty() {
            kv.batch_store_mappings(&mappings).await?;
            stats.batches_written += 1;
        }

        stats.blocks_processed += blocks.len() as u64;

        // Progress update
        let progress_pct = (stats.blocks_processed as f64 / stats.total_blocks as f64) * 100.0;
        info!(
            "📊 Progress: {}/{} blocks ({:.1}%) | Mappings: {} | Batches: {}",
            stats.blocks_processed,
            stats.total_blocks,
            progress_pct,
            stats.mappings_created,
            stats.batches_written
        );

        current_height = batch_end + 1;

        // Small delay to avoid overwhelming database
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    stats.duration_secs = start_time.elapsed().as_secs_f64();
    stats.blocks_per_sec = stats.blocks_processed as f64 / stats.duration_secs;

    Ok(stats)
}

async fn fetch_blocks_batch(
    kv: &RocksDBKV,
    start_height: u64,
    end_height: u64,
) -> Result<Vec<Block>> {
    let mut blocks = Vec::new();

    for height in start_height..=end_height {
        match fetch_block_by_height(kv, height).await {
            Ok(Some(block)) => blocks.push(block),
            Ok(None) => {
                debug!("Block at height {} not found", height);
            }
            Err(e) => {
                warn!("❌ Failed to fetch block at height {}: {}", height, e);
            }
        }
    }

    Ok(blocks)
}

async fn fetch_block_by_height(kv: &RocksDBKV, height: u64) -> Result<Option<Block>> {
    // Fetch block by height from CF_BLOCKS
    let key = height.to_be_bytes();

    match kv.get("blocks", &key).await? {
        Some(bytes) => {
            let block: Block = bincode::deserialize(&bytes)
                .context("Failed to deserialize block")?;
            Ok(Some(block))
        }
        None => Ok(None),
    }
}

async fn verify_mappings(
    kv: Arc<RocksDBKV>,
    start_height: u64,
    end_height: u64,
) -> Result<()> {
    let mut verified = 0u64;
    let mut missing = 0u64;
    let mut corrupted = 0u64;

    info!("🔍 Verifying {} blocks...", end_height - start_height + 1);

    for height in start_height..=end_height {
        if let Some(block) = fetch_block_by_height(&kv, height).await? {
            // Check if block has vertices (Phase 2 blocks)
            if !block.vertices.is_empty() {
                // Convert VertexId ([u8; 32]) to u64 using first 8 bytes
                let vertex_bytes = &block.vertices[0][..8];
                let expected_vertex_id = u64::from_le_bytes(vertex_bytes.try_into().unwrap_or([0u8; 8]));
                let block_hash_hex = hex::encode(&block.hash);
                // Verify block → vertex mapping
                match kv.get_vertex_for_block(&block_hash_hex).await? {
                    Some(actual_vertex_id) => {
                        if actual_vertex_id == expected_vertex_id {
                            verified += 1;

                            // Also verify reverse mapping
                            match kv.get_block_for_vertex(actual_vertex_id).await? {
                                Some(actual_hash_hex) => {
                                    if actual_hash_hex != block_hash_hex {
                                        error!(
                                            "❌ Reverse mapping mismatch: vertex {} → block {} (expected {})",
                                            actual_vertex_id, &actual_hash_hex[..16], &block_hash_hex[..16]
                                        );
                                        corrupted += 1;
                                    }
                                }
                                None => {
                                    error!("❌ Reverse mapping missing: vertex {} → ?", actual_vertex_id);
                                    corrupted += 1;
                                }
                            }
                        } else {
                            error!(
                                "❌ Vertex ID mismatch for block {}: expected {}, got {}",
                                &block_hash_hex[..16], expected_vertex_id, actual_vertex_id
                            );
                            corrupted += 1;
                        }
                    }
                    None => {
                        warn!("⚠️  Mapping missing for block {} (vertex {})", &block_hash_hex[..16], expected_vertex_id);
                        missing += 1;
                    }
                }
            }
        }

        // Progress update every 10,000 blocks
        if height % 10000 == 0 {
            info!(
                "📊 Verified: {} | Missing: {} | Corrupted: {}",
                verified, missing, corrupted
            );
        }
    }

    info!("");
    info!("✅ Verification complete:");
    info!("   Verified:  {}", verified);
    info!("   Missing:   {}", missing);
    info!("   Corrupted: {}", corrupted);

    if missing > 0 || corrupted > 0 {
        return Err(anyhow::anyhow!(
            "Verification failed: {} missing, {} corrupted",
            missing,
            corrupted
        ));
    }

    Ok(())
}

fn print_migration_stats(stats: &MigrationStats) {
    info!("");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("📊 Migration Statistics");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("Total blocks:       {}", stats.total_blocks);
    info!("Blocks processed:   {}", stats.blocks_processed);
    info!("Mappings created:   {}", stats.mappings_created);
    info!("Batches written:    {}", stats.batches_written);
    info!("Duration:           {:.2} seconds", stats.duration_secs);
    info!("Processing speed:   {:.2} blocks/sec", stats.blocks_per_sec);

    if !stats.errors.is_empty() {
        info!("");
        warn!("⚠️  Errors encountered: {}", stats.errors.len());
        for (i, error) in stats.errors.iter().enumerate().take(10) {
            warn!("   {}. {}", i + 1, error);
        }
        if stats.errors.len() > 10 {
            warn!("   ... and {} more errors", stats.errors.len() - 10);
        }
    }

    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
