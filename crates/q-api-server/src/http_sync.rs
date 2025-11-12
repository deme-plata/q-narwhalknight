// crates/q-api-server/src/http_sync.rs
//
// v0.9.59-beta: HTTP Fallback Sync for Gap Filling
//
// This module implements HTTP-based blockchain synchronization as a fallback
// when P2P turbo sync fails or for large gaps (>1000 blocks).

use anyhow::{Context, Result};
use q_storage::QStorage;
use q_types::SignedBlock;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Bootstrap genesis block from HTTP endpoint if database is empty
pub async fn bootstrap_genesis_if_needed(storage: Arc<QStorage>) -> Result<()> {
    let current_height = storage.get_latest_height().await.unwrap_or(0);

    if current_height == 0 {
        info!("🌱 Fresh database detected (height 0) - bootstrapping genesis block");

        let bootstrap_url = std::env::var("Q_BOOTSTRAP_URL")
            .unwrap_or_else(|_| "http://185.182.185.227:8080".to_string());

        let url = format!("{}/api/v1/blocks/1", bootstrap_url);

        info!("📥 Fetching genesis block from {}", url);

        match reqwest::get(&url).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<crate::handlers::ApiResponse<SignedBlock>>().await {
                        Ok(api_resp) => {
                            if let Some(genesis) = api_resp.data {
                                storage.insert_block(&genesis).await?;
                                info!("✅ Genesis block bootstrapped successfully (height 1)");
                                info!("   Block hash: {}", hex::encode(&genesis.header.hash));
                            } else {
                                error!("❌ Genesis block fetch returned empty data");
                            }
                        }
                        Err(e) => {
                            error!("❌ Failed to parse genesis block response: {}", e);
                        }
                    }
                } else {
                    error!("❌ Bootstrap server returned error: {}", resp.status());
                }
            }
            Err(e) => {
                error!("❌ Failed to fetch genesis block: {}", e);
                error!("   Bootstrap URL: {}", url);
                error!("   This node cannot start without genesis block!");
                error!("   Solutions:");
                error!("   1. Ensure {} is accessible", bootstrap_url);
                error!("   2. Set Q_BOOTSTRAP_URL environment variable to a working node");
                error!("   3. Manually download genesis block and place in database");
            }
        }
    } else {
        debug!("✅ Database already has genesis block (height: {})", current_height);
    }

    Ok(())
}

/// Fill a gap using HTTP fallback sync
/// Returns number of blocks successfully filled
pub async fn http_gap_fill(
    storage: Arc<QStorage>,
    start_height: u64,
    end_height: u64,
) -> Result<u64> {
    let gap_size = end_height - start_height + 1;

    info!("📊 [HTTP SYNC] Gap fill requested: {} → {} ({} blocks)",
          start_height, end_height, gap_size);

    // Determine bootstrap URL
    let bootstrap_url = std::env::var("Q_BOOTSTRAP_URL")
        .unwrap_or_else(|_| "http://185.182.185.227:8080".to_string());

    info!("📡 [HTTP SYNC] Using bootstrap node: {}", bootstrap_url);

    let mut filled = 0;
    let mut failed_heights = Vec::new();

    // Strategy: For large gaps, use batched requests with progress reporting
    let batch_size = if gap_size > 10000 { 100 } else if gap_size > 1000 { 50 } else { 10 };

    for height in start_height..=end_height {
        let url = format!("{}/api/v1/blocks/{}", bootstrap_url, height);

        match fetch_block_with_retry(&url, height, 3).await {
            Ok(block) => {
                // Validate block height matches
                if block.header.height != height {
                    warn!("⚠️ [HTTP SYNC] Height mismatch at {}: expected {}, got {}",
                          height, height, block.header.height);
                    failed_heights.push(height);
                    continue;
                }

                // Insert block
                match storage.insert_block(&block).await {
                    Ok(_) => {
                        filled += 1;

                        // Progress reporting every batch_size blocks
                        if filled % batch_size == 0 || filled == gap_size {
                            let percent = (filled as f64 / gap_size as f64) * 100.0;
                            info!("📥 [HTTP SYNC] Progress: {}/{} blocks ({:.1}%) - height {}",
                                  filled, gap_size, percent, height);
                        }
                    }
                    Err(e) => {
                        warn!("⚠️ [HTTP SYNC] Failed to insert block at height {}: {}", height, e);
                        failed_heights.push(height);
                    }
                }
            }
            Err(e) => {
                warn!("⚠️ [HTTP SYNC] Failed to fetch block at height {}: {}", height, e);
                failed_heights.push(height);

                // If we're failing too much, slow down
                if failed_heights.len() > 10 {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }

        // Rate limiting: don't hammer the bootstrap server
        if filled % 100 == 0 && filled > 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    // Summary
    if filled > 0 {
        info!("✅ [HTTP SYNC] Filled {} blocks successfully", filled);
    }

    if !failed_heights.is_empty() {
        warn!("⚠️ [HTTP SYNC] Failed to fetch {} blocks: {:?}",
              failed_heights.len(),
              if failed_heights.len() <= 10 { failed_heights.clone() } else { failed_heights[..10].to_vec() }
        );
    }

    Ok(filled)
}

/// Fetch a single block with retry logic
async fn fetch_block_with_retry(url: &str, height: u64, max_retries: u32) -> Result<SignedBlock> {
    let mut attempt = 0;

    loop {
        attempt += 1;

        match reqwest::get(url).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<crate::handlers::ApiResponse<SignedBlock>>().await {
                        Ok(api_resp) => {
                            if let Some(block) = api_resp.data {
                                return Ok(block);
                            } else {
                                return Err(anyhow::anyhow!("Empty response data"));
                            }
                        }
                        Err(e) => {
                            if attempt >= max_retries {
                                return Err(anyhow::anyhow!("Parse error after {} retries: {}", max_retries, e));
                            }
                            debug!("⚠️ Parse error for height {}, retry {}/{}", height, attempt, max_retries);
                            tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                        }
                    }
                } else {
                    return Err(anyhow::anyhow!("HTTP error: {}", resp.status()));
                }
            }
            Err(e) => {
                if attempt >= max_retries {
                    return Err(anyhow::anyhow!("Network error after {} retries: {}", max_retries, e));
                }
                debug!("⚠️ Network error for height {}, retry {}/{}", height, attempt, max_retries);
                tokio::time::sleep(Duration::from_millis(200 * attempt as u64)).await;
            }
        }
    }
}

/// Check if a gap is large enough to warrant HTTP fallback
pub fn should_use_http_fallback(gap_size: u64) -> bool {
    // Use HTTP fallback for gaps larger than 1000 blocks
    // or if P2P has failed multiple times (tracked externally)
    gap_size > 1000
}

/// Spawn background task to continuously monitor and fill gaps
pub async fn spawn_gap_monitor(storage: Arc<QStorage>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // ✅ v0.9.76-beta: More aggressive monitoring for early blocks
        // Check every 5 seconds instead of 30 to catch critical early block gaps quickly
        let mut check_interval = tokio::time::interval(Duration::from_secs(5));
        let mut consecutive_failures = 0;

        loop {
            check_interval.tick().await;

            let current_height = storage.get_latest_height().await.unwrap_or(0);

            // Check for gaps
            match storage.get_first_missing_height().await {
                Ok(Some(missing_height)) => {
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    info!("🚨 [GAP MONITOR] CRITICAL GAP DETECTED!");
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    info!("   Missing height: {}", missing_height);
                    info!("   Current height: {}", current_height);
                    info!("   Gap size: {} blocks", current_height.saturating_sub(missing_height));
                    info!("");

                    // ✅ v0.9.76-beta: AGGRESSIVE early block sync
                    // For early blocks (<100), always use HTTP fallback immediately
                    // For later blocks, use smarter batching
                    let gap_end = if missing_height < 100 {
                        // Early blocks: Fetch a small batch aggressively
                        missing_height + 50
                    } else {
                        // Later blocks: Use larger batches
                        missing_height + 1000
                    };

                    info!("📡 [GAP MONITOR] Starting HTTP fallback sync {} → {}", missing_height, gap_end);

                    match http_gap_fill(storage.clone(), missing_height, gap_end).await {
                        Ok(filled) => {
                            info!("✅ [GAP MONITOR] Successfully filled {} blocks via HTTP", filled);
                            info!("   Node should now advance from height {}", current_height);
                            consecutive_failures = 0;
                        }
                        Err(e) => {
                            consecutive_failures += 1;
                            error!("❌ [GAP MONITOR] HTTP gap fill failed (attempt {}): {}", consecutive_failures, e);

                            // Back off if we're failing repeatedly
                            if consecutive_failures > 5 {
                                warn!("⚠️ [GAP MONITOR] Too many failures - backing off for 30 seconds");
                                tokio::time::sleep(Duration::from_secs(30)).await;
                            }
                        }
                    }

                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                }
                Ok(None) => {
                    if current_height % 100 == 0 && current_height > 0 {
                        debug!("✅ [GAP MONITOR] No gaps detected (height: {})", current_height);
                    }
                    consecutive_failures = 0;
                }
                Err(e) => {
                    warn!("⚠️ [GAP MONITOR] Failed to check for gaps: {}", e);
                }
            }
        }
    })
}
