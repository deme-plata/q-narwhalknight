//! Centralized Solution Submitter
//!
//! All mining threads send solutions to a single mpsc channel.
//! One tokio task deduplicates by solution hash and submits exactly once
//! via HTTP POST (with fallback) + optional P2P broadcast.
//!
//! This reduces bandwidth from N×threads submissions per solution to 1.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::shared_state::DiagnosticEvent;

/// Message sent from mining threads to the centralized submitter.
#[derive(Clone)]
pub struct SolutionMessage {
    pub solution_json: serde_json::Value,
    pub solution_hash: [u8; 32],
    pub block_height: u64,
    pub nonce: u64,
    /// P2P submission payload (if P2P is enabled)
    pub p2p_submission: Option<q_types::mining_solution::P2PMiningSubmission>,
}

/// Centralized solution submitter — receives from all threads, deduplicates, submits once.
pub struct SolutionSubmitter {
    rx: mpsc::UnboundedReceiver<SolutionMessage>,
    client: reqwest::Client,
    primary_url: String,
    fallback_url: String,
    p2p_tx: Option<mpsc::UnboundedSender<q_types::mining_solution::P2PMiningSubmission>>,
    event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
    solutions_found: Arc<AtomicU64>,
    blocks_mined: Arc<AtomicU64>,
    /// Shared upload byte counter for bandwidth tracking (same as GLOBAL_BYTES_UPLOADED)
    bytes_uploaded: Arc<AtomicU64>,
}

impl SolutionSubmitter {
    pub fn new(
        rx: mpsc::UnboundedReceiver<SolutionMessage>,
        client: reqwest::Client,
        primary_url: String,
        fallback_url: String,
        p2p_tx: Option<mpsc::UnboundedSender<q_types::mining_solution::P2PMiningSubmission>>,
        event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
        solutions_found: Arc<AtomicU64>,
        blocks_mined: Arc<AtomicU64>,
        bytes_uploaded: Arc<AtomicU64>,
    ) -> Self {
        Self {
            rx,
            client,
            primary_url,
            fallback_url,
            p2p_tx,
            event_tx,
            solutions_found,
            blocks_mined,
            bytes_uploaded,
        }
    }

    /// Run the submitter loop. Spawned as a tokio task.
    pub async fn run(mut self) {
        let mut seen: HashSet<[u8; 32]> = HashSet::new();
        let mut batch: Vec<SolutionMessage> = Vec::new();

        loop {
            // Drain all available solutions (non-blocking after first)
            match self.rx.recv().await {
                Some(msg) => batch.push(msg),
                None => return, // All senders dropped — mining stopped
            }
            // Drain any additional queued solutions (500ms batch window)
            let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
            loop {
                match tokio::time::timeout_at(deadline, self.rx.recv()).await {
                    Ok(Some(msg)) => batch.push(msg),
                    _ => break,
                }
            }

            // Deduplicate by solution hash
            for msg in batch.drain(..) {
                if seen.contains(&msg.solution_hash) {
                    debug!("Solution deduped (hash already submitted)");
                    continue;
                }
                seen.insert(msg.solution_hash);

                // Submit via HTTP (with retry + fallback)
                self.submit_http(&msg).await;

                // Submit via P2P (single broadcast)
                if let (Some(ref tx), Some(p2p_sub)) = (&self.p2p_tx, msg.p2p_submission) {
                    let _ = tx.send(p2p_sub);
                }
            }

            // Prune dedup set to prevent unbounded growth
            if seen.len() > 2000 {
                seen.clear();
            }
        }
    }

    async fn submit_http(&self, msg: &SolutionMessage) {
        // Track upload bandwidth (~500 bytes per solution)
        let payload_size = msg.solution_json.to_string().len();
        self.bytes_uploaded.fetch_add(payload_size as u64, Ordering::Relaxed);

        let submit_url = format!("{}/api/v1/mining/submit", self.primary_url);
        let fallback_url = format!("{}/api/v1/mining/submit", self.fallback_url);

        // Try primary, then fallback, with up to 3 retries
        for attempt in 0..3u32 {
            let url = if attempt < 2 { &submit_url } else { &fallback_url };

            match self.client
                .post(url)
                .json(&msg.solution_json)
                .timeout(Duration::from_secs(10))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(result) = resp.json::<serde_json::Value>().await {
                        self.process_success(&result, msg.block_height);
                    }
                    return;
                }
                Ok(resp) => {
                    let status = resp.status();
                    debug!("Solution submit attempt {} returned HTTP {} — retrying", attempt + 1, status);
                }
                Err(e) => {
                    debug!("Solution submit attempt {} failed: {} — retrying", attempt + 1, e);
                }
            }

            // Backoff before retry (0s, 2s, 5s)
            if attempt < 2 {
                let delay = if attempt == 0 { 2 } else { 5 };
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }
        }

        warn!("⚠️ Solution submit failed after 3 attempts (mining continues)");
    }

    fn process_success(&self, result: &serde_json::Value, block_height: u64) {
        if let Some(data) = result.get("data") {
            let reward_qnk = data.get("reward_qnk")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if reward_qnk > 0.0 {
                info!("✅ Solution accepted! Earned {} QUG", reward_qnk);
            } else {
                info!("✅ Solution accepted at block #{}", block_height);
            }
            self.solutions_found.fetch_add(1, Ordering::Relaxed);
            self.blocks_mined.fetch_add(1, Ordering::Relaxed);
            let _ = self.event_tx.send(DiagnosticEvent::SolutionAccepted {
                block_height,
                reward_qnk,
            });
            if data.get("update_available").and_then(|v| v.as_bool()).unwrap_or(false) {
                warn!("╔══════════════════════════════════════════════════╗");
                warn!("║  📦 MINER UPDATE AVAILABLE                       ║");
                warn!("║  Your miner v{} may be outdated.              ", env!("CARGO_PKG_VERSION"));
                warn!("║  Download: https://dl.quillon.xyz/downloads/      ║");
                warn!("╚══════════════════════════════════════════════════╝");
            }
            if let Some(notice) = data.get("server_notice").and_then(|v| v.as_str()) {
                if !notice.is_empty() {
                    warn!("[SERVER] {}", notice);
                }
            }
        }
    }
}
