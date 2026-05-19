//! Live-node invariant tests.
//!
//! These tests hit a running q-api-server over HTTP and assert structural +
//! trend properties of `/metrics` and `/api/v1/status`. Use them to catch
//! regressions that pure unit tests can't: e.g., "binary boots but mesh never
//! forms peers", "sync stalls", "API returns wrong network".
//!
//! ## Running
//!
//! All tests are `#[ignore]`-gated so they don't run during normal `cargo test`.
//! Run explicitly:
//!
//! ```bash
//! # Against a local node on port 8080 (default):
//! cargo test --test live_metrics_invariants -- --ignored --nocapture
//!
//! # Against a Docker sync-test container on port 8181:
//! QNK_TEST_TARGET=http://localhost:8181 cargo test --test live_metrics_invariants -- --ignored --nocapture
//!
//! # Against production:
//! QNK_TEST_TARGET=https://quillon.xyz cargo test --test live_metrics_invariants -- --ignored --nocapture
//! ```
//!
//! ## Notes on counter deltas
//!
//! Prometheus counters are cumulative across the process lifetime. Trend tests
//! that want to assert "X advanced over the window" snapshot at t0 and assert
//! delta at t1. Reset is not supported by the prometheus crate.

use std::collections::HashMap;
use std::env;
use std::time::Duration;

const DEFAULT_TARGET: &str = "http://localhost:8080";
const WINDOW_SECS: u64 = 60;

fn target() -> String {
    env::var("QNK_TEST_TARGET").unwrap_or_else(|_| DEFAULT_TARGET.to_string())
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .user_agent("qnk-invariant-test/1.0")
        .build()
        .expect("client builder")
}

async fn fetch_metrics() -> String {
    let url = format!("{}/metrics", target());
    let body = client()
        .get(&url)
        .send()
        .await
        .unwrap_or_else(|e| panic!("GET {url} failed: {e}"))
        .text()
        .await
        .expect("metrics body");
    assert!(
        body.starts_with("# HELP") || body.contains("# TYPE"),
        "{url} did not return Prometheus text format (got {} bytes starting with {:?})",
        body.len(),
        body.chars().take(80).collect::<String>()
    );
    body
}

async fn fetch_status() -> serde_json::Value {
    let url = format!("{}/api/v1/status", target());
    let body = client()
        .get(&url)
        .send()
        .await
        .unwrap_or_else(|e| panic!("GET {url} failed: {e}"))
        .json::<serde_json::Value>()
        .await
        .unwrap_or_else(|e| panic!("status not JSON: {e}"));
    assert_eq!(body["success"], serde_json::json!(true), "/api/v1/status not success");
    body
}

/// Parse a single Prometheus text-format line into (metric_name, labels, value).
/// Handles plain counter/gauge lines; ignores HELP/TYPE/comment lines.
/// Returns `None` for non-data lines.
fn parse_metric_line(line: &str) -> Option<(String, HashMap<String, String>, f64)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (left, value_str) = line.rsplit_once(' ')?;
    let value: f64 = value_str.parse().ok()?;
    if let Some(open) = left.find('{') {
        let name = left[..open].to_string();
        let labels_str = left[open + 1..left.rfind('}').unwrap_or(left.len())].to_string();
        let labels = labels_str
            .split(',')
            .filter_map(|kv| {
                let (k, v) = kv.split_once('=')?;
                Some((
                    k.trim().to_string(),
                    v.trim().trim_matches('"').to_string(),
                ))
            })
            .collect();
        Some((name, labels, value))
    } else {
        Some((left.to_string(), HashMap::new(), value))
    }
}

/// Return ALL samples matching `metric_name` (one per label-set). Useful for gauges
/// with topic labels, etc.
fn samples_for(metrics: &str, metric_name: &str) -> Vec<(HashMap<String, String>, f64)> {
    metrics
        .lines()
        .filter_map(parse_metric_line)
        .filter(|(name, _, _)| name == metric_name)
        .map(|(_, labels, v)| (labels, v))
        .collect()
}

/// Single-value lookup for an unlabelled metric.
fn scalar(metrics: &str, metric_name: &str) -> Option<f64> {
    samples_for(metrics, metric_name)
        .into_iter()
        .find(|(labels, _)| labels.is_empty())
        .map(|(_, v)| v)
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural — assert /metrics + /api/v1/status are well-formed.
// These can run against any healthy node.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn metrics_endpoint_is_well_formed() {
    let m = fetch_metrics().await;
    let lines: Vec<&str> = m.lines().collect();
    assert!(lines.len() > 50, "/metrics body suspiciously short: {} lines", lines.len());

    // Every metric should have HELP+TYPE header(s) before its data lines.
    let help_lines = lines.iter().filter(|l| l.starts_with("# HELP")).count();
    let type_lines = lines.iter().filter(|l| l.starts_with("# TYPE")).count();
    assert!(help_lines > 5, "expected >5 # HELP lines, got {help_lines}");
    assert!(type_lines > 5, "expected >5 # TYPE lines, got {type_lines}");

    // At least one of the well-known qnk_* metrics is present.
    assert!(
        m.contains("qnk_peers_connected"),
        "missing core metric qnk_peers_connected"
    );
}

#[tokio::test]
#[ignore]
async fn status_endpoint_has_required_fields() {
    let s = fetch_status().await;
    let data = &s["data"];
    assert!(data["version"].is_string(), "status missing version");
    assert!(data["peer_id"].is_string(), "status missing peer_id");
    assert!(
        data["peer_id"].as_str().unwrap().starts_with("12D3KooW"),
        "peer_id wrong format: {:?}",
        data["peer_id"]
    );
    assert!(
        data["network_id"].is_string(),
        "status missing network_id"
    );
    assert_eq!(
        data["status"], serde_json::json!("ready"),
        "node not ready: {:?}",
        data["status"]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Connectivity — these will catch the kind of mesh failure we saw 2026-05-18.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn at_least_one_libp2p_peer_after_warmup() {
    // Wait up to 60s for the node to find at least one peer. Captures bootstrap
    // failures (firewall, wrong peer-id allowlist, dead bootstrap nodes).
    let deadline = std::time::Instant::now() + Duration::from_secs(WINDOW_SECS);
    let mut last_seen = 0.0_f64;
    while std::time::Instant::now() < deadline {
        let m = fetch_metrics().await;
        let peers = scalar(&m, "qnk_peers_connected").unwrap_or(0.0);
        last_seen = peers;
        if peers >= 1.0 {
            eprintln!("peers={peers} — invariant holds.");
            return;
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    panic!("qnk_peers_connected stayed at 0 for {WINDOW_SECS}s (last seen {last_seen}). Bootstrap is broken.");
}

#[tokio::test]
#[ignore]
async fn gossipsub_mesh_has_peers_for_blocks_topic() {
    // /metrics exposes qnk_peers_in_gossipsub_mesh per topic. The /blocks topic
    // is the most critical — if it's empty, this node won't see new blocks.
    // Catches the 2026-05-18 mesh failure.
    let deadline = std::time::Instant::now() + Duration::from_secs(WINDOW_SECS);
    while std::time::Instant::now() < deadline {
        let m = fetch_metrics().await;
        let mesh = samples_for(&m, "qnk_peers_in_gossipsub_mesh");
        let blocks = mesh
            .iter()
            .find(|(labels, _)| labels.get("topic").is_some_and(|t| t.contains("/blocks")));
        if let Some((_, count)) = blocks {
            if *count >= 1.0 {
                eprintln!("/blocks mesh has {count} peer(s) — invariant holds.");
                return;
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    panic!(
        "no peers in /blocks gossipsub mesh after {WINDOW_SECS}s — turbo-sync + new-block delivery will be broken"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Trend — assert sync is actually making forward progress.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn sync_height_advances_within_window() {
    // For a fresh-syncing node, height should monotonically increase.
    // We snapshot height now and require strict increase within the window.
    // For an already-synced node, this would be trivially true if blocks
    // continue to arrive. If blocks AREN'T arriving the node is dead in the
    // water and we want to know.
    let s0 = fetch_status().await;
    let h0 = s0["data"]["upgrades"]["current_height"]
        .as_u64()
        .or_else(|| s0["data"]["current_height"].as_u64())
        .expect("status missing current_height");

    eprintln!("starting height: {h0}, waiting {WINDOW_SECS}s for progress");
    tokio::time::sleep(Duration::from_secs(WINDOW_SECS)).await;

    let s1 = fetch_status().await;
    let h1 = s1["data"]["upgrades"]["current_height"]
        .as_u64()
        .or_else(|| s1["data"]["current_height"].as_u64())
        .expect("status missing current_height");

    assert!(
        h1 > h0,
        "height did not advance in {WINDOW_SECS}s (h0={h0}, h1={h1}). Sync is stalled."
    );
    eprintln!("height advanced {h0} → {h1} (+{})", h1 - h0);
}

#[tokio::test]
#[ignore]
async fn bootstrap_dial_attempts_increase() {
    // qnk_bootstrap_dial_total should be incrementing — even a healthy node
    // periodically refreshes Kademlia. If this is flat, the bootstrap loop is dead.
    let m0 = fetch_metrics().await;
    let d0: f64 = samples_for(&m0, "qnk_bootstrap_dial_total")
        .iter()
        .map(|(_, v)| v)
        .sum();
    tokio::time::sleep(Duration::from_secs(WINDOW_SECS)).await;
    let m1 = fetch_metrics().await;
    let d1: f64 = samples_for(&m1, "qnk_bootstrap_dial_total")
        .iter()
        .map(|(_, v)| v)
        .sum();

    // For a node with existing peers, dial attempts may pause. So we only
    // require strict increase when peers < 3 (still bootstrapping).
    let peers = scalar(&m1, "qnk_peers_connected").unwrap_or(0.0);
    if peers < 3.0 {
        assert!(
            d1 > d0,
            "bootstrap dials flat at peers={peers} (d0={d0}, d1={d1}) — bootstrap loop is dead"
        );
        eprintln!("bootstrap dials advanced {d0} → {d1}");
    } else {
        eprintln!("peers={peers} >= 3, skipping bootstrap-dial-increase check");
    }
}
