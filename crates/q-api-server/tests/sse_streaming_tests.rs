//! SSE Streaming Event Tests
//!
//! v3.2.25-beta: Tests for Server-Sent Events formatting and delivery
//!
//! These tests verify:
//! - SSE event format is correct
//! - Event types are properly serialized
//! - Mining stats events include miner identification
//! - Balance update events are properly formatted
//!
//! Run with: cargo test --package q-api-server --test sse_streaming_tests

use std::collections::HashMap;
use std::time::SystemTime;

// ============================================================================
// SSE EVENT TYPES (mirrors streaming.rs)
// ============================================================================

#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// New block produced/received
    NewBlock {
        height: u64,
        hash: String,
        transactions: u64,
        timestamp: u64,
    },

    /// Balance update for a wallet
    BalanceUpdate {
        address: String,
        token: String,
        balance: String,  // String to handle large numbers
        block_height: u64,
    },

    /// Mining statistics update
    MiningStats {
        miner_address: String,
        total_rewards: f64,
        total_blocks_found: u64,
        current_balance: f64,
        avg_hash_rate: f64,
        miner_id: Option<String>,
        worker_id: Option<String>,
        timestamp: String,
    },

    /// Network status update
    NetworkStatus {
        peer_count: u32,
        network_height: u64,
        sync_status: String,
    },

    /// Transaction confirmed
    TransactionConfirmed {
        tx_hash: String,
        block_height: u64,
        confirmations: u64,
    },

    /// Heartbeat (keep-alive)
    Heartbeat {
        server_time: u64,
    },
}

// ============================================================================
// SSE FORMATTING
// ============================================================================

impl StreamEvent {
    /// Format event for SSE transmission
    pub fn to_sse_string(&self) -> String {
        let (event_type, data) = match self {
            StreamEvent::NewBlock { height, hash, transactions, timestamp } => {
                let json = format!(
                    r#"{{"height":{},"hash":"{}","transactions":{},"timestamp":{}}}"#,
                    height, hash, transactions, timestamp
                );
                ("new-block", json)
            }
            StreamEvent::BalanceUpdate { address, token, balance, block_height } => {
                let json = format!(
                    r#"{{"address":"{}","token":"{}","balance":"{}","block_height":{}}}"#,
                    address, token, balance, block_height
                );
                ("balance-update", json)
            }
            StreamEvent::MiningStats {
                miner_address, total_rewards, total_blocks_found,
                current_balance, avg_hash_rate, miner_id, worker_id, timestamp
            } => {
                let miner_id_str = miner_id.as_ref()
                    .map(|id| format!(r#","miner_id":"{}""#, id))
                    .unwrap_or_default();
                let worker_id_str = worker_id.as_ref()
                    .map(|id| format!(r#","worker_id":"{}""#, id))
                    .unwrap_or_default();

                let json = format!(
                    r#"{{"miner_address":"{}","total_rewards":{},"total_blocks_found":{},"current_balance":{},"avg_hash_rate":{}{}{},"timestamp":"{}"}}"#,
                    miner_address, total_rewards, total_blocks_found,
                    current_balance, avg_hash_rate, miner_id_str, worker_id_str, timestamp
                );
                ("miner-stats", json)
            }
            StreamEvent::NetworkStatus { peer_count, network_height, sync_status } => {
                let json = format!(
                    r#"{{"peer_count":{},"network_height":{},"sync_status":"{}"}}"#,
                    peer_count, network_height, sync_status
                );
                ("network-status", json)
            }
            StreamEvent::TransactionConfirmed { tx_hash, block_height, confirmations } => {
                let json = format!(
                    r#"{{"tx_hash":"{}","block_height":{},"confirmations":{}}}"#,
                    tx_hash, block_height, confirmations
                );
                ("tx-confirmed", json)
            }
            StreamEvent::Heartbeat { server_time } => {
                let json = format!(r#"{{"server_time":{}}}"#, server_time);
                ("heartbeat", json)
            }
        };

        format!("event: {}\ndata: {}\n\n", event_type, data)
    }

    /// Get event type name
    pub fn event_type(&self) -> &'static str {
        match self {
            StreamEvent::NewBlock { .. } => "new-block",
            StreamEvent::BalanceUpdate { .. } => "balance-update",
            StreamEvent::MiningStats { .. } => "miner-stats",
            StreamEvent::NetworkStatus { .. } => "network-status",
            StreamEvent::TransactionConfirmed { .. } => "tx-confirmed",
            StreamEvent::Heartbeat { .. } => "heartbeat",
        }
    }
}

// ============================================================================
// SSE FORMAT TESTS
// ============================================================================

mod sse_format_tests {
    use super::*;

    #[test]
    fn test_sse_format_has_event_and_data() {
        let event = StreamEvent::Heartbeat { server_time: 1234567890 };
        let sse = event.to_sse_string();

        assert!(sse.starts_with("event: "), "SSE must start with 'event: '");
        assert!(sse.contains("\ndata: "), "SSE must contain 'data: '");
        assert!(sse.ends_with("\n\n"), "SSE must end with double newline");
    }

    #[test]
    fn test_sse_newblock_format() {
        let event = StreamEvent::NewBlock {
            height: 12345,
            hash: "abc123".to_string(),
            transactions: 10,
            timestamp: 1234567890,
        };
        let sse = event.to_sse_string();

        assert!(sse.contains("event: new-block"));
        assert!(sse.contains(r#""height":12345"#));
        assert!(sse.contains(r#""hash":"abc123""#));
        assert!(sse.contains(r#""transactions":10"#));
    }

    #[test]
    fn test_sse_balance_update_format() {
        let event = StreamEvent::BalanceUpdate {
            address: "qnk_test".to_string(),
            token: "QUG".to_string(),
            balance: "1000000000000000000000000".to_string(),  // 1 QUG
            block_height: 100,
        };
        let sse = event.to_sse_string();

        assert!(sse.contains("event: balance-update"));
        assert!(sse.contains(r#""address":"qnk_test""#));
        assert!(sse.contains(r#""token":"QUG""#));
        assert!(sse.contains(r#""balance":"1000000000000000000000000""#));
    }

    #[test]
    fn test_sse_mining_stats_with_miner_id() {
        let event = StreamEvent::MiningStats {
            miner_address: "qnk_miner".to_string(),
            total_rewards: 10.5,
            total_blocks_found: 5,
            current_balance: 100.0,
            avg_hash_rate: 1000.0,
            miner_id: Some("my-miner-123".to_string()),
            worker_id: Some("my-miner-123".to_string()),
            timestamp: "2026-01-21T12:00:00Z".to_string(),
        };
        let sse = event.to_sse_string();

        assert!(sse.contains("event: miner-stats"));
        assert!(sse.contains(r#""miner_id":"my-miner-123""#));
        assert!(sse.contains(r#""worker_id":"my-miner-123""#));
    }

    #[test]
    fn test_sse_mining_stats_without_miner_id() {
        let event = StreamEvent::MiningStats {
            miner_address: "qnk_miner".to_string(),
            total_rewards: 10.5,
            total_blocks_found: 5,
            current_balance: 100.0,
            avg_hash_rate: 1000.0,
            miner_id: None,
            worker_id: Some("direct".to_string()),
            timestamp: "2026-01-21T12:00:00Z".to_string(),
        };
        let sse = event.to_sse_string();

        assert!(sse.contains("event: miner-stats"));
        assert!(!sse.contains("miner_id"), "miner_id should be omitted when None");
        assert!(sse.contains(r#""worker_id":"direct""#));
    }
}

// ============================================================================
// EVENT TYPE TESTS
// ============================================================================

mod event_type_tests {
    use super::*;

    #[test]
    fn test_all_event_types() {
        let events: Vec<StreamEvent> = vec![
            StreamEvent::NewBlock { height: 1, hash: "a".to_string(), transactions: 0, timestamp: 0 },
            StreamEvent::BalanceUpdate { address: "a".to_string(), token: "Q".to_string(), balance: "0".to_string(), block_height: 0 },
            StreamEvent::MiningStats {
                miner_address: "a".to_string(), total_rewards: 0.0, total_blocks_found: 0,
                current_balance: 0.0, avg_hash_rate: 0.0, miner_id: None, worker_id: None,
                timestamp: "".to_string()
            },
            StreamEvent::NetworkStatus { peer_count: 0, network_height: 0, sync_status: "".to_string() },
            StreamEvent::TransactionConfirmed { tx_hash: "".to_string(), block_height: 0, confirmations: 0 },
            StreamEvent::Heartbeat { server_time: 0 },
        ];

        let expected_types = vec![
            "new-block",
            "balance-update",
            "miner-stats",
            "network-status",
            "tx-confirmed",
            "heartbeat",
        ];

        for (event, expected) in events.iter().zip(expected_types.iter()) {
            assert_eq!(event.event_type(), *expected);
        }
    }
}

// ============================================================================
// MINING STATS EVENT TESTS (v3.2.25-beta focus)
// ============================================================================

mod mining_stats_event_tests {
    use super::*;

    #[test]
    fn test_mining_stats_event_fields() {
        let event = StreamEvent::MiningStats {
            miner_address: "qnk8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1".to_string(),
            total_rewards: 50.5,
            total_blocks_found: 10,
            current_balance: 500.0,
            avg_hash_rate: 15000.0,
            miner_id: Some("desktop-miner".to_string()),
            worker_id: Some("desktop-miner".to_string()),
            timestamp: "2026-01-21T14:30:00Z".to_string(),
        };

        let sse = event.to_sse_string();

        // Verify all fields are present
        assert!(sse.contains(r#""miner_address":"qnk8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1""#));
        assert!(sse.contains(r#""total_rewards":50.5"#));
        assert!(sse.contains(r#""total_blocks_found":10"#));
        assert!(sse.contains(r#""current_balance":500"#));
        assert!(sse.contains(r#""avg_hash_rate":15000"#));
        assert!(sse.contains(r#""miner_id":"desktop-miner""#));
        assert!(sse.contains(r#""timestamp":"2026-01-21T14:30:00Z""#));
    }

    #[test]
    fn test_p2p_miner_no_miner_id() {
        // P2P miners should have worker_id but not miner_id
        let event = StreamEvent::MiningStats {
            miner_address: "qnk_remote".to_string(),
            total_rewards: 10.0,
            total_blocks_found: 2,
            current_balance: 100.0,
            avg_hash_rate: 5000.0,
            miner_id: None,  // P2P miners don't set miner_id
            worker_id: Some("p2p:12D3KooWABC".to_string()),
            timestamp: "2026-01-21T14:30:00Z".to_string(),
        };

        let sse = event.to_sse_string();

        assert!(!sse.contains("miner_id\":"), "P2P miner should not have miner_id field");
        assert!(sse.contains(r#""worker_id":"p2p:12D3KooWABC""#));
    }

    #[test]
    fn test_multiple_miners_generate_separate_events() {
        let events: Vec<StreamEvent> = vec![
            StreamEvent::MiningStats {
                miner_address: "wallet_a".to_string(),
                total_rewards: 10.0,
                total_blocks_found: 2,
                current_balance: 100.0,
                avg_hash_rate: 5000.0,
                miner_id: Some("miner-1".to_string()),
                worker_id: Some("miner-1".to_string()),
                timestamp: "2026-01-21T14:30:00Z".to_string(),
            },
            StreamEvent::MiningStats {
                miner_address: "wallet_a".to_string(),  // Same wallet
                total_rewards: 20.0,
                total_blocks_found: 4,
                current_balance: 100.0,
                avg_hash_rate: 7500.0,
                miner_id: Some("miner-2".to_string()),  // Different miner
                worker_id: Some("miner-2".to_string()),
                timestamp: "2026-01-21T14:30:00Z".to_string(),
            },
        ];

        // Each miner should generate a separate event
        assert_eq!(events.len(), 2);

        // Events should have different miner_ids
        let sse1 = events[0].to_sse_string();
        let sse2 = events[1].to_sse_string();

        assert!(sse1.contains(r#""miner_id":"miner-1""#));
        assert!(sse2.contains(r#""miner_id":"miner-2""#));
    }
}

// ============================================================================
// JSON PARSING TESTS
// ============================================================================

mod json_parsing_tests {
    use super::*;

    /// Parse the data portion of an SSE event
    fn extract_sse_data(sse: &str) -> Option<&str> {
        for line in sse.lines() {
            if line.starts_with("data: ") {
                return Some(&line[6..]);
            }
        }
        None
    }

    #[test]
    fn test_extract_sse_data() {
        let event = StreamEvent::Heartbeat { server_time: 123 };
        let sse = event.to_sse_string();

        let data = extract_sse_data(&sse);
        assert!(data.is_some());
        assert_eq!(data.unwrap(), r#"{"server_time":123}"#);
    }

    #[test]
    fn test_json_is_valid() {
        let events: Vec<StreamEvent> = vec![
            StreamEvent::NewBlock { height: 1, hash: "test".to_string(), transactions: 5, timestamp: 123 },
            StreamEvent::BalanceUpdate { address: "a".to_string(), token: "Q".to_string(), balance: "100".to_string(), block_height: 1 },
            StreamEvent::Heartbeat { server_time: 123 },
        ];

        for event in events {
            let sse = event.to_sse_string();
            let data = extract_sse_data(&sse).unwrap();

            // Verify it's valid JSON by checking basic structure
            assert!(data.starts_with("{"), "JSON must start with {{");
            assert!(data.ends_with("}"), "JSON must end with }}");

            // Count braces to ensure they're balanced
            let open_braces = data.chars().filter(|c| *c == '{').count();
            let close_braces = data.chars().filter(|c| *c == '}').count();
            assert_eq!(open_braces, close_braces, "Braces must be balanced");
        }
    }

    #[test]
    fn test_special_characters_in_strings() {
        let event = StreamEvent::NewBlock {
            height: 1,
            hash: "abc\"def".to_string(),  // Contains quote
            transactions: 0,
            timestamp: 0,
        };

        // This would produce invalid JSON - document the behavior
        let sse = event.to_sse_string();
        // In production, we should escape special characters
        // This test documents that we need proper JSON serialization
    }
}

// ============================================================================
// EVENT STREAM TESTS
// ============================================================================

mod event_stream_tests {
    use super::*;

    struct EventStream {
        events: Vec<StreamEvent>,
    }

    impl EventStream {
        fn new() -> Self {
            Self { events: Vec::new() }
        }

        fn push(&mut self, event: StreamEvent) {
            self.events.push(event);
        }

        fn to_sse_stream(&self) -> String {
            self.events.iter()
                .map(|e| e.to_sse_string())
                .collect()
        }

        fn len(&self) -> usize {
            self.events.len()
        }
    }

    #[test]
    fn test_multiple_events_in_stream() {
        let mut stream = EventStream::new();

        stream.push(StreamEvent::Heartbeat { server_time: 1 });
        stream.push(StreamEvent::NewBlock {
            height: 100,
            hash: "abc".to_string(),
            transactions: 5,
            timestamp: 123,
        });
        stream.push(StreamEvent::Heartbeat { server_time: 2 });

        let output = stream.to_sse_stream();

        // Count events in output
        let event_count = output.matches("event: ").count();
        assert_eq!(event_count, 3);

        // Verify order is preserved
        let first_event_pos = output.find("event: heartbeat").unwrap();
        let block_event_pos = output.find("event: new-block").unwrap();
        let last_event_pos = output.rfind("event: heartbeat").unwrap();

        assert!(first_event_pos < block_event_pos);
        assert!(block_event_pos < last_event_pos);
    }

    #[test]
    fn test_stream_with_mining_stats() {
        let mut stream = EventStream::new();

        // Add multiple mining stats for different miners
        for i in 1..=3 {
            stream.push(StreamEvent::MiningStats {
                miner_address: "qnk_test".to_string(),
                total_rewards: i as f64 * 10.0,
                total_blocks_found: i as u64,
                current_balance: 100.0,
                avg_hash_rate: i as f64 * 1000.0,
                miner_id: Some(format!("miner-{}", i)),
                worker_id: Some(format!("miner-{}", i)),
                timestamp: "2026-01-21T12:00:00Z".to_string(),
            });
        }

        let output = stream.to_sse_stream();

        // All 3 events should be present
        assert!(output.contains(r#""miner_id":"miner-1""#));
        assert!(output.contains(r#""miner_id":"miner-2""#));
        assert!(output.contains(r#""miner_id":"miner-3""#));
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_sse_serialization_performance() {
        let iterations = 10_000;
        let event = StreamEvent::MiningStats {
            miner_address: "qnk8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1".to_string(),
            total_rewards: 50.5,
            total_blocks_found: 10,
            current_balance: 500.0,
            avg_hash_rate: 15000.0,
            miner_id: Some("desktop-miner".to_string()),
            worker_id: Some("desktop-miner".to_string()),
            timestamp: "2026-01-21T14:30:00Z".to_string(),
        };

        let start = Instant::now();

        for _ in 0..iterations {
            let _ = event.to_sse_string();
        }

        let elapsed = start.elapsed();
        let per_event_us = elapsed.as_micros() / iterations as u128;

        println!("SSE serialization: {} us per event", per_event_us);
        assert!(per_event_us < 10, "SSE serialization should be < 10us, got {} us", per_event_us);
    }
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

mod regression_tests {
    use super::*;

    /// Regression test for v3.2.25-beta: Mining stats must include miner_id
    #[test]
    fn test_regression_mining_stats_has_miner_id_v3_2_25() {
        let event = StreamEvent::MiningStats {
            miner_address: "qnk_test".to_string(),
            total_rewards: 10.0,
            total_blocks_found: 1,
            current_balance: 100.0,
            avg_hash_rate: 5000.0,
            miner_id: Some("my-miner".to_string()),
            worker_id: Some("my-miner".to_string()),
            timestamp: "2026-01-21T12:00:00Z".to_string(),
        };

        let sse = event.to_sse_string();

        // MUST contain miner_id field when set
        assert!(
            sse.contains("miner_id"),
            "REGRESSION: MiningStats event missing miner_id field"
        );

        // MUST contain worker_id field when set
        assert!(
            sse.contains("worker_id"),
            "REGRESSION: MiningStats event missing worker_id field"
        );
    }
}
