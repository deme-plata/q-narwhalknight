//! 🎨 Console Visualization System
//!
//! Beautiful animated ASCII art visualization showing the Q-NarwhalKnight
//! consensus system in action when running the API server.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Live consensus statistics for visualization
#[derive(Debug, Clone, Default)]
pub struct ConsensusStats {
    pub total_transactions: u64,
    pub transactions_per_second: f64,
    pub total_blocks: u64,
    pub blocks_per_second: f64,
    pub connected_peers: usize,
    pub dag_vertices: u64,
    pub consensus_rounds: u64,
    pub average_latency_ms: f64,
    pub mempool_size: usize,
    pub resonance_enabled: bool,
    pub shadow_mode_active: bool,
    pub agreement_rate: f64,
}

/// Console visualizer for consensus system
pub struct ConsoleVisualizer {
    stats: Arc<RwLock<ConsensusStats>>,
    frame_count: Arc<RwLock<usize>>,
}

impl ConsoleVisualizer {
    /// Create new console visualizer
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(ConsensusStats::default())),
            frame_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get handle to update stats
    pub fn get_stats_handle(&self) -> Arc<RwLock<ConsensusStats>> {
        self.stats.clone()
    }

    /// Start animated visualization loop
    pub async fn start_animation(self) {
        let stats = self.stats;
        let frame_count = self.frame_count;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));

            loop {
                interval.tick().await;

                let stats_snapshot = stats.read().await.clone();
                let mut frame = frame_count.write().await;
                *frame = (*frame + 1) % 8;

                // Clear screen and move cursor to top
                print!("\x1B[2J\x1B[1;1H");

                // Render the visualization
                Self::render_frame(*frame, &stats_snapshot);
            }
        });
    }

    /// Render a single frame of the visualization
    fn render_frame(frame: usize, stats: &ConsensusStats) {
        // Header banner
        info!("╔═══════════════════════════════════════════════════════════════════════════╗");
        info!("║           🎻 Q-NARWHALKNIGHT QUANTUM CONSENSUS SYSTEM 🎻                 ║");
        info!("╚═══════════════════════════════════════════════════════════════════════════╝");
        info!("");

        // Animated DAG visualization
        Self::render_dag_graph(frame, stats);
        info!("");

        // Consensus metrics
        Self::render_metrics(stats);
        info!("");

        // Network topology
        Self::render_network(frame, stats);
        info!("");

        // Footer
        info!("═══════════════════════════════════════════════════════════════════════════");
        info!("  Press Ctrl+C to stop | Logs: /tmp/api-server.log");
        info!("═══════════════════════════════════════════════════════════════════════════");
    }

    /// Render animated DAG graph
    fn render_dag_graph(frame: usize, stats: &ConsensusStats) {
        info!("📊 DAG-KNIGHT CONSENSUS VISUALIZATION:");
        info!("");

        // Animated vertex creation
        let vertex_symbols = ["◯", "◉", "●", "◉", "◯", "○", "◌", "○"];
        let current_symbol = vertex_symbols[frame];

        // DAG structure with animation
        let vertices_per_round = 4;
        let rounds_to_show = 3;

        for round in (0..rounds_to_show).rev() {
            let round_num = stats.consensus_rounds.saturating_sub(round);
            let prefix = if round == 0 { "   ▶ " } else { "     " };

            info!("{}Round {}: ", prefix, round_num);

            // Show vertices in this round
            for v in 0..vertices_per_round {
                let symbol = if round == 0 && v == frame % vertices_per_round {
                    current_symbol
                } else {
                    "●"
                };

                let vertex_num = round_num * vertices_per_round as u64 + v as u64;
                if vertex_num <= stats.dag_vertices {
                    print!("  {} V{:<4}", symbol, vertex_num);
                }
            }
            println!();

            // Show connections to previous round
            if round < rounds_to_show - 1 {
                info!("     │  │  │  │");
            }
        }

        info!("");
        info!(
            "  Total Vertices: {} | Consensus Rounds: {}",
            stats.dag_vertices, stats.consensus_rounds
        );
    }

    /// Render consensus metrics
    fn render_metrics(stats: &ConsensusStats) {
        info!("⚡ PERFORMANCE METRICS:");
        info!("");

        // Transaction throughput with bar graph
        let tps_bar = Self::create_bar_graph(stats.transactions_per_second, 100_000.0, 40);
        info!(
            "  Transactions/sec: {:>8.0} TPS  {}",
            stats.transactions_per_second, tps_bar
        );

        // Block production rate
        let bps_bar = Self::create_bar_graph(stats.blocks_per_second * 10.0, 10.0, 40);
        info!(
            "  Blocks/sec:       {:>8.2} BPS  {}",
            stats.blocks_per_second, bps_bar
        );

        // Consensus latency
        let latency_bar =
            Self::create_bar_graph(100.0 - stats.average_latency_ms.min(100.0), 100.0, 40);
        info!(
            "  Avg Latency:      {:>8.2} ms   {}",
            stats.average_latency_ms, latency_bar
        );

        // Mempool status
        let mempool_bar = Self::create_bar_graph(stats.mempool_size as f64, 100_000.0, 40);
        info!(
            "  Mempool Size:     {:>8} txs  {}",
            stats.mempool_size, mempool_bar
        );

        info!("");
        info!(
            "  Total Transactions: {:>12} | Total Blocks: {:>8}",
            stats.total_transactions, stats.total_blocks
        );

        // Shadow mode status (if enabled)
        if stats.shadow_mode_active {
            info!("");
            info!("🎭 SHADOW MODE STATUS:");
            info!(
                "  Mode: Active | Agreement Rate: {:.1}%",
                stats.agreement_rate * 100.0
            );

            let agreement_bar = Self::create_bar_graph(stats.agreement_rate * 100.0, 100.0, 40);
            info!("  DAG-Knight vs Resonance: {}", agreement_bar);

            if stats.agreement_rate >= 0.95 {
                info!("  ✅ Excellent agreement - ready for migration!");
            } else if stats.agreement_rate >= 0.85 {
                info!("  ✓ Good agreement - validation in progress");
            } else {
                info!("  ⚠ Agreement below threshold - monitoring");
            }
        }
    }

    /// Render network topology
    fn render_network(frame: usize, stats: &ConsensusStats) {
        info!("🌐 NETWORK TOPOLOGY:");
        info!("");

        // Animated peer connections
        let connection_symbols = ["═", "─", "═", "━", "═", "─", "═", "━"];
        let conn_symbol = connection_symbols[frame];

        // Central node (this node)
        info!("                    ┌─────────┐");
        info!("                    │  THIS   │");
        info!("                    │  NODE   │");
        info!("                    └────┬────┘");
        info!("                         │");

        // Connected peers
        if stats.connected_peers > 0 {
            info!(
                "         ┌{}{}{}{}{}┼{}{}{}{}{}┐",
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol,
                conn_symbol
            );

            // Show up to 5 peers
            let peers_to_show = stats.connected_peers.min(5);
            for i in 0..peers_to_show {
                let symbol = if i == frame % peers_to_show {
                    "◉"
                } else {
                    "●"
                };
                if i == 0 && peers_to_show == 1 {
                    info!("         │");
                    info!("      ┌──▼──┐");
                } else if i == 0 {
                    info!("         │          │          │");
                    info!("      ┌──▼──┐    ┌──▼──┐    ┌──▼──┐");
                }

                if peers_to_show <= 3 {
                    print!("      │ {} {} │", symbol, format!("P{}", i + 1));
                    if i < peers_to_show - 1 {
                        print!(
                            "    │ {} {} │",
                            if (i + 1) == frame % peers_to_show {
                                "◉"
                            } else {
                                "●"
                            },
                            format!("P{}", i + 2)
                        );
                        if peers_to_show > 2 {
                            print!(
                                "    │ {} {} │",
                                if (i + 2) == frame % peers_to_show {
                                    "◉"
                                } else {
                                    "●"
                                },
                                format!("P{}", i + 3)
                            );
                        }
                    }
                    println!();
                    if i == 0 {
                        info!("      └─────┘    └─────┘    └─────┘");
                    }
                    break;
                }
            }

            if stats.connected_peers > 5 {
                info!("         ... and {} more peers", stats.connected_peers - 5);
            }
        } else {
            info!("         │");
            info!("         ▼");
            info!("   ⚠ No peers connected yet");
            info!("   Peer discovery in progress...");
        }

        info!("");
        info!(
            "  Connected Peers: {} | Network Status: {}",
            stats.connected_peers,
            if stats.connected_peers >= 4 {
                "✅ Healthy"
            } else if stats.connected_peers >= 1 {
                "⚠ Limited"
            } else {
                "❌ Isolated"
            }
        );
    }

    /// Create a text-based bar graph
    fn create_bar_graph(value: f64, max_value: f64, width: usize) -> String {
        let percentage = (value / max_value).min(1.0);
        let filled = (percentage * width as f64) as usize;
        let empty = width.saturating_sub(filled);

        let bar_filled = "█".repeat(filled);
        let bar_empty = "░".repeat(empty);

        format!("[{}{}] {:>5.1}%", bar_filled, bar_empty, percentage * 100.0)
    }
}

/// Helper function to update stats from various sources
pub async fn update_stats(
    stats_handle: Arc<RwLock<ConsensusStats>>,
    update: impl FnOnce(&mut ConsensusStats),
) {
    let mut stats = stats_handle.write().await;
    update(&mut stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_graph() {
        let bar = ConsoleVisualizer::create_bar_graph(50.0, 100.0, 20);
        assert!(bar.contains("50.0%"));

        let full_bar = ConsoleVisualizer::create_bar_graph(100.0, 100.0, 20);
        assert!(full_bar.contains("100.0%"));

        let empty_bar = ConsoleVisualizer::create_bar_graph(0.0, 100.0, 20);
        assert!(empty_bar.contains("0.0%"));
    }

    #[tokio::test]
    async fn test_stats_update() {
        let visualizer = ConsoleVisualizer::new();
        let stats_handle = visualizer.get_stats_handle();

        update_stats(stats_handle.clone(), |stats| {
            stats.total_transactions = 1000;
            stats.transactions_per_second = 50_000.0;
        })
        .await;

        let stats = stats_handle.read().await;
        assert_eq!(stats.total_transactions, 1000);
        assert_eq!(stats.transactions_per_second, 50_000.0);
    }
}
