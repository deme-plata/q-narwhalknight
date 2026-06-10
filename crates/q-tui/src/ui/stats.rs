use crate::app::App;
use crate::metrics::Metrics;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// Format number with comma separators
fn fmt_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

pub fn render(f: &mut Frame, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(9),  // Core Network Metrics
            Constraint::Length(9),  // Network Health & Mining
            Constraint::Length(9),  // Performance & Resources
            Constraint::Length(11), // Apollo Control Systems
            Constraint::Min(3),    // Sync status / footer
            Constraint::Length(3), // Footer
        ])
        .split(f.size());

    render_header(f, chunks[0], &metrics);
    render_core_metrics(f, chunks[1], &metrics);
    render_health_mining(f, chunks[2], &metrics);
    render_performance(f, chunks[3], &metrics);
    render_apollo_systems(f, chunks[4], &metrics);
    render_sync_status(f, chunks[5], &metrics);
    render_footer(f, chunks[6]);
}

fn render_header(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let net_str = if metrics.network_id.is_empty() {
        String::new()
    } else {
        format!(" ({})", metrics.network_id)
    };
    let version = if metrics.version.is_empty() {
        env!("CARGO_PKG_VERSION").to_string()
    } else {
        metrics.version.clone()
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " NETWORK STATISTICS ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("│ "),
        Span::styled(
            format!("v{}{}", version, net_str),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" │ "),
        Span::styled(
            Metrics::format_uptime(metrics.uptime_secs),
            Style::default().fg(Color::Green),
        ),
        Span::raw(" │ Press [D] Dashboard │ [S] Stats"),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Yellow)),
    )
    .alignment(Alignment::Center);

    f.render_widget(header, area);
}

fn render_core_metrics(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Current Height
    let height_color = if metrics.block_height > 0 {
        Color::Cyan
    } else {
        Color::Red
    };
    let height_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Current Height",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {}", fmt_num(metrics.block_height)),
            Style::default()
                .fg(height_color)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            " Latest committed block",
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(height_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("⛓️  Height")
                .style(Style::default().fg(Color::Cyan)),
        ),
        cols[0],
    );

    // Network Height
    let net_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Network Height",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {}", fmt_num(metrics.network_height)),
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            " Highest known block",
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(net_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("🌐 Net Height")
                .style(Style::default().fg(Color::Blue)),
        ),
        cols[1],
    );

    // TPS
    let tps_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Current TPS",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {}", metrics.current_tps),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            " Transactions/sec",
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(tps_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("📊 TPS")
                .style(Style::default().fg(Color::Green)),
        ),
        cols[2],
    );

    // Supply
    let supply_str = if metrics.total_supply > 0.0 {
        format!(" {:.2} QUG", metrics.total_supply)
    } else {
        " 0.00 QUG".to_string()
    };
    let supply_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Total Supply",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            supply_str,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {:.6} QUG/blk", metrics.emission_rate),
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(supply_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("💰 Supply")
                .style(Style::default().fg(Color::Yellow)),
        ),
        cols[3],
    );
}

fn render_health_mining(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // Peers
    let peer_color = if metrics.peer_count > 0 {
        Color::Green
    } else {
        Color::Red
    };
    let peer_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Active Peers",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {}", metrics.peer_count),
            Style::default()
                .fg(peer_color)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            " Connected validators",
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(peer_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("🔗 Peers")
                .style(Style::default().fg(Color::Cyan)),
        ),
        cols[0],
    );

    // Mining
    let (miner_str, miner_color) = if metrics.mining_enabled {
        (format!(" {}", metrics.active_miners), Color::Green)
    } else {
        (" OFF".to_string(), Color::DarkGray)
    };
    let hashrate_str = if metrics.hashrate > 1_000_000.0 {
        format!(" {:.2} MH/s", metrics.hashrate / 1_000_000.0)
    } else if metrics.hashrate > 1_000.0 {
        format!(" {:.1} KH/s", metrics.hashrate / 1_000.0)
    } else if metrics.hashrate > 0.0 {
        format!(" {:.0} H/s", metrics.hashrate)
    } else {
        " --".to_string()
    };
    let mining_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Active Miners",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            miner_str,
            Style::default()
                .fg(miner_color)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            hashrate_str,
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(mining_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("⛏️  Mining")
                .style(Style::default().fg(Color::Yellow)),
        ),
        cols[1],
    );

    // Blocks Mined
    let last_block_str = if metrics.last_block_secs == 0 {
        " N/A".to_string()
    } else if metrics.last_block_secs > 3600 {
        format!(" {}h ago", metrics.last_block_secs / 3600)
    } else if metrics.last_block_secs > 60 {
        format!(" {}m ago", metrics.last_block_secs / 60)
    } else {
        format!(" {}s ago", metrics.last_block_secs)
    };
    let blocks_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Blocks Mined",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {}", fmt_num(metrics.blocks_mined)),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" Last:{}", last_block_str),
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(blocks_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("📦 Blocks")
                .style(Style::default().fg(Color::Cyan)),
        ),
        cols[2],
    );

    // DAG
    let dag_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " DAG Size",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {:.1} MB", metrics.dag_size_mb),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {} vertices", fmt_num(metrics.vertex_count)),
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(dag_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("🔷 DAG")
                .style(Style::default().fg(Color::Magenta)),
        ),
        cols[3],
    );
}

fn render_performance(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // CPU
    let cpu_color = if metrics.cpu_usage_percent < 50.0 {
        Color::Green
    } else if metrics.cpu_usage_percent < 80.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let cpu_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " CPU Usage",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {:.1}%", metrics.cpu_usage_percent),
            Style::default()
                .fg(cpu_color)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            " System load",
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(cpu_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("🖥️  CPU")
                .style(Style::default().fg(cpu_color)),
        ),
        cols[0],
    );

    // RAM
    let ram_pct = if metrics.ram_total_gb > 0.0 {
        metrics.ram_usage_gb / metrics.ram_total_gb * 100.0
    } else {
        0.0
    };
    let ram_color = if ram_pct < 60.0 {
        Color::Green
    } else if ram_pct < 85.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let ram_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Memory",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {:.1}/{:.1} GB", metrics.ram_usage_gb, metrics.ram_total_gb),
            Style::default()
                .fg(ram_color)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" ({:.0}% used)", ram_pct),
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(ram_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("🧠 RAM")
                .style(Style::default().fg(ram_color)),
        ),
        cols[1],
    );

    // Disk
    let disk_pct = if metrics.disk_total_gb > 0.0 {
        metrics.disk_usage_gb / metrics.disk_total_gb * 100.0
    } else {
        0.0
    };
    let disk_color = if disk_pct < 70.0 {
        Color::Green
    } else if disk_pct < 90.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let disk_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Disk",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" {:.0}/{:.0} GB", metrics.disk_usage_gb, metrics.disk_total_gb),
            Style::default()
                .fg(disk_color)
                .add_modifier(Modifier::BOLD),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!(" ({:.0}% used)", disk_pct),
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(disk_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("💾 Disk")
                .style(Style::default().fg(disk_color)),
        ),
        cols[2],
    );

    // Latency / Network bandwidth
    let lat_items = vec![
        ListItem::new(""),
        ListItem::new(Line::from(vec![Span::styled(
            " Latency",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![
            Span::styled(" P50: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}ms", metrics.latency_p50_ms),
                Style::default().fg(Color::Green),
            ),
            Span::styled("  P99: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}ms", metrics.latency_p99_ms),
                Style::default().fg(Color::Yellow),
            ),
        ])),
        ListItem::new(Line::from(vec![Span::styled(
            " Round-trip time",
            Style::default().fg(Color::DarkGray),
        )])),
    ];
    f.render_widget(
        List::new(lat_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("⏱️  Latency")
                .style(Style::default().fg(Color::Green)),
        ),
        cols[3],
    );
}

/// Apollo Control Systems — Kalman predictor, PID controller, Gravity assist
/// Shows live network prediction data even when synced
fn render_apollo_systems(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(38),
            Constraint::Percentage(30),
            Constraint::Percentage(32),
        ])
        .split(area);

    // ─── Kalman Network Predictor ───
    let bw = metrics.apollo_kalman_bandwidth_mbps;
    let lat = metrics.apollo_kalman_latency_ms;
    let confidence = metrics.apollo_kalman_confidence;
    let loss = metrics.apollo_kalman_loss_pct;
    let timeout = metrics.apollo_kalman_timeout_ms;
    let chunk_kb = metrics.apollo_kalman_optimal_chunk_kb;
    let concurrency = metrics.apollo_kalman_concurrency;

    let conf_pct = (confidence * 100.0).round() as u64;
    let conf_color = if conf_pct > 70 { Color::Green } else if conf_pct > 30 { Color::Yellow } else { Color::Red };
    let conf_bar_w = ((confidence * 16.0).round() as usize).min(16);

    // Bandwidth trend indicator
    let bw_indicator = if bw > 50.0 { ">>>" } else if bw > 10.0 { ">>" } else if bw > 0.0 { ">" } else { "--" };
    let bw_color = if bw > 50.0 { Color::Green } else if bw > 10.0 { Color::Cyan } else if bw > 0.0 { Color::Yellow } else { Color::DarkGray };

    let kalman_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" BW ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} Mbps ", bw),
                Style::default().fg(bw_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(bw_indicator, Style::default().fg(bw_color)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Lat ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}ms", lat),
                Style::default().fg(if lat < 100.0 { Color::Green } else if lat < 500.0 { Color::Yellow } else { Color::Red }),
            ),
            Span::styled("  Loss ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}%", loss),
                Style::default().fg(if loss < 2.0 { Color::Green } else if loss < 10.0 { Color::Yellow } else { Color::Red }),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Tout ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}ms", timeout), Style::default().fg(Color::Yellow)),
            Span::styled("  Chunk ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}KB", chunk_kb), Style::default().fg(Color::Magenta)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Par ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}x", concurrency), Style::default().fg(Color::Cyan)),
            Span::styled("  Conf ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}%", conf_pct), Style::default().fg(conf_color)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "\u{2588}".repeat(conf_bar_w),
                Style::default().fg(conf_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(conf_bar_w)),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                if confidence > 0.7 { " LOCKED" } else if confidence > 0.3 { " TRACKING" } else if confidence > 0.0 { " ACQUIRING" } else { " IDLE" },
                Style::default().fg(conf_color),
            ),
        ])),
    ];

    let kalman_widget = List::new(kalman_items)
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled(" KALMAN Predictor ", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)))
            .style(Style::default().fg(Color::Blue)));
    f.render_widget(kalman_widget, cols[0]);

    // ─── PID Rate Controller ───
    let target = metrics.apollo_pid_target_bps;
    let current = metrics.apollo_pid_current_bps;
    let avg_err = metrics.apollo_pid_error;
    let drift = if target > 0.0 { (current - target) / target * 100.0 } else { 0.0 };
    let drift_color = if drift.abs() < 10.0 { Color::Green } else if drift.abs() < 30.0 { Color::Yellow } else { Color::Red };

    // Rate utilization bar
    let utilization = if target > 0.0 { (current / target).min(2.0) } else { 0.0 };
    let util_bar_w = ((utilization * 8.0).round() as usize).min(16);
    let util_color = if utilization > 1.2 { Color::Red } else if utilization > 0.8 { Color::Green } else { Color::Yellow };

    let mode_str = match metrics.apollo_sync_mode {
        1 => ("TURBO", Color::Yellow),
        2 => ("ENDGAME", Color::Magenta),
        3 => ("MICRO", Color::Cyan),
        _ => ("IDLE", Color::Green),
    };

    let pid_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Target ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0} BPS", target),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Actual ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0} BPS", current),
                Style::default().fg(util_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Drift  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:+.1}%", drift),
                Style::default().fg(drift_color),
            ),
            Span::styled("  Err ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}", avg_err),
                Style::default().fg(if avg_err.abs() < 50.0 { Color::Green } else { Color::Yellow }),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "\u{2588}".repeat(util_bar_w),
                Style::default().fg(util_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(util_bar_w)),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Mode ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                mode_str.0,
                Style::default().fg(mode_str.1).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if drift.abs() < 5.0 { "  STABLE" } else if drift > 0.0 { "  ABOVE" } else { "  BELOW" },
                Style::default().fg(drift_color),
            ),
        ])),
    ];

    let pid_widget = List::new(pid_items)
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled(" PID Controller ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)))
            .style(Style::default().fg(Color::Green)));
    f.render_widget(pid_widget, cols[1]);

    // ─── Gravity Assist (Peer Momentum) ───
    let peers = metrics.apollo_peers_tracked;
    let in_flight = metrics.apollo_in_flight;
    let queued = metrics.apollo_queued;
    let completed = metrics.apollo_chunks_completed;
    let total = metrics.apollo_chunks_total;
    let chunk_pct = if total > 0 { completed as f64 / total as f64 * 100.0 } else { 0.0 };
    let best_peer = &metrics.apollo_gravity_best_peer;
    let best_heat = metrics.apollo_gravity_best_heat;

    // Heat indicator
    let heat_bar_w = ((best_heat * 16.0).round() as usize).min(16);
    let heat_color = if best_heat > 0.7 { Color::Red } else if best_heat > 0.3 { Color::Yellow } else if best_heat > 0.0 { Color::Cyan } else { Color::DarkGray };

    let peer_display = if !best_peer.is_empty() {
        format!("{}...", best_peer)
    } else {
        "--".to_string()
    };

    let gravity_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Peers ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", peers),
                Style::default().fg(if peers > 3 { Color::Green } else if peers > 0 { Color::Yellow } else { Color::Red }).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" tracked", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Chunks ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", completed, total),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" F:{} Q:{}", in_flight, queued),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Best  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                peer_display,
                Style::default().fg(Color::Yellow),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Heat  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}% ", best_heat * 100.0),
                Style::default().fg(heat_color),
            ),
            Span::styled(
                "\u{2588}".repeat(heat_bar_w),
                Style::default().fg(heat_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(heat_bar_w)),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Done  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}%", chunk_pct),
                Style::default().fg(if chunk_pct > 90.0 { Color::Green } else if chunk_pct > 50.0 { Color::Cyan } else { Color::Yellow }).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if in_flight > 0 { "  ACTIVE" } else if peers > 0 { "  READY" } else { "  IDLE" },
                Style::default().fg(if in_flight > 0 { Color::Green } else { Color::DarkGray }),
            ),
        ])),
    ];

    let gravity_widget = List::new(gravity_items)
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled(" GRAVITY Assist ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)))
            .style(Style::default().fg(Color::Magenta)));
    f.render_widget(gravity_widget, cols[2]);
}

fn render_sync_status(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let status_color;
    let status_text;
    let detail;

    if metrics.is_syncing {
        status_color = Color::Yellow;
        let speed_str = if metrics.sync_speed_blocks_per_sec >= 1000.0 {
            format!("{:.1}K blk/s", metrics.sync_speed_blocks_per_sec / 1000.0)
        } else if metrics.sync_speed_blocks_per_sec >= 1.0 {
            format!("{:.0} blk/s", metrics.sync_speed_blocks_per_sec)
        } else {
            "calculating...".to_string()
        };

        let remaining = metrics
            .sync_target_height
            .saturating_sub(metrics.sync_current_height);
        let eta = if metrics.sync_speed_blocks_per_sec > 0.5 {
            let secs = remaining as f32 / metrics.sync_speed_blocks_per_sec;
            let mins = (secs / 60.0) as u32;
            if mins > 60 {
                format!("{}h {}m", mins / 60, mins % 60)
            } else {
                format!("{}m", mins)
            }
        } else {
            "...".to_string()
        };

        let mode_str = match metrics.apollo_sync_mode {
            1 => "TURBO",
            2 => "ENDGAME",
            3 => "MICRO",
            _ => "SYNC",
        };

        status_text = format!(
            "[{}] {}/{} ({:.1}%)",
            mode_str,
            fmt_num(metrics.sync_current_height),
            fmt_num(metrics.sync_target_height),
            metrics.sync_progress_percent
        );

        let apollo_detail = if metrics.apollo_kalman_confidence > 0.0 {
            format!(
                " │ Kalman: {:.1}Mbps/{:.0}ms (conf={:.0}%) │ Peers: {}",
                metrics.apollo_kalman_bandwidth_mbps,
                metrics.apollo_kalman_latency_ms,
                metrics.apollo_kalman_confidence * 100.0,
                metrics.apollo_peers_tracked,
            )
        } else {
            String::new()
        };

        detail = format!(
            "Speed: {} │ Remaining: {} │ ETA: {}{}",
            speed_str,
            fmt_num(remaining),
            eta,
            apollo_detail
        );
    } else if metrics.block_height > 0 && metrics.peer_count > 0 {
        status_color = Color::Green;
        status_text = format!(
            "SYNCED  Height: {} │ Peers: {}",
            fmt_num(metrics.block_height),
            metrics.peer_count
        );
        detail = "Node is fully synchronized with the network".to_string();
    } else if metrics.peer_count == 0 {
        status_color = Color::Red;
        status_text = "CONNECTING  Searching for peers...".to_string();
        detail = "Waiting for P2P connections to bootstrap nodes".to_string();
    } else {
        status_color = Color::Yellow;
        status_text = "STARTING  Initializing...".to_string();
        detail = "Node is starting up".to_string();
    };

    let status_widget = Paragraph::new(vec![
        Line::from(vec![Span::styled(
            format!("  {}", status_text),
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![Span::styled(
            format!("  {}", detail),
            Style::default().fg(Color::DarkGray),
        )]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("📡 Sync Status")
            .style(Style::default().fg(status_color)),
    );

    f.render_widget(status_widget, area);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[D] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Dashboard │ "),
        Span::styled("[S] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Stats │ "),
        Span::styled("[F] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Physics │ "),
        Span::styled("[L] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Logs │ "),
        Span::styled("[N] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Network │ "),
        Span::styled("[Q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(footer, area);
}
