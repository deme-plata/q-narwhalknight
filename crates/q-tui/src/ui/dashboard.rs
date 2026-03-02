use crate::app::{App, LogLevel};
use crate::metrics::Metrics;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline},
    Frame,
};

pub fn render(f: &mut Frame, app: &App) {
    let metrics = app.metrics.read().unwrap();

    // Dynamic layout based on sync state
    let constraints = if metrics.is_syncing {
        vec![
            Constraint::Length(3),   // Header
            Constraint::Length(4),   // Sync Progress Bar
            Constraint::Length(9),   // Metrics cards
            Constraint::Length(9),   // APOLLO Control Systems (replaces TPS during sync)
            Constraint::Min(4),      // Logs (smaller when syncing)
            Constraint::Length(3),   // Footer
        ]
    } else {
        vec![
            Constraint::Length(3),   // Header
            Constraint::Length(9),   // Metrics cards
            Constraint::Length(7),   // TPS Chart or AI Metrics
            Constraint::Min(8),      // Logs
            Constraint::Length(3),   // Footer
        ]
    };
    drop(metrics); // Release lock

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(f.size());

    let metrics = app.metrics.read().unwrap();
    let mut idx = 0;

    render_header(f, chunks[idx], app);
    idx += 1;

    if metrics.is_syncing {
        render_sync_progress(f, chunks[idx], app);
        idx += 1;
    }
    drop(metrics);

    render_metrics_grid(f, chunks[idx], app);
    idx += 1;

    // Always show APOLLO control systems — they track live network state even when synced
    {
        let m = app.metrics.read().unwrap();
        let has_apollo_data = m.apollo_kalman_confidence > 0.0
            || m.apollo_peers_tracked > 0
            || m.apollo_pid_current_bps > 0.0
            || m.is_syncing;
        drop(m);
        if has_apollo_data {
            render_apollo_control_systems(f, chunks[idx], app);
        } else {
            render_tps_or_ai_metrics(f, chunks[idx], app);
        }
    }
    idx += 1;

    render_recent_logs(f, chunks[idx], app);
    idx += 1;

    render_footer(f, chunks[idx], app);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    // Starship Flight Computer phase-based header
    let status_text = match metrics.starship_phase.as_str() {
        "SUPER_HEAVY"       => Span::styled("TURBO SYNC", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        "HOT_STAGING"       => Span::styled("ENDGAME", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        "STARSHIP_CRUISE"   => Span::styled("CRUISE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        "IGNITION"          => Span::styled("IGNITION", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        "ORBITAL_INSERTION" => Span::styled("INSERTING", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        "STATION_KEEPING" if metrics.starship_orbit_stable =>
            Span::styled("IN ORBIT", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        "STATION_KEEPING" =>
            Span::styled("STABILIZING", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        "PRELAUNCH" if metrics.peer_count > 0 =>
            Span::styled("PRELAUNCH", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
        _ if metrics.peer_count == 0 =>
            Span::styled("CONNECTING", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        _ => Span::styled("SYNCED", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
    };

    let version_str = if metrics.version.is_empty() {
        env!("CARGO_PKG_VERSION").to_string()
    } else {
        metrics.version.clone()
    };

    let net_str = if metrics.network_id.is_empty() {
        String::new()
    } else {
        format!(" ({}) ", metrics.network_id)
    };

    let peer_color = if metrics.peer_count > 0 { Color::Green } else { Color::Red };

    // v8.6.1: Operator earnings display
    let operator_spans = if metrics.operator_fee_promille > 0 && !metrics.admin_wallet_address.is_empty() {
        vec![
            Span::raw(" │ "),
            Span::styled("Earned: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4} QUG", metrics.operator_fee_total_qug + metrics.admin_wallet_balance),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            ),
        ]
    } else {
        vec![]
    };

    let mut header_spans = vec![
        Span::styled("Peers: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", metrics.peer_count),
            Style::default().fg(peer_color).add_modifier(Modifier::BOLD)
        ),
        Span::raw("  "),
        Span::styled(
            format!("{}", version_str),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        ),
        Span::styled(net_str, Style::default().fg(Color::DarkGray)),
        Span::raw(" │ "),
        status_text,
        Span::raw(" │ Uptime: "),
        Span::styled(
            Metrics::format_uptime(metrics.uptime_secs),
            Style::default().fg(Color::Green)
        ),
        Span::raw(" │ "),
        Span::styled(
            format!("Net Height: {}", metrics.network_height),
            Style::default().fg(Color::Blue)
        ),
    ];
    header_spans.extend(operator_spans);

    let header = Paragraph::new(Line::from(header_spans))
    .block(Block::default().borders(Borders::ALL).style(Style::default().fg(Color::Cyan)))
    .alignment(Alignment::Left);

    f.render_widget(header, area);
}

fn render_metrics_grid(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);

    // Network metrics (left panel) — or Mining panel if mining is enabled
    if metrics.mining_enabled {
        let mining_items = vec![
            ListItem::new(Line::from(vec![
                Span::raw("Miners:      "),
                Span::styled(
                    format!("{}", metrics.active_miners),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                ),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Hashrate:    "),
                Span::styled(
                    format!("{:.1} H/s", metrics.hashrate),
                    Style::default().fg(Color::Cyan)
                ),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Blocks:      "),
                Span::styled(
                    format!("{}", metrics.blocks_mined),
                    Style::default().fg(Color::Yellow)
                ),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("Peers:       "),
                Span::styled(
                    format!("{}", metrics.peer_count),
                    Style::default().fg(if metrics.peer_count > 0 { Color::Green } else { Color::Red })
                ),
            ])),
            ListItem::new(format!("├ In:       {}", metrics.inbound_peers)),
            ListItem::new(format!("└ Out:      {}", metrics.outbound_peers)),
        ];

        let mining_widget = List::new(mining_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("⛏️  Mining + Network")
                    .style(Style::default().fg(Color::Yellow))
            );
        f.render_widget(mining_widget, chunks[0]);
    } else {
        let network_items = vec![
            ListItem::new(Line::from(vec![
                Span::raw("Peers:        "),
                Span::styled(
                    format!("{}", metrics.peer_count),
                    Style::default().fg(if metrics.peer_count > 0 { Color::Green } else { Color::Red })
                ),
            ])),
            ListItem::new(format!("├ Inbound:   {}", metrics.inbound_peers)),
            ListItem::new(format!("└ Outbound:  {}", metrics.outbound_peers)),
            ListItem::new(format!("Tor Circuits: {}", metrics.tor_circuits)),
            ListItem::new(Line::from(vec![
                Span::raw("↓ "),
                Span::styled(
                    format!("{}/s", Metrics::format_bytes(metrics.bytes_received)),
                    Style::default().fg(Color::Cyan)
                ),
            ])),
            ListItem::new(Line::from(vec![
                Span::raw("↑ "),
                Span::styled(
                    format!("{}/s", Metrics::format_bytes(metrics.bytes_sent)),
                    Style::default().fg(Color::Magenta)
                ),
            ])),
        ];

        let network = List::new(network_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("🔗 Network")
                    .style(Style::default().fg(Color::Blue))
            );
        f.render_widget(network, chunks[0]);
    }

    // Helper: format number with comma separators
    let fmt_num = |n: u64| -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 { result.push(','); }
            result.push(c);
        }
        result.chars().rev().collect()
    };

    // Blockchain metrics
    let last_block_display = if metrics.last_block_secs == 0 && metrics.block_height == 0 {
        "N/A".to_string()
    } else if metrics.last_block_secs > 3600 {
        format!("{}h ago", metrics.last_block_secs / 3600)
    } else if metrics.last_block_secs > 60 {
        format!("{}m ago", metrics.last_block_secs / 60)
    } else {
        format!("{}s ago", metrics.last_block_secs)
    };
    let last_block_color = if metrics.last_block_secs == 0 {
        Color::DarkGray
    } else if metrics.last_block_secs < 5 {
        Color::Green
    } else {
        Color::Yellow
    };

    // Height with sync progress indicator
    let height_display = if metrics.is_syncing && metrics.sync_progress_percent > 0.0 {
        format!("{} ({:.1}%)", fmt_num(metrics.block_height), metrics.sync_progress_percent)
    } else {
        fmt_num(metrics.block_height)
    };

    let mut blockchain_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Height:      "),
            Span::styled(
                height_display,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Net Height:  "),
            Span::styled(
                fmt_num(metrics.network_height),
                Style::default().fg(Color::Blue)
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Last Block:  "),
            Span::styled(last_block_display, Style::default().fg(last_block_color)),
        ])),
        ListItem::new(format!("DAG Size:     {:.1} MB", metrics.dag_size_mb)),
    ];

    if metrics.total_supply > 0.0 {
        blockchain_items.push(ListItem::new(Line::from(vec![
            Span::raw("Supply:      "),
            Span::styled(
                format!("{:.2} QUG", metrics.total_supply),
                Style::default().fg(Color::Yellow)
            ),
        ])));
    }
    if metrics.emission_rate > 0.0 {
        blockchain_items.push(ListItem::new(format!("Emission:     {:.6} QUG/blk", metrics.emission_rate)));
    }
    // v8.6.1: Operator wallet balance
    if metrics.operator_fee_promille > 0 && metrics.admin_wallet_balance > 0.0 {
        blockchain_items.push(ListItem::new(Line::from(vec![
            Span::raw("Wallet:      "),
            Span::styled(
                format!("{:.4} QUG", metrics.admin_wallet_balance),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            ),
        ])));
    }

    let blockchain = List::new(blockchain_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("⛓️  Blockchain")
                .style(Style::default().fg(Color::Cyan))
        );
    f.render_widget(blockchain, chunks[1]);

    // Performance metrics
    let perf_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("TPS:         "),
            Span::styled(
                format!("{}", metrics.current_tps),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            ),
        ])),
        ListItem::new(format!("Latency P50:  {}ms", metrics.latency_p50_ms)),
        ListItem::new(format!("Latency P99:  {}ms", metrics.latency_p99_ms)),
        ListItem::new(format!("CPU:          {:.1}%", metrics.cpu_usage_percent)),
        ListItem::new(format!("RAM:          {:.1}/{:.1} GB", metrics.ram_usage_gb, metrics.ram_total_gb)),
        ListItem::new(format!("Disk:         {:.0}/{:.0} GB", metrics.disk_usage_gb, metrics.disk_total_gb)),
    ];

    let performance = List::new(perf_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("⚡ Performance")
                .style(Style::default().fg(Color::Green))
        );
    f.render_widget(performance, chunks[2]);
}

fn render_tps_chart(f: &mut Frame, area: Rect, app: &App) {
    let history = app.get_tps_history();

    // Convert f64 to u64 for sparkline
    let history_u64: Vec<u64> = history.iter().map(|&x| x as u64).collect();

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("📈 TPS Chart (Last 60s)")
                .style(Style::default().fg(Color::Yellow))
        )
        .data(&history_u64)
        .style(Style::default().fg(Color::Green))
        .max(500);

    f.render_widget(sparkline, area);
}

fn render_recent_logs(f: &mut Frame, area: Rect, app: &App) {
    let available_height = area.height.saturating_sub(2); // Account for borders
    let logs = app.get_recent_logs(available_height as usize);

    let log_items: Vec<ListItem> = logs
        .iter()
        .rev() // Show newest first
        .map(|log| {
            let time_str = log.timestamp.format("%H:%M:%S").to_string();
            let level_color = match log.level {
                LogLevel::Trace => Color::DarkGray,
                LogLevel::Debug => Color::Gray,
                LogLevel::Info => Color::Cyan,
                LogLevel::Warn => Color::Yellow,
                LogLevel::Error => Color::Red,
            };

            ListItem::new(Line::from(vec![
                Span::raw("["),
                Span::styled(time_str, Style::default().fg(Color::DarkGray)),
                Span::raw("] "),
                Span::styled(
                    log.level.as_str(),
                    Style::default().fg(level_color).add_modifier(Modifier::BOLD)
                ),
                Span::raw("  "),
                Span::raw(&log.message),
            ]))
        })
        .collect();

    let pause_indicator = if app.logs_paused {
        " [PAUSED]"
    } else {
        ""
    };

    let logs_list = List::new(log_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("📝 Recent Logs (Scroll: ↑↓ | [P] Pause){}",pause_indicator))
                .style(Style::default().fg(Color::White))
        );

    f.render_widget(logs_list, area);
}

fn render_sync_progress(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let progress = metrics.sync_progress_percent.clamp(0.0, 100.0);
    // bar_width accounts for left/right border (2 chars)
    let inner_width = area.width.saturating_sub(2);
    let bar_width = (inner_width as f32 * progress / 100.0) as u16;

    let speed = metrics.sync_speed_blocks_per_sec;
    let eta_str = if speed > 0.5 {
        let blocks_remaining = metrics.sync_target_height.saturating_sub(metrics.sync_current_height) as f32;
        let secs_remaining = blocks_remaining / speed;
        let mins = (secs_remaining / 60.0) as u32;
        if mins > 60 {
            format!("ETA: {}h {}m", mins / 60, mins % 60)
        } else if mins > 0 {
            format!("ETA: {}m", mins)
        } else {
            format!("ETA: <1m")
        }
    } else if metrics.sync_current_height > 0 {
        "Calculating...".to_string()
    } else {
        "Starting...".to_string()
    };

    // Format speed nicely
    let speed_str = if speed >= 1000.0 {
        format!("{:.1}K blk/s", speed / 1000.0)
    } else if speed >= 1.0 {
        format!("{:.0} blk/s", speed)
    } else if speed > 0.0 {
        format!("{:.1} blk/s", speed)
    } else {
        "-- blk/s".to_string()
    };

    // Format heights with comma separators
    let fmt_num = |n: u64| -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 { result.push(','); }
            result.push(c);
        }
        result.chars().rev().collect()
    };

    // Starship sync phase badge
    let (badge_text, badge_color) = match metrics.starship_phase.as_str() {
        "PRELAUNCH"         => ("[PRELAUNCH] ",    Color::DarkGray),
        "IGNITION"          => ("[IGNITION] ",     Color::Red),
        "SUPER_HEAVY"       => ("[SUPER HEAVY] ",  Color::Yellow),
        "HOT_STAGING"       => ("[HOT STAGING] ",  Color::Magenta),
        "STARSHIP_CRUISE"   => ("[CRUISE] ",       Color::Cyan),
        "ORBITAL_INSERTION" => ("[INSERTING] ",     Color::Blue),
        "STATION_KEEPING"   => ("[IN ORBIT] ",     Color::Green),
        _                   => ("[SYNC] ",         Color::DarkGray),
    };
    let mode_badge = Span::styled(badge_text, Style::default().fg(badge_color).add_modifier(Modifier::BOLD));

    // Chunk progress string
    let chunks_str = if metrics.apollo_chunks_total > 0 {
        format!("Chunks: {}/{} ({}fly/{}q)",
            metrics.apollo_chunks_completed, metrics.apollo_chunks_total,
            metrics.apollo_in_flight, metrics.apollo_queued)
    } else {
        String::new()
    };

    let sync_text = vec![
        Line::from(vec![
            mode_badge,
            Span::styled(
                fmt_num(metrics.sync_current_height),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
            Span::styled(
                format!("/{}", fmt_num(metrics.sync_target_height)),
                Style::default().fg(Color::DarkGray)
            ),
            Span::raw(" "),
            Span::styled(
                format!("({:.1}%)", progress),
                Style::default().fg(if progress > 50.0 { Color::Green } else { Color::Yellow })
            ),
            Span::raw(" │ "),
            Span::styled(speed_str, Style::default().fg(Color::Cyan)),
            Span::raw(" │ "),
            Span::styled(eta_str, Style::default().fg(Color::Magenta)),
            if !chunks_str.is_empty() {
                Span::raw(" │ ")
            } else {
                Span::raw("")
            },
            Span::styled(chunks_str, Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(
                "█".repeat(bar_width as usize),
                Style::default().fg(if progress > 90.0 { Color::Green } else if progress > 50.0 { Color::Cyan } else { Color::Yellow })
            ),
            Span::styled(
                "░".repeat(inner_width.saturating_sub(bar_width) as usize),
                Style::default().fg(Color::DarkGray)
            ),
        ]),
    ];

    let sync_widget = Paragraph::new(sync_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("⏳ Blockchain Sync")
                .style(Style::default().fg(Color::Yellow))
        );

    f.render_widget(sync_widget, area);
}

/// 🚀 v1.0.2: APOLLO Control Systems panel — shown during sync instead of TPS chart
fn render_tps_or_ai_metrics(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    if metrics.ai_enabled {
        drop(metrics);
        render_ai_metrics(f, area, app);
    } else {
        drop(metrics);
        render_tps_chart(f, area, app);
    }
}

fn render_ai_metrics(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let ai_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("AI Nodes:       "),
            Span::styled(
                format!("{}", metrics.ai_nodes_available),
                Style::default().fg(if metrics.ai_nodes_available > 0 { Color::Green } else { Color::Red })
            ),
        ])),
        ListItem::new(format!("Total Requests:  {}", metrics.ai_total_requests)),
        ListItem::new(format!("Nodes Used:      {}", metrics.ai_nodes_participated)),
        ListItem::new(format!("Avg Nodes/Req:   {:.1}", metrics.ai_avg_nodes_per_request)),
        ListItem::new(format!("Layers Processed: {}", metrics.ai_layers_processed)),
        ListItem::new(Line::from(vec![
            Span::raw("Active Requests: "),
            Span::styled(
                format!("{}", metrics.ai_active_requests),
                Style::default().fg(if metrics.ai_active_requests > 0 { Color::Yellow } else { Color::Green })
            ),
        ])),
    ];

    let ai_widget = List::new(ai_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("🤖 Distributed AI Metrics")
                .style(Style::default().fg(Color::Magenta))
        );

    f.render_widget(ai_widget, area);
}

fn render_apollo_control_systems(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    // Split into 3 columns: Kalman Predictor | PID Controller | Gravity Assist
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(38),
            Constraint::Percentage(30),
            Constraint::Percentage(32),
        ])
        .split(area);

    // --- Kalman Network Predictor ---
    let bw = metrics.apollo_kalman_bandwidth_mbps;
    let lat = metrics.apollo_kalman_latency_ms;
    let confidence = metrics.apollo_kalman_confidence;
    let loss = metrics.apollo_kalman_loss_pct;
    let conf_color = if confidence > 0.7 { Color::Green } else if confidence > 0.3 { Color::Yellow } else { Color::Red };
    let conf_bar_w = ((confidence * 14.0).round() as usize).min(14);
    let bw_color = if bw > 50.0 { Color::Green } else if bw > 10.0 { Color::Cyan } else if bw > 0.0 { Color::Yellow } else { Color::DarkGray };
    let bw_arrow = if bw > 50.0 { ">>>" } else if bw > 10.0 { ">>" } else if bw > 0.0 { ">" } else { "--" };

    let kalman_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("BW:   "),
            Span::styled(
                format!("{:.1} Mbps ", bw),
                Style::default().fg(bw_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(bw_arrow, Style::default().fg(bw_color)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Lat:  "),
            Span::styled(
                format!("{:.0}ms", lat),
                Style::default().fg(if lat < 100.0 { Color::Green } else if lat < 500.0 { Color::Yellow } else { Color::Red })
            ),
            Span::raw("  Loss: "),
            Span::styled(
                format!("{:.1}%", loss),
                Style::default().fg(if loss < 2.0 { Color::Green } else if loss < 10.0 { Color::Yellow } else { Color::Red })
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Chnk: "),
            Span::styled(
                format!("{}KB", metrics.apollo_kalman_optimal_chunk_kb),
                Style::default().fg(Color::Magenta)
            ),
            Span::raw("  Par: "),
            Span::styled(
                format!("{}x", metrics.apollo_kalman_concurrency),
                Style::default().fg(Color::Cyan)
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Conf: "),
            Span::styled(
                "\u{2588}".repeat(conf_bar_w),
                Style::default().fg(conf_color)
            ),
            Span::styled(
                "\u{2591}".repeat(14_usize.saturating_sub(conf_bar_w)),
                Style::default().fg(Color::DarkGray)
            ),
            Span::raw(" "),
            Span::styled(
                format!("{}%", (confidence * 100.0).round() as u64),
                Style::default().fg(conf_color)
            ),
        ])),
    ];

    let kalman_widget = List::new(kalman_items)
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled("KALMAN Predictor", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)))
            .style(Style::default().fg(Color::Blue)));
    f.render_widget(kalman_widget, cols[0]);

    // --- PID Rate Controller ---
    let pid_target = metrics.apollo_pid_target_bps;
    let pid_current = metrics.apollo_pid_current_bps;
    let pid_drift = if pid_target > 0.0 { (pid_current - pid_target) / pid_target * 100.0 } else { 0.0 };
    let drift_color = if pid_drift.abs() < 10.0 { Color::Green } else if pid_drift.abs() < 30.0 { Color::Yellow } else { Color::Red };
    let pid_avg_err = metrics.apollo_pid_error;

    // Utilization bar: how close current is to target
    let util_ratio = if pid_target > 0.0 { (pid_current / pid_target).min(2.0) } else { 0.0 };
    let util_bar_w = ((util_ratio * 7.0).round() as usize).min(14);
    let util_color = if util_ratio > 1.2 { Color::Red } else if util_ratio > 0.8 { Color::Green } else { Color::Yellow };

    // Starship phase for PID panel
    let mode_label = match metrics.starship_phase.as_str() {
        "SUPER_HEAVY"       => ("BOOST", Color::Yellow),
        "HOT_STAGING"       => ("STAGING", Color::Magenta),
        "STARSHIP_CRUISE"   => ("CRUISE", Color::Cyan),
        "IGNITION"          => ("IGNITE", Color::Red),
        "ORBITAL_INSERTION" => ("INSERT", Color::Blue),
        "STATION_KEEPING"   => ("ORBIT", Color::Green),
        _                   => ("IDLE", Color::DarkGray),
    };

    let pid_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Tgt: "),
            Span::styled(format!("{:.0}", pid_target), Style::default().fg(Color::White)),
            Span::raw(" Act: "),
            Span::styled(
                format!("{:.0} BPS", pid_current),
                Style::default().fg(util_color).add_modifier(Modifier::BOLD)
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Drft "),
            Span::styled(
                format!("{:+.1}%", pid_drift),
                Style::default().fg(drift_color)
            ),
            Span::raw(" Err: "),
            Span::styled(
                format!("{:.1}", pid_avg_err),
                Style::default().fg(if pid_avg_err.abs() < 50.0 { Color::Green } else { Color::Yellow })
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Rate "),
            Span::styled(
                "\u{2588}".repeat(util_bar_w),
                Style::default().fg(util_color)
            ),
            Span::styled(
                "\u{2591}".repeat(14_usize.saturating_sub(util_bar_w)),
                Style::default().fg(Color::DarkGray)
            ),
        ])),
        ListItem::new(Line::from({
            let phase_dur = metrics.starship_phase_duration_secs;
            let mission_t = metrics.starship_mission_elapsed_secs;
            let dur_str = if phase_dur >= 3600 {
                format!("[{}h{}m]", phase_dur / 3600, (phase_dur % 3600) / 60)
            } else if phase_dur >= 60 {
                format!("[{}m{}s]", phase_dur / 60, phase_dur % 60)
            } else {
                format!("[{}s]", phase_dur)
            };
            let mission_str = if mission_t >= 3600 {
                format!("T+{}h{}m", mission_t / 3600, (mission_t % 3600) / 60)
            } else if mission_t >= 60 {
                format!("T+{}m", mission_t / 60)
            } else {
                format!("T+{}s", mission_t)
            };
            // Phase-specific status suffix
            let (orbit_str, orbit_color) = if metrics.starship_phase == "STATION_KEEPING" {
                if metrics.starship_orbit_stable {
                    (" STABLE", Color::Green)
                } else {
                    (" STBLZ", Color::Blue)
                }
            } else if metrics.starship_phase_bps > 100.0 {
                (">100bps", Color::Green)
            } else if metrics.starship_phase_bps > 0.0 {
                ("", Color::Yellow)  // show bps below
            } else {
                ("", Color::DarkGray)
            };
            let mut spans = vec![
                Span::styled(mode_label.0, Style::default().fg(mode_label.1).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" {}", dur_str), Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(mission_str, Style::default().fg(Color::White)),
            ];
            if !orbit_str.is_empty() {
                spans.push(Span::styled(format!(" {}", orbit_str), Style::default().fg(orbit_color)));
            } else if metrics.starship_phase_bps > 0.0 {
                spans.push(Span::styled(
                    format!(" {:.0}bps", metrics.starship_phase_bps),
                    Style::default().fg(if metrics.starship_phase_bps > 50.0 { Color::Green } else { Color::Yellow })
                ));
            }
            spans
        })),
    ];

    let pid_widget = List::new(pid_items)
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled("PID Controller", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)))
            .style(Style::default().fg(Color::Green)));
    f.render_widget(pid_widget, cols[1]);

    // --- Gravity Assist (Peer Momentum) ---
    let peers_tracked = metrics.apollo_peers_tracked;
    let in_flight = metrics.apollo_in_flight;
    let queued = metrics.apollo_queued;
    let completed = metrics.apollo_chunks_completed;
    let total = metrics.apollo_chunks_total;
    let chunk_pct = if total > 0 { completed as f64 / total as f64 * 100.0 } else { 0.0 };
    let best_peer = &metrics.apollo_gravity_best_peer;
    let best_heat = metrics.apollo_gravity_best_heat;

    // Heat visualization bar
    let heat_bar_w = ((best_heat * 14.0).round() as usize).min(14);
    let heat_color = if best_heat > 0.7 { Color::Red } else if best_heat > 0.3 { Color::Yellow } else if best_heat > 0.0 { Color::Cyan } else { Color::DarkGray };

    let gravity_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Peers "),
            Span::styled(
                format!("{}", peers_tracked),
                Style::default().fg(if peers_tracked > 3 { Color::Green } else if peers_tracked > 0 { Color::Yellow } else { Color::Red }).add_modifier(Modifier::BOLD)
            ),
            Span::raw(" Fly:"),
            Span::styled(format!("{}", in_flight), Style::default().fg(Color::Cyan)),
            Span::raw(" Q:"),
            Span::styled(format!("{}", queued), Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Chnk "),
            Span::styled(
                format!("{}/{}", completed, total),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
            Span::raw(" "),
            Span::styled(
                format!("{:.0}%", chunk_pct),
                Style::default().fg(if chunk_pct > 90.0 { Color::Green } else if chunk_pct > 50.0 { Color::Cyan } else { Color::Yellow })
            ),
        ])),
        ListItem::new(Line::from(if !best_peer.is_empty() {
            vec![
                Span::raw("Best "),
                Span::styled(format!("{}.. ", best_peer), Style::default().fg(Color::Yellow)),
            ]
        } else {
            vec![
                Span::raw("Best "),
                Span::styled("--", Style::default().fg(Color::DarkGray)),
            ]
        })),
        ListItem::new(Line::from(vec![
            Span::raw("Heat "),
            Span::styled(
                "\u{2588}".repeat(heat_bar_w),
                Style::default().fg(heat_color)
            ),
            Span::styled(
                "\u{2591}".repeat(14_usize.saturating_sub(heat_bar_w)),
                Style::default().fg(Color::DarkGray)
            ),
            Span::styled(
                if in_flight > 0 { " ACT" } else if peers_tracked > 0 { " RDY" } else { " IDL" },
                Style::default().fg(if in_flight > 0 { Color::Green } else { Color::DarkGray })
            ),
        ])),
    ];

    let gravity_widget = List::new(gravity_items)
        .block(Block::default().borders(Borders::ALL)
            .title(Span::styled("GRAVITY Assist", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)))
            .style(Style::default().fg(Color::Magenta)));
    f.render_widget(gravity_widget, cols[2]);
}

fn render_footer(f: &mut Frame, area: Rect, _app: &App) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[Tab] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Switch View │ "),
        Span::styled("[W] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Wallet │ "),
        Span::styled("[L] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Logs │ "),
        Span::styled("[M] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Menu │ "),
        Span::styled("[N] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Network │ "),
        Span::styled("[P] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Pause │ "),
        Span::styled("[Q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(footer, area);
}
