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
            Constraint::Length(7),   // TPS Chart or AI Metrics
            Constraint::Min(6),      // Logs (smaller when syncing)
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

    render_tps_or_ai_metrics(f, chunks[idx], app);
    idx += 1;

    render_recent_logs(f, chunks[idx], app);
    idx += 1;

    render_footer(f, chunks[idx], app);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let status_text = if metrics.peer_count > 0 {
        Span::styled("✅ SYNCED", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
    } else {
        Span::styled("⚠ CONNECTING", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled("Q-NarwhalKnight ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("v0.0.7-beta │ "),
        Span::raw("Status: "),
        status_text,
        Span::raw(" │ Uptime: "),
        Span::styled(
            Metrics::format_uptime(metrics.uptime_secs),
            Style::default().fg(Color::Green)
        ),
        Span::raw(" │ "),
        Span::styled("[Q] Quit", Style::default().fg(Color::DarkGray)),
    ]))
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

    // Network metrics
    let network_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Peers:        "),
            Span::styled(
                format!("{}/{}", metrics.peer_count, 100),
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

    // Blockchain metrics
    let blockchain_items = vec![
        ListItem::new(Line::from(vec![
            Span::raw("Height:      "),
            Span::styled(
                format!("{}", metrics.block_height),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
        ])),
        ListItem::new(format!("DAG Size:     {:.1} MB", metrics.dag_size_mb)),
        ListItem::new(Line::from(vec![
            Span::raw("Last Block:  "),
            Span::styled(
                format!("{}s ago", metrics.last_block_secs),
                Style::default().fg(if metrics.last_block_secs < 5 { Color::Green } else { Color::Yellow })
            ),
        ])),
        ListItem::new(format!("Anchors:      {}", metrics.anchor_count)),
        ListItem::new(format!("Vertices:     {}", metrics.vertex_count)),
    ];

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
    let bar_width = (area.width as f32 * progress / 100.0) as u16;

    let eta_str = if metrics.sync_speed_blocks_per_sec > 0.0 {
        let blocks_remaining = metrics.sync_target_height.saturating_sub(metrics.sync_current_height) as f32;
        let secs_remaining = blocks_remaining / metrics.sync_speed_blocks_per_sec;
        let mins = (secs_remaining / 60.0) as u32;
        if mins > 60 {
            format!("ETA: {}h {}m", mins / 60, mins % 60)
        } else {
            format!("ETA: {}m", mins)
        }
    } else {
        "Calculating...".to_string()
    };

    let sync_text = vec![
        Line::from(vec![
            Span::raw("Syncing: "),
            Span::styled(
                format!("{}/{} ", metrics.sync_current_height, metrics.sync_target_height),
                Style::default().fg(Color::Cyan)
            ),
            Span::raw(format!("({:.1}%) │ Speed: {:.1} blocks/s │ {}",
                progress, metrics.sync_speed_blocks_per_sec, eta_str)),
        ]),
        Line::from(vec![
            Span::styled(
                "█".repeat(bar_width as usize),
                Style::default().fg(Color::Green)
            ),
            Span::styled(
                "░".repeat((area.width.saturating_sub(bar_width + 2)) as usize),
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

fn render_footer(f: &mut Frame, area: Rect, _app: &App) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[Tab] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Switch View │ "),
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
