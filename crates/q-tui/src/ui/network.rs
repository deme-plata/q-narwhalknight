use crate::app::App;
use crate::metrics::Metrics;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Sparkline, Table},
    Frame,
};

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header with throttle mode
            Constraint::Length(7),  // Bandwidth sparklines
            Constraint::Length(5),  // Usage analysis
            Constraint::Length(8),  // Peer table
            Constraint::Min(0),    // Flex space
            Constraint::Length(3),  // Footer
        ])
        .split(f.size());

    render_header(f, chunks[0], app);
    render_bandwidth_sparklines(f, chunks[1], app);
    render_usage_analysis(f, chunks[2], app);
    render_peer_table(f, chunks[3], app);
    render_footer(f, chunks[5]);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();
    let mode = app.network_throttle_mode;
    let mode_color = match mode {
        crate::metrics::NetworkThrottleMode::Conservative => Color::Yellow,
        crate::metrics::NetworkThrottleMode::Normal => Color::Green,
        crate::metrics::NetworkThrottleMode::Turbo => Color::Magenta,
    };

    // v8.5.9: Show disk I/O impact alongside throttle mode
    let disk_info = match mode {
        crate::metrics::NetworkThrottleMode::Conservative => " | Disk: SSD-safe",
        crate::metrics::NetworkThrottleMode::Normal => " | Disk: moderate",
        crate::metrics::NetworkThrottleMode::Turbo => " | Disk: full speed",
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled("Q-NarwhalKnight ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("| "),
        Span::styled("NETWORK", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        Span::raw(" | Peers: "),
        Span::styled(
            format!("{}/100", metrics.peer_count),
            Style::default().fg(if metrics.peer_count > 0 { Color::Green } else { Color::Red }),
        ),
        Span::raw(" | Mode: "),
        Span::styled(
            format!("[{}]", mode.as_str()),
            Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(disk_info.to_string(), Style::default().fg(mode_color)),
        Span::raw(" "),
        Span::styled("[T]", Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Left);

    f.render_widget(header, area);
}

fn render_bandwidth_sparklines(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();
    let bw_in_history = app.get_bw_in_history();
    let bw_out_history = app.get_bw_out_history();

    // Split into two rows: download and upload sparklines
    let spark_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Download sparkline
            Constraint::Length(3), // Upload sparkline
        ])
        .margin(0)
        .split(area);

    // Convert u64 history to u64 (already u64, just need to collect to vec)
    let in_data: Vec<u64> = bw_in_history;
    let out_data: Vec<u64> = bw_out_history;

    let in_rate = Metrics::format_bytes_rate(metrics.bytes_in_per_sec);
    let out_rate = Metrics::format_bytes_rate(metrics.bytes_out_per_sec);

    // Download sparkline
    let dl_sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Download  {} ", in_rate)),
        )
        .data(&in_data)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(dl_sparkline, spark_chunks[0]);

    // Upload sparkline
    let ul_sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Upload  {} ", out_rate)),
        )
        .data(&out_data)
        .style(Style::default().fg(Color::Green));
    f.render_widget(ul_sparkline, spark_chunks[1]);
}

fn render_usage_analysis(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let total_in = Metrics::format_bytes(metrics.total_bytes_in);
    let total_out = Metrics::format_bytes(metrics.total_bytes_out);
    let rate_in = Metrics::format_bytes_rate(metrics.bytes_in_per_sec);
    let rate_out = Metrics::format_bytes_rate(metrics.bytes_out_per_sec);
    let peak_in = Metrics::format_bytes_rate(app.peak_bw_in);
    let peak_out = Metrics::format_bytes_rate(app.peak_bw_out);
    let uptime = Metrics::format_uptime(metrics.uptime_secs);

    // Calculate average rate
    let avg_rate = if metrics.uptime_secs > 0 {
        let total = metrics.total_bytes_in + metrics.total_bytes_out;
        Metrics::format_bytes_rate(total / metrics.uptime_secs)
    } else {
        "0 B/s".to_string()
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("  Total In:  "),
            Span::styled(format!("{:<12}", total_in), Style::default().fg(Color::Cyan)),
            Span::raw("Rate: "),
            Span::styled(format!("{:<12}", rate_in), Style::default().fg(Color::Cyan)),
            Span::raw("Peak: "),
            Span::styled(peak_in, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("  Total Out: "),
            Span::styled(format!("{:<12}", total_out), Style::default().fg(Color::Green)),
            Span::raw("Rate: "),
            Span::styled(format!("{:<12}", rate_out), Style::default().fg(Color::Green)),
            Span::raw("Peak: "),
            Span::styled(peak_out, Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("  Session:   "),
            Span::styled(format!("{:<12}", uptime), Style::default().fg(Color::White)),
            Span::raw("Avg:  "),
            Span::styled(avg_rate, Style::default().fg(Color::White)),
        ]),
    ];

    let analysis = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Usage Analysis "),
    );

    f.render_widget(analysis, area);
}

fn render_peer_table(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let net_id = if metrics.network_id.is_empty() {
        "unknown"
    } else {
        &metrics.network_id
    };

    let bootstrap_str = if metrics.peer_count > 0 {
        "Connected"
    } else {
        "Searching..."
    };

    let mut rows: Vec<Row> = Vec::new();

    // Network info row
    rows.push(
        Row::new(vec![
            "network".to_string(),
            net_id.to_string(),
            bootstrap_str.to_string(),
            format!("h:{}", metrics.block_height),
            format!("net:{}", metrics.network_height),
        ])
        .style(Style::default().fg(Color::White)),
    );

    // Peer summary rows
    if metrics.inbound_peers > 0 {
        rows.push(Row::new(vec![
            "inbound".to_string(),
            format!("{} peers", metrics.inbound_peers),
            "In".to_string(),
            "-".to_string(),
            format!("{}", Metrics::format_bytes(metrics.total_bytes_in)),
        ]));
    }
    if metrics.outbound_peers > 0 {
        rows.push(Row::new(vec![
            "outbound".to_string(),
            format!("{} peers", metrics.outbound_peers),
            "Out".to_string(),
            "-".to_string(),
            format!("{}", Metrics::format_bytes(metrics.total_bytes_out)),
        ]));
    }
    if metrics.peer_count > 0 && metrics.inbound_peers == 0 && metrics.outbound_peers == 0 {
        rows.push(Row::new(vec![
            "peers".to_string(),
            format!("{} connected", metrics.peer_count),
            "-".to_string(),
            "-".to_string(),
            format!(
                "{}/{}",
                Metrics::format_bytes(metrics.total_bytes_in),
                Metrics::format_bytes(metrics.total_bytes_out)
            ),
        ]));
    }

    if metrics.tor_circuits > 0 {
        rows.push(
            Row::new(vec![
                "tor".to_string(),
                format!("{} circuits", metrics.tor_circuits),
                "Active".to_string(),
                "-".to_string(),
                "-".to_string(),
            ])
            .style(Style::default().fg(Color::Magenta)),
        );
    }

    if rows.len() <= 1 && metrics.peer_count == 0 {
        rows.push(Row::new(vec![
            "".to_string(),
            "No peers connected".to_string(),
            "".to_string(),
            "".to_string(),
            "".to_string(),
        ]));
    }

    let widths = [
        Constraint::Length(8),
        Constraint::Length(20),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Min(14),
    ];

    let table = Table::new(rows, widths)
        .header(
            Row::new(vec!["ID", "Info", "Type", "Height", "Traffic"])
                .style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
        )
        .column_spacing(1)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Connected Peers ({}) ", metrics.peer_count)),
        );

    f.render_widget(table, area);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[T] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Throttle | "),
        Span::styled("[Tab] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Next View | "),
        Span::styled("[D] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Dashboard | "),
        Span::styled("[Q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(footer, area);
}
