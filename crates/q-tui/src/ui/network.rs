use crate::app::App;
use crate::metrics::Metrics;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table},
    Frame,
};

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Header
            Constraint::Length(12),  // Peer table
            Constraint::Min(8),      // Network topology
            Constraint::Length(3),   // Footer
        ])
        .split(f.size());

    render_header(f, chunks[0], app);
    render_peer_table(f, chunks[1], app);
    render_network_info(f, chunks[2], app);
    render_footer(f, chunks[3]);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let header = Paragraph::new(Line::from(vec![
        Span::styled("Q-NarwhalKnight ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("│ "),
        Span::styled("🌐 NETWORK VIEW", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        Span::raw(" │ Peers: "),
        Span::styled(
            format!("{}/100", metrics.peer_count),
            Style::default().fg(if metrics.peer_count > 0 { Color::Green } else { Color::Red })
        ),
        Span::raw(" │ "),
        Span::styled("[Tab] ", Style::default().fg(Color::DarkGray)),
        Span::raw("Next │ "),
        Span::styled("[Esc] ", Style::default().fg(Color::DarkGray)),
        Span::raw("Dashboard"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Left);

    f.render_widget(header, area);
}

fn render_peer_table(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let peer_rows: Vec<Row> = if metrics.peer_count == 0 {
        vec![Row::new(vec!["", "No peers connected", "", "", ""])]
    } else {
        // Show summary row based on real peer counts
        let mut rows = Vec::new();
        if metrics.inbound_peers > 0 {
            rows.push(Row::new(vec![
                "inbound".to_string(),
                format!("{} peers", metrics.inbound_peers),
                "In".to_string(),
                "-".to_string(),
                format!("↓{}", Metrics::format_bytes(metrics.bytes_received)),
            ]));
        }
        if metrics.outbound_peers > 0 {
            rows.push(Row::new(vec![
                "outbound".to_string(),
                format!("{} peers", metrics.outbound_peers),
                "Out".to_string(),
                "-".to_string(),
                format!("↑{}", Metrics::format_bytes(metrics.bytes_sent)),
            ]));
        }
        if rows.is_empty() {
            rows.push(Row::new(vec![
                "peers".to_string(),
                format!("{} connected", metrics.peer_count),
                "-".to_string(),
                "-".to_string(),
                "-".to_string(),
            ]));
        }
        rows
    };

    let widths = [
        Constraint::Length(8),
        Constraint::Length(25),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(15),
    ];

    let table = Table::new(peer_rows, widths)
        .header(
            Row::new(vec!["ID", "Address", "Type", "Latency", "Traffic"])
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        )
        .column_spacing(2)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("🔗 Connected Peers ({})", metrics.peer_count))
        );

    f.render_widget(table, area);
}

fn render_network_info(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let bootstrap_status = if metrics.peer_count > 0 {
        Span::styled("Connected", Style::default().fg(Color::Green))
    } else {
        Span::styled("Searching...", Style::default().fg(Color::Yellow))
    };

    let net_id = if metrics.network_id.is_empty() { "unknown" } else { &metrics.network_id };

    let mut info_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("📊 Network Statistics", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ])),
        ListItem::new(""),
        ListItem::new(Line::from(vec![
            Span::raw("Network:    "),
            Span::styled(net_id.to_string(), Style::default().fg(Color::Cyan)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Bootstrap:  "),
            bootstrap_status,
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Peers:      "),
            Span::styled(
                format!("{} (in: {} / out: {})", metrics.peer_count, metrics.inbound_peers, metrics.outbound_peers),
                Style::default().fg(Color::Green)
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Height:     "),
            Span::styled(
                format!("{} / net: {}", metrics.block_height, metrics.network_height),
                Style::default().fg(Color::Cyan)
            ),
        ])),
    ];

    if metrics.tor_circuits > 0 {
        info_items.push(ListItem::new(Line::from(vec![
            Span::raw("Tor:        "),
            Span::styled(
                format!("{} circuits active", metrics.tor_circuits),
                Style::default().fg(Color::Green)
            ),
        ])));
    }

    info_items.push(ListItem::new(""));
    info_items.push(ListItem::new(Line::from(vec![
        Span::raw("Bandwidth:  "),
        Span::styled(
            format!("↓{}/s  ↑{}/s", Metrics::format_bytes(metrics.bytes_received), Metrics::format_bytes(metrics.bytes_sent)),
            Style::default().fg(Color::Blue)
        ),
    ])));

    let info = List::new(info_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Network Information")
        );

    f.render_widget(info, area);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[D] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Disconnect Peer │ "),
        Span::styled("[B] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Ban Peer │ "),
        Span::styled("[A] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Add Peer │ "),
        Span::styled("[Tab] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Next View │ "),
        Span::styled("[Q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(footer, area);
}
