use crate::app::App;
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

    // Sample peer data (in real implementation, this would come from actual peer manager)
    let peer_rows = vec![
        Row::new(vec!["node2", "12D3Koo...jgYmG", "Inbound", "12ms", "↓2.3MB ↑1.1MB"]),
        Row::new(vec!["node3", "185.182.185.227:8081", "Outbound", "45ms", "↓1.8MB ↑0.9MB"]),
        Row::new(vec!["node4", "abc123.onion:9050", "Tor", "234ms", "↓0.5MB ↑0.3MB"]),
    ];

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
                .title(format!("🔗 Connected Peers ({}/100)", metrics.peer_count))
        );

    f.render_widget(table, area);
}

fn render_network_info(f: &mut Frame, area: Rect, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let info_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("📊 Network Statistics", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ])),
        ListItem::new(""),
        ListItem::new(Line::from(vec![
            Span::raw("Bootstrap: "),
            Span::styled("✅ Connected to 185.182.185.227:8081", Style::default().fg(Color::Green)),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("Tor Status: "),
            Span::styled(
                format!("✅ {} circuits active", metrics.tor_circuits),
                Style::default().fg(Color::Green)
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw("mDNS: "),
            Span::styled("✅ Discovering local peers", Style::default().fg(Color::Green)),
        ])),
        ListItem::new(""),
        ListItem::new(Line::from(vec![
            Span::styled("🌐 Network Topology", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ])),
        ListItem::new(""),
        ListItem::new("       [You] ─────┬──────── node2 (12ms)"),
        ListItem::new("                  ├──────── node3 (45ms)"),
        ListItem::new("                  └──────── node4 [Tor] (234ms)"),
    ];

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
