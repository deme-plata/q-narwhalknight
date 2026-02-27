#[cfg(feature = "tui")]
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
    Frame,
};

#[cfg(feature = "tui")]
use super::super::tui_app::MinerTuiApp;

#[cfg(feature = "tui")]
use std::sync::atomic::Ordering;

#[cfg(feature = "tui")]
pub fn draw_network(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Latency sparkline
            Constraint::Length(7),  // Server status
            Constraint::Length(5),  // Throttle control
            Constraint::Min(3),    // Stats
        ])
        .split(area);

    draw_latency_sparkline(f, chunks[0], app);
    draw_server_status(f, chunks[1], app);
    draw_throttle_control(f, chunks[2], app);
    draw_network_stats(f, chunks[3], app);
}

#[cfg(feature = "tui")]
fn draw_latency_sparkline(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let history: Vec<u64> = app.latency_history.iter()
        .map(|&lat| lat as u64)
        .collect();

    let current_ms = if let Some(ref state) = app.state {
        state.last_challenge_latency_us.load(Ordering::Relaxed) / 1000
    } else {
        0
    };

    let max_val = history.iter().max().copied().unwrap_or(100).max(1);

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(format!(" Challenge Latency: {}ms ", current_ms)),
        )
        .data(&history)
        .style(Style::default().fg(Color::Magenta))
        .max(max_val);

    f.render_widget(sparkline, area);
}

#[cfg(feature = "tui")]
fn draw_server_status(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (primary_url, sse, ml, fallback) = if let Some(ref state) = app.state {
        (
            state.server_url.clone(),
            state.sse_connected.load(Ordering::Relaxed),
            state.miner_link_connected.load(Ordering::Relaxed),
            state.using_fallback.load(Ordering::Relaxed),
        )
    } else {
        ("...".to_string(), false, false, false)
    };

    let primary_display = primary_url.replace("https://", "").replace("http://", "");
    let primary_status = if fallback { ("FALLBACK", Color::Yellow) } else { ("PRIMARY", Color::Green) };

    let text = vec![
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!(" {} ", primary_status.0),
                Style::default().fg(Color::Black).bg(primary_status.1).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(&primary_display, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  SSE Stream:  "),
            if sse {
                Span::styled("● Connected", Style::default().fg(Color::Green))
            } else {
                Span::styled("○ Disconnected", Style::default().fg(Color::Red))
            },
            Span::raw("    MinerLink:  "),
            if ml {
                Span::styled("● Connected", Style::default().fg(Color::Green))
            } else {
                Span::styled("○ Not connected", Style::default().fg(Color::DarkGray))
            },
        ]),
        Line::from(vec![
            Span::raw("  Fallback:    "),
            if fallback {
                Span::styled("● Active (quillon.xyz)", Style::default().fg(Color::Yellow))
            } else {
                Span::styled("○ Standby", Style::default().fg(Color::DarkGray))
            },
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue))
        .title(" Server Status ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_throttle_control(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let mode = if let Some(ref state) = app.state {
        *state.throttle_mode.read()
    } else {
        crate::shared_state::MinerThrottleMode::Off
    };

    let (mode_label, mode_color) = match mode {
        crate::shared_state::MinerThrottleMode::Off => ("OFF", Color::Green),
        crate::shared_state::MinerThrottleMode::Light => ("LIGHT (100ms delay)", Color::Yellow),
        crate::shared_state::MinerThrottleMode::Heavy => ("HEAVY (500ms delay)", Color::Red),
    };

    let text = vec![
        Line::from(vec![
            Span::raw("  Network Throttle: "),
            Span::styled(
                mode_label,
                Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("[T]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Cycle: Off → Light → Heavy → Off"),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Throttle Control ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_network_stats(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let solutions = if let Some(ref state) = app.state {
        state.solutions_found.load(Ordering::Relaxed)
    } else {
        0
    };
    let blocks = if let Some(ref state) = app.state {
        state.blocks_mined.load(Ordering::Relaxed)
    } else {
        0
    };

    let text = vec![
        Line::from(vec![
            Span::styled("  Solutions Submitted: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", solutions),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Blocks Mined:        ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", blocks),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Mining Stats ");

    f.render_widget(Paragraph::new(text).block(block), area);
}
