use crate::app::{App, LogLevel};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use ringbuf::Rb;

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Header
            Constraint::Min(0),      // Logs (fill remaining space)
            Constraint::Length(3),   // Footer
        ])
        .split(f.size());

    render_header(f, chunks[0], app);
    render_logs(f, chunks[1], app);
    render_footer(f, chunks[2]);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let pause_text = if app.logs_paused {
        Span::styled(" [PAUSED]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    } else {
        Span::raw("")
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled("Q-NarwhalKnight ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("│ "),
        Span::styled("📝 LOGS VIEW", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        pause_text,
        Span::raw(" │ Filter: [ALL] │ "),
        Span::styled("[Esc] ", Style::default().fg(Color::DarkGray)),
        Span::raw("Dashboard"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Left);

    f.render_widget(header, area);
}

fn render_logs(f: &mut Frame, area: Rect, app: &App) {
    let available_height = area.height.saturating_sub(2);
    let logs = app.get_recent_logs(available_height as usize);

    let log_items: Vec<ListItem> = logs
        .iter()
        .rev()
        .map(|log| {
            let time_str = log.timestamp.format("%H:%M:%S.%3f").to_string();
            let level_color = match log.level {
                LogLevel::Trace => Color::DarkGray,
                LogLevel::Debug => Color::Gray,
                LogLevel::Info => Color::Cyan,
                LogLevel::Warn => Color::Yellow,
                LogLevel::Error => Color::Red,
            };

            // Format: [HH:MM:SS.mmm] LEVEL  target: message
            ListItem::new(vec![
                Line::from(vec![
                    Span::raw("["),
                    Span::styled(time_str, Style::default().fg(Color::DarkGray)),
                    Span::raw("] "),
                    Span::styled(
                        log.level.as_str(),
                        Style::default().fg(level_color).add_modifier(Modifier::BOLD)
                    ),
                    Span::raw(" "),
                    Span::styled(&log.target, Style::default().fg(Color::Blue)),
                    Span::raw(": "),
                ]),
                Line::from(vec![
                    Span::raw("           "),
                    Span::raw(&log.message),
                ]),
            ])
        })
        .collect();

    let total_logs = if let Ok(logs) = app.logs.read() {
        logs.len()
    } else {
        0
    };

    let logs_list = List::new(log_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("Logs ({} total, showing last {}) - Scroll: ↑↓ / PgUp/PgDn", total_logs, logs.len()))
        );

    f.render_widget(logs_list, area);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[P] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Pause │ "),
        Span::styled("[C] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Clear │ "),
        Span::styled("[F] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Filter │ "),
        Span::styled("[/] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Search │ "),
        Span::styled("[Tab] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Dashboard │ "),
        Span::styled("[Q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(footer, area);
}
