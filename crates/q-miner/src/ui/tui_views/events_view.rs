#[cfg(feature = "tui")]
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

#[cfg(feature = "tui")]
use super::super::tui_app::{LogLevel, MinerTuiApp};

#[cfg(feature = "tui")]
pub fn draw_events(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Filter bar + badge counts
            Constraint::Min(3),    // Scrollable log
        ])
        .split(area);

    draw_filter_bar(f, chunks[0], app);
    draw_log_entries(f, chunks[1], app);
}

#[cfg(feature = "tui")]
fn draw_filter_bar(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (info_count, warn_count, error_count, success_count) = count_by_level(app);

    let filter_labels = [
        ("1:All", 0),
        ("2:Info+", 1),
        ("3:Warn+", 2),
        ("4:Error", 3),
    ];

    let mut spans = vec![Span::raw("  Filter: ")];

    for (label, level) in &filter_labels {
        let is_active = app.log_filter == *level;
        let style = if is_active {
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        spans.push(Span::styled(format!(" {} ", label), style));
        spans.push(Span::raw(" "));
    }

    spans.push(Span::raw("    "));
    spans.push(Span::styled(format!("I:{}", info_count), Style::default().fg(Color::White)));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(format!("W:{}", warn_count), Style::default().fg(Color::Yellow)));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(format!("E:{}", error_count), Style::default().fg(Color::Red)));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(format!("S:{}", success_count), Style::default().fg(Color::Green)));

    let line = Line::from(spans);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    f.render_widget(Paragraph::new(line).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_log_entries(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let visible_height = area.height.saturating_sub(2) as usize;

    // Filter logs by current filter level
    let filtered: Vec<_> = app.logs.iter()
        .filter(|entry| match app.log_filter {
            0 => true,                // All
            1 => !matches!(entry.level, LogLevel::Success), // Info+
            2 => matches!(entry.level, LogLevel::Warn | LogLevel::Error), // Warn+
            3 => matches!(entry.level, LogLevel::Error),    // Error only
            _ => true,
        })
        .collect();

    let total_filtered = filtered.len();

    // Calculate scroll position
    let scroll_offset = if app.log_scroll_offset == 0 {
        // Auto-scroll: show latest
        if total_filtered > visible_height {
            total_filtered - visible_height
        } else {
            0
        }
    } else {
        // Manual scroll
        let max_scroll = if total_filtered > visible_height {
            total_filtered - visible_height
        } else {
            0
        };
        max_scroll.saturating_sub(app.log_scroll_offset)
    };

    let lines: Vec<Line> = filtered
        .iter()
        .skip(scroll_offset)
        .take(visible_height)
        .map(|entry| {
            let (prefix, color) = match entry.level {
                LogLevel::Error => ("ERR", Color::Red),
                LogLevel::Warn => ("WRN", Color::Yellow),
                LogLevel::Info => ("INF", Color::White),
                LogLevel::Success => (" OK", Color::Green),
            };
            Line::from(vec![
                Span::styled(
                    format!(" {} ", &entry.timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{} ", prefix),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(&entry.message, Style::default().fg(color)),
            ])
        })
        .collect();

    let scroll_indicator = if total_filtered > visible_height {
        let pos = scroll_offset + visible_height;
        format!(" {}/{} ", pos.min(total_filtered), total_filtered)
    } else {
        format!(" {}/{} ", total_filtered, total_filtered)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(format!(" Events Log {} [Up/Down/PgUp/PgDn] ", scroll_indicator));

    f.render_widget(Paragraph::new(lines).block(block), area);
}

#[cfg(feature = "tui")]
fn count_by_level(app: &MinerTuiApp) -> (usize, usize, usize, usize) {
    let mut info = 0;
    let mut warn = 0;
    let mut error = 0;
    let mut success = 0;
    for entry in &app.logs {
        match entry.level {
            LogLevel::Info => info += 1,
            LogLevel::Warn => warn += 1,
            LogLevel::Error => error += 1,
            LogLevel::Success => success += 1,
        }
    }
    (info, warn, error, success)
}
