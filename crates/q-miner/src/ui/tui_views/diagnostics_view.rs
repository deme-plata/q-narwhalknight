#[cfg(feature = "tui")]
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

#[cfg(feature = "tui")]
use super::super::tui_app::MinerTuiApp;

#[cfg(feature = "tui")]
use crate::diagnostics::CheckStatus;

#[cfg(feature = "tui")]
pub fn draw_diagnostics(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Summary bar
            Constraint::Min(5),    // Health checks
        ])
        .split(area);

    draw_summary(f, chunks[0], app);
    draw_checks(f, chunks[1], app);
}

#[cfg(feature = "tui")]
fn draw_summary(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let passed = app.diagnostics.passed_count();
    let total = app.diagnostics.total_count();
    let color = if passed == total {
        Color::Green
    } else if passed >= total / 2 {
        Color::Yellow
    } else {
        Color::Red
    };

    let elapsed = app.diagnostics.last_run.elapsed().as_secs();
    let ago = if elapsed < 60 {
        format!("{}s ago", elapsed)
    } else {
        format!("{}m ago", elapsed / 60)
    };

    let line = Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{}/{} checks passed", passed, total),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::raw("    "),
        Span::styled(format!("Last run: {}", ago), Style::default().fg(Color::DarkGray)),
        Span::raw("    "),
        Span::styled("[R] Re-run diagnostics", Style::default().fg(Color::Cyan)),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .title(" Diagnostics Summary ");

    f.render_widget(Paragraph::new(line).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_checks(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let mut lines: Vec<Line> = Vec::new();

    for (i, check) in app.diagnostics.checks.iter().enumerate() {
        let (prefix, color) = match &check.status {
            CheckStatus::Pass => (" PASS ", Color::Green),
            CheckStatus::Warn(_) => (" WARN ", Color::Yellow),
            CheckStatus::Fail(_) => (" FAIL ", Color::Red),
        };

        // Check name line
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                prefix,
                Style::default().fg(Color::Black).bg(color).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                check.name,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            if let Some(detail) = check.status.detail() {
                Span::styled(format!("  — {}", detail), Style::default().fg(Color::DarkGray))
            } else {
                Span::raw("")
            },
        ]));

        // Fix suggestion (only for non-pass)
        if let Some(ref fix) = check.fix_suggestion {
            for fix_line in fix.lines() {
                lines.push(Line::from(vec![
                    Span::raw("           "),
                    Span::styled(
                        format!("  {}", fix_line),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
            }
        }

        // Spacing between checks
        if i < app.diagnostics.checks.len() - 1 {
            lines.push(Line::from(""));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "Press [R] to run diagnostics",
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Health Checks ");

    f.render_widget(Paragraph::new(lines).block(block), area);
}
