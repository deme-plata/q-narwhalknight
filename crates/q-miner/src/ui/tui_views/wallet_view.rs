// ═══════════════════════════════════════════════════════════════════
// Wallet View: QR Code + Balance + Easy Send
// ═══════════════════════════════════════════════════════════════════

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect, Alignment},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use std::sync::atomic::Ordering;

use super::MinerTuiApp;

/// Render a QR code as unicode block characters (2 rows per line using half-blocks)
fn render_qr_lines(data: &str) -> Vec<String> {
    use qrcode::QrCode;

    let code = match QrCode::new(data.as_bytes()) {
        Ok(c) => c,
        Err(_) => return vec!["[QR Error]".into()],
    };

    let matrix = code.to_colors();
    let width = code.width();
    let mut lines = Vec::new();

    // Use half-block characters: top half = row i, bottom half = row i+1
    // Black module = dark pixel, White module = light pixel
    // In terminal: we use inverted — light bg for "dark" QR modules
    //
    // Unicode half blocks:
    // '\u{2580}' ▀ upper half block
    // '\u{2584}' ▄ lower half block
    // '\u{2588}' █ full block
    // ' '          space (empty)

    // Add quiet zone (1 char padding)
    let qz_line: String = " ".repeat(width + 4);
    lines.push(qz_line.clone());

    let mut row = 0;
    while row < width {
        let mut line = String::with_capacity(width + 4);
        line.push_str("  "); // left quiet zone

        for col in 0..width {
            let top_dark = matrix[row * width + col] == qrcode::Color::Dark;
            let bot_dark = if row + 1 < width {
                matrix[(row + 1) * width + col] == qrcode::Color::Dark
            } else {
                false
            };

            // In terminals with dark background:
            // Dark QR module = white/light character
            // Light QR module = space (dark bg shows through)
            match (top_dark, bot_dark) {
                (true, true) => line.push('\u{2588}'),   // █ both dark
                (true, false) => line.push('\u{2580}'),  // ▀ top dark only
                (false, true) => line.push('\u{2584}'),  // ▄ bottom dark only
                (false, false) => line.push(' '),         //   both light
            }
        }
        line.push_str("  "); // right quiet zone
        lines.push(line);
        row += 2;
    }

    lines.push(qz_line);
    lines
}

pub fn draw_wallet(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let wallet_addr = app.state.as_ref()
        .map(|s| s.wallet_address.clone())
        .unwrap_or_else(|| "unknown".into());

    let server_url = app.state.as_ref()
        .map(|s| s.server_url.clone())
        .unwrap_or_default();

    // Main layout: left (QR + address) | right (balance + send)
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(45), // QR code side
            Constraint::Percentage(55), // Balance + Send side
        ])
        .split(area);

    // ═══════════════════════════════════════════════
    // LEFT: QR Code + Receive Address
    // ═══════════════════════════════════════════════

    let left_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Receive — QR Code ");

    let left_inner = left_block.inner(cols[0]);
    f.render_widget(left_block, cols[0]);

    let qr_lines = render_qr_lines(&wallet_addr);
    let qr_height = qr_lines.len() as u16;

    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(qr_height.min(left_inner.height.saturating_sub(5))), // QR
            Constraint::Length(1), // spacer
            Constraint::Length(3), // address display
            Constraint::Min(0),   // fill
        ])
        .split(left_inner);

    // QR code rendered as text
    let qr_text: Vec<Line> = qr_lines.iter()
        .map(|l| Line::from(Span::styled(l.as_str(), Style::default().fg(Color::White))))
        .collect();
    f.render_widget(
        Paragraph::new(qr_text).alignment(Alignment::Center),
        left_rows[0],
    );

    // Wallet address (copyable display)
    let addr_display = if wallet_addr.len() > 50 {
        format!("{}...{}", &wallet_addr[..24], &wallet_addr[wallet_addr.len()-24..])
    } else {
        wallet_addr.clone()
    };

    let addr_lines = vec![
        Line::from(Span::styled("Your Address:", Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled(
            addr_display,
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
    ];
    f.render_widget(
        Paragraph::new(addr_lines).alignment(Alignment::Center),
        left_rows[2],
    );

    // ═══════════════════════════════════════════════
    // RIGHT: Balance + Send Form
    // ═══════════════════════════════════════════════

    let right_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green))
        .title(" Balance & Send ");

    let right_inner = right_block.inner(cols[1]);
    f.render_widget(right_block, cols[1]);

    if app.wallet_password_setting {
        draw_password_setting(f, right_inner, app);
    } else if app.wallet_send_mode {
        draw_send_form(f, right_inner, app, &wallet_addr);
    } else {
        draw_balance_view(f, right_inner, app, &wallet_addr, &server_url);
    }
}

fn draw_balance_view(f: &mut Frame, area: Rect, app: &MinerTuiApp, wallet_addr: &str, server_url: &str) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // spacer
            Constraint::Length(4),  // balance display
            Constraint::Length(1),  // divider
            Constraint::Length(6),  // mining stats
            Constraint::Length(1),  // divider
            Constraint::Length(4),  // server info
            Constraint::Length(1),  // spacer
            Constraint::Length(3),  // send button hint
            Constraint::Min(0),    // fill
        ])
        .split(area);

    // Balance — use smoothly interpolated display value ("Mercedes" feel)
    let balance = app.wallet_balance_display;
    let balance_str = format!("{:.8}", balance);
    let balance_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Balance: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} QUG", balance_str),
                Style::default()
                    .fg(if balance > 0.0 { Color::Green } else { Color::White })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("           ", Style::default()),
            Span::styled(
                format!("{:.4}% of 21M supply", balance / 21_000_000.0 * 100.0),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(balance_lines), rows[1]);

    // Divider
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  ─────────────────────────────────",
            Style::default().fg(Color::DarkGray),
        ))),
        rows[2],
    );

    // Mining stats
    let block_h = app.current_block_height;
    let reward = app.current_block_reward;
    let khs = app.current_hashrate_khs();
    let solutions = app.state.as_ref()
        .map(|s| s.solutions_found.load(Ordering::Relaxed))
        .unwrap_or(0);

    let stats_lines = vec![
        Line::from(vec![
            Span::styled("  Block:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("#{}", block_h), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Hashrate:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1} KH/s", khs), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Solutions: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", solutions), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("  Reward:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.8} QUG", reward),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(stats_lines), rows[3]);

    // Divider
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  ─────────────────────────────────",
            Style::default().fg(Color::DarkGray),
        ))),
        rows[4],
    );

    // Server info
    let server_display = if server_url.len() > 35 {
        format!("{}...", &server_url[..35])
    } else {
        server_url.to_string()
    };
    let addr_short = if wallet_addr.len() > 20 {
        format!("{}..{}", &wallet_addr[..10], &wallet_addr[wallet_addr.len()-8..])
    } else {
        wallet_addr.to_string()
    };

    let info_lines = vec![
        Line::from(vec![
            Span::styled("  Server:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(server_display, Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::styled("  Wallet:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(addr_short, Style::default().fg(Color::Yellow)),
        ]),
    ];
    f.render_widget(Paragraph::new(info_lines), rows[5]);

    // Send status or hint
    let hint_lines = if let Some(ref status) = app.wallet_send_status {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {}", status),
                Style::default().fg(if app.wallet_send_disabled { Color::Yellow } else { Color::Green }),
            )),
        ]
    } else if app.wallet_send_disabled {
        // v10.1.2: Send disabled when using community/master wallet
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Send disabled — community pool wallet",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    } else {
        let pw_status = if app.wallet_password_hash.is_some() { "set" } else { "none" };
        vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  [S]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(" Send  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[P]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" Password ({})", pw_status), Style::default().fg(Color::DarkGray)),
            ]),
        ]
    };
    f.render_widget(Paragraph::new(hint_lines), rows[7]);
}

fn draw_send_form(f: &mut Frame, area: Rect, app: &MinerTuiApp, _wallet_addr: &str) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),  // title
            Constraint::Length(1),  // spacer
            Constraint::Length(3),  // to address field
            Constraint::Length(1),  // spacer
            Constraint::Length(3),  // amount field
            Constraint::Length(1),  // spacer
            Constraint::Length(4),  // confirmation or buttons
            Constraint::Length(1),  // spacer
            Constraint::Length(2),  // hints
            Constraint::Min(0),    // fill
        ])
        .split(area);

    // Title
    let title_lines = vec![
        Line::from(Span::styled(
            "  Send QUG",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )),
    ];
    f.render_widget(Paragraph::new(title_lines), rows[0]);

    // Address field
    let addr_border_color = if app.wallet_send_field == 0 { Color::Cyan } else { Color::DarkGray };
    let addr_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(addr_border_color))
        .title(Span::styled(" To Address ", Style::default().fg(addr_border_color)));

    let cursor = if app.wallet_send_field == 0 && !app.wallet_send_confirming { "█" } else { "" };
    let addr_text = format!("{}{}", app.wallet_send_address, cursor);
    let addr_para = Paragraph::new(Line::from(Span::styled(
        addr_text,
        Style::default().fg(Color::Yellow),
    )))
    .block(addr_block);
    f.render_widget(addr_para, rows[2]);

    // Amount field
    let amt_border_color = if app.wallet_send_field == 1 { Color::Cyan } else { Color::DarkGray };
    let amt_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(amt_border_color))
        .title(Span::styled(" Amount (QUG) ", Style::default().fg(amt_border_color)));

    let cursor2 = if app.wallet_send_field == 1 && !app.wallet_send_confirming { "█" } else { "" };
    let amt_text = format!("{}{}", app.wallet_send_amount, cursor2);
    let amt_para = Paragraph::new(Line::from(Span::styled(
        amt_text,
        Style::default().fg(Color::Green),
    )))
    .block(amt_block);
    f.render_widget(amt_para, rows[4]);

    // Confirmation or action area
    if app.wallet_send_confirming {
        let mut confirm_lines = vec![
            Line::from(Span::styled(
                "  Confirm transaction?",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("  Send ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{} QUG", app.wallet_send_amount),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" to ", Style::default().fg(Color::White)),
                Span::styled(
                    if app.wallet_send_address.len() > 16 {
                        format!("{}..{}", &app.wallet_send_address[..8], &app.wallet_send_address[app.wallet_send_address.len()-6..])
                    } else {
                        app.wallet_send_address.clone()
                    },
                    Style::default().fg(Color::Yellow),
                ),
            ]),
        ];
        // v8.6.5: Show password prompt if password is set
        if app.wallet_password_hash.is_some() {
            let dots = "*".repeat(app.wallet_send_password.len());
            let pw_color = if app.wallet_send_password_err { Color::Red } else { Color::Cyan };
            confirm_lines.push(Line::from(vec![
                Span::styled("  Password: ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{}█", dots), Style::default().fg(pw_color)),
                if app.wallet_send_password_err {
                    Span::styled("  Wrong password!", Style::default().fg(Color::Red))
                } else {
                    Span::styled("", Style::default())
                },
            ]));
        }
        confirm_lines.push(Line::from(vec![
            Span::styled("  [Enter] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("Confirm   ", Style::default().fg(Color::White)),
            Span::styled("[Esc] ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled("Cancel", Style::default().fg(Color::White)),
        ]));
        f.render_widget(Paragraph::new(confirm_lines), rows[6]);
    } else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(vec![
                    Span::styled("  [Enter] ", Style::default().fg(Color::Cyan)),
                    Span::styled("Review & Send", Style::default().fg(Color::White)),
                ]),
            ]),
            rows[6],
        );
    }

    // Hints
    let hint_lines = vec![
        Line::from(vec![
            Span::styled("  [Tab]", Style::default().fg(Color::Cyan)),
            Span::styled(" Switch field  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[Esc]", Style::default().fg(Color::Cyan)),
            Span::styled(" Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];
    f.render_widget(Paragraph::new(hint_lines), rows[8]);
}

/// v8.6.5: Password setting screen
fn draw_password_setting(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),  // title
            Constraint::Length(1),  // spacer
            Constraint::Length(3),  // password field
            Constraint::Length(1),  // spacer
            Constraint::Length(3),  // info
            Constraint::Length(1),  // spacer
            Constraint::Length(2),  // hints
            Constraint::Min(0),    // fill
        ])
        .split(area);

    let title = if app.wallet_password_hash.is_some() {
        "  Change Wallet Send Password"
    } else {
        "  Set Wallet Send Password"
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            title,
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ))),
        rows[0],
    );

    // Password field
    let dots = "*".repeat(app.wallet_password_new.len());
    let pw_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(" New Password (min 4 chars) ", Style::default().fg(Color::Cyan)));
    let pw_para = Paragraph::new(Line::from(Span::styled(
        format!("{}█", dots),
        Style::default().fg(Color::White),
    )))
    .block(pw_block);
    f.render_widget(pw_para, rows[2]);

    // Info
    let info_lines = vec![
        Line::from(Span::styled(
            "  Password protects the Send function.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  You'll need to enter it before each send.",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    f.render_widget(Paragraph::new(info_lines), rows[4]);

    // Hints
    let hint_lines = vec![
        Line::from(vec![
            Span::styled("  [Enter]", Style::default().fg(Color::Cyan)),
            Span::styled(" Save  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[Esc]", Style::default().fg(Color::Cyan)),
            Span::styled(" Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];
    f.render_widget(Paragraph::new(hint_lines), rows[6]);
}
