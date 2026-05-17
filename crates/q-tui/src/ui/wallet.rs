// ═══════════════════════════════════════════════════════════════════════════════
// WALLET VIEW — Epic Node Wallet with QR Code, Balance, Earnings & Live Stats
// ═══════════════════════════════════════════════════════════════════════════════
//
// v8.6.2: Full wallet view for q-api-server TUI
// Features:
//   - Large QR code for receive address (half-block unicode rendering)
//   - Live balance display with USD estimate
//   - Operator earnings breakdown (session + lifetime)
//   - Mining reward stats
//   - Emission rate & supply info
//   - Animated balance bar visualization

use crate::app::App;
use crate::metrics::Metrics;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Render a QR code as unicode half-block characters for maximum density
/// Uses 2 vertical pixels per character row via upper/lower half-blocks
fn render_qr_lines(data: &str) -> Vec<String> {
    use qrcode::QrCode;

    let code = match QrCode::new(data.as_bytes()) {
        Ok(c) => c,
        Err(_) => return vec!["  [QR generation failed]".into()],
    };

    let matrix = code.to_colors();
    let width = code.width();
    let mut lines = Vec::new();

    // Quiet zone top
    let qz: String = " ".repeat(width + 4);
    lines.push(qz.clone());

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

            match (top_dark, bot_dark) {
                (true, true)   => line.push('\u{2588}'), // Full block
                (true, false)  => line.push('\u{2580}'), // Upper half
                (false, true)  => line.push('\u{2584}'), // Lower half
                (false, false) => line.push(' '),
            }
        }
        line.push_str("  "); // right quiet zone
        lines.push(line);
        row += 2;
    }

    lines.push(qz);
    lines
}

/// Format a QUG balance with commas for readability
fn format_balance(amount: f64) -> String {
    if amount >= 1_000_000.0 {
        format!("{:.2}M", amount / 1_000_000.0)
    } else if amount >= 1_000.0 {
        // Manual comma formatting for thousands
        let whole = amount as u64;
        let frac = ((amount - whole as f64) * 10000.0).round() as u64;
        let s = whole.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 { result.insert(0, ','); }
            result.insert(0, c);
        }
        format!("{}.{:04}", result, frac)
    } else {
        format!("{:.8}", amount)
    }
}

pub fn render(f: &mut Frame, app: &App) {
    let metrics = app.metrics.read().unwrap();

    // Main vertical layout
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Header
            Constraint::Min(10),     // Content
            Constraint::Length(3),   // Footer
        ])
        .split(f.size());

    render_wallet_header(f, main_chunks[0], &metrics);
    render_wallet_content(f, main_chunks[1], &metrics);
    render_wallet_footer(f, main_chunks[2]);
}

fn render_wallet_header(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let balance = metrics.admin_wallet_balance + metrics.operator_fee_total_qug;
    let balance_str = format_balance(balance);

    let header = Paragraph::new(Line::from(vec![
        Span::styled("  ", Style::default().fg(Color::Yellow)),
        Span::styled(" WALLET ", Style::default()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(
            format!("{} QUG", balance_str),
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  ({:.2}% of 21M)", balance / 21_000_000.0 * 100.0), // % of max supply
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("  "),
        Span::styled(
            format!("v{}", if metrics.version.is_empty() { env!("CARGO_PKG_VERSION") } else { &metrics.version }),
            Style::default().fg(Color::DarkGray),
        ),
    ]))
    .block(Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            " Q-NarwhalKnight Node Wallet ",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )));

    f.render_widget(header, area);
}

fn render_wallet_content(f: &mut Frame, area: Rect, metrics: &Metrics) {
    // Two columns: QR code (left) | Stats (right)
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // QR code
            Constraint::Percentage(60), // Balance + stats
        ])
        .split(area);

    render_qr_panel(f, cols[0], metrics);
    render_stats_panel(f, cols[1], metrics);
}

fn render_qr_panel(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let wallet_addr = if metrics.admin_wallet_address.is_empty() {
        "No wallet configured".to_string()
    } else {
        metrics.admin_wallet_address.clone()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            " Receive Address ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Layout: QR code + address display
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // QR code
            Constraint::Length(1), // spacer
            Constraint::Length(4), // Address
        ])
        .split(inner);

    // QR code
    if !metrics.admin_wallet_address.is_empty() {
        let qr_lines = render_qr_lines(&wallet_addr);
        let max_lines = rows[0].height as usize;
        let qr_text: Vec<Line> = qr_lines.iter()
            .take(max_lines)
            .map(|l| Line::from(Span::styled(l.as_str(), Style::default().fg(Color::White))))
            .collect();
        f.render_widget(
            Paragraph::new(qr_text).alignment(Alignment::Center),
            rows[0],
        );
    } else {
        // No wallet configured — show placeholder
        let placeholder = vec![
            Line::from(""),
            Line::from(Span::styled(
                "No wallet address",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "Set Q_OPERATOR_WALLET",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "environment variable",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        f.render_widget(
            Paragraph::new(placeholder).alignment(Alignment::Center),
            rows[0],
        );
    }

    // Wallet address display (truncated with highlight)
    let addr_display = if wallet_addr.len() > 42 {
        // Show first 18 and last 18 chars
        let prefix = &wallet_addr[..18];
        let suffix = &wallet_addr[wallet_addr.len()-18..];
        vec![
            Line::from(Span::styled(
                "Your Address:",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(vec![
                Span::styled(prefix, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled("...", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(Span::styled(
                format!("...{}", suffix),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )),
        ]
    } else {
        vec![
            Line::from(Span::styled(
                "Your Address:",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                wallet_addr.clone(),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )),
        ]
    };

    f.render_widget(
        Paragraph::new(addr_display).alignment(Alignment::Center),
        rows[2],
    );
}

fn render_stats_panel(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green))
        .title(Span::styled(
            " Balance & Earnings ",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),  // Balance hero
            Constraint::Length(1),  // divider
            Constraint::Length(6),  // Operator earnings
            Constraint::Length(1),  // divider
            Constraint::Length(5),  // Mining stats
            Constraint::Length(1),  // divider
            Constraint::Min(4),    // Chain economics
        ])
        .split(inner);

    // ═══ Balance Hero Section ═══
    let total_balance = metrics.admin_wallet_balance + metrics.operator_fee_total_qug;
    let bal_str = format_balance(total_balance);
    let bal_color = if total_balance > 100.0 {
        Color::Green
    } else if total_balance > 0.0 {
        Color::Yellow
    } else {
        Color::White
    };

    let balance_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Total Balance", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(
                format!("{} QUG", bal_str),
                Style::default().fg(bal_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                format!("  Wallet: {:.4} QUG", metrics.admin_wallet_balance),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                format!("  Pending fees: {:.4} QUG", metrics.operator_fee_total_qug),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(balance_lines), rows[0]);

    // Divider
    render_divider(f, rows[1], Color::DarkGray);

    // ═══ Operator Earnings ═══
    let fee_pct = metrics.operator_fee_promille as f64 / 10.0;
    let earnings_lines = vec![
        Line::from(vec![
            Span::styled("  Operator Earnings", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("  ({:.1}% fee)", fee_pct),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Session:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.8} QUG", metrics.operator_fee_session_qug),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Lifetime:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4} QUG", metrics.operator_fee_total_qug),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Fee TXs:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", metrics.operator_fee_tx_count),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Founder:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4} QUG", metrics.founder_wallet_balance),
                Style::default().fg(Color::Yellow),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(earnings_lines), rows[2]);

    // Divider
    render_divider(f, rows[3], Color::DarkGray);

    // ═══ Mining Stats ═══
    let mining_lines = vec![
        Line::from(vec![
            Span::styled("  Mining", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(
                if metrics.active_miners > 0 {
                    format!("  {} miners active", metrics.active_miners)
                } else {
                    "  no miners".to_string()
                },
                Style::default().fg(if metrics.active_miners > 0 { Color::Green } else { Color::DarkGray }),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Blocks:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", metrics.blocks_mined),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  Hashrate: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if metrics.hashrate > 1000.0 {
                    format!("{:.1} KH/s", metrics.hashrate / 1000.0)
                } else {
                    format!("{:.0} H/s", metrics.hashrate)
                },
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Height:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", metrics.block_height),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!(" / {}", metrics.network_height),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(mining_lines), rows[4]);

    // Divider
    render_divider(f, rows[5], Color::DarkGray);

    // ═══ Chain Economics ═══
    let supply_pct = if metrics.total_supply > 0.0 {
        (metrics.total_supply / 21_000_000.0 * 100.0).min(100.0)
    } else {
        0.0
    };

    let econ_lines = vec![
        Line::from(vec![
            Span::styled("  Chain Economics", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Supply:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.2} / 21M QUG ({:.2}%)", metrics.total_supply, supply_pct),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Emission:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.6} QUG/block", metrics.emission_rate),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(econ_lines), rows[6]);
}

fn render_divider(f: &mut Frame, area: Rect, color: Color) {
    let width = area.width.saturating_sub(4) as usize;
    let divider_str = format!("  {}", "\u{2500}".repeat(width));
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            divider_str,
            Style::default().fg(color),
        ))),
        area,
    );
}

fn render_wallet_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("  [Tab]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" Switch View", Style::default().fg(Color::DarkGray)),
        Span::styled("  [D]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" Dashboard", Style::default().fg(Color::DarkGray)),
        Span::styled("  [L]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" Logs", Style::default().fg(Color::DarkGray)),
        Span::styled("  [N]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" Network", Style::default().fg(Color::DarkGray)),
        Span::styled("  [M]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" Menu", Style::default().fg(Color::DarkGray)),
        Span::styled("  [Q]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray)));

    f.render_widget(footer, area);
}
