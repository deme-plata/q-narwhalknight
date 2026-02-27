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
pub fn draw_dashboard(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    // v8.5.5: Dynamic thread panel height — wraps to multiple rows for 192/384+ threads
    let thread_count = app.state.as_ref().map(|s| s.num_threads).unwrap_or(0);
    let thread_panel_width = (area.width / 2).saturating_sub(4) as usize;
    let dots_per_row = if thread_panel_width > 2 { thread_panel_width / 2 } else { 1 };
    let thread_rows = if thread_count > 0 { (thread_count + dots_per_row - 1) / dots_per_row } else { 1 };
    let thread_panel_height = (thread_rows as u16 + 3).max(5).min(14);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),                    // Hashrate sparkline
            Constraint::Length(5),                    // Physics metrics row
            Constraint::Length(thread_panel_height),  // Thread dots + block info (dynamic)
            Constraint::Length(3),                    // Connection status bar
            Constraint::Min(3),                       // Mini-log
        ])
        .split(area);

    draw_hashrate_sparkline(f, chunks[0], app);
    draw_physics_metrics(f, chunks[1], app);
    draw_thread_and_block_info(f, chunks[2], app);
    draw_connection_bar(f, chunks[3], app);
    draw_mini_log(f, chunks[4], app);
}

#[cfg(feature = "tui")]
fn draw_hashrate_sparkline(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let history: Vec<u64> = app.hashrate_history.iter()
        .map(|&rate| (rate * 1000.0) as u64)
        .collect();

    let current_khs = app.current_hashrate_khs();
    let current_mhs = current_khs / 1000.0;
    let peak_mhs = app.peak_hashrate_khs / 1000.0;

    // Total hashes computed
    let total_hashes = app.state.as_ref()
        .map(|s| s.hash_counter.load(Ordering::Relaxed))
        .unwrap_or(0);

    let title = format!(
        " {} {:.2} MH/s  Peak {:.2} MH/s  Total {} ",
        "\u{26A1}", // ⚡
        current_mhs, peak_mhs,
        format_hash_count(total_hashes),
    );

    let max_val = history.iter().max().copied().unwrap_or(1).max(1);

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(title),
        )
        .data(&history)
        .style(Style::default().fg(Color::Green))
        .max(max_val);

    f.render_widget(sparkline, area);
}

/// v8.5.5: Physics-inspired performance metrics row
#[cfg(feature = "tui")]
fn draw_physics_metrics(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    let uptime_secs = app.start_time.elapsed().as_secs();
    let current_khs = app.current_hashrate_khs();
    let total_hashes = app.state.as_ref()
        .map(|s| s.hash_counter.load(Ordering::Relaxed))
        .unwrap_or(0);
    let solutions = app.state.as_ref()
        .map(|s| s.solutions_found.load(Ordering::Relaxed))
        .unwrap_or(0);
    let blocks_mined = app.state.as_ref()
        .map(|s| s.blocks_mined.load(Ordering::Relaxed))
        .unwrap_or(0);
    let threads = app.state.as_ref().map(|s| s.num_threads).unwrap_or(0);
    let active = app.state.as_ref().map(|s| s.active_thread_count()).unwrap_or(0);

    // ── Card 1: Computational Entropy ──
    // Shannon entropy of thread states: measures how "spread out" the work is
    // H = -Sum(p_i * ln(p_i))  — max entropy = all threads in different states = diverse
    let entropy = compute_thread_entropy(app);
    let max_entropy = (threads as f64).ln().max(0.001);
    let entropy_pct = (entropy / max_entropy * 100.0).min(100.0);
    let entropy_bar = mini_bar(entropy_pct, 8);

    let card1 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{03A8} ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), // Ψ
            Span::styled("Entropy ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{:.1}%", entropy_pct), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(&entropy_bar, Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled(format!("  H={:.3} nat", entropy), Style::default().fg(Color::DarkGray)),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card1, cols[0]);

    // ── Card 2: Hashrate per Thread (Efficiency) ──
    let per_thread_khs = if active > 0 { current_khs / active as f64 } else { 0.0 };
    // VDF iterations per hash = 101 sequential BLAKE3 rounds
    let vdf_ops_per_sec = current_khs * 1000.0 * 101.0;
    let card2 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{03B7} ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), // η
            Span::styled("Efficiency", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {:.1} KH/t", per_thread_khs), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {} VDF/s", format_si(vdf_ops_per_sec)), Style::default().fg(Color::DarkGray)),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card2, cols[1]);

    // ── Card 3: Mining Yield & Difficulty ──
    // Hashes per solution = computational "proof of work" difficulty analogue
    let hashes_per_sol = if solutions > 0 { total_hashes as f64 / solutions as f64 } else { 0.0 };
    // Expected time to solution at current rate
    let est_secs = if current_khs > 0.0 && hashes_per_sol > 0.0 {
        hashes_per_sol / (current_khs * 1000.0)
    } else {
        0.0
    };
    let card3 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{0394} ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), // Δ
            Span::styled("Difficulty", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}/sol", format_hash_count(hashes_per_sol as u64)), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(
                if est_secs > 0.0 { format!("  ETA ~{}", format_duration(est_secs as u64)) }
                else { "  ETA --".to_string() },
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card3, cols[2]);

    // ── Card 4: Thermodynamic Summary ──
    // Landauer limit: minimum energy per bit erasure = kT ln(2)
    // At 300K: 2.85×10^-21 J per bit = 0.0178 eV
    // Each BLAKE3 round operates on 512-bit state → 512 bit-ops minimum
    // 101 VDF rounds × 512 bits = 51,712 bit-ops per hash
    let landauer_j_per_bit: f64 = 1.38e-23 * 300.0 * 0.693; // kT ln(2) at 300K
    let bits_per_hash: f64 = 101.0 * 512.0; // VDF rounds × state bits
    let landauer_energy_per_hash = landauer_j_per_bit * bits_per_hash;
    let landauer_power_w = landauer_energy_per_hash * current_khs * 1000.0;
    // Real power estimate: ~5W per thread typical x86 mining
    let est_power_w = active as f64 * 5.0;
    let carnot_ratio = if est_power_w > 0.0 { landauer_power_w / est_power_w } else { 0.0 };

    let card4 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{03A9} ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)), // Ω
            Span::styled("Thermo", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(format!("  ~{:.0}W est.", est_power_w), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("  {:.1e}W Landauer", landauer_power_w),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card4, cols[3]);
}

#[cfg(feature = "tui")]
fn draw_thread_and_block_info(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let halves = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    draw_thread_dots(f, halves[0], app);
    draw_block_info(f, halves[1], app);
}

#[cfg(feature = "tui")]
fn draw_thread_dots(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    if let Some(ref state) = app.state {
        let active = state.active_thread_count();
        let total = state.num_threads;
        let errored = state.errored_thread_count();

        // v8.5.5: Wrap thread dots to multiple rows for 192/384+ threads
        let inner_width = area.width.saturating_sub(4) as usize;
        let dots_per_row = if inner_width > 2 { inner_width / 2 } else { 1 };

        let mut all_dots: Vec<(&str, Color)> = Vec::new();
        for ts in &state.thread_states {
            let status = ts.get_status();
            let (symbol, color) = match &status {
                crate::shared_state::ThreadStatus::Mining { .. } => ("\u{25CF}", Color::Green),    // ●
                crate::shared_state::ThreadStatus::FetchingChallenge => ("\u{25CC}", Color::Yellow), // ◌
                crate::shared_state::ThreadStatus::WaitingForSync { .. } => ("\u{25CE}", Color::Cyan), // ◎
                crate::shared_state::ThreadStatus::Starting => ("\u{25CB}", Color::DarkGray),       // ○
                crate::shared_state::ThreadStatus::Error { .. } => ("\u{25CF}", Color::Red),        // ●
                crate::shared_state::ThreadStatus::Stopped => ("\u{25CB}", Color::DarkGray),        // ○
            };
            all_dots.push((symbol, color));
        }

        let mut lines: Vec<Line> = Vec::new();

        // Header with utilization percentage
        let util_pct = if total > 0 { (active as f64 / total as f64 * 100.0) as u32 } else { 0 };
        let mut header_spans = vec![
            Span::styled(
                format!("  {}/{} ", active, total),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{}% ", util_pct),
                Style::default().fg(if util_pct > 90 { Color::Green } else if util_pct > 50 { Color::Yellow } else { Color::Red }),
            ),
        ];
        if errored > 0 {
            header_spans.push(Span::styled(
                format!("{}err", errored),
                Style::default().fg(Color::Red),
            ));
        }
        lines.push(Line::from(header_spans));

        // Build rows of dots, wrapping when dots_per_row is exceeded
        for chunk in all_dots.chunks(dots_per_row) {
            let mut row_spans: Vec<Span> = vec![Span::raw(" ")];
            for (symbol, color) in chunk {
                row_spans.push(Span::styled(format!("{} ", symbol), Style::default().fg(*color)));
            }
            lines.push(Line::from(row_spans));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Threads ");

        f.render_widget(Paragraph::new(lines).block(block), area);
    } else {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Threads ");
        f.render_widget(Paragraph::new("  Initializing...").block(block), area);
    }
}

#[cfg(feature = "tui")]
fn draw_block_info(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let solutions = app.state.as_ref()
        .map(|s| s.solutions_found.load(Ordering::Relaxed))
        .unwrap_or(0);
    let blocks_mined = app.state.as_ref()
        .map(|s| s.blocks_mined.load(Ordering::Relaxed))
        .unwrap_or(0);
    let uptime = app.start_time.elapsed().as_secs();

    // Solutions per hour
    let sol_per_hr = if uptime > 0 { solutions as f64 / (uptime as f64 / 3600.0) } else { 0.0 };

    let mut text = vec![
        Line::from(vec![
            Span::styled("  Block   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("#{}", app.current_block_height),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Reward  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.4} QUG", app.current_block_reward),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Solved  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", solutions),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ({:.1}/hr)", sol_per_hr),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    // Uptime
    text.push(Line::from(vec![
        Span::styled("  Uptime  ", Style::default().fg(Color::Gray)),
        Span::styled(
            format_duration(uptime),
            Style::default().fg(Color::Cyan),
        ),
    ]));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" Block Info ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_connection_bar(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (sse_status, sse_color) = if let Some(ref state) = app.state {
        if state.sse_connected.load(Ordering::Relaxed) {
            ("\u{25CF}", Color::Green) // ●
        } else {
            ("\u{25CB}", Color::Red) // ○
        }
    } else {
        ("\u{25CB}", Color::DarkGray)
    };

    let (ml_status, ml_color) = if let Some(ref state) = app.state {
        if state.miner_link_connected.load(Ordering::Relaxed) {
            ("\u{25CF}", Color::Green)
        } else {
            ("\u{25CB}", Color::DarkGray)
        }
    } else {
        ("\u{25CB}", Color::DarkGray)
    };

    let latency_ms = if let Some(ref state) = app.state {
        state.last_challenge_latency_us.load(Ordering::Relaxed) / 1000
    } else {
        0
    };

    let server_display = if let Some(ref state) = app.state {
        state.server_url.replace("https://", "").replace("http://", "")
    } else {
        "...".to_string()
    };

    let latency_str = if latency_ms > 0 {
        format!("{}ms", latency_ms)
    } else {
        "...".to_string()
    };

    // Bandwidth limit display
    let bw_display = if let Some(ref state) = app.state {
        let throttle = *state.throttle_mode.read();
        match throttle {
            crate::shared_state::MinerThrottleMode::Off => String::new(),
            _ => format!("  \u{2502}  Throttle {}", throttle.label()),
        }
    } else {
        String::new()
    };

    let line = Line::from(vec![
        Span::raw("  "),
        Span::styled(&server_display, Style::default().fg(Color::Cyan)),
        Span::raw(" "),
        Span::styled(&latency_str, Style::default().fg(if latency_ms > 1000 { Color::Red } else { Color::Green })),
        Span::raw("  \u{2502}  SSE "),
        Span::styled(sse_status, Style::default().fg(sse_color)),
        Span::raw("  \u{2502}  Link "),
        Span::styled(ml_status, Style::default().fg(ml_color)),
        Span::styled(&bw_display, Style::default().fg(Color::DarkGray)),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    f.render_widget(Paragraph::new(line).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_mini_log(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let height = area.height.saturating_sub(2) as usize;
    let start = if app.logs.len() > height {
        app.logs.len() - height
    } else {
        0
    };

    let lines: Vec<Line> = app.logs.iter()
        .skip(start)
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

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Recent Activity ");

    f.render_widget(Paragraph::new(lines).block(block), area);
}

// ═══════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
use super::super::tui_app::LogLevel;

/// Compute Shannon entropy of thread status distribution
/// H = -Sum(p_i * ln(p_i))  where p_i = fraction of threads in state i
#[cfg(feature = "tui")]
fn compute_thread_entropy(app: &MinerTuiApp) -> f64 {
    let state = match &app.state {
        Some(s) => s,
        None => return 0.0,
    };
    let total = state.num_threads as f64;
    if total <= 1.0 { return 0.0; }

    // Count threads in each status category
    let mut counts = [0u32; 6]; // Mining, Fetching, Waiting, Starting, Error, Stopped
    for ts in &state.thread_states {
        let idx = match ts.get_status() {
            crate::shared_state::ThreadStatus::Mining { .. } => 0,
            crate::shared_state::ThreadStatus::FetchingChallenge => 1,
            crate::shared_state::ThreadStatus::WaitingForSync { .. } => 2,
            crate::shared_state::ThreadStatus::Starting => 3,
            crate::shared_state::ThreadStatus::Error { .. } => 4,
            crate::shared_state::ThreadStatus::Stopped => 5,
        };
        counts[idx] += 1;
    }

    let mut h = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Format large hash counts with SI suffixes
#[cfg(feature = "tui")]
fn format_hash_count(n: u64) -> String {
    if n >= 1_000_000_000_000 { format!("{:.2}T", n as f64 / 1e12) }
    else if n >= 1_000_000_000 { format!("{:.2}G", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { format!("{}", n) }
}

/// Format SI with decimal prefix for any f64
#[cfg(feature = "tui")]
fn format_si(n: f64) -> String {
    if n >= 1e12 { format!("{:.1}T", n / 1e12) }
    else if n >= 1e9 { format!("{:.1}G", n / 1e9) }
    else if n >= 1e6 { format!("{:.1}M", n / 1e6) }
    else if n >= 1e3 { format!("{:.1}K", n / 1e3) }
    else { format!("{:.0}", n) }
}

/// Format seconds into human-readable duration
#[cfg(feature = "tui")]
fn format_duration(secs: u64) -> String {
    if secs < 60 { format!("{}s", secs) }
    else if secs < 3600 { format!("{}m {}s", secs / 60, secs % 60) }
    else if secs < 86400 { format!("{}h {}m", secs / 3600, (secs % 3600) / 60) }
    else { format!("{}d {}h", secs / 86400, (secs % 86400) / 3600) }
}

/// Mini ASCII progress bar
#[cfg(feature = "tui")]
fn mini_bar(pct: f64, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("\u{2595}{}{}\u{258F}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
    // ▕████░░░░▏
}
