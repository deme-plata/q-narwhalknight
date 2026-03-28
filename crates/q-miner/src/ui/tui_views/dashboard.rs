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
    // v9.0.4: Show Starship sync panel when node is syncing
    if app.sync_info.is_some() {
        draw_syncing_dashboard(f, area, app);
        return;
    }

    // v9.9.0: Determine if update banner should show
    let has_update = app.update_version.is_some();
    let update_banner_height: u16 = if has_update { 1 } else { 0 };

    // v8.5.5: Dynamic thread panel height — wraps to multiple rows for 192/384+ threads
    let thread_count = app.state.as_ref().map(|s| s.num_threads).unwrap_or(0);
    let thread_panel_width = (area.width / 2).saturating_sub(4) as usize;
    let dots_per_row = if thread_panel_width > 2 { thread_panel_width / 2 } else { 1 };
    let thread_rows = if thread_count > 0 { (thread_count + dots_per_row - 1) / dots_per_row } else { 1 };
    let thread_panel_height = (thread_rows as u16 + 3).max(5).min(14);

    let mut constraints = Vec::new();
    if has_update {
        constraints.push(Constraint::Length(update_banner_height));
    }
    constraints.extend_from_slice(&[
        Constraint::Length(5),                    // Hashrate sparkline
        Constraint::Length(5),                    // Compute Power Layer cards
        Constraint::Length(5),                    // Physics metrics row
        Constraint::Length(thread_panel_height),  // Thread dots + block info (dynamic)
        Constraint::Length(3),                    // Connection status bar
        Constraint::Min(3),                       // Mini-log
    ]);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let offset = if has_update { 1 } else { 0 };

    if has_update {
        draw_update_banner(f, chunks[0], app);
    }

    draw_hashrate_sparkline(f, chunks[offset], app);
    draw_compute_power_cards(f, chunks[offset + 1], app);
    draw_physics_metrics(f, chunks[offset + 2], app);
    draw_thread_and_block_info(f, chunks[offset + 3], app);
    draw_connection_bar(f, chunks[offset + 4], app);
    draw_mini_log(f, chunks[offset + 5], app);
}

/// v9.9.0: Update banner — shows download progress or "press [U] to update"
#[cfg(feature = "tui")]
fn draw_update_banner(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let version = app.update_version.as_deref().unwrap_or("?");

    let (text, style) = if let Some(ref err) = app.update_error {
        if err == "__APPLY__" {
            (format!(" Applying v{}... ", version), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        } else {
            (format!(" Update v{} error: {} ", version, err), Style::default().fg(Color::Red))
        }
    } else if app.update_ready {
        (format!(" Update v{} ready — press [U] to apply and restart ", version), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
    } else {
        let (down, total) = app.update_progress;
        if total > 0 {
            let pct = (down as f64 / total as f64 * 100.0).min(100.0);
            let down_mb = down as f64 / 1_048_576.0;
            let total_mb = total as f64 / 1_048_576.0;
            (format!(" Downloading v{}: {:.1}/{:.1} MB ({:.0}%) ", version, down_mb, total_mb, pct), Style::default().fg(Color::Yellow))
        } else {
            (format!(" Downloading v{}... ", version), Style::default().fg(Color::Yellow))
        }
    };

    let banner = Paragraph::new(Line::from(vec![Span::styled(text, style)]));
    f.render_widget(banner, area);
}

/// v9.0.4: Full-screen Starship sync dashboard — shows when node is catching up
#[cfg(feature = "tui")]
fn draw_syncing_dashboard(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),   // Starship flight panel
            Constraint::Length(5),   // Phase timeline + stats
            Constraint::Length(3),   // Connection bar
            Constraint::Min(3),      // Mini-log
        ])
        .split(area);

    draw_starship_panel(f, chunks[0], app);
    draw_sync_stats(f, chunks[1], app);
    draw_connection_bar(f, chunks[2], app);
    draw_mini_log(f, chunks[3], app);
}

/// Starship flight computer display — the hero panel during sync
#[cfg(feature = "tui")]
fn draw_starship_panel(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let info = match &app.sync_info {
        Some(i) => i,
        None => return,
    };

    let pct = info.sync_progress.min(100.0).max(0.0);
    let bar_width = area.width.saturating_sub(6) as usize;
    let filled = ((pct as f64 / 100.0) * bar_width as f64).round() as usize;
    let empty = bar_width.saturating_sub(filled);

    // Phase emoji + color
    let (phase_icon, phase_color) = match info.phase.as_str() {
        "Prelaunch" => ("\u{1F4E1}", Color::DarkGray),  // 📡
        "Ignition" => ("\u{1F525}", Color::Red),          // 🔥
        "SuperHeavy" => ("\u{1F680}", Color::Cyan),       // 🚀
        "HotStaging" => ("\u{2604}", Color::Yellow),      // ☄
        "StarshipCruise" => ("\u{1F6F8}", Color::Magenta),// 🛸
        "StationKeeping" => ("\u{1F30D}", Color::Green),  // 🌍
        _ => ("\u{1F680}", Color::Cyan),                   // 🚀
    };

    // ETA display
    let eta_str = if info.eta_secs > 0 {
        format!("ETA {}", format_duration(info.eta_secs))
    } else if info.sync_speed_bps > 0.0 {
        "ETA calculating...".to_string()
    } else {
        "ETA --".to_string()
    };

    // Speed display
    let speed_str = if info.sync_speed_bps > 0.0 {
        format!("{:.0} blk/s", info.sync_speed_bps)
    } else {
        "-- blk/s".to_string()
    };

    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!("  {} STARSHIP FLIGHT COMPUTER", phase_icon),
                Style::default().fg(phase_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  Phase: {}", info.phase),
                Style::default().fg(phase_color),
            ),
        ]),
        Line::from(""),
        // Progress bar
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "\u{2595}",  // ▕
                Style::default().fg(phase_color),
            ),
            Span::styled(
                "\u{2588}".repeat(filled),
                Style::default().fg(phase_color),
            ),
            Span::styled(
                "\u{2591}".repeat(empty),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                "\u{258F}",  // ▏
                Style::default().fg(phase_color),
            ),
            Span::styled(
                format!(" {:.1}%", pct),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        // Height + blocks behind
        Line::from(vec![
            Span::styled("  Height  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("#{}", format_with_commas(info.local_height)),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" / #{}", format_with_commas(info.network_height)),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("  ({} behind)", format_with_commas(info.blocks_behind)),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        // Speed + ETA
        Line::from(vec![
            Span::styled("  Speed   ", Style::default().fg(Color::Gray)),
            Span::styled(
                &speed_str,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  \u{2502}  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                &eta_str,
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
        ]),
        // Peers + mission time
        Line::from(vec![
            Span::styled("  Peers   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", info.peer_count),
                Style::default().fg(if info.peer_count > 0 { Color::Green } else { Color::Red }).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  \u{2502}  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("Mission T+{}", format_duration(info.mission_elapsed_secs)),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    let border_color = if pct > 90.0 { Color::Green } else if pct > 50.0 { Color::Yellow } else { phase_color };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            " \u{2728} Node Syncing \u{2014} Mining paused ",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ));

    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// Phase timeline and sync statistics
#[cfg(feature = "tui")]
fn draw_sync_stats(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let info = match &app.sync_info {
        Some(i) => i,
        None => return,
    };

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Phase timeline
    let phases = ["Prelaunch", "Ignition", "SuperHeavy", "HotStaging", "StarshipCruise", "StationKeeping"];
    let phase_icons = ["\u{1F4E1}", "\u{1F525}", "\u{1F680}", "\u{2604}\u{FE0F}", "\u{1F6F8}", "\u{1F30D}"];
    let current_idx = phases.iter().position(|p| *p == info.phase.as_str()).unwrap_or(2);

    let mut phase_spans: Vec<Span> = vec![Span::raw("  ")];
    for (i, (phase, icon)) in phases.iter().zip(phase_icons.iter()).enumerate() {
        let (color, style) = if i < current_idx {
            (Color::Green, Modifier::DIM)       // completed
        } else if i == current_idx {
            (Color::Cyan, Modifier::BOLD)        // active
        } else {
            (Color::DarkGray, Modifier::DIM)     // future
        };
        phase_spans.push(Span::styled(
            format!("{} ", icon),
            Style::default().fg(color).add_modifier(style),
        ));
        if i < phases.len() - 1 {
            let arrow_color = if i < current_idx { Color::Green } else { Color::DarkGray };
            phase_spans.push(Span::styled("\u{2192} ", Style::default().fg(arrow_color)));
        }
    }

    let timeline = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("  Phase Timeline", Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(phase_spans),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(timeline, cols[0]);

    // Orbit status
    let orbit_icon = if info.orbit_stable { "\u{2705}" } else { "\u{1F504}" }; // ✅ or 🔄
    let orbit_text = if info.orbit_stable { "Stable" } else { "Acquiring..." };
    let phase_time = format_duration(info.phase_duration_secs);

    let status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("  Orbit  ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{} {}", orbit_icon, orbit_text), Style::default().fg(if info.orbit_stable { Color::Green } else { Color::Yellow })),
        ]),
        Line::from(vec![
            Span::styled("  Phase  ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{} for {}", info.phase, phase_time), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Status ", Style::default().fg(Color::Gray)),
            Span::styled("Mining auto-starts on sync", Style::default().fg(Color::DarkGray)),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(status, cols[1]);
}

/// Format number with commas: 7410000 → "7,410,000"
#[cfg(feature = "tui")]
fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

#[cfg(feature = "tui")]
fn draw_hashrate_sparkline(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let history: Vec<u64> = app.hashrate_history.iter()
        .map(|&rate| (rate * 1000.0) as u64)
        .collect();

    let current_khs = app.current_hashrate_khs();
    // v10.2.0: Combined CPU+GPU hashrate
    let combined_khs = current_khs + app.gpu_hashrate_khs;
    let combined_mhs = combined_khs / 1000.0;
    let peak_mhs = app.peak_hashrate_khs / 1000.0;

    // Total hashes computed
    let total_hashes = app.state.as_ref()
        .map(|s| s.hash_counter.load(Ordering::Relaxed))
        .unwrap_or(0);

    let mode_badge = if app.gpu_active { "[CPU+GPU]" } else { "" };

    let title = format!(
        " {} {:.2} MH/s {} Peak {:.2} MH/s  Total {} ",
        "\u{26A1}", // ⚡
        combined_mhs, mode_badge, peak_mhs,
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

/// v9.1.0: Compute Power Network Layer — 4 cards showing SIMD, network hashrate, security, peers
#[cfg(feature = "tui")]
fn draw_compute_power_cards(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    let current_khs = app.current_hashrate_khs();

    // ── Card 1: SIMD Tier / Hybrid Mode ──
    let (card1_title, card1_main, card1_sub, card1_color) = if app.gpu_active {
        let gpu_khs = app.gpu_hashrate_khs;
        let gpu_str = if gpu_khs >= 1000.0 { format!("{:.1} MH/s", gpu_khs / 1000.0) }
            else if gpu_khs > 0.0 { format!("{:.1} kH/s", gpu_khs) }
            else { "warming up".to_string() };
        // v10.1.7: Show GPU name in card if available
        let gpu_label = if let Some(dev) = app.gpu_devices.first() {
            // Truncate long names to fit card width
            let name = &dev.name;
            if name.len() > 18 { format!("{}...", &name[..15]) } else { name.clone() }
        } else if !app.gpu_device_name.is_empty() {
            let n = &app.gpu_device_name;
            if n.len() > 18 { format!("{}...", &n[..15]) } else { n.clone() }
        } else {
            "CPU+GPU".to_string()
        };
        ("Hybrid", gpu_label, gpu_str, Color::Magenta)
    } else {
        let batch_str = format!("{}x batch", app.simd_batch_size);
        ("SIMD", app.simd_tier.clone(), batch_str, Color::Cyan)
    };
    let card1 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{2301} ", Style::default().fg(card1_color).add_modifier(Modifier::BOLD)),
            Span::styled(card1_title, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}", card1_main), Style::default().fg(card1_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}", card1_sub), Style::default().fg(Color::DarkGray)),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(card1_color)));
    f.render_widget(card1, cols[0]);

    // ── Card 2: Network Compute Power (from P2P announcements) ──
    let net_hr = app.network_total_hashrate_hs;
    let hr_str = if net_hr >= 1e9 { format!("{:.2} GH/s", net_hr / 1e9) }
        else if net_hr >= 1e6 { format!("{:.2} MH/s", net_hr / 1e6) }
        else if net_hr >= 1e3 { format!("{:.1} kH/s", net_hr / 1e3) }
        else if net_hr > 0.0 { format!("{:.0} H/s", net_hr) }
        else { "--".to_string() };

    let card2 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{26A1} ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), // ⚡
            Span::styled("Network", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}", hr_str), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("  {} peers", app.network_compute_peers),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card2, cols[1]);

    // ── Card 3: Live Security Bits ──
    let sec_bits = app.live_security_bits;
    let sec_color = if sec_bits >= 128.0 { Color::Green }
        else if sec_bits >= 80.0 { Color::Yellow }
        else if sec_bits > 0.0 { Color::Red }
        else { Color::DarkGray };
    let tier = if sec_bits >= 256.0 { "FORTRESS" }
        else if sec_bits >= 192.0 { "FORTIFIED" }
        else if sec_bits >= 128.0 { "STRONG" }
        else if sec_bits >= 80.0 { "MODERATE" }
        else if sec_bits > 0.0 { "EMERGING" }
        else { "--" };

    let card3 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{1F6E1} ", Style::default().fg(sec_color).add_modifier(Modifier::BOLD)), // 🛡
            Span::styled("Security", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(
                if sec_bits > 0.0 { format!("  {:.0}-bit", sec_bits) } else { "  --".to_string() },
                Style::default().fg(sec_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}", tier), Style::default().fg(Color::DarkGray)),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card3, cols[2]);

    // ── Card 4: Compute Tunnel (local contribution vs network) ──
    let local_hs = current_khs * 1000.0; // kH/s → H/s
    let contribution_pct = if net_hr > 0.0 { (local_hs / net_hr * 100.0).min(100.0) } else { 0.0 };
    let contribution_bar = mini_bar(contribution_pct, 8);

    let card4 = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" \u{1F310} ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), // 🌐
            Span::styled("Tunnel", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::styled(
                if contribution_pct > 0.0 { format!("  {:.1}%", contribution_pct) } else { "  --".to_string() },
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(&contribution_bar, Style::default().fg(Color::Magenta)),
        ]),
    ]).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    f.render_widget(card4, cols[3]);
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
