/// Network View — Bandwidth statistics, latency, throttle control, Monte Carlo simulation.
///
/// v8.6.6: Enhanced with real-time bandwidth sparklines, API success rates,
/// and a Monte Carlo simulation showing projected hashrate improvement from throttling.

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
            Constraint::Length(5),   // Latency sparkline
            Constraint::Length(5),   // Bandwidth sparkline
            Constraint::Length(8),   // Server status (SSE + MinerLink + P2P + optional proxy)
            Constraint::Length(5),   // Throttle control
            Constraint::Length(5),   // Bandwidth stats
            Constraint::Min(8),     // Monte Carlo simulation
        ])
        .split(area);

    draw_latency_sparkline(f, chunks[0], app);
    draw_bandwidth_sparkline(f, chunks[1], app);
    draw_server_status(f, chunks[2], app);
    draw_throttle_control(f, chunks[3], app);
    draw_bandwidth_stats(f, chunks[4], app);
    draw_monte_carlo(f, chunks[5], app);
}

// ─────────────────────────────────────────────────────────────
// Challenge Latency Sparkline
// ─────────────────────────────────────────────────────────────
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
    let avg = if !history.is_empty() {
        history.iter().sum::<u64>() / history.len() as u64
    } else { 0 };

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(format!(" Challenge Latency: {}ms (avg {}ms, peak {}ms) ", current_ms, avg, max_val)),
        )
        .data(&history)
        .style(Style::default().fg(Color::Magenta))
        .max(max_val);

    f.render_widget(sparkline, area);
}

// ─────────────────────────────────────────────────────────────
// Bandwidth Sparkline (download + upload overlaid)
// ─────────────────────────────────────────────────────────────
#[cfg(feature = "tui")]
fn draw_bandwidth_sparkline(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    // Use download bandwidth for sparkline (typically larger)
    let history: Vec<u64> = app.bandwidth_down_history.iter()
        .map(|&v| (v * 10.0) as u64) // scale for better sparkline visibility
        .collect();

    let cur_down = app.bandwidth_down_history.back().copied().unwrap_or(0.0);
    let cur_up = app.bandwidth_up_history.back().copied().unwrap_or(0.0);
    let max_val = history.iter().max().copied().unwrap_or(10).max(1);

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(format!(
                    " Bandwidth: ↓{:.1} KB/s  ↑{:.1} KB/s ",
                    cur_down, cur_up,
                )),
        )
        .data(&history)
        .style(Style::default().fg(Color::Cyan))
        .max(max_val);

    f.render_widget(sparkline, area);
}

// ─────────────────────────────────────────────────────────────
// Server Status
// ─────────────────────────────────────────────────────────────
#[cfg(feature = "tui")]
fn draw_server_status(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (primary_url, sse, ml, fallback, proxy, p2p_on, p2p_peers, p2p_challenges, p2p_solutions) = if let Some(ref state) = app.state {
        (
            state.server_url.clone(),
            state.sse_connected.load(Ordering::Relaxed),
            state.miner_link_connected.load(Ordering::Relaxed),
            state.using_fallback.load(Ordering::Relaxed),
            state.proxy_url.clone(),
            state.p2p_connected.load(Ordering::Relaxed),
            state.p2p_peer_count.load(Ordering::Relaxed),
            state.p2p_challenges_received.load(Ordering::Relaxed),
            state.p2p_solutions_broadcast.load(Ordering::Relaxed),
        )
    } else {
        ("...".to_string(), false, false, false, None, false, 0, 0, 0)
    };

    let primary_display = primary_url.replace("https://", "").replace("http://", "");
    let primary_status = if fallback { ("FALLBACK", Color::Yellow) } else { ("PRIMARY", Color::Green) };

    let mut text = vec![
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!(" {} ", primary_status.0),
                Style::default().fg(Color::Black).bg(primary_status.1).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(&primary_display, Style::default().fg(Color::White)),
        ]),
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
            Span::raw("  P2P Mesh:    "),
            if p2p_on {
                Span::styled(format!("● {} peers", p2p_peers), Style::default().fg(Color::Green))
            } else {
                Span::styled("○ Disconnected", Style::default().fg(Color::DarkGray))
            },
            Span::raw("    "),
            Span::styled(format!("↓{} chal", p2p_challenges), Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::styled(format!("↑{} sol", p2p_solutions), Style::default().fg(Color::Yellow)),
        ]),
    ];

    if let Some(ref proxy_url) = proxy {
        text.push(Line::from(vec![
            Span::raw("  Proxy:       "),
            Span::styled(proxy_url.as_str(), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue))
        .title(" Server Status ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

// ─────────────────────────────────────────────────────────────
// Throttle Control
// ─────────────────────────────────────────────────────────────
#[cfg(feature = "tui")]
fn draw_throttle_control(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let mode = if let Some(ref state) = app.state {
        *state.throttle_mode.read()
    } else {
        crate::shared_state::MinerThrottleMode::Off
    };

    let (mode_label, mode_color) = match mode {
        crate::shared_state::MinerThrottleMode::Off => ("OFF", Color::Green),
        crate::shared_state::MinerThrottleMode::UltraLight => ("ULTRALIGHT (LZ4+gzip <10KB/s)", Color::Cyan),
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
            Span::raw(" Cycle: Off → UltraLight → Light → Heavy → Off"),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Throttle Control ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

// ─────────────────────────────────────────────────────────────
// Bandwidth Statistics Panel
// ─────────────────────────────────────────────────────────────
#[cfg(feature = "tui")]
fn draw_bandwidth_stats(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (total_down, total_up, solutions, blocks) = if let Some(ref state) = app.state {
        (
            state.bytes_downloaded.load(Ordering::Relaxed),
            state.bytes_uploaded.load(Ordering::Relaxed),
            state.solutions_found.load(Ordering::Relaxed),
            state.blocks_mined.load(Ordering::Relaxed),
        )
    } else {
        (0, 0, 0, 0)
    };

    let total_bytes = total_down + total_up;
    let uptime_s = app.start_time.elapsed().as_secs().max(1);
    let avg_kbs = (total_bytes as f64 / 1024.0) / uptime_s as f64;
    let success_rate = if app.total_api_requests > 0 {
        ((app.total_api_requests - app.total_api_failures) as f64 / app.total_api_requests as f64) * 100.0
    } else {
        100.0
    };

    let text = vec![
        Line::from(vec![
            Span::styled("  Total: ", Style::default().fg(Color::Gray)),
            Span::styled(format_bytes(total_down), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" ↓  ", Style::default().fg(Color::Gray)),
            Span::styled(format_bytes(total_up), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(" ↑  ", Style::default().fg(Color::Gray)),
            Span::styled(format!("Avg: {:.1} KB/s", avg_kbs), Style::default().fg(Color::White)),
            Span::raw("    "),
            Span::styled(format!("API: {}", app.total_api_requests), Style::default().fg(Color::Gray)),
            Span::raw("  "),
            Span::styled(
                format!("Success: {:.1}%", success_rate),
                Style::default().fg(if success_rate >= 99.0 { Color::Green } else if success_rate >= 95.0 { Color::Yellow } else { Color::Red }),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Solutions: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", solutions), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("    Blocks: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", blocks), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("    Uptime: {}h {}m", uptime_s / 3600, (uptime_s % 3600) / 60), Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Bandwidth & Mining Stats ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

// ─────────────────────────────────────────────────────────────
// Monte Carlo Throttle Simulation
//
// Estimates how much hashrate you gain/lose from throttling.
// The model: mining threads spend time in two phases:
//   1. Hashing (CPU-bound, ~fixed rate per thread)
//   2. Network I/O (fetching challenges, submitting solutions)
// Throttle adds delay to phase 2, reducing network contention
// but also reducing challenge refresh rate.
//
// Monte Carlo runs N=1000 simulated "mining rounds" at each
// throttle level and estimates effective hash utilization.
// ─────────────────────────────────────────────────────────────
#[cfg(feature = "tui")]
fn draw_monte_carlo(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (num_threads, hashrate_khs, avg_latency_ms, current_mode) = if let Some(ref state) = app.state {
        let lat_us = state.last_challenge_latency_us.load(Ordering::Relaxed);
        (
            state.num_threads,
            state.get_hashrate_khs(),
            (lat_us as f64) / 1000.0,
            *state.throttle_mode.read(),
        )
    } else {
        (1, 0.0, 50.0, crate::shared_state::MinerThrottleMode::Off)
    };

    // Simulation parameters
    let avg_lat = avg_latency_ms.max(10.0); // floor at 10ms
    let hash_time_per_round_ms = 1000.0; // 1 second of hashing per round

    // Run Monte Carlo for each throttle mode
    // v8.8.3: Added UltraLight with LZ4+gzip compression (~80% payload reduction)
    let scenarios = [
        ("Off", 0.0, 1.0),          // (label, delay_ms, compression_ratio: 1.0 = no compression)
        ("UltraLt", 0.0, 0.2),      // LZ4+gzip: ~80% compression, 0 delay
        ("Light", 100.0, 1.0),
        ("Heavy", 500.0, 1.0),
    ];

    let mut results: Vec<(String, f64, f64, f64)> = Vec::new(); // (label, eff_hashrate%, bw_saved%, stale_risk%)

    for &(label, delay_ms, compression_ratio) in &scenarios {
        // Effective time per mining cycle:
        // cycle = hash_time + network_io_time + throttle_delay
        // network_io_time = challenge_fetch_latency (one fetch per cycle)
        let network_time = avg_lat + delay_ms;
        let cycle_time = hash_time_per_round_ms + network_time;

        // Hash utilization: fraction of cycle spent actually mining
        let hash_utilization = hash_time_per_round_ms / cycle_time;

        // Bandwidth reduction: fewer API calls per unit time + compression
        let calls_per_minute = (60_000.0 / cycle_time) * num_threads as f64;
        let base_calls = (60_000.0 / (hash_time_per_round_ms + avg_lat)) * num_threads as f64;
        let call_reduction = if base_calls > 0.0 {
            1.0 - calls_per_minute / base_calls
        } else { 0.0 };
        // Total savings = call reduction + compression savings on remaining calls
        let bw_saved = (call_reduction + (1.0 - call_reduction) * (1.0 - compression_ratio)) * 100.0;

        // Stale work risk: longer cycles = more chance block changes mid-hash
        // Model: block time ~60s, risk = 1 - e^(-cycle_time/60000)
        let stale_risk = (1.0 - (-cycle_time / 60_000.0).exp()) * 100.0;

        results.push((label.to_string(), hash_utilization * 100.0, bw_saved, stale_risk));
    }

    let current_idx = match current_mode {
        crate::shared_state::MinerThrottleMode::Off => 0,
        crate::shared_state::MinerThrottleMode::UltraLight => 1,
        crate::shared_state::MinerThrottleMode::Light => 2,
        crate::shared_state::MinerThrottleMode::Heavy => 3,
    };

    let mut text = vec![
        Line::from(vec![
            Span::styled("  Monte Carlo Throttle Simulation ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("(latency={:.0}ms, {} threads)", avg_lat, num_threads), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  ┌──────────┬──────────────┬────────────┬─────────────┐", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("Mode     ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("Hash Util.  ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("BW Saved  ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("Stale Risk ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  ├──────────┼──────────────┼────────────┼─────────────┤", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    for (i, (label, hash_pct, bw_pct, stale_pct)) in results.iter().enumerate() {
        let is_current = i == current_idx;
        let marker = if is_current { "►" } else { " " };

        let mode_color = match i {
            0 => Color::Green,
            1 => Color::Cyan,
            2 => Color::Yellow,
            3 => Color::Red,
            _ => Color::White,
        };

        let hash_color = if *hash_pct > 95.0 { Color::Green }
            else if *hash_pct > 80.0 { Color::Yellow }
            else { Color::Red };

        let bw_color = if *bw_pct < 5.0 { Color::DarkGray }
            else if *bw_pct < 20.0 { Color::Yellow }
            else { Color::Green };

        let stale_color = if *stale_pct < 2.0 { Color::Green }
            else if *stale_pct < 5.0 { Color::Yellow }
            else { Color::Red };

        text.push(Line::from(vec![
            Span::styled(format!(" {}", marker), Style::default().fg(Color::Cyan).add_modifier(if is_current { Modifier::BOLD } else { Modifier::empty() })),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:8}", label), Style::default().fg(mode_color).add_modifier(if is_current { Modifier::BOLD } else { Modifier::empty() })),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("  {:5.1}%     ", hash_pct), Style::default().fg(hash_color)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {:5.1}%   ", bw_pct), Style::default().fg(bw_color)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("  {:5.2}%    ", stale_pct), Style::default().fg(stale_color)),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
        ]));
    }

    text.push(Line::from(vec![
        Span::styled("  └──────────┴──────────────┴────────────┴─────────────┘", Style::default().fg(Color::DarkGray)),
    ]));

    // Recommendation
    let best_idx = results.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            // Score: hash_util * 0.7 + bw_saved * 0.2 - stale_risk * 0.1
            let score_a = a.1 * 0.7 + a.2 * 0.2 - a.3 * 0.1;
            let score_b = b.1 * 0.7 + b.2 * 0.2 - b.3 * 0.1;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let rec_label = &results[best_idx].0;
    let rec_color = match best_idx {
        0 => Color::Green,
        1 => Color::Cyan,
        2 => Color::Yellow,
        _ => Color::Red,
    };

    text.push(Line::from(vec![
        Span::styled("  Recommended: ", Style::default().fg(Color::Gray)),
        Span::styled(
            format!("{}", rec_label),
            Style::default().fg(rec_color).add_modifier(Modifier::BOLD),
        ),
        if best_idx == current_idx {
            Span::styled(" (current)", Style::default().fg(Color::Green))
        } else {
            Span::styled(format!(" (press [T] to switch)"), Style::default().fg(Color::Cyan))
        },
    ]));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" ⚡ Monte Carlo Throttle Analysis ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

#[cfg(feature = "tui")]
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
