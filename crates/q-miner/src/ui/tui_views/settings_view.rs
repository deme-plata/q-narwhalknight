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
use std::sync::atomic::Ordering;

#[cfg(feature = "tui")]
pub fn draw_settings(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),  // Thread/intensity controls
            Constraint::Length(10), // Config info (includes proxy line)
            Constraint::Min(3),    // Hardware info
        ])
        .split(area);

    draw_controls(f, chunks[0], app);
    draw_config(f, chunks[1], app);
    draw_hardware(f, chunks[2], app);
}

#[cfg(feature = "tui")]
fn draw_controls(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (threads, intensity) = if let Some(ref state) = app.state {
        (
            state.target_threads.load(Ordering::Relaxed),
            state.target_intensity.load(Ordering::Relaxed),
        )
    } else {
        (0, 7)
    };

    let total_threads = if let Some(ref state) = app.state {
        state.num_threads
    } else {
        0
    };

    let text = vec![
        Line::from(vec![
            Span::styled("  Threads:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", threads),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("/{}", total_threads),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw("        "),
            Span::styled("[+]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Add  "),
            Span::styled("[-]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Remove"),
        ]),
        Line::from(vec![
            Span::styled("  Intensity: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", intensity),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled("/10", Style::default().fg(Color::DarkGray)),
            Span::raw("       "),
            Span::styled("[>]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Increase  "),
            Span::styled("[<]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" Decrease"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("Note:", Style::default().fg(Color::Yellow)),
            Span::raw(" Thread changes take effect on next challenge fetch"),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Controls ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_config(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (server, wallet, mode, miner_id, miner_name, proxy) = if let Some(ref state) = app.state {
        (
            state.server_url.clone(),
            state.wallet_address.clone(),
            state.mining_mode.clone(),
            state.miner_id.clone(),
            state.miner_name.clone(),
            state.proxy_url.clone(),
        )
    } else {
        ("...".into(), "...".into(), "...".into(), "...".into(), None, None)
    };

    let wallet_display = if wallet.len() > 20 {
        format!("{}...{}", &wallet[..10], &wallet[wallet.len()-10..])
    } else {
        wallet.clone()
    };

    let proxy_display = proxy.as_deref().unwrap_or("(direct)");

    let text = vec![
        Line::from(vec![
            Span::styled("  Server:     ", Style::default().fg(Color::Gray)),
            Span::styled(&server, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Wallet:     ", Style::default().fg(Color::Gray)),
            Span::styled(&wallet_display, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("  Mode:       ", Style::default().fg(Color::Gray)),
            Span::styled(&mode, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Miner ID:   ", Style::default().fg(Color::Gray)),
            Span::styled(&miner_id, Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  Name:       ", Style::default().fg(Color::Gray)),
            Span::styled(
                miner_name.as_deref().unwrap_or("(not set)"),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Proxy:      ", Style::default().fg(Color::Gray)),
            Span::styled(
                proxy_display,
                Style::default().fg(if proxy.is_some() { Color::Magenta } else { Color::DarkGray }),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Version:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("v{}", env!("CARGO_PKG_VERSION")),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue))
        .title(" Configuration ");

    f.render_widget(Paragraph::new(text).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_hardware(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let cores = num_cpus::get_physical();
    let threads = num_cpus::get();

    let mut text = vec![
        Line::from(vec![
            Span::styled("  CPU Cores:    ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", cores), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  CPU Threads:  ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", threads), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  SIMD:         ", Style::default().fg(Color::Gray)),
            Span::styled(&app.simd_tier, Style::default().fg(Color::Cyan)),
            Span::styled(format!(" ({}x batch)", app.simd_batch_size), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
    ];

    // GPU section
    if app.gpu_active {
        let gpu_khs = app.gpu_hashrate_khs;
        let gpu_hr_str = if gpu_khs >= 1000.0 { format!("{:.1} MH/s", gpu_khs / 1000.0) }
            else if gpu_khs > 0.0 { format!("{:.1} kH/s", gpu_khs) }
            else { "warming up...".to_string() };

        text.push(Line::from(vec![
            Span::styled("  GPU Mining:   ", Style::default().fg(Color::Gray)),
            Span::styled("ACTIVE", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  ({})", gpu_hr_str), Style::default().fg(Color::Magenta)),
        ]));

        if app.gpu_devices.is_empty() {
            // Fallback: just show device name
            if !app.gpu_device_name.is_empty() {
                text.push(Line::from(vec![
                    Span::styled("  Device:       ", Style::default().fg(Color::Gray)),
                    Span::styled(&app.gpu_device_name, Style::default().fg(Color::White)),
                ]));
            }
        } else {
            for dev in &app.gpu_devices {
                text.push(Line::from(vec![
                    Span::styled(format!("  GPU #{}:       ", dev.index), Style::default().fg(Color::Gray)),
                    Span::styled(&dev.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]));
                text.push(Line::from(vec![
                    Span::styled("    Vendor:     ", Style::default().fg(Color::DarkGray)),
                    Span::styled(&dev.vendor, Style::default().fg(Color::White)),
                ]));
                text.push(Line::from(vec![
                    Span::styled("    Memory:     ", Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{} MB", dev.global_memory_mb), Style::default().fg(Color::White)),
                ]));
                text.push(Line::from(vec![
                    Span::styled("    Compute:    ", Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{} CU @ {} MHz", dev.compute_units, dev.max_clock_mhz), Style::default().fg(Color::White)),
                ]));
                text.push(Line::from(vec![
                    Span::styled("    API:        ", Style::default().fg(Color::DarkGray)),
                    Span::styled(&dev.api, Style::default().fg(Color::Cyan)),
                ]));
            }
        }
    } else {
        text.push(Line::from(vec![
            Span::styled("  GPU Mining:   ", Style::default().fg(Color::Gray)),
            Span::styled("OFF", Style::default().fg(Color::DarkGray)),
            Span::styled("  (recompile with --features cuda-mining or opencl-mining)", Style::default().fg(Color::DarkGray)),
        ]));
    }

    let border_color = if app.gpu_active { Color::Magenta } else { Color::DarkGray };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(if app.gpu_active { " Hardware + GPU " } else { " Hardware " });

    f.render_widget(Paragraph::new(text).block(block), area);
}
