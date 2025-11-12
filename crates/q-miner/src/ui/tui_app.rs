/// Beautiful Terminal UI for Q-Miner
///
/// This module implements a real-time mining dashboard using ratatui with:
/// - Live hash rate graphs
/// - Per-GPU monitoring
/// - Block discovery notifications
/// - Interactive GPU configuration

#[cfg(feature = "tui")]
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span, Text},
    widgets::{
        Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, Gauge, List, ListItem,
        Paragraph, Sparkline, Tabs,
    },
    Frame, Terminal,
};

#[cfg(feature = "tui")]
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use crate::{DeviceStats, DeviceType, GlobalMiningStats, MiningEvent};
use anyhow::Result;
use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const HASH_RATE_HISTORY_SIZE: usize = 60; // 60 samples (1 minute at 1 sample/sec)

/// TUI Application State
pub struct TuiApp {
    /// Current tab index
    pub tab_index: usize,

    /// Hash rate history for sparkline
    pub hash_rate_history: VecDeque<f64>,

    /// Global mining statistics
    pub stats: GlobalMiningStats,

    /// Recent mining events
    pub events: VecDeque<MiningEvent>,

    /// Mining start time
    pub start_time: Instant,

    /// Selected GPU for detailed view
    pub selected_gpu: usize,

    /// TUI running state
    pub running: bool,

    /// Pause mining
    pub paused: bool,

    /// Show help overlay
    pub show_help: bool,

    /// GPU temperature history (per device)
    pub gpu_temp_history: Vec<VecDeque<f64>>,

    /// GPU power history (per device)
    pub gpu_power_history: Vec<VecDeque<f64>>,
}

impl TuiApp {
    pub fn new() -> Self {
        Self {
            tab_index: 0,
            hash_rate_history: VecDeque::with_capacity(HASH_RATE_HISTORY_SIZE),
            stats: GlobalMiningStats::default(),
            events: VecDeque::with_capacity(100),
            start_time: Instant::now(),
            selected_gpu: 0,
            running: true,
            paused: false,
            show_help: false,
            gpu_temp_history: vec![VecDeque::with_capacity(HASH_RATE_HISTORY_SIZE); 8],
            gpu_power_history: vec![VecDeque::with_capacity(HASH_RATE_HISTORY_SIZE); 8],
        }
    }

    pub fn update_stats(&mut self, stats: GlobalMiningStats) {
        // Update hash rate history
        self.hash_rate_history.push_back(stats.total_hash_rate);
        if self.hash_rate_history.len() > HASH_RATE_HISTORY_SIZE {
            self.hash_rate_history.pop_front();
        }

        // Update per-GPU temperature and power history
        for (idx, device) in stats.devices.iter().enumerate() {
            if idx < self.gpu_temp_history.len() {
                self.gpu_temp_history[idx].push_back(device.temperature);
                if self.gpu_temp_history[idx].len() > HASH_RATE_HISTORY_SIZE {
                    self.gpu_temp_history[idx].pop_front();
                }

                self.gpu_power_history[idx].push_back(device.power_usage);
                if self.gpu_power_history[idx].len() > HASH_RATE_HISTORY_SIZE {
                    self.gpu_power_history[idx].pop_front();
                }
            }
        }

        self.stats = stats;
    }

    pub fn add_event(&mut self, event: MiningEvent) {
        self.events.push_back(event);
        if self.events.len() > 100 {
            self.events.pop_front();
        }
    }

    pub fn next_tab(&mut self) {
        self.tab_index = (self.tab_index + 1) % 4;
    }

    pub fn previous_tab(&mut self) {
        if self.tab_index > 0 {
            self.tab_index -= 1;
        } else {
            self.tab_index = 3;
        }
    }

    pub fn next_gpu(&mut self) {
        if !self.stats.devices.is_empty() {
            self.selected_gpu = (self.selected_gpu + 1) % self.stats.devices.len();
        }
    }

    pub fn previous_gpu(&mut self) {
        if !self.stats.devices.is_empty() && self.selected_gpu > 0 {
            self.selected_gpu -= 1;
        } else if !self.stats.devices.is_empty() {
            self.selected_gpu = self.stats.devices.len() - 1;
        }
    }
}

#[cfg(feature = "tui")]
pub async fn run_tui(
    mut stats_rx: mpsc::UnboundedReceiver<GlobalMiningStats>,
    mut event_rx: mpsc::UnboundedReceiver<MiningEvent>,
) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = TuiApp::new();

    // Main TUI loop
    let tick_rate = Duration::from_millis(250); // 4 FPS
    let mut last_tick = Instant::now();

    while app.running {
        // Draw UI
        terminal.draw(|f| draw_ui(f, &app))?;

        // Handle input with timeout
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            app.running = false;
                        }
                        KeyCode::Char('p') => {
                            app.paused = !app.paused;
                        }
                        KeyCode::Char('h') | KeyCode::Char('?') => {
                            app.show_help = !app.show_help;
                        }
                        KeyCode::Tab | KeyCode::Right => {
                            app.next_tab();
                        }
                        KeyCode::Left => {
                            app.previous_tab();
                        }
                        KeyCode::Up => {
                            app.previous_gpu();
                        }
                        KeyCode::Down => {
                            app.next_gpu();
                        }
                        _ => {}
                    }
                }
            }
        }

        // Update stats if available
        while let Ok(stats) = stats_rx.try_recv() {
            app.update_stats(stats);
        }

        // Receive events
        while let Ok(event) = event_rx.try_recv() {
            app.add_event(event);
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

#[cfg(feature = "tui")]
fn draw_ui(f: &mut Frame, app: &TuiApp) {
    let size = f.area();

    // Create main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header/Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer/Controls
        ])
        .split(size);

    // Draw header with tabs
    draw_header(f, chunks[0], app);

    // Draw content based on selected tab
    match app.tab_index {
        0 => draw_overview_tab(f, chunks[1], app),
        1 => draw_gpu_details_tab(f, chunks[1], app),
        2 => draw_events_tab(f, chunks[1], app),
        3 => draw_settings_tab(f, chunks[1], app),
        _ => {}
    }

    // Draw footer
    draw_footer(f, chunks[2], app);

    // Draw help overlay if active
    if app.show_help {
        draw_help_overlay(f, size, app);
    }
}

#[cfg(feature = "tui")]
fn draw_header(f: &mut Frame, area: Rect, app: &TuiApp) {
    let tab_titles = vec!["Overview", "GPU Details", "Events", "Settings"];
    let tabs = Tabs::new(tab_titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title("⛏️  Q-Miner Dashboard"),
        )
        .select(app.tab_index)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

#[cfg(feature = "tui")]
fn draw_overview_tab(f: &mut Frame, area: Rect, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40), // Hash rate graph
            Constraint::Percentage(30), // GPU status
            Constraint::Percentage(30), // Stats
        ])
        .split(area);

    // Hash rate graph
    draw_hashrate_graph(f, chunks[0], app);

    // GPU status bars
    draw_gpu_status(f, chunks[1], app);

    // Mining statistics
    draw_mining_stats(f, chunks[2], app);
}

#[cfg(feature = "tui")]
fn draw_hashrate_graph(f: &mut Frame, area: Rect, app: &TuiApp) {
    let history: Vec<u64> = app
        .hash_rate_history
        .iter()
        .map(|&rate| (rate / 1_000_000.0) as u64) // Convert to MH/s
        .collect();

    let current_rate = app.stats.total_hash_rate / 1_000_000.0;
    let max_rate = history.iter().max().copied().unwrap_or(1);
    let avg_rate: f64 = if !history.is_empty() {
        history.iter().sum::<u64>() as f64 / history.len() as f64
    } else {
        0.0
    };

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(format!(
                    "Hashrate: {:.1} MH/s  │  Avg: {:.1} MH/s  │  Peak: {} MH/s",
                    current_rate, avg_rate, max_rate
                )),
        )
        .data(&history)
        .style(Style::default().fg(Color::Green))
        .max(max_rate);

    f.render_widget(sparkline, area);
}

#[cfg(feature = "tui")]
fn draw_gpu_status(f: &mut Frame, area: Rect, app: &TuiApp) {
    let mut gpu_widgets = Vec::new();

    for (idx, device) in app.stats.devices.iter().enumerate() {
        let device_name = match &device.device_type {
            DeviceType::CPU => "CPU".to_string(),
            DeviceType::CUDA(name) | DeviceType::OpenCL(name) | DeviceType::Vulkan(name) => {
                name.clone()
            }
        };

        let hash_rate_mhs = device.hash_rate / 1_000_000.0;
        let utilization_pct = (device.utilization * 100.0) as u16;

        let label = format!(
            "GPU {} ({})  {:.1} MH/s  {}°C  {:.0}W",
            idx, device_name, hash_rate_mhs, device.temperature as u32, device.power_usage
        );

        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::NONE))
            .gauge_style(
                Style::default()
                    .fg(temperature_color(device.temperature))
                    .bg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            )
            .percent(utilization_pct)
            .label(label);

        gpu_widgets.push(gauge);
    }

    // Create vertical layout for GPUs
    let gpu_count = gpu_widgets.len();
    if gpu_count > 0 {
        let constraints: Vec<Constraint> = (0..gpu_count)
            .map(|_| Constraint::Length(1))
            .collect();

        let gpu_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area.inner(ratatui::layout::Margin {
                horizontal: 1,
                vertical: 1,
            }));

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta))
            .title("GPU Status");

        f.render_widget(block, area);

        for (idx, gauge) in gpu_widgets.iter().enumerate() {
            if idx < gpu_chunks.len() {
                f.render_widget(gauge.clone(), gpu_chunks[idx]);
            }
        }
    }
}

#[cfg(feature = "tui")]
fn draw_mining_stats(f: &mut Frame, area: Rect, app: &TuiApp) {
    let uptime = app.start_time.elapsed();
    let uptime_str = format!(
        "{}h {}m {}s",
        uptime.as_secs() / 3600,
        (uptime.as_secs() % 3600) / 60,
        uptime.as_secs() % 60
    );

    let efficiency = app.stats.efficiency * 100.0;
    let accepted = app.stats.accepted_shares;
    let rejected = app.stats.rejected_shares;
    let total_power = app.stats.power_usage;

    let text = vec![
        Line::from(vec![
            Span::styled("Shares Accepted: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", accepted),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  │  "),
            Span::styled("Rejected: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", rejected),
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Efficiency: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2}%", efficiency),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  │  "),
            Span::styled("Power Usage: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.0}W", total_power),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Uptime: ", Style::default().fg(Color::Gray)),
            Span::styled(
                uptime_str,
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title("Mining Statistics"),
        )
        .alignment(Alignment::Left);

    f.render_widget(paragraph, area);
}

#[cfg(feature = "tui")]
fn draw_gpu_details_tab(f: &mut Frame, area: Rect, _app: &TuiApp) {
    let placeholder = Paragraph::new("GPU detailed view (coming soon)")
        .block(Block::default().borders(Borders::ALL).title("GPU Details"))
        .alignment(Alignment::Center);

    f.render_widget(placeholder, area);
}

#[cfg(feature = "tui")]
fn draw_events_tab(f: &mut Frame, area: Rect, app: &TuiApp) {
    let events: Vec<ListItem> = app
        .events
        .iter()
        .rev()
        .take(20)
        .map(|event| {
            let content = format_event(event);
            ListItem::new(content)
        })
        .collect();

    let list = List::new(events)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow))
                .title("Recent Events"),
        )
        .style(Style::default().fg(Color::White));

    f.render_widget(list, area);
}

#[cfg(feature = "tui")]
fn draw_settings_tab(f: &mut Frame, area: Rect, _app: &TuiApp) {
    let text = vec![
        Line::from("Settings"),
        Line::from(""),
        Line::from("(Use 'g' to toggle GPU settings)"),
    ];

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title("Settings"),
        )
        .alignment(Alignment::Center);

    f.render_widget(paragraph, area);
}

#[cfg(feature = "tui")]
fn draw_footer(f: &mut Frame, area: Rect, app: &TuiApp) {
    let status = if app.paused { "⏸️  PAUSED" } else { "⛏️  MINING" };
    let status_color = if app.paused { Color::Yellow } else { Color::Green };

    let footer_text = vec![
        Span::styled(status, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::raw("  │  "),
        Span::raw("[q] Quit  [p] Pause  [h] Help  [Tab] Next"),
    ];

    let footer = Paragraph::new(Line::from(footer_text))
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Left);

    f.render_widget(footer, area);
}

#[cfg(feature = "tui")]
fn draw_help_overlay(f: &mut Frame, area: Rect, _app: &TuiApp) {
    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Keyboard Shortcuts", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  q, Esc     ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit application"),
        ]),
        Line::from(vec![
            Span::styled("  p          ", Style::default().fg(Color::Yellow)),
            Span::raw("Pause/Resume mining"),
        ]),
        Line::from(vec![
            Span::styled("  h, ?       ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle this help"),
        ]),
        Line::from(vec![
            Span::styled("  Tab, →     ", Style::default().fg(Color::Yellow)),
            Span::raw("Next tab"),
        ]),
        Line::from(vec![
            Span::styled("  ←          ", Style::default().fg(Color::Yellow)),
            Span::raw("Previous tab"),
        ]),
        Line::from(vec![
            Span::styled("  ↑/↓        ", Style::default().fg(Color::Yellow)),
            Span::raw("Select GPU"),
        ]),
    ];

    let help_block = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title("Help"),
        )
        .alignment(Alignment::Left);

    // Center the help overlay
    let popup_area = centered_rect(50, 50, area);
    f.render_widget(ratatui::widgets::Clear, popup_area);
    f.render_widget(help_block, popup_area);
}

#[cfg(feature = "tui")]
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

#[cfg(feature = "tui")]
fn temperature_color(temp: f64) -> Color {
    if temp >= 85.0 {
        Color::Red
    } else if temp >= 75.0 {
        Color::Yellow
    } else if temp >= 65.0 {
        Color::Green
    } else {
        Color::Cyan
    }
}

#[cfg(feature = "tui")]
fn format_event(event: &MiningEvent) -> Line<'static> {
    match event {
        MiningEvent::NewWork(work) => Line::from(vec![
            Span::styled("[NEW]  ", Style::default().fg(Color::Blue)),
            Span::raw(format!("New work received: {}", work.job_id)),
        ]),
        MiningEvent::SolutionFound {
            device_id,
            hash_rate,
            nonce,
        } => Line::from(vec![
            Span::styled("[✓]    ", Style::default().fg(Color::Green)),
            Span::raw(format!(
                "Solution found on {}: nonce {} ({:.1} MH/s)",
                device_id,
                nonce,
                hash_rate / 1_000_000.0
            )),
        ]),
        MiningEvent::ShareAccepted {
            job_id,
            difficulty,
            reward,
        } => Line::from(vec![
            Span::styled("[ACC]  ", Style::default().fg(Color::Green)),
            Span::raw(format!(
                "Share accepted: {} (diff: {:.0}, reward: {:.2} QNK)",
                job_id, difficulty, reward
            )),
        ]),
        MiningEvent::ShareRejected { job_id, reason } => Line::from(vec![
            Span::styled("[REJ]  ", Style::default().fg(Color::Red)),
            Span::raw(format!("Share rejected: {} ({})", job_id, reason)),
        ]),
        MiningEvent::DeviceUpdate { device_id, stats } => Line::from(vec![
            Span::styled("[UPD]  ", Style::default().fg(Color::Yellow)),
            Span::raw(format!(
                "{}: {:.1} MH/s, {}°C, {:.0}W",
                device_id,
                stats.hash_rate / 1_000_000.0,
                stats.temperature as u32,
                stats.power_usage
            )),
        ]),
        MiningEvent::NetworkEvent {
            connected,
            peer_count,
            pool_latency,
        } => {
            let status = if *connected { "Connected" } else { "Disconnected" };
            let color = if *connected { Color::Green } else { Color::Red };

            Line::from(vec![
                Span::styled("[NET]  ", Style::default().fg(color)),
                Span::raw(format!(
                    "{} - {} peers, {:.0}ms latency",
                    status, peer_count, pool_latency
                )),
            ])
        }
    }
}

// Fallback for when TUI feature is disabled
#[cfg(not(feature = "tui"))]
pub async fn run_tui(
    _stats_rx: mpsc::UnboundedReceiver<GlobalMiningStats>,
    _event_rx: mpsc::UnboundedReceiver<MiningEvent>,
) -> Result<()> {
    anyhow::bail!("TUI feature is not enabled. Compile with --features tui");
}
