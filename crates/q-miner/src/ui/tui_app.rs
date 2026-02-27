// ═══════════════════════════════════════════════════════════════════
// MinerTuiApp: Beautiful diagnostic dashboard for Q-NarwhalKnight miner
//
// Features:
// - 5-tab dashboard (Dashboard, Diagnostics, Network, Events, Settings)
// - Tracing log capture via MinerTuiLogLayer
// - 10 health checks with fix suggestions
// - Network throttle toggle [T]
// - Thread/intensity live adjustments
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Tabs},
    Frame, Terminal,
};

#[cfg(feature = "tui")]
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use crate::diagnostics::MinerDiagnostics;
use crate::shared_state::{DiagnosticEvent, MinerThrottleMode, SharedMinerState};
use anyhow::Result;
use std::collections::VecDeque;
use std::io;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing;

// Also re-export for non-tui builds
use crate::{GlobalMiningStats, MiningEvent};

const HASHRATE_HISTORY_SIZE: usize = 120;
const LATENCY_HISTORY_SIZE: usize = 60;
const MAX_LOG_ENTRIES: usize = 1000;
const TAB_COUNT: usize = 5;

// ═══════════════════════════════════════════════════════════════════
// Log entry types for the TUI log viewer
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
    Success,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub message: String,
}

// ═══════════════════════════════════════════════════════════════════
// MinerTuiLogLayer: Captures tracing events into the TUI log
// Pattern from q-tui/src/lib.rs
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
pub struct MinerTuiLogLayer {
    tx: mpsc::UnboundedSender<LogEntry>,
}

#[cfg(feature = "tui")]
impl MinerTuiLogLayer {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<LogEntry>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }
}

#[cfg(feature = "tui")]
impl<S> tracing_subscriber::Layer<S> for MinerTuiLogLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let level = match *event.metadata().level() {
            tracing::Level::ERROR => LogLevel::Error,
            tracing::Level::WARN => LogLevel::Warn,
            _ => LogLevel::Info,
        };

        // Extract the message from the event
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let message = visitor.message;
        if message.is_empty() {
            return;
        }

        // Detect success messages by content
        let level = if message.contains("Solution accepted")
            || message.contains("MINING REWARD")
            || message.contains("Balance Updated")
            || message.contains("accepted!")
        {
            LogLevel::Success
        } else {
            level
        };

        let now = chrono::Local::now();
        let timestamp = now.format("%H:%M:%S").to_string();

        let _ = self.tx.send(LogEntry {
            timestamp,
            level,
            message,
        });
    }
}

#[cfg(feature = "tui")]
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

#[cfg(feature = "tui")]
impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
            // Remove surrounding quotes from format!("{:?}")
            if self.message.starts_with('"') && self.message.ends_with('"') {
                self.message = self.message[1..self.message.len()-1].to_string();
            }
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MinerTuiApp: Main application state
// ═══════════════════════════════════════════════════════════════════

pub struct MinerTuiApp {
    pub current_tab: usize,
    pub state: Option<Arc<SharedMinerState>>,
    pub diagnostics: MinerDiagnostics,

    // Sparkline data
    pub hashrate_history: VecDeque<f64>,   // KH/s values
    pub latency_history: VecDeque<f64>,    // ms values
    pub peak_hashrate_khs: f64,

    // Block info (updated from events)
    pub current_block_height: u64,
    pub current_block_reward: f64,

    // Logs
    pub logs: VecDeque<LogEntry>,
    pub log_filter: usize,         // 0=All, 1=Info+, 2=Warn+, 3=Error
    pub log_scroll_offset: usize,  // 0=auto-scroll (latest), >0=manual scroll

    // UI state
    pub running: bool,
    pub show_help: bool,

    // Timing
    pub start_time: Instant,
    last_diagnostics_run: Instant,
}

impl MinerTuiApp {
    pub fn new(state: Option<Arc<SharedMinerState>>) -> Self {
        Self {
            current_tab: 0,
            state,
            diagnostics: MinerDiagnostics::new(),
            hashrate_history: VecDeque::with_capacity(HASHRATE_HISTORY_SIZE),
            latency_history: VecDeque::with_capacity(LATENCY_HISTORY_SIZE),
            peak_hashrate_khs: 0.0,
            current_block_height: 0,
            current_block_reward: 0.0,
            logs: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            log_filter: 0,
            log_scroll_offset: 0,
            running: true,
            show_help: false,
            start_time: Instant::now(),
            last_diagnostics_run: Instant::now(),
        }
    }

    pub fn current_hashrate_khs(&self) -> f64 {
        if let Some(ref state) = self.state {
            state.get_hashrate_khs()
        } else {
            0.0
        }
    }

    pub fn tick(&mut self) {
        // Update hashrate history
        let khs = self.current_hashrate_khs();
        self.hashrate_history.push_back(khs);
        if self.hashrate_history.len() > HASHRATE_HISTORY_SIZE {
            self.hashrate_history.pop_front();
        }
        if khs > self.peak_hashrate_khs {
            self.peak_hashrate_khs = khs;
        }

        // Update latency history
        if let Some(ref state) = self.state {
            let lat_us = state.last_challenge_latency_us.load(Ordering::Relaxed);
            let lat_ms = lat_us as f64 / 1000.0;
            self.latency_history.push_back(lat_ms);
            if self.latency_history.len() > LATENCY_HISTORY_SIZE {
                self.latency_history.pop_front();
            }
        }

        // Auto-run diagnostics every 10 seconds
        if self.last_diagnostics_run.elapsed() >= Duration::from_secs(10) {
            if let Some(ref state) = self.state {
                self.diagnostics.run_checks(state);
            }
            self.last_diagnostics_run = Instant::now();
        }
    }

    pub fn add_log(&mut self, entry: LogEntry) {
        self.logs.push_back(entry);
        while self.logs.len() > MAX_LOG_ENTRIES {
            self.logs.pop_front();
        }
    }

    pub fn process_event(&mut self, event: DiagnosticEvent) {
        let now = chrono::Local::now().format("%H:%M:%S").to_string();

        match event {
            DiagnosticEvent::SolutionAccepted { block_height, reward_qnk } => {
                self.current_block_height = block_height;
                if reward_qnk > 0.0 {
                    self.current_block_reward = reward_qnk;
                }
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Success,
                    message: format!("Solution accepted! {:.6} QNK at block #{}", reward_qnk, block_height),
                });
            }
            DiagnosticEvent::SolutionFound { thread_id, block_height, nonce } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: format!("Solution found! Block #{}, Thread {}", block_height, thread_id),
                });
            }
            DiagnosticEvent::SolutionRejected { block_height, reason } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Warn,
                    message: format!("Solution rejected at block #{}: {}", block_height, reason),
                });
            }
            DiagnosticEvent::ChallengeFetched { thread_id, block_height, latency_ms } => {
                self.current_block_height = block_height;
            }
            DiagnosticEvent::ChallengeFetchFailed { thread_id, error } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Error,
                    message: format!("Thread {} challenge fetch failed: {}", thread_id, error),
                });
            }
            DiagnosticEvent::NewBlockSignal { block_height } => {
                self.current_block_height = block_height;
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: format!("New block #{} detected", block_height),
                });
            }
            DiagnosticEvent::MiningReward { reward_qnk, block_height } => {
                self.current_block_height = block_height;
                if reward_qnk > 0.0 {
                    self.current_block_reward = reward_qnk;
                }
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Success,
                    message: format!("Mining reward: {:.8} QNK at block #{}", reward_qnk, block_height),
                });
            }
            DiagnosticEvent::BalanceUpdated { new_balance } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Success,
                    message: format!("Balance updated: {:.8} QNK", new_balance),
                });
            }
            DiagnosticEvent::SseConnected { url } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: format!("SSE connected: {}", url),
                });
            }
            DiagnosticEvent::SseDisconnected { error } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Warn,
                    message: format!("SSE disconnected: {}", error),
                });
            }
            DiagnosticEvent::MinerLinkConnected => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: "MinerLink connected".into(),
                });
            }
            DiagnosticEvent::MinerLinkDisconnected => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Warn,
                    message: "MinerLink disconnected".into(),
                });
            }
            DiagnosticEvent::ServerNotice { message } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Warn,
                    message: format!("[SERVER] {}", message),
                });
            }
            DiagnosticEvent::UpdateAvailable { min_miner_version } => {
                self.diagnostics.min_miner_version = Some(min_miner_version.clone());
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Warn,
                    message: format!("Miner update required: minimum v{}, you have v{}",
                        min_miner_version, env!("CARGO_PKG_VERSION")),
                });
            }
            DiagnosticEvent::ServerSyncing { blocks_behind } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Warn,
                    message: format!("Server syncing: {} blocks behind", blocks_behind),
                });
            }
            DiagnosticEvent::ServerSyncComplete => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Success,
                    message: "Server sync complete - mining starts".into(),
                });
            }
            DiagnosticEvent::ThreadStarted { thread_id } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: format!("Thread {} started", thread_id),
                });
            }
            DiagnosticEvent::ThreadStopped { thread_id } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: format!("Thread {} stopped", thread_id),
                });
            }
            DiagnosticEvent::ThreadError { thread_id, message } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Error,
                    message: format!("Thread {} error: {}", thread_id, message),
                });
            }
            DiagnosticEvent::ThrottleChanged { mode } => {
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Info,
                    message: format!("Throttle changed to: {}", mode.label()),
                });
            }
        }
    }

    fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % TAB_COUNT;
    }

    fn prev_tab(&mut self) {
        if self.current_tab > 0 {
            self.current_tab -= 1;
        } else {
            self.current_tab = TAB_COUNT - 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// TUI Runner: Terminal init, event loop, cleanup
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
pub async fn run_miner_tui(
    state: Arc<SharedMinerState>,
    mut event_rx: mpsc::UnboundedReceiver<DiagnosticEvent>,
    mut log_rx: mpsc::UnboundedReceiver<LogEntry>,
) -> Result<()> {
    // Windows: Enable VT processing so ANSI escape sequences work in cmd.exe/PowerShell.
    // Without this, EnterAlternateScreen silently fails and the TUI never appears.
    #[cfg(target_os = "windows")]
    {
        use crossterm::execute;
        // crossterm::terminal::enable_raw_mode already tries to set VT, but we
        // also need it on stdout BEFORE entering the alternate screen.  The
        // simplest cross-version way is to call the Windows API directly.
        unsafe {
            extern "system" {
                fn GetStdHandle(nStdHandle: u32) -> *mut std::ffi::c_void;
                fn GetConsoleMode(hConsoleHandle: *mut std::ffi::c_void, lpMode: *mut u32) -> i32;
                fn SetConsoleMode(hConsoleHandle: *mut std::ffi::c_void, dwMode: u32) -> i32;
            }
            const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5u32; // -11i32 as u32
            const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
            let handle = GetStdHandle(STD_OUTPUT_HANDLE);
            if !handle.is_null() {
                let mut mode: u32 = 0;
                if GetConsoleMode(handle, &mut mode) != 0 {
                    let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }
        }
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = MinerTuiApp::new(Some(state.clone()));

    // Initial diagnostics run
    app.diagnostics.run_checks(&state);

    // Main TUI loop
    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    while app.running {
        // Draw UI
        terminal.draw(|f| draw_ui(f, &app))?;

        // Handle input with timeout
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_millis(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if app.show_help {
                        // Any key closes help
                        app.show_help = false;
                    } else {
                        handle_key_press(&mut app, key.code, key.modifiers);
                    }
                }
            }
        }

        // Process diagnostic events (cap per frame to avoid stalling render)
        for _ in 0..64 {
            match event_rx.try_recv() {
                Ok(ev) => app.process_event(ev),
                Err(_) => break,
            }
        }

        // Process log entries from tracing layer (cap per frame)
        for _ in 0..64 {
            match log_rx.try_recv() {
                Ok(log) => app.add_log(log),
                Err(_) => break,
            }
        }

        // Tick (update histories, auto-diagnostics)
        if last_tick.elapsed() >= tick_rate {
            app.tick();
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

    // Signal shutdown
    state.is_running.store(false, Ordering::SeqCst);

    Ok(())
}

#[cfg(feature = "tui")]
fn handle_key_press(app: &mut MinerTuiApp, code: KeyCode, modifiers: KeyModifiers) {
    match code {
        // Quit
        KeyCode::Char('q') | KeyCode::Esc => {
            app.running = false;
        }

        // Tab navigation
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('1') if app.current_tab == 3 => app.log_filter = 0,
        KeyCode::Char('2') if app.current_tab == 3 => app.log_filter = 1,
        KeyCode::Char('3') if app.current_tab == 3 => app.log_filter = 2,
        KeyCode::Char('4') if app.current_tab == 3 => app.log_filter = 3,

        // Pause/resume
        KeyCode::Char('p') => {
            if let Some(ref state) = app.state {
                let current = state.is_paused.load(Ordering::Relaxed);
                state.is_paused.store(!current, Ordering::SeqCst);
            }
        }

        // Help
        KeyCode::Char('h') | KeyCode::Char('?') => {
            app.show_help = !app.show_help;
        }

        // Re-run diagnostics
        KeyCode::Char('r') | KeyCode::Char('R') => {
            if let Some(ref state) = app.state {
                app.diagnostics.run_checks(state);
            }
        }

        // Throttle toggle
        KeyCode::Char('t') | KeyCode::Char('T') => {
            if let Some(ref state) = app.state {
                let mut mode = state.throttle_mode.write();
                let new_mode = mode.next();
                *mode = new_mode;
                state.send_event(DiagnosticEvent::ThrottleChanged { mode: new_mode });
            }
        }

        // Thread count
        KeyCode::Char('+') | KeyCode::Char('=') => {
            if let Some(ref state) = app.state {
                let current = state.target_threads.load(Ordering::Relaxed);
                let max = num_cpus::get();
                if current < max {
                    state.target_threads.store(current + 1, Ordering::SeqCst);
                }
            }
        }
        KeyCode::Char('-') | KeyCode::Char('_') => {
            if let Some(ref state) = app.state {
                let current = state.target_threads.load(Ordering::Relaxed);
                if current > 1 {
                    state.target_threads.store(current - 1, Ordering::SeqCst);
                }
            }
        }

        // Intensity
        KeyCode::Char('>') | KeyCode::Char('.') => {
            if let Some(ref state) = app.state {
                let current = state.target_intensity.load(Ordering::Relaxed);
                if current < 10 {
                    state.target_intensity.store(current + 1, Ordering::SeqCst);
                }
            }
        }
        KeyCode::Char('<') | KeyCode::Char(',') => {
            if let Some(ref state) = app.state {
                let current = state.target_intensity.load(Ordering::Relaxed);
                if current > 1 {
                    state.target_intensity.store(current - 1, Ordering::SeqCst);
                }
            }
        }

        // Scroll (Events tab)
        KeyCode::Up => {
            if app.current_tab == 3 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_add(1);
            }
        }
        KeyCode::Down => {
            if app.current_tab == 3 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_sub(1);
            }
        }
        KeyCode::PageUp => {
            if app.current_tab == 3 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_add(10);
            }
        }
        KeyCode::PageDown => {
            if app.current_tab == 3 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_sub(10);
            }
        }
        KeyCode::Home => {
            if app.current_tab == 3 {
                app.log_scroll_offset = app.logs.len(); // Scroll to top
            }
        }
        KeyCode::End => {
            if app.current_tab == 3 {
                app.log_scroll_offset = 0; // Auto-scroll (latest)
            }
        }

        _ => {}
    }
}

// ═══════════════════════════════════════════════════════════════════
// Drawing
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn draw_ui(f: &mut Frame, app: &MinerTuiApp) {
    let size = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header + tabs
            Constraint::Min(0),   // Content
            Constraint::Length(3), // Footer
        ])
        .split(size);

    draw_header(f, chunks[0], app);
    super::tui_views::draw_tab_content(f, chunks[1], app);
    draw_footer(f, chunks[2], app);

    if app.show_help {
        draw_help_overlay(f, size);
    }
}

#[cfg(feature = "tui")]
fn draw_header(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let uptime = app.start_time.elapsed();
    let hrs = uptime.as_secs() / 3600;
    let mins = (uptime.as_secs() % 3600) / 60;

    let is_paused = app.state.as_ref()
        .map(|s| s.is_paused.load(Ordering::Relaxed))
        .unwrap_or(false);

    let status = if is_paused { "PAUSED" } else { "MINING" };
    let status_color = if is_paused { Color::Yellow } else { Color::Green };

    let title = format!(
        " Q-NarwhalKnight Miner v{} --- {} --- Uptime: {}h {:02}m ",
        env!("CARGO_PKG_VERSION"), status, hrs, mins
    );

    let tab_titles = vec!["Dashboard", "Diagnostics", "Network", "Events", "Settings"];
    let tabs = Tabs::new(tab_titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(status_color))
                .title(title),
        )
        .select(app.current_tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )
        .divider("|");

    f.render_widget(tabs, area);
}

#[cfg(feature = "tui")]
fn draw_footer(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let is_paused = app.state.as_ref()
        .map(|s| s.is_paused.load(Ordering::Relaxed))
        .unwrap_or(false);

    let status_span = if is_paused {
        Span::styled(" PAUSED ", Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD))
    } else {
        Span::styled(" MINING ", Style::default().fg(Color::Black).bg(Color::Green).add_modifier(Modifier::BOLD))
    };

    let line = Line::from(vec![
        Span::raw(" "),
        status_span,
        Span::raw("  "),
        Span::styled("[q]", Style::default().fg(Color::Cyan)),
        Span::raw("Quit "),
        Span::styled("[Tab]", Style::default().fg(Color::Cyan)),
        Span::raw("Next "),
        Span::styled("[h]", Style::default().fg(Color::Cyan)),
        Span::raw("Help "),
        Span::styled("[p]", Style::default().fg(Color::Cyan)),
        Span::raw("Pause "),
        Span::styled("[+/-]", Style::default().fg(Color::Cyan)),
        Span::raw("Threads "),
        Span::styled("[T]", Style::default().fg(Color::Cyan)),
        Span::raw("Throttle "),
        Span::styled("[R]", Style::default().fg(Color::Cyan)),
        Span::raw("Diag"),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    f.render_widget(Paragraph::new(line).block(block), area);
}

#[cfg(feature = "tui")]
fn draw_help_overlay(f: &mut Frame, area: Rect) {
    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            " Keyboard Shortcuts",
            Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  q, Esc      ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit miner"),
        ]),
        Line::from(vec![
            Span::styled("  Tab         ", Style::default().fg(Color::Yellow)),
            Span::raw("Next tab"),
        ]),
        Line::from(vec![
            Span::styled("  Shift+Tab   ", Style::default().fg(Color::Yellow)),
            Span::raw("Previous tab"),
        ]),
        Line::from(vec![
            Span::styled("  p           ", Style::default().fg(Color::Yellow)),
            Span::raw("Pause/Resume mining"),
        ]),
        Line::from(vec![
            Span::styled("  h, ?        ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle this help"),
        ]),
        Line::from(vec![
            Span::styled("  T           ", Style::default().fg(Color::Yellow)),
            Span::raw("Cycle network throttle (Off/Light/Heavy)"),
        ]),
        Line::from(vec![
            Span::styled("  R           ", Style::default().fg(Color::Yellow)),
            Span::raw("Re-run diagnostics"),
        ]),
        Line::from(vec![
            Span::styled("  +/-         ", Style::default().fg(Color::Yellow)),
            Span::raw("Adjust thread count"),
        ]),
        Line::from(vec![
            Span::styled("  >/<         ", Style::default().fg(Color::Yellow)),
            Span::raw("Adjust intensity"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Events Tab",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  1-4         ", Style::default().fg(Color::Yellow)),
            Span::raw("Filter: All / Info+ / Warn+ / Error"),
        ]),
        Line::from(vec![
            Span::styled("  Up/Down     ", Style::default().fg(Color::Yellow)),
            Span::raw("Scroll log"),
        ]),
        Line::from(vec![
            Span::styled("  PgUp/PgDn   ", Style::default().fg(Color::Yellow)),
            Span::raw("Scroll fast"),
        ]),
        Line::from(vec![
            Span::styled("  Home/End    ", Style::default().fg(Color::Yellow)),
            Span::raw("Jump to top/bottom"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Press any key to close",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let popup_area = centered_rect(60, 70, area);
    let block = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(" Help "),
        )
        .alignment(Alignment::Left);

    f.render_widget(Clear, popup_area);
    f.render_widget(block, popup_area);
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

// ═══════════════════════════════════════════════════════════════════
// Legacy compatibility: Keep TuiApp and run_tui for old code
// ═══════════════════════════════════════════════════════════════════

/// Legacy TUI Application State (kept for backward compatibility)
pub struct TuiApp {
    pub tab_index: usize,
    pub hash_rate_history: VecDeque<f64>,
    pub stats: GlobalMiningStats,
    pub events: VecDeque<MiningEvent>,
    pub start_time: Instant,
    pub selected_gpu: usize,
    pub running: bool,
    pub paused: bool,
    pub show_help: bool,
    pub gpu_temp_history: Vec<VecDeque<f64>>,
    pub gpu_power_history: Vec<VecDeque<f64>>,
}

impl TuiApp {
    pub fn new() -> Self {
        Self {
            tab_index: 0,
            hash_rate_history: VecDeque::with_capacity(60),
            stats: GlobalMiningStats::default(),
            events: VecDeque::with_capacity(100),
            start_time: Instant::now(),
            selected_gpu: 0,
            running: true,
            paused: false,
            show_help: false,
            gpu_temp_history: vec![VecDeque::with_capacity(60); 8],
            gpu_power_history: vec![VecDeque::with_capacity(60); 8],
        }
    }

    pub fn update_stats(&mut self, stats: GlobalMiningStats) {
        self.hash_rate_history.push_back(stats.total_hash_rate);
        if self.hash_rate_history.len() > 60 {
            self.hash_rate_history.pop_front();
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
        if self.tab_index > 0 { self.tab_index -= 1; } else { self.tab_index = 3; }
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

// Fallback run_tui (old API, still works)
#[cfg(feature = "tui")]
pub async fn run_tui(
    mut _stats_rx: mpsc::UnboundedReceiver<GlobalMiningStats>,
    mut _event_rx: mpsc::UnboundedReceiver<MiningEvent>,
) -> Result<()> {
    anyhow::bail!("Legacy run_tui is deprecated. Use run_miner_tui() instead.");
}

#[cfg(not(feature = "tui"))]
pub async fn run_tui(
    _stats_rx: mpsc::UnboundedReceiver<GlobalMiningStats>,
    _event_rx: mpsc::UnboundedReceiver<MiningEvent>,
) -> Result<()> {
    anyhow::bail!("TUI feature is not enabled. Compile with --features tui");
}
