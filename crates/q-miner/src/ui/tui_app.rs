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
use crate::shared_state::{DiagnosticEvent, MinerThrottleMode, SharedMinerState, StarshipSyncInfo};
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
const BANDWIDTH_HISTORY_SIZE: usize = 60; // 60 ticks of bandwidth samples
const MAX_LOG_ENTRIES: usize = 1000;
const TAB_COUNT: usize = 6;

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
// Wallet password persistence (SHA-256 hash stored in .wallet_password file)
// ═══════════════════════════════════════════════════════════════════

fn wallet_password_path() -> std::path::PathBuf {
    // Store next to node data in current working directory
    std::path::PathBuf::from(".wallet_password")
}

fn load_wallet_password_hash() -> Option<String> {
    let path = wallet_password_path();
    std::fs::read_to_string(&path).ok().map(|s| s.trim().to_string()).filter(|s| s.len() == 64)
}

fn save_wallet_password_hash(hash: &str) {
    let path = wallet_password_path();
    let _ = std::fs::write(&path, hash);
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

    // Wallet tab state
    pub wallet_balance: f64,
    pub wallet_send_mode: bool,       // true = showing send form
    pub wallet_send_address: String,  // recipient address being typed
    pub wallet_send_amount: String,   // amount being typed
    pub wallet_send_field: u8,        // 0=address, 1=amount
    pub wallet_send_status: Option<String>, // result message
    pub wallet_send_confirming: bool, // awaiting Enter to confirm
    // v8.6.5: Password protection for sends
    pub wallet_send_password: String,      // password being typed in confirm step
    pub wallet_send_password_err: bool,    // true if last password check failed
    pub wallet_password_hash: Option<String>, // stored SHA-256 hash of wallet password
    pub wallet_password_setting: bool,      // true = currently setting a new password
    pub wallet_password_new: String,        // new password being typed

    // v8.6.6: Bandwidth statistics
    pub bandwidth_down_history: VecDeque<f64>,  // KB/s download per tick
    pub bandwidth_up_history: VecDeque<f64>,    // KB/s upload per tick
    pub prev_bytes_down: u64,                    // previous tick's total bytes
    pub prev_bytes_up: u64,
    pub total_api_requests: u64,
    pub total_api_failures: u64,

    // v9.0.4: Starship sync telemetry for TUI
    pub sync_info: Option<StarshipSyncInfo>,

    // v9.1.0: Compute Power Layer stats for TUI cards
    pub simd_tier: String,
    pub simd_batch_size: usize,
    pub network_compute_peers: u32,
    pub network_total_hashrate_hs: f64,
    pub live_security_bits: f64,

    // UI state
    pub running: bool,
    pub show_help: bool,

    // Timing
    pub start_time: Instant,
    last_diagnostics_run: Instant,
    last_bandwidth_tick: Instant,
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
            wallet_balance: 0.0,
            wallet_send_mode: false,
            wallet_send_address: String::new(),
            wallet_send_amount: String::new(),
            wallet_send_field: 0,
            wallet_send_status: None,
            wallet_send_confirming: false,
            wallet_send_password: String::new(),
            wallet_send_password_err: false,
            wallet_password_hash: load_wallet_password_hash(),
            wallet_password_setting: false,
            wallet_password_new: String::new(),
            bandwidth_down_history: VecDeque::with_capacity(BANDWIDTH_HISTORY_SIZE),
            bandwidth_up_history: VecDeque::with_capacity(BANDWIDTH_HISTORY_SIZE),
            prev_bytes_down: 0,
            prev_bytes_up: 0,
            total_api_requests: 0,
            total_api_failures: 0,
            sync_info: None,
            simd_tier: {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") { "AVX-512".to_string() }
                    else if is_x86_feature_detected!("avx2") { "AVX2".to_string() }
                    else { "SSE2".to_string() }
                }
                #[cfg(target_arch = "aarch64")]
                { "NEON".to_string() }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                { "Scalar".to_string() }
            },
            simd_batch_size: crate::cpu::optimal_mining_batch_size(),
            network_compute_peers: 0,
            network_total_hashrate_hs: 0.0,
            live_security_bits: 0.0,
            running: true,
            show_help: false,
            start_time: Instant::now(),
            last_diagnostics_run: Instant::now(),
            last_bandwidth_tick: Instant::now(),
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

        // Update bandwidth statistics
        if let Some(ref state) = self.state {
            let elapsed = self.last_bandwidth_tick.elapsed().as_secs_f64().max(0.1);
            self.last_bandwidth_tick = Instant::now();

            let cur_down = state.bytes_downloaded.load(Ordering::Relaxed);
            let cur_up = state.bytes_uploaded.load(Ordering::Relaxed);

            let delta_down = cur_down.saturating_sub(self.prev_bytes_down);
            let delta_up = cur_up.saturating_sub(self.prev_bytes_up);

            let kbs_down = (delta_down as f64 / 1024.0) / elapsed;
            let kbs_up = (delta_up as f64 / 1024.0) / elapsed;

            self.bandwidth_down_history.push_back(kbs_down);
            self.bandwidth_up_history.push_back(kbs_up);
            if self.bandwidth_down_history.len() > BANDWIDTH_HISTORY_SIZE {
                self.bandwidth_down_history.pop_front();
            }
            if self.bandwidth_up_history.len() > BANDWIDTH_HISTORY_SIZE {
                self.bandwidth_up_history.pop_front();
            }

            self.prev_bytes_down = cur_down;
            self.prev_bytes_up = cur_up;

            self.total_api_requests = state.api_requests_total.load(Ordering::Relaxed);
            self.total_api_failures = state.api_requests_failed.load(Ordering::Relaxed);
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
                self.wallet_balance = new_balance;
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
            DiagnosticEvent::ServerSyncing { sync_info } => {
                // Only log every ~30s (6 polls at 5s interval) to avoid spam
                let should_log = self.sync_info.is_none()
                    || sync_info.blocks_behind % 500 < 5
                    || sync_info.blocks_behind < 20;
                if should_log {
                    self.add_log(LogEntry {
                        timestamp: now,
                        level: LogLevel::Warn,
                        message: format!("\u{1F680} {} | {:.1}% | {:.0} blk/s | {} behind",
                            sync_info.phase, sync_info.sync_progress,
                            sync_info.sync_speed_bps, sync_info.blocks_behind),
                    });
                }
                self.sync_info = Some(sync_info);
            }
            DiagnosticEvent::ServerSyncComplete => {
                self.sync_info = None;
                self.add_log(LogEntry {
                    timestamp: now,
                    level: LogLevel::Success,
                    message: "\u{1F30D} Orbit achieved - mining online!".into(),
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
            DiagnosticEvent::ComputePowerUpdate { network_hashrate_hs, connected_miners, live_security_bits } => {
                self.network_total_hashrate_hs = network_hashrate_hs;
                self.network_compute_peers = connected_miners;
                self.live_security_bits = live_security_bits;
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

        // Tab navigation — Tab, Shift+Tab, Left/Right arrows, or F1-F6
        KeyCode::Tab if !app.wallet_send_mode => app.next_tab(),
        KeyCode::BackTab if !app.wallet_send_mode => app.prev_tab(),
        KeyCode::Right if !app.wallet_send_mode => app.next_tab(),
        KeyCode::Left if !app.wallet_send_mode => app.prev_tab(),
        KeyCode::F(n) if n >= 1 && n <= 6 => { app.current_tab = (n as usize) - 1; }

        // Events tab log filters (only when on Events tab)
        KeyCode::Char('1') if app.current_tab == 4 => app.log_filter = 0,
        KeyCode::Char('2') if app.current_tab == 4 => app.log_filter = 1,
        KeyCode::Char('3') if app.current_tab == 4 => app.log_filter = 2,
        KeyCode::Char('4') if app.current_tab == 4 => app.log_filter = 3,

        // Wallet tab: password setting mode
        _ if app.current_tab == 1 && app.wallet_password_setting => {
            handle_password_setting(app, code);
            return;
        }
        // Wallet tab input handling
        _ if app.current_tab == 1 && app.wallet_send_mode => {
            handle_wallet_input(app, code, modifiers);
            return;
        }
        // Toggle send mode with 's' on wallet tab
        KeyCode::Char('s') | KeyCode::Char('S') if app.current_tab == 1 => {
            app.wallet_send_mode = !app.wallet_send_mode;
            if app.wallet_send_mode {
                app.wallet_send_address.clear();
                app.wallet_send_amount.clear();
                app.wallet_send_field = 0;
                app.wallet_send_status = None;
                app.wallet_send_confirming = false;
            }
            return;
        }
        // v8.6.5: Set/change wallet send password with 'P' on wallet tab
        KeyCode::Char('P') if app.current_tab == 1 && !app.wallet_send_mode => {
            app.wallet_password_setting = true;
            app.wallet_password_new.clear();
            return;
        }

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
            if app.current_tab == 4 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_add(1);
            }
        }
        KeyCode::Down => {
            if app.current_tab == 4 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_sub(1);
            }
        }
        KeyCode::PageUp => {
            if app.current_tab == 4 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_add(10);
            }
        }
        KeyCode::PageDown => {
            if app.current_tab == 4 {
                app.log_scroll_offset = app.log_scroll_offset.saturating_sub(10);
            }
        }
        KeyCode::Home => {
            if app.current_tab == 4 {
                app.log_scroll_offset = app.logs.len(); // Scroll to top
            }
        }
        KeyCode::End => {
            if app.current_tab == 4 {
                app.log_scroll_offset = 0; // Auto-scroll (latest)
            }
        }

        _ => {}
    }
}

// ═══════════════════════════════════════════════════════════════════
// Wallet Send Input Handler
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn handle_wallet_input(app: &mut MinerTuiApp, code: KeyCode, _modifiers: KeyModifiers) {
    match code {
        KeyCode::Esc => {
            app.wallet_send_mode = false;
            app.wallet_send_confirming = false;
            app.wallet_send_password.clear();
            app.wallet_send_password_err = false;
        }
        KeyCode::Tab => {
            if !app.wallet_send_confirming {
                // Toggle between address and amount fields
                app.wallet_send_field = if app.wallet_send_field == 0 { 1 } else { 0 };
            }
        }
        KeyCode::Backspace => {
            if app.wallet_send_confirming {
                // In confirmation mode: backspace on password field
                if app.wallet_password_hash.is_some() {
                    app.wallet_send_password.pop();
                    app.wallet_send_password_err = false;
                } else {
                    app.wallet_send_confirming = false;
                }
                return;
            }
            if app.wallet_send_field == 0 {
                app.wallet_send_address.pop();
            } else {
                app.wallet_send_amount.pop();
            }
        }
        KeyCode::Enter => {
            if app.wallet_send_confirming {
                // v8.6.5: If password is required, verify before sending
                if let Some(ref stored_hash) = app.wallet_password_hash {
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(app.wallet_send_password.as_bytes());
                    let computed = hex::encode(hasher.finalize());
                    if computed != *stored_hash {
                        app.wallet_send_password_err = true;
                        app.wallet_send_password.clear();
                        return;
                    }
                }
                // Password verified (or no password set) — execute the send
                let to = app.wallet_send_address.clone();
                let amt = app.wallet_send_amount.clone();
                if let Some(ref state) = app.state {
                    let server_url = state.server_url.clone();
                    let from = state.wallet_address.clone();
                    let proxy = state.proxy_url.clone();
                    let to_clone = to.clone();
                    let amt_clone = amt.clone();
                    tokio::spawn(async move {
                        match send_qug_transfer(&server_url, &from, &to_clone, &amt_clone, proxy.as_deref()).await {
                            Ok(msg) => tracing::info!("Transfer sent: {}", msg),
                            Err(e) => tracing::error!("Transfer failed: {}", e),
                        }
                    });
                    app.wallet_send_status = Some(format!("Sending {} QUG to {}...", amt, &to[..20.min(to.len())]));
                }
                app.wallet_send_confirming = false;
                app.wallet_send_mode = false;
                app.wallet_send_password.clear();
                app.wallet_send_password_err = false;
            } else if !app.wallet_send_address.is_empty() && !app.wallet_send_amount.is_empty() {
                // Show confirmation (with password field if password is set)
                app.wallet_send_confirming = true;
                app.wallet_send_password.clear();
                app.wallet_send_password_err = false;
            }
        }
        KeyCode::Char(c) => {
            if app.wallet_send_confirming {
                // v8.6.5: Type into password field during confirmation
                if app.wallet_password_hash.is_some() {
                    app.wallet_send_password.push(c);
                    app.wallet_send_password_err = false;
                }
                return;
            }
            if app.wallet_send_field == 0 {
                // Address field — only hex chars
                if c.is_ascii_hexdigit() && app.wallet_send_address.len() < 64 {
                    app.wallet_send_address.push(c);
                }
            } else {
                // Amount field — digits and dot
                if (c.is_ascii_digit() || c == '.') && app.wallet_send_amount.len() < 20 {
                    app.wallet_send_amount.push(c);
                }
            }
        }
        _ => {}
    }
}

// ═══════════════════════════════════════════════════════════════════
// Password Setting Input Handler
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn handle_password_setting(app: &mut MinerTuiApp, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.wallet_password_setting = false;
            app.wallet_password_new.clear();
        }
        KeyCode::Backspace => {
            app.wallet_password_new.pop();
        }
        KeyCode::Enter => {
            if app.wallet_password_new.len() >= 4 {
                // Hash and save the password
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(app.wallet_password_new.as_bytes());
                let hash = hex::encode(hasher.finalize());
                save_wallet_password_hash(&hash);
                app.wallet_password_hash = Some(hash);
                app.wallet_password_setting = false;
                app.wallet_password_new.clear();
                app.wallet_send_status = Some("Wallet password set successfully".to_string());
            }
        }
        KeyCode::Char(c) => {
            if app.wallet_password_new.len() < 64 {
                app.wallet_password_new.push(c);
            }
        }
        _ => {}
    }
}

// ═══════════════════════════════════════════════════════════════════
// Send QUG Transfer via API
// ═══════════════════════════════════════════════════════════════════

async fn send_qug_transfer(
    server_url: &str,
    from: &str,
    to: &str,
    amount: &str,
    proxy_url: Option<&str>,
) -> anyhow::Result<String> {
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .danger_accept_invalid_certs(true);
    if let Some(proxy) = proxy_url {
        builder = builder.proxy(reqwest::Proxy::all(proxy)?);
    }
    let client = builder.build()?;
    let url = format!("{}/api/v1/transfer", server_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "from": from,
        "to": to,
        "amount": amount,
    });
    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.is_success() {
        Ok(format!("Success: {}", text))
    } else {
        Err(anyhow::anyhow!("HTTP {}: {}", status, text))
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

    let tab_titles = vec!["Dashboard", "Wallet", "Diagnostics", "Network", "Events", "Settings"];
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
            Span::raw("Cycle network throttle (Off/UltraLight/Light/Heavy)"),
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
            " Wallet Tab",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  S           ", Style::default().fg(Color::Yellow)),
            Span::raw("Open/close Send form"),
        ]),
        Line::from(vec![
            Span::styled("  Tab         ", Style::default().fg(Color::Yellow)),
            Span::raw("Switch Address/Amount field"),
        ]),
        Line::from(vec![
            Span::styled("  Enter       ", Style::default().fg(Color::Yellow)),
            Span::raw("Confirm → Send"),
        ]),
        Line::from(vec![
            Span::styled("  Esc         ", Style::default().fg(Color::Yellow)),
            Span::raw("Cancel send"),
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
