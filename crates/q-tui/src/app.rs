use crate::metrics::{Metrics, NetworkThrottleMode};
use chrono::{DateTime, Utc};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ringbuf::{HeapRb, Rb};
use std::sync::{Arc, RwLock};

/// Application view modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Dashboard,
    Wallet,
    FullLogs,
    Network,
    Stats,
    Physics,
    Menu,
    Bounty,
}

/// Log entry with timestamp and message
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO ",
            LogLevel::Warn => "WARN ",
            LogLevel::Error => "ERROR",
        }
    }

    pub fn should_display(&self, filter: LogLevel) -> bool {
        *self >= filter
    }
}

/// Main application state
pub struct App {
    /// Current view mode
    pub view_mode: ViewMode,

    /// Node metrics
    pub metrics: Arc<RwLock<Metrics>>,

    /// Log buffer (last 1000 entries)
    pub logs: Arc<RwLock<HeapRb<LogEntry>>>,

    /// Logs paused?
    pub logs_paused: bool,

    /// Log scroll offset
    pub log_scroll: usize,

    /// Log level filter (defaults to Info)
    pub log_filter: LogLevel,

    /// Menu selection
    pub menu_selection: usize,

    /// TPS history (last 60 data points)
    pub tps_history: Arc<RwLock<HeapRb<f64>>>,

    /// Bandwidth in history (last 60 samples of bytes/s)
    pub bw_in_history: Arc<RwLock<HeapRb<u64>>>,

    /// Bandwidth out history (last 60 samples of bytes/s)
    pub bw_out_history: Arc<RwLock<HeapRb<u64>>>,

    /// Peak bandwidth in (bytes/s) this session
    pub peak_bw_in: u64,

    /// Peak bandwidth out (bytes/s) this session
    pub peak_bw_out: u64,

    /// Current network throttle mode
    pub network_throttle_mode: NetworkThrottleMode,

    /// Should quit?
    pub should_quit: bool,

    /// Bounty registration state
    pub bounty_testnet_address: String,
    pub bounty_mainnet_address: String,
    pub bounty_input_field: BountyInputField,
    pub bounty_status_message: String,

    /// v9.8.4: Auto-update animation overlay (rainbow progress bar + success modal)
    pub update_animation: crate::ui::update_animation::UpdateAnimation,

    /// v9.8.5: Starship sync animation (rocket launch + orbital visualization)
    pub starship_animation: crate::ui::starship_animation::StarshipAnimation,

    /// v9.8.6: Water robots animation (quantum marine creatures on important events)
    pub water_robots: crate::ui::water_robots_animation::WaterRobotsAnimation,

    /// v9.9.0: Command Center — radar, ocean, topology for network view
    pub command_center: crate::ui::command_center::CommandCenterState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BountyInputField {
    TestnetAddress,
    MainnetAddress,
}

impl App {
    pub fn new() -> Self {
        Self::with_metrics(Arc::new(RwLock::new(Metrics::default())))
    }

    /// Create a new App with externally managed metrics (for live updates)
    pub fn with_metrics(metrics: Arc<RwLock<Metrics>>) -> Self {
        Self {
            view_mode: ViewMode::Dashboard,
            metrics,
            logs: Arc::new(RwLock::new(HeapRb::new(1000))),
            logs_paused: false,
            log_scroll: 0,
            log_filter: LogLevel::Info, // Default: filter out Debug and Trace
            menu_selection: 0,
            tps_history: Arc::new(RwLock::new(HeapRb::new(60))),
            bw_in_history: Arc::new(RwLock::new(HeapRb::new(60))),
            bw_out_history: Arc::new(RwLock::new(HeapRb::new(60))),
            peak_bw_in: 0,
            peak_bw_out: 0,
            network_throttle_mode: NetworkThrottleMode::default(), // v8.5.4: Turbo by default
            should_quit: false,
            bounty_testnet_address: String::new(),
            bounty_mainnet_address: String::new(),
            bounty_input_field: BountyInputField::TestnetAddress,
            bounty_status_message: String::new(),
            update_animation: crate::ui::update_animation::UpdateAnimation::new(),
            starship_animation: crate::ui::starship_animation::StarshipAnimation::new(),
            water_robots: crate::ui::water_robots_animation::WaterRobotsAnimation::new(),
            command_center: crate::ui::command_center::CommandCenterState::new(),
        }
    }

    /// Create a new App with externally managed metrics AND log buffer.
    /// The log buffer should be the same Arc passed to `TuiLogLayer` so that
    /// tracing events appear in the TUI log panel.
    pub fn with_metrics_and_logs(
        metrics: Arc<RwLock<Metrics>>,
        logs: Arc<RwLock<HeapRb<LogEntry>>>,
    ) -> Self {
        Self {
            view_mode: ViewMode::Dashboard,
            metrics,
            logs,
            logs_paused: false,
            log_scroll: 0,
            log_filter: LogLevel::Info,
            menu_selection: 0,
            tps_history: Arc::new(RwLock::new(HeapRb::new(60))),
            bw_in_history: Arc::new(RwLock::new(HeapRb::new(60))),
            bw_out_history: Arc::new(RwLock::new(HeapRb::new(60))),
            peak_bw_in: 0,
            peak_bw_out: 0,
            network_throttle_mode: NetworkThrottleMode::default(), // v8.5.4: Turbo by default
            should_quit: false,
            bounty_testnet_address: String::new(),
            bounty_mainnet_address: String::new(),
            bounty_input_field: BountyInputField::TestnetAddress,
            bounty_status_message: String::new(),
            update_animation: crate::ui::update_animation::UpdateAnimation::new(),
            starship_animation: crate::ui::starship_animation::StarshipAnimation::new(),
            water_robots: crate::ui::water_robots_animation::WaterRobotsAnimation::new(),
            command_center: crate::ui::command_center::CommandCenterState::new(),
        }
    }

    /// Add a log entry (with level filtering)
    pub fn add_log(&mut self, level: LogLevel, target: String, message: String) {
        // Filter out logs below the current filter level
        if !level.should_display(self.log_filter) {
            return;
        }

        if let Ok(mut logs) = self.logs.write() {
            logs.push_overwrite(LogEntry {
                timestamp: Utc::now(),
                level,
                target,
                message,
            });
        }
    }

    /// Handle keyboard events
    pub fn handle_key_event(&mut self, key: KeyEvent) -> bool {
        // v9.9.0: Command Center panel keys when in Network view
        if self.view_mode == ViewMode::Network {
            match key.code {
                KeyCode::Char(c @ ('r' | 'R' | 'o' | 'O' | 'v' | 'V')) => {
                    if self.command_center.handle_key(c) {
                        return false;
                    }
                }
                KeyCode::Esc => {
                    if self.command_center.focused_panel != crate::ui::command_center::Panel::Overview {
                        self.command_center.back_to_overview();
                        return false;
                    }
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.should_quit = true;
                return true;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
                return true;
            }
            KeyCode::Tab => {
                self.cycle_view();
            }
            KeyCode::Char('m') | KeyCode::Char('M') => {
                self.view_mode = ViewMode::Menu;
            }
            KeyCode::Char('l') | KeyCode::Char('L') => {
                self.view_mode = ViewMode::FullLogs;
            }
            KeyCode::Char('w') | KeyCode::Char('W') => {
                self.view_mode = ViewMode::Wallet;
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                self.view_mode = ViewMode::Dashboard;
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                self.view_mode = ViewMode::Network;
            }
            KeyCode::Char('b') | KeyCode::Char('B') => {
                self.view_mode = ViewMode::Bounty;
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                self.view_mode = ViewMode::Stats;
            }
            KeyCode::Char('f') | KeyCode::Char('F') => {
                self.view_mode = ViewMode::Physics;
            }
            KeyCode::Char('t') | KeyCode::Char('T') => {
                // 🎚️ v8.5.4: Throttle toggle works on ALL views (not just Network)
                self.network_throttle_mode = self.network_throttle_mode.next();
                // Write throttle mode into metrics so the server can read it
                if let Ok(mut metrics) = self.metrics.write() {
                    metrics.network_throttle_mode = self.network_throttle_mode;
                }
            }
            KeyCode::Char('p') | KeyCode::Char('P') => {
                self.logs_paused = !self.logs_paused;
            }
            KeyCode::Up => {
                self.handle_up();
            }
            KeyCode::Down => {
                self.handle_down();
            }
            KeyCode::PageUp => {
                self.log_scroll = self.log_scroll.saturating_add(10);
            }
            KeyCode::PageDown => {
                self.log_scroll = self.log_scroll.saturating_sub(10);
            }
            KeyCode::Esc => {
                if self.view_mode == ViewMode::Menu {
                    self.view_mode = ViewMode::Dashboard;
                }
            }
            KeyCode::Enter => {
                match self.view_mode {
                    ViewMode::Menu => self.handle_menu_selection(),
                    ViewMode::Bounty => {
                        // Trigger async registration - this needs to be handled in the main loop
                        // For now, just set a status message indicating we need async handling
                        self.bounty_status_message = "⏳ Submitting registration... (requires async)".to_string();
                    }
                    _ => {}
                }
            }
            KeyCode::Char(c) if self.view_mode == ViewMode::Bounty => {
                match self.bounty_input_field {
                    BountyInputField::TestnetAddress => self.bounty_testnet_address.push(c),
                    BountyInputField::MainnetAddress => self.bounty_mainnet_address.push(c),
                }
            }
            KeyCode::Backspace if self.view_mode == ViewMode::Bounty => {
                match self.bounty_input_field {
                    BountyInputField::TestnetAddress => { self.bounty_testnet_address.pop(); }
                    BountyInputField::MainnetAddress => { self.bounty_mainnet_address.pop(); }
                }
            }
            _ => {}
        }
        false
    }

    fn cycle_view(&mut self) {
        self.view_mode = match self.view_mode {
            ViewMode::Dashboard => ViewMode::Wallet,
            ViewMode::Wallet => ViewMode::FullLogs,
            ViewMode::FullLogs => ViewMode::Network,
            ViewMode::Network => ViewMode::Stats,
            ViewMode::Stats => ViewMode::Physics,
            ViewMode::Physics => ViewMode::Bounty,
            ViewMode::Bounty => ViewMode::Dashboard,
            ViewMode::Menu => ViewMode::Dashboard,
        };
    }

    fn handle_up(&mut self) {
        match self.view_mode {
            ViewMode::Menu => {
                if self.menu_selection > 0 {
                    self.menu_selection -= 1;
                }
            }
            ViewMode::FullLogs => {
                self.log_scroll = self.log_scroll.saturating_add(1);
            }
            ViewMode::Bounty => {
                // Switch between input fields
                self.bounty_input_field = BountyInputField::TestnetAddress;
            }
            _ => {}
        }
    }

    fn handle_down(&mut self) {
        match self.view_mode {
            ViewMode::Menu => {
                if self.menu_selection < 9 {  // Updated from 8 to 9 for new menu item
                    self.menu_selection += 1;
                }
            }
            ViewMode::FullLogs => {
                self.log_scroll = self.log_scroll.saturating_sub(1);
            }
            ViewMode::Bounty => {
                // Switch between input fields
                self.bounty_input_field = BountyInputField::MainnetAddress;
            }
            _ => {}
        }
    }

    fn handle_menu_selection(&mut self) {
        match self.menu_selection {
            0 => { /* Node Control - TODO */ }
            1 => self.view_mode = ViewMode::Network,
            2 => { /* Mining Status - TODO */ }
            3 => { /* Blockchain Explorer - TODO */ }
            4 => self.view_mode = ViewMode::Wallet,
            5 => self.view_mode = ViewMode::Bounty,  // Bounty Campaign Registration
            6 => { /* Performance Metrics - TODO */ }
            7 => { /* Configuration - TODO */ }
            8 => { /* Export Logs - TODO */ }
            9 => self.should_quit = true,
            _ => {}
        }
    }

    /// Called on every tick (250ms)
    pub fn on_tick(&mut self) {
        // v9.8.4: Advance update animation frame
        self.update_animation.tick();
        // v9.8.5: Advance starship sync animation
        self.starship_animation.tick();
        // v9.8.6: Advance water robots animation
        self.water_robots.tick();

        // v9.9.0: Advance Command Center animations (radar sweep, ocean creatures)
        {
            let (pc, connected) = if let Ok(m) = self.metrics.read() {
                (m.peer_count as u32, m.peer_count > 0)
            } else {
                (0, false)
            };
            self.command_center.tick(pc, connected);
        }

        if let Ok(metrics) = self.metrics.read() {
            // Update TPS history
            if let Ok(mut history) = self.tps_history.write() {
                history.push_overwrite(metrics.current_tps as f64);
            }
            // Update bandwidth history
            let bw_in = metrics.bytes_in_per_sec;
            let bw_out = metrics.bytes_out_per_sec;
            if let Ok(mut history) = self.bw_in_history.write() {
                history.push_overwrite(bw_in);
            }
            if let Ok(mut history) = self.bw_out_history.write() {
                history.push_overwrite(bw_out);
            }
            // Track peaks
            if bw_in > self.peak_bw_in {
                self.peak_bw_in = bw_in;
            }
            if bw_out > self.peak_bw_out {
                self.peak_bw_out = bw_out;
            }
        }
    }

    /// Get recent logs for display
    pub fn get_recent_logs(&self, count: usize) -> Vec<LogEntry> {
        if let Ok(logs) = self.logs.read() {
            logs.iter()
                .skip(self.log_scroll)
                .take(count)
                .cloned()
                .collect()
        } else {
            vec![]
        }
    }

    /// Get TPS history for chart
    pub fn get_tps_history(&self) -> Vec<f64> {
        if let Ok(history) = self.tps_history.read() {
            history.iter().copied().collect()
        } else {
            vec![]
        }
    }

    /// Get bandwidth in history for chart
    pub fn get_bw_in_history(&self) -> Vec<u64> {
        if let Ok(history) = self.bw_in_history.read() {
            history.iter().copied().collect()
        } else {
            vec![]
        }
    }

    /// Get bandwidth out history for chart
    pub fn get_bw_out_history(&self) -> Vec<u64> {
        if let Ok(history) = self.bw_out_history.read() {
            history.iter().copied().collect()
        } else {
            vec![]
        }
    }

    /// Submit bounty registration to API
    pub async fn submit_bounty_registration(&mut self) -> Result<String, String> {
        if self.bounty_testnet_address.is_empty() {
            return Err("Testnet address is required".to_string());
        }

        let client = reqwest::Client::new();
        let api_url = std::env::var("BOUNTY_API_URL")
            .unwrap_or_else(|_| "https://bounty.quillonq.xyz".to_string());

        let payload = serde_json::json!({
            "testnet_address": self.bounty_testnet_address,
            "mainnet_address": if self.bounty_mainnet_address.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(self.bounty_mainnet_address.clone())
            }
        });

        match client
            .post(format!("{}/v1/testnet/register", api_url))
            .json(&payload)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<serde_json::Value>().await {
                        Ok(data) => {
                            if let Some(user_id) = data.get("user_id").and_then(|v| v.as_str()) {
                                Ok(format!("✅ Registration successful! User ID: {}", user_id))
                            } else {
                                Ok("✅ Registration successful!".to_string())
                            }
                        }
                        Err(e) => Err(format!("Failed to parse response: {}", e)),
                    }
                } else {
                    Err(format!("Server returned error: {}", response.status()))
                }
            }
            Err(e) => Err(format!("Network error: {}", e)),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
