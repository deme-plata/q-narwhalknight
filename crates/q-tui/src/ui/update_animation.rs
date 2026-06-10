//! Update Animation: Wicked cool auto-update progress visualization (Node Edition)
//!
//! Extended version for the node TUI with additional states:
//! - Quorum collection (shield + signature dots)
//! - Preflight database check (spinning verification)
//! - Restart countdown (draining timer)
//! - Rollback warning (amber alert)
//!
//! Same visual language as the miner version: rainbow borders, flowing progress
//! bars, quantum sparkles, pulsing text.

use ratatui::{
    buffer::Buffer,
    style::Color,
};

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const SEED: u64 = 0x517E_CA11_40DE_F100;
const MODAL_WIDTH: u16 = 58;
const SPARKLE_CHARS: [char; 8] = ['✦', '✧', '·', '⚛', '✦', '·', '★', '✧'];
const SPINNER: [char; 8] = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];
const BAR_FILL: [char; 9] = ['█', '▉', '▊', '▋', '▌', '▍', '▎', '▏', ' '];
const SHIELD_ART: [&str; 5] = [
    "  ╔═══╗  ",
    "  ║ ⛨ ║  ",
    "  ║   ║  ",
    "  ╚═╤═╝  ",
    "    │    ",
];

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum AnimState {
    Hidden,
    WaitingForQuorum {
        version: String,
        signers: usize,
        needed: usize,
    },
    Downloading {
        version: String,
        progress: f32,
        bytes_downloaded: u64,
        bytes_total: u64,
    },
    Verifying {
        version: String,
    },
    PreflightCheck {
        version: String,
    },
    Success {
        version: String,
    },
    RestartCountdown {
        version: String,
        secs_remaining: u64,
        secs_total: u64,
    },
    Error {
        version: String,
        message: String,
    },
    RollingBack {
        version: String,
        reason: String,
    },
}

pub struct UpdateAnimation {
    frame: u16,
    state: AnimState,
    seed: u64,
    success_frame: u16,
}

impl UpdateAnimation {
    pub fn new() -> Self {
        Self {
            frame: 0,
            state: AnimState::Hidden,
            seed: SEED,
            success_frame: 0,
        }
    }

    // ─── State setters ───────────────────────────────────────────

    pub fn set_waiting_for_quorum(&mut self, version: String, signers: usize, needed: usize) {
        self.state = AnimState::WaitingForQuorum { version, signers, needed };
    }

    pub fn set_downloading(&mut self, version: String, progress: f32, bytes_downloaded: u64, bytes_total: u64) {
        self.state = AnimState::Downloading { version, progress, bytes_downloaded, bytes_total };
    }

    pub fn set_verifying(&mut self, version: String) {
        self.state = AnimState::Verifying { version };
    }

    pub fn set_preflight(&mut self, version: String) {
        self.state = AnimState::PreflightCheck { version };
    }

    pub fn set_success(&mut self, version: String) {
        if !matches!(self.state, AnimState::Success { .. }) {
            self.success_frame = 0;
        }
        self.state = AnimState::Success { version };
    }

    pub fn set_restart_countdown(&mut self, version: String, secs_remaining: u64, secs_total: u64) {
        self.state = AnimState::RestartCountdown { version, secs_remaining, secs_total };
    }

    pub fn set_error(&mut self, version: String, message: String) {
        self.state = AnimState::Error { version, message };
    }

    pub fn set_rolling_back(&mut self, version: String, reason: String) {
        self.state = AnimState::RollingBack { version, reason };
    }

    pub fn set_hidden(&mut self) {
        self.state = AnimState::Hidden;
    }

    pub fn is_visible(&self) -> bool {
        !matches!(self.state, AnimState::Hidden)
    }

    // ─── Tick ────────────────────────────────────────────────────

    pub fn tick(&mut self) {
        if self.is_visible() {
            self.frame = self.frame.wrapping_add(1);
            if matches!(self.state, AnimState::Success { .. }) {
                self.success_frame = self.success_frame.saturating_add(1);
            }
        }
    }

    // ─── Main render ─────────────────────────────────────────────

    pub fn render(&self, buf: &mut Buffer) {
        if !self.is_visible() {
            return;
        }

        let area = buf.area;
        if area.width < 40 || area.height < 12 {
            return;
        }

        match &self.state {
            AnimState::Hidden => {}
            AnimState::WaitingForQuorum { version, signers, needed } => {
                let modal = centered_rect(MODAL_WIDTH, 14, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Magenta);
                self.draw_quorum(buf, modal, version, *signers, *needed);
            }
            AnimState::Downloading { version, progress, bytes_downloaded, bytes_total } => {
                let modal = centered_rect(MODAL_WIDTH, 12, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Cyan);
                self.draw_downloading(buf, modal, version, *progress, *bytes_downloaded, *bytes_total);
            }
            AnimState::Verifying { version } => {
                let modal = centered_rect(MODAL_WIDTH, 10, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Yellow);
                self.draw_verifying(buf, modal, version);
            }
            AnimState::PreflightCheck { version } => {
                let modal = centered_rect(MODAL_WIDTH, 12, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Yellow);
                self.draw_preflight(buf, modal, version);
            }
            AnimState::Success { version } => {
                let modal = centered_rect(MODAL_WIDTH, 16, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Rainbow);
                self.draw_success(buf, modal, version);
            }
            AnimState::RestartCountdown { version, secs_remaining, secs_total } => {
                let modal = centered_rect(MODAL_WIDTH, 12, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Green);
                self.draw_restart_countdown(buf, modal, version, *secs_remaining, *secs_total);
            }
            AnimState::Error { version, message } => {
                let modal = centered_rect(MODAL_WIDTH, 10, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Red);
                self.draw_error(buf, modal, version, message);
            }
            AnimState::RollingBack { version, reason } => {
                let modal = centered_rect(MODAL_WIDTH, 12, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Amber);
                self.draw_rollback(buf, modal, version, reason);
            }
        }
    }

    // ─── Background dimming ──────────────────────────────────────

    fn dim_background(&self, buf: &mut Buffer, area: ratatui::layout::Rect) {
        let buf_area = buf.area;
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                if x < buf_area.x + buf_area.width && y < buf_area.y + buf_area.height {
                    let cell = buf.get_mut(x, y);
                    cell.set_fg(dim_color(cell.fg, 0.25));
                    cell.set_bg(Color::Rgb(8, 8, 12));
                }
            }
        }
    }

    // ─── Animated border ─────────────────────────────────────────

    fn draw_border(&self, buf: &mut Buffer, area: ratatui::layout::Rect, theme: BorderTheme) {
        let w = area.width;
        let h = area.height;
        if w < 4 || h < 3 { return; }

        let perimeter = 2 * (w as u32 + h as u32) - 4;
        let mut idx: u32 = 0;

        // Top
        for x in 0..w {
            let ch = if x == 0 { '╔' } else if x == w - 1 { '╗' } else { '═' };
            self.set_cell(buf, area.x + x, area.y, ch, border_color(&theme, idx, perimeter, self.frame));
            idx += 1;
        }
        // Right
        for y in 1..h.saturating_sub(1) {
            self.set_cell(buf, area.x + w - 1, area.y + y, '║', border_color(&theme, idx, perimeter, self.frame));
            idx += 1;
        }
        // Bottom
        for x in (0..w).rev() {
            let ch = if x == w - 1 { '╝' } else if x == 0 { '╚' } else { '═' };
            self.set_cell(buf, area.x + x, area.y + h - 1, ch, border_color(&theme, idx, perimeter, self.frame));
            idx += 1;
        }
        // Left
        for y in (1..h.saturating_sub(1)).rev() {
            self.set_cell(buf, area.x, area.y + y, '║', border_color(&theme, idx, perimeter, self.frame));
            idx += 1;
        }

        // Fill interior
        let buf_area = buf.area;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let px = area.x + x;
                let py = area.y + y;
                if px < buf_area.x + buf_area.width && py < buf_area.y + buf_area.height {
                    let cell = buf.get_mut(px, py);
                    cell.set_char(' ');
                    cell.set_fg(Color::White);
                    cell.set_bg(Color::Rgb(12, 12, 18));
                }
            }
        }
    }

    // ─── QUORUM state (node-specific) ────────────────────────────

    fn draw_quorum(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                   version: &str, signers: usize, needed: usize) {
        // Title
        let title = format!("\u{1F6E1}\u{FE0F}  COLLECTING QUORUM  {}", version);
        self.draw_text_centered(buf, area, 1, &title, Color::Magenta);

        // Shield ASCII art (pulsing)
        let pulse = ((self.frame as f32 * 0.12).sin() * 0.3 + 0.7).clamp(0.5, 1.0);
        for (i, line) in SHIELD_ART.iter().enumerate() {
            let y = area.y + 3 + i as u16;
            let x = area.x + (area.width.saturating_sub(line.len() as u16)) / 2;
            for (j, ch) in line.chars().enumerate() {
                if ch != ' ' {
                    let color = Color::Rgb(
                        (180.0 * pulse) as u8,
                        (80.0 * pulse) as u8,
                        (220.0 * pulse) as u8,
                    );
                    self.set_cell(buf, x + j as u16, y, ch, color);
                }
            }
        }

        // Signer progress dots
        let dots_y = area.y + 9;
        let signer_names = ["Beta", "Gamma", "Delta"];
        let total_w: usize = signer_names.iter().map(|n| n.len() + 4).sum::<usize>() + 8;
        let mut cx = area.x + (area.width.saturating_sub(total_w as u16)) / 2;
        for (i, name) in signer_names.iter().enumerate() {
            let signed = i < signers;
            let icon = if signed { "\u{2713}" } else { "\u{25CB}" };
            let color = if signed { Color::Green } else { Color::DarkGray };
            let label = format!("{} {}", icon, name);
            for ch in label.chars() {
                if cx < area.x + area.width - 1 {
                    self.set_cell(buf, cx, dots_y, ch, color);
                    cx += 1;
                }
            }
            cx += 4;
        }

        // Status text
        let status = format!("Waiting for {} of {} signatures...", needed, needed);
        self.draw_text_centered(buf, area, 11, &status, Color::DarkGray);
    }

    // ─── DOWNLOADING state ───────────────────────────────────────

    fn draw_downloading(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                        version: &str, progress: f32, bytes_dl: u64, bytes_total: u64) {
        let inner_w = area.width.saturating_sub(4);
        let title = format!("\u{2B07}  DOWNLOADING UPDATE {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Cyan);

        let bar_y = area.y + 5;
        let bar_x = area.x + 4;
        let bar_w = inner_w.saturating_sub(4);
        self.draw_progress_bar(buf, bar_x, bar_y, bar_w, progress);

        let pct_text = format!("{:.1}%", progress * 100.0);
        self.draw_text_centered(buf, area, 7, &pct_text, Color::White);

        let dl_mb = bytes_dl as f64 / 1_048_576.0;
        let total_mb = bytes_total as f64 / 1_048_576.0;
        let stats = if bytes_total > 0 {
            format!("{:.1} / {:.1} MB", dl_mb, total_mb)
        } else {
            format!("{:.1} MB downloaded", dl_mb)
        };
        self.draw_text_centered(buf, area, 9, &stats, Color::DarkGray);
    }

    // ─── VERIFYING state ─────────────────────────────────────────

    fn draw_verifying(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str) {
        let title = format!("\u{1F50D}  VERIFYING {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Yellow);

        let spinner_ch = SPINNER[self.frame as usize % SPINNER.len()];
        let msg = format!("{}  Checking integrity...", spinner_ch);
        self.draw_text_centered(buf, area, 5, &msg, Color::White);

        let checks_y = area.y + 7;
        let checks = [
            ("\u{2713} SHA-256", self.frame > 8),
            ("\u{2713} BLAKE3", self.frame > 16),
            ("\u{25CC} Preflight", false),
        ];
        let full_w: usize = checks.iter().map(|(s, _)| s.len() + 4).sum();
        let mut cx = area.x + (area.width.saturating_sub(full_w as u16)) / 2;
        for (label, done) in &checks {
            let color = if *done { Color::Green } else { Color::DarkGray };
            for ch in label.chars() {
                if cx < area.x + area.width - 1 {
                    self.set_cell(buf, cx, checks_y, ch, color);
                    cx += 1;
                }
            }
            cx += 4;
        }
    }

    // ─── PREFLIGHT CHECK state (node-specific) ───────────────────

    fn draw_preflight(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str) {
        let title = format!("\u{1F6E1}\u{FE0F}  PREFLIGHT CHECK  {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Yellow);

        // Animated database icon
        let db_lines = ["┌─────┐", "│ ≡≡≡ │", "│ ≡≡≡ │", "└─────┘"];
        let pulse = ((self.frame as f32 * 0.2).sin() * 0.3 + 0.7).clamp(0.5, 1.0);
        for (i, line) in db_lines.iter().enumerate() {
            let y = area.y + 4 + i as u16;
            let x = area.x + (area.width.saturating_sub(line.len() as u16)) / 2;
            for (j, ch) in line.chars().enumerate() {
                let color = Color::Rgb(
                    (200.0 * pulse) as u8,
                    (200.0 * pulse) as u8,
                    0,
                );
                self.set_cell(buf, x + j as u16, y, ch, color);
            }
        }

        let spinner_ch = SPINNER[self.frame as usize % SPINNER.len()];
        let msg = format!("{}  Verifying database integrity...", spinner_ch);
        self.draw_text_centered(buf, area, 9, &msg, Color::White);
    }

    // ─── SUCCESS state ───────────────────────────────────────────

    fn draw_success(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str) {
        // Top sparkle row
        let sparkle_y = area.y + 1;
        for x in (area.x + 8)..(area.x + area.width.saturating_sub(8)) {
            if self.sparkle(x, sparkle_y) {
                let hue = (x as f32 / area.width as f32 + self.frame as f32 * 0.05) % 1.0;
                let ch = SPARKLE_CHARS[self.hash(x as u64, sparkle_y as u64, self.frame as u64) as usize % SPARKLE_CHARS.len()];
                self.set_cell(buf, x, sparkle_y, ch, hue_to_color(hue));
            }
        }

        // Inner success box
        let box_w: u16 = 26;
        let box_x = area.x + (area.width.saturating_sub(box_w)) / 2;
        let box_y = area.y + 3;
        self.set_cell(buf, box_x, box_y, '┏', Color::Green);
        for x in 1..box_w.saturating_sub(1) {
            self.set_cell(buf, box_x + x, box_y, '━', Color::Green);
        }
        self.set_cell(buf, box_x + box_w - 1, box_y, '┓', Color::Green);
        self.set_cell(buf, box_x, box_y + 1, '┃', Color::Green);
        self.set_cell(buf, box_x + box_w - 1, box_y + 1, '┃', Color::Green);
        for x in 1..box_w.saturating_sub(1) {
            let px = box_x + x;
            let py = box_y + 1;
            let buf_area = buf.area;
            if px < buf_area.x + buf_area.width && py < buf_area.y + buf_area.height {
                let cell = buf.get_mut(px, py);
                cell.set_char(' ');
                cell.set_bg(Color::Rgb(0, 30, 0));
            }
        }
        let inner_text = "\u{2705} UPDATE VERIFIED";
        let text_x = box_x + (box_w.saturating_sub(inner_text.chars().count() as u16)) / 2;
        let pulse = ((self.frame as f32 * 0.15).sin() * 0.3 + 0.7).clamp(0.5, 1.0);
        let green = (180.0 * pulse) as u8;
        for (i, ch) in inner_text.chars().enumerate() {
            self.set_cell(buf, text_x + i as u16, box_y + 1, ch, Color::Rgb(green, 255, green));
        }
        self.set_cell(buf, box_x, box_y + 2, '┗', Color::Green);
        for x in 1..box_w.saturating_sub(1) {
            self.set_cell(buf, box_x + x, box_y + 2, '━', Color::Green);
        }
        self.set_cell(buf, box_x + box_w - 1, box_y + 2, '┛', Color::Green);

        // Version (rainbow)
        let ver_text = format!("Q-NarwhalKnight  {}", version);
        self.draw_text_rainbow(buf, area, 7, &ver_text);

        // All verification steps passed
        let checks = "\u{2713} Quorum  \u{2713} SHA-256  \u{2713} BLAKE3  \u{2713} Preflight";
        self.draw_text_centered(buf, area, 9, checks, Color::Green);

        // Divider
        let div_y = area.y + 11;
        let div_text = "\u{2500}\u{2500}\u{2500} \u{269B} \u{2500}\u{2500}\u{2500}";
        let div_x = area.x + (area.width.saturating_sub(div_text.chars().count() as u16)) / 2;
        for (i, ch) in div_text.chars().enumerate() {
            let hue = (i as f32 / div_text.len() as f32 + self.frame as f32 * 0.04) % 1.0;
            self.set_cell(buf, div_x + i as u16, div_y, ch, hue_to_color(hue));
        }

        // Restart message
        let msg = "SIGUSR1 restart will be triggered automatically";
        self.draw_text_centered(buf, area, 13, msg, Color::DarkGray);

        // Bottom sparkles
        let sparkle_y2 = area.y + area.height.saturating_sub(2);
        for x in (area.x + 8)..(area.x + area.width.saturating_sub(8)) {
            if self.sparkle(x, sparkle_y2 + 100) {
                let hue = (x as f32 / area.width as f32 + self.frame as f32 * 0.07) % 1.0;
                let ch = SPARKLE_CHARS[self.hash(x as u64, (sparkle_y2 + 100) as u64, self.frame as u64) as usize % SPARKLE_CHARS.len()];
                self.set_cell(buf, x, sparkle_y2, ch, hue_to_color(hue));
            }
        }
    }

    // ─── RESTART COUNTDOWN state (node-specific) ─────────────────

    fn draw_restart_countdown(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                              version: &str, secs_remaining: u64, secs_total: u64) {
        let title = format!("\u{23F3}  RESTARTING IN {}s", secs_remaining);
        self.draw_text_centered(buf, area, 2, &title, Color::Green);

        let subtitle = format!("Q-NarwhalKnight {}", version);
        self.draw_text_centered(buf, area, 3, &subtitle, Color::DarkGray);

        // Countdown progress bar (fills as time passes)
        let progress = if secs_total > 0 {
            1.0 - (secs_remaining as f32 / secs_total as f32)
        } else {
            1.0
        };
        let bar_y = area.y + 6;
        let bar_x = area.x + 6;
        let bar_w = area.width.saturating_sub(12);
        self.draw_progress_bar(buf, bar_x, bar_y, bar_w, progress);

        let status = "Connections draining \u{00B7} SIGUSR1 pending";
        self.draw_text_centered(buf, area, 9, status, Color::DarkGray);
    }

    // ─── ERROR state ─────────────────────────────────────────────

    fn draw_error(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str, message: &str) {
        let title = format!("\u{274C}  UPDATE ERROR  {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Red);

        let max_w = area.width.saturating_sub(6) as usize;
        let msg = if message.len() > max_w { &message[..max_w] } else { message };
        self.draw_text_centered(buf, area, 5, msg, Color::LightRed);

        let retry = "Will retry in 5 minutes";
        self.draw_text_centered(buf, area, 7, retry, Color::DarkGray);
    }

    // ─── ROLLBACK state (node-specific) ──────────────────────────

    fn draw_rollback(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                     version: &str, reason: &str) {
        // Flashing warning
        let flash = self.frame % 8 < 4;
        let title_color = if flash { Color::Yellow } else { Color::Rgb(180, 140, 0) };
        let title = format!("\u{26A0}\u{FE0F}  ROLLING BACK  {}", version);
        self.draw_text_centered(buf, area, 2, &title, title_color);

        // Reason
        let max_w = area.width.saturating_sub(6) as usize;
        let reason_text = if reason.len() > max_w { &reason[..max_w] } else { reason };
        self.draw_text_centered(buf, area, 5, reason_text, Color::Yellow);

        // Spinner + status
        let spinner_ch = SPINNER[self.frame as usize % SPINNER.len()];
        let msg = format!("{}  Restoring previous version...", spinner_ch);
        self.draw_text_centered(buf, area, 8, &msg, Color::White);

        let warn = "DO NOT restart the node manually";
        self.draw_text_centered(buf, area, 10, warn, Color::Red);
    }

    // ─── Progress bar ────────────────────────────────────────────

    fn draw_progress_bar(&self, buf: &mut Buffer, x: u16, y: u16, width: u16, progress: f32) {
        let progress = progress.clamp(0.0, 1.0);
        let buf_area = buf.area;
        let in_bounds = |px: u16, py: u16| -> bool {
            px >= buf_area.x && px < buf_area.x + buf_area.width
            && py >= buf_area.y && py < buf_area.y + buf_area.height
        };

        let lx = x.saturating_sub(1);
        if in_bounds(lx, y) {
            let cell = buf.get_mut(lx, y);
            cell.set_char('┃');
            cell.set_fg(Color::DarkGray);
        }
        let rx = x + width;
        if in_bounds(rx, y) {
            let cell = buf.get_mut(rx, y);
            cell.set_char('┃');
            cell.set_fg(Color::DarkGray);
        }

        let filled_f = width as f32 * progress;
        let filled = filled_f as u16;
        let fractional = filled_f - filled as f32;

        for i in 0..width {
            let px = x + i;
            if !in_bounds(px, y) { continue; }
            let cell = buf.get_mut(px, y);
            if i < filled {
                let hue = (i as f32 / width as f32 * 0.8 + self.frame as f32 * 0.025) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                let shimmer_pos = (self.frame as f32 * 0.3) % (width as f32 * 1.5);
                let dist_to_shimmer = (i as f32 - shimmer_pos).abs();
                let shimmer = if dist_to_shimmer < 3.0 { 1.0 - dist_to_shimmer / 3.0 } else { 0.0 };
                let bright = 1.0 + shimmer * 0.4;
                cell.set_char('█');
                cell.set_fg(Color::Rgb(
                    (r as f32 * bright).min(255.0) as u8,
                    (g as f32 * bright).min(255.0) as u8,
                    (b as f32 * bright).min(255.0) as u8,
                ));
                cell.set_bg(Color::Rgb(
                    (r as f32 * 0.15) as u8,
                    (g as f32 * 0.15) as u8,
                    (b as f32 * 0.15) as u8,
                ));
            } else if i == filled {
                let frac_idx = ((1.0 - fractional) * 8.0).min(7.0) as usize;
                let hue = (i as f32 / width as f32 * 0.8 + self.frame as f32 * 0.025) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                cell.set_char(BAR_FILL[frac_idx]);
                cell.set_fg(Color::Rgb(r, g, b));
                cell.set_bg(Color::Rgb(20, 20, 28));
            } else {
                cell.set_char('░');
                cell.set_fg(Color::Rgb(30, 30, 40));
                cell.set_bg(Color::Rgb(12, 12, 18));
            }
        }
    }

    // ─── Text helpers ────────────────────────────────────────────

    fn draw_text_centered(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                          row: u16, text: &str, color: Color) {
        let y = area.y + row;
        if y >= area.y + area.height { return; }
        let char_count = text.chars().count() as u16;
        let x = area.x + (area.width.saturating_sub(char_count)) / 2;
        for (i, ch) in text.chars().enumerate() {
            let px = x + i as u16;
            if px > area.x && px < area.x + area.width - 1 {
                self.set_cell(buf, px, y, ch, color);
            }
        }
    }

    fn draw_text_rainbow(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                         row: u16, text: &str) {
        let y = area.y + row;
        if y >= area.y + area.height { return; }
        let char_count = text.chars().count() as u16;
        let x = area.x + (area.width.saturating_sub(char_count)) / 2;
        for (i, ch) in text.chars().enumerate() {
            let px = x + i as u16;
            if px > area.x && px < area.x + area.width - 1 {
                let hue = (i as f32 / text.len() as f32 + self.frame as f32 * 0.02) % 1.0;
                self.set_cell(buf, px, y, ch, hue_to_color(hue));
            }
        }
    }

    fn set_cell(&self, buf: &mut Buffer, x: u16, y: u16, ch: char, fg: Color) {
        let area = buf.area;
        if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
            let cell = buf.get_mut(x, y);
            cell.set_char(ch);
            cell.set_fg(fg);
            cell.set_bg(Color::Rgb(12, 12, 18));
        }
    }

    // ─── Sparkle / hash ──────────────────────────────────────────

    fn sparkle(&self, x: u16, y: u16) -> bool {
        let h = self.hash(x as u64, y as u64, self.frame as u64);
        h % 5 == 0
    }

    fn hash(&self, x: u64, y: u64, frame: u64) -> u64 {
        let mut h = self.seed;
        h = h.wrapping_mul(6364136223846793005).wrapping_add(x);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(y);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(frame);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }
}

// ═══════════════════════════════════════════════════════════════════
// Border Theme
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
enum BorderTheme {
    Rainbow,
    Cyan,
    Yellow,
    Green,
    Red,
    Magenta,
    Amber,
}

fn border_color(theme: &BorderTheme, idx: u32, perimeter: u32, frame: u16) -> Color {
    let t = idx as f32 / perimeter as f32;
    let flow = frame as f32 * 0.04;

    match theme {
        BorderTheme::Rainbow => {
            let hue = (t + flow) % 1.0;
            hue_to_color(hue)
        }
        BorderTheme::Cyan => {
            let wave = ((t * 8.0 + flow).sin() * 0.3 + 0.7).clamp(0.4, 1.0);
            Color::Rgb(0, (180.0 * wave) as u8, (220.0 * wave) as u8)
        }
        BorderTheme::Yellow => {
            let wave = ((t * 6.0 + flow).sin() * 0.3 + 0.7).clamp(0.4, 1.0);
            Color::Rgb((220.0 * wave) as u8, (200.0 * wave) as u8, 0)
        }
        BorderTheme::Green => {
            let wave = ((t * 6.0 + flow).sin() * 0.3 + 0.7).clamp(0.4, 1.0);
            Color::Rgb(0, (220.0 * wave) as u8, (100.0 * wave) as u8)
        }
        BorderTheme::Red => {
            let wave = ((t * 6.0 + flow).sin() * 0.3 + 0.7).clamp(0.4, 1.0);
            Color::Rgb((220.0 * wave) as u8, (40.0 * wave) as u8, (40.0 * wave) as u8)
        }
        BorderTheme::Magenta => {
            let wave = ((t * 8.0 + flow).sin() * 0.3 + 0.7).clamp(0.4, 1.0);
            Color::Rgb((200.0 * wave) as u8, (60.0 * wave) as u8, (220.0 * wave) as u8)
        }
        BorderTheme::Amber => {
            let wave = ((t * 10.0 + flow).sin() * 0.4 + 0.6).clamp(0.3, 1.0);
            Color::Rgb((255.0 * wave) as u8, (180.0 * wave) as u8, 0)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Color Utilities
// ═══════════════════════════════════════════════════════════════════

fn hue_to_color(hue: f32) -> Color {
    let (r, g, b) = hue_to_rgb(hue);
    Color::Rgb(r, g, b)
}

fn hue_to_rgb(hue: f32) -> (u8, u8, u8) {
    let h = (hue.fract() + 1.0).fract() * 6.0;
    let c: u8 = 255;
    let x = (255.0 * (1.0 - (h % 2.0 - 1.0).abs())) as u8;
    match h as u32 % 6 {
        0 => (c, x, 0),
        1 => (x, c, 0),
        2 => (0, c, x),
        3 => (0, x, c),
        4 => (x, 0, c),
        _ => (c, 0, x),
    }
}

fn dim_color(c: Color, factor: f32) -> Color {
    let (r, g, b) = color_to_rgb(c);
    let f = factor.clamp(0.0, 1.0);
    Color::Rgb(
        (r as f32 * f) as u8,
        (g as f32 * f) as u8,
        (b as f32 * f) as u8,
    )
}

fn color_to_rgb(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::White => (220, 220, 220),
        Color::Cyan => (0, 255, 255),
        Color::Green => (0, 200, 0),
        Color::Yellow => (255, 255, 0),
        Color::Red => (255, 0, 0),
        Color::LightRed => (255, 128, 128),
        Color::Magenta => (255, 0, 255),
        Color::Blue => (0, 0, 255),
        Color::Gray => (128, 128, 128),
        Color::DarkGray => (64, 64, 64),
        Color::Black => (0, 0, 0),
        Color::Reset => (180, 180, 180),
        _ => (180, 180, 180),
    }
}

fn centered_rect(width: u16, height: u16, area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let w = width.min(area.width.saturating_sub(2));
    let h = height.min(area.height.saturating_sub(2));
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    ratatui::layout::Rect::new(x, y, w, h)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animation_lifecycle() {
        let mut anim = UpdateAnimation::new();
        assert!(!anim.is_visible());

        anim.set_waiting_for_quorum("v9.8.4".into(), 1, 2);
        assert!(anim.is_visible());

        anim.set_downloading("v9.8.4".into(), 0.5, 1000, 2000);
        assert!(anim.is_visible());

        anim.set_verifying("v9.8.4".into());
        anim.set_preflight("v9.8.4".into());
        anim.set_success("v9.8.4".into());
        assert!(anim.is_visible());

        anim.set_restart_countdown("v9.8.4".into(), 20, 30);
        assert!(anim.is_visible());

        anim.set_hidden();
        assert!(!anim.is_visible());
    }

    #[test]
    fn test_rollback_state() {
        let mut anim = UpdateAnimation::new();
        anim.set_rolling_back("v9.8.4".into(), "Watchdog timeout".into());
        assert!(anim.is_visible());
    }

    #[test]
    fn test_centered_rect() {
        let area = ratatui::layout::Rect::new(0, 0, 100, 40);
        let modal = centered_rect(58, 12, area);
        assert_eq!(modal.width, 58);
        assert_eq!(modal.height, 12);
        assert_eq!(modal.x, 21);
        assert_eq!(modal.y, 14);
    }

    #[test]
    fn test_all_border_themes() {
        for frame in 0..50u16 {
            let _ = border_color(&BorderTheme::Rainbow, 0, 100, frame);
            let _ = border_color(&BorderTheme::Magenta, 50, 100, frame);
            let _ = border_color(&BorderTheme::Amber, 99, 100, frame);
            let _ = border_color(&BorderTheme::Cyan, 25, 100, frame);
            let _ = border_color(&BorderTheme::Green, 75, 100, frame);
        }
    }

    #[test]
    fn test_tick_advances() {
        let mut anim = UpdateAnimation::new();
        anim.set_downloading("v1".into(), 0.0, 0, 100);
        assert_eq!(anim.frame, 0);
        anim.tick();
        assert_eq!(anim.frame, 1);
        anim.tick();
        assert_eq!(anim.frame, 2);
    }
}
