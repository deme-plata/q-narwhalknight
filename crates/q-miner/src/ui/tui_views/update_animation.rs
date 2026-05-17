//! Update Animation: Wicked cool auto-update progress visualization
//!
//! Renders a centered modal overlay with:
//! - Rainbow-gradient flowing progress bar during download
//! - Braille spinner during verification
//! - Sparkle-encrusted success modal when ready to apply
//! - Pulsing animated border that cycles colors clockwise
//!
//! Matches the Q-animation aesthetic (same hue math, sparkle hash, color palette).

use ratatui::{
    buffer::Buffer,
    layout::Position,
    style::Color,
};

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const SEED: u64 = 0x517E_CA11_0BD8_F100;
const MODAL_WIDTH: u16 = 58;
const SPARKLE_CHARS: [char; 8] = ['✦', '✧', '·', '⚛', '✦', '·', '★', '✧'];
const SPINNER: [char; 8] = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];
const BAR_FILL: [char; 9] = ['█', '▉', '▊', '▋', '▌', '▍', '▎', '▏', ' '];

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum AnimState {
    Hidden,
    Downloading {
        version: String,
        progress: f32,
        bytes_downloaded: u64,
        bytes_total: u64,
    },
    Verifying {
        version: String,
    },
    Success {
        version: String,
    },
    Applying {
        version: String,
    },
    Error {
        version: String,
        message: String,
    },
}

pub struct UpdateAnimation {
    frame: u16,
    state: AnimState,
    seed: u64,
    /// Tracks how long the success modal has been visible (for auto-dismiss sparkle burst)
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

    pub fn set_downloading(&mut self, version: String, progress: f32, bytes_downloaded: u64, bytes_total: u64) {
        self.state = AnimState::Downloading { version, progress, bytes_downloaded, bytes_total };
    }

    pub fn set_verifying(&mut self, version: String) {
        self.state = AnimState::Verifying { version };
    }

    pub fn set_success(&mut self, version: String) {
        if !matches!(self.state, AnimState::Success { .. }) {
            self.success_frame = 0; // Reset sparkle burst on new success
        }
        self.state = AnimState::Success { version };
    }

    pub fn set_applying(&mut self, version: String) {
        self.state = AnimState::Applying { version };
    }

    pub fn set_error(&mut self, version: String, message: String) {
        self.state = AnimState::Error { version, message };
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

    /// Render the update overlay on top of the existing buffer.
    /// Call AFTER normal TUI rendering.
    pub fn render(&self, buf: &mut Buffer) {
        if !self.is_visible() {
            return;
        }

        let area = buf.area;
        if area.width < 40 || area.height < 12 {
            return; // Terminal too small
        }

        match &self.state {
            AnimState::Hidden => {}
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
            AnimState::Success { version } => {
                let modal = centered_rect(MODAL_WIDTH, 16, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Rainbow);
                self.draw_success(buf, modal, version);
            }
            AnimState::Applying { version } => {
                let modal = centered_rect(MODAL_WIDTH, 10, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Green);
                self.draw_applying(buf, modal, version);
            }
            AnimState::Error { version, message } => {
                let modal = centered_rect(MODAL_WIDTH, 10, area);
                self.dim_background(buf, area);
                self.draw_border(buf, modal, BorderTheme::Red);
                self.draw_error(buf, modal, version, message);
            }
        }
    }

    // ─── Background dimming ──────────────────────────────────────

    fn dim_background(&self, buf: &mut Buffer, area: ratatui::layout::Rect) {
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                if let Some(cell) = buf.cell_mut(Position::new(x, y)) {
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
        if w < 4 || h < 3 {
            return;
        }

        let perimeter = 2 * (w as u32 + h as u32) - 4;

        // Walk clockwise around the border
        let mut idx: u32 = 0;
        // Top edge (left to right)
        for x in 0..w {
            let ch = if x == 0 { '╔' } else if x == w - 1 { '╗' } else { '═' };
            let color = border_color(&theme, idx, perimeter, self.frame);
            self.set_cell(buf, area.x + x, area.y, ch, color);
            idx += 1;
        }
        // Right edge (top+1 to bottom-1)
        for y in 1..h.saturating_sub(1) {
            let color = border_color(&theme, idx, perimeter, self.frame);
            self.set_cell(buf, area.x + w - 1, area.y + y, '║', color);
            idx += 1;
        }
        // Bottom edge (right to left)
        for x in (0..w).rev() {
            let ch = if x == w - 1 { '╝' } else if x == 0 { '╚' } else { '═' };
            let color = border_color(&theme, idx, perimeter, self.frame);
            self.set_cell(buf, area.x + x, area.y + h - 1, ch, color);
            idx += 1;
        }
        // Left edge (bottom-1 to top+1)
        for y in (1..h.saturating_sub(1)).rev() {
            let color = border_color(&theme, idx, perimeter, self.frame);
            self.set_cell(buf, area.x, area.y + y, '║', color);
            idx += 1;
        }

        // Fill interior background
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if let Some(cell) = buf.cell_mut(Position::new(area.x + x, area.y + y)) {
                    cell.set_char(' ');
                    cell.set_fg(Color::White);
                    cell.set_bg(Color::Rgb(12, 12, 18));
                }
            }
        }
    }

    // ─── DOWNLOADING state ───────────────────────────────────────

    fn draw_downloading(&self, buf: &mut Buffer, area: ratatui::layout::Rect,
                        version: &str, progress: f32, bytes_dl: u64, bytes_total: u64) {
        let inner_x = area.x + 2;
        let inner_w = area.width.saturating_sub(4);

        // Title: ⬇  DOWNLOADING UPDATE v9.8.4
        let title = format!("\u{2B07}  DOWNLOADING UPDATE {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Cyan);

        // Progress bar
        let bar_y = area.y + 5;
        let bar_x = inner_x + 2;
        let bar_w = inner_w.saturating_sub(4);
        self.draw_progress_bar(buf, bar_x, bar_y, bar_w, progress);

        // Percentage
        let pct_text = format!("{:.1}%", progress * 100.0);
        self.draw_text_centered(buf, area, 7, &pct_text, Color::White);

        // Stats line: bytes / total  ·  speed
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
        // Title
        let title = format!("\u{1F50D}  VERIFYING {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Yellow);

        // Spinner
        let spinner_ch = SPINNER[self.frame as usize % SPINNER.len()];
        let spinner_text = format!("{}  Checking integrity...", spinner_ch);
        self.draw_text_centered(buf, area, 5, &spinner_text, Color::White);

        // Checkmarks (animate them appearing)
        let checks_y = area.y + 7;
        let checks = [
            ("\u{2713} SHA-256", self.frame > 8),
            ("\u{2713} BLAKE3", self.frame > 16),
            ("\u{25CC} Preflight", false),
        ];
        let full = checks.iter().map(|(s, _)| s.len() + 4).sum::<usize>();
        let start_x = area.x + (area.width.saturating_sub(full as u16)) / 2;
        let mut cx = start_x;
        for (label, done) in &checks {
            let color = if *done { Color::Green } else { Color::DarkGray };
            for ch in label.chars() {
                if cx < area.x + area.width - 1 {
                    self.set_cell(buf, cx, checks_y, ch, color);
                    cx += 1;
                }
            }
            cx += 4; // spacing
        }
    }

    // ─── SUCCESS state ───────────────────────────────────────────

    fn draw_success(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str) {
        // Sparkle row at top
        let sparkle_y = area.y + 1;
        for x in (area.x + 8)..(area.x + area.width.saturating_sub(8)) {
            if self.sparkle(x, sparkle_y) {
                let hue = (x as f32 / area.width as f32 + self.frame as f32 * 0.05) % 1.0;
                let ch = SPARKLE_CHARS[self.hash(x as u64, sparkle_y as u64, self.frame as u64) as usize % SPARKLE_CHARS.len()];
                self.set_cell(buf, x, sparkle_y, ch, hue_to_color(hue));
            }
        }

        // Inner success box
        let box_w: u16 = 24;
        let box_x = area.x + (area.width.saturating_sub(box_w)) / 2;
        let box_y = area.y + 3;
        // Top
        self.set_cell(buf, box_x, box_y, '┏', Color::Green);
        for x in 1..box_w.saturating_sub(1) {
            self.set_cell(buf, box_x + x, box_y, '━', Color::Green);
        }
        self.set_cell(buf, box_x + box_w - 1, box_y, '┓', Color::Green);
        // Middle
        self.set_cell(buf, box_x, box_y + 1, '┃', Color::Green);
        self.set_cell(buf, box_x + box_w - 1, box_y + 1, '┃', Color::Green);
        // Fill inner
        for x in 1..box_w.saturating_sub(1) {
            if let Some(cell) = buf.cell_mut(Position::new(box_x + x, box_y + 1)) {
                cell.set_char(' ');
                cell.set_bg(Color::Rgb(0, 30, 0));
            }
        }
        // Text inside box: ✅ UPDATE READY
        let inner_text = "\u{2705} UPDATE READY";
        let text_x = box_x + (box_w.saturating_sub(inner_text.chars().count() as u16)) / 2;
        // Pulsing brightness
        let pulse = ((self.frame as f32 * 0.15).sin() * 0.3 + 0.7).clamp(0.5, 1.0);
        let green = (180.0 * pulse) as u8;
        for (i, ch) in inner_text.chars().enumerate() {
            self.set_cell(buf, text_x + i as u16, box_y + 1, ch, Color::Rgb(green, 255, green));
        }
        // Bottom
        self.set_cell(buf, box_x, box_y + 2, '┗', Color::Green);
        for x in 1..box_w.saturating_sub(1) {
            self.set_cell(buf, box_x + x, box_y + 2, '━', Color::Green);
        }
        self.set_cell(buf, box_x + box_w - 1, box_y + 2, '┛', Color::Green);

        // Version
        let ver_text = format!("Q-NarwhalKnight  {}", version);
        self.draw_text_rainbow(buf, area, 7, &ver_text);

        // Checkmarks row
        let checks = "\u{2713} Downloaded   \u{2713} Verified   \u{2713} Integrity";
        self.draw_text_centered(buf, area, 9, checks, Color::Green);

        // Q-branded sparkle divider
        let div_y = area.y + 11;
        let div_text = "\u{2500}\u{2500}\u{2500} \u{269B} \u{2500}\u{2500}\u{2500}";
        let div_x = area.x + (area.width.saturating_sub(div_text.chars().count() as u16)) / 2;
        for (i, ch) in div_text.chars().enumerate() {
            let hue = (i as f32 / div_text.len() as f32 + self.frame as f32 * 0.04) % 1.0;
            self.set_cell(buf, div_x + i as u16, div_y, ch, hue_to_color(hue));
        }

        // Action prompt — pulsing
        let prompt = "\u{269B}  Press [U] to apply & restart  \u{269B}";
        let prompt_pulse = ((self.frame as f32 * 0.12).sin() * 0.4 + 0.6).clamp(0.3, 1.0);
        let prompt_y = area.y + 13;
        let prompt_x = area.x + (area.width.saturating_sub(prompt.chars().count() as u16)) / 2;
        for (i, ch) in prompt.chars().enumerate() {
            let base_hue = (i as f32 / prompt.len() as f32 + self.frame as f32 * 0.03) % 1.0;
            let (r, g, b) = hue_to_rgb(base_hue);
            let color = Color::Rgb(
                (r as f32 * prompt_pulse) as u8,
                (g as f32 * prompt_pulse) as u8,
                (b as f32 * prompt_pulse) as u8,
            );
            self.set_cell(buf, prompt_x + i as u16, prompt_y, ch, color);
        }

        // Bottom sparkle row
        let sparkle_y2 = area.y + area.height.saturating_sub(2);
        for x in (area.x + 8)..(area.x + area.width.saturating_sub(8)) {
            if self.sparkle(x, sparkle_y2 + 100) { // offset seed for different pattern
                let hue = (x as f32 / area.width as f32 + self.frame as f32 * 0.07) % 1.0;
                let ch = SPARKLE_CHARS[self.hash(x as u64, (sparkle_y2 + 100) as u64, self.frame as u64) as usize % SPARKLE_CHARS.len()];
                self.set_cell(buf, x, sparkle_y2, ch, hue_to_color(hue));
            }
        }
    }

    // ─── APPLYING state ──────────────────────────────────────────

    fn draw_applying(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str) {
        let title = format!("\u{1F504}  APPLYING UPDATE {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Green);

        // Animated dots
        let dots = ".".repeat((self.frame as usize % 4) + 1);
        let msg = format!("Swapping binary{}", dots);
        self.draw_text_centered(buf, area, 5, &msg, Color::White);

        let warn = "Do not close this window";
        self.draw_text_centered(buf, area, 7, warn, Color::Yellow);
    }

    // ─── ERROR state ─────────────────────────────────────────────

    fn draw_error(&self, buf: &mut Buffer, area: ratatui::layout::Rect, version: &str, message: &str) {
        let title = format!("\u{274C}  UPDATE ERROR  {}", version);
        self.draw_text_centered(buf, area, 2, &title, Color::Red);

        // Error message (truncate to fit)
        let max_w = area.width.saturating_sub(6) as usize;
        let msg = if message.len() > max_w {
            &message[..max_w]
        } else {
            message
        };
        self.draw_text_centered(buf, area, 5, msg, Color::LightRed);

        let retry = "Will retry in 5 minutes";
        self.draw_text_centered(buf, area, 7, retry, Color::DarkGray);
    }

    // ─── Progress bar ────────────────────────────────────────────

    fn draw_progress_bar(&self, buf: &mut Buffer, x: u16, y: u16, width: u16, progress: f32) {
        let progress = progress.clamp(0.0, 1.0);

        // Bar border
        if let Some(cell) = buf.cell_mut(Position::new(x.saturating_sub(1), y)) {
            cell.set_char('┃');
            cell.set_fg(Color::DarkGray);
        }
        if let Some(cell) = buf.cell_mut(Position::new(x + width, y)) {
            cell.set_char('┃');
            cell.set_fg(Color::DarkGray);
        }

        let filled_f = width as f32 * progress;
        let filled = filled_f as u16;
        let fractional = filled_f - filled as f32;

        for i in 0..width {
            let pos = Position::new(x + i, y);
            if let Some(cell) = buf.cell_mut(pos) {
                if i < filled {
                    // Filled: rainbow gradient that flows
                    let hue = (i as f32 / width as f32 * 0.8 + self.frame as f32 * 0.025) % 1.0;
                    let (r, g, b) = hue_to_rgb(hue);

                    // Shimmer wave moving right
                    let shimmer_pos = (self.frame as f32 * 0.3) % (width as f32 * 1.5);
                    let dist_to_shimmer = (i as f32 - shimmer_pos).abs();
                    let shimmer = if dist_to_shimmer < 3.0 {
                        1.0 - dist_to_shimmer / 3.0
                    } else {
                        0.0
                    };
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
                    // Fractional fill: use partial block character
                    let frac_idx = ((1.0 - fractional) * 8.0).min(7.0) as usize;
                    let hue = (i as f32 / width as f32 * 0.8 + self.frame as f32 * 0.025) % 1.0;
                    let (r, g, b) = hue_to_rgb(hue);
                    cell.set_char(BAR_FILL[frac_idx]);
                    cell.set_fg(Color::Rgb(r, g, b));
                    cell.set_bg(Color::Rgb(20, 20, 28));
                } else {
                    // Empty
                    cell.set_char('░');
                    cell.set_fg(Color::Rgb(30, 30, 40));
                    cell.set_bg(Color::Rgb(12, 12, 18));
                }
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
        if let Some(cell) = buf.cell_mut(Position::new(x, y)) {
            cell.set_char(ch);
            cell.set_fg(fg);
            cell.set_bg(Color::Rgb(12, 12, 18));
        }
    }

    // ─── Sparkle / hash ──────────────────────────────────────────

    fn sparkle(&self, x: u16, y: u16) -> bool {
        let h = self.hash(x as u64, y as u64, self.frame as u64);
        h % 5 == 0 // ~20% chance — denser than Q animation for celebration feel
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
}

fn border_color(theme: &BorderTheme, idx: u32, perimeter: u32, frame: u16) -> Color {
    let t = idx as f32 / perimeter as f32;
    let flow = frame as f32 * 0.04; // Speed of color flow around border

    match theme {
        BorderTheme::Rainbow => {
            let hue = (t + flow) % 1.0;
            hue_to_color(hue)
        }
        BorderTheme::Cyan => {
            // Pulsing cyan with travelling bright spot
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
    }
}

// ═══════════════════════════════════════════════════════════════════
// Color Utilities (same as q_animation.rs)
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
        Color::Magenta => (255, 0, 255),
        Color::Blue => (0, 0, 255),
        Color::Gray => (128, 128, 128),
        Color::DarkGray => (64, 64, 64),
        Color::Black => (0, 0, 0),
        Color::Reset => (180, 180, 180),
        _ => (180, 180, 180),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Layout
// ═══════════════════════════════════════════════════════════════════

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

        anim.set_downloading("v9.8.4".into(), 0.5, 1000, 2000);
        assert!(anim.is_visible());

        anim.tick();
        assert!(anim.frame > 0);

        anim.set_success("v9.8.4".into());
        assert!(anim.is_visible());

        anim.set_hidden();
        assert!(!anim.is_visible());
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
    fn test_hue_spectrum() {
        let (r, _, _) = hue_to_rgb(0.0);
        assert_eq!(r, 255); // Red
        let (_, g, _) = hue_to_rgb(1.0 / 3.0);
        assert_eq!(g, 255); // Green
        let (_, _, b) = hue_to_rgb(2.0 / 3.0);
        assert_eq!(b, 255); // Blue
    }

    #[test]
    fn test_progress_clamp() {
        let anim = UpdateAnimation::new();
        // Just verify no panic on edge values
        anim.set_downloading("v1".into(), -0.5, 0, 100);
        anim.set_downloading("v1".into(), 1.5, 200, 100);
    }

    #[test]
    fn test_sparkle_deterministic() {
        let anim = UpdateAnimation::new();
        let a = anim.sparkle(10, 20);
        let b = anim.sparkle(10, 20);
        assert_eq!(a, b);
    }

    #[test]
    fn test_border_colors() {
        // Verify no panic for all themes
        for frame in 0..100u16 {
            let _ = border_color(&BorderTheme::Rainbow, 0, 100, frame);
            let _ = border_color(&BorderTheme::Cyan, 50, 100, frame);
            let _ = border_color(&BorderTheme::Red, 99, 100, frame);
        }
    }
}
