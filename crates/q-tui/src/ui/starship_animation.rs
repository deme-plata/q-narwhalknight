//! Starship Sync Animation: Full-screen rocket launch & orbital insertion visualization
//!
//! Node-side version for q-tui. Uses ratatui 0.25 API (buf.get_mut(x, y) with bounds checking).
//!
//! Renders an immersive animated overlay during node sync with 7 phases:
//!   Prelaunch → Ignition → SuperHeavy → HotStaging → StarshipCruise → OrbitalInsertion → StationKeeping

use ratatui::{
    buffer::Buffer,
    style::Color,
};

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const SEED: u64 = 0x57A4_5419_BEEF_CA5E;
const SPARKLE_CHARS: [char; 8] = ['✦', '✧', '·', '⚛', '✦', '·', '★', '✧'];
const STAR_CHARS: [char; 6] = ['·', '∙', '•', '✦', '★', '✧'];
const STAR_DIM: [char; 4] = ['.', '·', '∙', ','];

const ROCKET: [&str; 9] = [
    "    /\\    ",
    "   /  \\   ",
    "  | QN |  ",
    "  | KT |  ",
    "  |    |  ",
    " /|    |\\ ",
    "/ |    | \\",
    "  |/\\/\\|  ",
    "   \\  /   ",
];
const ROCKET_WIDTH: u16 = 10;

const FLAMES: [&str; 4] = [
    "  \\\\||//  ",
    "   \\|||/  ",
    "  //||\\\\  ",
    "   /||\\   ",
];
const FLAME_TIPS: [&str; 4] = [
    "   \\::/   ",
    "    \\/    ",
    "   /::\\   ",
    "    /\\    ",
];

const HORIZON_CHARS: [char; 6] = ['▁', '▂', '▃', '▄', '▅', '▆'];

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarshipPhase {
    Hidden,
    Prelaunch,
    Ignition,
    SuperHeavy,
    HotStaging,
    StarshipCruise,
    OrbitalInsertion,
    StationKeeping,
}

impl StarshipPhase {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Prelaunch" => Self::Prelaunch,
            "Ignition" => Self::Ignition,
            "SuperHeavy" => Self::SuperHeavy,
            "HotStaging" => Self::HotStaging,
            "StarshipCruise" => Self::StarshipCruise,
            "OrbitalInsertion" => Self::OrbitalInsertion,
            "StationKeeping" => Self::StationKeeping,
            _ => Self::SuperHeavy,
        }
    }
}

#[derive(Clone)]
struct Star {
    x: f32,
    y: f32,
    speed: f32,
    brightness: u8,
    char_idx: usize,
}

pub struct StarshipAnimation {
    frame: u16,
    phase: StarshipPhase,
    seed: u64,
    pub sync_progress: f32,
    pub sync_speed: f32,
    pub local_height: u64,
    pub network_height: u64,
    pub blocks_behind: u64,
    pub peer_count: u64,
    pub eta_secs: u64,
    pub mission_elapsed: u64,
    pub orbit_stable: bool,
    stars: Vec<Star>,
    stars_initialized: bool,
}

impl StarshipAnimation {
    pub fn new() -> Self {
        Self {
            frame: 0,
            phase: StarshipPhase::Hidden,
            seed: SEED,
            sync_progress: 0.0,
            sync_speed: 0.0,
            local_height: 0,
            network_height: 0,
            blocks_behind: 0,
            peer_count: 0,
            eta_secs: 0,
            mission_elapsed: 0,
            orbit_stable: false,
            stars: Vec::new(),
            stars_initialized: false,
        }
    }

    pub fn set_phase(&mut self, phase_str: &str) {
        let new_phase = StarshipPhase::from_str(phase_str);
        if new_phase != self.phase {
            self.phase = new_phase;
            if matches!(new_phase, StarshipPhase::HotStaging | StarshipPhase::StationKeeping) {
                self.stars_initialized = false;
            }
            if !self.stars_initialized {
                self.init_stars(120, 40);
            }
        }
    }

    pub fn set_hidden(&mut self) {
        self.phase = StarshipPhase::Hidden;
    }

    pub fn is_visible(&self) -> bool {
        self.phase != StarshipPhase::Hidden
    }

    pub fn tick(&mut self) {
        if !self.is_visible() { return; }
        self.frame = self.frame.wrapping_add(1);

        let speed_mult = match self.phase {
            StarshipPhase::Prelaunch => 0.0,
            StarshipPhase::Ignition => 0.3,
            StarshipPhase::SuperHeavy => 1.0,
            StarshipPhase::HotStaging => 0.6,
            StarshipPhase::StarshipCruise => 1.5,
            StarshipPhase::OrbitalInsertion => 0.8,
            StarshipPhase::StationKeeping => 0.15,
            StarshipPhase::Hidden => 0.0,
        };

        for star in &mut self.stars {
            star.y += star.speed * speed_mult * 0.02;
            if star.y > 1.1 {
                star.y = -0.1;
                let h = hash_raw(SEED, star.x.to_bits() as u64, star.brightness as u64, 0);
                star.x = (h % 1000) as f32 / 1000.0;
            }
        }
    }

    fn init_stars(&mut self, w: u16, h: u16) {
        if self.stars_initialized { return; }
        self.stars.clear();
        let num_stars = ((w as u32 * h as u32) / 12).min(400) as usize;
        for i in 0..num_stars {
            let h1 = hash_raw(self.seed, i as u64, 0, 0);
            let h2 = hash_raw(self.seed, i as u64, 1, 0);
            let h3 = hash_raw(self.seed, i as u64, 2, 0);
            self.stars.push(Star {
                x: (h1 % 1000) as f32 / 1000.0,
                y: (h2 % 1000) as f32 / 1000.0,
                speed: 0.2 + (h3 % 80) as f32 / 100.0,
                brightness: (80 + (h1 % 175)) as u8,
                char_idx: (h2 % STAR_CHARS.len() as u64) as usize,
            });
        }
        self.stars_initialized = true;
    }

    pub fn render(&self, buf: &mut Buffer) {
        if !self.is_visible() { return; }
        let area = buf.area;
        let w = area.width;
        let h = area.height;
        if w < 40 || h < 15 { return; }

        match self.phase {
            StarshipPhase::Hidden => {}
            StarshipPhase::Prelaunch => self.render_prelaunch(buf, w, h),
            StarshipPhase::Ignition => self.render_ignition(buf, w, h),
            StarshipPhase::SuperHeavy => self.render_superheavy(buf, w, h),
            StarshipPhase::HotStaging => self.render_hotstaging(buf, w, h),
            StarshipPhase::StarshipCruise => self.render_cruise(buf, w, h),
            StarshipPhase::OrbitalInsertion => self.render_orbital(buf, w, h),
            StarshipPhase::StationKeeping => self.render_stationkeeping(buf, w, h),
        }
    }

    // ─── Phase renderers ──────────────────────────────────────────

    fn render_prelaunch(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.85);
        self.draw_starfield(buf, w, h, 0.3);

        let rocket_x = (w.saturating_sub(ROCKET_WIDTH)) / 2;
        let rocket_y = h.saturating_sub(ROCKET.len() as u16 + 4);
        self.draw_rocket(buf, rocket_x, rocket_y, w, h, false);

        // Launch pad
        let pad_y = rocket_y + ROCKET.len() as u16;
        for x in rocket_x.saturating_sub(3)..=(rocket_x + ROCKET_WIDTH + 2).min(w - 1) {
            self.set_cell(buf, x, pad_y, '▀', Color::Rgb(80, 80, 80));
        }

        // Countdown
        let countdown = if self.mission_elapsed < 10 { 10u64.saturating_sub(self.mission_elapsed) } else { 0 };
        let cd_text = format!("T-{}", countdown);
        self.draw_text_centered(buf, w, 2, &cd_text, Color::Rgb(255, 120, 0));

        self.draw_phase_header(buf, w, "\u{1F4E1} PRELAUNCH", Color::Rgb(120, 120, 120));
        self.draw_telemetry_footer(buf, w, h);
    }

    fn render_ignition(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.9);
        self.draw_starfield(buf, w, h, 0.4);

        let rise = (self.frame as f32 * 0.3).min(8.0) as u16;
        let rocket_x = (w.saturating_sub(ROCKET_WIDTH)) / 2;
        let rocket_y = h.saturating_sub(ROCKET.len() as u16 + 2 + rise);
        self.draw_rocket(buf, rocket_x, rocket_y, w, h, true);
        self.draw_flame_plume(buf, rocket_x, rocket_y + ROCKET.len() as u16, w, h, 1.0);

        let g_force = 1.0 + (self.frame as f32 * 0.15).min(3.0);
        self.draw_text_at(buf, 3, 4, &format!("{:.1}G", g_force), Color::Yellow);

        self.draw_phase_header(buf, w, "\u{1F525} IGNITION", Color::Red);
        self.draw_telemetry_footer(buf, w, h);
    }

    fn render_superheavy(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.95);
        self.draw_starfield(buf, w, h, 1.0);
        self.draw_speed_lines(buf, w, h);

        let rocket_x = (w.saturating_sub(ROCKET_WIDTH)) / 2;
        let rocket_y = (h.saturating_sub(ROCKET.len() as u16)) / 2 - 2;
        self.draw_rocket(buf, rocket_x, rocket_y, w, h, true);
        self.draw_flame_plume(buf, rocket_x, rocket_y + ROCKET.len() as u16, w, h, 0.6);

        self.draw_progress_arc(buf, w, 1, self.sync_progress / 100.0);
        self.draw_text_at(buf, 3, 3, &format!("\u{26A1} {:.0} blk/s", self.sync_speed), Color::Cyan);
        self.draw_text_at(buf, 3, 4, &format!("\u{2191} #{}", format_commas(self.local_height)), Color::White);

        self.draw_phase_header(buf, w, "\u{1F680} SUPERHEAVY BOOST", Color::Cyan);
        self.draw_telemetry_footer(buf, w, h);
    }

    fn render_hotstaging(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.92);
        self.draw_starfield(buf, w, h, 0.5);

        // Separation flash
        if self.frame % 120 < 8 {
            let flash = 1.0 - (self.frame % 120) as f32 / 8.0;
            let bright = (255.0 * flash) as u8;
            let cx = w / 2;
            let cy = h / 2;
            let radius = (self.frame % 120) as f32 * 2.0;
            for step in 0..36 {
                let angle = step as f32 * std::f32::consts::PI / 18.0;
                let px = (cx as f32 + angle.cos() * radius) as u16;
                let py = (cy as f32 + angle.sin() * radius / 2.2) as u16;
                if px < w && py < h {
                    self.set_cell(buf, px, py, '✦', Color::Rgb(bright, bright, bright));
                }
            }
        }

        self.draw_debris(buf, w, h);
        let rocket_x = (w.saturating_sub(ROCKET_WIDTH)) / 2;
        let rocket_y = h / 3;
        self.draw_rocket(buf, rocket_x, rocket_y, w, h, true);
        self.draw_flame_plume(buf, rocket_x, rocket_y + ROCKET.len() as u16, w, h, 0.4);

        self.draw_progress_arc(buf, w, 1, self.sync_progress / 100.0);
        self.draw_phase_header(buf, w, "\u{2604}\u{FE0F} HOT STAGING", Color::Yellow);
        self.draw_telemetry_footer(buf, w, h);
    }

    fn render_cruise(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.97);
        self.draw_nebula(buf, w, h);
        self.draw_warp_trails(buf, w, h);

        let rocket_x = (w.saturating_sub(ROCKET_WIDTH)) / 2;
        let rocket_y = (h.saturating_sub(ROCKET.len() as u16)) / 2 - 1;
        self.draw_rocket(buf, rocket_x, rocket_y, w, h, true);
        self.draw_flame_plume(buf, rocket_x, rocket_y + ROCKET.len() as u16, w, h, 0.3);

        self.draw_progress_arc(buf, w, 1, self.sync_progress / 100.0);
        let warp = self.sync_speed / 100.0;
        self.draw_text_at(buf, 3, 3, &format!("\u{1F6F8} WARP {:.1}", warp.max(0.1)), Color::Magenta);

        self.draw_phase_header(buf, w, "\u{1F6F8} STARSHIP CRUISE", Color::Magenta);
        self.draw_telemetry_footer(buf, w, h);
    }

    fn render_orbital(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.95);
        self.draw_starfield(buf, w, h, 0.3);
        self.draw_planet_horizon(buf, w, h);
        self.draw_orbit_path(buf, w, h);

        // Orbiting rocket
        let angle = (self.frame as f32 * 0.03) % (2.0 * std::f32::consts::PI);
        let cx = w as f32 / 2.0;
        let cy = h as f32 * 0.45;
        let rx_orbit = w as f32 * 0.35;
        let ry_orbit = h as f32 * 0.2;
        let px = (cx + angle.cos() * rx_orbit) as u16;
        let py = (cy + angle.sin() * ry_orbit) as u16;
        if px > 1 && px < w - 2 && py > 1 && py < h - 3 {
            self.set_cell(buf, px, py, '\u{1F680}', Color::White);
            for t in 1..4u16 {
                let tx = (cx + (angle - t as f32 * 0.08).cos() * rx_orbit) as u16;
                let ty = (cy + (angle - t as f32 * 0.08).sin() * ry_orbit) as u16;
                if tx > 0 && tx < w && ty > 0 && ty < h {
                    let b = 200u8.saturating_sub(t as u8 * 50);
                    self.set_cell(buf, tx, ty, '·', Color::Rgb(b, b / 2, 0));
                }
            }
        }

        self.draw_progress_arc(buf, w, 1, self.sync_progress / 100.0);
        self.draw_phase_header(buf, w, "\u{1F30D} ORBITAL INSERTION", Color::Rgb(100, 180, 255));
        self.draw_telemetry_footer(buf, w, h);
    }

    fn render_stationkeeping(&self, buf: &mut Buffer, w: u16, h: u16) {
        self.dim_to_space(buf, w, h, 0.9);
        self.draw_starfield(buf, w, h, 0.15);
        self.draw_planet_horizon(buf, w, h);
        self.draw_orbit_path(buf, w, h);

        // Orbiting satellite
        let angle = (self.frame as f32 * 0.015) % (2.0 * std::f32::consts::PI);
        let cx = w as f32 / 2.0;
        let cy = h as f32 * 0.4;
        let px = (cx + angle.cos() * w as f32 * 0.3) as u16;
        let py = (cy + angle.sin() * h as f32 * 0.15) as u16;
        if px > 0 && px < w && py > 0 && py < h {
            self.set_cell(buf, px, py, '\u{1F6F0}', Color::White);
        }

        // SYNCED banner
        let pulse = ((self.frame as f32 * 0.1).sin() * 0.3 + 0.7).clamp(0.5, 1.0);
        let synced_text = "\u{2705} ORBIT STABLE \u{2014} NODE SYNCED";
        let sx = (w.saturating_sub(synced_text.chars().count() as u16)) / 2;
        for (i, ch) in synced_text.chars().enumerate() {
            let green = (200.0 * pulse) as u8;
            self.set_cell(buf, sx + i as u16, 2, ch, Color::Rgb(0, green, 0));
        }

        // Sparkle celebration
        for x in 0..w {
            for y in 0..(h / 3) {
                if self.sparkle(x, y) {
                    let hue = (x as f32 / w as f32 + self.frame as f32 * 0.03) % 1.0;
                    let ch = SPARKLE_CHARS[self.hash(x as u64, y as u64, self.frame as u64) as usize % SPARKLE_CHARS.len()];
                    self.set_cell(buf, x, y, ch, hue_to_color(hue));
                }
            }
        }

        self.draw_progress_arc(buf, w, 4, 1.0);
        self.draw_telemetry_footer(buf, w, h);
    }

    // ═══════════════════════════════════════════════════════════════
    // Drawing Primitives (ratatui 0.25 compatible — get_mut with bounds check)
    // ═══════════════════════════════════════════════════════════════

    fn dim_to_space(&self, buf: &mut Buffer, w: u16, h: u16, intensity: f32) {
        let area = buf.area;
        for y in 0..h {
            for x in 0..w {
                if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
                    let cell = buf.get_mut(x, y);
                    let (r, g, b) = color_to_rgb(cell.fg);
                    let dim = 1.0 - intensity;
                    cell.set_fg(Color::Rgb(
                        (r as f32 * dim) as u8,
                        (g as f32 * dim) as u8,
                        (b as f32 * dim) as u8,
                    ));
                    cell.set_bg(Color::Rgb(4, 4, 10));
                    if intensity > 0.8 {
                        cell.set_char(' ');
                    }
                }
            }
        }
    }

    fn draw_starfield(&self, buf: &mut Buffer, w: u16, h: u16, density: f32) {
        for star in &self.stars {
            if star.y < 0.0 || star.y > 1.0 { continue; }
            let sx = (star.x * w as f32) as u16;
            let sy = (star.y * h as f32) as u16;
            if sx >= w || sy >= h { continue; }

            let vis = self.hash(sx as u64, sy as u64, 0) % 100;
            if (vis as f32) > density * 100.0 { continue; }

            let twinkle = ((self.frame as f32 * 0.1 + star.x * 17.3 + star.y * 31.7).sin() * 40.0) as i16;
            let b = (star.brightness as i16 + twinkle).clamp(30, 255) as u8;

            let ch = if star.speed > 0.7 {
                STAR_CHARS[star.char_idx % STAR_CHARS.len()]
            } else {
                STAR_DIM[star.char_idx % STAR_DIM.len()]
            };

            self.set_cell(buf, sx, sy, ch, Color::Rgb(b, b, (b as f32 * 0.9) as u8));
        }
    }

    fn draw_speed_lines(&self, buf: &mut Buffer, w: u16, h: u16) {
        let num_lines = (w / 8).min(15);
        for i in 0..num_lines {
            let lx = self.hash(i as u64, 9999, 0) as u16 % w;
            let start_y = (self.frame.wrapping_mul(3).wrapping_add(i * 97)) % h;
            let length = 3 + (self.hash(i as u64, 8888, 0) % 5) as u16;

            for dy in 0..length {
                let ly = (start_y + dy) % h;
                let brightness = 255 - (dy * 40).min(200);
                let hue = (i as f32 / num_lines as f32 + self.frame as f32 * 0.01) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                self.set_cell(buf, lx, ly, '│', Color::Rgb(
                    (r as f32 * brightness as f32 / 255.0) as u8,
                    (g as f32 * brightness as f32 / 255.0) as u8,
                    (b as f32 * brightness as f32 / 255.0) as u8,
                ));
            }
        }
    }

    fn draw_warp_trails(&self, buf: &mut Buffer, w: u16, h: u16) {
        let cx = w / 2;
        let cy = h / 2;
        for i in 0..20u16 {
            let angle = self.hash(i as u64, 7777, self.frame as u64 / 30) as f32 * 0.0001;
            let dist = (self.hash(i as u64, 6666, 0) % (w as u64 / 2)) as f32;
            let trail_len = 4 + (self.hash(i as u64, 5555, 0) % 8) as u16;
            let hue = (i as f32 / 20.0 + self.frame as f32 * 0.02) % 1.0;
            let (r, g, b) = hue_to_rgb(hue);

            for t in 0..trail_len {
                let d = dist + t as f32;
                let tx = (cx as f32 + angle.cos() * d) as u16;
                let ty = (cy as f32 + angle.sin() * d / 2.2) as u16;
                if tx < w && ty < h {
                    let fade = 1.0 - t as f32 / trail_len as f32;
                    self.set_cell(buf, tx, ty, '─', Color::Rgb(
                        (r as f32 * fade) as u8,
                        (g as f32 * fade) as u8,
                        (b as f32 * fade) as u8,
                    ));
                }
            }
        }
    }

    fn draw_nebula(&self, buf: &mut Buffer, w: u16, h: u16) {
        let area = buf.area;
        for y in 0..h {
            for x in 0..w {
                let n1 = (self.hash(x as u64 / 8, y as u64 / 4, 0) % 100) as f32 / 100.0;
                let n2 = (self.hash(x as u64 / 12, y as u64 / 6, 1) % 100) as f32 / 100.0;

                if n1 > 0.7 && n2 > 0.6 {
                    let hue = (x as f32 / w as f32 * 0.3 + y as f32 / h as f32 * 0.2 + self.frame as f32 * 0.005) % 1.0;
                    let (r, g, b) = hue_to_rgb(hue);
                    let intensity = (n1 - 0.7) * 3.0 * 0.08;

                    if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
                        let cell = buf.get_mut(x, y);
                        let (cr, cg, cb) = color_to_rgb(cell.bg);
                        cell.set_bg(Color::Rgb(
                            (cr as f32 + r as f32 * intensity).min(255.0) as u8,
                            (cg as f32 + g as f32 * intensity).min(255.0) as u8,
                            (cb as f32 + b as f32 * intensity).min(255.0) as u8,
                        ));
                    }
                }
            }
        }
    }

    fn draw_debris(&self, buf: &mut Buffer, w: u16, h: u16) {
        for i in 0..12u16 {
            let base_x = w / 2 + (self.hash(i as u64, 3333, 0) % 20) as u16 - 10;
            let base_y = h / 2 + (self.frame / 2 + i * 3) % (h / 2);
            let drift_x = ((self.frame as f32 * 0.1 + i as f32).sin() * 3.0) as i16;
            let dx = (base_x as i16 + drift_x).clamp(0, w as i16 - 1) as u16;
            if base_y < h {
                let brightness = 200u8.saturating_sub((base_y.saturating_sub(h / 2)) as u8 * 4);
                let ch = ['▪', '▫', '·', '∙'][i as usize % 4];
                self.set_cell(buf, dx, base_y, ch, Color::Rgb(brightness, brightness / 2, 0));
            }
        }
    }

    fn draw_planet_horizon(&self, buf: &mut Buffer, w: u16, h: u16) {
        let horizon_y = h.saturating_sub(5);
        let cx = w as f32 / 2.0;
        for y in horizon_y..h {
            let depth = (y - horizon_y) as f32;
            for x in 0..w {
                let dx = (x as f32 - cx).abs();
                let curve = (1.0 - (dx / (w as f32 * 0.6)).powi(2)).max(0.0);
                if depth < curve * 5.0 {
                    let glow = (1.0 - depth / 5.0).clamp(0.0, 1.0);
                    let r = (30.0 * glow) as u8;
                    let g = (80.0 * glow + 40.0 * depth) as u8;
                    let b = (180.0 * glow + 60.0 * curve) as u8;
                    let ch = HORIZON_CHARS[(depth as usize).min(HORIZON_CHARS.len() - 1)];
                    self.set_cell(buf, x, y, ch, Color::Rgb(r, g, b));
                }
            }
        }
    }

    fn draw_orbit_path(&self, buf: &mut Buffer, w: u16, h: u16) {
        let cx = w as f32 / 2.0;
        let cy = h as f32 * 0.4;
        let rx = w as f32 * 0.3;
        let ry = h as f32 * 0.15;
        for step in 0..72 {
            let angle = step as f32 * std::f32::consts::PI / 36.0;
            let px = (cx + angle.cos() * rx) as u16;
            let py = (cy + angle.sin() * ry) as u16;
            if px < w && py < h {
                let hue = (step as f32 / 72.0 + self.frame as f32 * 0.008) % 1.0;
                let ch = if step % 3 == 0 { '·' } else { '─' };
                let (r, g, b) = hue_to_rgb(hue);
                self.set_cell(buf, px, py, ch, Color::Rgb(r / 3, g / 3, b / 3));
            }
        }
    }

    fn draw_rocket(&self, buf: &mut Buffer, x: u16, y: u16, w: u16, h: u16, with_glow: bool) {
        for (row_idx, row) in ROCKET.iter().enumerate() {
            let ry = y + row_idx as u16;
            if ry >= h { continue; }
            for (col_idx, ch) in row.chars().enumerate() {
                let rx = x + col_idx as u16;
                if rx >= w || ch == ' ' { continue; }
                let color = match ch {
                    '/' | '\\' => {
                        if row_idx < 2 { Color::Rgb(220, 220, 230) }
                        else { Color::Rgb(160, 160, 170) }
                    }
                    'Q' | 'N' | 'K' | 'T' => {
                        let hue = (col_idx as f32 / ROCKET_WIDTH as f32 + self.frame as f32 * 0.03) % 1.0;
                        hue_to_color(hue)
                    }
                    '|' => Color::Rgb(180, 180, 190),
                    _ => Color::Rgb(200, 200, 210),
                };
                self.set_cell(buf, rx, ry, ch, color);
            }
        }
        let _ = with_glow; // Glow handled via background in miner version
    }

    fn draw_flame_plume(&self, buf: &mut Buffer, rocket_x: u16, flame_start_y: u16, w: u16, h: u16, intensity: f32) {
        let flame_idx = self.frame as usize % FLAMES.len();
        let tip_idx = (self.frame as usize + 1) % FLAME_TIPS.len();

        if flame_start_y < h {
            for (ci, ch) in FLAMES[flame_idx].chars().enumerate() {
                let fx = rocket_x + ci as u16;
                if fx < w && ch != ' ' {
                    let heat = Color::Rgb(255, (180.0 * intensity) as u8, 0);
                    self.set_cell(buf, fx, flame_start_y, ch, heat);
                }
            }
        }
        let tip_y = flame_start_y + 1;
        if tip_y < h {
            for (ci, ch) in FLAME_TIPS[tip_idx].chars().enumerate() {
                let fx = rocket_x + ci as u16;
                if fx < w && ch != ' ' {
                    self.set_cell(buf, fx, tip_y, ch, Color::Rgb(255, (140.0 * intensity) as u8, 0));
                }
            }
        }

        let particle_count = (8.0 * intensity) as u16;
        for i in 0..particle_count {
            let px = rocket_x + ROCKET_WIDTH / 2 +
                ((self.hash(i as u64, self.frame as u64, 4444) % 8) as u16).wrapping_sub(4);
            let py = tip_y + 2 + (self.hash(i as u64, self.frame as u64, 5555) % 4) as u16;
            if px < w && py < h {
                let fade = 1.0 - (py.saturating_sub(tip_y) as f32) / 6.0;
                let ch = ['·', '∙', '.', ','][i as usize % 4];
                self.set_cell(buf, px, py, ch, Color::Rgb(
                    (200.0 * fade * intensity) as u8,
                    (100.0 * fade * intensity) as u8,
                    0,
                ));
            }
        }
    }

    fn draw_progress_arc(&self, buf: &mut Buffer, w: u16, y: u16, progress: f32) {
        let progress = progress.clamp(0.0, 1.0);
        let bar_start = 2u16;
        let bar_end = w.saturating_sub(2);
        let bar_width = bar_end.saturating_sub(bar_start);

        self.set_cell(buf, bar_start.saturating_sub(1), y, '╞', Color::DarkGray);
        self.set_cell(buf, bar_end, y, '╡', Color::DarkGray);

        let filled = (bar_width as f32 * progress) as u16;
        for i in 0..bar_width {
            let x = bar_start + i;
            if i < filled {
                let hue = (i as f32 / bar_width as f32 * 0.8 + self.frame as f32 * 0.02) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                let shimmer_pos = (self.frame as f32 * 0.4) % (bar_width as f32 * 1.5);
                let dist = (i as f32 - shimmer_pos).abs();
                let shimmer = if dist < 2.5 { 1.0 - dist / 2.5 } else { 0.0 };
                let bright = 1.0 + shimmer * 0.5;
                self.set_cell(buf, x, y, '█', Color::Rgb(
                    (r as f32 * bright).min(255.0) as u8,
                    (g as f32 * bright).min(255.0) as u8,
                    (b as f32 * bright).min(255.0) as u8,
                ));
            } else {
                self.set_cell(buf, x, y, '░', Color::Rgb(25, 25, 35));
            }
        }

        let pct_text = format!("{:.1}%", progress * 100.0);
        let pct_x = bar_end.saturating_sub(pct_text.len() as u16 + 1);
        if y + 1 < buf.area.height {
            for (i, ch) in pct_text.chars().enumerate() {
                self.set_cell(buf, pct_x + i as u16, y + 1, ch, Color::White);
            }
        }
    }

    fn draw_phase_header(&self, buf: &mut Buffer, w: u16, text: &str, color: Color) {
        let char_count = text.chars().count() as u16;
        let x = (w.saturating_sub(char_count)) / 2;
        for (i, ch) in text.chars().enumerate() {
            let px = x + i as u16;
            if px < w {
                self.set_cell(buf, px, 0, ch, color);
            }
        }
    }

    fn draw_telemetry_footer(&self, buf: &mut Buffer, w: u16, h: u16) {
        let footer_y = h.saturating_sub(1);
        let height_text = format!(
            " #{} / #{} ({} behind) ",
            format_commas(self.local_height),
            format_commas(self.network_height),
            format_commas(self.blocks_behind),
        );
        for (i, ch) in height_text.chars().enumerate() {
            if (i as u16) < w {
                self.set_cell(buf, i as u16, footer_y, ch, Color::DarkGray);
            }
        }
        let right_text = format!(
            " {} peers | ETA {} | T+{} ",
            self.peer_count,
            format_duration(self.eta_secs),
            format_duration(self.mission_elapsed),
        );
        let rx = w.saturating_sub(right_text.chars().count() as u16);
        for (i, ch) in right_text.chars().enumerate() {
            let px = rx + i as u16;
            if px < w {
                self.set_cell(buf, px, footer_y, ch, Color::DarkGray);
            }
        }
    }

    fn draw_text_centered(&self, buf: &mut Buffer, w: u16, y: u16, text: &str, color: Color) {
        let char_count = text.chars().count() as u16;
        let x = (w.saturating_sub(char_count)) / 2;
        for (i, ch) in text.chars().enumerate() {
            let px = x + i as u16;
            if px < w && y < buf.area.height {
                self.set_cell(buf, px, y, ch, color);
            }
        }
    }

    fn draw_text_at(&self, buf: &mut Buffer, x: u16, y: u16, text: &str, color: Color) {
        for (i, ch) in text.chars().enumerate() {
            let px = x + i as u16;
            if px < buf.area.width && y < buf.area.height {
                self.set_cell(buf, px, y, ch, color);
            }
        }
    }

    /// ratatui 0.25 compatible cell setter with bounds checking
    fn set_cell(&self, buf: &mut Buffer, x: u16, y: u16, ch: char, fg: Color) {
        let area = buf.area;
        if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
            let cell = buf.get_mut(x, y);
            cell.set_char(ch);
            cell.set_fg(fg);
        }
    }

    fn sparkle(&self, x: u16, y: u16) -> bool {
        let h = self.hash(x as u64, y as u64, self.frame as u64);
        h % 80 == 0
    }

    fn hash(&self, x: u64, y: u64, frame: u64) -> u64 {
        hash_raw(self.seed, x, y, frame)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Free Functions
// ═══════════════════════════════════════════════════════════════════

fn hash_raw(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut h = seed;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(a);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(b);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(c);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

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
        Color::DarkGray => (64, 64, 64),
        Color::Black => (0, 0, 0),
        Color::Reset => (180, 180, 180),
        _ => (180, 180, 180),
    }
}

fn format_commas(n: u64) -> String {
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

fn format_duration(secs: u64) -> String {
    if secs == 0 { return "--".to_string(); }
    if secs < 60 { format!("{}s", secs) }
    else if secs < 3600 { format!("{}m{}s", secs / 60, secs % 60) }
    else if secs < 86400 { format!("{}h{}m", secs / 3600, (secs % 3600) / 60) }
    else { format!("{}d{}h", secs / 86400, (secs % 86400) / 3600) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animation_lifecycle() {
        let mut anim = StarshipAnimation::new();
        assert!(!anim.is_visible());
        anim.set_phase("Prelaunch");
        assert!(anim.is_visible());
        anim.set_hidden();
        assert!(!anim.is_visible());
    }

    #[test]
    fn test_phase_parsing() {
        assert_eq!(StarshipPhase::from_str("SuperHeavy"), StarshipPhase::SuperHeavy);
        assert_eq!(StarshipPhase::from_str("StationKeeping"), StarshipPhase::StationKeeping);
    }

    #[test]
    fn test_hash_deterministic() {
        let a = hash_raw(SEED, 1, 2, 3);
        let b = hash_raw(SEED, 1, 2, 3);
        assert_eq!(a, b);
    }
}
