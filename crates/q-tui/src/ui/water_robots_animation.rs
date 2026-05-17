//! Water Robots Animation for q-tui (ratatui 0.25 compatible)
//!
//! Quantum marine creatures surfing across the node TUI on important events.
//! Same visuals as q-miner version, adapted for ratatui 0.25 API
//! (get_mut with manual bounds checking instead of cell_mut(Position)).

use ratatui::{
    buffer::Buffer,
    style::Color,
};

const SEED: u64 = 0xAC0A_B075_F150_CA5E;

const ENTER_TICKS: u16 = 12;
const DISPLAY_TICKS: u16 = 16;
const EXIT_TICKS: u16 = 10;
const TOTAL_TICKS: u16 = ENTER_TICKS + DISPLAY_TICKS + EXIT_TICKS;

const WAVE_CHARS: [char; 8] = ['~', '≈', '∿', '≋', '〜', '∼', '～', '≈'];
const BUBBLE_CHARS: [char; 5] = ['°', '•', '○', '◦', '∘'];
const SPLASH_CHARS: [char; 6] = ['*', '✦', '·', '∙', '⁕', '✧'];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotSpecies {
    QuantumJellyfish,
    EntangledDolphin,
    TunnelingOctopus,
    WaveParticleWhale,
    SuperpositionSeahorse,
    NanoQuantumonas,
    SchoolingRobotichthys,
    CyberCetus,
}

impl RobotSpecies {
    fn all() -> [Self; 8] {
        [
            Self::QuantumJellyfish, Self::EntangledDolphin, Self::TunnelingOctopus,
            Self::WaveParticleWhale, Self::SuperpositionSeahorse, Self::NanoQuantumonas,
            Self::SchoolingRobotichthys, Self::CyberCetus,
        ]
    }

    fn icon(&self) -> char {
        match self {
            Self::QuantumJellyfish => '~',
            Self::EntangledDolphin => '^',
            Self::TunnelingOctopus => '*',
            Self::WaveParticleWhale => '#',
            Self::SuperpositionSeahorse => '?',
            Self::NanoQuantumonas => '.',
            Self::SchoolingRobotichthys => '>',
            Self::CyberCetus => '@',
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::QuantumJellyfish => "Quantum Jellyfish",
            Self::EntangledDolphin => "Entangled Dolphin",
            Self::TunnelingOctopus => "Tunneling Octopus",
            Self::WaveParticleWhale => "Wave-Particle Whale",
            Self::SuperpositionSeahorse => "Superposition Seahorse",
            Self::NanoQuantumonas => "Nano Quantumonas",
            Self::SchoolingRobotichthys => "Schooling Robotichthys",
            Self::CyberCetus => "Cyber Cetus",
        }
    }

    fn primary_color(&self) -> Color {
        match self {
            Self::QuantumJellyfish => Color::Rgb(200, 100, 255),
            Self::EntangledDolphin => Color::Rgb(0, 200, 255),
            Self::TunnelingOctopus => Color::Rgb(255, 80, 120),
            Self::WaveParticleWhale => Color::Rgb(60, 120, 200),
            Self::SuperpositionSeahorse => Color::Rgb(255, 200, 0),
            Self::NanoQuantumonas => Color::Rgb(0, 255, 150),
            Self::SchoolingRobotichthys => Color::Rgb(255, 165, 0),
            Self::CyberCetus => Color::Rgb(0, 255, 255),
        }
    }

    fn secondary_color(&self) -> Color {
        match self {
            Self::QuantumJellyfish => Color::Rgb(120, 40, 200),
            Self::EntangledDolphin => Color::Rgb(0, 100, 200),
            Self::TunnelingOctopus => Color::Rgb(180, 30, 80),
            Self::WaveParticleWhale => Color::Rgb(30, 60, 120),
            Self::SuperpositionSeahorse => Color::Rgb(200, 150, 0),
            Self::NanoQuantumonas => Color::Rgb(0, 180, 100),
            Self::SchoolingRobotichthys => Color::Rgb(200, 120, 0),
            Self::CyberCetus => Color::Rgb(0, 180, 200),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WaterRobotEvent {
    BlockFound { height: u64 },
    RewardReceived { amount: f64 },
    SyncMilestone { progress: f32 },
    NewPeerConnected { peer_count: u64 },
}

impl WaterRobotEvent {
    fn banner_text(&self) -> String {
        match self {
            Self::BlockFound { height } => format!("BLOCK #{} FOUND!", format_commas(*height)),
            Self::RewardReceived { amount } => format!("REWARD: {:.4} QUG", amount),
            Self::SyncMilestone { progress } => format!("SYNC {:.0}% REACHED!", progress),
            Self::NewPeerConnected { peer_count } => format!("{} PEERS CONNECTED", peer_count),
        }
    }
}

pub struct WaterRobotsAnimation {
    frame: u16,
    active: bool,
    species: RobotSpecies,
    event: Option<WaterRobotEvent>,
    seed: u64,
    last_species_idx: usize,
    cooldown: u16,
}

impl WaterRobotsAnimation {
    pub fn new() -> Self {
        Self {
            frame: 0,
            active: false,
            species: RobotSpecies::QuantumJellyfish,
            event: None,
            seed: SEED,
            last_species_idx: 99,
            cooldown: 0,
        }
    }

    pub fn trigger(&mut self, event: WaterRobotEvent) {
        if self.active || self.cooldown > 0 { return; }

        let all = RobotSpecies::all();
        let h = hash_raw(self.seed, self.last_species_idx as u64, self.frame as u64, 42);
        let mut idx = (h % all.len() as u64) as usize;
        if idx == self.last_species_idx {
            idx = (idx + 1) % all.len();
        }
        self.species = all[idx];
        self.last_species_idx = idx;
        self.event = Some(event);
        self.frame = 0;
        self.active = true;
    }

    pub fn is_visible(&self) -> bool { self.active }

    pub fn tick(&mut self) {
        if self.cooldown > 0 { self.cooldown -= 1; }
        if !self.active { return; }
        self.frame += 1;
        if self.frame >= TOTAL_TICKS {
            self.active = false;
            self.event = None;
            self.cooldown = 80;
        }
    }

    pub fn render(&self, buf: &mut Buffer) {
        if !self.active { return; }

        let area = buf.area;
        let w = area.width;
        let h = area.height;
        if w < 40 || h < 12 { return; }

        let t = self.frame as f32 / TOTAL_TICKS as f32;

        let (phase, phase_t) = if self.frame < ENTER_TICKS {
            (0, self.frame as f32 / ENTER_TICKS as f32)
        } else if self.frame < ENTER_TICKS + DISPLAY_TICKS {
            (1, (self.frame - ENTER_TICKS) as f32 / DISPLAY_TICKS as f32)
        } else {
            (2, (self.frame - ENTER_TICKS - DISPLAY_TICKS) as f32 / EXIT_TICKS as f32)
        };

        self.draw_water_surface(buf, w, h, t);

        match self.species {
            RobotSpecies::QuantumJellyfish => self.draw_jellyfish(buf, w, h, phase, phase_t, t),
            RobotSpecies::EntangledDolphin => self.draw_dolphin(buf, w, h, t),
            RobotSpecies::TunnelingOctopus => self.draw_octopus(buf, w, h, t),
            RobotSpecies::WaveParticleWhale => self.draw_whale(buf, w, h, t),
            RobotSpecies::SuperpositionSeahorse => self.draw_seahorse(buf, w, h, t),
            RobotSpecies::NanoQuantumonas => self.draw_nano_swarm(buf, w, h, t),
            RobotSpecies::SchoolingRobotichthys => self.draw_school(buf, w, h, t),
            RobotSpecies::CyberCetus => self.draw_guardian(buf, w, h, t),
        }

        self.draw_event_banner(buf, w, phase, phase_t);
        self.draw_species_label(buf, w, h);
    }

    // ─── Helper: bounds-checked set_cell for ratatui 0.25 ───────────

    fn set_cell(&self, buf: &mut Buffer, x: u16, y: u16, ch: char, fg: Color) {
        let area = buf.area;
        if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
            let cell = buf.get_mut(x, y);
            cell.set_char(ch);
            cell.set_fg(fg);
        }
    }

    fn set_cell_bg(&self, buf: &mut Buffer, x: u16, y: u16, ch: char, fg: Color, bg: Color) {
        let area = buf.area;
        if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
            let cell = buf.get_mut(x, y);
            cell.set_char(ch);
            cell.set_fg(fg);
            cell.set_bg(bg);
        }
    }

    fn get_fg(&self, buf: &Buffer, x: u16, y: u16) -> Color {
        let area = buf.area;
        if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
            buf.get(x, y).fg
        } else {
            Color::Reset
        }
    }

    fn hash(&self, x: u64, y: u64, frame: u64) -> u64 {
        hash_raw(self.seed, x, y, frame)
    }

    // ─── Water surface ──────────────────────────────────────────────

    fn draw_water_surface(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;

        for x in 0..w {
            let wave_offset = ((x as f32 * 0.15 + self.frame as f32 * 0.3).sin() * 1.5) as i16;
            let wy = (water_y as i16 + wave_offset).clamp(0, h as i16 - 1) as u16;

            let hue = (x as f32 / w as f32 * 0.3 + 0.55 + t * 0.05) % 1.0;
            let wave_ch = WAVE_CHARS[(x as usize + self.frame as usize) % WAVE_CHARS.len()];
            self.set_cell(buf, x, wy, wave_ch, hue_to_color(hue));

            for y in (wy + 1)..h {
                let depth = (y - wy) as f32 / (h - wy) as f32;
                let alpha = (depth * 0.3).min(0.25);
                let fg = self.get_fg(buf, x, y);
                let (r, g, b) = color_to_rgb(fg);
                self.set_cell(buf, x, y,
                    buf.get(x, y).symbol().chars().next().unwrap_or(' '),
                    Color::Rgb(
                        (r as f32 * (1.0 - alpha) + 20.0 * alpha) as u8,
                        (g as f32 * (1.0 - alpha) + 60.0 * alpha) as u8,
                        (b as f32 * (1.0 - alpha) + 120.0 * alpha) as u8,
                    ));
            }
        }

        for i in 0..8u16 {
            let bx = (hash_raw(self.seed, i as u64, 1111, 0) % w as u64) as u16;
            let rise = (self.frame / 3 + i * 5) % (h - water_y).max(1);
            let bubble_y = h.saturating_sub(rise + 1);
            if bubble_y > water_y && bubble_y < h && bx < w {
                let ch = BUBBLE_CHARS[(i as usize + self.frame as usize / 2) % BUBBLE_CHARS.len()];
                self.set_cell(buf, bx, bubble_y, ch, Color::Rgb(100, 180, 255));
            }
        }
    }

    // ─── Species animations ─────────────────────────────────────────

    fn draw_jellyfish(&self, buf: &mut Buffer, w: u16, h: u16, phase: u8, pt: f32, _t: f32) {
        let water_y = h * 2 / 3;
        let cx = match phase {
            0 => w + 5 - (pt * (w / 3) as f32) as u16,
            1 => (w * 2 / 3) - (pt * (w / 3) as f32) as u16,
            _ => (w / 3).saturating_sub((pt * (w / 2) as f32) as u16),
        };
        let cy = water_y.saturating_sub(3) + ((self.frame as f32 * 0.2).sin() * 2.0) as u16;
        if cx >= w { return; }

        let bell = ["  ,---,  ", " ,-----. ", ",-------.", "'~~~~~'  "];
        for (i, line) in bell.iter().enumerate() {
            let ly = cy + i as u16;
            if ly >= h { continue; }
            for (j, ch) in line.chars().enumerate() {
                let lx = cx.saturating_sub(4) + j as u16;
                if lx < w && ch != ' ' {
                    let glow_hue = (j as f32 / 9.0 + self.frame as f32 * 0.05) % 1.0;
                    self.set_cell(buf, lx, ly, ch, hue_to_color(glow_hue * 0.3 + 0.7));
                }
            }
        }

        for tent in 0..5u16 {
            let tx = cx.saturating_sub(3) + tent * 2;
            for dy in 0..4u16 {
                let ty = cy + 4 + dy;
                if ty >= h || tx >= w { continue; }
                let wave = ((self.frame as f32 * 0.3 + tent as f32 * 1.5 + dy as f32).sin() * 1.0) as i16;
                let tentx = (tx as i16 + wave).clamp(0, w as i16 - 1) as u16;
                let brightness = 255u8.saturating_sub(dy as u8 * 40);
                self.set_cell(buf, tentx, ty, '|', Color::Rgb(brightness, brightness / 3, brightness));
            }
        }

        for i in 0..6u16 {
            let angle = self.frame as f32 * 0.15 + i as f32 * std::f32::consts::PI / 3.0;
            let gx = (cx as f32 + angle.cos() * 5.0) as u16;
            let gy = (cy as f32 + angle.sin() * 3.0) as u16;
            if gx < w && gy < h {
                let hue = (i as f32 / 6.0 + self.frame as f32 * 0.04) % 1.0;
                self.set_cell(buf, gx, gy, '*', hue_to_color(hue));
            }
        }
    }

    fn draw_dolphin(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;
        let progress = t;
        let cx = (progress * w as f32 * 1.2 - w as f32 * 0.1) as i16;
        let leap_height = (h / 3) as f32;
        let arc = -4.0 * (progress - 0.5).powi(2) + 1.0;
        let cy = (water_y as f32 - arc * leap_height) as i16;

        let dolphin = if arc > 0.3 {
            [" ,--. ", ",| ^ |,", "'----'"]
        } else {
            ["       ", " ~^~   ", "  ~~~  "]
        };

        for (i, line) in dolphin.iter().enumerate() {
            let dy = cy + i as i16;
            if dy < 0 || dy >= h as i16 { continue; }
            for (j, ch) in line.chars().enumerate() {
                let dx = cx + j as i16;
                if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                self.set_cell(buf, dx as u16, dy as u16, ch,
                    RobotSpecies::EntangledDolphin.primary_color());
            }
        }

        if arc.abs() < 0.15 {
            let splash_x = cx.clamp(0, w as i16 - 1) as u16;
            for i in 0..8u16 {
                let sx = splash_x + (hash_raw(self.seed, i as u64, self.frame as u64, 0) % 10) as u16;
                let sy = water_y.saturating_sub((hash_raw(self.seed, i as u64, self.frame as u64, 1) % 4) as u16);
                if sx < w && sy < h {
                    let ch = SPLASH_CHARS[i as usize % SPLASH_CHARS.len()];
                    self.set_cell(buf, sx, sy, ch, Color::Rgb(200, 230, 255));
                }
            }
        }
    }

    fn draw_octopus(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let cx = (t * w as f32 * 1.3 - w as f32 * 0.15) as i16;
        let cy = (h as f32 * 0.3 + (t * 6.0).sin() * h as f32 * 0.15) as i16;

        for i in 0..15u16 {
            let trail_t = t - i as f32 * 0.015;
            if trail_t < 0.0 { continue; }
            let ix = (trail_t * w as f32 * 1.3 - w as f32 * 0.15) as i16;
            let iy = (h as f32 * 0.3 + (trail_t * 6.0).sin() * h as f32 * 0.15) as i16;
            let spread = i as i16;
            for _ in 0..3 {
                let dx = ix + (hash_raw(self.seed, i as u64, self.frame as u64, 222) % (spread as u64 * 2 + 1)) as i16 - spread;
                let dy = iy + (hash_raw(self.seed, i as u64, self.frame as u64, 333) % (spread as u64 + 1)) as i16;
                if dx >= 0 && dx < w as i16 && dy >= 0 && dy < h as i16 {
                    let fade = 1.0 - i as f32 / 15.0;
                    self.set_cell(buf, dx as u16, dy as u16, '.', Color::Rgb(
                        (80.0 * fade) as u8, (20.0 * fade) as u8, (60.0 * fade) as u8,
                    ));
                }
            }
        }

        let octo = ["  ,-.  ", " /* \\ ", "'|'|'|"];
        for (i, line) in octo.iter().enumerate() {
            let dy = cy + i as i16;
            if dy < 0 || dy >= h as i16 { continue; }
            for (j, ch) in line.chars().enumerate() {
                let dx = cx + j as i16;
                if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                self.set_cell(buf, dx as u16, dy as u16, ch,
                    RobotSpecies::TunnelingOctopus.primary_color());
            }
        }

        for arm in 0..8u16 {
            for seg in 0..3u16 {
                let wave = ((self.frame as f32 * 0.25 + arm as f32 * 0.8 + seg as f32).sin() * 2.0) as i16;
                let ax = cx - 3 + arm as i16 + wave;
                let ay = cy + 3 + seg as i16;
                if ax >= 0 && ax < w as i16 && ay >= 0 && ay < h as i16 {
                    let fade = 1.0 - seg as f32 / 3.0;
                    let color = RobotSpecies::TunnelingOctopus.primary_color();
                    let (r, g, b) = color_to_rgb(color);
                    self.set_cell(buf, ax as u16, ay as u16, '\\', Color::Rgb(
                        (r as f32 * fade) as u8, (g as f32 * fade) as u8, (b as f32 * fade) as u8,
                    ));
                }
            }
        }
    }

    fn draw_whale(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;
        let cx = (t * w as f32 * 1.4 - w as f32 * 0.2) as i16;
        let cy = water_y as i16 - 4;

        let wave_front = cx + 8;
        if wave_front > 0 && wave_front < w as i16 {
            for dy in -3i16..4 {
                let wy = (water_y as i16 + dy).clamp(0, h as i16 - 1) as u16;
                let wave_h = (3.0 - (dy as f32).abs()) / 3.0;
                let wave_x = (wave_front + dy.abs() * 2) as u16;
                if wave_x < w {
                    let ch = if dy < 0 { '/' } else { '\\' };
                    let blue = (150.0 * wave_h) as u8;
                    self.set_cell(buf, wave_x, wy, ch, Color::Rgb(80, 150 + blue / 3, 200 + blue / 5));
                }
            }
        }

        let whale_art = [
            "      ,--------,      ",
            "   ,--| WHALE  |--,   ",
            "  ,|  |        |  |,  ",
            "--|   '--------'   |--",
            "  '-----,    ,-----'  ",
            "        '----'        ",
        ];

        for (i, line) in whale_art.iter().enumerate() {
            let dy = cy + i as i16;
            if dy < 0 || dy >= h as i16 { continue; }
            for (j, ch) in line.chars().enumerate() {
                let dx = cx - 11 + j as i16;
                if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                let depth = i as f32 / whale_art.len() as f32;
                self.set_cell(buf, dx as u16, dy as u16, ch, Color::Rgb(
                    (60.0 + 40.0 * depth) as u8,
                    (120.0 + 80.0 * (1.0 - depth)) as u8,
                    (200.0 + 40.0 * depth) as u8,
                ));
            }
        }

        if self.frame % 20 < 10 {
            let spout_x = (cx + 2).clamp(0, w as i16 - 1) as u16;
            for dy in 1..5u16 {
                let sy = cy.saturating_sub(dy as i16).max(0) as u16;
                if sy < h && spout_x < w {
                    let fade = 1.0 - dy as f32 / 5.0;
                    self.set_cell(buf, spout_x, sy, '|', Color::Rgb(
                        (180.0 * fade) as u8, (220.0 * fade) as u8, (255.0 * fade) as u8,
                    ));
                }
            }
            let spray_y = cy.saturating_sub(5).max(0) as u16;
            if spray_y < h {
                for dx in -2i16..3 {
                    let sx = (spout_x as i16 + dx).clamp(0, w as i16 - 1) as u16;
                    if sx < w {
                        self.set_cell(buf, sx, spray_y, '.', Color::Rgb(200, 230, 255));
                    }
                }
            }
        }
    }

    fn draw_seahorse(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;

        let positions: [(i16, i16, f32); 3] = [
            ((w as f32 * 0.25 + (t * 3.0).sin() * 5.0) as i16,
             (water_y as f32 - 4.0 + (t * 4.0).cos() * 2.0) as i16, 0.4),
            ((w as f32 * 0.5 + (t * 2.5).cos() * 4.0) as i16,
             (water_y as f32 - 6.0 + (t * 3.0).sin() * 3.0) as i16, 1.0),
            ((w as f32 * 0.75 + (t * 3.5).sin() * 3.0) as i16,
             (water_y as f32 - 3.0 + (t * 5.0).cos() * 2.0) as i16, 0.4),
        ];

        let (x1, y1, _) = positions[0];
        let (x2, y2, _) = positions[1];
        let (x3, y3, _) = positions[2];
        self.draw_entangle_line(buf, w, h, x1 as u16, y1 as u16, x2 as u16, y2 as u16);
        self.draw_entangle_line(buf, w, h, x2 as u16, y2 as u16, x3 as u16, y3 as u16);

        for (px, py, opacity) in &positions {
            let seahorse = ["  ? ", " S  ", " '' "];
            for (i, line) in seahorse.iter().enumerate() {
                let dy = py + i as i16;
                if dy < 0 || dy >= h as i16 { continue; }
                for (j, ch) in line.chars().enumerate() {
                    let dx = px + j as i16;
                    if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                    let (r, g, b) = color_to_rgb(RobotSpecies::SuperpositionSeahorse.primary_color());
                    self.set_cell(buf, dx as u16, dy as u16, ch, Color::Rgb(
                        (r as f32 * opacity) as u8, (g as f32 * opacity) as u8, (b as f32 * opacity) as u8,
                    ));
                }
            }

            if self.hash(*px as u64, *py as u64, self.frame as u64) % 3 == 0 {
                let sx = (*px + 3).clamp(0, w as i16 - 1) as u16;
                let sy = (*py - 1).clamp(0, h as i16 - 1) as u16;
                if sx < w && sy < h {
                    let hue = (self.frame as f32 * 0.06) % 1.0;
                    self.set_cell(buf, sx, sy, '*', hue_to_color(hue));
                }
            }
        }
    }

    fn draw_nano_swarm(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;
        let num_particles = 60u16;

        for i in 0..num_particles {
            let base_x = (hash_raw(self.seed, i as u64, 0, 0) % w as u64) as f32;
            let base_y = water_y as f32 - 2.0 + (hash_raw(self.seed, i as u64, 1, 0) % (h as u64 / 3)) as f32;
            let flow_speed = 0.5 + (hash_raw(self.seed, i as u64, 2, 0) % 50) as f32 / 100.0;
            let px = ((base_x + self.frame as f32 * flow_speed) % (w as f32 + 10.0)) as u16;
            let wave = ((self.frame as f32 * 0.2 + i as f32 * 0.5).sin() * 2.0) as i16;
            let py = (base_y as i16 + wave).clamp(0, h as i16 - 1) as u16;

            if px < w && py < h {
                let hue = (i as f32 / num_particles as f32 * 0.3 + 0.25 + self.frame as f32 * 0.01) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                let brightness = 0.5 + (hash_raw(self.seed, i as u64, 3, 0) % 50) as f32 / 100.0;
                let ch = if i % 3 == 0 { '*' } else if i % 3 == 1 { '.' } else { ':' };
                self.set_cell(buf, px, py, ch, Color::Rgb(
                    (r as f32 * brightness) as u8,
                    (g as f32 * brightness) as u8,
                    (b as f32 * brightness) as u8,
                ));
            }
        }

        let icon_x = ((t * w as f32 * 1.2 - w as f32 * 0.1) as u16).min(w.saturating_sub(1));
        let icon_y = water_y.saturating_sub(4) + ((self.frame as f32 * 0.15).sin() * 2.0) as u16;
        if icon_x < w && icon_y < h {
            self.set_cell(buf, icon_x, icon_y, 'o', Color::Rgb(0, 255, 150));
        }
    }

    fn draw_school(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;
        let school_size = 12u16;
        let leader_x = (t * w as f32 * 1.3 - w as f32 * 0.15) as f32;
        let leader_y = (water_y as f32 - 5.0 + (t * 5.0).sin() * 3.0) as f32;

        for i in 0..school_size {
            let row = i / 2 + 1;
            let side = if i % 2 == 0 { 1.0f32 } else { -1.0 };
            let offset_x = -(row as f32) * 3.0;
            let offset_y = side * row as f32 * 1.5;
            let wobble_x = (self.frame as f32 * 0.3 + i as f32 * 1.7).sin() * 0.5;
            let wobble_y = (self.frame as f32 * 0.4 + i as f32 * 2.3).cos() * 0.3;
            let fx = (leader_x + offset_x + wobble_x) as i16;
            let fy = (leader_y + offset_y + wobble_y) as i16;

            if fx >= 0 && fx < w as i16 && fy >= 0 && fy < h as i16 {
                let ch = if (fx + self.frame as i16) % 2 == 0 { '>' } else { '<' };
                self.set_cell(buf, fx as u16, fy as u16, ch,
                    Color::Rgb(255, 165 + (i as u8 * 5), 0));
            }
        }

        let lx = leader_x as i16;
        let ly = leader_y as i16;
        if lx >= 0 && lx < w as i16 - 2 && ly >= 0 && ly < h as i16 {
            self.set_cell(buf, lx as u16, ly as u16, '>', Color::Rgb(255, 200, 50));
        }
    }

    fn draw_guardian(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3;
        let cx = (t * w as f32 * 1.2 - w as f32 * 0.1) as i16;
        let cy = water_y as i16 - 3;

        for ring in 0..3u16 {
            let ring_age = (self.frame + ring * 8) % 24;
            let radius = ring_age as f32 * 1.5;
            let opacity = 1.0 - ring_age as f32 / 24.0;
            if opacity <= 0.0 { continue; }

            for step in 0..24 {
                let angle = step as f32 * std::f32::consts::PI / 12.0;
                let rx = (cx as f32 + angle.cos() * radius) as i16;
                let ry = (cy as f32 + angle.sin() * radius / 2.2) as i16;
                if rx >= 0 && rx < w as i16 && ry >= 0 && ry < h as i16 {
                    let (r, g, b) = color_to_rgb(RobotSpecies::CyberCetus.primary_color());
                    self.set_cell(buf, rx as u16, ry as u16, '.', Color::Rgb(
                        (r as f32 * opacity) as u8, (g as f32 * opacity) as u8, (b as f32 * opacity) as u8,
                    ));
                }
            }
        }

        let shark = [
            "    /^\\     ",
            "  /-@-\\   ",
            "/-------\\--\\",
            "\\-------/--/",
            "  \\-----/   ",
        ];

        for (i, line) in shark.iter().enumerate() {
            let dy = cy + i as i16;
            if dy < 0 || dy >= h as i16 { continue; }
            for (j, ch) in line.chars().enumerate() {
                let dx = cx - 6 + j as i16;
                if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                self.set_cell(buf, dx as u16, dy as u16, ch,
                    RobotSpecies::CyberCetus.primary_color());
            }
        }
    }

    // ─── UI elements ────────────────────────────────────────────────

    fn draw_event_banner(&self, buf: &mut Buffer, w: u16, phase: u8, pt: f32) {
        let event = match &self.event {
            Some(e) => e,
            None => return,
        };

        let text = event.banner_text();
        let char_count = text.chars().count() as u16;
        let banner_w = char_count + 6;
        let bx = (w.saturating_sub(banner_w)) / 2;
        let by: u16 = 1;

        let opacity = match phase {
            0 => pt.min(1.0),
            2 => (1.0 - pt).max(0.0),
            _ => 1.0,
        };
        if opacity < 0.1 { return; }

        for x in bx..bx + banner_w {
            if x < w && by < buf.area.height {
                self.set_cell_bg(buf, x, by, ' ', Color::Reset, Color::Rgb(10, 20, 40));
            }
        }

        let text_x = bx + 3;
        for (i, ch) in text.chars().enumerate() {
            let px = text_x + i as u16;
            if px < w {
                let hue = (i as f32 / text.len() as f32 + self.frame as f32 * 0.03) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                self.set_cell(buf, px, by, ch, Color::Rgb(
                    (r as f32 * opacity) as u8, (g as f32 * opacity) as u8, (b as f32 * opacity) as u8,
                ));
            }
        }

        let icon = self.species.icon();
        if bx > 0 {
            self.set_cell(buf, bx, by, icon, self.species.primary_color());
        }
        let end_x = bx + banner_w - 1;
        if end_x < w {
            self.set_cell(buf, end_x, by, icon, self.species.primary_color());
        }
    }

    fn draw_species_label(&self, buf: &mut Buffer, w: u16, h: u16) {
        let label = format!("[{}] {}", self.species.icon(), self.species.name());
        let label_y = h.saturating_sub(2);
        let label_x = 2u16;

        for (i, ch) in label.chars().enumerate() {
            let px = label_x + i as u16;
            if px < w && label_y < h {
                self.set_cell(buf, px, label_y, ch, self.species.secondary_color());
            }
        }
    }

    fn draw_entangle_line(&self, buf: &mut Buffer, w: u16, h: u16, x1: u16, y1: u16, x2: u16, y2: u16) {
        let steps = ((x2 as i16 - x1 as i16).abs().max((y2 as i16 - y1 as i16).abs())) as u16;
        if steps == 0 { return; }

        for s in 0..steps {
            let t = s as f32 / steps as f32;
            let px = (x1 as f32 + (x2 as f32 - x1 as f32) * t) as u16;
            let py = (y1 as f32 + (y2 as f32 - y1 as f32) * t) as u16;
            if px < w && py < h && s % 2 == 0 {
                let hue = (t + self.frame as f32 * 0.05) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                self.set_cell(buf, px, py, '.', Color::Rgb(r / 2, g / 2, b / 2));
            }
        }
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
