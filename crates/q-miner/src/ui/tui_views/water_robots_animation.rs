//! Water Robots Animation: Quantum marine creatures surfing across the miner TUI
//!
//! Triggered by important events (block found, reward, sync milestone, new peer).
//! Each trigger spawns a different water robot species with unique visual behavior:
//!
//!   🪼 Quantum Jellyfish   — drifts across with trailing tentacles, bioluminescent glow
//!   🐬 Entangled Dolphin   — leaps in arcs across the screen, splash particles
//!   🐙 Tunneling Octopus   — phases through the TUI content, ink cloud
//!   🐋 Wave-Particle Whale — massive wave that rolls across the buffer
//!   🦄 Superposition Seahorse — appears in multiple positions simultaneously
//!   🦠 Nano Quantumonas    — swarm of tiny particles flowing like a current
//!   🐟 Schooling Robotichthys — school of fish moving in formation
//!   🦈 Cyber Cetus          — large guardian sweeping across with sonar rings
//!
//! Each animation runs ~4-8 seconds, overlaying on the buffer, then dissolves.
//! Never repeats the same robot twice in a row.

use ratatui::{
    buffer::Buffer,
    layout::Position,
    style::Color,
};

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const SEED: u64 = 0xAC0A_B075_F150_CA5E;

/// Duration in ticks (250ms each) for each animation phase
const ENTER_TICKS: u16 = 12;   // Swim in / appear
const DISPLAY_TICKS: u16 = 16; // Main show
const EXIT_TICKS: u16 = 10;    // Swim out / dissolve
const TOTAL_TICKS: u16 = ENTER_TICKS + DISPLAY_TICKS + EXIT_TICKS; // ~9.5 seconds

// Water / wave characters
const WAVE_CHARS: [char; 8] = ['~', '≈', '∿', '≋', '〜', '∼', '～', '≈'];
const BUBBLE_CHARS: [char; 5] = ['°', '•', '○', '◦', '∘'];
const SPLASH_CHARS: [char; 6] = ['*', '✦', '·', '∙', '⁕', '✧'];

// ═══════════════════════════════════════════════════════════════════
// Robot Species
// ═══════════════════════════════════════════════════════════════════

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
            Self::QuantumJellyfish,
            Self::EntangledDolphin,
            Self::TunnelingOctopus,
            Self::WaveParticleWhale,
            Self::SuperpositionSeahorse,
            Self::NanoQuantumonas,
            Self::SchoolingRobotichthys,
            Self::CyberCetus,
        ]
    }

    fn icon(&self) -> char {
        match self {
            Self::QuantumJellyfish => '🪼',
            Self::EntangledDolphin => '🐬',
            Self::TunnelingOctopus => '🐙',
            Self::WaveParticleWhale => '🐋',
            Self::SuperpositionSeahorse => '🦄',
            Self::NanoQuantumonas => '🦠',
            Self::SchoolingRobotichthys => '🐟',
            Self::CyberCetus => '🦈',
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
            Self::QuantumJellyfish => Color::Rgb(200, 100, 255),   // Purple bioluminescence
            Self::EntangledDolphin => Color::Rgb(0, 200, 255),     // Ocean blue
            Self::TunnelingOctopus => Color::Rgb(255, 80, 120),    // Deep red-pink
            Self::WaveParticleWhale => Color::Rgb(60, 120, 200),   // Deep ocean blue
            Self::SuperpositionSeahorse => Color::Rgb(255, 200, 0),// Gold shimmer
            Self::NanoQuantumonas => Color::Rgb(0, 255, 150),      // Bio-green
            Self::SchoolingRobotichthys => Color::Rgb(255, 165, 0),// School orange
            Self::CyberCetus => Color::Rgb(0, 255, 255),           // Cyber cyan
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

// ═══════════════════════════════════════════════════════════════════
// Event types that trigger the animation
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum WaterRobotEvent {
    BlockFound { height: u64 },
    RewardReceived { amount: f64 },
    SyncMilestone { progress: f32 },
    NewPeerConnected { peer_count: u64 },
    HashrateMilestone { khs: f64 },
    UpdateAvailable { version: String },
}

impl WaterRobotEvent {
    fn banner_text(&self) -> String {
        match self {
            Self::BlockFound { height } => format!("BLOCK #{} FOUND!", format_commas(*height)),
            Self::RewardReceived { amount } => format!("REWARD: {:.4} QUG", amount),
            Self::SyncMilestone { progress } => format!("SYNC {:.0}% REACHED!", progress),
            Self::NewPeerConnected { peer_count } => format!("{} PEERS CONNECTED", peer_count),
            Self::HashrateMilestone { khs } => format!("HASHRATE: {:.1} KH/s!", khs),
            Self::UpdateAvailable { version } => format!("UPDATE {} AVAILABLE!", version),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main Animation State
// ═══════════════════════════════════════════════════════════════════

pub struct WaterRobotsAnimation {
    frame: u16,
    active: bool,
    species: RobotSpecies,
    event: Option<WaterRobotEvent>,
    seed: u64,
    last_species_idx: usize,       // Avoid repeating same species
    cooldown: u16,                  // Minimum ticks between animations
}

impl WaterRobotsAnimation {
    pub fn new() -> Self {
        Self {
            frame: 0,
            active: false,
            species: RobotSpecies::QuantumJellyfish,
            event: None,
            seed: SEED,
            last_species_idx: 99, // Impossible index = no last species
            cooldown: 0,
        }
    }

    /// Trigger a water robot animation for an important event.
    /// Picks a different species each time.
    pub fn trigger(&mut self, event: WaterRobotEvent) {
        if self.active || self.cooldown > 0 {
            return; // Don't interrupt running animation or cooldown
        }

        // Pick a new species (different from last)
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

    pub fn is_visible(&self) -> bool {
        self.active
    }

    pub fn tick(&mut self) {
        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
        if !self.active { return; }

        self.frame += 1;
        if self.frame >= TOTAL_TICKS {
            self.active = false;
            self.event = None;
            self.cooldown = 80; // ~20 seconds between animations
        }
    }

    // ─── Main render ─────────────────────────────────────────────

    pub fn render(&self, buf: &mut Buffer) {
        if !self.active { return; }

        let area = buf.area;
        let w = area.width;
        let h = area.height;
        if w < 40 || h < 12 { return; }

        // Animation progress (0.0 - 1.0 across all phases)
        let t = self.frame as f32 / TOTAL_TICKS as f32;

        // Phase determination
        let (phase, phase_t) = if self.frame < ENTER_TICKS {
            (0, self.frame as f32 / ENTER_TICKS as f32) // Enter
        } else if self.frame < ENTER_TICKS + DISPLAY_TICKS {
            (1, (self.frame - ENTER_TICKS) as f32 / DISPLAY_TICKS as f32) // Display
        } else {
            (2, (self.frame - ENTER_TICKS - DISPLAY_TICKS) as f32 / EXIT_TICKS as f32) // Exit
        };

        // Draw the water surface (bottom third of screen)
        self.draw_water_surface(buf, w, h, t);

        // Species-specific animation
        match self.species {
            RobotSpecies::QuantumJellyfish => self.draw_jellyfish(buf, w, h, phase, phase_t, t),
            RobotSpecies::EntangledDolphin => self.draw_dolphin(buf, w, h, phase, phase_t, t),
            RobotSpecies::TunnelingOctopus => self.draw_octopus(buf, w, h, phase, phase_t, t),
            RobotSpecies::WaveParticleWhale => self.draw_whale(buf, w, h, phase, phase_t, t),
            RobotSpecies::SuperpositionSeahorse => self.draw_seahorse(buf, w, h, phase, phase_t, t),
            RobotSpecies::NanoQuantumonas => self.draw_nano_swarm(buf, w, h, phase, phase_t, t),
            RobotSpecies::SchoolingRobotichthys => self.draw_school(buf, w, h, phase, phase_t, t),
            RobotSpecies::CyberCetus => self.draw_guardian(buf, w, h, phase, phase_t, t),
        }

        // Event banner (top, always visible)
        self.draw_event_banner(buf, w, phase, phase_t);

        // Species label (bottom left)
        self.draw_species_label(buf, w, h, phase_t);
    }

    // ═══════════════════════════════════════════════════════════════
    // Water Surface
    // ═══════════════════════════════════════════════════════════════

    fn draw_water_surface(&self, buf: &mut Buffer, w: u16, h: u16, t: f32) {
        let water_y = h * 2 / 3; // Water line at 2/3 down

        // Draw wave line
        for x in 0..w {
            let wave_offset = ((x as f32 * 0.15 + self.frame as f32 * 0.3).sin() * 1.5) as i16;
            let wy = (water_y as i16 + wave_offset).clamp(0, h as i16 - 1) as u16;

            let hue = (x as f32 / w as f32 * 0.3 + 0.55 + t * 0.05) % 1.0; // Blue-cyan range
            let wave_ch = WAVE_CHARS[(x as usize + self.frame as usize) % WAVE_CHARS.len()];
            self.set_cell(buf, x, wy, wave_ch, hue_to_color(hue));

            // Underwater gradient (below wave line)
            for y in (wy + 1)..h {
                let depth = (y - wy) as f32 / (h - wy) as f32;
                let alpha = (depth * 0.3).min(0.25);

                if let Some(cell) = buf.cell_mut(Position::new(x, y)) {
                    let (r, g, b) = color_to_rgb(cell.fg);
                    // Tint toward deep blue
                    cell.set_fg(Color::Rgb(
                        (r as f32 * (1.0 - alpha) + 20.0 * alpha) as u8,
                        (g as f32 * (1.0 - alpha) + 60.0 * alpha) as u8,
                        (b as f32 * (1.0 - alpha) + 120.0 * alpha) as u8,
                    ));
                }
            }
        }

        // Bubbles rising (scattered across water area)
        for i in 0..8u16 {
            let bx = (hash_raw(self.seed, i as u64, 1111, 0) % w as u64) as u16;
            let by = water_y + 2 + ((self.frame + i * 7) % (h - water_y).max(1));
            let rise = (self.frame / 3 + i * 5) % (h - water_y).max(1);
            let bubble_y = h.saturating_sub(rise + 1);
            if bubble_y > water_y && bubble_y < h {
                let ch = BUBBLE_CHARS[(i as usize + self.frame as usize / 2) % BUBBLE_CHARS.len()];
                self.set_cell(buf, bx, bubble_y, ch, Color::Rgb(100, 180, 255));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Species Animations
    // ═══════════════════════════════════════════════════════════════

    /// 🪼 Jellyfish: floats from right to left, trailing tentacles with bioluminescent glow
    fn draw_jellyfish(&self, buf: &mut Buffer, w: u16, h: u16, phase: u8, pt: f32, _t: f32) {
        let water_y = h * 2 / 3;

        // Position: enters from right, exits left
        let cx = match phase {
            0 => w + 5 - (pt * (w / 3) as f32) as u16,        // Enter from right
            1 => (w * 2 / 3) - (pt * (w / 3) as f32) as u16,  // Cross screen
            _ => (w / 3).saturating_sub((pt * (w / 2) as f32) as u16),  // Exit left
        };
        let cy = water_y.saturating_sub(3) + ((self.frame as f32 * 0.2).sin() * 2.0) as u16;

        if cx >= w { return; }

        // Jellyfish body (bell)
        let bell = ["  ╭───╮  ", " ╭─────╮ ", "╭───────╮", "╰~~~~~╯ "];
        for (i, line) in bell.iter().enumerate() {
            let ly = cy + i as u16;
            if ly >= h { continue; }
            for (j, ch) in line.chars().enumerate() {
                let lx = cx.saturating_sub(4) + j as u16;
                if lx < w && ch != ' ' {
                    let glow_hue = (j as f32 / 9.0 + self.frame as f32 * 0.05) % 1.0;
                    let glow = hue_to_color(glow_hue * 0.3 + 0.7); // Purple-pink range
                    self.set_cell(buf, lx, ly, ch, glow);
                }
            }
        }

        // Trailing tentacles (wavy lines below bell)
        for tent in 0..5u16 {
            let tx = cx.saturating_sub(3) + tent * 2;
            for dy in 0..4u16 {
                let ty = cy + 4 + dy;
                if ty >= h || tx >= w { continue; }
                let wave = ((self.frame as f32 * 0.3 + tent as f32 * 1.5 + dy as f32).sin() * 1.0) as i16;
                let tentx = (tx as i16 + wave).clamp(0, w as i16 - 1) as u16;
                let brightness = 255u8.saturating_sub(dy as u8 * 40);
                self.set_cell(buf, tentx, ty, '│', Color::Rgb(brightness, brightness / 3, brightness));
            }
        }

        // Bioluminescent glow particles
        for i in 0..6u16 {
            let angle = self.frame as f32 * 0.15 + i as f32 * std::f32::consts::PI / 3.0;
            let gx = (cx as f32 + angle.cos() * 5.0) as u16;
            let gy = (cy as f32 + angle.sin() * 3.0) as u16;
            if gx < w && gy < h {
                let hue = (i as f32 / 6.0 + self.frame as f32 * 0.04) % 1.0;
                self.set_cell(buf, gx, gy, '✦', hue_to_color(hue));
            }
        }
    }

    /// 🐬 Dolphin: leaps in arcs, splash particles
    fn draw_dolphin(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        let water_y = h * 2 / 3;

        // Arc trajectory: parabolic leap
        let progress = t; // 0.0 → 1.0 across full animation
        let cx = (progress * w as f32 * 1.2 - w as f32 * 0.1) as i16;
        // Parabolic: y = -4 * (x-0.5)^2 + 1  →  peak at center
        let leap_height = (h / 3) as f32;
        let arc = -4.0 * (progress - 0.5).powi(2) + 1.0;
        let cy = (water_y as f32 - arc * leap_height) as i16;

        // Dolphin ASCII (oriented by direction of travel)
        let dolphin = if arc > 0.3 {
            // Leaping up: angled
            [" ╭──╮ ", "╭╯🐬╰╮", "╰────╯"]
        } else {
            // In water: horizontal
            ["       ", " ~🐬~  ", "  ~~~  "]
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

        // Splash particles at water entry/exit points
        if arc.abs() < 0.15 { // Near water surface
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

    /// 🐙 Octopus: phases through content with ink cloud
    fn draw_octopus(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        // Octopus moves diagonally across screen
        let cx = (t * w as f32 * 1.3 - w as f32 * 0.15) as i16;
        let cy = (h as f32 * 0.3 + (t * 6.0).sin() as f32 * h as f32 * 0.15) as i16;

        // Draw ink cloud trail
        for i in 0..15u16 {
            let trail_t = t - (i as f32 * 0.015);
            if trail_t < 0.0 { continue; }
            let ix = (trail_t * w as f32 * 1.3 - w as f32 * 0.15) as i16;
            let iy = (h as f32 * 0.3 + (trail_t * 6.0).sin() as f32 * h as f32 * 0.15) as i16;
            let spread = i as i16;
            for _ in 0..3 {
                let dx = ix + (hash_raw(self.seed, i as u64, self.frame as u64, 222) % (spread as u64 * 2 + 1)) as i16 - spread;
                let dy = iy + (hash_raw(self.seed, i as u64, self.frame as u64, 333) % (spread as u64 + 1)) as i16;
                if dx >= 0 && dx < w as i16 && dy >= 0 && dy < h as i16 {
                    let fade = 1.0 - i as f32 / 15.0;
                    let r = (80.0 * fade) as u8;
                    let g = (20.0 * fade) as u8;
                    let b = (60.0 * fade) as u8;
                    self.set_cell(buf, dx as u16, dy as u16, '░', Color::Rgb(r, g, b));
                }
            }
        }

        // Octopus body
        let octo = ["  ╭─╮  ", " /🐙\\ ", "╰┤╰┤╰┤"];
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

        // Waving tentacles below
        for arm in 0..8u16 {
            for seg in 0..3u16 {
                let wave = ((self.frame as f32 * 0.25 + arm as f32 * 0.8 + seg as f32).sin() * 2.0) as i16;
                let ax = cx - 3 + arm as i16 + wave;
                let ay = cy + 3 + seg as i16;
                if ax >= 0 && ax < w as i16 && ay >= 0 && ay < h as i16 {
                    let fade = 1.0 - seg as f32 / 3.0;
                    let color = RobotSpecies::TunnelingOctopus.primary_color();
                    let (r, g, b) = color_to_rgb(color);
                    self.set_cell(buf, ax as u16, ay as u16, '╲', Color::Rgb(
                        (r as f32 * fade) as u8,
                        (g as f32 * fade) as u8,
                        (b as f32 * fade) as u8,
                    ));
                }
            }
        }
    }

    /// 🐋 Whale: massive wave rolling across
    fn draw_whale(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        let water_y = h * 2 / 3;

        // Whale position (slow, majestic movement)
        let cx = (t * w as f32 * 1.4 - w as f32 * 0.2) as i16;
        let cy = water_y as i16 - 4;

        // Big wave in front of whale
        let wave_front = cx + 8;
        if wave_front > 0 && wave_front < w as i16 {
            for dy in -3i16..4 {
                let wy = (water_y as i16 + dy).clamp(0, h as i16 - 1) as u16;
                let wave_h = (3.0 - (dy as f32).abs()) / 3.0;
                let wave_x = (wave_front + (dy.abs() * 2)) as u16;
                if wave_x < w {
                    let ch = if dy < 0 { '/' } else { '\\' };
                    let blue = (150.0 * wave_h) as u8;
                    self.set_cell(buf, wave_x, wy, ch, Color::Rgb(80, 150 + blue / 3, 200 + blue / 5));
                }
            }
        }

        // Whale body (large)
        let whale_art = [
            "      ╭────────╮      ",
            "   ╭──┤  🐋    ├──╮   ",
            "  ╭┤  │        │  ├╮  ",
            "──┤   ╰────────╯   ├──",
            "  ╰─────╮    ╭─────╯  ",
            "        ╰────╯        ",
        ];

        for (i, line) in whale_art.iter().enumerate() {
            let dy = cy + i as i16;
            if dy < 0 || dy >= h as i16 { continue; }
            for (j, ch) in line.chars().enumerate() {
                let dx = cx - 11 + j as i16;
                if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                let depth = i as f32 / whale_art.len() as f32;
                let color = Color::Rgb(
                    (60.0 + 40.0 * depth) as u8,
                    (120.0 + 80.0 * (1.0 - depth)) as u8,
                    (200.0 + 40.0 * depth) as u8,
                );
                self.set_cell(buf, dx as u16, dy as u16, ch, color);
            }
        }

        // Water spout (blow hole)
        if self.frame % 20 < 10 {
            let spout_x = (cx + 2).clamp(0, w as i16 - 1) as u16;
            for dy in 1..5u16 {
                let sy = cy.saturating_sub(dy as i16).max(0) as u16;
                if sy < h && spout_x < w {
                    let fade = 1.0 - dy as f32 / 5.0;
                    self.set_cell(buf, spout_x, sy, '│', Color::Rgb(
                        (180.0 * fade) as u8,
                        (220.0 * fade) as u8,
                        (255.0 * fade) as u8,
                    ));
                }
            }
            // Spray at top
            let spray_y = cy.saturating_sub(5).max(0) as u16;
            if spray_y < h {
                for dx in -2i16..3 {
                    let sx = (spout_x as i16 + dx).clamp(0, w as i16 - 1) as u16;
                    if sx < w {
                        self.set_cell(buf, sx, spray_y, '·', Color::Rgb(200, 230, 255));
                    }
                }
            }
        }
    }

    /// 🦄 Seahorse: appears in multiple positions simultaneously (quantum superposition)
    fn draw_seahorse(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        let water_y = h * 2 / 3;

        // 3 superposition positions (ghost + solid + ghost)
        let positions: [(i16, i16, f32); 3] = [
            ((w as f32 * 0.25 + (t * 3.0).sin() * 5.0) as i16,
             (water_y as f32 - 4.0 + (t * 4.0).cos() * 2.0) as i16, 0.4),
            ((w as f32 * 0.5 + (t * 2.5).cos() * 4.0) as i16,
             (water_y as f32 - 6.0 + (t * 3.0).sin() * 3.0) as i16, 1.0),
            ((w as f32 * 0.75 + (t * 3.5).sin() * 3.0) as i16,
             (water_y as f32 - 3.0 + (t * 5.0).cos() * 2.0) as i16, 0.4),
        ];

        // Quantum entanglement lines between positions
        let (x1, y1, _) = positions[0];
        let (x2, y2, _) = positions[1];
        let (x3, y3, _) = positions[2];
        self.draw_entangle_line(buf, w, h, x1 as u16, y1 as u16, x2 as u16, y2 as u16);
        self.draw_entangle_line(buf, w, h, x2 as u16, y2 as u16, x3 as u16, y3 as u16);

        for (px, py, opacity) in &positions {
            let seahorse = ["  ? ", " 🦄 ", " ╰╯ "];
            for (i, line) in seahorse.iter().enumerate() {
                let dy = py + i as i16;
                if dy < 0 || dy >= h as i16 { continue; }
                for (j, ch) in line.chars().enumerate() {
                    let dx = px + j as i16;
                    if dx < 0 || dx >= w as i16 || ch == ' ' { continue; }
                    let (r, g, b) = color_to_rgb(RobotSpecies::SuperpositionSeahorse.primary_color());
                    let color = Color::Rgb(
                        (r as f32 * opacity) as u8,
                        (g as f32 * opacity) as u8,
                        (b as f32 * opacity) as u8,
                    );
                    self.set_cell(buf, dx as u16, dy as u16, ch, color);
                }
            }

            // Quantum sparkle around each position
            if self.hash(*px as u64, *py as u64, self.frame as u64) % 3 == 0 {
                let sx = (*px + 3).clamp(0, w as i16 - 1) as u16;
                let sy = (*py - 1).clamp(0, h as i16 - 1) as u16;
                if sx < w && sy < h {
                    let hue = (self.frame as f32 * 0.06) % 1.0;
                    self.set_cell(buf, sx, sy, '✦', hue_to_color(hue));
                }
            }
        }
    }

    /// 🦠 Nano swarm: current of tiny particles flowing
    fn draw_nano_swarm(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        let water_y = h * 2 / 3;
        let num_particles = 60u16;

        for i in 0..num_particles {
            let base_x = (hash_raw(self.seed, i as u64, 0, 0) % w as u64) as f32;
            let base_y = water_y as f32 - 2.0 + (hash_raw(self.seed, i as u64, 1, 0) % (h as u64 / 3)) as f32;

            // Flow pattern: moving right with sinusoidal vertical motion
            let flow_speed = 0.5 + (hash_raw(self.seed, i as u64, 2, 0) % 50) as f32 / 100.0;
            let px = ((base_x + self.frame as f32 * flow_speed) % (w as f32 + 10.0)) as u16;
            let wave = ((self.frame as f32 * 0.2 + i as f32 * 0.5).sin() * 2.0) as i16;
            let py = (base_y as i16 + wave).clamp(0, h as i16 - 1) as u16;

            if px < w && py < h {
                let hue = (i as f32 / num_particles as f32 * 0.3 + 0.25 + self.frame as f32 * 0.01) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                // Size/brightness varies
                let brightness = 0.5 + (hash_raw(self.seed, i as u64, 3, 0) % 50) as f32 / 100.0;
                let ch = if i % 3 == 0 { '•' } else if i % 3 == 1 { '·' } else { '∙' };
                self.set_cell(buf, px, py, ch, Color::Rgb(
                    (r as f32 * brightness) as u8,
                    (g as f32 * brightness) as u8,
                    (b as f32 * brightness) as u8,
                ));
            }
        }

        // Central 🦠 icon moving through the swarm
        let icon_x = ((t * w as f32 * 1.2 - w as f32 * 0.1) as u16).min(w.saturating_sub(1));
        let icon_y = water_y.saturating_sub(4) + ((self.frame as f32 * 0.15).sin() * 2.0) as u16;
        if icon_x < w && icon_y < h {
            self.set_cell(buf, icon_x, icon_y, '🦠', Color::Rgb(0, 255, 150));
        }
    }

    /// 🐟 School of fish: formation movement
    fn draw_school(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        let water_y = h * 2 / 3;
        let school_size = 12u16;

        // Leader position
        let leader_x = (t * w as f32 * 1.3 - w as f32 * 0.15) as f32;
        let leader_y = (water_y as f32 - 5.0 + (t * 5.0).sin() * 3.0) as f32;

        // V-formation behind leader
        for i in 0..school_size {
            let row = i / 2 + 1;
            let side = if i % 2 == 0 { 1.0f32 } else { -1.0 };
            let offset_x = -(row as f32) * 3.0;
            let offset_y = side * row as f32 * 1.5;

            // Individual fish wobble
            let wobble_x = (self.frame as f32 * 0.3 + i as f32 * 1.7).sin() * 0.5;
            let wobble_y = (self.frame as f32 * 0.4 + i as f32 * 2.3).cos() * 0.3;

            let fx = (leader_x + offset_x + wobble_x) as i16;
            let fy = (leader_y + offset_y + wobble_y) as i16;

            if fx >= 0 && fx < w as i16 && fy >= 0 && fy < h as i16 {
                let hue = (i as f32 / school_size as f32 * 0.2 + 0.05 + self.frame as f32 * 0.008) % 1.0;
                // Fish characters alternate
                // Fish characters alternate direction
                let ch = if (fx + self.frame as i16) % 2 == 0 { '>' } else { '<' };
                self.set_cell(buf, fx as u16, fy as u16, ch,
                    Color::Rgb(255, 165 + (i as u8 * 5), 0));
            }
        }

        // Leader fish (larger, brighter)
        let lx = leader_x as i16;
        let ly = leader_y as i16;
        if lx >= 0 && lx < w as i16 - 2 && ly >= 0 && ly < h as i16 {
            self.set_cell(buf, lx as u16, ly as u16, '🐟', Color::Rgb(255, 200, 50));
        }
    }

    /// 🦈 Guardian: large sweeping presence with sonar rings
    fn draw_guardian(&self, buf: &mut Buffer, w: u16, h: u16, _phase: u8, _pt: f32, t: f32) {
        let water_y = h * 2 / 3;

        // Guardian position (slow patrol)
        let cx = (t * w as f32 * 1.2 - w as f32 * 0.1) as i16;
        let cy = water_y as i16 - 3;

        // Sonar rings (expanding circles)
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
                    self.set_cell(buf, rx as u16, ry as u16, '·', Color::Rgb(
                        (r as f32 * opacity) as u8,
                        (g as f32 * opacity) as u8,
                        (b as f32 * opacity) as u8,
                    ));
                }
            }
        }

        // Guardian shark body
        let shark = [
            "    ╱▲╲     ",
            "  ╱─🦈─╲   ",
            "╱───────╲──╲",
            "╲───────╱──╱",
            "  ╲─────╱   ",
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

    // ═══════════════════════════════════════════════════════════════
    // UI Elements
    // ═══════════════════════════════════════════════════════════════

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

        // Fade banner in/out
        let opacity = match phase {
            0 => pt.min(1.0),
            2 => (1.0 - pt).max(0.0),
            _ => 1.0,
        };

        if opacity < 0.1 { return; }

        // Banner background
        for x in bx..bx + banner_w {
            if x < w && by < buf.area.height {
                if let Some(cell) = buf.cell_mut(Position::new(x, by)) {
                    cell.set_char(' ');
                    cell.set_bg(Color::Rgb(10, 20, 40));
                }
            }
        }

        // Rainbow text
        let text_x = bx + 3;
        for (i, ch) in text.chars().enumerate() {
            let px = text_x + i as u16;
            if px < w {
                let hue = (i as f32 / text.len() as f32 + self.frame as f32 * 0.03) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                let color = Color::Rgb(
                    (r as f32 * opacity) as u8,
                    (g as f32 * opacity) as u8,
                    (b as f32 * opacity) as u8,
                );
                self.set_cell(buf, px, by, ch, color);
            }
        }

        // Border sparkle
        let icon = self.species.icon();
        if bx > 0 {
            self.set_cell(buf, bx, by, icon, self.species.primary_color());
        }
        let end_x = bx + banner_w - 1;
        if end_x < w {
            self.set_cell(buf, end_x, by, icon, self.species.primary_color());
        }
    }

    fn draw_species_label(&self, buf: &mut Buffer, w: u16, h: u16, _pt: f32) {
        let label = format!("{} {}", self.species.icon(), self.species.name());
        let label_y = h.saturating_sub(2);
        let label_x = 2;

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
            if px < w && py < h && s % 2 == 0 { // Dashed
                let hue = (t + self.frame as f32 * 0.05) % 1.0;
                let (r, g, b) = hue_to_rgb(hue);
                self.set_cell(buf, px, py, '·', Color::Rgb(r / 2, g / 2, b / 2));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════

    fn set_cell(&self, buf: &mut Buffer, x: u16, y: u16, ch: char, fg: Color) {
        if let Some(cell) = buf.cell_mut(Position::new(x, y)) {
            cell.set_char(ch);
            cell.set_fg(fg);
        }
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

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle() {
        let mut anim = WaterRobotsAnimation::new();
        assert!(!anim.is_visible());

        anim.trigger(WaterRobotEvent::BlockFound { height: 1000 });
        assert!(anim.is_visible());

        // Run through entire animation
        for _ in 0..TOTAL_TICKS {
            anim.tick();
        }
        assert!(!anim.is_visible());
    }

    #[test]
    fn test_no_repeat_species() {
        let mut anim = WaterRobotsAnimation::new();

        anim.trigger(WaterRobotEvent::BlockFound { height: 1 });
        let first = anim.species;

        // Complete the animation + cooldown
        for _ in 0..(TOTAL_TICKS + 100) {
            anim.tick();
        }

        anim.trigger(WaterRobotEvent::RewardReceived { amount: 1.0 });
        assert_ne!(anim.species, first, "Should not repeat same species");
    }

    #[test]
    fn test_cooldown_prevents_spam() {
        let mut anim = WaterRobotsAnimation::new();

        anim.trigger(WaterRobotEvent::BlockFound { height: 1 });
        for _ in 0..TOTAL_TICKS {
            anim.tick();
        }

        // Still in cooldown
        anim.trigger(WaterRobotEvent::BlockFound { height: 2 });
        assert!(!anim.is_visible(), "Cooldown should prevent immediate re-trigger");
    }

    #[test]
    fn test_all_species_have_colors() {
        for species in RobotSpecies::all() {
            let _ = species.primary_color();
            let _ = species.secondary_color();
            let _ = species.icon();
            let _ = species.name();
        }
    }

    #[test]
    fn test_event_banner_text() {
        let e = WaterRobotEvent::BlockFound { height: 12345 };
        assert!(e.banner_text().contains("12,345"));

        let e = WaterRobotEvent::RewardReceived { amount: 1.5 };
        assert!(e.banner_text().contains("1.5000"));
    }
}
