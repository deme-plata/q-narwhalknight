//! Swarm Ocean: Animated ASCII ocean showing P2P network nodes as sea creatures
//!
//! Part of the miner TUI Command Center. Each peer is a sea creature swimming
//! through an animated wave background, exchanging data particles:
//!
//!   Narwhal  --=<(((*>   You (the local miner)
//!   Whale    ~~~<((({o>  Supernode (10Gbit)
//!   Dolphin  ~<(({o>    Full node (1Gbit)
//!   Shark    -<({o>     Bootstrap (100Mbit)
//!   Fish     <{o>       Light node


use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Color,
};


const WAVE_CHARS: [char; 4] = ['~', '=', '-', '~'];

const WAVE_BG: Color = Color::Rgb(20, 40, 80);

const WAVE_FG_DARK: Color = Color::Rgb(30, 60, 110);

const WAVE_FG_MID: Color = Color::Rgb(40, 80, 130);

const SEED: u64 = 0xDE4D_0CE4_F150_CA5E;


fn splitmix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ═══════════════════════════════════════════════════════════════════
// Creature Kind
// ═══════════════════════════════════════════════════════════════════


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CreatureKind {
    Narwhal,
    Whale,
    Dolphin,
    Shark,
    Fish,
}


impl CreatureKind {
    fn sprite_right(&self) -> &'static str {
        match self {
            Self::Narwhal => "--=<(((*>",
            Self::Whale   => "~~~<((({o>",
            Self::Dolphin => "~<(({o>",
            Self::Shark   => "-<({o>",
            Self::Fish    => "<{o>",
        }
    }

    fn sprite_left(&self) -> &'static str {
        match self {
            Self::Narwhal => "<*)))))>=--",
            Self::Whale   => "<o})))>~~~",
            Self::Dolphin => "<o})>~",
            Self::Shark   => "<o})>-",
            Self::Fish    => "<o}>",
        }
    }

    fn color(&self) -> Color {
        match self {
            Self::Narwhal => Color::Rgb(0, 220, 255),
            Self::Whale   => Color::Rgb(60, 120, 220),
            Self::Dolphin => Color::Rgb(100, 180, 255),
            Self::Shark   => Color::Rgb(160, 170, 180),
            Self::Fish    => Color::Rgb(80, 220, 120),
        }
    }

    fn random_peer(seed: u64) -> Self {
        match seed % 4 {
            0 => Self::Whale,
            1 => Self::Dolphin,
            2 => Self::Shark,
            _ => Self::Fish,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Particle Kind
// ═══════════════════════════════════════════════════════════════════


#[derive(Debug, Clone, Copy)]
enum ParticleKind {
    Block,
    Gossip,
    Solution,
}


impl ParticleKind {
    fn ch(&self) -> char {
        match self {
            Self::Block    => '*',
            Self::Gossip   => '.',
            Self::Solution => '+',
        }
    }

    fn color(&self) -> Color {
        match self {
            Self::Block    => Color::Rgb(255, 220, 50),
            Self::Gossip   => Color::Rgb(0, 200, 220),
            Self::Solution => Color::Rgb(50, 255, 100),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Data Particle & Swarm Creature
// ═══════════════════════════════════════════════════════════════════


#[derive(Debug, Clone)]
struct DataParticle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    lifetime: u8,
    kind: ParticleKind,
}


#[derive(Debug, Clone)]
struct SwarmCreature {
    kind: CreatureKind,
    name: String,
    x: f32,
    y: f32,
    vx: f32,
    facing_right: bool,
    bob_phase: f32,
}


impl SwarmCreature {
    fn new(kind: CreatureKind, name: String, x: f32, y: f32, vx: f32) -> Self {
        Self { kind, name, x, y, vx, facing_right: vx >= 0.0, bob_phase: x * 0.7 }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Swarm Ocean (public API)
// ═══════════════════════════════════════════════════════════════════


pub struct SwarmOcean {
    tick: u32,
    creatures: Vec<SwarmCreature>,
    particles: Vec<DataParticle>,
    wave_phase: f64,
    rng_state: u64,
}


impl SwarmOcean {
    /// Create with the local narwhal at center.
    pub fn new() -> Self {
        let narwhal = SwarmCreature::new(
            CreatureKind::Narwhal, "You".to_string(), 40.0, 6.0, 0.3,
        );
        Self {
            tick: 0,
            creatures: vec![narwhal],
            particles: Vec::new(),
            wave_phase: 0.0,
            rng_state: SEED,
        }
    }

    /// Advance one animation frame.
    pub fn tick(&mut self, peer_count: u32) {
        self.tick = self.tick.wrapping_add(1);
        self.wave_phase += 0.15;
        self.rng_state = splitmix(self.rng_state.wrapping_add(self.tick as u64));

        // Move creatures with sinusoidal bob
        for c in self.creatures.iter_mut() {
            c.x += c.vx;
            c.bob_phase += 0.12;
        }

        // Bounce off edges
        for c in self.creatures.iter_mut() {
            if c.x < 2.0 {
                c.x = 2.0;
                c.vx = c.vx.abs();
                c.facing_right = true;
            } else if c.x > 75.0 {
                c.x = 75.0;
                c.vx = -c.vx.abs();
                c.facing_right = false;
            }
        }

        // Advance particles, remove dead
        for p in self.particles.iter_mut() {
            p.x += p.vx;
            p.y += p.vy;
            p.lifetime = p.lifetime.saturating_sub(1);
        }
        self.particles.retain(|p| p.lifetime > 0);
        if self.particles.len() > 80 {
            self.particles.drain(0..(self.particles.len() - 60));
        }

        // Spawn gossip particles between creatures (1 in 8 ticks)
        if self.tick % 8 == 0 && self.creatures.len() >= 2 {
            let idx_a = (self.rng_state as usize) % self.creatures.len();
            let idx_b = (splitmix(self.rng_state) as usize) % self.creatures.len();
            if idx_a != idx_b {
                let (ax, ay) = (self.creatures[idx_a].x, self.creatures[idx_a].y);
                let (bx, by) = (self.creatures[idx_b].x, self.creatures[idx_b].y);
                let (dx, dy) = (bx - ax, by - ay);
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                self.particles.push(DataParticle {
                    x: ax, y: ay,
                    vx: dx / dist * 1.2, vy: dy / dist * 1.2,
                    lifetime: (dist / 1.2) as u8 + 2,
                    kind: ParticleKind::Gossip,
                });
            }
        }

        // Sync creature count to peer_count (+1 for our narwhal)
        let target = (peer_count as usize).saturating_add(1);
        let known_names = ["Epsilon", "Beta", "Gamma", "Delta", "Alpha",
                           "Relay-1", "Relay-2", "Relay-3"];
        let known_kinds = [
            CreatureKind::Whale, CreatureKind::Shark, CreatureKind::Dolphin,
            CreatureKind::Dolphin, CreatureKind::Fish, CreatureKind::Fish,
            CreatureKind::Fish, CreatureKind::Fish,
        ];
        while self.creatures.len() < target {
            let idx = self.creatures.len() - 1; // -1 because narwhal is index 0
            let s = splitmix(self.rng_state.wrapping_add(self.creatures.len() as u64 * 7));
            let kind = if idx < known_kinds.len() { known_kinds[idx] } else { CreatureKind::random_peer(s) };
            let name = if idx < known_names.len() { known_names[idx].to_string() } else { format!("Peer {}", idx) };
            let x = 5.0 + (s % 65) as f32;
            let y = 3.0 + (splitmix(s) % 10) as f32;
            let vx_mag = 0.2 + (s % 5) as f32 * 0.08;
            let vx = if s % 2 == 0 { vx_mag } else { -vx_mag };
            self.creatures.push(SwarmCreature::new(kind, name, x, y, vx));
        }
        while self.creatures.len() > target && self.creatures.len() > 1 {
            self.creatures.pop();
        }
    }

    /// Spawn block particles radiating from a random creature.
    pub fn spawn_block_event(&mut self) {
        if self.creatures.is_empty() { return; }
        let idx = (self.rng_state as usize) % self.creatures.len();
        let (cx, cy) = (self.creatures[idx].x, self.creatures[idx].y);
        for i in 0..6 {
            let angle = (i as f32) * std::f32::consts::TAU / 6.0;
            self.particles.push(DataParticle {
                x: cx, y: cy,
                vx: angle.cos() * 0.8, vy: angle.sin() * 0.4,
                lifetime: 12, kind: ParticleKind::Block,
            });
        }
    }

    /// Spawn solution particles from the narwhal (index 0).
    pub fn spawn_solution_event(&mut self) {
        if self.creatures.is_empty() { return; }
        let (cx, cy) = (self.creatures[0].x, self.creatures[0].y);
        for i in 0..8 {
            let angle = (i as f32) * std::f32::consts::TAU / 8.0;
            self.particles.push(DataParticle {
                x: cx, y: cy,
                vx: angle.cos() * 1.0, vy: angle.sin() * 0.5,
                lifetime: 15, kind: ParticleKind::Solution,
            });
        }
    }

    /// Render the ocean into the given buffer region.
    pub fn draw(&self, buf: &mut Buffer, area: Rect) {
        if area.width < 10 || area.height < 5 { return; }

        let (left, top, right, bottom) = (area.left(), area.top(), area.right(), area.bottom());
        let (w, h) = (area.width as f32, area.height as f32);
        let tick = self.tick;

        // 1) Animated wave background
        for row in top..bottom {
            let ry = (row - top) as u32;
            for col in left..right {
                let cx = (col - left) as u32;
                let idx = (cx.wrapping_add(tick).wrapping_add(ry.wrapping_mul(3))) % 4;
                let fg = if (ry + cx) % 3 == 0 { WAVE_FG_MID } else { WAVE_FG_DARK };
                let cell = buf.get_mut(col, row);
                cell.set_char(WAVE_CHARS[idx as usize]).set_fg(fg).set_bg(WAVE_BG);
            }
        }

        // 2) Draw creatures
        for creature in &self.creatures {
            let cx = creature.x.clamp(0.0, w - 1.0);
            let bob = (creature.bob_phase as f64).sin() as f32 * 0.8;
            let cy = (creature.y + bob).clamp(0.0, h - 2.0);
            let (px, py) = (left + cx as u16, top + cy as u16);

            let sprite = if creature.facing_right {
                creature.kind.sprite_right()
            } else {
                creature.kind.sprite_left()
            };
            let color = creature.kind.color();

            for (i, ch) in sprite.chars().enumerate() {
                let sx = px.wrapping_add(i as u16);
                if sx >= right { break; }
                if sx >= left && py >= top && py < bottom {
                    let cell = buf.get_mut(sx, py);
                    cell.set_char(ch).set_fg(color).set_bg(WAVE_BG);
                }
            }

            // Name label one row below (dim)
            let label_y = py.saturating_add(1);
            if label_y < bottom {
                let lx_start = px.saturating_add(1);
                let lc = Color::Rgb(100, 120, 140);
                for (i, ch) in creature.name.chars().enumerate() {
                    let lx = lx_start.wrapping_add(i as u16);
                    if lx >= right { break; }
                    if lx >= left {
                        let cell = buf.get_mut(lx, label_y);
                        cell.set_char(ch).set_fg(lc).set_bg(WAVE_BG);
                    }
                }
            }
        }

        // 2b) Bubble trails behind larger creatures
        let bubble_chars = ['o', '.', '*', '`'];
        for creature in &self.creatures {
            let is_large = matches!(creature.kind, CreatureKind::Narwhal | CreatureKind::Whale | CreatureKind::Dolphin);
            if !is_large { continue; }
            let cx = creature.x.clamp(0.0, w - 1.0);
            let bob = (creature.bob_phase as f64).sin() as f32 * 0.8;
            let cy = (creature.y + bob).clamp(0.0, h - 2.0);
            let trail_dir: f32 = if creature.facing_right { -1.0 } else { 1.0 };
            let sprite_len = creature.kind.sprite_right().len() as f32;
            for b in 0..3 {
                let bx = cx + trail_dir * (sprite_len + 1.0 + b as f32 * 2.0);
                let by = cy + ((self.tick as f32 * 0.3 + b as f32).sin() * 0.5);
                let px = left + bx.clamp(0.0, w - 1.0) as u16;
                let py = top + by.clamp(0.0, h - 1.0) as u16;
                if px >= left && px < right && py >= top && py < bottom {
                    let bc = bubble_chars[((self.tick as usize + b) / 2) % bubble_chars.len()];
                    let dim = 80u8.saturating_sub(b as u8 * 20);
                    let cell = buf.get_mut(px, py);
                    cell.set_char(bc).set_fg(Color::Rgb(dim, dim + 40, dim + 80)).set_bg(WAVE_BG);
                }
            }
        }

        // 3) Draw data particles with lifetime fade
        for p in &self.particles {
            let px = left + (p.x.clamp(0.0, w - 1.0)) as u16;
            let py = top + (p.y.clamp(0.0, h - 1.0)) as u16;
            if px >= left && px < right && py >= top && py < bottom {
                let base = p.kind.color();
                let fg = if p.lifetime > 8 { base } else {
                    match base {
                        Color::Rgb(r, g, b) => {
                            let f = p.lifetime as f32 / 8.0;
                            Color::Rgb((r as f32 * f) as u8, (g as f32 * f) as u8, (b as f32 * f) as u8)
                        }
                        c => c,
                    }
                };
                let cell = buf.get_mut(px, py);
                cell.set_char(p.kind.ch()).set_fg(fg).set_bg(WAVE_BG);
            }
        }

        // 4) Title and peer count overlay
        let title_color = Color::Rgb(80, 160, 200);
        for (i, ch) in "[ Swarm Ocean ]".chars().enumerate() {
            let tx = left + i as u16;
            if tx < right && top < bottom {
                let cell = buf.get_mut(tx, top);
                cell.set_char(ch).set_fg(title_color).set_bg(WAVE_BG);
            }
        }
        let peer_str = format!("Peers: {}", self.creatures.len().saturating_sub(1));
        let ps = right.saturating_sub(peer_str.len() as u16 + 1);
        for (i, ch) in peer_str.chars().enumerate() {
            let tx = ps + i as u16;
            if tx >= left && tx < right && top < bottom {
                let cell = buf.get_mut(tx, top);
                cell.set_char(ch).set_fg(title_color).set_bg(WAVE_BG);
            }
        }
    }
}


impl Default for SwarmOcean {
    fn default() -> Self { Self::new() }
}
