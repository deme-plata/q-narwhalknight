//! Radar/Sonar Display: P2P peer visualization for the Command Center network tab.
//!
//! Shows connected peers as blips on a rotating radar sweep. Each peer is placed
//! at an angle (determined by name hash) and distance (proportional to latency).
//! The sweep line rotates once every ~4 seconds (16 ticks at 250ms each).
//! Zoom levels: 1x (full range), 2x, 4x. ASCII-only for terminal compatibility.


use ratatui::{buffer::Buffer, layout::Rect, style::Color};


use std::f64::consts::PI;


const TWO_PI: f64 = 2.0 * PI;

const SWEEP_STEP: f64 = TWO_PI / 16.0;

const FADE_DECAY: f64 = 0.06;

const FADE_FLOOR: f64 = 0.2;

const ILLUMINATE_THRESHOLD: f64 = 0.45;

const ASPECT: f64 = 2.0;

const RING_COUNT: usize = 3;


struct PeerBlip {
    name: String,
    angle: f64,
    distance: f64,
    strength: f64,
    connected: bool,
    fade: f64,
}


pub struct RadarDisplay {
    pub tick: u32,
    peer_blips: Vec<PeerBlip>,
    sweep_angle: f64,
    zoom_level: u8,
    ping_radius: f64,  // expanding ping ring (0.0 = no ping, >0 = active)
}


impl RadarDisplay {
    pub fn new() -> Self {
        Self { tick: 0, peer_blips: Vec::new(), sweep_angle: 0.0, zoom_level: 1, ping_radius: 0.0 }
    }

    /// Advance sweep angle, decay fades, illuminate blips near sweep.
    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        self.sweep_angle = (self.sweep_angle + SWEEP_STEP) % TWO_PI;
        for blip in &mut self.peer_blips {
            blip.fade = (blip.fade - FADE_DECAY).max(FADE_FLOOR);
            let mut d = (self.sweep_angle - blip.angle).abs();
            if d > PI { d = TWO_PI - d; }
            if d < ILLUMINATE_THRESHOLD { blip.fade = 1.0; }
        }
        // Decay ping ring
        if self.ping_radius > 0.0 {
            self.ping_radius += 0.15;
            if self.ping_radius > 1.2 { self.ping_radius = 0.0; }
        }
    }

    /// Trigger a sonar ping animation (expanding ring from center).
    pub fn ping(&mut self) {
        self.ping_radius = 0.05;
    }

    /// Update blips from network state. Synthetic peers based on peer_count.
    pub fn update_peers(&mut self, peer_count: u32, connected: bool) {
        let canon: &[(&str, f64, f64)] = &[
            ("Beta", 12.0, 0.95), ("Epsilon", 23.0, 0.99),
            ("Gamma", 45.0, 0.85), ("Delta", 67.0, 0.78),
            ("Alpha", 89.0, 0.60), ("Relay-1", 110.0, 0.55),
            ("Relay-2", 135.0, 0.50), ("Relay-3", 160.0, 0.42),
        ];
        let count = (peer_count as usize).min(canon.len());
        let old: Vec<(String, f64)> = self.peer_blips.iter()
            .map(|b| (b.name.clone(), b.fade)).collect();
        self.peer_blips.clear();
        for &(name, lat, str_) in &canon[..count] {
            let fade = old.iter().find(|(n, _)| n == name)
                .map(|(_, f)| *f).unwrap_or(FADE_FLOOR);
            self.peer_blips.push(PeerBlip {
                name: name.to_string(),
                angle: name_to_angle(name),
                distance: (lat / 200.0).clamp(0.15, 1.0),
                strength: str_,
                connected: connected && str_ > 0.3,
                fade,
            });
        }
    }

    /// Cycle zoom level: 1 -> 2 -> 4 -> 1
    pub fn cycle_zoom(&mut self) {
        self.zoom_level = match self.zoom_level { 1 => 2, 2 => 4, _ => 1 };
    }

    /// Render the radar display into the given buffer area.
    pub fn draw(&self, buf: &mut Buffer, area: Rect) {
        if area.width < 10 || area.height < 6 { return; }
        self.draw_border(buf, area);
        let inner = Rect {
            x: area.x + 1, y: area.y + 1,
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(2),
        };
        if inner.width < 8 || inner.height < 4 { return; }
        let cx = inner.x + inner.width / 2;
        let cy = inner.y + inner.height / 2;
        let rad = (inner.height / 2).saturating_sub(1) as f64;
        self.draw_rings(buf, inner, cx, cy, rad);
        self.draw_crosshairs(buf, inner, cx, cy, rad);
        self.draw_sweep(buf, inner, cx, cy, rad);
        self.draw_ping(buf, inner, cx, cy, rad);
        self.draw_blips(buf, inner, cx, cy, rad);
        self.draw_center(buf, cx, cy);
        self.draw_zoom(buf, inner);
    }

    fn draw_border(&self, buf: &mut Buffer, area: Rect) {
        let (x0, y0) = (area.x, area.y);
        let (x1, y1) = (area.right().saturating_sub(1), area.bottom().saturating_sub(1));
        for x in x0..=x1 { put(buf, x, y0, '-', Color::DarkGray); put(buf, x, y1, '-', Color::DarkGray); }
        for y in y0..=y1 { put(buf, x0, y, '|', Color::DarkGray); put(buf, x1, y, '|', Color::DarkGray); }
        for &(x, y) in &[(x0,y0),(x1,y0),(x0,y1),(x1,y1)] { put(buf, x, y, '+', Color::DarkGray); }
        let title = " RADAR ";
        let ts = x0 + (area.width.saturating_sub(title.len() as u16)) / 2;
        for (i, c) in title.chars().enumerate() {
            let tx = ts + i as u16;
            if tx < x1 { put(buf, tx, y0, c, Color::White); }
        }
    }

    fn draw_rings(&self, buf: &mut Buffer, inner: Rect, cx: u16, cy: u16, rad: f64) {
        let labels = ["50ms", "100ms", "150ms"];
        for ring in 1..=RING_COUNT {
            let r = rad * ring as f64 / (RING_COUNT + 1) as f64;
            let steps = ((r * ASPECT * 4.0) as usize).max(24);
            for s in 0..steps {
                let th = TWO_PI * s as f64 / steps as f64;
                let (px, py) = polar_to_cell(cx, cy, r, th);
                if hit(inner, px, py) { put(buf, px as u16, py as u16, '.', Color::DarkGray); }
            }
            // Distance label on right side of ring
            if ring <= labels.len() {
                let lbl = labels[ring - 1];
                let lx = (cx as f64 + r * ASPECT + 1.0).round() as i32;
                for (i, c) in lbl.chars().enumerate() {
                    let x = lx + i as i32;
                    if hit(inner, x, cy as i32) {
                        put(buf, x as u16, cy, c, Color::Rgb(60, 60, 60));
                    }
                }
            }
        }
    }

    fn draw_crosshairs(&self, buf: &mut Buffer, inner: Rect, cx: u16, cy: u16, rad: f64) {
        let mr = rad as i32;
        for dy in -mr..=mr {
            let py = cy as i32 + dy;
            if py != cy as i32 && hit(inner, cx as i32, py) {
                put(buf, cx, py as u16, ':', Color::Rgb(30, 30, 30));
            }
        }
        let mrx = (rad * ASPECT) as i32;
        for dx in -mrx..=mrx {
            let px = cx as i32 + dx;
            if px != cx as i32 && hit(inner, px, cy as i32) {
                put(buf, px as u16, cy, '-', Color::Rgb(30, 30, 30));
            }
        }
    }

    fn draw_sweep(&self, buf: &mut Buffer, inner: Rect, cx: u16, cy: u16, rad: f64) {
        let steps = (rad * ASPECT * 1.5) as usize;
        let (ca, sa) = (self.sweep_angle.cos(), self.sweep_angle.sin());
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let (px, py) = (
                (cx as f64 + ca * t * rad * ASPECT).round() as i32,
                (cy as f64 + sa * t * rad).round() as i32,
            );
            if !hit(inner, px, py) { break; }
            let g = 80 + ((t * 175.0) as u8);
            put(buf, px as u16, py as u16, '/', Color::Rgb(0, g, 0));
        }
        // Faint trail behind sweep
        for trail in 1..=3u32 {
            let ta = (self.sweep_angle - trail as f64 * 0.12 + TWO_PI) % TWO_PI;
            let (tc, ts_) = (ta.cos(), ta.sin());
            let dim = 60u8.saturating_sub(trail as u8 * 15);
            for i in 1..=steps {
                let t = i as f64 / steps as f64;
                let (px, py) = (
                    (cx as f64 + tc * t * rad * ASPECT).round() as i32,
                    (cy as f64 + ts_ * t * rad).round() as i32,
                );
                if !hit(inner, px, py) { break; }
                put(buf, px as u16, py as u16, '.', Color::Rgb(0, dim, 0));
            }
        }
    }

    fn draw_ping(&self, buf: &mut Buffer, inner: Rect, cx: u16, cy: u16, rad: f64) {
        if self.ping_radius <= 0.0 { return; }
        let r = self.ping_radius * rad;
        let brightness = ((1.0 - self.ping_radius) * 200.0) as u8;
        let steps = ((r * ASPECT * 4.0) as usize).max(16);
        for s in 0..steps {
            let th = TWO_PI * s as f64 / steps as f64;
            let (px, py) = polar_to_cell(cx, cy, r, th);
            if hit(inner, px, py) {
                put(buf, px as u16, py as u16, 'o', Color::Rgb(0, brightness, brightness));
            }
        }
    }

    fn draw_blips(&self, buf: &mut Buffer, inner: Rect, cx: u16, cy: u16, rad: f64) {
        let zoom = self.zoom_level as f64;
        for blip in &self.peer_blips {
            let r = (blip.distance / zoom).min(1.0) * rad;
            let (px, py) = polar_to_cell(cx, cy, r, blip.angle);
            if !hit(inner, px, py) { continue; }
            let (ch, col) = blip_char(blip);
            put(buf, px as u16, py as u16, ch, col);
            if blip.fade > 0.5 {
                let end = blip.name.char_indices().nth(2).map(|(i, _)| i)
                    .unwrap_or(blip.name.len());
                let lbl = &blip.name[..end];
                let lx = if px + 2 + lbl.len() as i32 <= inner.right() as i32
                    { px + 2 } else { px - 1 - lbl.len() as i32 };
                let lc = if blip.connected { Color::Green } else { Color::Red };
                for (i, c) in lbl.chars().enumerate() {
                    let x = lx + i as i32;
                    if hit(inner, x, py) { put(buf, x as u16, py as u16, c, lc); }
                }
            }
        }
    }

    fn draw_center(&self, buf: &mut Buffer, cx: u16, cy: u16) {
        if cx >= 1 { put(buf, cx - 1, cy, 'Y', Color::Cyan); }
        put(buf, cx, cy, 'O', Color::Cyan);
        put(buf, cx + 1, cy, 'U', Color::Cyan);
    }

    fn draw_zoom(&self, buf: &mut Buffer, inner: Rect) {
        let lbl = match self.zoom_level { 2 => "2x", 4 => "4x", _ => "1x" };
        let sx = inner.right().saturating_sub(lbl.len() as u16 + 1);
        let sy = inner.bottom().saturating_sub(1);
        for (i, c) in lbl.chars().enumerate() {
            let x = sx + i as u16;
            if x < inner.right() && sy < inner.bottom() && sy >= inner.y {
                put(buf, x, sy, c, Color::Yellow);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════


fn name_to_angle(name: &str) -> f64 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for b in name.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
    (h % 6283) as f64 / 1000.0
}


fn blip_char(blip: &PeerBlip) -> (char, Color) {
    if !blip.connected { return ('x', Color::Red); }
    if blip.fade > 0.7 {
        let c = if blip.strength > 0.8 { Color::Green }
                else if blip.strength > 0.5 { Color::Yellow }
                else { Color::Rgb(255, 165, 0) };
        ('O', c)
    } else if blip.fade > 0.4 {
        let c = if blip.strength > 0.6 { Color::Rgb(0, 100, 0) }
                else { Color::Rgb(100, 100, 0) };
        ('o', c)
    } else {
        ('.', Color::Rgb(0, 60, 0))
    }
}


fn polar_to_cell(cx: u16, cy: u16, r: f64, angle: f64) -> (i32, i32) {
    ((cx as f64 + angle.cos() * r * ASPECT).round() as i32,
     (cy as f64 + angle.sin() * r).round() as i32)
}


fn put(buf: &mut Buffer, x: u16, y: u16, ch: char, fg: Color) {
    let cell = buf.get_mut(x, y);
    cell.set_char(ch).set_fg(fg);
}


fn hit(area: Rect, px: i32, py: i32) -> bool {
    px >= area.x as i32 && px < area.right() as i32
        && py >= area.y as i32 && py < area.bottom() as i32
}
