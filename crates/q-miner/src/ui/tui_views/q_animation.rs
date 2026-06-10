//! Q-Animation: Fullscreen animated Q logo transformation
//!
//! Periodically transforms the entire TUI into a massive glowing "Q" by
//! keeping characters inside the Q shape (recolored with a quantum gradient)
//! and dissolving everything outside into darkness.
//!
//! The Q is defined mathematically (circle ring + diagonal tail) and scales
//! to any terminal size. The effect runs ~10 seconds every 10 minutes or can
//! be triggered manually (e.g., on block found).

use ratatui::{
    buffer::Buffer,
    layout::Position,
    style::Color,
};

// ═══════════════════════════════════════════════════════════════════
// Animation State Machine
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No animation — normal rendering
    Idle,
    /// Characters outside Q dissolve, Q chars recolor
    Converge,
    /// Full Q visible, pulsing quantum glow
    Glow,
    /// Q dissolves back to normal view
    Dissolve,
}

pub struct QAnimation {
    phase: Phase,
    frame: u16,
    idle_ticks: u32,
    trigger_interval: u32,
    seed: u64,
}

impl QAnimation {
    pub fn new() -> Self {
        Self {
            phase: Phase::Idle,
            frame: 0,
            idle_ticks: 0,
            trigger_interval: 2400, // 2400 ticks * 250ms = 10 minutes
            seed: 0x517E_CA11_4E49_6874,
        }
    }

    /// Advance animation state. Call once per tick (250ms).
    pub fn tick(&mut self) {
        match self.phase {
            Phase::Idle => {
                self.idle_ticks += 1;
                if self.idle_ticks >= self.trigger_interval {
                    self.phase = Phase::Converge;
                    self.frame = 0;
                    self.idle_ticks = 0;
                }
            }
            Phase::Converge => {
                self.frame += 1;
                if self.frame >= 14 {
                    self.phase = Phase::Glow;
                    self.frame = 0;
                }
            }
            Phase::Glow => {
                self.frame += 1;
                if self.frame >= 16 {
                    self.phase = Phase::Dissolve;
                    self.frame = 0;
                }
            }
            Phase::Dissolve => {
                self.frame += 1;
                if self.frame >= 10 {
                    self.phase = Phase::Idle;
                    self.frame = 0;
                }
            }
        }
    }

    /// Trigger animation immediately (e.g., on block found)
    pub fn trigger(&mut self) {
        if self.phase == Phase::Idle {
            self.phase = Phase::Converge;
            self.frame = 0;
            self.idle_ticks = 0;
        }
    }

    /// Is the animation currently active?
    pub fn is_active(&self) -> bool {
        self.phase != Phase::Idle
    }

    /// Apply the animation overlay to the frame buffer.
    /// Call AFTER normal rendering completes. This modifies the buffer in-place,
    /// transforming existing characters into the Q shape.
    pub fn apply_overlay(&self, buf: &mut Buffer) {
        if !self.is_active() {
            return;
        }

        let area = buf.area;
        let w = area.width;
        let h = area.height;
        if w < 30 || h < 15 {
            return; // Terminal too small for the effect
        }

        let (progress, is_glow) = match self.phase {
            Phase::Converge => (self.frame as f32 / 14.0, false),
            Phase::Glow => (1.0, true),
            Phase::Dissolve => (1.0 - self.frame as f32 / 10.0, false),
            Phase::Idle => return,
        };

        // Glow pulse: sinusoidal intensity cycling
        let glow_t = if is_glow {
            let t = self.frame as f32 / 16.0;
            (t * std::f32::consts::PI * 3.0).sin() * 0.5 + 0.5
        } else {
            0.0
        };

        let wf = w as f32;
        let hf = h as f32;

        for y in 0..h {
            for x in 0..w {
                let xf = x as f32;
                let yf = y as f32;

                let q_info = q_shape_info(xf, yf, wf, hf);
                let pos = Position::new(area.x + x, area.y + y);

                if let Some(cell) = buf.cell_mut(pos) {
                    match q_info {
                        QRegion::Edge => {
                            // Q outline: replace with block characters, beautiful gradient
                            let edge_progress = (progress * 1.3).min(1.0);
                            if edge_progress > 0.3 {
                                let t = ((edge_progress - 0.3) / 0.7).min(1.0);
                                let color = edge_gradient(xf / wf, yf / hf, glow_t);
                                let fg = lerp_color(cell.fg, color, t);
                                cell.set_fg(fg);

                                // At high progress, replace with block chars for sharp outline
                                if t > 0.6 {
                                    cell.set_char(edge_char(xf, yf, wf, hf));
                                    if is_glow {
                                        let bg = dim_color(color, 0.15);
                                        cell.set_bg(bg);
                                    }
                                }
                            }
                        }
                        QRegion::Interior => {
                            // Inside the Q: keep original text, recolor with gradient
                            let color = interior_gradient(xf / wf, yf / hf, glow_t);
                            let fg = lerp_color(cell.fg, color, progress);
                            cell.set_fg(fg);

                            if is_glow {
                                // Subtle background glow
                                let bg = dim_color(color, 0.08 + glow_t * 0.07);
                                cell.set_bg(bg);
                            }

                            // Quantum sparkle: random bright flashes near the edge
                            if is_glow && self.sparkle(x, y) {
                                cell.set_fg(Color::White);
                                cell.set_char(sparkle_char(self.frame, x, y));
                            }
                        }
                        QRegion::Exterior => {
                            // Outside Q: dissolve to darkness
                            let dist = q_distance(xf, yf, wf, hf);
                            let fade_start = 0.15;
                            if progress > fade_start {
                                let fade = ((progress - fade_start) / (1.0 - fade_start)).min(1.0);

                                // Closer cells fade faster (wave effect from Q outward)
                                let dist_factor = (1.0 - (dist / (wf * 0.6)).min(1.0)).max(0.0);
                                let cell_fade = (fade * (1.0 + dist_factor)).min(1.0);

                                if self.should_blank(x, y, cell_fade) {
                                    cell.set_char(' ');
                                    cell.set_fg(Color::Reset);
                                    cell.set_bg(Color::Reset);
                                } else if cell_fade > 0.3 {
                                    // Dim before fully blanking
                                    let dim = 1.0 - cell_fade;
                                    cell.set_fg(dim_color(cell.fg, dim));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Draw branded subtitle during glow phase
        if is_glow {
            self.draw_subtitle(buf, w, h, glow_t);
        }
    }

    /// Deterministic sparkle test for quantum particle effect
    fn sparkle(&self, x: u16, y: u16) -> bool {
        let h = self.hash(x as u64, y as u64, self.frame as u64);
        // ~3% chance per cell per frame
        h % 33 == 0
    }

    /// Deterministic test for whether to blank an exterior cell
    fn should_blank(&self, x: u16, y: u16, probability: f32) -> bool {
        let h = self.hash(x as u64, y as u64, self.frame as u64);
        (h % 1000) as f32 / 1000.0 < probability
    }

    /// Draw "Q - N A R W H A L K N I G H T" centered below the Q
    fn draw_subtitle(&self, buf: &mut Buffer, w: u16, h: u16, pulse: f32) {
        let text = "Q - N A R W H A L K N I G H T";
        let text_y = (h as f32 * 0.90) as u16;
        if text_y >= h { return; }

        let text_x = w.saturating_sub(text.len() as u16) / 2;
        let area = buf.area;

        for (i, ch) in text.chars().enumerate() {
            let px = text_x + i as u16;
            if px >= w { break; }

            let pos = Position::new(area.x + px, area.y + text_y);
            if let Some(cell) = buf.cell_mut(pos) {
                cell.set_char(ch);
                // Rainbow cycling color for each character
                let hue = (i as f32 / text.len() as f32 + pulse * 0.4) % 1.0;
                cell.set_fg(hue_to_color(hue));
                cell.set_bg(Color::Reset);
            }
        }

        // Version subtitle
        let version = concat!("v", env!("CARGO_PKG_VERSION"));
        let ver_y = text_y + 1;
        if ver_y < h {
            let ver_x = w.saturating_sub(version.len() as u16) / 2;
            for (i, ch) in version.chars().enumerate() {
                let px = ver_x + i as u16;
                if px >= w { break; }
                let pos = Position::new(area.x + px, area.y + ver_y);
                if let Some(cell) = buf.cell_mut(pos) {
                    cell.set_char(ch);
                    cell.set_fg(Color::DarkGray);
                    cell.set_bg(Color::Reset);
                }
            }
        }
    }

    /// Fast deterministic hash for visual effects (not cryptographic)
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
// Q Shape Mathematics
// ═══════════════════════════════════════════════════════════════════

/// Region classification for a point relative to the Q shape

#[derive(Debug, Clone, Copy, PartialEq)]
enum QRegion {
    /// On the Q outline (ring stroke or tail stroke)
    Edge,
    /// Inside the O of the Q (enclosed space)
    Interior,
    /// Outside the Q entirely
    Exterior,
}

/// Classify a point relative to the Q shape.
/// (x, y) in terminal coordinates, (w, h) is terminal size.

fn q_shape_info(x: f32, y: f32, w: f32, h: f32) -> QRegion {
    // Terminal characters are ~2.2x taller than wide
    let aspect: f32 = 2.2;

    // Center the O part slightly above middle (tail goes below)
    let cx = w / 2.0;
    let cy = h * 0.42;

    // Radius: use ~80% of the smaller effective dimension
    let max_r = (w / 2.0).min(h * aspect / 2.0) * 0.78;
    let r_outer = max_r;
    let r_inner = max_r * 0.70; // Ring thickness
    let stroke = max_r * 0.12;  // Edge detection width

    // Distance from center (aspect-corrected)
    let dx = x - cx;
    let dy = (y - cy) * aspect;
    let dist = (dx * dx + dy * dy).sqrt();

    // ── Tail of the Q ──
    // Diagonal from lower-right of circle toward bottom-right
    let tail_angle: f32 = 0.75; // ~43 degrees (radians)
    let cos_a = tail_angle.cos();
    let sin_a = tail_angle.sin();
    let tail_sx = cx + r_inner * 0.4 * cos_a;
    let tail_sy = cy + r_inner * 0.4 * sin_a / aspect;
    let tail_ex = cx + r_outer * 1.55 * cos_a;
    let tail_ey = cy + r_outer * 1.55 * sin_a / aspect;

    let tail_width = max_r * 0.18;
    let in_tail = point_near_segment(x, y * aspect, tail_sx, tail_sy * aspect,
                                      tail_ex, tail_ey * aspect, tail_width);

    // ── Ring of the O ──
    let on_outer_edge = (dist - r_outer).abs() < stroke;
    let on_inner_edge = (dist - r_inner).abs() < stroke;
    let in_ring = dist >= r_inner && dist <= r_outer;
    let in_interior = dist < r_inner;

    if in_tail {
        QRegion::Edge
    } else if on_outer_edge || on_inner_edge {
        QRegion::Edge
    } else if in_ring {
        QRegion::Edge
    } else if in_interior {
        QRegion::Interior
    } else {
        QRegion::Exterior
    }
}

/// Distance from a point to the nearest part of the Q shape

fn q_distance(x: f32, y: f32, w: f32, h: f32) -> f32 {
    let aspect: f32 = 2.2;
    let cx = w / 2.0;
    let cy = h * 0.42;
    let max_r = (w / 2.0).min(h * aspect / 2.0) * 0.78;

    let dx = x - cx;
    let dy = (y - cy) * aspect;
    let dist = (dx * dx + dy * dy).sqrt();

    // Distance to the ring
    let dist_to_ring = if dist > max_r {
        dist - max_r
    } else if dist < max_r * 0.70 {
        max_r * 0.70 - dist
    } else {
        0.0
    };

    dist_to_ring / aspect // Convert back to terminal units
}

/// Check if point is within `width` of a line segment

fn point_near_segment(px: f32, py: f32, sx: f32, sy: f32, ex: f32, ey: f32, width: f32) -> bool {
    let seg_dx = ex - sx;
    let seg_dy = ey - sy;
    let len_sq = seg_dx * seg_dx + seg_dy * seg_dy;
    if len_sq < 0.001 { return false; }

    let t = ((px - sx) * seg_dx + (py - sy) * seg_dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = sx + t * seg_dx;
    let proj_y = sy + t * seg_dy;
    let d = ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt();
    t > 0.02 && d <= width
}

// ═══════════════════════════════════════════════════════════════════
// Color Functions
// ═══════════════════════════════════════════════════════════════════

/// Gradient for Q edge: cyan at top → magenta at bottom

fn edge_gradient(_nx: f32, ny: f32, pulse: f32) -> Color {
    let hue = (ny * 0.5 + 0.5 + pulse * 0.1) % 1.0;
    let (r, g, b) = hue_to_rgb(hue);
    // Bright edge
    Color::Rgb(
        (r as f32 * 0.9 + 25.0) as u8,
        (g as f32 * 0.9 + 25.0) as u8,
        (b as f32 * 0.9 + 25.0) as u8,
    )
}

/// Gradient for Q interior: softer version of edge

fn interior_gradient(nx: f32, ny: f32, pulse: f32) -> Color {
    let hue = (ny * 0.45 + 0.55 + pulse * 0.08 + nx * 0.05) % 1.0;
    let (r, g, b) = hue_to_rgb(hue);
    // Slightly dimmer than edge
    Color::Rgb(
        (r as f32 * 0.75) as u8,
        (g as f32 * 0.75) as u8,
        (b as f32 * 0.75) as u8,
    )
}

/// Choose the best block character for edge cells based on position

fn edge_char(x: f32, y: f32, w: f32, h: f32) -> char {
    let aspect: f32 = 2.2;
    let cx = w / 2.0;
    let cy = h * 0.42;

    let dx = x - cx;
    let dy = (y - cy) * aspect;
    let angle = dy.atan2(dx);

    // Use different block chars based on angle for a sculpted look
    let blocks = ['█', '▓', '▒', '░', '▓', '█', '▓', '▒'];
    let idx = ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 8.0) as usize % 8;
    blocks[idx]
}

/// Sparkle characters for quantum particle effect

fn sparkle_char(frame: u16, x: u16, y: u16) -> char {
    let chars = ['✦', '✧', '·', '⚛', '✦', '·', '★', '✧'];
    let idx = (frame as usize + x as usize * 7 + y as usize * 13) % chars.len();
    chars[idx]
}

/// Convert hue [0.0, 1.0] to Color

fn hue_to_color(hue: f32) -> Color {
    let (r, g, b) = hue_to_rgb(hue);
    Color::Rgb(r, g, b)
}

/// Convert hue [0.0, 1.0] to (R, G, B)

fn hue_to_rgb(hue: f32) -> (u8, u8, u8) {
    let h = (hue.fract() + 1.0).fract() * 6.0; // Ensure positive
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

/// Linearly interpolate between two colors

fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let (fr, fg, fb) = color_to_rgb(from);
    let (tr, tg, tb) = color_to_rgb(to);
    let t = t.clamp(0.0, 1.0);
    Color::Rgb(
        (fr as f32 + (tr as f32 - fr as f32) * t) as u8,
        (fg as f32 + (tg as f32 - fg as f32) * t) as u8,
        (fb as f32 + (tb as f32 - fb as f32) * t) as u8,
    )
}

/// Dim a color by a factor (0.0 = black, 1.0 = original)

fn dim_color(c: Color, factor: f32) -> Color {
    let (r, g, b) = color_to_rgb(c);
    let f = factor.clamp(0.0, 1.0);
    Color::Rgb(
        (r as f32 * f) as u8,
        (g as f32 * f) as u8,
        (b as f32 * f) as u8,
    )
}

/// Extract RGB from any ratatui Color

fn color_to_rgb(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::White => (220, 220, 220),
        Color::Cyan => (0, 255, 255),
        Color::LightCyan => (128, 255, 255),
        Color::Blue => (0, 0, 255),
        Color::LightBlue => (128, 128, 255),
        Color::Magenta => (255, 0, 255),
        Color::LightMagenta => (255, 128, 255),
        Color::Yellow => (255, 255, 0),
        Color::LightYellow => (255, 255, 128),
        Color::Green => (0, 200, 0),
        Color::LightGreen => (128, 255, 128),
        Color::Red => (255, 0, 0),
        Color::LightRed => (255, 128, 128),
        Color::Gray => (128, 128, 128),
        Color::DarkGray => (64, 64, 64),
        Color::Black => (0, 0, 0),
        Color::Reset => (180, 180, 180),
        _ => (180, 180, 180),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_shape_center_is_interior() {
        // Center of a 100x40 terminal should be inside the Q
        let info = q_shape_info(50.0, 16.0, 100.0, 40.0);
        assert_eq!(info, QRegion::Interior);
    }

    #[test]
    fn test_q_shape_corner_is_exterior() {
        let info = q_shape_info(0.0, 0.0, 100.0, 40.0);
        assert_eq!(info, QRegion::Exterior);
    }

    #[test]
    fn test_q_shape_ring_is_edge() {
        // A point on the outer ring (right side of O)
        let info = q_shape_info(85.0, 17.0, 100.0, 40.0);
        assert_eq!(info, QRegion::Edge);
    }

    #[test]
    fn test_animation_lifecycle() {
        let mut anim = QAnimation::new();
        assert!(!anim.is_active());

        // Trigger manually
        anim.trigger();
        assert!(anim.is_active());
        assert_eq!(anim.phase, Phase::Converge);

        // Advance through converge (14 frames)
        for _ in 0..14 {
            anim.tick();
        }
        assert_eq!(anim.phase, Phase::Glow);

        // Advance through glow (16 frames)
        for _ in 0..16 {
            anim.tick();
        }
        assert_eq!(anim.phase, Phase::Dissolve);

        // Advance through dissolve (10 frames)
        for _ in 0..10 {
            anim.tick();
        }
        assert_eq!(anim.phase, Phase::Idle);
        assert!(!anim.is_active());
    }

    #[test]
    fn test_auto_trigger_after_interval() {
        let mut anim = QAnimation::new();
        anim.trigger_interval = 10; // Short interval for test

        for _ in 0..10 {
            anim.tick();
        }
        assert!(anim.is_active());
    }

    #[test]
    fn test_hue_to_rgb_red() {
        let (r, g, b) = hue_to_rgb(0.0);
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_hue_to_rgb_green() {
        let (r, g, b) = hue_to_rgb(1.0 / 3.0);
        assert_eq!(r, 0);
        assert_eq!(g, 255);
    }

    #[test]
    fn test_hue_to_rgb_blue() {
        let (r, g, b) = hue_to_rgb(2.0 / 3.0);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_lerp_color() {
        let black = Color::Rgb(0, 0, 0);
        let white = Color::Rgb(255, 255, 255);
        let mid = lerp_color(black, white, 0.5);
        if let Color::Rgb(r, g, b) = mid {
            assert!((r as i32 - 127).abs() <= 1);
            assert!((g as i32 - 127).abs() <= 1);
        }
    }

    #[test]
    fn test_dim_color() {
        let white = Color::Rgb(200, 200, 200);
        let dimmed = dim_color(white, 0.5);
        if let Color::Rgb(r, g, b) = dimmed {
            assert_eq!(r, 100);
            assert_eq!(g, 100);
        }
    }

    #[test]
    fn test_q_distance_far_corner() {
        let dist = q_distance(0.0, 0.0, 100.0, 40.0);
        assert!(dist > 10.0); // Corner should be far from Q
    }

    #[test]
    fn test_sparkle_deterministic() {
        let anim = QAnimation::new();
        let a = anim.sparkle(10, 20);
        let b = anim.sparkle(10, 20);
        assert_eq!(a, b); // Same input → same result
    }
}
