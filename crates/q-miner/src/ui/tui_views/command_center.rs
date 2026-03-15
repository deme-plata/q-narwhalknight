//! Command Center: Main network visualization view (Tab #3) for the miner TUI.
//!
//! Composes the radar display, swarm ocean animation, and swarm status panel
//! into a single cohesive layout. Supports panel focus modes:
//!   [R] Radar full-screen  [S] Swarm ocean full-screen
//!   [T] Topology view      [P] Overview (default, all panels)

#[cfg(feature = "tui")]
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

#[cfg(feature = "tui")]
use super::radar::RadarDisplay;
#[cfg(feature = "tui")]
use super::swarm_ocean::SwarmOcean;
#[cfg(feature = "tui")]
use super::super::tui_app::MinerTuiApp;
#[cfg(feature = "tui")]
use std::sync::atomic::Ordering;

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Overview,   // Default - shows all panels
    RadarFull,  // [R] - radar takes full screen
    OceanFull,  // [S] - ocean takes full screen
    Topology,   // [T] - topology graph view
}

#[cfg(feature = "tui")]
pub struct CommandCenterState {
    pub radar: RadarDisplay,
    pub ocean: SwarmOcean,
    pub show_topology: bool,
    pub focused_panel: Panel,
}

#[cfg(feature = "tui")]
impl CommandCenterState {
    pub fn new() -> Self {
        Self {
            radar: RadarDisplay::new(),
            ocean: SwarmOcean::new(),
            show_topology: false,
            focused_panel: Panel::Overview,
        }
    }

    /// Advance animations by one tick.
    pub fn tick(&mut self, peer_count: u32, connected: bool) {
        self.radar.tick();
        self.radar.update_peers(peer_count, connected);
        self.ocean.tick(peer_count);
    }

    /// Handle panel-switching keys. Returns true if the key was consumed.
    pub fn handle_key(&mut self, key: char) -> bool {
        let new = match key {
            'r' | 'R' if self.focused_panel != Panel::RadarFull => Panel::RadarFull,
            'r' | 'R' => Panel::Overview,
            's' | 'S' if self.focused_panel != Panel::OceanFull => Panel::OceanFull,
            's' | 'S' => Panel::Overview,
            't' | 'T' if self.focused_panel != Panel::Topology => Panel::Topology,
            't' | 'T' => Panel::Overview,
            'p' | 'P' => Panel::Overview,
            _ => return false,
        };
        self.focused_panel = new;
        true
    }

    pub fn spawn_block_event(&mut self) { self.ocean.spawn_block_event(); }
    pub fn spawn_solution_event(&mut self) { self.ocean.spawn_solution_event(); }
}

// ═══════════════════════════════════════════════════════════════════
// Known peer roster (canonical servers)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
struct PeerEntry {
    symbol: char,
    name: &'static str,
    bandwidth: &'static str,
    latency_ms: u32,
    reliability: u8,
}

#[cfg(feature = "tui")]
const KNOWN_PEERS: [PeerEntry; 5] = [
    PeerEntry { symbol: 'E', name: "Epsilon", bandwidth: "10G",  latency_ms: 23, reliability: 99 },
    PeerEntry { symbol: 'B', name: "Beta",    bandwidth: "100M", latency_ms: 12, reliability: 95 },
    PeerEntry { symbol: 'G', name: "Gamma",   bandwidth: "1G",   latency_ms: 45, reliability: 82 },
    PeerEntry { symbol: 'D', name: "Delta",   bandwidth: "1G",   latency_ms: 67, reliability: 71 },
    PeerEntry { symbol: 'A', name: "Alpha",   bandwidth: "1G",   latency_ms: 89, reliability: 60 },
];

// ═══════════════════════════════════════════════════════════════════
// Main draw entry point (called from mod.rs when tab == 3)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
pub fn draw_command_center(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    match app.command_center.focused_panel {
        Panel::RadarFull => app.command_center.radar.draw(f.buffer_mut(), area),
        Panel::OceanFull => app.command_center.ocean.draw(f.buffer_mut(), area),
        Panel::Topology  => draw_topology(f, area, app),
        Panel::Overview  => draw_overview(f, area, app),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Overview layout: radar + status (top), ocean (bottom), footer
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn draw_overview(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(15),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(main[0]);

    app.command_center.radar.draw(f.buffer_mut(), top[0]);
    draw_swarm_status(f, top[1], app);
    app.command_center.ocean.draw(f.buffer_mut(), main[1]);
    draw_footer(f, main[2]);
}

// ═══════════════════════════════════════════════════════════════════
// Swarm Status + Peer Roster panel
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn fmt_bytes(total: u64) -> String {
    let kb = total as f64 / 1024.0;
    if kb >= 1024.0 { format!("{:.1} MB", kb / 1024.0) } else { format!("{:.0} KB", kb) }
}

#[cfg(feature = "tui")]
fn draw_swarm_status(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let (peer_count, connected, challenges, solutions, bytes_down, bytes_up) =
        if let Some(ref st) = app.state {
            (
                st.p2p_peer_count.load(Ordering::Relaxed),
                st.p2p_connected.load(Ordering::Relaxed),
                st.p2p_challenges_received.load(Ordering::Relaxed),
                st.p2p_solutions_broadcast.load(Ordering::Relaxed),
                st.bytes_downloaded.load(Ordering::Relaxed),
                st.bytes_uploaded.load(Ordering::Relaxed),
            )
        } else {
            (0, false, 0, 0, 0, 0)
        };

    let total_peers = 5u32;
    let health = ((peer_count as f64 / total_peers.max(1) as f64) * 100.0).min(100.0) as u32;
    let hc = if health >= 80 { Color::Green } else if health >= 50 { Color::Yellow } else { Color::Red };

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(" Active Nodes: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}/{}", peer_count, total_peers),
                Style::default().fg(if connected { Color::Green } else { Color::Red })),
            Span::styled("    Mesh Health: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}%", health), Style::default().fg(hc)),
        ]),
        Line::from(vec![
            Span::styled(" Challenges: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", challenges), Style::default().fg(Color::Cyan)),
            Span::styled("   Solutions: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", solutions), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled(" Bandwidth: ", Style::default().fg(Color::Gray)),
            Span::styled(fmt_bytes(bytes_down), Style::default().fg(Color::Blue)),
            Span::styled("   ", Style::default()),
            Span::styled(fmt_bytes(bytes_up), Style::default().fg(Color::Magenta)),
        ]),
    ];

    // Separator + roster header
    let sep_w = area.width.saturating_sub(3) as usize;
    lines.push(Line::from(Span::styled(
        format!(" {}", "-".repeat(sep_w)), Style::default().fg(Color::DarkGray))));
    lines.push(Line::from(Span::styled(
        " PEER ROSTER", Style::default().fg(Color::White).add_modifier(Modifier::BOLD))));

    // Peer entries with reliability bar
    let visible = (peer_count as usize).min(KNOWN_PEERS.len());
    for p in &KNOWN_PEERS[..visible] {
        let filled = ((p.reliability as usize) * 10 / 100).min(10);
        let bar_f: String = (0..filled).map(|_| '#').collect();
        let bar_e: String = (0..(10 - filled)).map(|_| '-').collect();
        let bc = if p.reliability >= 90 { Color::Green }
                 else if p.reliability >= 70 { Color::Yellow }
                 else { Color::Red };
        lines.push(Line::from(vec![
            Span::styled(format!(" {} ", p.symbol),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:<8}", p.name), Style::default().fg(Color::White)),
            Span::styled(format!("{:>4} ", p.bandwidth), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:>3}ms ", p.latency_ms), Style::default().fg(Color::Gray)),
            Span::styled(bar_f, Style::default().fg(bc)),
            Span::styled(bar_e, Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {}%", p.reliability), Style::default().fg(bc)),
        ]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" SWARM STATUS ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

// ═══════════════════════════════════════════════════════════════════
// Topology graph (Panel::Topology)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn draw_topology(f: &mut Frame, area: Rect, app: &MinerTuiApp) {
    let pc = app.state.as_ref()
        .map(|s| s.p2p_peer_count.load(Ordering::Relaxed)).unwrap_or(0);
    let g = Style::default().fg(Color::DarkGray);
    let green = Style::default().fg(Color::Green);
    let yellow = Style::default().fg(Color::Yellow);
    let cyan = Style::default().fg(Color::Cyan);
    let dim = Style::default().fg(Color::Rgb(60, 80, 100));

    // Animated data flow dots on edges
    let tick = app.command_center.radar.tick; // borrow tick for animation
    let dot_pos = (tick % 8) as usize;
    let flow_chars = ['.', '.', '*', '.', '.', '.', '.', '.'];

    // Build edge strings with animated dot
    let mut edge1 = String::from("           ");
    for i in 0..4 { edge1.push(if i == dot_pos % 4 { '*' } else { '-' }); }
    let mut edge2 = String::from("           |    \\          |       ");
    let mut edge3 = String::from("           |     \\         |       ");

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", g),
            Span::styled("MESH TOPOLOGY", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("  —  Real-time P2P Network Graph", dim),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("          ", g),
            Span::styled("[Epsilon", green),
            Span::styled(" 10G", cyan),
            Span::styled("]", green),
            Span::styled(
                format!("{}",
                    (0..8).map(|i| if i == dot_pos % 8 { '*' } else { '-' }).collect::<String>()),
                Style::default().fg(Color::Rgb(0, 180, 80))),
            Span::styled("[Beta", green),
            Span::styled(" 100M", cyan),
            Span::styled("]", green),
        ]),
        Line::from(vec![
            Span::styled("           |  ", g),
            Span::styled(if dot_pos % 3 == 0 { "*" } else { "\\" }, Style::default().fg(Color::Rgb(0, 120, 60))),
            Span::styled("            |       ", g),
        ]),
        Line::from(vec![
            Span::styled("           |   ", g),
            Span::styled(if dot_pos % 3 == 1 { "*" } else { "\\" }, Style::default().fg(Color::Rgb(0, 120, 60))),
            Span::styled("           |       ", g),
        ]),
        Line::from(vec![
            Span::styled("          ", g),
            Span::styled("[Gamma", yellow),
            Span::styled("  1G", cyan),
            Span::styled("]", yellow),
            Span::styled(
                format!("{}",
                    (0..8).map(|i| if i == (dot_pos + 4) % 8 { '*' } else { '-' }).collect::<String>()),
                Style::default().fg(Color::Rgb(180, 180, 0))),
            Span::styled("[Delta", yellow),
            Span::styled("  1G", cyan),
            Span::styled("]", yellow),
        ]),
        Line::from(Span::styled("           |                       ", g)),
        Line::from(vec![
            Span::styled("          ", g),
            Span::styled("[Alpha", dim),
            Span::styled("  1G", dim),
            Span::styled("]", dim),
            Span::styled("  (canary)", Style::default().fg(Color::Rgb(80, 80, 80))),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Connected: ", g),
            Span::styled(format!("{}", pc), cyan),
            Span::styled("  |  ", g),
            Span::styled(if pc >= 3 { "MESH HEALTHY" } else if pc >= 1 { "PARTIAL MESH" } else { "DISCONNECTED" },
                if pc >= 3 { green } else if pc >= 1 { yellow } else { Style::default().fg(Color::Red) }),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Legend: ", g),
            Span::styled("*", Style::default().fg(Color::Rgb(0, 180, 80))),
            Span::styled(" = data flow   ", g),
            Span::styled("----", g),
            Span::styled(" = connection   ", g),
            Span::styled("[P]", cyan),
            Span::styled(" = back to overview", g),
        ]),
    ];
    let block = Block::default()
        .borders(Borders::ALL).border_style(g)
        .title(Span::styled(" TOPOLOGY ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

// ═══════════════════════════════════════════════════════════════════
// Footer key hints
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn draw_footer(f: &mut Frame, area: Rect) {
    let c = Style::default().fg(Color::Cyan);
    let d = Style::default().fg(Color::DarkGray);
    let line = Line::from(vec![
        Span::styled(" [R]", c), Span::styled("adar  ", d),
        Span::styled("[S]", c),  Span::styled("warm  ", d),
        Span::styled("[T]", c),  Span::styled("opology  ", d),
        Span::styled("[P]", c),  Span::styled("eer details  ", d),
        Span::styled("[Tab]", c), Span::styled(" Next tab", d),
    ]);
    f.render_widget(Paragraph::new(line), area);
}
