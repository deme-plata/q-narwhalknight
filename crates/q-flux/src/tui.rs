use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
};
use ratatui::{
    prelude::*,
    widgets::*,
};

use crate::metrics::Metrics;

/// Sparkline history: 120 data points × 500ms tick = 60 seconds of history
const SPARKLINE_LEN: usize = 120;
const TICK: Duration = Duration::from_millis(500);

pub struct TuiApp {
    metrics: Metrics,
    rate_history: VecDeque<u64>,
    conn_history: VecDeque<u64>,
    prev_requests: u64,
    prev_bytes_rx: u64,
    prev_bytes_tx: u64,
    prev_time: Instant,
    current_rate: f64,
    peak_rate: f64,
    peak_connections: u64,
    current_rx_rate: f64,
    current_tx_rate: f64,
    should_quit: bool,
    worker_count: usize,
}

impl TuiApp {
    pub fn new(metrics: Metrics, worker_count: usize) -> Self {
        Self {
            metrics,
            rate_history: VecDeque::with_capacity(SPARKLINE_LEN),
            conn_history: VecDeque::with_capacity(SPARKLINE_LEN),
            prev_requests: 0,
            prev_bytes_rx: 0,
            prev_bytes_tx: 0,
            prev_time: Instant::now(),
            current_rate: 0.0,
            peak_rate: 0.0,
            peak_connections: 0,
            current_rx_rate: 0.0,
            current_tx_rate: 0.0,
            should_quit: false,
            worker_count,
        }
    }

    pub fn run(mut self) -> io::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.run_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        result
    }

    fn run_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        let mut last_tick = Instant::now();
        loop {
            terminal.draw(|f| self.draw(f))?;

            let timeout = TICK.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                            KeyCode::Char('r') => self.reset_peaks(),
                            _ => {}
                        }
                    }
                }
            }

            if last_tick.elapsed() >= TICK {
                self.on_tick();
                last_tick = Instant::now();
            }

            if self.should_quit {
                return Ok(());
            }
        }
    }

    fn on_tick(&mut self) {
        let snap = self.metrics.snapshot();
        let now = Instant::now();
        let elapsed = now.duration_since(self.prev_time).as_secs_f64();

        if elapsed > 0.01 {
            let delta_reqs = snap.total_requests.saturating_sub(self.prev_requests);
            self.current_rate = delta_reqs as f64 / elapsed;
            if self.current_rate > self.peak_rate {
                self.peak_rate = self.current_rate;
            }

            let delta_rx = snap.bytes_received.saturating_sub(self.prev_bytes_rx);
            let delta_tx = snap.bytes_sent.saturating_sub(self.prev_bytes_tx);
            self.current_rx_rate = delta_rx as f64 / elapsed;
            self.current_tx_rate = delta_tx as f64 / elapsed;

            self.prev_requests = snap.total_requests;
            self.prev_bytes_rx = snap.bytes_received;
            self.prev_bytes_tx = snap.bytes_sent;
            self.prev_time = now;
        }

        if snap.active_connections > self.peak_connections {
            self.peak_connections = snap.active_connections;
        }

        push_bounded(&mut self.rate_history, self.current_rate as u64, SPARKLINE_LEN);
        push_bounded(&mut self.conn_history, snap.active_connections, SPARKLINE_LEN);
    }

    fn reset_peaks(&mut self) {
        self.rate_history.clear();
        self.conn_history.clear();
        self.peak_rate = 0.0;
        self.peak_connections = 0;
    }

    // ── Drawing ──────────────────────────────────────────────────────

    fn draw(&self, f: &mut Frame) {
        let snap = self.metrics.snapshot();
        let area = f.area();

        // Main layout: header | stats | sparklines | footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header bar
                Constraint::Length(12),  // Stats panels
                Constraint::Min(8),     // Sparkline charts
                Constraint::Length(1),   // Footer / keybindings
            ])
            .split(area);

        self.draw_header(f, chunks[0], &snap);
        self.draw_stats(f, chunks[1], &snap);
        self.draw_charts(f, chunks[2]);
        self.draw_footer(f, chunks[3]);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect, snap: &crate::metrics::MetricsSnapshot) {
        let uptime = fmt_duration(snap.uptime_secs);

        // Build a single-line status bar with key indicators
        let status_color = if snap.upstream_connect_failures == 0 && snap.requests_5xx == 0 {
            Color::Green
        } else if snap.requests_5xx < 100 {
            Color::Yellow
        } else {
            Color::Red
        };

        let status_dot = Span::styled("●", Style::default().fg(status_color));
        let title_line = Line::from(vec![
            Span::styled(" ⚡ q-flux ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("v{}", env!("CARGO_PKG_VERSION")),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            status_dot,
            Span::styled(
                format!(" {} workers", self.worker_count),
                Style::default().fg(Color::White),
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("⏱ {}", uptime),
                Style::default().fg(Color::White),
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/s ", fmt_num(self.current_rate as u64)),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} conns", fmt_num(snap.active_connections)),
                Style::default().fg(Color::Cyan),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Double)
            .border_style(Style::default().fg(Color::Cyan));

        let paragraph = Paragraph::new(title_line).block(block).alignment(Alignment::Left);
        f.render_widget(paragraph, area);
    }

    fn draw_stats(&self, f: &mut Frame, area: Rect, snap: &crate::metrics::MetricsSnapshot) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area);

        self.draw_connections_panel(f, cols[0], snap);
        self.draw_requests_panel(f, cols[1], snap);
        self.draw_bandwidth_panel(f, cols[2], snap);
        self.draw_upstream_panel(f, cols[3], snap);
    }

    fn draw_connections_panel(&self, f: &mut Frame, area: Rect, snap: &crate::metrics::MetricsSnapshot) {
        let block = styled_block(" CONNECTIONS ", Color::Blue);

        let lines = vec![
            stat_line("●", Color::Green, "Active ", &fmt_num(snap.active_connections), Color::White, true),
            stat_line("○", Color::DarkGray, "Total  ", &fmt_num(snap.total_connections), Color::Gray, false),
            stat_line("◆", Color::Yellow, "Peak   ", &fmt_num(self.peak_connections), Color::Yellow, false),
            Line::from(""),
            Line::from(Span::styled("  TLS", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))),
            stat_line("✓", Color::Green, "OK     ", &fmt_num(snap.tls_handshakes), Color::Green, false),
            stat_line("✗", Color::Red, "Fail   ", &fmt_num(snap.tls_handshake_failures), Color::Red, false),
            stat_line("⊘", Color::DarkGray, "Limit  ", &fmt_num(snap.rate_limited), Color::DarkGray, false),
        ];

        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_requests_panel(&self, f: &mut Frame, area: Rect, snap: &crate::metrics::MetricsSnapshot) {
        let rate_color = rate_to_color(self.current_rate);
        let block = styled_block(" REQUESTS ", Color::Green);

        let total_resp = snap.requests_2xx + snap.requests_4xx + snap.requests_5xx;
        let success_pct = if total_resp > 0 {
            format!("{}%", snap.requests_2xx * 100 / total_resp)
        } else {
            "—".into()
        };

        let lines = vec![
            stat_line("⚡", rate_color, "Rate   ", &format!("{}/s", fmt_num(self.current_rate as u64)), rate_color, true),
            stat_line("↑", Color::Yellow, "Peak   ", &format!("{}/s", fmt_num(self.peak_rate as u64)), Color::Yellow, false),
            stat_line("Σ", Color::DarkGray, "Total  ", &fmt_num(snap.total_requests), Color::Gray, false),
            Line::from(""),
            Line::from(vec![
                Span::styled("  2xx ", Style::default().fg(Color::Green)),
                Span::styled(fmt_num(snap.requests_2xx), Style::default().fg(Color::Green)),
                Span::styled(format!(" ({})", success_pct), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("  4xx ", Style::default().fg(Color::Yellow)),
                Span::styled(fmt_num(snap.requests_4xx), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("  5xx ", Style::default().fg(Color::Red)),
                Span::styled(fmt_num(snap.requests_5xx), Style::default().fg(if snap.requests_5xx > 0 { Color::Red } else { Color::DarkGray })),
            ]),
        ];

        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_bandwidth_panel(&self, f: &mut Frame, area: Rect, snap: &crate::metrics::MetricsSnapshot) {
        let block = styled_block(" BANDWIDTH ", Color::Cyan);

        let lines = vec![
            stat_line("↓", Color::Green, "RX     ", &fmt_bytes(snap.bytes_received), Color::Green, true),
            stat_line("↑", Color::Cyan, "TX     ", &fmt_bytes(snap.bytes_sent), Color::Cyan, true),
            Line::from(""),
            stat_line("↓", Color::Green, "RX/s   ", &format!("{}/s", fmt_bytes(self.current_rx_rate as u64)), Color::Green, false),
            stat_line("↑", Color::Cyan, "TX/s   ", &format!("{}/s", fmt_bytes(self.current_tx_rate as u64)), Color::Cyan, false),
            Line::from(""),
            Line::from(Span::styled("  WEBSOCKET", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))),
            stat_line("●", Color::Green, "Active ", &fmt_num(snap.active_websockets), Color::White, false),
            stat_line("Σ", Color::DarkGray, "Total  ", &fmt_num(snap.websocket_upgrades), Color::Gray, false),
        ];

        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_upstream_panel(&self, f: &mut Frame, area: Rect, snap: &crate::metrics::MetricsSnapshot) {
        let health = if snap.upstream_connect_failures == 0 && snap.upstream_timeouts == 0 {
            Color::Green
        } else if snap.upstream_connect_failures < 10 {
            Color::Yellow
        } else {
            Color::Red
        };

        let block = styled_block(" UPSTREAM ", health);

        let lines = vec![
            stat_line("●", Color::Green, "Active   ", &fmt_num(snap.upstream_active), Color::White, true),
            Line::from(""),
            stat_line("✗", Color::Red, "ConnFail ", &fmt_num(snap.upstream_connect_failures),
                if snap.upstream_connect_failures > 0 { Color::Red } else { Color::DarkGray }, false),
            stat_line("⏱", Color::Yellow, "Timeout  ", &fmt_num(snap.upstream_timeouts),
                if snap.upstream_timeouts > 0 { Color::Yellow } else { Color::DarkGray }, false),
            Line::from(""),
            // Health gauge
            {
                let error_total = snap.upstream_connect_failures + snap.upstream_timeouts;
                let total = snap.total_requests.max(1);
                let health_pct = if total > 0 {
                    ((total.saturating_sub(error_total)) as f64 / total as f64 * 100.0) as u64
                } else {
                    100
                };
                let bar_width = 12;
                let filled = (health_pct as usize * bar_width / 100).min(bar_width);
                let empty = bar_width - filled;
                let bar_color = if health_pct >= 99 { Color::Green } else if health_pct >= 95 { Color::Yellow } else { Color::Red };
                Line::from(vec![
                    Span::styled("  ", Style::default()),
                    Span::styled("█".repeat(filled), Style::default().fg(bar_color)),
                    Span::styled("░".repeat(empty), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!(" {}%", health_pct), Style::default().fg(bar_color).add_modifier(Modifier::BOLD)),
                ])
            },
            Line::from(Span::styled("  health", Style::default().fg(Color::DarkGray))),
        ];

        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_charts(&self, f: &mut Frame, area: Rect) {
        // Split into two sparkline charts: request rate + active connections
        let charts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(area);

        self.draw_rate_chart(f, charts[0]);
        self.draw_conn_chart(f, charts[1]);
    }

    fn draw_rate_chart(&self, f: &mut Frame, area: Rect) {
        let data_vec: Vec<u64> = if self.rate_history.is_empty() {
            vec![0]
        } else {
            self.rate_history.iter().copied().collect()
        };
        let data: &[u64] = &data_vec;

        let min_r = data.iter().copied().min().unwrap_or(0);
        let max_r = data.iter().copied().max().unwrap_or(0);
        let avg_r: u64 = if data.is_empty() { 0 } else { data.iter().sum::<u64>() / data.len() as u64 };

        let footer_line = Line::from(vec![
            Span::styled("  min ", Style::default().fg(Color::DarkGray)),
            Span::styled(fmt_num(min_r), Style::default().fg(Color::Blue)),
            Span::styled("  avg ", Style::default().fg(Color::DarkGray)),
            Span::styled(fmt_num(avg_r), Style::default().fg(Color::Yellow)),
            Span::styled("  max ", Style::default().fg(Color::DarkGray)),
            Span::styled(fmt_num(max_r), Style::default().fg(Color::Red)),
            Span::styled("  now ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/s", fmt_num(self.current_rate as u64)),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " REQUEST RATE (/sec) ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ))
            .title_bottom(footer_line);

        let sparkline = Sparkline::default()
            .block(block)
            .data(data)
            .style(Style::default().fg(Color::Cyan));

        f.render_widget(sparkline, area);
    }

    fn draw_conn_chart(&self, f: &mut Frame, area: Rect) {
        let data_vec: Vec<u64> = if self.conn_history.is_empty() {
            vec![0]
        } else {
            self.conn_history.iter().copied().collect()
        };
        let data: &[u64] = &data_vec;

        let snap = self.metrics.snapshot();
        let min_c = data.iter().copied().min().unwrap_or(0);
        let max_c = data.iter().copied().max().unwrap_or(0);

        let footer_line = Line::from(vec![
            Span::styled("  min ", Style::default().fg(Color::DarkGray)),
            Span::styled(fmt_num(min_c), Style::default().fg(Color::Blue)),
            Span::styled("  max ", Style::default().fg(Color::DarkGray)),
            Span::styled(fmt_num(max_c), Style::default().fg(Color::Red)),
            Span::styled("  now ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                fmt_num(snap.active_connections),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  peak ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                fmt_num(self.peak_connections),
                Style::default().fg(Color::Yellow),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " ACTIVE CONNECTIONS ",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ))
            .title_bottom(footer_line);

        let sparkline = Sparkline::default()
            .block(block)
            .data(data)
            .style(Style::default().fg(Color::Magenta));

        f.render_widget(sparkline, area);
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let footer = Line::from(vec![
            Span::styled(" q", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(" quit  ", Style::default().fg(Color::DarkGray)),
            Span::styled("r", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(" reset peaks  ", Style::default().fg(Color::DarkGray)),
            Span::styled("⚡ q-flux", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" — worker-per-core TLS reverse proxy", Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(footer), area);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn styled_block(title: &str, color: Color) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .title(Span::styled(
            title,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
}

fn stat_line(icon: &str, icon_color: Color, label: &str, value: &str, value_color: Color, bold: bool) -> Line<'static> {
    let mut style = Style::default().fg(value_color);
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    Line::from(vec![
        Span::styled(format!(" {} ", icon), Style::default().fg(icon_color)),
        Span::raw(label.to_string()),
        Span::styled(value.to_string(), style),
    ])
}

fn rate_to_color(rate: f64) -> Color {
    if rate > 5000.0 { Color::Green }
    else if rate > 1000.0 { Color::Yellow }
    else if rate > 0.0 { Color::White }
    else { Color::DarkGray }
}

fn push_bounded(buf: &mut VecDeque<u64>, val: u64, max: usize) {
    buf.push_back(val);
    if buf.len() > max {
        buf.pop_front();
    }
}

pub fn fmt_num(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1000, n % 1000)
    } else {
        format!("{}", n)
    }
}

pub fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_099_511_627_776 {
        format!("{:.1} TB", bytes as f64 / 1_099_511_627_776.0)
    } else if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn fmt_duration(secs: u64) -> String {
    let d = secs / 86400;
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if d > 0 {
        format!("{}d {}h {}m", d, h, m)
    } else if h > 0 {
        format!("{}h {}m {}s", h, m, s)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}
