use crate::app::App;
use crate::metrics::Metrics;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

pub fn render(f: &mut Frame, app: &App) {
    let metrics = app.metrics.read().unwrap();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(9),  // Row 1: Hamiltonian | Phase Transition | Temperature
            Constraint::Length(9),  // Row 2: Gossip Diffusion | Convergence | Thermodynamics
            Constraint::Length(9),  // Row 3: Security | Privacy | Network Params
            Constraint::Min(3),    // Equation / status
            Constraint::Length(3), // Footer
        ])
        .split(f.size());

    render_header(f, chunks[0], &metrics);
    render_row1(f, chunks[1], &metrics);
    render_row2(f, chunks[2], &metrics);
    render_row3(f, chunks[3], &metrics);
    render_equations(f, chunks[4], &metrics);
    render_footer(f, chunks[5]);
}

fn render_header(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let phase_color = if metrics.physics_phase == "ordered" { Color::Green } else { Color::Red };
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " THEORETICAL PHYSICS DASHBOARD ",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::styled(
            format!("H_DAG = {:.0}", metrics.physics_h_total),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | Phase: "),
        Span::styled(
            metrics.physics_phase.to_uppercase(),
            Style::default().fg(phase_color).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | [F] Physics [S] Stats [D] Dash"),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Magenta)),
    )
    .alignment(Alignment::Center);

    f.render_widget(header, area);
}

/// Row 1: Consensus Hamiltonian | Phase Transition | Effective Temperature
fn render_row1(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(35),
            Constraint::Percentage(25),
        ])
        .split(area);

    // ─── Consensus Hamiltonian ───
    let h_total = metrics.physics_h_total;
    let h_color = if h_total < 0.0 { Color::Green } else { Color::Red };

    let hamiltonian_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" H_DAG  = ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", h_total),
                Style::default().fg(h_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" H_parent   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.2}", metrics.physics_h_parent),
                Style::default().fg(Color::White),
            ),
            Span::styled("  (causal)", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" H_anticone ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", metrics.physics_h_anticone),
                Style::default().fg(Color::Yellow),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" H_blue     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", metrics.physics_h_blue),
                Style::default().fg(Color::Blue),
            ),
            Span::styled("  (vertices)", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" H_vdf      ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", metrics.physics_h_vdf),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  (anchors)", Style::default().fg(Color::DarkGray)),
        ])),
    ];

    f.render_widget(
        List::new(hamiltonian_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Consensus Hamiltonian ",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(Color::Cyan)),
        ),
        cols[0],
    );

    // ─── Phase Transition ───
    let phase_color = if metrics.physics_phase == "ordered" { Color::Green } else { Color::Red };
    let margin = metrics.physics_phase_margin;
    let margin_color = if margin > 10.0 { Color::Green } else if margin > 1.0 { Color::Yellow } else { Color::Red };

    let phase_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Phase   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                metrics.physics_phase.to_uppercase(),
                Style::default().fg(phase_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" kappa   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}", metrics.physics_kappa),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" kappa_c ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", metrics.physics_kappa_c),
                Style::default().fg(Color::Yellow),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Margin  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", margin),
                Style::default().fg(margin_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" phi     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", metrics.physics_order_param),
                Style::default().fg(Color::Green),
            ),
            Span::styled("  (order)", Style::default().fg(Color::DarkGray)),
        ])),
    ];

    f.render_widget(
        List::new(phase_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Phase Transition ",
                    Style::default().fg(phase_color).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(phase_color)),
        ),
        cols[1],
    );

    // ─── Effective Temperature ───
    let t_eff = metrics.physics_t_eff;
    let temp_color = if t_eff < 0.1 { Color::Cyan } else if t_eff < 1.0 { Color::Green } else if t_eff < 5.0 { Color::Yellow } else { Color::Red };

    // Temperature "gauge" bar
    let temp_level = ((t_eff * 8.0).round() as usize).min(16);
    let temp_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" T_eff", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                format!(" {:.6}", t_eff),
                Style::default().fg(temp_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "\u{2588}".repeat(temp_level),
                Style::default().fg(temp_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(temp_level)),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                if t_eff < 0.01 { " GROUND STATE" }
                else if t_eff < 1.0 { " LOW TEMP" }
                else { " HIGH TEMP" },
                Style::default().fg(temp_color),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                " Strong consensus",
                Style::default().fg(Color::DarkGray),
            ),
        ])),
    ];

    f.render_widget(
        List::new(temp_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Temperature ",
                    Style::default().fg(temp_color).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(temp_color)),
        ),
        cols[2],
    );
}

/// Row 2: Gossip Diffusion | Convergence | Thermodynamics
fn render_row2(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(area);

    // ─── Gossip Diffusion ───
    let d = metrics.physics_diffusion_d;
    let tau = metrics.physics_tau_gossip_ms;
    let rho_200 = metrics.physics_info_density_200ms;
    let rho_1s = metrics.physics_info_density_1s;

    // Diffusion health bar
    let rho_bar = ((rho_1s * 16.0).round() as usize).min(16);
    let rho_color = if rho_1s > 0.99 { Color::Green } else if rho_1s > 0.9 { Color::Cyan } else { Color::Yellow };

    let diffusion_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" D       ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}", d),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  diffusion const", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" tau     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}ms", tau),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("  gossip time", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" mesh    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", metrics.physics_mesh_degree),
                Style::default().fg(Color::White),
            ),
            Span::styled("  degree", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" rho@.2s ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", rho_200),
                Style::default().fg(Color::Green),
            ),
            Span::styled("  @1s ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", rho_1s),
                Style::default().fg(rho_color),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "\u{2588}".repeat(rho_bar),
                Style::default().fg(rho_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(rho_bar)),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(" coverage", Style::default().fg(Color::DarkGray)),
        ])),
    ];

    f.render_widget(
        List::new(diffusion_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Gossip Diffusion ",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(Color::Cyan)),
        ),
        cols[0],
    );

    // ─── Convergence ───
    let gap = metrics.physics_spectral_gap;
    let conv_t = metrics.physics_convergence_time_s;
    let gap_color = if gap > 10.0 { Color::Green } else if gap > 1.0 { Color::Yellow } else { Color::Red };
    let conv_color = if conv_t < 1.0 { Color::Green } else if conv_t < 10.0 { Color::Yellow } else { Color::Red };

    let gap_bar = ((gap.min(100.0) / 100.0 * 16.0).round() as usize).min(16);

    let conv_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Spectral Gap", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                format!(" {:.4}", gap),
                Style::default().fg(gap_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "\u{2588}".repeat(gap_bar),
                Style::default().fg(gap_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(gap_bar)),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Conv time", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                format!(" {:.2}s", conv_t.min(999.0)),
                Style::default().fg(conv_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if conv_t < 1.0 { "  FAST" } else if conv_t < 10.0 { "  OK" } else { "  SLOW" },
                Style::default().fg(conv_color),
            ),
        ])),
    ];

    f.render_widget(
        List::new(conv_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Convergence ",
                    Style::default().fg(gap_color).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(gap_color)),
        ),
        cols[1],
    );

    // ─── Thermodynamics ───
    let fe = metrics.physics_free_energy;
    let entropy = metrics.physics_entropy;
    let energy = metrics.physics_h_total;
    let fe_color = if fe < 0.0 { Color::Green } else { Color::Yellow };

    let thermo_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" F = E - TS", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Free E  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", fe),
                Style::default().fg(fe_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Entropy  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", entropy),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(" bits", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Energy   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", energy),
                Style::default().fg(Color::Cyan),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                if fe < energy { " FAVORABLE" } else { " UNSTABLE" },
                Style::default().fg(fe_color).add_modifier(Modifier::BOLD),
            ),
        ])),
    ];

    f.render_widget(
        List::new(thermo_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Thermodynamics ",
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(Color::Yellow)),
        ),
        cols[2],
    );
}

/// Row 3: Security Bounds | Privacy | Network Parameters
fn render_row3(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35),
            Constraint::Percentage(30),
            Constraint::Percentage(35),
        ])
        .split(area);

    // ─── Security Bounds ───
    let sig_bits = metrics.physics_sig_forgery_bits;
    let key_bits = metrics.physics_key_recovery_bits;
    let dag_bits = metrics.physics_dag_manipulation_bits;

    let sig_color = if sig_bits >= 256.0 { Color::Green } else if sig_bits >= 128.0 { Color::Yellow } else { Color::Red };
    let key_color = if key_bits >= 192.0 { Color::Green } else if key_bits >= 128.0 { Color::Yellow } else { Color::Red };
    let dag_str = if dag_bits == 0.0 || dag_bits.is_infinite() {
        "infinity".to_string()
    } else {
        format!("{:.1}", dag_bits)
    };

    let security_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Attack Cost Bounds", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Sig forge  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("2^{:.0}", sig_bits),
                Style::default().fg(sig_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  bits", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Key recov  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("2^{:.0}", key_bits),
                Style::default().fg(key_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  bits", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" DAG manip  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("2^{}", dag_str),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                " Lattice    ",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                "2^Theta(n)",
                Style::default().fg(Color::Green),
            ),
        ])),
    ];

    f.render_widget(
        List::new(security_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Security Bounds ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(Color::Red)),
        ),
        cols[0],
    );

    // ─── Privacy (Dandelion++) ───
    let stem = metrics.physics_stem_length;
    let p_de = metrics.physics_p_deanon;
    let priv_color = if p_de < 0.01 { Color::Green } else if p_de < 0.1 { Color::Yellow } else { Color::Red };

    let priv_bar = ((1.0 - p_de) * 16.0).round() as usize;

    let privacy_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Dandelion++ Privacy", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" Stem hops  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", stem),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" P(deanon)  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.6}", p_de),
                Style::default().fg(priv_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "\u{2588}".repeat(priv_bar.min(16)),
                Style::default().fg(priv_color),
            ),
            Span::styled(
                "\u{2591}".repeat(16_usize.saturating_sub(priv_bar.min(16))),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(
                if p_de < 0.01 { " ANONYMOUS" } else if p_de < 0.1 { " PRIVATE" } else { " EXPOSED" },
                Style::default().fg(priv_color).add_modifier(Modifier::BOLD),
            ),
        ])),
    ];

    f.render_widget(
        List::new(privacy_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Privacy ",
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(Color::Magenta)),
        ),
        cols[1],
    );

    // ─── Network Parameters ───
    let br = metrics.physics_block_rate;
    let bf = metrics.physics_byzantine_fraction;
    let br_color = if br > 0.5 { Color::Green } else if br > 0.0 { Color::Yellow } else { Color::DarkGray };

    let net_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" Block rate  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4} b/s", br),
                Style::default().fg(br_color).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" delta       ", Style::default().fg(Color::DarkGray)),
            Span::styled("0.2s", Style::default().fg(Color::White)),
            Span::styled("  propagation", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" f/n         ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", bf),
                Style::default().fg(if bf < 0.01 { Color::Green } else { Color::Red }),
            ),
            Span::styled("  byzantine", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" k           ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", metrics.physics_kappa),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  protocol K", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled(" heartbeat   ", Style::default().fg(Color::DarkGray)),
            Span::styled("50ms", Style::default().fg(Color::White)),
            Span::styled("  gossip", Style::default().fg(Color::DarkGray)),
        ])),
    ];

    f.render_widget(
        List::new(net_items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    " Network Params ",
                    Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().fg(Color::Blue)),
        ),
        cols[2],
    );
}

fn render_equations(f: &mut Frame, area: Rect, metrics: &Metrics) {
    let phase_sym = if metrics.physics_phase == "ordered" { ">" } else { "<" };

    let eq_widget = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("  H_DAG", Style::default().fg(Color::Cyan)),
            Span::raw(" = H_p + H_a + H_b + H_vdf   "),
            Span::styled("kappa", Style::default().fg(Color::Green)),
            Span::raw(format!(" = {:.0} {} {:.4} = ", metrics.physics_kappa, phase_sym, metrics.physics_kappa_c)),
            Span::styled("kappa_c", Style::default().fg(Color::Yellow)),
            Span::raw("   "),
            Span::styled("F", Style::default().fg(Color::Yellow)),
            Span::raw(format!(" = {:.0} - {:.6} * {:.4}", metrics.physics_h_total, metrics.physics_t_eff, metrics.physics_entropy)),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Equations ")
            .style(Style::default().fg(Color::DarkGray)),
    );

    f.render_widget(eq_widget, area);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("[D] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Dashboard | "),
        Span::styled("[S] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Stats | "),
        Span::styled("[F] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Physics | "),
        Span::styled("[L] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Logs | "),
        Span::styled("[N] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Network | "),
        Span::styled("[Q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(footer, area);
}
