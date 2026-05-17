use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span, Text},
    widgets::{
        BarChart, Block, Borders, Chart, Dataset, Gauge, List, ListItem, ListState,
        Paragraph, Sparkline, Table, Row, Cell,
    },
    Frame, Terminal,
};
use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};
use tokio::time::{interval, sleep};
use tracing::{debug, info, warn};

use crate::robot::{RobotManager, SensorData};
use crate::swarm::SwarmController;
use crate::quantum::QuantumStateMonitor;

/// Terminal UI application for robot control
pub struct TerminalUI {
    robot_manager: RobotManager,
    swarm_controller: SwarmController,
    quantum_monitor: QuantumStateMonitor,
    app_state: AppState,
}

/// Application state for UI
#[derive(Debug)]
struct AppState {
    current_tab: TabIndex,
    robot_list_state: ListState,
    swarm_list_state: ListState,
    quantum_entities_state: ListState,
    sensor_history: VecDeque<SensorReading>,
    performance_history: VecDeque<PerformanceMetric>,
    last_update: Instant,
    selected_robot: Option<String>,
    selected_swarm: Option<String>,
    log_messages: VecDeque<LogMessage>,
    show_help: bool,
}

#[derive(Debug, Clone, Copy)]
enum TabIndex {
    Robots = 0,
    Swarms = 1,
    Quantum = 2,
    Sensors = 3,
    Environment = 4,
    Logs = 5,
}

#[derive(Debug, Clone)]
struct SensorReading {
    timestamp: Instant,
    robot_id: String,
    temperature: f64,
    pressure: f64,
    ph: f64,
    dissolved_oxygen: f64,
    quantum_coherence: f64,
}

#[derive(Debug, Clone)]
struct PerformanceMetric {
    timestamp: Instant,
    metric_type: String,
    value: f64,
    unit: String,
}

#[derive(Debug, Clone)]
struct LogMessage {
    timestamp: Instant,
    level: LogLevel,
    message: String,
    source: String,
}

#[derive(Debug, Clone, Copy)]
enum LogLevel {
    Info,
    Warn,
    Error,
    Debug,
}

impl LogLevel {
    fn color(self) -> Color {
        match self {
            Self::Info => Color::Green,
            Self::Warn => Color::Yellow,
            Self::Error => Color::Red,
            Self::Debug => Color::Cyan,
        }
    }
    
    fn prefix(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERR",
            Self::Debug => "DBG",
        }
    }
}

impl TerminalUI {
    pub async fn new(
        robot_manager: RobotManager,
        swarm_controller: SwarmController, 
        quantum_monitor: QuantumStateMonitor
    ) -> Result<Self> {
        let mut app_state = AppState {
            current_tab: TabIndex::Robots,
            robot_list_state: ListState::default(),
            swarm_list_state: ListState::default(),
            quantum_entities_state: ListState::default(),
            sensor_history: VecDeque::with_capacity(1000),
            performance_history: VecDeque::with_capacity(1000),
            last_update: Instant::now(),
            selected_robot: None,
            selected_swarm: None,
            log_messages: VecDeque::with_capacity(500),
            show_help: false,
        };
        
        // Initialize with some sample data
        app_state.log_messages.push_back(LogMessage {
            timestamp: Instant::now(),
            level: LogLevel::Info,
            message: "Terminal UI initialized".to_string(),
            source: "UI".to_string(),
        });
        
        Ok(Self {
            robot_manager,
            swarm_controller,
            quantum_monitor,
            app_state,
        })
    }
    
    pub async fn run(&mut self, fullscreen: bool) -> Result<()> {
        if fullscreen {
            self.run_fullscreen().await
        } else {
            self.run_windowed().await
        }
    }
    
    async fn run_fullscreen(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        
        info!("Starting full-screen Terminal UI");
        
        let result = self.run_app(&mut terminal).await;
        
        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;
        
        if let Err(err) = result {
            println!("Application error: {}", err);
        }
        
        Ok(())
    }
    
    async fn run_windowed(&mut self) -> Result<()> {
        info!("Starting windowed Terminal UI (basic mode)");
        
        // Simple text-based interface
        loop {
            self.display_basic_interface().await?;
            
            println!("\nPress Enter to refresh, 'q' to quit, 'h' for help:");
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            match input.trim() {
                "q" | "quit" => break,
                "h" | "help" => self.show_help(),
                _ => continue,
            }
        }
        
        Ok(())
    }
    
    async fn run_app<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        let mut update_interval = interval(Duration::from_millis(250));
        
        loop {
            // Update data
            tokio::select! {
                _ = update_interval.tick() => {
                    self.update_data().await?;
                }
                result = async {
                    if event::poll(Duration::from_millis(50)).unwrap_or(false) {
                        if let Ok(event) = event::read() {
                            match event {
                                Event::Key(key) => {
                                    match key.code {
                                        KeyCode::Char('q') => return Ok::<bool, anyhow::Error>(true),
                                        KeyCode::Char('h') | KeyCode::F(1) => {
                                            self.app_state.show_help = !self.app_state.show_help;
                                        }
                                        KeyCode::Tab => {
                                            self.next_tab();
                                        }
                                        KeyCode::BackTab => {
                                            self.previous_tab();
                                        }
                                        KeyCode::Up => {
                                            self.handle_up_key();
                                        }
                                        KeyCode::Down => {
                                            self.handle_down_key();
                                        }
                                        KeyCode::Enter => {
                                            self.handle_enter_key().await?;
                                        }
                                        KeyCode::Char('r') => {
                                            self.refresh_data().await?;
                                        }
                                        _ => {}
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok::<bool, anyhow::Error>(false)
                } => {
                    if result? {
                        break Ok(());
                    }
                }
            }
            
            // Draw UI
            terminal.draw(|f| self.draw_ui(f))?;
        }
    }
    
    async fn update_data(&mut self) -> Result<()> {
        // Update sensor data
        let robots = self.robot_manager.list_robots().await?;
        
        for robot_info in robots {
            if let Ok(status) = self.robot_manager.get_robot_status(&robot_info.id).await {
                let sensor_reading = SensorReading {
                    timestamp: Instant::now(),
                    robot_id: robot_info.id.clone(),
                    temperature: status.sensor_data.temperature,
                    pressure: status.sensor_data.pressure,
                    ph: status.sensor_data.ph,
                    dissolved_oxygen: status.sensor_data.dissolved_oxygen,
                    quantum_coherence: status.quantum_coherence,
                };
                
                self.app_state.sensor_history.push_back(sensor_reading);
                
                // Keep only recent data
                if self.app_state.sensor_history.len() > 1000 {
                    self.app_state.sensor_history.pop_front();
                }
            }
        }
        
        // Add performance metrics
        let performance_metric = PerformanceMetric {
            timestamp: Instant::now(),
            metric_type: "System Load".to_string(),
            value: rand::random::<f64>() * 100.0,
            unit: "%".to_string(),
        };
        
        self.app_state.performance_history.push_back(performance_metric);
        
        if self.app_state.performance_history.len() > 1000 {
            self.app_state.performance_history.pop_front();
        }
        
        self.app_state.last_update = Instant::now();
        Ok(())
    }
    
    async fn refresh_data(&mut self) -> Result<()> {
        self.add_log_message(LogLevel::Info, "Refreshing all data", "UI");
        self.update_data().await
    }
    
    fn draw_ui(&mut self, frame: &mut Frame) {
        if self.app_state.show_help {
            self.draw_help_screen(frame);
            return;
        }

        let size = frame.size();
        
        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(0),     // Main content
                Constraint::Length(3),  // Status bar
            ])
            .split(size);
        
        // Draw header
        self.draw_header(frame, chunks[0]);
        
        // Draw main content based on current tab
        match self.app_state.current_tab {
            TabIndex::Robots => self.draw_robots_tab(frame, chunks[1]),
            TabIndex::Swarms => self.draw_swarms_tab(frame, chunks[1]),
            TabIndex::Quantum => self.draw_quantum_tab(frame, chunks[1]),
            TabIndex::Sensors => self.draw_sensors_tab(frame, chunks[1]),
            TabIndex::Environment => self.draw_environment_tab(frame, chunks[1]),
            TabIndex::Logs => self.draw_logs_tab(frame, chunks[1]),
        }
        
        // Draw status bar
        self.draw_status_bar(frame, chunks[2]);
    }
    
    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let title = "🌊🤖 Quantum Water Robot Control System 🤖🌊";
        let tabs = vec![
            "🤖 Robots", "🐟 Swarms", "⚛️  Quantum", "📊 Sensors", "🌊 Environment", "📝 Logs"
        ];
        
        let header = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .style(Style::default().fg(Color::Cyan));
        
        frame.render_widget(header, area);
        
        // Draw tab indicators
        let tab_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Percentage(16); 6])
            .margin(1)
            .split(area);
        
        for (i, tab_name) in tabs.iter().enumerate() {
            let is_selected = i == self.app_state.current_tab as usize;
            let style = if is_selected {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            
            let tab = Paragraph::new(*tab_name)
                .style(style)
                .block(Block::default());
            
            frame.render_widget(tab, tab_area[i]);
        }
    }
    
    fn draw_robots_tab(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        
        // Robot list
        let robot_items: Vec<ListItem> = vec![
            ListItem::new("quantum_jelly_001 - Active"),
            ListItem::new("dolphin_alpha_002 - Idle"),
            ListItem::new("octopus_stealth_003 - Mission"),
            ListItem::new("whale_song_004 - Charging"),
            ListItem::new("seahorse_precision_005 - Active"),
        ];
        
        let robots_list = List::new(robot_items)
            .block(Block::default().title("Connected Robots").borders(Borders::ALL))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
        
        frame.render_stateful_widget(robots_list, chunks[0], &mut self.app_state.robot_list_state);
        
        // Robot details
        let details = vec![
            Line::from(vec![Span::styled("Robot ID: ", Style::default().fg(Color::Cyan)), 
                          Span::raw("quantum_jelly_001")]),
            Line::from(vec![Span::styled("Type: ", Style::default().fg(Color::Cyan)), 
                          Span::raw("Quantum Jellyfish")]),
            Line::from(vec![Span::styled("Position: ", Style::default().fg(Color::Cyan)), 
                          Span::raw("(45.2, -12.8, -15.5)")]),
            Line::from(vec![Span::styled("Battery: ", Style::default().fg(Color::Cyan)), 
                          Span::styled("87%", Style::default().fg(Color::Green))]),
            Line::from(vec![Span::styled("Quantum Coherence: ", Style::default().fg(Color::Cyan)), 
                          Span::raw("0.125 ms")]),
            Line::from(vec![Span::styled("Active Abilities: ", Style::default().fg(Color::Cyan)), 
                          Span::raw("Bioluminescence, Superposition")]),
        ];
        
        let details_paragraph = Paragraph::new(details)
            .block(Block::default().title("Robot Details").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(details_paragraph, chunks[1]);
    }
    
    fn draw_swarms_tab(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area);
        
        // Swarm list
        let swarm_items: Vec<ListItem> = vec![
            ListItem::new("exploration_alpha - 5 robots - Active"),
            ListItem::new("patrol_beta - 8 robots - Patrol"),
            ListItem::new("research_gamma - 12 robots - Research"),
            ListItem::new("rescue_delta - 6 robots - Standby"),
        ];
        
        let swarms_list = List::new(swarm_items)
            .block(Block::default().title("Active Swarms").borders(Borders::ALL))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
        
        frame.render_stateful_widget(swarms_list, chunks[0], &mut self.app_state.swarm_list_state);
        
        // Swarm formation visualization
        let formation_area = chunks[1];
        let formation_block = Block::default()
            .title("Swarm Formation")
            .borders(Borders::ALL);
        
        frame.render_widget(formation_block.clone(), formation_area);
        
        // ASCII art formation display
        let formation_text = vec![
            Line::from("      🤖 Leader"),
            Line::from("    🤖   🤖"),
            Line::from("  🤖       🤖"),
            Line::from("🤖           🤖"),
            Line::from("  🤖       🤖"),
            Line::from("    🤖   🤖"),
            Line::from(""),
            Line::from("Formation: School"),
            Line::from("Cohesion: 94%"),
            Line::from("Entanglement Fidelity: 87%"),
        ];
        
        let formation_content = Paragraph::new(formation_text)
            .block(formation_block)
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(formation_content, formation_area);
    }
    
    fn draw_quantum_tab(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        
        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[0]);
        
        // Quantum state visualization
        let quantum_states = vec![
            Line::from("🌈 Superposition States:"),
            Line::from(""),
            Line::from(vec![
                Span::raw("|0⟩ "),
                Span::styled("████████████████", Style::default().fg(Color::Blue)),
                Span::raw(" 65%")
            ]),
            Line::from(vec![
                Span::raw("|1⟩ "),
                Span::styled("██████████", Style::default().fg(Color::Red)),
                Span::raw(" 35%")
            ]),
            Line::from(""),
            Line::from("Coherence Time: 0.245 ms"),
            Line::from("Position Uncertainty: ±1.8m"),
        ];
        
        let quantum_block = Paragraph::new(quantum_states)
            .block(Block::default().title("Quantum States").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(quantum_block, top_chunks[0]);
        
        // Entanglement matrix
        let entanglement_data = vec![
            Row::new(vec!["Robot", "A", "B", "C", "D"]),
            Row::new(vec!["A", "1.00", "0.87", "0.23", "0.45"]),
            Row::new(vec!["B", "0.87", "1.00", "0.91", "0.12"]),
            Row::new(vec!["C", "0.23", "0.91", "1.00", "0.78"]),
            Row::new(vec!["D", "0.45", "0.12", "0.78", "1.00"]),
        ];
        
        let entanglement_table = Table::new(
                entanglement_data,
                &[
                    Constraint::Length(8),
                    Constraint::Length(6),
                    Constraint::Length(6),
                    Constraint::Length(6),
                    Constraint::Length(6),
                ]
            )
            .block(Block::default().title("Entanglement Matrix").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(entanglement_table, top_chunks[1]);
        
        // Quantum measurements
        let measurements = vec![
            Line::from("Recent Quantum Measurements:"),
            Line::from(""),
            Line::from("Position: 12.45 ± 2.1 m"),
            Line::from("Momentum: 0.87 ± 0.3 kg⋅m/s"),
            Line::from("Spin: ↑ (probability: 73%)"),
            Line::from("Phase: 0.42π radians"),
            Line::from("Energy: 2.1 × 10⁻²¹ J"),
        ];
        
        let measurements_block = Paragraph::new(measurements)
            .block(Block::default().title("Quantum Measurements").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(measurements_block, chunks[1]);
    }
    
    fn draw_sensors_tab(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);
        
        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[0]);
        
        // Temperature gauge
        let temperature_gauge = Gauge::default()
            .block(Block::default().title("Water Temperature").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Blue))
            .percent(72)
            .label("22.4°C");
        
        frame.render_widget(temperature_gauge, top_chunks[0]);
        
        // pH gauge  
        let ph_gauge = Gauge::default()
            .block(Block::default().title("pH Level").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Green))
            .percent(83)
            .label("8.1 pH");
        
        frame.render_widget(ph_gauge, top_chunks[1]);
        
        // Sensor data table
        let sensor_data = vec![
            Row::new(vec!["Sensor", "Value", "Unit", "Status"]),
            Row::new(vec!["Dissolved O₂", "7.2", "mg/L", "✓ Good"]),
            Row::new(vec!["Salinity", "35.1", "PSU", "✓ Normal"]),  
            Row::new(vec!["Turbidity", "2.8", "NTU", "✓ Clear"]),
            Row::new(vec!["Pressure", "1.23", "atm", "✓ Normal"]),
            Row::new(vec!["Quantum Field", "0.87", "arb", "✓ Stable"]),
        ];
        
        let sensor_table = Table::new(
                sensor_data,
                &[
                    Constraint::Length(12),
                    Constraint::Length(8),
                    Constraint::Length(6),
                    Constraint::Length(10),
                ]
            )
            .block(Block::default().title("Sensor Readings").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(sensor_table, chunks[1]);
    }
    
    fn draw_environment_tab(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);
        
        // Marine life detection
        let marine_life = vec![
            Line::from("🐠 Marine Life Detected:"),
            Line::from(""),
            Line::from("• Bluefin Tuna - 12 individuals"),
            Line::from("  Location: (34.2, -45.1, -12.0)"),
            Line::from("  Behavior: Feeding"),
            Line::from(""),
            Line::from("• Giant Octopus - 1 individual"),  
            Line::from("  Location: (28.7, -50.3, -25.8)"),
            Line::from("  Behavior: Hunting"),
            Line::from(""),
            Line::from("• Coral Colony - Healthy"),
            Line::from("  Coverage: 87% - Excellent condition"),
        ];
        
        let marine_block = Paragraph::new(marine_life)
            .block(Block::default().title("Marine Environment").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(marine_block, chunks[0]);
        
        // Water quality summary
        let water_quality = vec![
            Row::new(vec!["Parameter", "Current", "Optimal Range", "Status"]),
            Row::new(vec!["Temperature", "22.4°C", "20-25°C", "✓ Good"]),
            Row::new(vec!["pH", "8.1", "7.5-8.5", "✓ Excellent"]),
            Row::new(vec!["Dissolved O₂", "7.2 mg/L", ">6.0 mg/L", "✓ Good"]),
            Row::new(vec!["Salinity", "35.1 PSU", "34-36 PSU", "✓ Normal"]),
            Row::new(vec!["Turbidity", "2.8 NTU", "<4.0 NTU", "✓ Clear"]),
        ];
        
        let quality_table = Table::new(
                water_quality,
                &[
                    Constraint::Length(12),
                    Constraint::Length(10),
                    Constraint::Length(12),
                    Constraint::Length(10),
                ]
            )
            .block(Block::default().title("Water Quality Assessment").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(quality_table, chunks[1]);
    }
    
    fn draw_logs_tab(&mut self, frame: &mut Frame, area: Rect) {
        let log_items: Vec<ListItem> = self.app_state.log_messages
            .iter()
            .rev() // Show newest first
            .take(50) // Limit display
            .map(|log| {
                let time_str = format!("{:02}:{:02}:{:02}", 
                    log.timestamp.elapsed().as_secs() / 3600 % 24,
                    log.timestamp.elapsed().as_secs() / 60 % 60,
                    log.timestamp.elapsed().as_secs() % 60);
                
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("[{}] ", time_str),
                        Style::default().fg(Color::Gray)
                    ),
                    Span::styled(
                        format!("{} ", log.level.prefix()),
                        Style::default().fg(log.level.color()).add_modifier(Modifier::BOLD)
                    ),
                    Span::styled(
                        format!("[{}] ", log.source),
                        Style::default().fg(Color::Blue)
                    ),
                    Span::raw(&log.message),
                ]))
            })
            .collect();
        
        let logs_list = List::new(log_items)
            .block(Block::default().title("System Logs").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));
        
        frame.render_widget(logs_list, area);
    }
    
    fn draw_status_bar(&self, frame: &mut Frame, area: Rect) {
        let status_text = format!(
            " Connected Robots: {} | Active Swarms: {} | Last Update: {:.1}s ago | Press 'h' for help, 'q' to quit",
            5, // Would be dynamic
            3, // Would be dynamic
            self.app_state.last_update.elapsed().as_secs_f64()
        );
        
        let status_paragraph = Paragraph::new(status_text)
            .style(Style::default().fg(Color::Yellow))
            .block(Block::default().borders(Borders::ALL));
        
        frame.render_widget(status_paragraph, area);
    }
    
    fn draw_help_screen(&self, frame: &mut Frame) {
        let help_text = vec![
            Line::from("🌊🤖 Quantum Water Robot Control System - Help 🤖🌊"),
            Line::from(""),
            Line::from("KEYBOARD SHORTCUTS:"),
            Line::from(""),
            Line::from("  Tab / Shift+Tab  - Switch between tabs"),
            Line::from("  ↑ ↓             - Navigate lists"),
            Line::from("  Enter           - Select/Activate item"),
            Line::from("  r               - Refresh data"),
            Line::from("  h / F1          - Toggle this help screen"),
            Line::from("  q               - Quit application"),
            Line::from(""),
            Line::from("TABS:"),
            Line::from(""),
            Line::from("  🤖 Robots       - View and control individual robots"),
            Line::from("  🐟 Swarms       - Manage robot swarms and formations"),
            Line::from("  ⚛️  Quantum      - Monitor quantum states and entanglement"),
            Line::from("  📊 Sensors      - Real-time sensor data and analytics"),
            Line::from("  🌊 Environment  - Marine life and water quality"),
            Line::from("  📝 Logs         - System logs and messages"),
            Line::from(""),
            Line::from("FEATURES:"),
            Line::from(""),
            Line::from("  • Real-time robot monitoring and control"),
            Line::from("  • Quantum state visualization and measurement"),
            Line::from("  • Swarm intelligence coordination"),
            Line::from("  • Environmental monitoring and conservation"),
            Line::from("  • Post-quantum cryptographic security"),
            Line::from(""),
            Line::from("Press 'h' again to close this help screen"),
        ];
        
        let help_paragraph = Paragraph::new(help_text)
            .block(Block::default()
                .title("Help")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
            )
            .style(Style::default().fg(Color::White))
            .alignment(ratatui::layout::Alignment::Left);
        
        let area = centered_rect(80, 80, frame.size());
        frame.render_widget(help_paragraph, area);
    }
    
    fn next_tab(&mut self) {
        self.app_state.current_tab = match self.app_state.current_tab {
            TabIndex::Robots => TabIndex::Swarms,
            TabIndex::Swarms => TabIndex::Quantum,
            TabIndex::Quantum => TabIndex::Sensors,
            TabIndex::Sensors => TabIndex::Environment,
            TabIndex::Environment => TabIndex::Logs,
            TabIndex::Logs => TabIndex::Robots,
        };
    }
    
    fn previous_tab(&mut self) {
        self.app_state.current_tab = match self.app_state.current_tab {
            TabIndex::Robots => TabIndex::Logs,
            TabIndex::Swarms => TabIndex::Robots,
            TabIndex::Quantum => TabIndex::Swarms,
            TabIndex::Sensors => TabIndex::Quantum,
            TabIndex::Environment => TabIndex::Sensors,
            TabIndex::Logs => TabIndex::Environment,
        };
    }
    
    fn handle_up_key(&mut self) {
        match self.app_state.current_tab {
            TabIndex::Robots => {
                let i = self.app_state.robot_list_state.selected().map_or(0, |i| {
                    if i == 0 { 4 } else { i - 1 }
                });
                self.app_state.robot_list_state.select(Some(i));
            }
            TabIndex::Swarms => {
                let i = self.app_state.swarm_list_state.selected().map_or(0, |i| {
                    if i == 0 { 3 } else { i - 1 }
                });
                self.app_state.swarm_list_state.select(Some(i));
            }
            _ => {}
        }
    }
    
    fn handle_down_key(&mut self) {
        match self.app_state.current_tab {
            TabIndex::Robots => {
                let i = self.app_state.robot_list_state.selected().map_or(0, |i| {
                    if i >= 4 { 0 } else { i + 1 }
                });
                self.app_state.robot_list_state.select(Some(i));
            }
            TabIndex::Swarms => {
                let i = self.app_state.swarm_list_state.selected().map_or(0, |i| {
                    if i >= 3 { 0 } else { i + 1 }
                });
                self.app_state.swarm_list_state.select(Some(i));
            }
            _ => {}
        }
    }
    
    async fn handle_enter_key(&mut self) -> Result<()> {
        match self.app_state.current_tab {
            TabIndex::Robots => {
                if let Some(selected) = self.app_state.robot_list_state.selected() {
                    let robot_names = vec![
                        "quantum_jelly_001",
                        "dolphin_alpha_002", 
                        "octopus_stealth_003",
                        "whale_song_004",
                        "seahorse_precision_005"
                    ];
                    
                    if let Some(robot_name) = robot_names.get(selected) {
                        self.app_state.selected_robot = Some(robot_name.to_string());
                        self.add_log_message(LogLevel::Info, 
                            &format!("Selected robot: {}", robot_name), "UI");
                    }
                }
            }
            TabIndex::Swarms => {
                if let Some(selected) = self.app_state.swarm_list_state.selected() {
                    let swarm_names = vec![
                        "exploration_alpha",
                        "patrol_beta",
                        "research_gamma", 
                        "rescue_delta"
                    ];
                    
                    if let Some(swarm_name) = swarm_names.get(selected) {
                        self.app_state.selected_swarm = Some(swarm_name.to_string());
                        self.add_log_message(LogLevel::Info,
                            &format!("Selected swarm: {}", swarm_name), "UI");
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn add_log_message(&mut self, level: LogLevel, message: &str, source: &str) {
        let log_msg = LogMessage {
            timestamp: Instant::now(),
            level,
            message: message.to_string(),
            source: source.to_string(),
        };
        
        self.app_state.log_messages.push_back(log_msg);
        
        // Keep only recent messages
        if self.app_state.log_messages.len() > 500 {
            self.app_state.log_messages.pop_front();
        }
    }
    
    async fn display_basic_interface(&mut self) -> Result<()> {
        println!("\n🌊🤖 Quantum Water Robot Control System 🤖🌊");
        println!("═══════════════════════════════════════════════");
        
        // Display robot status
        println!("\n🤖 Connected Robots:");
        let robots = self.robot_manager.list_robots().await?;
        if robots.is_empty() {
            println!("  No robots currently connected");
        } else {
            for robot in robots {
                println!("  • {} ({}) - {} - Battery: {:.1}%", 
                    robot.id, robot.robot_type, robot.status, robot.battery_level);
            }
        }
        
        // Display basic stats
        println!("\n📊 System Status:");
        println!("  Active Robots: {}", 5);
        println!("  Active Swarms: {}", 3); 
        println!("  System Load: {:.1}%", rand::random::<f64>() * 100.0);
        println!("  Quantum Coherence: {:.3} ms", 0.125);
        
        Ok(())
    }
    
    fn show_help(&self) {
        println!("\n🆘 Help - Quantum Water Robot Control System");
        println!("═══════════════════════════════════════════════");
        println!("Commands:");
        println!("  Enter/Return - Refresh display");
        println!("  h/help      - Show this help");
        println!("  q/quit      - Exit application");
        println!("\nFor full interactive mode, restart with --fullscreen flag");
    }
}

// Helper function to create centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}