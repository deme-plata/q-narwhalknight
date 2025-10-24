use crate::app::{App, BountyInputField};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Length(7),  // Testnet address input
            Constraint::Length(7),  // Mainnet address input (optional)
            Constraint::Length(5),  // Instructions
            Constraint::Min(0),     // Status message
            Constraint::Length(3),  // Controls
        ])
        .split(f.size());

    // Title
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "⚛️ Q-NarwhalKnight Testnet Bounty Registration",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL).style(Style::default().fg(Color::Cyan)));

    f.render_widget(title, chunks[0]);

    // Testnet Address Input
    let testnet_focused = matches!(app.bounty_input_field, BountyInputField::TestnetAddress);
    let testnet_style = if testnet_focused {
        Style::default().fg(Color::Yellow).bg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let testnet_block = Block::default()
        .borders(Borders::ALL)
        .title(" Testnet Wallet Address (Required) ")
        .title_style(if testnet_focused {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        })
        .border_style(if testnet_focused {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let testnet_input = Paragraph::new(Line::from(vec![Span::styled(
        if app.bounty_testnet_address.is_empty() {
            "qnk...".to_string()
        } else {
            app.bounty_testnet_address.clone()
        },
        testnet_style,
    )]))
    .block(testnet_block)
    .wrap(Wrap { trim: false });

    f.render_widget(testnet_input, chunks[1]);

    // Mainnet Address Input (Optional)
    let mainnet_focused = matches!(app.bounty_input_field, BountyInputField::MainnetAddress);
    let mainnet_style = if mainnet_focused {
        Style::default().fg(Color::Yellow).bg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let mainnet_block = Block::default()
        .borders(Borders::ALL)
        .title(" Mainnet Wallet Address (Optional - for future rewards) ")
        .title_style(if mainnet_focused {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        })
        .border_style(if mainnet_focused {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    let mainnet_input = Paragraph::new(Line::from(vec![Span::styled(
        if app.bounty_mainnet_address.is_empty() {
            "qnk... (optional)".to_string()
        } else {
            app.bounty_mainnet_address.clone()
        },
        mainnet_style,
    )]))
    .block(mainnet_block)
    .wrap(Wrap { trim: false });

    f.render_widget(mainnet_input, chunks[2]);

    // Instructions
    let instructions = Paragraph::new(vec![
        Line::from(vec![
            Span::raw("Register your node for the "),
            Span::styled("Q-NarwhalKnight Testnet Bounty Campaign", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Earn points for: "),
            Span::styled("Node Operations • Transactions • Bug Reports • Community • Social Media", Style::default().fg(Color::Green)),
        ]),
    ])
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)));

    f.render_widget(instructions, chunks[3]);

    // Status Message
    if !app.bounty_status_message.is_empty() {
        let status_color = if app.bounty_status_message.starts_with('✅') {
            Color::Green
        } else if app.bounty_status_message.starts_with('❌') {
            Color::Red
        } else {
            Color::Yellow
        };

        let status = Paragraph::new(Line::from(vec![Span::styled(
            app.bounty_status_message.clone(),
            Style::default().fg(status_color).add_modifier(Modifier::BOLD),
        )]))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(status_color)));

        f.render_widget(status, chunks[4]);
    } else {
        // Show campaign info
        let info = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("💰 Total Reward Pool: ", Style::default().fg(Color::Yellow)),
                Span::styled("100,000 QNK", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("🏆 Distribution: ", Style::default().fg(Color::Yellow)),
                Span::raw("Pioneer 20% • Contributor 30% • Participant 40% • Supporter 10%"),
            ]),
        ])
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Magenta)));

        f.render_widget(info, chunks[4]);
    }

    // Controls
    let controls = Paragraph::new(Line::from(vec![
        Span::styled("↑/↓", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw(" Switch fields  "),
        Span::styled("Enter", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::raw(" Submit  "),
        Span::styled("Esc", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::raw(" Back  "),
        Span::styled("B", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" Bounty page"),
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));

    f.render_widget(controls, chunks[5]);
}
