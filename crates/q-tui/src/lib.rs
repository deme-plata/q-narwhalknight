/// Q-NarwhalKnight Beautiful Terminal UI
///
/// Provides an interactive, real-time dashboard for monitoring node status,
/// network metrics, blockchain state, and streaming logs.

pub mod app;
pub mod ui;
pub mod events;
pub mod metrics;

pub use app::{App, LogEntry, LogLevel};
pub use events::{Event, EventHandler};
pub use metrics::Metrics;

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use ringbuf::{HeapRb, Rb};
use std::io;
use std::sync::{Arc, RwLock};

/// A tracing layer that captures log events into a ring buffer for the TUI.
///
/// Instead of writing to stdout (which corrupts ratatui's alternate screen),
/// this layer converts tracing events into `LogEntry` values and pushes them
/// into a shared ring buffer that the TUI log panel reads from.
pub struct TuiLogLayer {
    log_buffer: Arc<RwLock<HeapRb<LogEntry>>>,
}

impl TuiLogLayer {
    pub fn new(log_buffer: Arc<RwLock<HeapRb<LogEntry>>>) -> Self {
        Self { log_buffer }
    }
}

impl<S> tracing_subscriber::Layer<S> for TuiLogLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let level = match *event.metadata().level() {
            tracing::Level::ERROR => LogLevel::Error,
            tracing::Level::WARN => LogLevel::Warn,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::DEBUG => LogLevel::Debug,
            tracing::Level::TRACE => LogLevel::Trace,
        };

        let target = event.metadata().target().to_string();

        // Extract the message from the event fields
        let mut message = String::new();
        let mut visitor = MessageVisitor(&mut message);
        event.record(&mut visitor);

        if let Ok(mut buf) = self.log_buffer.write() {
            buf.push_overwrite(LogEntry {
                timestamp: chrono::Utc::now(),
                level,
                target,
                message,
            });
        }
    }
}

/// Visitor that extracts the `message` field from a tracing event.
struct MessageVisitor<'a>(&'a mut String);

impl<'a> tracing::field::Visit for MessageVisitor<'a> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = format!("{:?}", value);
        } else if self.0.is_empty() {
            *self.0 = format!("{}={:?}", field.name(), value);
        } else {
            *self.0 = format!("{} {}={:?}", self.0, field.name(), value);
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            *self.0 = value.to_string();
        } else if self.0.is_empty() {
            *self.0 = format!("{}={}", field.name(), value);
        } else {
            *self.0 = format!("{} {}={}", self.0, field.name(), value);
        }
    }
}

/// Initialize the terminal for TUI mode
pub fn init_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

/// Restore the terminal to normal mode
pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Run the TUI application
pub async fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = init_terminal()?;
    let mut event_handler = EventHandler::new(250); // 250ms tick rate

    loop {
        // Draw UI
        terminal.draw(|f| ui::render(f, &mut app))?;

        // Handle events
        match event_handler.next().await? {
            Event::Tick => {
                app.on_tick();
            }
            Event::Key(key) => {
                if app.handle_key_event(key) {
                    break; // User requested quit
                }
            }
            Event::Mouse(_) => {}
            Event::Resize(_, _) => {}
        }
    }

    restore_terminal(&mut terminal)?;
    Ok(())
}
