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

        // Apply privacy redaction to log messages before storing in ring buffer
        let message = q_log_privacy::PrivacyRedactionLayer::redact(&message);

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

/// Check if stdout is a real terminal (not a pipe, not headless)
pub fn is_terminal_available() -> bool {
    use std::io::IsTerminal;
    io::stdout().is_terminal() && io::stdin().is_terminal()
}

/// Initialize the terminal for TUI mode
pub fn init_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    if !is_terminal_available() {
        return Err(anyhow::anyhow!(
            "TUI requires a real terminal (TTY). Ubuntu Server headless or piped output detected. \
             Run without --tui for headless/systemd operation."
        ));
    }
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

/// Restore the terminal to normal mode
pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    // v8.3.0: Don't panic if terminal is already gone (SSH disconnect)
    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let _ = terminal.show_cursor();
    Ok(())
}

/// Run the TUI application
pub async fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = init_terminal()?;
    let mut event_handler = EventHandler::new(250); // 250ms tick rate
    let mut draw_errors = 0u32;

    loop {
        // Draw UI — handle errors gracefully (terminal may vanish)
        if let Err(e) = terminal.draw(|f| ui::render(f, &mut app)) {
            draw_errors += 1;
            if draw_errors >= 3 {
                eprintln!("⚠️ [TUI] Terminal draw failed {} times: {}. Exiting TUI gracefully.", draw_errors, e);
                break;
            }
        } else {
            draw_errors = 0;
        }

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
            Event::TerminalLost => {
                eprintln!("⚠️ [TUI] Terminal lost (SSH disconnect?). Node continues running without TUI.");
                break;
            }
        }
    }

    restore_terminal(&mut terminal)?;
    Ok(())
}
