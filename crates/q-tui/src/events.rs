use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyEvent, MouseEvent};
use std::time::Duration;
use tokio::sync::mpsc;

/// Terminal events
#[derive(Debug, Clone)]
pub enum Event {
    /// Tick event (periodic update)
    Tick,
    /// Key press
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
    /// Terminal resize
    Resize(u16, u16),
    /// Terminal lost (SSH disconnect, no TTY, etc.)
    TerminalLost,
}

/// Event handler for terminal events
pub struct EventHandler {
    rx: mpsc::UnboundedReceiver<Event>,
    _tx: mpsc::UnboundedSender<Event>,
}

impl EventHandler {
    /// Create a new event handler with the given tick rate (in milliseconds)
    pub fn new(tick_rate: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let event_tx = tx.clone();

        // Spawn event listener thread
        tokio::spawn(async move {
            let tick_duration = Duration::from_millis(tick_rate);
            let mut consecutive_errors = 0u32;
            loop {
                // v8.3.0: Handle poll/read errors gracefully instead of unwrap().
                // On Ubuntu Server, SSH disconnect or missing TTY causes crossterm
                // to return errors — previous .unwrap() killed the entire node.
                match event::poll(tick_duration) {
                    Ok(true) => {
                        consecutive_errors = 0;
                        match event::read() {
                            Ok(CrosstermEvent::Key(key)) => {
                                event_tx.send(Event::Key(key)).ok();
                            }
                            Ok(CrosstermEvent::Mouse(mouse)) => {
                                event_tx.send(Event::Mouse(mouse)).ok();
                            }
                            Ok(CrosstermEvent::Resize(w, h)) => {
                                event_tx.send(Event::Resize(w, h)).ok();
                            }
                            Ok(_) => {}
                            Err(_) => {
                                consecutive_errors += 1;
                                if consecutive_errors >= 5 {
                                    eprintln!("⚠️ [TUI] Terminal read failed {} times — terminal lost", consecutive_errors);
                                    event_tx.send(Event::TerminalLost).ok();
                                    break;
                                }
                            }
                        }
                    }
                    Ok(false) => {
                        // Normal timeout — send tick
                        consecutive_errors = 0;
                        event_tx.send(Event::Tick).ok();
                    }
                    Err(_) => {
                        consecutive_errors += 1;
                        // Still send ticks so the UI keeps updating
                        event_tx.send(Event::Tick).ok();
                        if consecutive_errors >= 10 {
                            eprintln!("⚠️ [TUI] Terminal poll failed {} times — terminal lost (SSH disconnect?)", consecutive_errors);
                            event_tx.send(Event::TerminalLost).ok();
                            break;
                        }
                        // Brief sleep to avoid busy-loop on persistent errors
                        std::thread::sleep(Duration::from_millis(500));
                    }
                }
            }
        });

        Self { rx, _tx: tx }
    }

    /// Get the next event
    pub async fn next(&mut self) -> Result<Event> {
        self.rx
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("Event channel closed"))
    }
}
