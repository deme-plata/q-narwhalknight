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
            loop {
                // Poll for events with timeout
                if event::poll(tick_duration).unwrap() {
                    match event::read().unwrap() {
                        CrosstermEvent::Key(key) => {
                            event_tx.send(Event::Key(key)).ok();
                        }
                        CrosstermEvent::Mouse(mouse) => {
                            event_tx.send(Event::Mouse(mouse)).ok();
                        }
                        CrosstermEvent::Resize(w, h) => {
                            event_tx.send(Event::Resize(w, h)).ok();
                        }
                        _ => {}
                    }
                } else {
                    // Timeout - send tick event
                    event_tx.send(Event::Tick).ok();
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
