//! Cross-thread notification for blocking consumers.
//!
//! When a consumer has no work, it can park on a [`Notifier`] instead of
//! busy-spinning. The producer calls `notify()` after enqueuing to wake
//! the consumer with minimal latency.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, Thread};

/// A lightweight notification primitive for producer->consumer wakeup.
pub struct Notifier {
    consumer_thread: Thread,
    parked: AtomicBool,
}

impl Default for Notifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Notifier {
    /// Create a notifier for the current thread (call from consumer thread).
    pub fn new() -> Self {
        Self {
            consumer_thread: thread::current(),
            parked: AtomicBool::new(false),
        }
    }

    /// Create a notifier targeting a specific thread.
    pub fn for_thread(thread: Thread) -> Self {
        Self {
            consumer_thread: thread,
            parked: AtomicBool::new(false),
        }
    }

    /// Wake the consumer if parked. Called by producer after enqueuing.
    #[inline]
    pub fn notify(&self) {
        if self.parked.swap(false, Ordering::AcqRel) {
            self.consumer_thread.unpark();
        }
    }

    /// Park until notified. Called by consumer when queue is empty.
    #[inline]
    pub fn wait(&self) {
        self.parked.store(true, Ordering::Release);
        thread::park();
        self.parked.store(false, Ordering::Relaxed);
    }

    /// Park with timeout. Returns `true` if woken by notify.
    #[inline]
    pub fn wait_timeout(&self, duration: std::time::Duration) -> bool {
        self.parked.store(true, Ordering::Release);
        thread::park_timeout(duration);
        // If parked is still true, we timed out (nobody called notify)
        !self.parked.swap(false, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn notify_wakes_parked_thread() {
        // Use for_thread so we can notify from the spawned thread
        let main_thread = thread::current();
        let notifier = Arc::new(Notifier::for_thread(main_thread));

        let n = notifier.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            n.notify();
        });

        notifier.wait_timeout(Duration::from_secs(2));
        handle.join().unwrap();
    }

    #[test]
    fn wait_timeout_returns_false_on_timeout() {
        let notifier = Notifier::new();
        let woken = notifier.wait_timeout(Duration::from_millis(10));
        assert!(!woken);
    }
}
