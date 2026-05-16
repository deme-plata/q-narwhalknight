//! Single-instance enforcement plus a tiny IPC channel for forwarding URLs and
//! CLI args from a second launch to the running instance.
//!
//! Flow:
//!   1. First launch acquires the single-instance lock and starts an IPC server
//!      that listens on a well-known local socket name.
//!   2. Second launch fails to acquire the lock. It connects to the server,
//!      sends its argv (typically a `quillon://...` URL passed by the OS
//!      protocol handler), then exits cleanly.
//!   3. The running instance receives the forwarded message, raises its window,
//!      and routes the URL into the UI (deep-link to send screen, etc.).
//!
//! Cross-platform IPC via the `interprocess` crate (Windows named pipe, Linux
//! abstract Unix socket, macOS Unix socket).

use std::io::{BufRead, BufReader, Write};
use std::time::Duration;

use interprocess::local_socket::{
    prelude::*, GenericNamespaced, ListenerOptions, Stream,
};
use single_instance::SingleInstance;

const LOCK_ID: &str = "quillon-wallet-single-instance-v1";
const SOCKET_NAME: &str = "quillon-wallet-ipc-v1.sock";

/// Held for the lifetime of the wallet process. Dropping it releases the lock.
pub struct InstanceLock(#[allow(dead_code)] SingleInstance);

/// Try to acquire the single-instance lock. Returns `None` if another wallet
/// is already running.
pub fn acquire() -> Option<InstanceLock> {
    match SingleInstance::new(LOCK_ID) {
        Ok(instance) if instance.is_single() => Some(InstanceLock(instance)),
        Ok(_) => None,
        Err(e) => {
            eprintln!("[single-instance] WARN: lock creation failed ({e}); proceeding without enforcement");
            None
        }
    }
}

/// Send a message (typically a `quillon://...` URL or the literal "focus") to
/// the running instance. Best-effort; logs and swallows errors.
pub fn forward(message: &str) {
    let name = match SOCKET_NAME.to_ns_name::<GenericNamespaced>() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[single-instance] WARN: bad socket name: {e}");
            return;
        }
    };
    let mut stream = match Stream::connect(name) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[single-instance] WARN: cannot reach running instance ({e})");
            return;
        }
    };
    if let Err(e) = writeln!(stream, "{}", message.trim()) {
        eprintln!("[single-instance] WARN: forward write failed: {e}");
        return;
    }
    let _ = stream.flush();
}

/// Start the IPC server thread on the running instance. Each message received
/// is passed to `on_message` (typically a closure that posts to the Slint UI
/// thread via `slint::invoke_from_event_loop`).
///
/// The server is fire-and-forget: errors are logged, the thread keeps running.
pub fn start_server<F>(on_message: F)
where
    F: Fn(String) + Send + 'static,
{
    std::thread::spawn(move || {
        let name = match SOCKET_NAME.to_ns_name::<GenericNamespaced>() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[single-instance] ERROR: bad socket name: {e}");
                return;
            }
        };
        let listener = match ListenerOptions::new().name(name).create_sync() {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[single-instance] ERROR: cannot bind IPC socket: {e}");
                return;
            }
        };
        for incoming in listener.incoming() {
            match incoming {
                Ok(stream) => {
                    let mut reader = BufReader::new(stream);
                    let mut line = String::new();
                    // Cap at a small read so a malicious client cannot exhaust memory.
                    if let Ok(n) = reader.read_line(&mut line) {
                        if n > 0 {
                            let msg = line.trim().to_string();
                            if !msg.is_empty() && msg.len() < 4096 {
                                on_message(msg);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[single-instance] WARN: accept failed: {e}");
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    });
}
