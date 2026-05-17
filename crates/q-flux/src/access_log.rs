//! Structured JSON access logging (Issue #13).
//!
//! Each request produces one log line with timestamp, client IP, method,
//! path, status, latency, bytes, and TLS info. Uses a dedicated writer
//! thread with a bounded channel to keep I/O off the hot path.

use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::net::SocketAddr;
use std::sync::mpsc;
use std::time::Duration;

/// A single access log entry.
pub struct AccessEntry {
    pub timestamp: String,
    pub client_addr: SocketAddr,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub request_bytes: u64,
    pub response_bytes: u64,
    pub latency: Duration,
    pub tls_version: Option<String>,
    pub user_agent: Option<String>,
    pub upstream_backend: Option<String>,
    pub request_id: Option<String>,
}

impl AccessEntry {
    /// Serialize to a single JSON line (no trailing newline).
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("{\"ts\":\"");
        out.push_str(&self.timestamp);
        out.push_str("\",\"ip\":\"");
        write!(out, "{}", self.client_addr.ip()).ok();
        out.push_str("\",\"method\":\"");
        out.push_str(&self.method);
        out.push_str("\",\"path\":\"");
        // Escape quotes and control characters in path
        for c in self.path.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c.is_control() => { /* skip control chars */ }
                c => out.push(c),
            }
        }
        out.push_str("\",\"status\":");
        write!(out, "{}", self.status).ok();
        out.push_str(",\"rx\":");
        write!(out, "{}", self.request_bytes).ok();
        out.push_str(",\"tx\":");
        write!(out, "{}", self.response_bytes).ok();
        out.push_str(",\"latency_ms\":");
        write!(out, "{:.2}", self.latency.as_secs_f64() * 1000.0).ok();
        if let Some(ref tls) = self.tls_version {
            out.push_str(",\"tls\":\"");
            out.push_str(tls);
            out.push('"');
        }
        if let Some(ref ua) = self.user_agent {
            out.push_str(",\"ua\":\"");
            for c in ua.chars().take(128) {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c.is_control() => { /* skip control chars */ }
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        if let Some(ref backend) = self.upstream_backend {
            out.push_str(",\"upstream\":\"");
            out.push_str(backend);
            out.push('"');
        }
        if let Some(ref rid) = self.request_id {
            out.push_str(",\"rid\":\"");
            out.push_str(rid);
            out.push('"');
        }
        out.push('}');
        out
    }
}

/// Access log writer that runs on a dedicated thread.
/// Entries are sent via a bounded channel — producers never block on I/O.
pub struct AccessLogger {
    tx: mpsc::SyncSender<AccessEntry>,
}

impl AccessLogger {
    /// Create a new access logger writing to stdout.
    /// `buffer_size`: max queued entries before producers drop logs.
    pub fn new_stdout(buffer_size: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel::<AccessEntry>(buffer_size);

        std::thread::Builder::new()
            .name("q-flux-access-log".into())
            .spawn(move || {
                let stdout = std::io::stdout();
                let mut out = std::io::BufWriter::new(stdout.lock());
                while let Ok(entry) = rx.recv() {
                    let line = entry.to_json();
                    let _ = writeln!(out, "{}", line);
                    // Flush every 64 entries for batched I/O
                    // (BufWriter handles the actual batching)
                }
            })
            .expect("failed to spawn access log thread");

        Self { tx }
    }

    /// Create a new access logger writing to a file.
    pub fn new_file(path: &str, buffer_size: usize) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let (tx, rx) = mpsc::sync_channel::<AccessEntry>(buffer_size);

        std::thread::Builder::new()
            .name("q-flux-access-log".into())
            .spawn(move || {
                let mut out = std::io::BufWriter::new(file);
                while let Ok(entry) = rx.recv() {
                    let line = entry.to_json();
                    let _ = writeln!(out, "{}", line);
                }
            })
            .expect("failed to spawn access log thread");

        Ok(Self { tx })
    }

    /// Log an access entry. Non-blocking — drops the entry if the channel is full.
    #[inline]
    pub fn log(&self, entry: AccessEntry) {
        let _ = self.tx.try_send(entry); // Drop if buffer full — never block hot path
    }
}

impl Clone for AccessLogger {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

/// Epoch-millisecond timestamp without external dependencies.
fn epoch_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", now.as_secs(), now.subsec_millis())
}

/// Emit an access log entry if a logger is configured.
///
/// Shared by both the HTTP/1.1 and HTTP/2 proxy paths.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn log_access(
    logger: Option<&AccessLogger>,
    client_addr: SocketAddr,
    method: &str,
    path: &str,
    status: u16,
    request_bytes: u64,
    response_bytes: u64,
    latency: Duration,
    user_agent: Option<&str>,
    upstream_backend: Option<&str>,
    request_id: Option<&str>,
) {
    if let Some(logger) = logger {
        logger.log(AccessEntry {
            timestamp: epoch_timestamp(),
            client_addr,
            method: method.to_string(),
            path: path.to_string(),
            status,
            request_bytes,
            response_bytes,
            latency,
            tls_version: None,
            user_agent: user_agent.map(|s| s.to_string()),
            upstream_backend: upstream_backend.map(|s| s.to_string()),
            request_id: request_id.map(|s| s.to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn json_serialization() {
        let entry = AccessEntry {
            timestamp: "2026-03-07T12:00:00Z".to_string(),
            client_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4)), 54321),
            method: "POST".to_string(),
            path: "/api/v1/mining/submit".to_string(),
            status: 200,
            request_bytes: 256,
            response_bytes: 128,
            latency: Duration::from_micros(4200),
            tls_version: Some("TLS1.3".to_string()),
            user_agent: Some("q-miner/9.2.4".to_string()),
            upstream_backend: Some("127.0.0.1:8080".to_string()),
            request_id: Some("abc123".to_string()),
        };

        let json = entry.to_json();
        assert!(json.contains("\"ts\":\"2026-03-07T12:00:00Z\""));
        assert!(json.contains("\"ip\":\"1.2.3.4\""));
        assert!(json.contains("\"method\":\"POST\""));
        assert!(json.contains("\"path\":\"/api/v1/mining/submit\""));
        assert!(json.contains("\"status\":200"));
        assert!(json.contains("\"latency_ms\":4.20"));
        assert!(json.contains("\"tls\":\"TLS1.3\""));
        assert!(json.contains("\"ua\":\"q-miner/9.2.4\""));
        assert!(json.contains("\"upstream\":\"127.0.0.1:8080\""));
        assert!(json.contains("\"rid\":\"abc123\""));
    }

    #[test]
    fn json_escapes_special_chars() {
        let entry = AccessEntry {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            client_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1234),
            method: "GET".to_string(),
            path: "/path?q=\"test\"\nnewline\ttab".to_string(),
            status: 200,
            request_bytes: 0,
            response_bytes: 0,
            latency: Duration::from_millis(1),
            tls_version: None,
            user_agent: Some("agent\r\ninjection".to_string()),
            upstream_backend: None,
            request_id: None,
        };

        let json = entry.to_json();
        // Quotes escaped
        assert!(json.contains("\\\"test\\\""));
        // Newline escaped in path
        assert!(json.contains("\\n"));
        // Tab escaped in path
        assert!(json.contains("\\t"));
        // Carriage return escaped in user_agent
        assert!(json.contains("\\r"));
        // No raw control characters in output
        assert!(!json.contains('\n'));
        assert!(!json.contains('\r'));
        assert!(!json.contains('\t'));
    }
}
