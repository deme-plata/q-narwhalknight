/// Quillon Mail SMTP Server: Inbound email receiving
/// v7.3.2: SMTP server for receiving mail on ports 25 (MX) and 587 (submission)
///
/// Ported from axum-mail-server with wallet-based auth for Quillon ecosystem.
/// Handles SMTP state machine: HELO -> AUTH -> MAIL FROM -> RCPT TO -> DATA
/// Local delivery via storage_engine, outbound queued for MTA.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, error, info, warn};

use crate::AppState;
use q_types::*;

// ---------------------------------------------------------------------------
// SMTP session state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum SmtpState {
    Connected,
    Greeted,       // after HELO/EHLO
    Authenticated, // after AUTH
    MailFrom,      // after MAIL FROM
    RcptTo,        // after RCPT TO (ready for DATA)
}

struct SmtpSession {
    state: SmtpState,
    client_addr: SocketAddr,
    helo_domain: Option<String>,
    authenticated_wallet: Option<[u8; 32]>,
    mail_from: Option<String>,
    rcpt_to: Vec<String>,
    data: Option<String>,
}

impl SmtpSession {
    fn new(addr: SocketAddr) -> Self {
        Self {
            state: SmtpState::Connected,
            client_addr: addr,
            helo_domain: None,
            authenticated_wallet: None,
            mail_from: None,
            rcpt_to: Vec::new(),
            data: None,
        }
    }

    fn reset_transaction(&mut self) {
        self.mail_from = None;
        self.rcpt_to.clear();
        self.data = None;
        if self.authenticated_wallet.is_some() {
            self.state = SmtpState::Authenticated;
        } else {
            self.state = SmtpState::Greeted;
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Spawns SMTP listeners on the specified port (typically 25 or 587).
/// Runs until the tokio runtime shuts down.
pub async fn start_smtp_server(state: Arc<AppState>, port: u16) {
    let bind = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) => {
            error!("SMTP: failed to bind {}: {}", bind, e);
            return;
        }
    };
    info!("SMTP server listening on {}", bind);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let st = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(st, stream, addr).await {
                        debug!("SMTP connection error from {}: {}", addr, e);
                    }
                });
            }
            Err(e) => {
                error!("SMTP accept error: {}", e);
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

async fn handle_connection(
    state: Arc<AppState>,
    stream: TcpStream,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    let mut session = SmtpSession::new(addr);
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    // Greeting
    send(&mut writer, "220 mail.quillon.xyz ESMTP Quillon Mail\r\n").await?;

    loop {
        line.clear();
        let n = tokio::time::timeout(Duration::from_secs(300), reader.read_line(&mut line)).await;
        let n = match n {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                let _ = send(&mut writer, "421 Timeout\r\n").await;
                return Ok(());
            }
        };
        if n == 0 {
            break; // client disconnected
        }

        let trimmed = line.trim().to_string();
        debug!("SMTP [{}] << {}", addr, trimmed);

        let upper = trimmed.to_uppercase();

        // --- QUIT (always allowed) ---
        if upper.starts_with("QUIT") {
            send(&mut writer, "221 Bye\r\n").await?;
            break;
        }

        // --- NOOP / RSET ---
        if upper.starts_with("NOOP") {
            send(&mut writer, "250 OK\r\n").await?;
            continue;
        }
        if upper.starts_with("RSET") {
            session.reset_transaction();
            send(&mut writer, "250 OK\r\n").await?;
            continue;
        }

        // --- EHLO / HELO ---
        if upper.starts_with("EHLO") || upper.starts_with("HELO") {
            let domain = trimmed.splitn(2, ' ').nth(1).unwrap_or("unknown").to_string();
            session.helo_domain = Some(domain.clone());
            session.state = SmtpState::Greeted;
            if upper.starts_with("EHLO") {
                let resp = format!(
                    "250-mail.quillon.xyz Hello {}\r\n250-AUTH PLAIN LOGIN\r\n250-SIZE 10485760\r\n250 OK\r\n",
                    domain
                );
                send(&mut writer, &resp).await?;
            } else {
                send(&mut writer, &format!("250 Hello {}\r\n", domain)).await?;
            }
            continue;
        }

        // --- AUTH (wallet-based: AUTH PLAIN base64(\0wallet_hex\0signature_hex)) ---
        if upper.starts_with("AUTH") {
            if session.state == SmtpState::Connected {
                send(&mut writer, "503 Send EHLO first\r\n").await?;
                continue;
            }
            match handle_auth(&state, &mut session, &trimmed).await {
                Ok(resp) => send(&mut writer, &resp).await?,
                Err(e) => {
                    warn!("SMTP AUTH error from {}: {}", addr, e);
                    send(&mut writer, "535 Authentication failed\r\n").await?;
                }
            }
            continue;
        }

        // --- MAIL FROM ---
        if upper.starts_with("MAIL FROM:") {
            if session.state == SmtpState::Connected {
                send(&mut writer, "503 Send EHLO first\r\n").await?;
                continue;
            }
            let from = extract_angle_addr(&trimmed[10..]);
            session.mail_from = Some(from.clone());
            session.state = SmtpState::MailFrom;
            send(&mut writer, &format!("250 Sender <{}> OK\r\n", from)).await?;
            continue;
        }

        // --- RCPT TO ---
        if upper.starts_with("RCPT TO:") {
            if session.mail_from.is_none() {
                send(&mut writer, "503 Need MAIL FROM first\r\n").await?;
                continue;
            }
            let to = extract_angle_addr(&trimmed[8..]);

            // Local domain check: allow delivery. External: require auth.
            if !is_local_domain(&to) && session.authenticated_wallet.is_none() {
                send(&mut writer, "550 Relay denied\r\n").await?;
                continue;
            }
            session.rcpt_to.push(to.clone());
            session.state = SmtpState::RcptTo;
            send(&mut writer, &format!("250 Recipient <{}> OK\r\n", to)).await?;
            continue;
        }

        // --- DATA ---
        if upper.starts_with("DATA") {
            if session.rcpt_to.is_empty() {
                send(&mut writer, "503 Need RCPT TO first\r\n").await?;
                continue;
            }
            send(&mut writer, "354 End data with <CR><LF>.<CR><LF>\r\n").await?;

            // Read message body until lone "."
            let body = read_data(&mut reader).await?;
            session.data = Some(body);

            // Process and deliver
            match process_message(&state, &session).await {
                Ok(()) => send(&mut writer, "250 OK message accepted\r\n").await?,
                Err(e) => {
                    error!("SMTP message processing error: {}", e);
                    send(&mut writer, "451 Processing error\r\n").await?;
                }
            }
            session.reset_transaction();
            continue;
        }

        // Unknown command
        send(&mut writer, "500 Unknown command\r\n").await?;
    }

    debug!("SMTP connection closed: {}", addr);
    Ok(())
}

// ---------------------------------------------------------------------------
// AUTH handler (wallet-based)
// ---------------------------------------------------------------------------

async fn handle_auth(
    state: &Arc<AppState>,
    session: &mut SmtpSession,
    line: &str,
) -> anyhow::Result<String> {
    // We support AUTH PLAIN with base64 payload: \0<wallet_hex>\0<password_or_sig>
    let parts: Vec<&str> = line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Ok("501 Syntax error\r\n".into());
    }
    let mechanism = parts[1].to_uppercase();
    if mechanism != "PLAIN" {
        return Ok("504 Only AUTH PLAIN supported\r\n".into());
    }

    let b64data = if parts.len() >= 3 {
        parts[2].to_string()
    } else {
        return Ok("501 Missing credentials\r\n".into());
    };

    let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &b64data)?;
    let auth_str = String::from_utf8_lossy(&decoded);

    // PLAIN format: \0username\0password
    let fields: Vec<&str> = auth_str.split('\0').collect();
    if fields.len() < 3 {
        return Ok("535 Bad credentials format\r\n".into());
    }
    let username = fields[1]; // wallet hex or email
    let password = fields[2];

    // Try wallet-based auth: username is hex wallet address
    if username.len() == 64 {
        if let Ok(wallet_bytes) = hex::decode(username) {
            if wallet_bytes.len() == 32 {
                let mut addr = [0u8; 32];
                addr.copy_from_slice(&wallet_bytes);

                // Verify password against stored hash
                let pw_hashes = state.wallet_password_hashes.read().await;
                if let Some(hash) = pw_hashes.get(&addr) {
                    if bcrypt::verify(password, hash).unwrap_or(false) {
                        session.authenticated_wallet = Some(addr);
                        session.state = SmtpState::Authenticated;
                        return Ok("235 Authentication successful\r\n".into());
                    }
                }
            }
        }
    }

    Ok("535 Authentication failed\r\n".into())
}

// ---------------------------------------------------------------------------
// Message processing
// ---------------------------------------------------------------------------

async fn process_message(state: &Arc<AppState>, session: &SmtpSession) -> anyhow::Result<()> {
    let raw_data = session.data.as_deref().unwrap_or("");
    let mail_from = session.mail_from.as_deref().unwrap_or("unknown@unknown");
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Parse subject from headers
    let subject = parse_header(raw_data, "Subject").unwrap_or_else(|| "No Subject".into());

    // Split headers from body
    let body = if let Some(pos) = raw_data.find("\r\n\r\n") {
        &raw_data[pos + 4..]
    } else if let Some(pos) = raw_data.find("\n\n") {
        &raw_data[pos + 2..]
    } else {
        raw_data
    };

    for recipient in &session.rcpt_to {
        if is_local_domain(recipient) {
            // Local delivery: store as EmailMessage
            let wallet = resolve_wallet_for_email(state, recipient).await;
            let email_id = uuid::Uuid::new_v4().to_string();

            let email = EmailMessage {
                id: email_id,
                from_wallet: session.authenticated_wallet.unwrap_or([0u8; 32]),
                from_email: Some(mail_from.to_string()),
                to_wallet: wallet,
                to_email: Some(recipient.clone()),
                subject: subject.clone(),
                body: body.to_string(),
                body_html: None,
                encrypted: false,
                signature: vec![],
                timestamp,
                read: false,
                folder: "inbox".to_string(),
                thread_id: parse_header(raw_data, "Thread-Id"),
                in_reply_to: parse_header(raw_data, "In-Reply-To"),
                crypto_transfer: None,
                delivery_method: DeliveryMethod::SmtpInbound,
            };

            if let Err(e) = state.storage_engine.save_email(&email).await {
                error!("Failed to save inbound email for {}: {}", recipient, e);
            } else {
                info!("SMTP: delivered local email to {}", recipient);
            }
        } else {
            // Outbound: queue for MTA delivery
            let outbound = OutboundEmail {
                id: uuid::Uuid::new_v4().to_string(),
                from_wallet: session.authenticated_wallet.unwrap_or([0u8; 32]),
                from_email: mail_from.to_string(),
                to_email: recipient.clone(),
                subject: subject.clone(),
                body: body.to_string(),
                body_html: None,
                timestamp,
                status: OutboundStatus::Pending,
                retry_count: 0,
                last_error: None,
                next_retry_at: None,
                email_id: None,
            };

            if let Err(e) = state.storage_engine.save_outbound_email(&outbound).await {
                error!("Failed to queue outbound email to {}: {}", recipient, e);
            } else {
                info!("SMTP: queued outbound email to {}", recipient);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn send(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    msg: &str,
) -> anyhow::Result<()> {
    writer.write_all(msg.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_data(reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>) -> anyhow::Result<String> {
    let mut data = String::new();
    let mut line = String::new();
    let max_size: usize = 10 * 1024 * 1024; // 10 MB

    loop {
        line.clear();
        let n = tokio::time::timeout(Duration::from_secs(300), reader.read_line(&mut line)).await;
        match n {
            Ok(Ok(0)) => break,
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err(anyhow::anyhow!("DATA timeout")),
        }

        if line.trim() == "." {
            break;
        }

        // Dot-stuffing removal
        let clean = if line.starts_with("..") {
            &line[1..]
        } else {
            &line
        };

        if data.len() + clean.len() > max_size {
            return Err(anyhow::anyhow!("Message too large"));
        }
        data.push_str(clean);
    }

    Ok(data)
}

fn extract_angle_addr(s: &str) -> String {
    let s = s.trim();
    if let Some(start) = s.find('<') {
        if let Some(end) = s.find('>') {
            return s[start + 1..end].trim().to_string();
        }
    }
    // No angle brackets: return trimmed
    s.to_string()
}

fn is_local_domain(email: &str) -> bool {
    let local = ["quillon.xyz", "mail.quillon.xyz"];
    if let Some(domain) = email.split('@').nth(1) {
        local.iter().any(|d| d.eq_ignore_ascii_case(domain))
    } else {
        false
    }
}

fn parse_header(raw: &str, name: &str) -> Option<String> {
    let prefix = format!("{}:", name);
    let prefix_lower = prefix.to_lowercase();
    for line in raw.lines() {
        if line.is_empty() || line == "\r" {
            break; // end of headers
        }
        if line.to_lowercase().starts_with(&prefix_lower) {
            return Some(line[prefix.len()..].trim().to_string());
        }
    }
    None
}

/// Resolve a wallet address for a local email address.
/// Returns `Some([u8; 32])` if a wallet is registered for the address, else `None`.
async fn resolve_wallet_for_email(state: &Arc<AppState>, email: &str) -> Option<[u8; 32]> {
    // Email-to-wallet lookup not yet implemented in storage layer
    let _ = (state, email); // suppress unused warnings
    None
}
