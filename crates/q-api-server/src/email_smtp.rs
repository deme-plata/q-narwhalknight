/// Quillon Mail SMTP Server: Inbound email receiving
/// v7.3.2: SMTP server for receiving mail on ports 25 (MX) and 587 (submission)
/// v8.7.5: STARTTLS support — required for Gmail/Outlook/Yahoo inbound delivery
///
/// Handles SMTP state machine: HELO -> STARTTLS -> AUTH -> MAIL FROM -> RCPT TO -> DATA
/// Local delivery via storage_engine, outbound queued for MTA.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
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
    tls_active: bool,
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
            tls_active: false,
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
// TLS configuration
// ---------------------------------------------------------------------------

/// Load TLS config from Let's Encrypt certs.
/// Tries mail.quillon.xyz first, falls back to quillon.xyz.
fn load_tls_config() -> Option<Arc<rustls::ServerConfig>> {
    let cert_paths = [
        (
            "/etc/letsencrypt/live/mail.quillon.xyz/fullchain.pem",
            "/etc/letsencrypt/live/mail.quillon.xyz/privkey.pem",
        ),
        (
            "/etc/letsencrypt/live/quillon.xyz/fullchain.pem",
            "/etc/letsencrypt/live/quillon.xyz/privkey.pem",
        ),
        (
            "/etc/letsencrypt/live/beta.quillon.xyz/fullchain.pem",
            "/etc/letsencrypt/live/beta.quillon.xyz/privkey.pem",
        ),
    ];

    // Also check environment override
    let env_cert = std::env::var("SMTP_TLS_CERT").ok();
    let env_key = std::env::var("SMTP_TLS_KEY").ok();

    let mut all_paths: Vec<(&str, &str)> = Vec::new();
    if let (Some(c), Some(k)) = (env_cert.as_deref(), env_key.as_deref()) {
        all_paths.push((c, k));
    }
    for (c, k) in &cert_paths {
        all_paths.push((c, k));
    }

    // Ensure ring crypto provider is installed for rustls
    let _ = rustls::crypto::ring::default_provider().install_default();

    for (cert_path, key_path) in &all_paths {
        match load_certs_and_key(cert_path, key_path) {
            Ok((certs, key)) => {
                match rustls::ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(certs, key)
                {
                    Ok(config) => {
                        info!("📧 [SMTP TLS] Loaded certificate from {}", cert_path);
                        return Some(Arc::new(config));
                    }
                    Err(e) => {
                        warn!("📧 [SMTP TLS] Failed to build config from {}: {}", cert_path, e);
                    }
                }
            }
            Err(e) => {
                debug!("📧 [SMTP TLS] Cert not at {}: {}", cert_path, e);
            }
        }
    }

    warn!("📧 [SMTP TLS] No TLS certificates found — STARTTLS will be unavailable");
    warn!("📧 [SMTP TLS] Gmail/Outlook/Yahoo may refuse to deliver mail without TLS!");
    None
}

fn load_certs_and_key(
    cert_path: &str,
    key_path: &str,
) -> anyhow::Result<(Vec<rustls::pki_types::CertificateDer<'static>>, rustls::pki_types::PrivateKeyDer<'static>)> {
    use std::io::BufReader as StdBufReader;

    let cert_file = std::fs::File::open(cert_path)?;
    let mut cert_reader = StdBufReader::new(cert_file);
    let certs: Vec<_> = rustls_pemfile::certs(&mut cert_reader)
        .filter_map(|r| r.ok())
        .collect();
    if certs.is_empty() {
        return Err(anyhow::anyhow!("No certificates found in {}", cert_path));
    }

    let key_file = std::fs::File::open(key_path)?;
    let mut key_reader = StdBufReader::new(key_file);
    let key = rustls_pemfile::private_key(&mut key_reader)?
        .ok_or_else(|| anyhow::anyhow!("No private key found in {}", key_path))?;

    Ok((certs, key))
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
            error!("📧 SMTP: failed to bind {}: {}", bind, e);
            return;
        }
    };

    // Load TLS config once at startup
    let tls_config = load_tls_config();
    let tls_acceptor = tls_config.map(tokio_rustls::TlsAcceptor::from);

    if tls_acceptor.is_some() {
        info!("📧 SMTP server listening on {} (STARTTLS enabled)", bind);
    } else {
        warn!("📧 SMTP server listening on {} (NO TLS — inbound mail may be rejected by senders)", bind);
    }

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let st = state.clone();
                let acceptor = tls_acceptor.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(st, stream, addr, acceptor).await {
                        debug!("📧 SMTP connection error from {}: {}", addr, e);
                    }
                });
            }
            Err(e) => {
                error!("📧 SMTP accept error: {}", e);
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Connection handler — plain text phase, then optional TLS upgrade
// ---------------------------------------------------------------------------

async fn handle_connection(
    state: Arc<AppState>,
    stream: TcpStream,
    addr: SocketAddr,
    tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
) -> anyhow::Result<()> {
    let mut session = SmtpSession::new(addr);
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    // Greeting
    send_raw(&mut writer, "220 mail.quillon.xyz ESMTP Quillon Mail\r\n").await?;

    let has_tls = tls_acceptor.is_some();
    let mut line = String::new();

    loop {
        line.clear();
        let n = tokio::time::timeout(Duration::from_secs(300), reader.read_line(&mut line)).await;
        let n = match n {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                let _ = send_raw(&mut writer, "421 Timeout\r\n").await;
                return Ok(());
            }
        };
        if n == 0 {
            break;
        }

        let trimmed = line.trim().to_string();
        debug!("SMTP [{}] << {}", addr, trimmed);
        let upper = trimmed.to_uppercase();

        // --- QUIT ---
        if upper.starts_with("QUIT") {
            send_raw(&mut writer, "221 Bye\r\n").await?;
            break;
        }

        // --- NOOP / RSET ---
        if upper.starts_with("NOOP") {
            send_raw(&mut writer, "250 OK\r\n").await?;
            continue;
        }
        if upper.starts_with("RSET") {
            session.reset_transaction();
            send_raw(&mut writer, "250 OK\r\n").await?;
            continue;
        }

        // --- EHLO / HELO ---
        if upper.starts_with("EHLO") || upper.starts_with("HELO") {
            let domain = trimmed.splitn(2, ' ').nth(1).unwrap_or("unknown").to_string();
            session.helo_domain = Some(domain.clone());
            session.state = SmtpState::Greeted;
            if upper.starts_with("EHLO") {
                let mut resp = format!("250-mail.quillon.xyz Hello {}\r\n", domain);
                if has_tls {
                    resp.push_str("250-STARTTLS\r\n");
                }
                resp.push_str("250-AUTH PLAIN LOGIN\r\n");
                resp.push_str("250-SIZE 10485760\r\n");
                resp.push_str("250 OK\r\n");
                send_raw(&mut writer, &resp).await?;
            } else {
                send_raw(&mut writer, &format!("250 Hello {}\r\n", domain)).await?;
            }
            continue;
        }

        // --- STARTTLS ---
        if upper.starts_with("STARTTLS") {
            if let Some(ref acceptor) = tls_acceptor {
                send_raw(&mut writer, "220 Ready to start TLS\r\n").await?;

                // Reunite the split halves back into a TcpStream for TLS handshake
                let tcp_stream = reader.into_inner().reunite(writer)
                    .map_err(|e| anyhow::anyhow!("Failed to reunite TCP stream: {}", e))?;

                // Perform TLS handshake
                let tls_stream = match tokio::time::timeout(
                    Duration::from_secs(30),
                    acceptor.accept(tcp_stream),
                ).await {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        warn!("📧 SMTP STARTTLS handshake failed from {}: {}", addr, e);
                        return Ok(());
                    }
                    Err(_) => {
                        warn!("📧 SMTP STARTTLS handshake timeout from {}", addr);
                        return Ok(());
                    }
                };

                info!("📧 SMTP STARTTLS active for {}", addr);
                session.tls_active = true;
                // Reset state — RFC 3207 requires client to re-EHLO after STARTTLS
                session.state = SmtpState::Connected;
                session.helo_domain = None;

                // Continue SMTP session over TLS
                let (tls_reader, tls_writer) = tokio::io::split(tls_stream);
                let tls_reader = BufReader::new(tls_reader);
                return smtp_loop(state, tls_reader, tls_writer, &mut session).await;
            } else {
                send_raw(&mut writer, "454 TLS not available\r\n").await?;
                continue;
            }
        }

        // --- AUTH ---
        if upper.starts_with("AUTH") {
            if session.state == SmtpState::Connected {
                send_raw(&mut writer, "503 Send EHLO first\r\n").await?;
                continue;
            }
            match handle_auth(&state, &mut session, &trimmed).await {
                Ok(resp) => send_raw(&mut writer, &resp).await?,
                Err(e) => {
                    warn!("SMTP AUTH error from {}: {}", addr, e);
                    send_raw(&mut writer, "535 Authentication failed\r\n").await?;
                }
            }
            continue;
        }

        // --- MAIL FROM ---
        if upper.starts_with("MAIL FROM:") {
            if session.state == SmtpState::Connected {
                send_raw(&mut writer, "503 Send EHLO first\r\n").await?;
                continue;
            }
            let from = extract_angle_addr(&trimmed[10..]);
            session.mail_from = Some(from.clone());
            session.state = SmtpState::MailFrom;
            send_raw(&mut writer, &format!("250 Sender <{}> OK\r\n", from)).await?;
            continue;
        }

        // --- RCPT TO ---
        if upper.starts_with("RCPT TO:") {
            if session.mail_from.is_none() {
                send_raw(&mut writer, "503 Need MAIL FROM first\r\n").await?;
                continue;
            }
            let to = extract_angle_addr(&trimmed[8..]);
            if !is_local_domain(&to) && session.authenticated_wallet.is_none() && !is_trusted_relay(&addr) {
                send_raw(&mut writer, "550 Relay denied\r\n").await?;
                continue;
            }
            session.rcpt_to.push(to.clone());
            session.state = SmtpState::RcptTo;
            send_raw(&mut writer, &format!("250 Recipient <{}> OK\r\n", to)).await?;
            continue;
        }

        // --- DATA ---
        if upper.starts_with("DATA") {
            if session.rcpt_to.is_empty() {
                send_raw(&mut writer, "503 Need RCPT TO first\r\n").await?;
                continue;
            }
            send_raw(&mut writer, "354 End data with <CR><LF>.<CR><LF>\r\n").await?;
            let body = read_data_generic(&mut reader).await?;
            session.data = Some(body);
            match process_message(&state, &session).await {
                Ok(()) => send_raw(&mut writer, "250 OK message accepted\r\n").await?,
                Err(e) => {
                    error!("SMTP message processing error: {}", e);
                    send_raw(&mut writer, "451 Processing error\r\n").await?;
                }
            }
            session.reset_transaction();
            continue;
        }

        send_raw(&mut writer, "500 Unknown command\r\n").await?;
    }

    debug!("SMTP connection closed: {}", addr);
    Ok(())
}

// ---------------------------------------------------------------------------
// Generic SMTP command loop (used for TLS-upgraded connections)
// ---------------------------------------------------------------------------

async fn smtp_loop<R, W>(
    state: Arc<AppState>,
    mut reader: BufReader<R>,
    mut writer: W,
    session: &mut SmtpSession,
) -> anyhow::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let addr = session.client_addr;
    let mut line = String::new();

    loop {
        line.clear();
        let n = tokio::time::timeout(Duration::from_secs(300), reader.read_line(&mut line)).await;
        let n = match n {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                let _ = send_generic(&mut writer, "421 Timeout\r\n").await;
                return Ok(());
            }
        };
        if n == 0 {
            break;
        }

        let trimmed = line.trim().to_string();
        debug!("SMTP [{}] (TLS) << {}", addr, trimmed);
        let upper = trimmed.to_uppercase();

        if upper.starts_with("QUIT") {
            send_generic(&mut writer, "221 Bye\r\n").await?;
            break;
        }
        if upper.starts_with("NOOP") {
            send_generic(&mut writer, "250 OK\r\n").await?;
            continue;
        }
        if upper.starts_with("RSET") {
            session.reset_transaction();
            send_generic(&mut writer, "250 OK\r\n").await?;
            continue;
        }

        // --- EHLO / HELO (re-issued after STARTTLS per RFC 3207) ---
        if upper.starts_with("EHLO") || upper.starts_with("HELO") {
            let domain = trimmed.splitn(2, ' ').nth(1).unwrap_or("unknown").to_string();
            session.helo_domain = Some(domain.clone());
            session.state = SmtpState::Greeted;
            if upper.starts_with("EHLO") {
                let resp = format!(
                    "250-mail.quillon.xyz Hello {}\r\n250-AUTH PLAIN LOGIN\r\n250-SIZE 10485760\r\n250 OK\r\n",
                    domain
                );
                send_generic(&mut writer, &resp).await?;
            } else {
                send_generic(&mut writer, &format!("250 Hello {}\r\n", domain)).await?;
            }
            continue;
        }

        // STARTTLS already done — reject duplicate
        if upper.starts_with("STARTTLS") {
            send_generic(&mut writer, "503 TLS already active\r\n").await?;
            continue;
        }

        // --- AUTH ---
        if upper.starts_with("AUTH") {
            if session.state == SmtpState::Connected {
                send_generic(&mut writer, "503 Send EHLO first\r\n").await?;
                continue;
            }
            match handle_auth(&state, session, &trimmed).await {
                Ok(resp) => send_generic(&mut writer, &resp).await?,
                Err(e) => {
                    warn!("SMTP AUTH error from {}: {}", addr, e);
                    send_generic(&mut writer, "535 Authentication failed\r\n").await?;
                }
            }
            continue;
        }

        // --- MAIL FROM ---
        if upper.starts_with("MAIL FROM:") {
            if session.state == SmtpState::Connected {
                send_generic(&mut writer, "503 Send EHLO first\r\n").await?;
                continue;
            }
            let from = extract_angle_addr(&trimmed[10..]);
            session.mail_from = Some(from.clone());
            session.state = SmtpState::MailFrom;
            send_generic(&mut writer, &format!("250 Sender <{}> OK\r\n", from)).await?;
            continue;
        }

        // --- RCPT TO ---
        if upper.starts_with("RCPT TO:") {
            if session.mail_from.is_none() {
                send_generic(&mut writer, "503 Need MAIL FROM first\r\n").await?;
                continue;
            }
            let to = extract_angle_addr(&trimmed[8..]);
            if !is_local_domain(&to) && session.authenticated_wallet.is_none() && !is_trusted_relay(&session.client_addr) {
                send_generic(&mut writer, "550 Relay denied\r\n").await?;
                continue;
            }
            session.rcpt_to.push(to.clone());
            session.state = SmtpState::RcptTo;
            send_generic(&mut writer, &format!("250 Recipient <{}> OK\r\n", to)).await?;
            continue;
        }

        // --- DATA ---
        if upper.starts_with("DATA") {
            if session.rcpt_to.is_empty() {
                send_generic(&mut writer, "503 Need RCPT TO first\r\n").await?;
                continue;
            }
            send_generic(&mut writer, "354 End data with <CR><LF>.<CR><LF>\r\n").await?;
            let body = read_data_generic(&mut reader).await?;
            session.data = Some(body);
            match process_message(&state, session).await {
                Ok(()) => send_generic(&mut writer, "250 OK message accepted\r\n").await?,
                Err(e) => {
                    error!("SMTP message processing error: {}", e);
                    send_generic(&mut writer, "451 Processing error\r\n").await?;
                }
            }
            session.reset_transaction();
            continue;
        }

        send_generic(&mut writer, "500 Unknown command\r\n").await?;
    }

    debug!("SMTP (TLS) connection closed: {}", addr);
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
    let username = fields[1];
    let password = fields[2];

    // Try wallet-based auth: username is hex wallet address
    if username.len() == 64 {
        if let Ok(wallet_bytes) = hex::decode(username) {
            if wallet_bytes.len() == 32 {
                let mut addr = [0u8; 32];
                addr.copy_from_slice(&wallet_bytes);

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

    let subject = parse_header(raw_data, "Subject").unwrap_or_else(|| "No Subject".into());

    let body = if let Some(pos) = raw_data.find("\r\n\r\n") {
        &raw_data[pos + 4..]
    } else if let Some(pos) = raw_data.find("\n\n") {
        &raw_data[pos + 2..]
    } else {
        raw_data
    };

    for recipient in &session.rcpt_to {
        if is_local_domain(recipient) {
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
                info!("📧 SMTP: delivered local email to {} (TLS={})", recipient, session.tls_active);
            }
        } else {
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
                info!("📧 SMTP: queued outbound email to {}", recipient);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Send on a concrete OwnedWriteHalf (plain text phase)
async fn send_raw(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    msg: &str,
) -> anyhow::Result<()> {
    writer.write_all(msg.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

/// Send on any AsyncWrite (TLS phase)
async fn send_generic<W: AsyncWrite + Unpin>(
    writer: &mut W,
    msg: &str,
) -> anyhow::Result<()> {
    writer.write_all(msg.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

/// Read DATA body from any AsyncBufRead
async fn read_data_generic<R: AsyncBufRead + Unpin>(reader: &mut R) -> anyhow::Result<String> {
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

/// Check if a client IP is in the trusted relay list (Q_SMTP_TRUSTED_RELAY_IPS).
/// Trusted relays can send mail to external recipients without authentication.
fn is_trusted_relay(addr: &SocketAddr) -> bool {
    let ip_str = addr.ip().to_string();
    if let Ok(trusted) = std::env::var("Q_SMTP_TRUSTED_RELAY_IPS") {
        trusted.split(',').any(|t| t.trim() == ip_str)
    } else {
        false
    }
}

fn parse_header(raw: &str, name: &str) -> Option<String> {
    let prefix = format!("{}:", name);
    let prefix_lower = prefix.to_lowercase();
    for line in raw.lines() {
        if line.is_empty() || line == "\r" {
            break;
        }
        if line.to_lowercase().starts_with(&prefix_lower) {
            return Some(line[prefix.len()..].trim().to_string());
        }
    }
    None
}

async fn resolve_wallet_for_email(state: &Arc<AppState>, email: &str) -> Option<[u8; 32]> {
    // Extract local part (before @)
    let local_part = email.split('@').next().unwrap_or("").trim().to_lowercase();
    if local_part.is_empty() {
        return None;
    }

    // 1. Try as custom alias (e.g. "demetri@quillon.xyz" → alias "demetri")
    if let Ok(Some(wallet_hex)) = state.storage_engine.get_email_alias_wallet(&local_part).await {
        if let Ok(bytes) = hex::decode(&wallet_hex) {
            if bytes.len() == 32 {
                let mut addr = [0u8; 32];
                addr.copy_from_slice(&bytes);
                info!("📧 [SMTP INBOUND] Resolved alias '{}' to wallet {}", local_part, &wallet_hex[..8]);
                return Some(addr);
            }
        }
    }

    // 2. Try as direct wallet hex (e.g. "efca1e8c@quillon.xyz" → first 8 chars of wallet)
    //    Check all known wallets for a prefix match
    if local_part.len() >= 8 && local_part.chars().all(|c| c.is_ascii_hexdigit()) {
        let balances = state.wallet_balances.read().await;
        for addr in balances.keys() {
            let addr_hex = hex::encode(addr);
            if addr_hex.starts_with(&local_part) {
                info!("📧 [SMTP INBOUND] Resolved wallet prefix '{}' to {}", local_part, &addr_hex[..16]);
                return Some(*addr);
            }
        }
    }

    warn!("📧 [SMTP INBOUND] Could not resolve '{}' to any wallet", local_part);
    None
}
