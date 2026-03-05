/// Quillon Mail: Mail Transport Agent (MTA)
/// v8.9.5: Outbound SMTP delivery with relay support and MX resolution
///
/// If Q_SMTP_RELAY is set (e.g., "89.149.241.126"), all outbound mail is
/// forwarded to that host on port 25 instead of connecting to MX directly.
/// This works around Contabo blocking outbound port 25.

use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tracing::{debug, error, info, warn};

use crate::AppState;
use q_types::*;

/// Mail Transport Agent for outbound SMTP delivery
pub struct MailTransportAgent {
    state: Arc<AppState>,
    /// Optional SMTP relay host (from Q_SMTP_RELAY env var)
    relay_host: Option<String>,
}

impl MailTransportAgent {
    pub fn new(state: Arc<AppState>) -> Self {
        let relay_host = std::env::var("Q_SMTP_RELAY").ok().filter(|s| !s.is_empty());
        if let Some(ref relay) = relay_host {
            info!("📤 [MTA] Using SMTP relay: {}", relay);
        }
        Self { state, relay_host }
    }

    /// Start the MTA delivery loop (runs every 30 seconds)
    pub async fn start(&self) {
        info!("📤 [MTA] Mail Transport Agent started (relay: {})",
            self.relay_host.as_deref().unwrap_or("direct MX"));

        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;

            match self.process_outbound_queue().await {
                Ok(delivered) => {
                    if delivered > 0 {
                        info!("📤 [MTA] Delivered {} outbound emails", delivered);
                    }
                }
                Err(e) => {
                    warn!("📤 [MTA] Queue processing error: {}", e);
                }
            }
        }
    }

    /// Process pending outbound emails
    async fn process_outbound_queue(&self) -> anyhow::Result<usize> {
        let messages = self.state.storage_engine.claim_outbound_emails(10).await?;

        if messages.is_empty() {
            return Ok(0);
        }

        info!("📤 [MTA] Processing {} outbound emails", messages.len());
        let mut delivered = 0;
        for msg in &messages {
            match self.deliver_message(msg).await {
                Ok(()) => {
                    self.state
                        .storage_engine
                        .mark_outbound_delivered(&msg.id)
                        .await?;
                    delivered += 1;
                }
                Err(e) => {
                    let err_str = format!("{}", e);
                    warn!(
                        "📤 [MTA] Delivery failed for {} to {}: {}",
                        msg.id, msg.to_email, err_str
                    );
                    self.state
                        .storage_engine
                        .mark_outbound_failed(&msg.id, &err_str)
                        .await?;
                }
            }
        }

        Ok(delivered)
    }

    /// Deliver a single message via SMTP
    async fn deliver_message(&self, msg: &OutboundEmail) -> anyhow::Result<()> {
        // If relay is configured, use it directly instead of MX lookup
        if let Some(ref relay) = self.relay_host {
            info!("📤 [MTA] Relaying email {} to {} via relay {}", msg.id, msg.to_email, relay);
            return self.attempt_smtp_delivery(msg, relay).await;
        }

        // Direct MX delivery (only works if port 25 outbound is open)
        let domain = msg
            .to_email
            .split('@')
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("Invalid recipient email: {}", msg.to_email))?;

        let mx_hosts = self.resolve_mx(domain).await?;

        if mx_hosts.is_empty() {
            return Err(anyhow::anyhow!("No MX records found for domain: {}", domain));
        }

        let mut last_error = String::new();
        for (priority, host) in &mx_hosts {
            debug!(
                "📤 [MTA] Trying MX {} (priority {}) for {}",
                host, priority, msg.to_email
            );

            match self.attempt_smtp_delivery(msg, host).await {
                Ok(()) => {
                    info!(
                        "✅ [MTA] Delivered email {} to {} via MX {}",
                        msg.id, msg.to_email, host
                    );
                    return Ok(());
                }
                Err(e) => {
                    last_error = format!("{}", e);
                    warn!("📤 [MTA] MX {} failed: {}", host, last_error);
                }
            }
        }

        Err(anyhow::anyhow!(
            "All MX hosts failed for {}: {}",
            domain,
            last_error
        ))
    }

    /// Resolve MX records for a domain
    async fn resolve_mx(&self, domain: &str) -> anyhow::Result<Vec<(u16, String)>> {
        use hickory_resolver::TokioAsyncResolver;
        use hickory_resolver::config::*;

        let resolver = TokioAsyncResolver::tokio(
            ResolverConfig::default(),
            ResolverOpts::default(),
        );

        match resolver.mx_lookup(domain).await {
            Ok(mx_records) => {
                let mut records: Vec<(u16, String)> = mx_records
                    .iter()
                    .map(|mx| {
                        let host = mx.exchange().to_string();
                        let host = host.trim_end_matches('.').to_string();
                        (mx.preference(), host)
                    })
                    .collect();

                records.sort_by_key(|(priority, _)| *priority);
                Ok(records)
            }
            Err(e) => {
                warn!(
                    "📤 [MTA] MX lookup failed for {}: {}. Trying domain directly.",
                    domain, e
                );
                Ok(vec![(99, domain.to_string())])
            }
        }
    }

    /// Attempt SMTP delivery to a specific host (MX server or relay)
    async fn attempt_smtp_delivery(
        &self,
        msg: &OutboundEmail,
        smtp_host: &str,
    ) -> anyhow::Result<()> {
        let port = std::env::var("Q_SMTP_RELAY_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(25);
        let addr = format!("{}:{}", smtp_host, port);

        info!("📤 [MTA] Connecting to {} for {}", addr, msg.to_email);

        let stream = tokio::time::timeout(
            Duration::from_secs(30),
            TcpStream::connect(&addr),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Connection timeout to {}", addr))?
        .map_err(|e| anyhow::anyhow!("Connection failed to {}: {}", addr, e))?;

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);

        // Read greeting
        let greeting = read_smtp_response(&mut reader).await?;
        if !greeting.starts_with("220") {
            return Err(anyhow::anyhow!("Bad greeting from {}: {}", smtp_host, greeting));
        }

        // EHLO
        let hostname = std::env::var("MAIL_HOSTNAME").unwrap_or_else(|_| "mail.quillon.xyz".to_string());
        write_smtp_command(&mut writer, &format!("EHLO {}\r\n", hostname)).await?;
        let ehlo_resp = read_smtp_response(&mut reader).await?;
        if !ehlo_resp.starts_with("250") {
            return Err(anyhow::anyhow!("EHLO rejected: {}", ehlo_resp));
        }

        // MAIL FROM
        write_smtp_command(
            &mut writer,
            &format!("MAIL FROM:<{}>\r\n", msg.from_email),
        )
        .await?;
        let from_resp = read_smtp_response(&mut reader).await?;
        if !from_resp.starts_with("250") {
            return Err(anyhow::anyhow!("MAIL FROM rejected: {}", from_resp));
        }

        // RCPT TO
        write_smtp_command(
            &mut writer,
            &format!("RCPT TO:<{}>\r\n", msg.to_email),
        )
        .await?;
        let rcpt_resp = read_smtp_response(&mut reader).await?;
        if rcpt_resp.starts_with("5") {
            return Err(anyhow::anyhow!(
                "Permanent RCPT failure: {}",
                rcpt_resp
            ));
        }
        if !rcpt_resp.starts_with("250") {
            return Err(anyhow::anyhow!("RCPT TO rejected: {}", rcpt_resp));
        }

        // DATA
        write_smtp_command(&mut writer, "DATA\r\n").await?;
        let data_resp = read_smtp_response(&mut reader).await?;
        if !data_resp.starts_with("354") {
            return Err(anyhow::anyhow!("DATA rejected: {}", data_resp));
        }

        // Send message headers + body
        let message_id = format!(
            "<{}@quillon.xyz>",
            msg.email_id.as_deref().unwrap_or(&msg.id)
        );
        let date = chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S +0000").to_string();

        let headers = format!(
            "From: {}\r\n\
             To: {}\r\n\
             Subject: {}\r\n\
             Date: {}\r\n\
             Message-ID: {}\r\n\
             MIME-Version: 1.0\r\n\
             Content-Type: text/plain; charset=UTF-8\r\n\
             X-Mailer: Quillon Mail v8.9.5\r\n\
             X-Blockchain: Q-NarwhalKnight\r\n\
             \r\n",
            msg.from_email, msg.to_email, msg.subject, date, message_id
        );

        writer.write_all(headers.as_bytes()).await?;

        // Body (with dot-stuffing: lines starting with "." get an extra ".")
        for line in msg.body.lines() {
            if line.starts_with('.') {
                writer.write_all(b".").await?;
            }
            writer.write_all(line.as_bytes()).await?;
            writer.write_all(b"\r\n").await?;
        }

        // End of message
        writer.write_all(b"\r\n.\r\n").await?;
        writer.flush().await?;

        let end_resp = read_smtp_response(&mut reader).await?;
        if !end_resp.starts_with("250") {
            return Err(anyhow::anyhow!("Message rejected: {}", end_resp));
        }

        // QUIT
        let _ = write_smtp_command(&mut writer, "QUIT\r\n").await;

        info!("✅ [MTA] Email {} delivered to {} via {}", msg.id, msg.to_email, smtp_host);
        Ok(())
    }
}

/// Read an SMTP response (handles multi-line responses)
async fn read_smtp_response<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
) -> anyhow::Result<String> {
    let mut response = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        let n = tokio::time::timeout(Duration::from_secs(30), reader.read_line(&mut line))
            .await
            .map_err(|_| anyhow::anyhow!("SMTP read timeout"))?
            .map_err(|e| anyhow::anyhow!("SMTP read error: {}", e))?;

        if n == 0 {
            return Err(anyhow::anyhow!("SMTP connection closed"));
        }

        response.push_str(&line);

        // Multi-line responses have a dash after the code (e.g., "250-")
        // Final line has a space (e.g., "250 ")
        if line.len() >= 4 && line.chars().nth(3) != Some('-') {
            break;
        }
    }

    Ok(response.trim().to_string())
}

/// Write an SMTP command
async fn write_smtp_command<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    command: &str,
) -> anyhow::Result<()> {
    writer.write_all(command.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}
