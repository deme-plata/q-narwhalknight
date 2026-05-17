//! Automatic OCSP staple fetching and periodic refresh.
//!
//! Extracts the OCSP responder URL from the server certificate's Authority
//! Information Access (AIA) extension, builds an OCSP request, and POSTs it
//! to the responder. The DER-encoded response is saved to disk so that
//! `build_tls_config()` can staple it into the TLS handshake.
//!
//! Runs as a background task that refreshes the staple periodically
//! (default: every 12 hours) and triggers a TLS hot-reload on success.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};
use x509_parser::prelude::*;

use crate::acceptor::SharedTlsConfig;
use crate::config::TlsConfig;

/// Status of the OCSP auto-fetcher, exposed to the admin endpoint.
#[derive(Debug, Clone)]
pub struct OcspStatus {
    pub responder_url: Option<String>,
    pub last_fetch: Option<std::time::Instant>,
    pub last_error: Option<String>,
    pub response_bytes: usize,
    pub refresh_count: u64,
}

impl Default for OcspStatus {
    fn default() -> Self {
        Self {
            responder_url: None,
            last_fetch: None,
            last_error: None,
            response_bytes: 0,
            refresh_count: 0,
        }
    }
}

/// Shared OCSP status for admin visibility.
pub type SharedOcspStatus = Arc<parking_lot::RwLock<OcspStatus>>;

/// Extract the OCSP responder URL from a PEM certificate file.
///
/// Parses the leaf certificate's Authority Information Access (AIA) extension
/// and returns the OCSP responder HTTP URL if present.
pub fn extract_ocsp_responder_url(cert_path: &Path) -> Result<Option<String>> {
    let pem_data = std::fs::read(cert_path)
        .with_context(|| format!("Cannot read cert {}", cert_path.display()))?;

    // Parse first PEM block (leaf certificate)
    let (_, pem) = x509_parser::pem::parse_x509_pem(&pem_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse PEM: {:?}", e))?;

    let cert = pem.parse_x509()
        .map_err(|e| anyhow::anyhow!("Failed to parse X.509: {:?}", e))?;

    // Look for Authority Information Access extension
    for ext in cert.extensions() {
        if let ParsedExtension::AuthorityInfoAccess(aia) = ext.parsed_extension() {
            for access_desc in aia.accessdescs.iter() {
                // OID 1.3.6.1.5.5.7.48.1 = id-ad-ocsp
                if access_desc.access_method.to_id_string() == "1.3.6.1.5.5.7.48.1" {
                    if let GeneralName::URI(url) = access_desc.access_location {
                        return Ok(Some(url.to_string()));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Build a DER-encoded OCSP request for the given certificate.
///
/// The OCSP request contains:
/// - SHA-256 hash of the issuer's distinguished name
/// - SHA-256 hash of the issuer's public key
/// - Serial number of the certificate to check
///
/// Returns the raw DER bytes ready to POST to the OCSP responder.
pub fn build_ocsp_request_der(cert_path: &Path) -> Result<Vec<u8>> {
    let pem_data = std::fs::read(cert_path)
        .with_context(|| format!("Cannot read cert {}", cert_path.display()))?;

    // Parse all PEM blocks — first is leaf, second is issuer
    let pems = Pem::iter_from_buffer(&pem_data).collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to parse PEM chain: {:?}", e))?;

    if pems.len() < 2 {
        anyhow::bail!(
            "Certificate chain needs at least 2 certs (leaf + issuer) for OCSP request, found {}",
            pems.len()
        );
    }

    let leaf = pems[0].parse_x509()
        .map_err(|e| anyhow::anyhow!("Failed to parse leaf cert: {:?}", e))?;
    let issuer = pems[1].parse_x509()
        .map_err(|e| anyhow::anyhow!("Failed to parse issuer cert: {:?}", e))?;

    // Hash issuer name (DER-encoded)
    let issuer_name_hash = {
        let mut hasher = Sha256::new();
        hasher.update(issuer.subject().as_raw());
        hasher.finalize()
    };

    // Hash issuer public key (the BIT STRING content, not the whole SubjectPublicKeyInfo)
    let issuer_key_hash = {
        let mut hasher = Sha256::new();
        hasher.update(&*issuer.public_key().subject_public_key.data);
        hasher.finalize()
    };

    let serial = leaf.raw_serial();

    // Build the OCSP request DER manually.
    // Structure (RFC 6960):
    //   OCSPRequest ::= SEQUENCE {
    //     tbsRequest TBSRequest
    //   }
    //   TBSRequest ::= SEQUENCE {
    //     requestList SEQUENCE OF Request
    //   }
    //   Request ::= SEQUENCE {
    //     reqCert CertID
    //   }
    //   CertID ::= SEQUENCE {
    //     hashAlgorithm AlgorithmIdentifier (SHA-256 = 2.16.840.1.101.3.4.2.1)
    //     issuerNameHash OCTET STRING
    //     issuerKeyHash  OCTET STRING
    //     serialNumber   INTEGER
    //   }

    // SHA-256 AlgorithmIdentifier (OID 2.16.840.1.101.3.4.2.1 + NULL)
    let sha256_alg_id: &[u8] = &[
        0x30, 0x0D,                                             // SEQUENCE (13 bytes)
        0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,       // OID 2.16.840.1.101.3.4.2.1
        0x04, 0x02, 0x01,
        0x05, 0x00,                                             // NULL
    ];

    // CertID
    let cert_id = encode_sequence(&[
        sha256_alg_id,
        &encode_octet_string(&issuer_name_hash),
        &encode_octet_string(&issuer_key_hash),
        &encode_integer(serial),
    ]);

    // Request (just wraps CertID)
    let request = encode_sequence(&[&cert_id]);

    // requestList (SEQUENCE OF Request)
    let request_list = encode_sequence(&[&request]);

    // TBSRequest
    let tbs_request = encode_sequence(&[&request_list]);

    // OCSPRequest
    let ocsp_request = encode_sequence(&[&tbs_request]);

    Ok(ocsp_request)
}

/// Fetch an OCSP response from the responder and save to disk.
///
/// Returns the number of bytes in the response on success.
pub async fn fetch_ocsp_response(
    responder_url: &str,
    ocsp_request_der: &[u8],
    output_path: &Path,
) -> Result<usize> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .context("Failed to build HTTP client for OCSP")?;

    let resp = client
        .post(responder_url)
        .header("Content-Type", "application/ocsp-request")
        .body(ocsp_request_der.to_vec())
        .send()
        .await
        .with_context(|| format!("OCSP request to {} failed", responder_url))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "OCSP responder {} returned HTTP {}",
            responder_url,
            resp.status()
        );
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !content_type.contains("ocsp-response") {
        warn!(
            content_type = content_type,
            "OCSP responder returned unexpected content-type (expected application/ocsp-response)"
        );
    }

    let body = resp.bytes().await.context("Failed to read OCSP response body")?;

    if body.is_empty() {
        anyhow::bail!("OCSP responder returned empty response");
    }

    // Basic validation: OCSP response starts with SEQUENCE tag (0x30)
    if body[0] != 0x30 {
        anyhow::bail!(
            "OCSP response does not start with SEQUENCE tag (got 0x{:02X}), likely not valid DER",
            body[0]
        );
    }

    // Atomic write: write to temp file then rename
    let tmp_path = output_path.with_extension("tmp");
    std::fs::write(&tmp_path, &body)
        .with_context(|| format!("Failed to write OCSP response to {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, output_path)
        .with_context(|| format!("Failed to rename OCSP file to {}", output_path.display()))?;

    Ok(body.len())
}

/// Background task: periodically fetch OCSP staple and trigger TLS reload.
///
/// Runs forever. On each cycle:
/// 1. Extracts OCSP responder URL from the certificate
/// 2. Builds an OCSP request
/// 3. POSTs to the responder
/// 4. Saves the DER response to the configured path
/// 5. Triggers a TLS config hot-reload to pick up the new staple
pub async fn ocsp_refresh_task(
    tls_config: TlsConfig,
    shared_tls: SharedTlsConfig,
    status: SharedOcspStatus,
    refresh_interval: Duration,
) {
    // Initial delay — let TLS initialize first
    tokio::time::sleep(Duration::from_secs(5)).await;

    let ocsp_path = match tls_config.ocsp_staple {
        Some(ref p) => p.clone(),
        None => {
            // No OCSP file configured — generate a default path next to the cert
            tls_config.cert.with_extension("ocsp")
        }
    };

    loop {
        match do_ocsp_refresh(&tls_config.cert, &ocsp_path, &shared_tls, &tls_config, &status).await {
            Ok(bytes) => {
                info!(
                    bytes = bytes,
                    path = %ocsp_path.display(),
                    "OCSP staple refreshed successfully"
                );
            }
            Err(e) => {
                let msg = format!("{:#}", e);
                error!(err = %msg, "OCSP staple refresh failed");
                status.write().last_error = Some(msg);
            }
        }

        tokio::time::sleep(refresh_interval).await;
    }
}

async fn do_ocsp_refresh(
    cert_path: &Path,
    ocsp_path: &Path,
    shared_tls: &SharedTlsConfig,
    tls_config: &TlsConfig,
    status: &SharedOcspStatus,
) -> Result<usize> {
    // 1. Extract OCSP responder URL
    let url = extract_ocsp_responder_url(cert_path)?
        .ok_or_else(|| anyhow::anyhow!(
            "No OCSP responder URL in certificate AIA extension (cert: {})",
            cert_path.display()
        ))?;

    {
        let mut s = status.write();
        s.responder_url = Some(url.clone());
    }

    // 2. Build OCSP request
    let request_der = build_ocsp_request_der(cert_path)?;

    // 3. Fetch from responder
    let bytes = fetch_ocsp_response(&url, &request_der, ocsp_path).await?;

    // 4. Trigger TLS reload to pick up the new staple
    // Create a temporary TlsConfig with the OCSP path set
    let tls_with_ocsp = TlsConfig {
        ocsp_staple: Some(ocsp_path.to_path_buf()),
        ..tls_config.clone()
    };
    if let Err(e) = shared_tls.reload(&tls_with_ocsp) {
        warn!(err = %e, "TLS reload after OCSP refresh failed (staple saved but not active)");
    }

    // 5. Update status
    {
        let mut s = status.write();
        s.last_fetch = Some(std::time::Instant::now());
        s.response_bytes = bytes;
        s.refresh_count += 1;
        s.last_error = None;
    }

    Ok(bytes)
}

// ─── DER encoding helpers ────────────────────────────────────────────────

fn encode_length(len: usize) -> Vec<u8> {
    if len < 0x80 {
        vec![len as u8]
    } else if len <= 0xFF {
        vec![0x81, len as u8]
    } else if len <= 0xFFFF {
        vec![0x82, (len >> 8) as u8, len as u8]
    } else {
        vec![0x83, (len >> 16) as u8, (len >> 8) as u8, len as u8]
    }
}

fn encode_sequence(items: &[&[u8]]) -> Vec<u8> {
    let total_len: usize = items.iter().map(|i| i.len()).sum();
    let mut out = vec![0x30]; // SEQUENCE tag
    out.extend_from_slice(&encode_length(total_len));
    for item in items {
        out.extend_from_slice(item);
    }
    out
}

fn encode_octet_string(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x04]; // OCTET STRING tag
    out.extend_from_slice(&encode_length(data.len()));
    out.extend_from_slice(data);
    out
}

fn encode_integer(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x02]; // INTEGER tag
    // If high bit set, prepend 0x00 to indicate positive
    if !data.is_empty() && data[0] & 0x80 != 0 {
        out.extend_from_slice(&encode_length(data.len() + 1));
        out.push(0x00);
    } else {
        out.extend_from_slice(&encode_length(data.len()));
    }
    out.extend_from_slice(data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_length() {
        assert_eq!(encode_length(0), vec![0x00]);
        assert_eq!(encode_length(5), vec![0x05]);
        assert_eq!(encode_length(127), vec![0x7F]);
        assert_eq!(encode_length(128), vec![0x81, 0x80]);
        assert_eq!(encode_length(256), vec![0x82, 0x01, 0x00]);
    }

    #[test]
    fn test_encode_sequence() {
        let inner = [0x02, 0x01, 0x05]; // INTEGER 5
        let seq = encode_sequence(&[&inner]);
        assert_eq!(seq, vec![0x30, 0x03, 0x02, 0x01, 0x05]);
    }

    #[test]
    fn test_encode_octet_string() {
        let data = [0xAA, 0xBB];
        let enc = encode_octet_string(&data);
        assert_eq!(enc, vec![0x04, 0x02, 0xAA, 0xBB]);
    }

    #[test]
    fn test_encode_integer_positive() {
        // Serial number with high bit set needs leading 0x00
        let serial = [0x80, 0x01];
        let enc = encode_integer(&serial);
        assert_eq!(enc, vec![0x02, 0x03, 0x00, 0x80, 0x01]);

        // Normal serial without high bit
        let serial2 = [0x01, 0x02, 0x03];
        let enc2 = encode_integer(&serial2);
        assert_eq!(enc2, vec![0x02, 0x03, 0x01, 0x02, 0x03]);
    }
}
