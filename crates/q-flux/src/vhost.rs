//! Virtual host routing — SNI-based TLS cert selection + per-domain backend/static overrides.

use std::collections::HashMap;
use std::io::BufReader;
use std::sync::Arc;
use anyhow::Result;

use crate::config::{TlsConfig, VhostConfig};

/// A VhostRouter maps domain names → TLS certified keys.
/// Implements `rustls::server::ResolvesServerCert` so rustls picks the right
/// certificate during the TLS handshake based on the client's SNI extension.
///
/// If the SNI name doesn't match any configured vhost, the default (main) cert
/// is returned so the connection still succeeds.
#[derive(Debug)]
pub struct VhostRouter {
    /// domain (lowercase) → certified key
    cert_map: HashMap<String, Arc<rustls::sign::CertifiedKey>>,
    /// Default cert (main [tls] section)
    default_cert: Arc<rustls::sign::CertifiedKey>,
}

impl rustls::server::ResolvesServerCert for VhostRouter {
    fn resolve(
        &self,
        hello: rustls::server::ClientHello<'_>,
    ) -> Option<Arc<rustls::sign::CertifiedKey>> {
        if let Some(sni) = hello.server_name() {
            let key = sni.to_lowercase();
            if let Some(cert) = self.cert_map.get(&key) {
                tracing::trace!(sni = %sni, "SNI: vhost cert selected");
                return Some(cert.clone());
            }
        }
        // Fall back to default cert — connection still works, just with main cert
        Some(self.default_cert.clone())
    }
}

impl VhostRouter {
    /// Build a VhostRouter from the default TLS config and a list of vhost configs.
    /// Loads all certs/keys from disk. Call this once at startup.
    pub fn from_config(default_tls: &TlsConfig, vhosts: &[VhostConfig]) -> Result<Arc<Self>> {
        let default_cert = Arc::new(load_certified_key(&default_tls.cert, &default_tls.key)?);

        let mut cert_map = HashMap::new();
        for vhost in vhosts {
            if vhost.domains.is_empty() {
                continue;
            }
            let cert_key = Arc::new(load_certified_key(&vhost.cert, &vhost.key)?);
            for domain in &vhost.domains {
                cert_map.insert(domain.to_lowercase(), cert_key.clone());
                tracing::info!(domain = %domain, cert = %vhost.cert.display(), "Vhost cert loaded");
            }
        }

        tracing::info!(
            vhosts = vhosts.len(),
            "VhostRouter: {} vhost(s) + default cert loaded",
            vhosts.len(),
        );

        Ok(Arc::new(Self { cert_map, default_cert }))
    }
}

/// Load a PEM cert chain + private key and produce a rustls CertifiedKey.
pub fn load_certified_key(
    cert_path: &std::path::Path,
    key_path: &std::path::Path,
) -> Result<rustls::sign::CertifiedKey> {
    // Load certificate chain
    let cert_file = std::fs::File::open(cert_path)
        .map_err(|e| anyhow::anyhow!("Cannot open cert {}: {}", cert_path.display(), e))?;
    let certs: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(cert_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to parse certs in {}: {}", cert_path.display(), e))?;
    if certs.is_empty() {
        anyhow::bail!("No certificates found in {}", cert_path.display());
    }

    // Load private key
    let key_file = std::fs::File::open(key_path)
        .map_err(|e| anyhow::anyhow!("Cannot open key {}: {}", key_path.display(), e))?;
    let key = rustls_pemfile::private_key(&mut BufReader::new(key_file))
        .map_err(|e| anyhow::anyhow!("Failed to parse key in {}: {}", key_path.display(), e))?
        .ok_or_else(|| anyhow::anyhow!("No private key found in {}", key_path.display()))?;

    // Build signing key
    let signing_key = rustls::crypto::ring::sign::any_supported_type(&key)
        .map_err(|e| anyhow::anyhow!("Unsupported key type in {}: {:?}", key_path.display(), e))?;

    Ok(rustls::sign::CertifiedKey::new(certs, signing_key))
}
