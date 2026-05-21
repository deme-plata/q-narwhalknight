//! Framework-agnostic HTTP adapter for [`TipProofService`].
//!
//! [`TipProofService`]: crate::TipProofService
//!
//! The q-api-server uses axum; other deployments might use warp, actix,
//! hyper directly, or a custom router. Rather than couple this crate to
//! any one of them, the adapter returns a plain [`HttpResponse`] struct
//! the caller wraps into their framework's response type:
//!
//! ```ignore
//! // axum handler
//! async fn get_tip(State(svc): State<TipProofService>) -> impl IntoResponse {
//!     let resp = tip_proof_http::handle_get_tip_bytes(&svc);
//!     let mut builder = axum::response::Response::builder().status(resp.status);
//!     for (k, v) in &resp.headers {
//!         builder = builder.header(k, v);
//!     }
//!     builder.body(axum::body::Body::from(resp.body)).unwrap()
//! }
//! ```
//!
//! # Endpoint shapes (suggested)
//!
//! | Path                | Body                                  | Returned by                |
//! |---------------------|---------------------------------------|----------------------------|
//! | `GET /tip`          | bincode'd `LatticeTipProofV2`         | [`handle_get_tip_bytes`]   |
//! | `GET /tip/json`     | JSON envelope w/ tip metadata         | [`handle_get_tip_json`]    |
//! | `GET /tip/stats`    | JSON `TipProofServiceStats`           | [`handle_get_tip_stats`]   |
//! | `GET /tip/health`   | JSON liveness object                  | [`handle_get_tip_health`]  |
//!
//! # Caching
//!
//! Every body-bearing response includes an `ETag` header computed as
//! `blake3(body)`. Clients should pass `If-None-Match: <etag>` on
//! repeat requests; [`if_none_match_matches`] returns true when the
//! caller should respond with 304 Not Modified.
//!
//! # Content negotiation
//!
//! `/tip` always returns `application/octet-stream` (the bincode wire
//! format). `/tip/json` returns `application/json` with the proof's
//! header fields + step count (NOT the full step proofs — those are
//! large and binary). Clients that want both call both endpoints.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tip_proof_service::{TipProofService, TipProofServiceStats};
use crate::tip_proof_v2::PROOF_VERSION;

// ════════════════════════════════════════════════════════════════════════════
// Response shape
// ════════════════════════════════════════════════════════════════════════════

/// Framework-agnostic HTTP response envelope. Map onto your framework's
/// response builder at the call site.
#[derive(Clone, Debug)]
pub struct HttpResponse {
    /// HTTP status code as a `u16`. Stays primitive so this crate doesn't
    /// pull in `http::StatusCode`.
    pub status: u16,
    /// Header name → value. `BTreeMap` for deterministic header order
    /// (matters for snapshot tests + ETag stability).
    pub headers: BTreeMap<String, String>,
    /// Response body bytes. Empty for 204/304.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Construct an `application/octet-stream` response with an `ETag`
    /// header computed from the body.
    pub fn octet_stream(body: Vec<u8>) -> Self {
        let etag = compute_etag(&body);
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".to_string(), "application/octet-stream".to_string());
        headers.insert("Content-Length".to_string(), body.len().to_string());
        headers.insert("ETag".to_string(), etag);
        headers.insert("X-Proof-Version".to_string(), PROOF_VERSION.to_string());
        Self {
            status: 200,
            headers,
            body,
        }
    }

    /// Construct an `application/json` response with an `ETag` header.
    pub fn json(body: Vec<u8>) -> Self {
        let etag = compute_etag(&body);
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Content-Length".to_string(), body.len().to_string());
        headers.insert("ETag".to_string(), etag);
        headers.insert("X-Proof-Version".to_string(), PROOF_VERSION.to_string());
        Self {
            status: 200,
            headers,
            body,
        }
    }

    /// `304 Not Modified` — body empty, ETag included so the client
    /// can confirm match.
    pub fn not_modified(etag: String) -> Self {
        let mut headers = BTreeMap::new();
        headers.insert("ETag".to_string(), etag);
        Self {
            status: 304,
            headers,
            body: Vec::new(),
        }
    }

    /// `500 Internal Server Error` with a JSON error body. Used when
    /// the service surfaces an unexpected error (serialization failure,
    /// poisoned lock, etc.) — these should be rare in steady state.
    pub fn server_error(reason: &str) -> Self {
        let body = serde_json::to_vec(&JsonError {
            error: reason.to_string(),
        })
        .unwrap_or_else(|_| br#"{"error":"unknown"}"#.to_vec());
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("Content-Length".to_string(), body.len().to_string());
        Self {
            status: 500,
            headers,
            body,
        }
    }
}

/// Compute the ETag string for a body. Format: `W/"<hex>"` (weak ETag)
/// using the first 16 hex chars of BLAKE3(body). Weak because two
/// semantically-equivalent serializations of the same proof may differ
/// at the byte level (none today, but reserves the option).
pub fn compute_etag(body: &[u8]) -> String {
    let h = blake3::hash(body);
    let hex_full = hex::encode(h.as_bytes());
    format!("W/\"{}\"", &hex_full[..16])
}

/// Check whether the caller's `If-None-Match` value matches the body's
/// computed ETag. If true, respond with 304 instead of re-serving the
/// full body. Accepts the value verbatim — the caller is responsible
/// for extracting the header from their framework.
pub fn if_none_match_matches(if_none_match: Option<&str>, etag: &str) -> bool {
    match if_none_match {
        Some(value) => value.trim() == etag,
        None => false,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// JSON envelopes
// ════════════════════════════════════════════════════════════════════════════

/// `/tip/json` body — header metadata + step count. Excludes the
/// full step proofs (large + binary).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TipProofJson {
    pub anchor_height: u64,
    pub anchor_state_hex: String,
    pub tip_height: u64,
    pub tip_state_hex: String,
    pub step_count: usize,
    pub version: String,
    /// `Content-Length` the octet-stream body would have. Helps the
    /// caller decide whether to fetch the full proof.
    pub octet_body_size_estimate: usize,
}

/// `/tip/health` body — liveness check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TipProofHealthJson {
    pub status: &'static str,
    pub proof_version: &'static str,
    pub tip_height: u64,
    pub step_count: usize,
    pub anchor_height: u64,
    pub persistence_backend: Option<String>,
}

#[derive(Serialize)]
struct JsonError {
    error: String,
}

// ════════════════════════════════════════════════════════════════════════════
// Handlers
// ════════════════════════════════════════════════════════════════════════════

/// `GET /tip` — bincode'd `LatticeTipProofV2`.
///
/// If `if_none_match` matches the computed ETag, returns 304 with the
/// ETag echoed back. Otherwise returns 200 with body + ETag header.
pub fn handle_get_tip_bytes(
    service: &TipProofService,
    if_none_match: Option<&str>,
) -> HttpResponse {
    let bytes = match service.current_proof_bytes() {
        Ok(b) => b,
        Err(e) => {
            return HttpResponse::server_error(&format!("serialise: {e}"));
        }
    };
    let etag = compute_etag(&bytes);
    if if_none_match_matches(if_none_match, &etag) {
        return HttpResponse::not_modified(etag);
    }
    HttpResponse::octet_stream(bytes)
}

/// `GET /tip/json` — header metadata as JSON.
pub fn handle_get_tip_json(
    service: &TipProofService,
    if_none_match: Option<&str>,
) -> HttpResponse {
    let proof = service.current_proof();
    let envelope = TipProofJson {
        anchor_height: proof.anchor_height,
        anchor_state_hex: hex::encode(proof.anchor_state),
        tip_height: proof.tip_height,
        tip_state_hex: hex::encode(proof.tip_state),
        step_count: proof.delta_proofs.len(),
        version: proof.version.clone(),
        octet_body_size_estimate: service.current_proof_size_estimate(),
    };
    let body = match serde_json::to_vec(&envelope) {
        Ok(b) => b,
        Err(e) => return HttpResponse::server_error(&format!("json serialise: {e}")),
    };
    let etag = compute_etag(&body);
    if if_none_match_matches(if_none_match, &etag) {
        return HttpResponse::not_modified(etag);
    }
    HttpResponse::json(body)
}

/// `GET /tip/stats` — service stats as JSON.
pub fn handle_get_tip_stats(service: &TipProofService) -> HttpResponse {
    let stats: TipProofServiceStats = service.stats();
    let body = match serde_json::to_vec(&stats) {
        Ok(b) => b,
        Err(e) => return HttpResponse::server_error(&format!("json serialise: {e}")),
    };
    HttpResponse::json(body)
}

/// `GET /tip/health` — liveness probe. Always 200 if the service is
/// constructed (the act of calling already proves Send+Sync + lock
/// liveness). Body carries the headline metrics for a Kubernetes
/// liveness/readiness probe.
pub fn handle_get_tip_health(service: &TipProofService) -> HttpResponse {
    let (anchor_height, _) = service.anchor();
    let envelope = TipProofHealthJson {
        status: "ok",
        proof_version: service.proof_version(),
        tip_height: service.tip_height(),
        step_count: service.step_count(),
        anchor_height,
        persistence_backend: service.persistence_backend_id(),
    };
    let body = match serde_json::to_vec(&envelope) {
        Ok(b) => b,
        Err(e) => return HttpResponse::server_error(&format!("json serialise: {e}")),
    };
    HttpResponse::json(body)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tip_proof_service::TipProofServiceConfig;
    use crate::{tip_proof_v2, TipProofService};
    use q_ivc::recursion::{LatticeStepProof, StepIO};
    use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};

    fn dummy_lattice_proof() -> LatticeGuardProof {
        LatticeGuardProof {
            commitments: Vec::new(),
            evaluations: (0, 0, 0),
            product_proofs: Vec::new(),
            transcript_state: [0u8; 32],
            metadata: ProofMetadata {
                num_constraints: 0,
                num_public_inputs: 0,
                security_level: SecurityLevel::PQ128,
                generation_time_ms: 0,
            },
        }
    }

    fn step(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
        LatticeStepProof {
            proof: dummy_lattice_proof(),
            z_in: z_in.pack(),
            z_out: z_out.pack(),
            public_input_count: 9,
        }
    }

    fn root(seed: u8) -> [u8; 32] {
        let mut r = [0u8; 32];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
        }
        r
    }

    fn produce(service: &TipProofService, n: u64) {
        let mut prev = [0u8; 32];
        for h in 0u64..n {
            let next = root((h + 1) as u8);
            service
                .extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
                .expect("honest extend");
            prev = next;
        }
    }

    #[test]
    fn etag_is_deterministic_and_weak() {
        let body = b"hello world";
        let etag1 = compute_etag(body);
        let etag2 = compute_etag(body);
        assert_eq!(etag1, etag2);
        assert!(etag1.starts_with("W/\""));
        assert!(etag1.ends_with('"'));
    }

    #[test]
    fn etag_changes_when_body_changes() {
        let a = compute_etag(b"hello");
        let b = compute_etag(b"hellp");
        assert_ne!(a, b);
    }

    #[test]
    fn if_none_match_matches_exact_etag() {
        let etag = compute_etag(b"hello");
        assert!(if_none_match_matches(Some(&etag), &etag));
        assert!(if_none_match_matches(Some(&format!("  {etag}  ")), &etag)); // trim
        assert!(!if_none_match_matches(Some("W/\"different\""), &etag));
        assert!(!if_none_match_matches(None, &etag));
    }

    #[test]
    fn get_tip_bytes_returns_octet_stream() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let resp = handle_get_tip_bytes(&service, None);
        assert_eq!(resp.status, 200);
        assert_eq!(
            resp.headers.get("Content-Type").map(String::as_str),
            Some("application/octet-stream")
        );
        assert!(resp.headers.contains_key("ETag"));
        assert!(resp.headers.contains_key("Content-Length"));
        assert_eq!(
            resp.headers.get("X-Proof-Version").map(String::as_str),
            Some("latticeguard-rlwe-v1")
        );
        assert!(!resp.body.is_empty());

        // Body round-trips through bincode.
        let proof: tip_proof_v2::LatticeTipProofV2 =
            bincode::deserialize(&resp.body).expect("valid bincode");
        assert_eq!(proof.tip_height, 0);
    }

    #[test]
    fn get_tip_bytes_304_when_if_none_match_matches() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let resp = handle_get_tip_bytes(&service, None);
        let etag = resp.headers.get("ETag").cloned().unwrap();

        let cached = handle_get_tip_bytes(&service, Some(&etag));
        assert_eq!(cached.status, 304);
        assert!(cached.body.is_empty());
        assert_eq!(cached.headers.get("ETag").cloned().unwrap(), etag);
    }

    #[test]
    fn get_tip_bytes_returns_new_etag_after_extend() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let etag_before = handle_get_tip_bytes(&service, None)
            .headers
            .get("ETag")
            .cloned()
            .unwrap();

        produce(&service, 1);
        let etag_after = handle_get_tip_bytes(&service, None)
            .headers
            .get("ETag")
            .cloned()
            .unwrap();

        assert_ne!(etag_before, etag_after, "extend must invalidate ETag");
    }

    #[test]
    fn get_tip_json_returns_envelope() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        produce(&service, 3);

        let resp = handle_get_tip_json(&service, None);
        assert_eq!(resp.status, 200);
        assert_eq!(
            resp.headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );

        let envelope: TipProofJson = serde_json::from_slice(&resp.body).expect("json");
        assert_eq!(envelope.tip_height, 3);
        assert_eq!(envelope.anchor_height, 0);
        assert_eq!(envelope.step_count, 3);
        assert_eq!(envelope.version, "latticeguard-rlwe-v1");
        assert_eq!(envelope.anchor_state_hex.len(), 64); // 32 bytes hex
        assert!(envelope.octet_body_size_estimate > 0);
    }

    #[test]
    fn get_tip_json_supports_etag_caching() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        let resp = handle_get_tip_json(&service, None);
        let etag = resp.headers.get("ETag").cloned().unwrap();

        let cached = handle_get_tip_json(&service, Some(&etag));
        assert_eq!(cached.status, 304);
    }

    #[test]
    fn get_tip_stats_returns_service_stats_json() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        produce(&service, 2);

        let resp = handle_get_tip_stats(&service);
        assert_eq!(resp.status, 200);
        let stats: TipProofServiceStats =
            serde_json::from_slice(&resp.body).expect("json");
        assert_eq!(stats.total_extends_succeeded, 2);
        assert_eq!(stats.current_tip_height, 2);
    }

    #[test]
    fn get_tip_health_returns_liveness_object() {
        let service = TipProofService::new(TipProofServiceConfig::genesis());
        produce(&service, 5);

        let resp = handle_get_tip_health(&service);
        assert_eq!(resp.status, 200);
        let health: TipProofHealthJson =
            serde_json::from_slice(&resp.body).expect("json");
        assert_eq!(health.status, "ok");
        assert_eq!(health.tip_height, 5);
        assert_eq!(health.step_count, 5);
        assert_eq!(health.anchor_height, 0);
        assert_eq!(health.proof_version, "latticeguard-rlwe-v1");
        assert!(
            health.persistence_backend.is_none(),
            "no persistence configured on this test service"
        );
    }

    #[test]
    fn server_error_returns_500_with_json_body() {
        let resp = HttpResponse::server_error("test failure");
        assert_eq!(resp.status, 500);
        assert_eq!(
            resp.headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
        let body: JsonErrorOut =
            serde_json::from_slice(&resp.body).expect("json");
        assert_eq!(body.error, "test failure");
    }

    #[test]
    fn etag_format_is_w_quote_hex() {
        let body = b"some bytes";
        let etag = compute_etag(body);
        // Format: W/"<16 hex chars>"
        assert!(etag.starts_with("W/\""));
        assert_eq!(etag.len(), 4 + 16 + 1); // 'W/"' + 16 hex + closing '"'
        assert!(etag.ends_with('"'));
        // Middle 16 chars are valid hex.
        let middle = &etag[3..3 + 16];
        assert!(middle.chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// Local copy of JsonError for round-trip testing — the production
    /// type is private to the module.
    #[derive(Deserialize)]
    struct JsonErrorOut {
        error: String,
    }
}
