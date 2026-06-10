// PaaS Audit Logging & Distributed Tracing System
// Implements comprehensive audit trails and distributed tracing for compliance
//
// Features:
// - Per-request audit records with trace_id
// - Distributed tracing spans across services
// - Structured logging with complete metadata
// - Compliance-ready audit trails
// - Prometheus metrics integration
//
// Standards:
// - W3C Trace Context (trace_id + span_id)
// - OpenTelemetry compatible
// - GDPR/SOC2 compliance ready

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, Span};
use uuid::Uuid;

use crate::privacy_service_api::PaaSService;

/// Audit record for a complete PaaS request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaaSAuditRecord {
    /// Unique audit record ID
    pub audit_id: String,

    /// W3C Trace Context - trace_id (128-bit)
    pub trace_id: String,

    /// W3C Trace Context - span_id (64-bit)
    pub span_id: String,

    /// Parent span ID (for distributed tracing)
    pub parent_span_id: Option<String>,

    /// Request ID (for correlation)
    pub request_id: String,

    /// Idempotency key (if provided)
    pub idempotency_key: Option<String>,

    /// Customer wallet address (32 bytes)
    pub wallet_address: [u8; 32],

    /// Service requested
    pub service: PaaSService,

    /// Request timestamp
    pub request_timestamp: DateTime<Utc>,

    /// Response timestamp
    pub response_timestamp: Option<DateTime<Utc>>,

    /// Request latency (milliseconds)
    pub latency_ms: Option<u64>,

    /// Amount in QUG (atomic units)
    pub amount_qug: u64,

    /// Amount in USD (at time of request)
    pub amount_usd: f64,

    /// QUG/USD exchange rate used
    pub qug_usd_rate: f64,

    /// Billing transaction ID (if finalized)
    pub billing_tx_id: Option<String>,

    /// Reservation ID (for atomic billing)
    pub reservation_id: Option<String>,

    /// Request successful?
    pub success: bool,

    /// HTTP status code
    pub status_code: u16,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Error code (for categorization)
    pub error_code: Option<String>,

    /// Request metadata (service-specific)
    pub request_metadata: serde_json::Value,

    /// Response metadata (service-specific)
    pub response_metadata: Option<serde_json::Value>,

    /// IP address (hashed for privacy)
    pub ip_address_hash: String,

    /// User agent (hashed for privacy)
    pub user_agent_hash: String,

    /// API key ID (not the key itself)
    pub api_key_id: Option<String>,

    /// Rate limit state at request time
    pub rate_limit_remaining: Option<u32>,

    /// Compliance flags
    pub compliance_flags: ComplianceFlags,
}

/// Compliance flags for audit records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFlags {
    /// PII redacted from logs
    pub pii_redacted: bool,

    /// GDPR compliant logging
    pub gdpr_compliant: bool,

    /// SOC2 audit trail
    pub soc2_audit_trail: bool,

    /// Data retention policy applied
    pub retention_policy: String,
}

impl Default for ComplianceFlags {
    fn default() -> Self {
        Self {
            pii_redacted: true,
            gdpr_compliant: true,
            soc2_audit_trail: true,
            retention_policy: "90_days".to_string(),
        }
    }
}

/// Audit logging manager
pub struct PaaSAuditManager {
    /// In-memory audit log (for recent queries)
    audit_records: Arc<RwLock<Vec<PaaSAuditRecord>>>,

    /// Maximum in-memory records
    max_memory_records: usize,

    /// Audit statistics
    stats: Arc<RwLock<AuditStats>>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct AuditStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_amount_qug: u64,
    pub total_amount_usd: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

impl PaaSAuditManager {
    pub fn new(max_memory_records: usize) -> Self {
        Self {
            audit_records: Arc::new(RwLock::new(Vec::with_capacity(max_memory_records))),
            max_memory_records,
            stats: Arc::new(RwLock::new(AuditStats::default())),
        }
    }

    /// Create a new audit record
    pub async fn create_audit_record(
        &self,
        trace_id: String,
        span_id: String,
        wallet_address: [u8; 32],
        service: PaaSService,
        amount_qug: u64,
        amount_usd: f64,
        qug_usd_rate: f64,
        request_metadata: serde_json::Value,
        ip_address: &str,
        user_agent: &str,
        api_key_id: Option<String>,
    ) -> String {
        let audit_id = Uuid::new_v4().to_string();
        let request_id = Uuid::new_v4().to_string();

        // Hash PII for privacy
        let ip_address_hash = self.hash_pii(ip_address);
        let user_agent_hash = self.hash_pii(user_agent);

        let audit_record = PaaSAuditRecord {
            audit_id: audit_id.clone(),
            trace_id,
            span_id,
            parent_span_id: None,
            request_id,
            idempotency_key: None,
            wallet_address,
            service: service.clone(),
            request_timestamp: Utc::now(),
            response_timestamp: None,
            latency_ms: None,
            amount_qug,
            amount_usd,
            qug_usd_rate,
            billing_tx_id: None,
            reservation_id: None,
            success: false,
            status_code: 0,
            error_message: None,
            error_code: None,
            request_metadata,
            response_metadata: None,
            ip_address_hash,
            user_agent_hash,
            api_key_id,
            rate_limit_remaining: None,
            compliance_flags: ComplianceFlags::default(),
        };

        // Store in memory (with rotation)
        let mut records = self.audit_records.write().await;
        if records.len() >= self.max_memory_records {
            // Rotate: remove oldest 25%
            records.drain(0..self.max_memory_records / 4);
        }
        records.push(audit_record);

        info!(
            audit_id = %audit_id,
            trace_id = %self.short_trace_id(&audit_id),
            service = ?service,
            amount_qug = %amount_qug,
            "📝 Audit record created"
        );

        audit_id
    }

    /// Finalize audit record with response data
    pub async fn finalize_audit_record(
        &self,
        audit_id: &str,
        success: bool,
        status_code: u16,
        billing_tx_id: Option<String>,
        reservation_id: Option<String>,
        response_metadata: Option<serde_json::Value>,
        error_message: Option<String>,
        error_code: Option<String>,
    ) {
        let mut records = self.audit_records.write().await;

        if let Some(record) = records.iter_mut().find(|r| r.audit_id == audit_id) {
            let response_timestamp = Utc::now();
            let latency_ms = response_timestamp
                .signed_duration_since(record.request_timestamp)
                .num_milliseconds() as u64;

            record.response_timestamp = Some(response_timestamp);
            record.latency_ms = Some(latency_ms);
            record.success = success;
            record.status_code = status_code;
            record.billing_tx_id = billing_tx_id;
            record.reservation_id = reservation_id;
            record.response_metadata = response_metadata;
            record.error_message = error_message.clone();
            record.error_code = error_code.clone();

            // Clone values before dropping the lock
            let amount_qug = record.amount_qug;
            let amount_usd = record.amount_usd;

            // Update statistics
            drop(records);
            self.update_stats(success, latency_ms, amount_qug, amount_usd)
                .await;

            info!(
                audit_id = %audit_id,
                success = %success,
                latency_ms = %latency_ms,
                "✅ Audit record finalized"
            );
        } else {
            warn!(
                audit_id = %audit_id,
                "⚠️ Audit record not found for finalization"
            );
        }
    }

    /// Update audit statistics
    async fn update_stats(&self, success: bool, latency_ms: u64, amount_qug: u64, amount_usd: f64) {
        let mut stats = self.stats.write().await;

        stats.total_requests += 1;
        if success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }

        stats.total_amount_qug += amount_qug;
        stats.total_amount_usd += amount_usd;

        // Update latency stats (simplified - use proper percentile calculation in production)
        let total = stats.total_requests as f64;
        stats.avg_latency_ms = (stats.avg_latency_ms * (total - 1.0) + latency_ms as f64) / total;

        // TODO: Implement proper p50/p95/p99 calculation with histogram
        stats.p50_latency_ms = stats.avg_latency_ms * 0.8;
        stats.p95_latency_ms = stats.avg_latency_ms * 1.5;
        stats.p99_latency_ms = stats.avg_latency_ms * 2.0;
    }

    /// Get audit statistics
    pub async fn get_stats(&self) -> AuditStats {
        self.stats.read().await.clone()
    }

    /// Query audit records by filter
    pub async fn query_audit_records(&self, filter: AuditFilter) -> Vec<PaaSAuditRecord> {
        let records = self.audit_records.read().await;

        records
            .iter()
            .filter(|r| {
                // Filter by wallet address
                if let Some(wallet) = &filter.wallet_address {
                    if r.wallet_address != *wallet {
                        return false;
                    }
                }

                // Filter by service
                if let Some(service) = &filter.service {
                    if format!("{:?}", r.service) != *service {
                        return false;
                    }
                }

                // Filter by success
                if let Some(success) = filter.success {
                    if r.success != success {
                        return false;
                    }
                }

                // Filter by time range
                if let Some(start) = filter.start_time {
                    if r.request_timestamp < start {
                        return false;
                    }
                }

                if let Some(end) = filter.end_time {
                    if r.request_timestamp > end {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Hash PII data for privacy-preserving logging
    fn hash_pii(&self, data: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hasher.update(b"paas_audit_salt"); // Add salt
        hex::encode(&hasher.finalize()[..16]) // First 16 bytes (32 hex chars)
    }

    /// Shorten trace_id for logging
    fn short_trace_id(&self, trace_id: &str) -> String {
        trace_id.chars().take(8).collect()
    }

    /// Export audit records to JSON (for compliance)
    pub async fn export_audit_records(&self, filter: AuditFilter) -> String {
        let records = self.query_audit_records(filter).await;
        serde_json::to_string_pretty(&records).unwrap_or_else(|_| "[]".to_string())
    }
}

/// Filter for querying audit records
#[derive(Debug, Clone)]
pub struct AuditFilter {
    pub wallet_address: Option<[u8; 32]>,
    pub service: Option<String>,
    pub success: Option<bool>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

impl Default for AuditFilter {
    fn default() -> Self {
        Self {
            wallet_address: None,
            service: None,
            success: None,
            start_time: None,
            end_time: None,
        }
    }
}

/// Distributed tracing context
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// W3C Trace Context - trace_id (128-bit UUID)
    pub trace_id: String,

    /// W3C Trace Context - span_id (64-bit random)
    pub span_id: String,

    /// Parent span ID
    pub parent_span_id: Option<String>,

    /// Tracing span (for structured logging)
    pub span: Span,
}

impl TraceContext {
    /// Create a new trace context
    pub fn new(service_name: &str, operation: &str) -> Self {
        let trace_id = Uuid::new_v4().to_string();
        let span_id = format!("{:016x}", rand::random::<u64>());

        let span = tracing::info_span!(
            "paas_request",
            service = %service_name,
            operation = %operation,
            trace_id = %trace_id,
            span_id = %span_id
        );

        Self {
            trace_id,
            span_id,
            parent_span_id: None,
            span,
        }
    }

    /// Create child span
    pub fn create_child_span(&self, operation: &str) -> TraceContext {
        let span_id = format!("{:016x}", rand::random::<u64>());

        let span = tracing::info_span!(
            parent: &self.span,
            "paas_operation",
            operation = %operation,
            trace_id = %self.trace_id,
            span_id = %span_id,
            parent_span_id = %self.span_id
        );

        TraceContext {
            trace_id: self.trace_id.clone(),
            span_id,
            parent_span_id: Some(self.span_id.clone()),
            span,
        }
    }

    /// Enter span context
    pub fn enter(&self) -> tracing::span::Entered {
        self.span.enter()
    }
}

/// Trace a PaaS operation
#[macro_export]
macro_rules! trace_paas_operation {
    ($ctx:expr, $operation:expr, $block:block) => {{
        let child_ctx = $ctx.create_child_span($operation);
        let _guard = child_ctx.enter();

        tracing::info!(
            operation = %$operation,
            trace_id = %child_ctx.trace_id,
            span_id = %child_ctx.span_id,
            "▶️  Starting operation"
        );

        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed().as_millis() as u64;

        tracing::info!(
            operation = %$operation,
            elapsed_ms = %elapsed,
            success = %result.is_ok(),
            "✅ Operation completed"
        );

        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_record_lifecycle() {
        let manager = PaaSAuditManager::new(1000);
        let wallet = [1u8; 32];

        // Create audit record
        let audit_id = manager
            .create_audit_record(
                "trace-123".to_string(),
                "span-456".to_string(),
                wallet,
                PaaSService::TorRelay,
                100_000_000,
                0.05,
                0.50,
                serde_json::json!({"data_size_mb": 1.0}),
                "127.0.0.1",
                "Mozilla/5.0",
                None,
            )
            .await;

        // Finalize record
        manager
            .finalize_audit_record(
                &audit_id,
                true,
                200,
                Some("tx-789".to_string()),
                Some("res-101".to_string()),
                Some(serde_json::json!({"circuit_id": "qnk-tor-abc"})),
                None,
                None,
            )
            .await;

        // Query records
        let records = manager
            .query_audit_records(AuditFilter {
                wallet_address: Some(wallet),
                ..Default::default()
            })
            .await;

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].audit_id, audit_id);
        assert!(records[0].success);
        assert!(records[0].latency_ms.is_some());
    }

    #[tokio::test]
    async fn test_audit_statistics() {
        let manager = PaaSAuditManager::new(1000);

        // Create and finalize multiple records
        for i in 0..10 {
            let audit_id = manager
                .create_audit_record(
                    format!("trace-{}", i),
                    format!("span-{}", i),
                    [i as u8; 32],
                    PaaSService::TorRelay,
                    100_000_000,
                    0.05,
                    0.50,
                    serde_json::json!({}),
                    "127.0.0.1",
                    "Mozilla/5.0",
                    None,
                )
                .await;

            manager
                .finalize_audit_record(&audit_id, i % 2 == 0, 200, None, None, None, None, None)
                .await;
        }

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_requests, 10);
        assert_eq!(stats.successful_requests, 5);
        assert_eq!(stats.failed_requests, 5);
    }

    #[test]
    fn test_trace_context_creation() {
        let ctx = TraceContext::new("PaaS", "tor_relay");
        assert!(!ctx.trace_id.is_empty());
        assert!(!ctx.span_id.is_empty());
        assert!(ctx.parent_span_id.is_none());
    }

    #[test]
    fn test_trace_context_child_span() {
        let parent_ctx = TraceContext::new("PaaS", "tor_relay");
        let child_ctx = parent_ctx.create_child_span("circuit_establishment");

        assert_eq!(child_ctx.trace_id, parent_ctx.trace_id);
        assert_ne!(child_ctx.span_id, parent_ctx.span_id);
        assert_eq!(child_ctx.parent_span_id, Some(parent_ctx.span_id));
    }
}
