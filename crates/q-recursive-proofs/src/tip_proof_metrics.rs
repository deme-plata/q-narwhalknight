//! Pluggable metrics sink for [`TipProofService`] and
//! [`TipProofClient`] stats.
//!
//! [`TipProofService`]: crate::TipProofService
//! [`TipProofClient`]: crate::TipProofClient
//!
//! Goals:
//!
//! - Don't depend on `prometheus` / `metrics` / any specific framework.
//!   The trait is small enough that callers wire their own adapter in
//!   ~10 LOC.
//! - Easy to test — [`VecSink`] collects observations into memory so
//!   tests can assert on what was emitted.
//! - Easy to deploy nothing — [`NopSink`] makes "metrics off" the
//!   default.
//!
//! # Wiring
//!
//! ```ignore
//! use q_recursive_proofs::{TipProofService, MetricsSink, StdoutSink};
//!
//! let svc = TipProofService::new(...);
//! let sink = Arc::new(StdoutSink::new("tip_proof.service"));
//!
//! // Pump on a tokio interval:
//! tokio::spawn(async move {
//!     let mut tick = tokio::time::interval(Duration::from_secs(15));
//!     loop {
//!         tick.tick().await;
//!         pump_service_stats(&svc, sink.as_ref());
//!     }
//! });
//! ```
//!
//! # Metric naming
//!
//! Names are dotted paths with `_total` suffix for counters and
//! `_current` for gauges — compatible with Prometheus naming when the
//! sink converts the dot to underscore.

use std::sync::Mutex;

use tracing::debug;

use crate::tip_proof_client::TipProofClientStats;
use crate::tip_proof_service::TipProofServiceStats;

// ════════════════════════════════════════════════════════════════════════════
// Trait
// ════════════════════════════════════════════════════════════════════════════

/// Sink for metric observations. Implementors are responsible for
/// thread safety; both counters and gauges may be observed concurrently
/// from many threads.
///
/// Implementor contract:
///
/// - `counter`: a monotonically increasing value. Sinks that map to
///   Prometheus should treat this as a `_total` counter.
/// - `gauge`: an instantaneous value that can go up or down. Sinks
///   should overwrite the previous observation.
pub trait MetricsSink: Send + Sync + std::fmt::Debug {
    /// Observe a counter value at the given metric name.
    fn counter(&self, name: &str, value: u64);

    /// Observe a gauge value at the given metric name.
    fn gauge(&self, name: &str, value: u64);

    /// Human-readable sink identifier for logs / diagnostics.
    fn sink_id(&self) -> &str;
}

// ════════════════════════════════════════════════════════════════════════════
// NopSink
// ════════════════════════════════════════════════════════════════════════════

/// Discard all observations. Default sink for callers that don't care
/// about metrics.
#[derive(Clone, Copy, Debug, Default)]
pub struct NopSink;

impl MetricsSink for NopSink {
    fn counter(&self, _name: &str, _value: u64) {}
    fn gauge(&self, _name: &str, _value: u64) {}
    fn sink_id(&self) -> &str {
        "nop"
    }
}

// ════════════════════════════════════════════════════════════════════════════
// StdoutSink
// ════════════════════════════════════════════════════════════════════════════

/// Log observations through the `tracing` framework at DEBUG level.
/// Suitable for local dev + small deployments where Prometheus would
/// be overkill.
#[derive(Clone, Debug)]
pub struct StdoutSink {
    prefix: String,
}

impl StdoutSink {
    /// `prefix` is prepended to every metric name (e.g. "tip_proof").
    /// Empty string is allowed.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    fn full_name<'a>(&self, name: &'a str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }
}

impl MetricsSink for StdoutSink {
    fn counter(&self, name: &str, value: u64) {
        debug!(metric_kind = "counter", metric = %self.full_name(name), value);
    }
    fn gauge(&self, name: &str, value: u64) {
        debug!(metric_kind = "gauge", metric = %self.full_name(name), value);
    }
    fn sink_id(&self) -> &str {
        "stdout"
    }
}

// ════════════════════════════════════════════════════════════════════════════
// VecSink — for tests
// ════════════════════════════════════════════════════════════════════════════

/// Single observation captured by [`VecSink`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub kind: ObservationKind,
    pub name: String,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationKind {
    Counter,
    Gauge,
}

/// In-memory observation collector. Used by tests to assert on what was
/// emitted. Thread-safe via internal `Mutex`.
#[derive(Debug, Default)]
pub struct VecSink {
    observations: Mutex<Vec<Observation>>,
}

impl VecSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot all collected observations.
    pub fn snapshot(&self) -> Vec<Observation> {
        self.observations
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Snapshot then clear.
    pub fn drain(&self) -> Vec<Observation> {
        self.observations
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default()
    }

    /// Look up the most recent observation for `name`.
    pub fn last_value(&self, name: &str) -> Option<u64> {
        self.observations
            .lock()
            .ok()?
            .iter()
            .rev()
            .find(|o| o.name == name)
            .map(|o| o.value)
    }

    /// Count of observations recorded.
    pub fn len(&self) -> usize {
        self.observations.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// True if no observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MetricsSink for VecSink {
    fn counter(&self, name: &str, value: u64) {
        if let Ok(mut g) = self.observations.lock() {
            g.push(Observation {
                kind: ObservationKind::Counter,
                name: name.to_string(),
                value,
            });
        }
    }
    fn gauge(&self, name: &str, value: u64) {
        if let Ok(mut g) = self.observations.lock() {
            g.push(Observation {
                kind: ObservationKind::Gauge,
                name: name.to_string(),
                value,
            });
        }
    }
    fn sink_id(&self) -> &str {
        "vec"
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Stats → sink pumps
// ════════════════════════════════════════════════════════════════════════════

/// Pump [`TipProofServiceStats`] into the sink. Call this on a periodic
/// interval (typically 15s) from a background task.
///
/// Counter metrics:
///   - `service.extends.attempted_total`
///   - `service.extends.succeeded_total`
///   - `service.extends.rejected_total`
///   - `service.anchor_resets_total`
///   - `service.prefix_drops_total`
///
/// Gauge metrics:
///   - `service.tip_height_current`
///   - `service.step_count_current`
///   - `service.last_extend_at_unix_current` (0 if `None`)
pub fn pump_service_stats(stats: &TipProofServiceStats, sink: &dyn MetricsSink) {
    sink.counter("service.extends.attempted_total", stats.total_extends_attempted);
    sink.counter("service.extends.succeeded_total", stats.total_extends_succeeded);
    sink.counter("service.extends.rejected_total", stats.total_extends_rejected);
    sink.counter("service.anchor_resets_total", stats.total_anchor_resets);
    sink.counter("service.prefix_drops_total", stats.total_prefix_drops);

    sink.gauge("service.tip_height_current", stats.current_tip_height);
    sink.gauge("service.step_count_current", stats.current_step_count as u64);
    sink.gauge(
        "service.last_extend_at_unix_current",
        stats.last_extend_at_unix.unwrap_or(0),
    );
}

/// Pump [`TipProofClientStats`] into the sink.
///
/// Counter metrics:
///   - `client.proofs.fetched_total`
///   - `client.proofs.accepted_total`
///   - `client.proofs.rejected_total`
///   - `client.proofs.rollbacks_rejected_total`
///
/// Gauge metrics:
///   - `client.last_accepted_tip_height_current` (0 if `None`)
///   - `client.last_accepted_at_unix_current` (0 if `None`)
pub fn pump_client_stats(stats: &TipProofClientStats, sink: &dyn MetricsSink) {
    sink.counter("client.proofs.fetched_total", stats.total_proofs_fetched);
    sink.counter("client.proofs.accepted_total", stats.total_proofs_accepted);
    sink.counter("client.proofs.rejected_total", stats.total_proofs_rejected);
    sink.counter(
        "client.proofs.rollbacks_rejected_total",
        stats.total_rollbacks_rejected,
    );

    sink.gauge(
        "client.last_accepted_tip_height_current",
        stats.last_accepted_tip_height.unwrap_or(0),
    );
    sink.gauge(
        "client.last_accepted_at_unix_current",
        stats.last_accepted_at_unix.unwrap_or(0),
    );
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nop_sink_discards_silently() {
        let sink = NopSink;
        sink.counter("foo", 42);
        sink.gauge("bar", 7);
        assert_eq!(sink.sink_id(), "nop");
    }

    #[test]
    fn stdout_sink_with_prefix_prefixes_metric_name() {
        let sink = StdoutSink::new("test_prefix");
        assert_eq!(sink.full_name("foo"), "test_prefix.foo");

        let no_prefix = StdoutSink::new("");
        assert_eq!(no_prefix.full_name("foo"), "foo");
    }

    #[test]
    fn vec_sink_records_observations_in_order() {
        let sink = VecSink::new();
        sink.counter("a", 1);
        sink.gauge("b", 2);
        sink.counter("c", 3);

        let obs = sink.snapshot();
        assert_eq!(obs.len(), 3);
        assert_eq!(
            obs[0],
            Observation {
                kind: ObservationKind::Counter,
                name: "a".to_string(),
                value: 1
            }
        );
        assert_eq!(obs[1].kind, ObservationKind::Gauge);
        assert_eq!(obs[1].name, "b");
        assert_eq!(obs[2].name, "c");
    }

    #[test]
    fn vec_sink_drain_clears_observations() {
        let sink = VecSink::new();
        sink.counter("x", 1);
        let drained = sink.drain();
        assert_eq!(drained.len(), 1);
        assert!(sink.is_empty());
    }

    #[test]
    fn vec_sink_last_value_returns_most_recent_observation() {
        let sink = VecSink::new();
        sink.counter("metric", 1);
        sink.counter("metric", 5);
        sink.counter("metric", 9);
        assert_eq!(sink.last_value("metric"), Some(9));
        assert_eq!(sink.last_value("nonexistent"), None);
    }

    #[test]
    fn pump_service_stats_emits_all_expected_metrics() {
        let sink = VecSink::new();
        let mut stats = TipProofServiceStats::default();
        stats.total_extends_attempted = 100;
        stats.total_extends_succeeded = 95;
        stats.total_extends_rejected = 5;
        stats.total_anchor_resets = 2;
        stats.total_prefix_drops = 1;
        stats.current_tip_height = 1000;
        stats.current_step_count = 50;
        stats.last_extend_at_unix = Some(1700000000);

        pump_service_stats(&stats, &sink);

        // 5 counters + 3 gauges = 8 observations
        assert_eq!(sink.len(), 8);
        assert_eq!(
            sink.last_value("service.extends.attempted_total"),
            Some(100)
        );
        assert_eq!(
            sink.last_value("service.extends.succeeded_total"),
            Some(95)
        );
        assert_eq!(sink.last_value("service.tip_height_current"), Some(1000));
        assert_eq!(sink.last_value("service.step_count_current"), Some(50));
        assert_eq!(
            sink.last_value("service.last_extend_at_unix_current"),
            Some(1700000000)
        );
    }

    #[test]
    fn pump_service_stats_none_timestamp_emits_zero() {
        let sink = VecSink::new();
        let stats = TipProofServiceStats::default(); // last_extend_at_unix: None
        pump_service_stats(&stats, &sink);
        assert_eq!(
            sink.last_value("service.last_extend_at_unix_current"),
            Some(0)
        );
    }

    #[test]
    fn pump_client_stats_emits_all_expected_metrics() {
        let sink = VecSink::new();
        let mut stats = TipProofClientStats::default();
        stats.total_proofs_fetched = 50;
        stats.total_proofs_accepted = 45;
        stats.total_proofs_rejected = 3;
        stats.total_rollbacks_rejected = 2;
        stats.last_accepted_tip_height = Some(123);
        stats.last_accepted_at_unix = Some(1700000000);

        pump_client_stats(&stats, &sink);

        assert_eq!(sink.len(), 6);
        assert_eq!(sink.last_value("client.proofs.fetched_total"), Some(50));
        assert_eq!(sink.last_value("client.proofs.accepted_total"), Some(45));
        assert_eq!(
            sink.last_value("client.proofs.rollbacks_rejected_total"),
            Some(2)
        );
        assert_eq!(
            sink.last_value("client.last_accepted_tip_height_current"),
            Some(123)
        );
    }

    #[test]
    fn pump_client_stats_with_none_values_emits_zeros() {
        let sink = VecSink::new();
        let stats = TipProofClientStats::default();
        pump_client_stats(&stats, &sink);

        assert_eq!(
            sink.last_value("client.last_accepted_tip_height_current"),
            Some(0)
        );
        assert_eq!(
            sink.last_value("client.last_accepted_at_unix_current"),
            Some(0)
        );
    }

    #[test]
    fn trait_object_dispatch_works() {
        // Sanity: verify that all sinks implement the trait identically
        // from a `dyn MetricsSink` reference.
        let sinks: Vec<Box<dyn MetricsSink>> = vec![
            Box::new(NopSink),
            Box::new(StdoutSink::new("test")),
            Box::new(VecSink::new()),
        ];
        for s in &sinks {
            s.counter("trait_test", 1);
            s.gauge("trait_test_gauge", 2);
            assert!(!s.sink_id().is_empty());
        }
    }

    #[test]
    fn vec_sink_concurrent_writes_dont_lose_observations() {
        use std::sync::Arc;
        use std::thread;

        let sink: Arc<dyn MetricsSink> = Arc::new(VecSink::new());
        let mut handles = Vec::new();
        for tid in 0..4 {
            let s = Arc::clone(&sink);
            handles.push(thread::spawn(move || {
                for i in 0..100u64 {
                    s.counter(&format!("t{tid}"), i);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Each thread emits 100 observations → 400 total. The VecSink
        // wraps a Mutex so all writes are serialised.
        let vec_sink: &VecSink =
            (sink.as_ref() as &dyn std::any::Any)
                .downcast_ref::<VecSink>()
                .expect("type known");
        assert_eq!(vec_sink.len(), 400);
    }
}
