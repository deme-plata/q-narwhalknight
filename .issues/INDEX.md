# Q-Flux Issues

| # | Title | Priority | Status | Labels |
|---|-------|----------|--------|--------|
| 001 | Wire BandwidthLimiter into WebSocket Splice | High | Done | libp2p, performance |
| 002 | Wire Circuit Breaker into Upstream Forward Path | Medium | Done | reliability |
| 003 | Per-Peer Byte Tracking | Low | Done | observability |
| 004 | Auto-Tier Classification from Traffic Patterns | Low | Done | libp2p |
| 005 | Expose PeerTracker Stats via Admin /peers | Medium | Done | observability, admin |
| 006 | libp2p Detection for HTTP/2 Connections | Medium | Deferred | libp2p, h2 |
| 007 | Wire GossipsubDedup into WebSocket Data Path | Low | Deferred | libp2p, performance |
| 008 | Weighted Routing for Super-Cluster Peers | Low | Deferred | super-cluster |
| 009 | Prometheus Metrics for libp2p Peers | Medium | Done | prometheus |
| 010 | Graceful Connection Draining on Config Reload | Medium | Done | reliability |
| 011 | Automatic Request Retry on Upstream Failure | High | Done | reliability |
| 012 | Add X-Request-ID Header for Request Tracing | Low | Done | observability |
| 013 | Upstream Retry Metrics (retries/successes) | Medium | Done | observability, prometheus |
| 014 | Splice Zero-Copy for WebSocket/SSE Passthrough | High | Open | performance, io_uring, zero-copy |
| 015 | io_uring Config Section + Feature Detection | Medium | Open | io_uring, config |
| 016 | Splice Zero-Copy Metrics | Low | Open | observability, io_uring, prometheus |
| 017 | kTLS Kernel TLS Offload | Medium | Planned | performance, tls, kernel |
| 018 | Connection Draining Improvements | Medium | Planned | reliability, deployment |
| 019 | OCSP Auto-Fetch and Periodic Refresh | High | Done | tls, security, performance |
| 020 | Duration Parser Bug — "ms" Matched by "s" | Medium | Done | bug, config |
| 021 | ACME Certificate Automation | Medium | Planned | tls, security, automation |
| 022 | Upstream Round-Robin and Failover Tests | Medium | Open | testing, reliability |

## How to pick up an issue

```bash
# Read the issue
cat .issues/001-flux-bandwidth-limiter.md

# Create a branch
git checkout -b flux/issue-001-bandwidth-limiter

# Work, then commit
cargo check --package q-flux
git add crates/q-flux/
git commit -m "feat(q-flux): wire BandwidthLimiter into WebSocket splice (closes #001)"
```
