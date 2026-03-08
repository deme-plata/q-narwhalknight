# Q-Flux Issues

| # | Title | Priority | Status | Labels |
|---|-------|----------|--------|--------|
| 001 | Wire BandwidthLimiter into WebSocket Splice | High | Done | libp2p, performance |
| 002 | Wire Circuit Breaker into Upstream Forward Path | Medium | Done | reliability |
| 003 | Per-Peer Byte Tracking | Low | Done | observability |
| 004 | Auto-Tier Classification from Traffic Patterns | Low | Done | libp2p |
| 005 | Expose PeerTracker Stats via Admin /peers | Medium | Done | observability, admin |
| 006 | libp2p Detection for HTTP/2 Connections | Medium | Open | libp2p, h2 |
| 007 | Wire GossipsubDedup into WebSocket Data Path | Low | Open | libp2p, performance |
| 008 | Weighted Routing for Super-Cluster Peers | Low | Open | super-cluster |
| 009 | Prometheus Metrics for libp2p Peers | Medium | Done | prometheus |
| 010 | Graceful Connection Draining on Config Reload | Medium | Open | reliability |
| 011 | Automatic Request Retry on Upstream Failure | High | Done | reliability |
| 012 | Add X-Request-ID Header for Request Tracing | Low | Done | observability |

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
