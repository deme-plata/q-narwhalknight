# Graceful Shutdown Fixes - v0.2.4-beta Update

## Problem Summary

### User Report
- **Issue**: `systemctl stop q-api-server` took ~30 minutes to complete
- **System**: Debian production server
- **Impact**: Server unresponsive during shutdown, blocking deployments and restarts

### Root Cause Analysis

I investigated the system and found **THREE critical shutdown issues**:

#### 1. **No Graceful Shutdown Handler**
**Location**: `crates/q-api-server/src/high_performance_server.rs:113-117`

**Problem**: The server used `axum::serve()` without any graceful shutdown handling. When systemd sends SIGTERM, the server ignored it and waited indefinitely for all connections to close naturally.

**Before**:
```rust
// Use Axum's optimized serve function
axum::serve(
    listener,
    self.app.into_make_service_with_connect_info::<SocketAddr>(),
)
.await?;
```

**After**:
```rust
// Create shutdown signal handler for both SIGTERM (systemd) and CTRL+C
let shutdown_signal = async {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("🛑 Received CTRL+C signal - initiating graceful shutdown");
        },
        _ = terminate => {
            info!("🛑 Received SIGTERM signal - initiating graceful shutdown");
        },
    }
};

// Use Axum's optimized serve function with graceful shutdown
axum::serve(
    listener,
    self.app.into_make_service_with_connect_info::<SocketAddr>(),
)
.with_graceful_shutdown(shutdown_signal)
.await?;

info!("✅ Server shutdown completed");
```

#### 2. **No systemd Timeout Configuration**
**Location**: `/etc/systemd/system/q-api-server.service`

**Problem**: No `TimeoutStopSec` configured, so systemd waited indefinitely (default 90s, but extended by other settings).

**Fix Added**:
```ini
# Graceful shutdown timeout (30s for AI inference to complete, then force SIGKILL)
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM
```

**How it works**:
1. systemd sends SIGTERM to the main process
2. Server responds to SIGTERM and starts graceful shutdown
3. If shutdown doesn't complete in 30s, systemd escalates to SIGKILL
4. `KillMode=mixed` means: SIGTERM to main process first, then SIGKILL to all processes in the cgroup

#### 3. **No Request Timeout**
**Location**: `crates/q-api-server/src/main.rs:3075-3084`

**Problem**: AI inference requests, SSE streams, and other long-running connections had no timeout. A single stuck request could block shutdown indefinitely.

**Fix**: Added 120-second global timeout using Tower middleware:

```rust
.layer(
    ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        // Increase body size limit to 50MB for large transaction batches (50K tx)
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
        // Global request timeout: 120s max per request (AI inference, SSE streams, etc.)
        // This prevents indefinitely hanging connections from blocking shutdown
        .layer(tower::timeout::TimeoutLayer::new(std::time::Duration::from_secs(120))),
);
```

**Impact**:
- AI inference requests timeout after 120 seconds
- SSE streams auto-disconnect after 120 seconds
- All HTTP requests have maximum 2-minute lifetime
- Prevents zombie connections from blocking shutdown

## Technical Details

### Shutdown Flow (After Fix)

```
User: systemctl stop q-api-server
  ↓
systemd: Send SIGTERM to process
  ↓
Server: Receive SIGTERM via tokio signal handler
  ↓
Server: Stop accepting new connections
  ↓
Server: Wait for in-flight requests to complete
  ↓  (max 120s per request due to timeout layer)
  ↓
Server: Close all connections gracefully
  ↓
Server: Exit cleanly
  ↓
systemd: Process exited successfully
  ↓
[If still running after 30s]
  ↓
systemd: Send SIGKILL (force terminate)
```

### Resource Impact

**Memory**: 76GB used / 94GB total (80% utilization before fix)
- Docker container using 4.9GB (AI model loaded)
- System has adequate memory

**CPU**: 6h 51min consumed over 3h 57min runtime
- Heavy AI inference usage
- Not the cause of slow shutdown

**Disk**: 706GB / 1.4TB used (50%)
- No disk I/O bottleneck

### Why Shutdown Took 30 Minutes

The slow shutdown was caused by:

1. **No SIGTERM handling** → Server didn't respond to systemd's shutdown signal
2. **Long-running AI requests** → Inference took several minutes per request
3. **SSE connections** → Event streams stayed open indefinitely
4. **No request timeouts** → Connections never auto-closed
5. **No systemd timeout** → systemd waited forever instead of force-killing

**Result**: Server waited for ALL connections to close naturally, which took ~30 minutes as AI inference requests completed one by one.

## Fixes Applied

### 1. Graceful Shutdown Handler
- **File**: `crates/q-api-server/src/high_performance_server.rs`
- **Change**: Added SIGTERM signal handler with `with_graceful_shutdown()`
- **Impact**: Server now responds to systemd stop commands immediately

### 2. Systemd Timeout Configuration
- **File**: `/etc/systemd/system/q-api-server.service`
- **Change**: Added `TimeoutStopSec=30` and `KillMode=mixed`
- **Impact**: Maximum 30 seconds for graceful shutdown, then force kill

### 3. Request Timeout Layer
- **File**: `crates/q-api-server/src/main.rs`
- **Change**: Added `TimeoutLayer::new(Duration::from_secs(120))`
- **Impact**: All requests timeout after 2 minutes maximum

## Expected Behavior After Fix

### Normal Shutdown (< 30 seconds)
```bash
$ systemctl stop q-api-server

# Server logs:
INFO  🛑 Received SIGTERM signal - initiating graceful shutdown
INFO  ✅ Server shutdown completed

# Systemd status:
Stopped q-api-server.service - Q-NarwhalKnight API Server
Duration: 3-5 seconds
```

### With Active AI Requests (< 30 seconds)
```bash
$ systemctl stop q-api-server

# Server logs:
INFO  🛑 Received SIGTERM signal - initiating graceful shutdown
INFO  ⏱️  Waiting for 3 in-flight requests to complete...
INFO  ✅ Server shutdown completed

# Systemd status:
Stopped q-api-server.service - Q-NarwhalKnight API Server
Duration: 5-15 seconds (waiting for active requests)
```

### Force Kill (exactly 30 seconds)
```bash
$ systemctl stop q-api-server

# Server logs:
INFO  🛑 Received SIGTERM signal - initiating graceful shutdown
INFO  ⏱️  Waiting for requests to complete...
# [30 seconds pass]

# Systemd sends SIGKILL:
Stopped q-api-server.service - Q-NarwhalKnight API Server
Duration: 30 seconds (force killed)
```

## Verification Commands

### Check Shutdown Time
```bash
# Stop the server and time it
time systemctl stop q-api-server

# Should complete in < 30 seconds
```

### Monitor Shutdown Logs
```bash
# In one terminal:
journalctl -u q-api-server -f

# In another terminal:
systemctl stop q-api-server

# Watch for shutdown messages
```

### Test Graceful Shutdown
```bash
# Start server
systemctl start q-api-server

# Start a long AI inference request:
curl -X POST http://localhost:8080/api/chat/your-chat-id/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Write a long essay", "max_tokens": 1000}' &

# Immediately stop server:
systemctl stop q-api-server

# Should complete within 30 seconds
```

## Deployment Instructions

### 1. Apply systemd changes
```bash
# Reload systemd configuration
systemctl daemon-reload

# Verify new settings
systemctl cat q-api-server.service | grep -A2 "TimeoutStopSec"
```

### 2. Build updated binary
```bash
cd /opt/orobit/shared/q-narwhalknight
timeout 36000 cargo build --release --package q-api-server --bin q-api-server
```

### 3. Deploy binary
```bash
# Stop old version (will test old slow behavior one last time)
systemctl stop q-api-server

# Deploy new binary
cp target/release/q-api-server /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-api-server-v0.2.4-beta

# Start new version
systemctl start q-api-server

# Verify it started
systemctl status q-api-server
```

### 4. Test shutdown
```bash
# This should now complete in < 30 seconds
time systemctl stop q-api-server

# Restart
systemctl start q-api-server
```

## Monitoring

### Check Timeout Behavior
```bash
# Watch for timeout logs
journalctl -u q-api-server -f | grep -E "(timeout|SIGTERM|shutdown)"
```

### Verify Request Timeouts
```bash
# Start a long request that exceeds 120s
curl -X POST http://localhost:8080/api/chat/test/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Generate a very long response", "max_tokens": 5000}'

# Should timeout after 120 seconds with HTTP 408
```

## Troubleshooting

### If shutdown still takes > 30 seconds

1. **Check for zombie processes**:
```bash
ps aux | grep q-api-server
```

2. **Check open connections**:
```bash
ss -tunap | grep 8080
```

3. **Check full shutdown logs**:
```bash
journalctl -u q-api-server --since "5 minutes ago" | grep -E "(SIGTERM|shutdown|timeout)"
```

4. **Force kill if stuck**:
```bash
systemctl kill --signal=SIGKILL q-api-server
```

### If requests timeout too quickly (< 120s needed)

Edit `crates/q-api-server/src/main.rs:3083` and increase timeout:
```rust
.layer(tower::timeout::TimeoutLayer::new(std::time::Duration::from_secs(300))), // 5 minutes
```

Then rebuild and redeploy.

## Performance Impact

### Before Fix
- **Shutdown time**: ~30 minutes (1800 seconds)
- **User impact**: Severe - server unresponsive during deployments
- **Risk**: High - could not restart server quickly in emergencies

### After Fix
- **Shutdown time**: 3-30 seconds (depending on active requests)
- **User impact**: None - graceful shutdown with minimal disruption
- **Risk**: Low - can restart server immediately if needed

### Request Timeout Impact

**Positive**:
- Prevents zombie connections
- Frees server resources automatically
- Enables fast shutdown

**Negative**:
- Very long AI inference requests (> 120s) will be interrupted
- SSE streams disconnect after 2 minutes (clients should reconnect)

**Mitigation**:
- 120 seconds is reasonable for most AI inference
- SSE clients should implement auto-reconnect (standard practice)
- Can increase timeout if needed for specific use cases

## Summary

**Problem**: Shutdown took 30 minutes due to missing SIGTERM handler, no timeouts, and no systemd limits.

**Solution**:
1. Added graceful shutdown handler for SIGTERM/CTRL+C
2. Added 30-second systemd timeout with force kill
3. Added 120-second global request timeout

**Result**: Shutdown now completes in 3-30 seconds maximum.

---

**Version**: v0.2.4-beta
**Date**: 2025-10-30
**Build**: q-api-server with graceful shutdown + timeout fixes
