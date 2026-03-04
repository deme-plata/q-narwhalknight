# QNK Infrastructure Bottleneck Analysis — Technical Review
**Date:** 2026-03-04 13:15 UTC
**Network Age:** 10 days (mainnet-genesis launched Feb 22, 2026)
**Active Miners:** ~400+ (7+ threads each = ~2,800 req/s)
**Chain Height:** ~7,440,000 blocks
**Version:** v9.0.0 (Epsilon), v9.0.0 (Beta), v8.9.6 (Gamma/Delta)

---

## 1. EXECUTIVE SUMMARY — THE BOTTLENECK

**Miners cannot submit solutions.** The q-api-server on Epsilon (the primary node serving all traffic via `quillon.xyz`) returns **23,405 HTTP 503 errors per minute** at the application layer. Nginx is healthy — the 503s originate from the Rust application's tower middleware stack, not from reverse proxy errors.

**Root cause chain:**
1. Epsilon was restarted ~20 min before data collection → entered **sync mode** (6,875 blocks behind)
2. Sync processes thousands of coinbase TXs per block → saturates CPU + RocksDB I/O
3. Mining HTTP handlers compete for the same RocksDB locks and tokio runtime
4. Application-level `ConcurrencyLimitLayer(5000)` + mining handler `MAX_MINING_CONCURRENCY(500)` starts rejecting requests
5. Every rejected request has exactly **50ms latency** — correlates with batch commit window size (50 blocks × ~1ms = 50ms per batch)
6. Miners receive 503 → retry immediately → amplifies load (thundering herd)

---

## 2. SERVER STATUS MATRIX

| Server | Role | IP | Version | Height | Network | Behind | Status | Load | RAM | CPU |
|--------|------|----|---------|--------|---------|--------|--------|------|-----|-----|
| **Epsilon** | Primary (10Gbit) | 89.149.241.126 | v9.0.0 | 7,433,289 | 7,440,164 | **6,875** | **syncing** | **23.25** | 25/62 GB + 5.6G swap | 48-core Xeon Gold 5118 |
| **Beta** | Bootstrap | 185.182.185.227 | v9.0.0 | 7,439,710 | 7,439,710 | 0 | ready | 4.16 | 14/94 GB | 18-core EPYC |
| **Gamma** | Backup | 109.205.176.60 | **v8.9.6** | 7,439,729 | 7,439,729 | 0 | ready | 5.92 | 4.5/7.8 GB | 4-core |
| **Delta** | Bootstrap #4 | 5.79.79.158 | **v8.9.6** | **6,200,299** | 6,209,299 | **1,239,865** | **syncing** | **26.89** | 12/31 GB + 8G swap | 8-core |

### Critical observations:
- **Epsilon is the ONLY server handling user traffic** (quillon.xyz DNS → Epsilon). Beta/Gamma/Delta are all marked `down` in nginx upstream.
- **Delta is 1.24M blocks behind** — effectively useless for load sharing. Running v8.9.6 on 8 cores at load 27.
- **Gamma is at tip but on old v8.9.6** — could share load if upgraded to v9.0.0 and added to nginx upstream.
- **Beta is healthy and at tip (v9.0.0)** — but not in Epsilon's upstream (marked `down`).

---

## 3. THE 503 PROBLEM IN DETAIL

### 3.1 Application-Level Concurrency Stack

```
Incoming HTTP request
  │
  ├─ Layer 1: tower::limit::ConcurrencyLimitLayer(5000)
  │   └─ Semaphore with 5000 permits. If all in use, request QUEUES (waits).
  │
  ├─ Layer 2: TraceLayer (logs request + response + latency)
  │
  ├─ Layer 3: TimeoutLayer(30s) — kills requests hanging >30s
  │
  ├─ Layer 4: CorsLayer::permissive()
  │
  ├─ Layer 5: BodyLimit(50MB)
  │
  └─ Handler (e.g., mining submit)
      │
      ├─ MINING_IN_FLIGHT atomic counter
      │   └─ if in_flight >= 500 → return 503 immediately
      │
      └─ SYNC GATE check
          └─ if local_height + 10 < network_height && !Q_ALLOW_SOLO_MINING → return 503
              (Epsilon has Q_ALLOW_SOLO_MINING=true, so this is BYPASSED)
```

### 3.2 The 50ms Signature

Every 503 has exactly 50ms latency. This is not random — it correlates with the batch sync commit window:

```rust
// main.rs line 1872:
// "New: 1000 blocks ÷ 50 = 20 commits × 50ms = 1 second"
```

During sync, blocks are committed in batches of 50. Each batch commit holds RocksDB write locks for ~50ms. Any mining/API request that needs RocksDB access during this window stalls for exactly one batch duration, then either succeeds or gets shed.

### 3.3 Request volume vs capacity

```
Incoming request rate:     ~2,800 req/s (400 miners × 7 threads)
Mining concurrency cap:    500 simultaneous
Global concurrency cap:    5,000 simultaneous
503 rate:                  ~390/sec (23,405/min)
Reject ratio:              ~14% of all requests
```

The mining cap of 500 is the tighter bottleneck. With 2,800 req/s and each request taking 50ms+ (blocked on sync batches), the in-flight count quickly hits 500:
```
500 in-flight × 50ms average = 10,000 req/s theoretical throughput
Actual incoming: 2,800 req/s → should handle this
BUT: During heavy sync, latency spikes to 200-500ms:
500 in-flight × 200ms average = 2,500 req/s max → OVERFLOW → 503s
```

---

## 4. EPSILON TCP/NETWORK STATE

```
Total TCP sockets:     133,082
  ├─ ESTABLISHED:       7,559
  ├─ TIME-WAIT:       119,680  ← 90% of all sockets
  ├─ LAST-ACK:          5,429
  ├─ SYN-RECV:            530
  ├─ CLOSING:             336
  ├─ FIN-WAIT-1:          128
  └─ Other:                ~50

Connections to backend (:8080):    1,193  (nginx → app)
Connections on :443 (miners):      5,864  (external TLS)
Connections on :9001 (P2P):           24

Kernel TCP settings:
  tcp_max_tw_buckets:    131,072  (increased from 30,000)
  tcp_fin_timeout:       15s      (reduced from 60s)
  tcp_tw_reuse:          2        (reuse TIME-WAIT for outbound)
  somaxconn:             65,535
  ip_local_port_range:   10000-65535  (55,535 ephemeral ports)
  nf_conntrack_max:      1,048,576
  nf_conntrack_count:    301,348   (29% of max)

File descriptors:
  q-api-server open FDs:   18,179
  q-api-server FD limit:   131,072

iptables:
  INPUT policy DROP:       11M packets dropped (cumulative)
  INVALID state drops:     26M packets (cumulative)
```

### 4.1 TIME-WAIT Analysis

119,680 TIME-WAIT sockets is the legacy of the previous `Connection ""` (keepalive) configuration. With `tcp_fin_timeout=15s`, these will drain within 15 seconds of the nginx restart. However, with `Connection "close"` now set, each mining request creates a new TCP connection → closes it → TIME-WAIT. At 2,800 req/s × 15s fin_timeout = ~42,000 TIME-WAIT sockets at steady state. This is manageable within the 131K tw_buckets limit.

### 4.2 Ephemeral Port Exhaustion Risk

`ip_local_port_range: 10000-65535` = 55,535 ports. With 119K TIME-WAIT, ports are being reused via `tcp_tw_reuse=2`. But the TIME-WAIT sockets are on `sport=:443` (server-side), not ephemeral. The risk is on the **nginx → backend** direction:
```
nginx → 127.0.0.1:8080 uses ephemeral ports
With Connection "close", each request uses a new ephemeral port
At 2,800 req/s × 15s fin_timeout = 42,000 ephemeral ports consumed
Max available: 55,535
Utilization: 75% → approaching danger zone
```

**Recommendation:** Reduce `tcp_fin_timeout` to 5s, or switch nginx→backend back to keepalive (the pileup problem was backend→client, not nginx→backend).

---

## 5. EPSILON NGINX CONFIGURATION — FULL ANALYSIS

### 5.1 Current Configuration (sites-enabled)

```nginx
# Upstream — single active server
upstream qnk_api_backend {
    least_conn;
    server 127.0.0.1:8080 weight=20 max_fails=3 fail_timeout=5s;   # ONLY active server
    server 185.182.185.227:8080 weight=5 ... down;   # Beta — marked down
    server 109.205.176.60:8080 weight=2 ... down;    # Gamma — marked down
    server 5.79.79.158:8080 weight=8 ... down;       # Delta — marked down
    keepalive 32;
    keepalive_requests 1000;
    keepalive_timeout 10s;
}
```

### 5.2 Issues Found

#### ISSUE 1: SINGLE POINT OF FAILURE (CRITICAL)
All traffic goes to localhost:8080 (Epsilon). Beta is at tip on v9.0.0 but marked `down`. When Epsilon syncs or restarts, 100% of miners fail. There is zero failover capacity.

**Fix:** Add Beta as active upstream:
```nginx
server 185.182.185.227:8080 weight=5 max_fails=3 fail_timeout=10s;  # Remove "down"
```

#### ISSUE 2: SSE LOCATION USES `Connection "close"` (BUG)
```nginx
location /api/v1/events {
    proxy_set_header Connection "close";  # ← WRONG for SSE!
}
```
SSE (Server-Sent Events) is a long-lived connection. Sending `Connection: close` tells the backend to close after the first event chunk. This breaks real-time event streaming.

**Fix:**
```nginx
proxy_set_header Connection "";  # Keep-alive for SSE (long-lived)
```

#### ISSUE 3: KEEPALIVE STILL IN UPSTREAM BLOCK
```nginx
keepalive 32;
keepalive_requests 1000;
keepalive_timeout 10s;
```
With `Connection "close"` on all location blocks, the upstream keepalive pool is never used. These directives are dead config. Not harmful but confusing.

**Fix:** Either remove them, or restore keepalive for specific locations (sync, blocks) where connection reuse helps.

#### ISSUE 4: MINING TIMEOUT TOO TIGHT FOR SYNCING NODE
```nginx
location /api/v1/mining/ {
    proxy_read_timeout 5s;
    proxy_send_timeout 5s;
    proxy_connect_timeout 3s;
}
```
When Epsilon is syncing, RocksDB contention causes 50-500ms latency spikes. A 5s timeout is fine for a healthy node but marginal under sync load. Miners retry immediately on timeout → amplifies thundering herd.

**Recommendation:** Keep 5s timeouts but enable `proxy_next_upstream` to failover to Beta instead of returning error to client.

#### ISSUE 5: NO `proxy_next_upstream` ON EPSILON (CRITICAL)
Beta's nginx has:
```nginx
proxy_next_upstream error timeout http_429 http_502 http_503 http_504 non_idempotent;
proxy_next_upstream_timeout 3s;
proxy_next_upstream_tries 3;
```
**Epsilon's nginx has NO `proxy_next_upstream`** on any location block. When Epsilon's local app returns 503, nginx just forwards the 503 to the client. If Beta were un-downed in the upstream, nginx could retry the request on Beta automatically.

**Fix:** Add to mining and API location blocks:
```nginx
proxy_next_upstream error timeout http_503;
proxy_next_upstream_timeout 3s;
proxy_next_upstream_tries 2;
```

#### ISSUE 6: CACHEABLE ENDPOINTS DON'T INCLUDE MINING CHALLENGE
```nginx
location ~ ^/api/v1/(health|status|network/supply|...|mining/stats|dex/pools) {
    proxy_cache api_cache;
    proxy_cache_valid 200 5s;
}
```
`/api/v1/mining/challenge` is NOT in the cacheable regex. Each miner thread fetches a new challenge every few seconds. With 400 miners × 7 threads = 2,800 challenge requests competing with submit requests. Challenges could be cached for 1-2s since they only change per block (~1s).

**Fix:** Add `/api/v1/mining/challenge` to cacheable endpoints (or add a separate location with 1s cache):
```nginx
location = /api/v1/mining/challenge {
    proxy_pass http://qnk_api_backend;
    proxy_cache api_cache;
    proxy_cache_valid 200 1s;
    proxy_cache_use_stale error timeout updating http_502 http_503;
    proxy_cache_lock on;
    # ... standard headers ...
}
```
This **alone could reduce backend load by ~50%** — 2,800 challenge fetches/sec → 1 fetch/sec (served from nginx cache).

#### ISSUE 7: NO RATE LIMITING ON MINING (ALL DISABLED)
```nginx
# limit_conn perip_conn 15;  # DISABLED
# limit_conn_status 503;     # DISABLED
# limit_req zone=api burst=500 nodelay;  # REMOVED
```
All rate limiting was disabled due to previous issues. This means a single miner with 100 threads can monopolize the backend. There should be per-IP rate limiting with a generous burst:
```nginx
limit_req zone=mining burst=100 nodelay;  # 50r/s per IP, burst 100
limit_conn perip_mining 30;               # 30 concurrent per IP
```

#### ISSUE 8: `gzip` ON FOR API RESPONSES
```nginx
gzip on;
gzip_types ... application/json ...;
```
Mining submit/challenge are tiny JSON payloads (100-500 bytes). Compressing these wastes CPU cycles. At 2,800 req/s, gzip burns significant CPU on content that's already small.

**Fix:** Add minimum length or exclude mining endpoints:
```nginx
gzip_min_length 1024;  # Already set — good, but verify it applies to proxied responses
gzip_proxied any;       # May need this for proxied responses
```
The `gzip_min_length 1024` is already set, but verify it's applied to proxied responses too.

---

## 6. APPLICATION-LEVEL BOTTLENECKS

### 6.1 Sync + Mining Contention (PRIMARY ISSUE)

The q-api-server runs sync and mining on the **same tokio runtime**. During heavy sync:

```
Sync task:        Read block from P2P → deserialize → validate → write to RocksDB
                  Batch of 50 blocks → single RocksDB write batch → ~50ms commit

Mining handler:   Read challenge from memory → validate solution → write to RocksDB
                  Needs RocksDB access → blocks during batch commit → 50ms stall
```

Both tasks compete for:
1. **Tokio runtime threads** (48 cores, but sync uses `spawn_blocking` which has its own pool)
2. **RocksDB read/write access** (single database instance, shared across all tasks)
3. **Memory** (25GB used + 5.6GB swap — swap thrashing adds latency)

### 6.2 Mining Handler Concurrency Cap

```rust
const MAX_MINING_CONCURRENCY: u32 = 500;
```

With 400 miners × 7 threads, the instantaneous in-flight count often exceeds 500 during sync-induced latency spikes. The cap triggers, returning instant 503. Miners retry → more 503s → thundering herd.

### 6.3 RocksDB Block Cache

```
ROCKSDB_BLOCK_CACHE_MB=16384  (16GB)
```

16GB block cache on a 62GB machine using 25GB for the process = 41GB committed. With OS page cache and other allocations, this pushes into swap (5.6GB in swap). Swap I/O adds latency to every RocksDB read.

**Recommendation:** Reduce to 8192MB (8GB), freeing RAM for OS page cache and reducing swap pressure.

### 6.4 Batch Sync Parameters (Supernode config)

```
Q_TURBO_PARALLEL_STREAMS=64
Q_TURBO_CHUNK_SIZE=5000
Q_SYNC_RAYON_THREADS=16
Q_SYNC_MAX_CONCURRENCY=16
```

64 parallel sync streams + 16 rayon threads + 16 max concurrency is VERY aggressive. During sync, this creates massive I/O and CPU pressure, leaving little headroom for mining requests.

**Recommendation:** During catch-up sync (>100 blocks behind):
- Reduce parallel streams to 8
- Reduce chunk size to 1000
- Or: implement sync throttling when mining load is high

---

## 7. PER-SERVER DEEP DIVE

### 7.1 Epsilon (89.149.241.126) — Primary

| Metric | Value | Assessment |
|--------|-------|------------|
| Load average | 23.25 / 21.06 / 17.65 | HIGH for 48 cores (48% utilization) |
| RAM | 25GB / 62GB + 5.6GB swap | CONCERNING — swap thrashing |
| Disk (/) | 35/40GB (94%) | CRITICAL — root partition nearly full |
| Disk (/home) | 300GB/1.8TB (18%) | OK |
| FDs | 18,179 / 131,072 | OK (14%) |
| TCP estab | 7,559 | MODERATE |
| TIME-WAIT | 119,680 | HIGH but draining |
| nf_conntrack | 301,348 / 1,048,576 | OK (29%) |
| Uptime | 20 min (restarted 12:53) | JUST RESTARTED — syncing |
| 503 rate | 23,405/min | CRITICAL |
| P2P peers | 30 | Good |

**Critical issues:**
1. Root partition 94% full — may cause service crashes if log files or temp data fill it
2. 5.6GB in swap — RocksDB block cache too large
3. Syncing — 6,875 blocks behind, every block triggers heavy I/O
4. 503s — app-level rejection of mining requests during sync

### 7.2 Beta (185.182.185.227) — Bootstrap

| Metric | Value | Assessment |
|--------|-------|------------|
| Load average | 4.16 / 4.31 / 5.09 | OK for 18 cores |
| RAM | 14GB / 94GB | EXCELLENT — tons of headroom |
| Disk | 1.1TB / 1.4TB (84%) | OK |
| FDs | 28,964 / 65,536 | MODERATE (44%) — approaching limit |
| TCP estab | 341 | LOW |
| Height | 7,439,710 = network tip | FULLY SYNCED |
| Status | ready | HEALTHY |
| 503 rate | 0 | No 503s |
| P2P peers | 8 | Adequate |

**Issues:**
- FD usage at 44% of 65,536 limit — should increase to 131,072
- Not serving any mining traffic despite being healthy and at tip
- Log shows bogus "deep fork" warnings from peers announcing inflated heights (11T blocks)
- Gamma NOT marked down in Beta's nginx upstream (`qnk_mining`), which could route traffic to Gamma

### 7.3 Gamma (109.205.176.60) — Backup

| Metric | Value | Assessment |
|--------|-------|------------|
| Load average | 5.92 / 6.10 / 6.10 | HIGH for 4 cores (148%!) |
| RAM | 4.5GB / 7.8GB + 63MB swap | TIGHT but OK |
| Disk | 221GB / 296GB (78%) | OK |
| Height | 7,439,729 = network tip | FULLY SYNCED |
| Status | ready | HEALTHY |
| Version | **8.9.6** | OUTDATED (needs v9.0.0) |
| P2P peers | 6 | Adequate |

**Issues:**
- Running v8.9.6 — two minor versions behind (missing v9.0.0 optimizations)
- Load 5.92 on 4 cores = overloaded just from P2P processing
- Only 4 cores and 7.8GB RAM — limited capacity for mining traffic
- Supply mismatch: `128,040 QUG` vs Beta's `72,300 QUG` — state divergence!

### 7.4 Delta (5.79.79.158) — Bootstrap #4

| Metric | Value | Assessment |
|--------|-------|------------|
| Load average | 26.89 / 24.80 / 24.32 | EXTREME for 8 cores (336%!) |
| RAM | 12GB / 31GB + 8GB swap | SWAP HEAVY |
| Disk | 24GB / 49GB (51%) | OK |
| Height | **6,200,299** | **1.24M blocks behind!** |
| Status | syncing | FAR BEHIND |
| Version | **8.9.6** | OUTDATED |
| P2P peers | 17 | Good |
| Mining | Stalled (772 min without solutions) | NOT MINING |

**Issues:**
- 1.24 MILLION blocks behind — will take days to sync at current rate
- Load 27 on 8 cores — completely overloaded from sync
- 8GB in swap — severe performance degradation
- `Q_P2P_ONLY=1` but still has HTTP port open — wasting resources
- HEIGHT CLAMP active: clamping poisoned heights (7.4M → 6.2M)
- CoinGecko oracle failing (price parse errors) — non-critical but noisy

---

## 8. RECOMMENDED FIXES — PRIORITY ORDER

### P0: IMMEDIATE (fix mining NOW)

#### 8a. Enable Beta in Epsilon's nginx upstream
```nginx
# In /etc/nginx/sites-enabled/quillon.xyz on Epsilon:
upstream qnk_api_backend {
    least_conn;
    server 127.0.0.1:8080 weight=20 max_fails=3 fail_timeout=5s;
    server 185.182.185.227:8080 weight=5 max_fails=3 fail_timeout=10s;  # REMOVE "down"
    # Gamma and Delta stay down until upgraded
}
```
Add `proxy_next_upstream error timeout http_503;` to the mining location block. When Epsilon 503s, nginx retries on Beta automatically.

#### 8b. Cache mining challenges in nginx
```nginx
location = /api/v1/mining/challenge {
    proxy_pass http://qnk_api_backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "close";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_cache api_cache;
    proxy_cache_valid 200 1s;
    proxy_cache_use_stale error timeout updating http_502 http_503;
    proxy_cache_lock on;
    add_header Access-Control-Allow-Origin "*" always;
}
```
Reduces challenge fetch load from ~2,800/s to ~1/s.

#### 8c. Reduce RocksDB block cache on Epsilon
```bash
# In /etc/systemd/system/q-api-server.service on Epsilon:
Environment="ROCKSDB_BLOCK_CACHE_MB=8192"  # Was 16384
```
Frees ~8GB RAM, reduces swap pressure.

### P1: SHORT-TERM (next few hours)

#### 8d. Fix SSE Connection header
```nginx
location /api/v1/events {
    proxy_set_header Connection "";  # Was "close" — SSE needs keepalive
}
```

#### 8e. Reduce sync aggressiveness on Epsilon
```bash
Environment="Q_TURBO_PARALLEL_STREAMS=16"   # Was 64
Environment="Q_SYNC_MAX_CONCURRENCY=8"       # Was 16
```

#### 8f. Increase FD limit on Beta
```bash
# In /etc/systemd/system/q-api-server.service on Beta:
LimitNOFILE=131072  # Was 65536
```

### P2: MEDIUM-TERM (next day)

#### 8g. Upgrade Gamma to v9.0.0
Deploy v9.0.0 binary to Gamma, then add to Epsilon's nginx upstream as weight=2 backup.

#### 8h. Implement adaptive mining concurrency
Instead of fixed `MAX_MINING_CONCURRENCY=500`, scale based on current sync state:
```rust
let cap = if syncing { 200 } else { 1000 };
```

#### 8i. Add per-IP rate limiting back to mining
```nginx
limit_req zone=mining burst=50 nodelay;
limit_conn perip_mining 20;
```

### P3: LONG-TERM (architectural)

#### 8j. Separate sync and mining onto different tokio runtimes
Run block sync on a dedicated runtime with its own thread pool, preventing sync I/O from starving mining handlers.

#### 8k. Read-replica RocksDB for mining
Mining reads (challenge generation) can use a read-only RocksDB secondary instance, completely decoupling from sync writes.

#### 8l. Load balancer in front of Epsilon
Add a dedicated HAProxy/nginx LB that health-checks all nodes and automatically routes around unhealthy ones.

---

## 9. EPSILON NGINX CONFIG — FULL FILE

```nginx
# Quillon.xyz — Epsilon Supernode (10Gbit Intel Xeon)
# v1.0.0: Optimized firewall + nginx for 10Gbit dedicated server
# Applied: 2026-03-02

upstream qnk_api_backend {
    least_conn;
    server 127.0.0.1:8080 weight=20 max_fails=3 fail_timeout=5s;
    server 185.182.185.227:8080 weight=5 max_fails=3 fail_timeout=10s down;
    server 109.205.176.60:8080 weight=2 max_fails=3 fail_timeout=10s down;
    server 5.79.79.158:8080 weight=8 max_fails=3 fail_timeout=10s down;
    keepalive 32;
    keepalive_requests 1000;
    keepalive_timeout 10s;
}

server {
    server_name quillon.xyz www.quillon.xyz;
    root /home/orobit/q-narwhalknight/dist-final;
    index index.html;

    # --- Static/SPA ---
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location /downloads/ {
        alias /home/orobit/q-narwhalknight/dist-final/downloads/;
        autoindex off;
        add_header Content-Disposition "attachment";
        add_header Cache-Control "no-cache";
    }
    location / {
        try_files $uri $uri/ /index.html;
        location = / {
            add_header Cache-Control "no-cache, no-store, must-revalidate";
            try_files /index.html =404;
        }
    }

    # --- API (general) ---
    location /api/ {
        proxy_pass http://qnk_api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "close";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        # MISSING: proxy_next_upstream
        add_header Access-Control-Allow-Origin "*" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "..." always;
    }

    # --- SSE ---
    location /api/v1/events {
        proxy_pass http://qnk_api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "close";  # BUG: Should be "" for SSE
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        add_header X-Accel-Buffering "no" always;
        add_header Content-Type "text/event-stream" always;
    }

    # --- Mining ---
    location /api/v1/mining/ {
        # All rate limiting DISABLED
        proxy_pass http://qnk_api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "close";
        proxy_read_timeout 5s;
        proxy_send_timeout 5s;
        proxy_connect_timeout 3s;
        # MISSING: proxy_next_upstream error timeout http_503;
        # MISSING: proxy_next_upstream_timeout 3s;
        add_header Connection "close" always;
        add_header Access-Control-Allow-Origin "*" always;
    }

    # --- Block sync (300s timeout) ---
    location /api/v1/blocks { ... proxy_set_header Connection "close"; ... }
    location /api/v1/turbo-sync { ... proxy_set_header Connection "close"; ... }
    location /api/v1/state-sync { ... proxy_set_header Connection "close"; ... }

    # --- Cacheable endpoints (5s cache) ---
    location ~ ^/api/v1/(health|status|...|mining/stats|dex/pools) {
        proxy_cache api_cache;
        proxy_cache_valid 200 5s;
        proxy_cache_use_stale error timeout updating http_502 http_503;
        # NOTE: mining/challenge NOT included in cache regex
    }

    # --- WebSocket ---
    location /ws { ... Connection "upgrade"; ... }

    # --- AIOC proxy (port 3080) ---
    location /aioc/ { ... }

    # --- Security ---
    location ~ /\. { deny all; }
    location ~* \.(env|git|svn|htaccess)$ { deny all; }

    # SSL
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/quillon.xyz/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/quillon.xyz/privkey.pem;
}

# HTTP → HTTPS redirect
server { listen 80; ... return 301 https://...; }

# WSS proxy (port 9443 → 9002)
server { listen 9443 ssl; ... proxy_pass http://127.0.0.1:9002; ... }

# Internal stub_status (port 81)
server { listen 127.0.0.1:81; ... stub_status on; ... }
```

---

## 10. APPLICATION CODE — RELEVANT SECTIONS

### 10.1 Middleware stack (main.rs:20341-20352)
```rust
.layer(
    ServiceBuilder::new()
        .layer(tower::limit::ConcurrencyLimitLayer::new(5000))   // Global cap
        .layer(TraceLayer::new_for_http())
        .layer(tower_http::timeout::TimeoutLayer::new(Duration::from_secs(30)))
        .layer(CorsLayer::permissive())
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024)),
)
```

### 10.2 Mining concurrency cap (handlers.rs:8381-8400)
```rust
static MINING_IN_FLIGHT: AtomicU32 = AtomicU32::new(0);
const MAX_MINING_CONCURRENCY: u32 = 500;

let in_flight = MINING_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
if in_flight >= MAX_MINING_CONCURRENCY {
    MINING_IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    return Err(StatusCode::SERVICE_UNAVAILABLE);  // → 503
}
```

### 10.3 Mining sync gate (handlers.rs:8402-8416)
```rust
let local_h = state.current_height_atomic.load(Relaxed);
let raw_net_h = state.highest_network_height.load(Relaxed);
let max_reasonable = local_h + 5_000;
let net_h = if raw_net_h > max_reasonable { local_h } else { raw_net_h };
let allow_solo = env::var("Q_ALLOW_SOLO_MINING").map(|v| v == "true" || v == "1").unwrap_or(false);
if net_h > 0 && local_h + 10 < net_h && !allow_solo {
    return Err(StatusCode::SERVICE_UNAVAILABLE);  // → 503 if behind
}
// Epsilon has Q_ALLOW_SOLO_MINING=true → this gate is BYPASSED
```

### 10.4 Mining pipeline sharding (main.rs:6068-6099)
```rust
let num_mining_shards = num_cpus::get().min(64).max(4);  // 48 on Epsilon
let total_capacity = 1_000_000 + (48 - 16) * 50_000;     // = 2,600,000
let shard_capacity = 2_600_000 / 48;                      // = 54,166 per shard
// 48 shards × 54K buffer each = 2.6M total queue capacity
```

---

## 11. KEY QUESTIONS FOR DEEPSEEK

1. **Sync/Mining isolation:** What's the best way to prevent RocksDB batch writes during block sync from blocking mining request handlers? Options: separate tokio runtimes, read-replica, priority queues, or yield points in sync loops?

2. **ConcurrencyLimit behavior:** tower's `ConcurrencyLimitLayer` queues when full (doesn't reject). The 503s appear to come from the mining handler's own 500-cap. Should we remove the tower ConcurrencyLimit entirely and rely on per-handler caps?

3. **nginx → backend connection model:** We switched from `Connection ""` (keepalive) to `Connection "close"` to fix a 75K connection pileup. But this creates a TIME-WAIT socket per request. At 2,800 req/s × 15s = 42K TIME-WAIT. Is there a middle ground (e.g., keepalive with max_requests=10)?

4. **Mining challenge caching:** If we cache `/api/v1/mining/challenge` for 1s in nginx, does this cause duplicate work (multiple miners getting the same challenge and only one winning)?

5. **Adaptive concurrency:** What's the best pattern for dynamically adjusting `MAX_MINING_CONCURRENCY` based on current system load (CPU, sync status, RocksDB latency)?

6. **Multi-upstream with sticky mining:** Miners need to submit solutions to the same server that issued the challenge. With `proxy_next_upstream`, a failed submit could retry on Beta which doesn't know the challenge context. How to handle this?

---

## 12. SUMMARY

| Problem | Root Cause | Fix Priority |
|---------|-----------|--------------|
| **503 on mining submit** | App-level mining concurrency cap (500) hit during sync | P0 |
| **Single point of failure** | Only Epsilon serves traffic | P0 |
| **No challenge caching** | 2,800 challenge fetches/s hit backend | P0 |
| **SSE broken** | `Connection "close"` on SSE location | P1 |
| **Sync overwhelms mining** | Shared tokio runtime, aggressive sync params | P1 |
| **Swap thrashing** | 16GB RocksDB cache on 62GB RAM | P1 |
| **Delta 1.24M behind** | Old version, overwhelmed by sync | P2 |
| **Gamma on old version** | v8.9.6 vs v9.0.0 | P2 |
| **No rate limiting** | All limit_req/limit_conn disabled | P2 |
| **Root disk 94% full** | Epsilon / partition | P1 |
