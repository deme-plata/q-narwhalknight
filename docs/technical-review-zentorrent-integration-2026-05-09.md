# Technical Review: ZenTorrent Integration into Q-NarwhalKnight Node
**Date**: 2026-05-09  
**Author**: Server Beta  
**Status**: Design — pending implementation

---

## 1. Executive Summary

ZenTorrent (`/opt/orobit/shared/ZenTorrent/`) is a fully-functional Rust BitTorrent client
with a Warp REST API (port 3040) and a React frontend. Its feature set covers everything
needed to:

- Distribute Q-NarwhalKnight node binaries as torrents (redundant, bandwidth-efficient)
- Let users seed downloaded content back to the network
- Provide admin-gated torrent management inside the quantum wallet UI

This review documents what ZenTorrent currently does, what integration work is required, and
what the Torrent admin tab in the wallet needs to connect to.

---

## 2. ZenTorrent Feature Inventory

### 2.1 Core Download Engine

| Module | File | Status |
|--------|------|--------|
| BitTorrent protocol (v1) | `torrent/torrent.rs` | Complete |
| DHT (Kademlia) | `torrent/dht.rs` | Complete |
| Peer Exchange (PEX) | `torrent/pex_enhanced.rs` | Complete |
| Metadata exchange (BEP-10) | `torrent/metadata_exchange.rs` | Complete |
| Magnet link resolution | `torrent/magnet.rs` | Complete |
| Fast-resume (BEP-6) | `torrent/fast_resume.rs` | Complete |
| Endgame mode | `torrent/endgame_mode.rs` | Complete |
| Piece selection | `torrent/piece_selector.rs` | Complete |
| Bandwidth limiter | `torrent/bandwidth_manager.rs` | Complete |
| MSE/PE stream encryption | `network/mse.rs` | Complete |
| uTP (µTorrent Transport Protocol) | `torrent/utp.rs` | Complete |
| Port forwarding (UPnP/NAT-PMP) | `torrent/port_forwarding.rs` | Complete |

### 2.2 REST API (Warp, port 3040)

All endpoints are under `/api/`:

| Method | Path | Function |
|--------|------|----------|
| GET | `/api/torrents` | List all torrents + stats |
| GET | `/api/torrents/:hash` | Single torrent details |
| POST | `/api/torrents` | Add by `.torrent` file (multipart) |
| POST | `/api/torrents/magnet` | Add by magnet URI |
| DELETE | `/api/torrents/:hash?delete_data=bool` | Remove torrent |
| POST | `/api/torrents/:hash/start` | Start downloading |
| POST | `/api/torrents/:hash/pause` | Pause |
| POST | `/api/torrents/:hash/resume` | Resume |
| POST | `/api/torrents/:hash/stop` | Stop |
| GET | `/api/torrents/:hash/files` | File list with progress |
| POST | `/api/torrents/:hash/file-priorities` | Set per-file priority |
| GET | `/api/torrents/:hash/peers` | Connected peers |
| GET | `/api/settings` | Current settings |
| PUT | `/api/settings` | Update settings |
| WS  | `/api/ws` | Real-time updates (WebSocket) |

### 2.3 WebSocket Events

The `/api/ws` stream emits:

```
TorrentAdded(TorrentInfo)    — new torrent registered
TorrentRemoved(info_hash)    — torrent deleted
TorrentUpdated(TorrentInfo)  — progress, speed, peer count updated
TorrentCompleted(info_hash)  — download finished
StatsUpdate(ClientStats)     — global download/upload totals
Error(String)                — async error
```

### 2.4 TorrentInfo Model

```
info_hash, name, status, download_state, size, progress (0.0–1.0),
download_rate (bytes/s), upload_rate (bytes/s), downloaded, uploaded,
peers_connected, peers_total, seeds_connected, seeds_total,
pieces_completed, pieces_total, eta (seconds), added_date, completed_date,
download_dir
```

### 2.5 Settings

```
download_dir, max_connections, max_download_rate (KB/s, optional cap),
max_upload_rate (KB/s, optional cap), port (random 16000–60000),
dht_enabled, pex_enabled, encryption_mode, tor_enabled, tor_socks_addr
```

### 2.6 Tor Support

ZenTorrent has Tor integration via `network/tor.rs` using the system SOCKS5 proxy
(`127.0.0.1:9050`). Disabled by default, enabled via `Settings.tor_enabled = true`.

---

## 3. Integration Architecture

### 3.1 Process Model (Sidecar)

ZenTorrent runs as a separate process alongside `q-api-server`. They share the same host but
have independent lifecycles. There is no Cargo workspace dependency between them.

```
┌─────────────────────────────────────────────┐
│  VPS / Server Beta / Epsilon                │
│                                             │
│  ┌──────────────────┐  ┌────────────────┐   │
│  │  q-api-server    │  │  zentorrent    │   │
│  │  port 8080       │  │  port 3040     │   │
│  └──────────────────┘  └────────────────┘   │
│           │                    │            │
│           └──────────────┬─────┘            │
│                          ▼                  │
│              nginx / q-flux (port 443/80)   │
│              /api/*    → :8080              │
│              /torrent-api/* → :3040         │
└─────────────────────────────────────────────┘
```

### 3.2 Required nginx/q-flux Proxy Rule

Add to nginx site config (Beta) or q-flux routes (Epsilon):

**nginx (Beta):**
```nginx
location /torrent-api/ {
    proxy_pass http://127.0.0.1:3040/api/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

**q-flux (Epsilon) — add to routes config:**
```toml
[[routes]]
prefix = "/torrent-api/"
upstream = "http://127.0.0.1:3040/api/"
strip_prefix = true
```

The wallet frontend calls `/torrent-api/torrents` which nginx rewrites to
`http://127.0.0.1:3040/api/torrents`. This eliminates cross-origin and mixed-content issues.

### 3.3 ZenTorrent Startup (systemd service or manual)

```bash
# Add to server startup or a separate systemd unit:
cd /opt/orobit/shared/ZenTorrent/torrent-backend
RUST_LOG=info ./target/release/torrent-backend &

# Or build and run:
cargo build --release --package torrent-backend
./target/release/torrent-backend
```

Environment defaults: `DOWNLOAD_DIR=./downloads`, listens on `127.0.0.1:3040`.

---

## 4. Wallet UI Integration

### 4.1 Admin-Gated Torrent Tab

The Torrent tab is only shown when the logged-in wallet matches the master wallet:

```
MASTER_WALLET = efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723
```

This check is identical to the Deploy Control Panel (`DeployControlPanel.tsx:1485`).

### 4.2 TorrentTab.tsx Component Plan

Sections:
- **Software Distribution** panel — hardcoded list of q-api-server / q-miner torrents with
  "Seed" buttons. Creates torrents from existing downloaded binaries.
- **Active Torrents** table — live list from `GET /torrent-api/torrents`, auto-refreshed via
  WebSocket. Columns: name, size, progress bar, speed ↓/↑, ETA, peers, actions.
- **Add Torrent** toolbar — two buttons: "Add .torrent file" (file picker) and "Add magnet
  link" (text input dialog).
- **Settings** panel — bandwidth caps, connection limit, download directory display.

### 4.3 WebSocket URL

```
ws://[same-host]/torrent-api/ws
```

With the proxy rule in place the WebSocket upgrade works transparently.

---

## 5. Binary Distribution Use Case

### Current State (Single-Point-of-Failure)

```
User → wget https://quillon.xyz/downloads/q-api-server-v10.7.6
```
All downloads go through Epsilon's `/home/orobit/q-narwhalknight/dist-final/downloads/`.
If Epsilon is offline or the 1.8TB partition fills, downloads break.

### With ZenTorrent (Redundant P2P Distribution)

```
User → download q-api-server.torrent from quillon.xyz
     → BitTorrent swarm (Epsilon seeds + any other node seeds)
     → integrity guaranteed by info_hash
```

Steps to publish a binary release as a torrent:
1. Put the binary in ZenTorrent's download dir
2. `POST /torrent-api/torrents` with the file → ZenTorrent generates the torrent file +
   starts seeding
3. Store the resulting `.torrent` file and magnet link in `dist-final/downloads/`
4. Users download the `.torrent` file from `quillon.xyz`, load it in any BitTorrent client
   or paste the magnet link into the wallet Torrent tab

### Integrity

BitTorrent's SHA-1 piece verification (v1 torrents) or SHA-256 (v2/hybrid) means every
downloaded byte is verified — stronger than a plain `wget` without checksum verification.

---

## 6. Gap Analysis: What ZenTorrent Still Needs for Production

| Gap | Severity | Effort |
|-----|----------|--------|
| No authentication on port 3040 API | HIGH | 1 day — add bearer token header check in Warp filter |
| `torrent-backend` binary has no version flag | LOW | 30 min |
| No graceful shutdown signal handler | MEDIUM | 2 hours |
| WebSocket auth (exposed WS) | HIGH | 1 day — same auth token as REST |
| Per-user download dirs | LOW | not needed for admin-only use |
| systemd service unit | MEDIUM | 1 hour |
| Tracker announce for seeding | LOW | built-in, needs tracker URL list |
| ZenTorrent compiled for Debian 12 | MEDIUM | 1 build with `rust:bookworm` Docker image |

**Critical path**: Add API auth (token) before exposing `/torrent-api/` on nginx. Without
auth, anyone can POST magnet links to port 3040.

Simplest auth: shared secret via `X-Torrent-Token` header. The wallet includes this header
from a config constant; nginx can also validate it before forwarding.

---

## 7. Recommended Implementation Order

1. **Now**: Add nginx proxy rule + basic `X-Torrent-Token` auth in Warp routes
2. **Now**: Implement `TorrentTab.tsx` in wallet (admin-gated)
3. **Day 1**: Build ZenTorrent for Debian 12 / Epsilon; write systemd service unit
4. **Week 1**: Publish first binary torrent (v10.7.6 q-api-server)
5. **Week 2**: Add magnet link to `DownloadNodeScreen` for public users

---

## 8. Test Plan

- [ ] `POST /torrent-api/torrents/magnet` with Arch Linux magnet link → torrent appears in list
- [ ] WebSocket emits `TorrentUpdated` events with progressing `progress` field
- [ ] Pause/resume cycle returns torrent to previous progress
- [ ] Delete with `delete_data=false` removes metadata only
- [ ] Auth header required: request without `X-Torrent-Token` → 401
- [ ] Wallet admin tab visible when logged in as master wallet
- [ ] Wallet torrent tab NOT visible for any other wallet
- [ ] Seeded binary: download via magnet, verify SHA-256 matches original file
