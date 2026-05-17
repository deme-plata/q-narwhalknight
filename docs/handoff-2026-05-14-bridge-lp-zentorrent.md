# Handoff — 2026-05-14
## v10.9.21 honest-liquidity LP flow + ZenTorrent direction

This is a session handoff. Pick up exactly where this leaves off.

---

## TL;DR — state on disk right now

- **Working tree on Beta** (`/opt/orobit/shared/q-narwhalknight/`, branch `feature/safe-batched-sync-v1.0.2`) contains all of v10.9.21 uncommitted.
- **`cargo check --package q-api-server`** completed cleanly on **2026-05-14** in `bhbbv1ub3.output` ("Finished `dev` profile in 66m 18s", 0 errors). The code as it stands compiles.
- **Nothing is deployed.** Production Beta is still running v10.9.15, still crash-looping on the RocksDB lock held by the TUI process (PID 1197979 on port 62003 — see earlier session). That blocker has not been touched.

---

## Commits already on the branch

| Commit | Title |
|---|---|
| `3abf7d12` | feat(bridge+dex): make QUG/wBTC AMM pool work without configured Bitcoin RPC |
| `3b601969` | chore: bump workspace 10.9.19 → 10.9.20 + rebadge bridge-fix log strings |

`git update-server-info` was run so Epsilon can HTTP-pull these.

---

## Working tree (uncommitted, ready to commit as v10.9.21)

### Backend
- `Cargo.toml` — workspace `version = "10.9.20"` → **`"10.9.21"`** with full changelog string.
- `crates/q-api-server/src/lib.rs` — `bootstrap_bridge_pools` no longer mints fake wBTC. Pools start as empty shells (`reserve0 = reserve1 = 0`). Legacy fake-seeded reserves drain once on upgrade (detected via `lp_token_supply > 0 && provider == [0u8;32]`). `qug_price` param renamed `_qug_price` since it's no longer used for seeding.
- `crates/q-api-server/src/handlers.rs` — swap path adds **empty-pool guard for bridge tokens** right after pool resolution (~line 11100): if matched pool has `reserve0==0 || reserve1==0` and either side is a bridge token, return an actionable error pointing the user to the LP flow. The v10.9.20 `use_oracle` gate is preserved.
- `crates/q-api-server/src/bitcoin_lp_api.rs` — **NEW**, ~1000 lines. Self-contained module:
  - `LpIntent` + `LpIntentStatus` types
  - Storage layout in CF_MANIFEST: `btc_lp_intent:<id>`, `btc_lp_intent_by_wallet:<wallet>:<id>`, `btc_lp_intent_by_deposit:<dep>`
  - Endpoints:
    - `POST /api/v1/bitcoin/lp/intent` — escrow QUG, generate Knots deposit address
    - `GET /api/v1/bitcoin/lp/intent/:id` — single status
    - `GET /api/v1/bitcoin/lp/intents` — list user's intents
    - `POST /api/v1/bitcoin/lp/intent/:id/cancel` — refund escrowed QUG
    - `POST /api/v1/bitcoin/lp/intent/:id/finalize` — verify confs, mint wBTC, run internal `mint_lp_first_or_proportional` (matches `liquidity_api::generate_lp_token_address` SHA-256 convention so LP tokens are fungible), mark deposit minted, status → Completed
  - Uses `BalanceStorage::set_balance` (not `save_wallet_balance`) for the hex-string API.
- `crates/q-api-server/src/main.rs` — five new routes registered under `/api/v1/bitcoin/lp/...` (just above the Zcash bridge block).

### Frontend
- `gui/quantum-wallet/src/components/BitcoinSwapModal.tsx` — new **"Bridge LP"** tab with:
  - Pool reserve card (live `reserve0`/`reserve1` from `/api/v1/defi/dex/pools`, USD valuations, "Empty — awaiting first LP" badge when `lp_supply == 0`).
  - One-click wizard: BTC amount input → auto-computed paired QUG at oracle rate → "Lock QUG & Get BTC Address" button → polls intent list → per-intent row with cancel + "Claim LP tokens" actions.
  - Status helper `lpStatusLabel` handles the tagged-enum form `{kind: "...", ...}`.
- `gui/quantum-wallet/src/services/api.ts` — five new client methods: `createLpIntent`, `listLpIntents`, `getLpIntent`, `cancelLpIntent`, `finalizeLpIntent`.

### Docs
- `CLAUDE.md` — new "ALWAYS COMPILE / CARGO-CHECK ON EPSILON DOCKER (Debian 12), NEVER ON BETA!" block added at top of `#### COMPILATION & BUILD REQUIREMENTS` (rule of the day from this session). Plus a follow-up patch to the dev-mode `cargo` examples noting the same.

### Untouched (other people's in-flight work, leave alone)
- `Cargo.lock`, `crates/q-api-server/src/{block_producer,integrity_api,main}.rs` engine-pulse stuff
- `crates/q-consensus-guard/`, `crates/q-crypto-simd/`, `crates/q-ivc/`, `crates/q-network/`, `crates/q-storage/`, `crates/q-tui/`
- `crates/q-storage/src/balance_smt.rs` — still untracked; pre-existing 743-line WIP. We patched its E0505 borrow-checker bug in an earlier commit but the file itself is the user's work, not ours.
- All `gui/quantum-wallet/src/components/{DexScreen,LoginScreen}.tsx`

---

## Verification status

| Check | Result |
|---|---|
| `cargo check --package q-bitcoin-bridge` | ✅ 0 errors, 70 warnings (5m05s) |
| `cargo check --package q-api-server` | ✅ 0 errors (66m18s, ran 2026-05-14, output `/tmp/claude-0/.../bhbbv1ub3.output`) |
| `tsc -p tsconfig.app.json` | ✅ 0 errors in `BitcoinSwapModal.tsx` and `services/api.ts` |

---

## Next step when picking up

### 1. Commit v10.9.21 (5 min)

The hunks are mixed with unrelated in-flight work in `handlers.rs`, `lib.rs`, `main.rs`, so use the same `git add -p` patch-selection pattern we used for `3abf7d12`. Pseudo-script:

```bash
# Files that are entirely mine — stage whole
git add crates/q-api-server/src/bitcoin_lp_api.rs \
        gui/quantum-wallet/src/components/BitcoinSwapModal.tsx \
        gui/quantum-wallet/src/services/api.ts \
        CLAUDE.md

# Cargo.toml — only stage the version-bump hunk (use `s` to split if needed, then y/n).
# handlers.rs — stage only the empty-pool guard hunk near line 11100 (after pool_id resolution)
# lib.rs — stage the bootstrap_bridge_pools rewrite (two loops: QUG bridge pools, QUGUSD bridge pools)
# main.rs — stage only the route block under "// v10.9.21: One-click ..."

# Then:
git commit -m "feat(bridge): honest liquidity + one-click LP intent (v10.9.21)" -m "..."
git update-server-info
```

Commit body template:

```
v10.9.21 makes wBTC AMM swaps honest and ships a user-friendly LP onboarding flow.

bootstrap_bridge_pools (lib.rs)
  Stop minting unbacked wBTC at boot. Pools start as empty shells; real LPs
  must deposit. Legacy fake-seeded reserves are drained once on upgrade
  (detected via lp_token_supply > 0 && provider == [0u8;32]).

handlers.rs
  Add empty-pool guard for bridge tokens after pool resolution. Returns
  "No wBTC liquidity yet — be the first LP" instead of silently trading
  against zero reserves.

bitcoin_lp_api.rs (new)
  POST /api/v1/bitcoin/lp/intent     — escrow QUG, generate BTC deposit address
  GET  /api/v1/bitcoin/lp/intent/:id — status
  GET  /api/v1/bitcoin/lp/intents    — list mine
  POST /api/v1/bitcoin/lp/intent/:id/cancel   — refund QUG
  POST /api/v1/bitcoin/lp/intent/:id/finalize — verify confs, mint wBTC,
                                                 add liquidity, mint LP tokens

Frontend: new "Bridge LP" tab in BitcoinSwapModal with the wizard, real-time
pool reserve display, oracle-paired QUG suggestion, and per-intent actions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

### 2. Open questions to decide before further work

The user gave a clear direction at end of session but didn't pick which order:

1. **Commit v10.9.21 first?** (recommended yes)
2. **Auto-torrent feature: ZenTorrent or q-flux?** I argued for ZenTorrent (it already has bencoding + piece hashing); user is open to either.
3. **Order: ZenTorrent production-wiring first, or auto-torrent module first?**

The user's exact words: *"a speical feature for the node operatores is the zentorrent feature in dashboard tab torrent. how far are we with that feature. also i qout like q-flux to enable a feature tht is turn every folder with for instance album or movie in to a torrent file automatically and dynamically that eveeryone can seed and leech with zentorrent"*

### 3. ZenTorrent status (already analyzed in session, summarized here)

**Backend:** `/opt/orobit/shared/ZenTorrent/torrent-backend/` — full BitTorrent stack (DHT, PEX, MSE, uTP, NAT, fast-resume, endgame). REST API on :3040. Binary at `target/release/torrent-backend` dated Apr 4. NOT git-tracked — it's a sibling shared dir.

**Frontend:** `gui/quantum-wallet/src/components/TorrentTab.tsx` (437 lines), admin-gated, merged in commit `27d21362`.

**Live state:** **not running anywhere.**

- No listener on :3040 on Beta
- No `/torrent-api/` location in `/etc/nginx/`
- No `[[routes]]` for `/torrent-api/` in `crates/q-flux/q-flux-epsilon.toml`
- No systemd service unit
- No API auth (HIGH severity gap per `docs/technical-review-zentorrent-integration-2026-05-09.md`)
- No binary ever published as a torrent

**Minimum-to-live checklist** (from the design doc):
1. Build ZenTorrent for Debian 12 using `rust:bookworm` Docker on Epsilon
2. Write `/etc/systemd/system/zentorrent.service` with `ZENTORRENT_TOKEN` env var
3. Add `/torrent-api/` proxy block in nginx (Beta) + `[[routes]]` in `q-flux-epsilon.toml`
4. Add `X-Torrent-Token` Warp filter to torrent-backend (~20 lines)
5. Inject the same token from `TorrentTab.tsx` (read from wallet setting)

### 4. Auto-torrent feature design (from session)

User wants: any folder dropped into a configured seed dir → automatic torrent file → everyone seeds/leeches.

**Recommended architecture (in ZenTorrent, not q-flux):**

- New file `torrent-backend/src/torrent/auto_seed.rs`
- `AutoSeedWatcher` using `notify` crate (debounce 30s after last write so partial album rips don't churn)
- On startup: scan `auto_seed_dir`, build/load `.torrent` per subfolder, add each to torrent manager in seeding state
- On folder create/modify/delete: regenerate / remove
- New endpoint `GET /api/auto-seeded` → array of `{ name, info_hash, magnet_uri, torrent_url, size, file_count, peers, added_date }`
- Settings extension: `auto_seed_dir`, `auto_seed_debounce_secs`, `tracker_announce_urls` (optional — pure DHT works)

**q-flux side** — new module `crates/q-flux/src/seed_index.rs`:
- Route `/seed/` → render HTML listing (pulls from `http://127.0.0.1:3040/api/auto-seeded`)
- Each row: cover thumbnail (if `cover.jpg`/`folder.jpg` exists in the subfolder), name, size, file count, leecher/seeder counts, magnet button, `.torrent` download
- `/seed/*.torrent` proxy to `/torrent-api/auto-seeded/<info_hash>.torrent`

**Frontend TorrentTab.tsx** — add "Public Library" subsection mirroring `/seed/` with operator controls.

**Nice extras (later):**
- Web seed (BEP-19) — q-flux serves raw files at `/seed/raw/<folder>/` for cold-start clients
- Signed torrents — sign `.torrent` with master wallet's Ed25519 key
- Seed-to-earn — small QUG drip to wallets that prove they're seeding

**Estimated effort:** ~5 days focused.

---

## Runtime blockers still outstanding (none resolved this session)

From the earlier session:

1. **Beta crash-loop** — q-api-server v10.9.15 systemd service crash-looping every ~30s on `IO error: While lock file: ./data-mainnet-genesis/hot/LOCK`. The lock is held by another q-api-server process running in TUI mode on port 62003 (PID was 1197979 in the earlier session — verify with `pgrep -f "q-api-server.*--tui"` before acting). User preference was **SIGTERM, wait for graceful exit** but I never executed it.

2. **Bitcoin Knots on Delta** uses rpcauth (hashed). To make the bridge actually do BTC↔wBTC mint/redeem, we still need plaintext `BTC_RPC_USER` / `BTC_RPC_PASS` in `/.env` on Beta. User preference was **generate a new rpcauth entry on Delta and give plaintext to me**; I never executed it.

3. **Deploy** — user explicitly chose "commit locally, don't deploy" at end of last task, so no `ha-deploy.sh` runs.

---

## Build / cargo-check rule (now in CLAUDE.md)

**Compile and `cargo check` ONLY inside Epsilon's `rust:bookworm` Docker.** Beta is a live mainnet bootstrap node — running multi-hour rustc on it steals CPU/RAM from block production. The CLAUDE.md update from this session enforces this. Source repo on Epsilon: `/home/orobit/q-narwhalknight-src/`, persistent target cache: `/home/orobit/target-debian12/`.

Sync workflow:
```bash
# Edit on Beta → commit + push to local git server → pull on Epsilon → cargo check on Epsilon Docker
git add ...; git commit -m ...; git update-server-info
ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && git pull origin <branch>"
# Or rsync a single file for fast iteration (no commit).
```

The exception we made this session — running the long `cargo check` on Beta — was a mistake. Do NOT repeat. Use Epsilon Docker.

---

## File map (where to look first when picking up)

| Need to | Read |
|---|---|
| Re-orient on v10.9.21 changes | `crates/q-api-server/src/bitcoin_lp_api.rs` (the new module — read top to bottom) |
| See the empty-pool guard | `crates/q-api-server/src/handlers.rs` around line 11100 |
| See the LP frontend wizard | `gui/quantum-wallet/src/components/BitcoinSwapModal.tsx` — search for `tab === 'lp'` |
| ZenTorrent design + gaps | `docs/technical-review-zentorrent-integration-2026-05-09.md` |
| ZenTorrent frontend | `gui/quantum-wallet/src/components/TorrentTab.tsx` |
| Build rules | `CLAUDE.md` § COMPILATION & BUILD REQUIREMENTS (start of section) |
| Earlier session context | This session's commits `3abf7d12` and `3b601969` |
