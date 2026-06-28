# Quillon Graph — Addendum: Explorer "zero blocks" + SSE never-loads + slow balance push

> **Hand-off to Codex 5.5 / ChatGPT — continuation of `MASTER-DIAGNOSIS.md` /
> `REMEDIATION-codex.md` (2026‑06‑13).** Same host (Epsilon `89.149.241.126`,
> `quillon.xyz` → single A-record). Self-contained: every claim cited to `file:line`
> or a live read taken **2026‑06‑15 ~07:25 UTC**. Source tree of record on Beta:
> `/opt/orobit/shared/q-narwhalknight` (frontend `gui/quantum-wallet`, node
> `crates/q-api-server`); same tree mirrored on Epsilon at `/home/orobit/q-narwhalknight-src`.
>
> **Why a new file:** the master diagnosis closed on the death-loop / peering / fork /
> telemetry-unit lane. The operator now reports the node **runs more stably**, but two
> *new* symptoms remain, both in the **SSE / explorer-frontend lane** which the master
> doc did not cover: (1) the Explorer menu item shows **zero blocks** and **SSE data
> never loads**; (2) **balance over SSE updates too slowly** (operator wants 3–5×/s).
> These are largely a **frontend** problem with one node-side cadence lever — disjoint
> from, and shippable independently of, the Tier‑1 death-loop work.

## Status delta vs 2026‑06‑13 (live, 2026‑06‑15)
- Node is **up and `status:"ready"`**, height **18,554,297 → climbing** (`GET /api/v1/status`, peer_id `12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM`). Materially calmer than the 3–41‑min death-loop snapshot.
- **Backend SSE WORKS** — direct and through the proxy. Proven:
  - `curl -sN http://89.149.241.126:8080/api/v1/events` (no wallet) → emits `event: new-block` (e.g. h=18554303) + `: keep-alive`.
  - `curl -sN https://quillon.xyz/api/v1/events` → **also streams** new-block (h=18554306). **q-flux is NOT buffering SSE** — the "proxy eats SSE" hypothesis is **REJECTED**.
- `GET /api/v1/blocks/recent?limit=5` → **HTTP 200** with real `{success:true,data:[{height,dag_round,…}]}`. The explorer's REST block source is healthy server-side.

So: server, proxy, and REST are all fine. **The defects are in the browser SSE wiring and in broadcast cadence**, below.

---

## E‑1 — Explorer opens **no SSE stream at all** when no wallet is connected  🟢 FRONTEND · **ROOT CAUSE of "SSE never loads" / "no live blocks"**

> **✅ SHIPPED 2026‑06‑15 (Claude).** Fixed in `gui/quantum-wallet/src/services/sseManager.ts`
> `connect()`: when no wallet, it now opens a PUBLIC `EventSource` to `/api/v1/events`
> (no `wallet_address`, no auth) and upgrades to the per-wallet signed stream on login.
> Built on Beta (`npm run build`), rsynced to Epsilon `dist-final` (downloads/ preserved,
> no `--delete`). Verified live: `https://quillon.xyz` serves the new bundle
> (`index-DBFxwjSQ.js`) containing the "Connecting to PUBLIC stream" path. **Codex: do not
> redo E‑1.** Remaining E‑2/E‑3 below still open. Detail kept for the record:

**Problem.** The shared `sseManager` **refuses to open any `EventSource` unless a wallet
address exists**, but the Explorer is a public view that subscribes through that same
manager. With no wallet (logged-out, or `localStorage.walletAddress` empty) the explorer
never receives a single SSE event.

**Evidence (`file:line`).**
- `gui/quantum-wallet/src/services/sseManager.ts:125‑129`:
  ```js
  const walletAddress = state.walletAddress || localStorage.getItem('walletAddress') || '';
  if (!walletAddress) {
    console.log('[SSE Manager] No wallet address, deferring connection');
    return;                       // ← never opens EventSource
  }
  ```
  The only URL it ever builds is per-wallet: `…/api/v1/events?wallet_address=${walletAddress}${authParam}` (`sseManager.ts:151`).
- Explorer subscribes via that manager only — `gui/quantum-wallet/src/components/ExplorerScreen.tsx:3024‑3052` (`sseManager.on('node-status'…)`, `sseManager.on('mining_reward'…)`, `sseManager.on('new-block'…)`). Comment at `:3021` notes a prior second EventSource was removed to avoid duplicate connections — so this manager is the **only** live feed.
- The **backend already supports a public, unauthenticated stream**: with no `wallet_address`, `sse_events` forwards exactly the public allow-list — `NodeStatusUpdate | BlockFinalized | NewBlock | MetricsUpdate | TokenPriceUpdate | LiquidityPoolUpdate | NitroBoostsUpdate | ServerVersion | StateSyncComplete` (`crates/q-api-server/src/streaming.rs` — public branch in `is_event_relevant`, ~lines 808‑870). Proven live by the no-wallet `curl` above.

**Fix (🟢 LOW — display/transport only, no consensus/balance path).**
In `sseManager.connect()`, when `walletAddress` is empty, **open a PUBLIC EventSource to
`/api/v1/events` with no `wallet_address` and no `auth`** instead of early-returning.
Upgrade/reconnect to the per-wallet URL when a wallet later connects (the manager already
has a `reconnect()` at `:296`). All data the Explorer needs (`new-block`, `node-status`,
`metrics`) is in the public allow-list, so no auth is required.

**Verify.** Logged-out, Explorer height tile + recent-blocks live-append update on their
own; browser devtools shows an open `EventSource` to `/api/v1/events` (no `wallet_address`)
receiving `event: new-block`. (Server side already proven: the no-wallet `curl` streams.)

---

## E‑2 — Explorer "zero **blocks**" on initial render  🟢 FRONTEND · **SHIPPED 2026‑06‑15**

> **✅ SHIPPED 2026‑06‑15 (Claude). Real root cause found (two faults), E‑1 alone did NOT fix it:**
> 1. The explorer's height (`networkStats.currentHeight`, shown as the big "blocks"/height
>    number, init 0) was sourced from `qnkAPI.getNodeStatus()` → **`GET /api/v1/node/status`**,
>    which **HANGS** (HTTP 000, no response) — that handler "may wait for locks"
>    (`main.rs:25380` → `handlers::node_status`) and is a victim of the master **§N‑1b lock
>    contention**. Both the initial fetch and the 3 s `heightPoll` used it → height stuck at 0.
> 2. The SSE fallback handlers (`ExplorerScreen.tsx` node-status/mining_reward/new-block) read
>    `data.height` / `data.current_height` **directly**, but the SSE payload is
>    `{type, data:{…}}` — so they read `undefined` and never updated the height. (Inconsistent
>    with `Dashboard.tsx` which correctly does `const p = data?.data ?? data`.)
>
> **Fix (frontend, `ExplorerScreen.tsx`):** (a) `heightPoll` now uses the fast, always-
> responsive `qnkAPI.getRecentBlocks(1)` tip instead of the hanging `getNodeStatus`;
> (b) all 3 SSE handlers now unwrap `const p = data?.data ?? data`. Built + deployed to
> Epsilon `dist-final`; verified live: served `ExplorerScreen-4dqmlO_W.js` contains the
> `?.data??` unwrap + `getRecentBlocks`; tip read = 18,554,584. **Codex: do not redo E‑2.**
>
> **Note for Codex (node-side, NOT done):** `/api/v1/node/status` hanging is itself a real
> bug = master **§N‑1b**. The frontend now routes around it, but other callers
> (`main.rs:15065/21728/22769` peer gap-fill/bootstrap) still hit it — fixing the lock
> contention remains in the Tier‑1 plan.
>
> Original analysis kept for the record:

**What is NOT the cause.** The initial REST list works: `qnkAPI.getRecentBlocks(5)` →
`this.request('/v1/blocks/recent?limit=5')` (`api.ts:2086‑2088`), called from
`ExplorerScreen.tsx:2834` inside a `Promise.allSettled([... getRecentBlocks ...])`
(`:2833`), mapped at `:2850‑2858` guarded by `value.success && value.data`. Endpoint
returns 200 + correct shape (proven). So the **recent-blocks list** should populate from
REST regardless of SSE.

**Most-likely actual cause = E‑1.** The Explorer's headline block **counter / "live height"**
is driven by SSE (`new-block` / `node-status` raise `highestKnownHeightRef`,
`ExplorerScreen.tsx:3044‑3052`). With SSE dead (E‑1) the counter sticks at its initial
value → reads as "zero / not advancing." **Fixing E‑1 most likely fixes the visible
"zero blocks."**

**Two things for Codex to confirm in-browser (can't be seen from a shell):**
1. Whether the empty thing is the **list** (REST) or the **counter** (SSE). If the list is
   empty too, check `qnkAPI.request` base-URL resolution: `nodeUrl` from `localStorage`
   (`sseManager.ts:134` reads the same key) — if unset it must fall back to a **relative**
   `/api/...` so the call goes through q-flux at the apex. Test the REST path *through the
   proxy*: `curl -s https://quillon.xyz/api/v1/blocks/recent?limit=5` (I only confirmed
   `:8080` directly).
2. **Latent data oddity (don't let it mislead):** `/blocks/recent` returns `timestamp`
   as **10-digit unix seconds** (e.g. `1780717771`) ~7 days behind wall-clock, while the
   SSE `NewBlock.timestamp` is correct RFC3339. Same `_ms`/seconds + clock-source family
   as master **Tier‑0 T‑1/T‑2**. If the explorer renders/sorts/age-filters by that field
   it could *hide* blocks. Worth a glance; fix belongs with the Tier‑0 unit cleanup.

---

## E‑3 — Balance over SSE updates too slowly (operator wants 3–5×/s)  🟡 NODE-CADENCE + product decision

**How balance push works.** `StreamEvent::BalanceUpdated` is broadcast on several node
paths: local mining `main.rs:17524`, production `main.rs:21121` / `:21331`, plus
`:7233/:10632/:12352/:13075`. The browser receives them on the same per-wallet
`sseManager` stream (so E‑3 only matters once a wallet is connected; the public explorer
stream doesn't carry per-wallet balances by design — `streaming.rs` privacy hardening).

**What is NOT throttling it.**
- The only balance rate-limit is on the **P2P-ingest** path: `check_balance_update_rate_limit`
  (`main.rs:588`), applied at `main.rs:10433` (`[P2P BALANCE]`, keyed by `origin_node_id`).
  Epsilon has **peers=0** (master §1) → this path is inactive → it is **not** capping the
  operator's own balance pushes.
- `Q_SKIP_MINING_GOSSIP=1` only skips **gossipsub publish** (`main.rs:3405‑3424`), **not**
  the local `event_broadcaster` SSE path. So the death-loop env flags do **not** suppress
  SSE balance events.

**The real ceiling.** A `BalanceUpdated` only fires when the wallet's balance *actually
changes* (a mined reward or an inbound/outbound tx). Live, the SSE `new-block` cadence was
observed at **~1 event / 25–30 s**, while `/blocks/recent` shows blocks only seconds apart
— i.e. the **live broadcast lags real block production**, consistent with master **§N‑1c**
(durable `qblock:latest` contiguity gap ~768; broadcast riding the lagging
finalization/durable tick rather than the production hot-path).

**DECISION (Viktor, 2026‑06‑15): TRUE per-change events. Do NOT fabricate balances.**
The 3–5×/s target is to be met by emitting a real `BalanceUpdated` for **every actual
balance-changing event** as fast as it happens — never by a server/client "heartbeat" that
re-emits the same value. The "looks great" requirement is satisfied **purely in the UI
rendering layer** (animate between two *real* balances), not in the data.

1. **Node-side (the real lever — couples to master Tier‑1/N‑1c):** emit the operator-wallet
   `BalanceUpdated` on the **production hot path per block** (`main.rs:21121` region),
   **not** gated behind the lagging durable/finalization tick. Today the live SSE cadence
   lags real production (~1 event/25–30 s observed vs blocks seconds apart) precisely
   because the broadcast rides the lagging tick. Once N‑1c closes the durable gap and the
   broadcast fires on production, every real reward/tx for the wallet streams within <1 s —
   which IS 3–5×/s whenever the wallet is mining/transacting at that rate. No new event
   type, no synthetic values.
   - While auditing that path, confirm nothing **coalesces** multiple same-wallet
     `BalanceUpdated` within a block batch into one (we want one event per real delta);
     and that the local mining path `main.rs:17524` isn't dropped when
     `Q_SKIP_MINING_GOSSIP=1` (it isn't — that flag only skips gossipsub, `main.rs:3405‑3424`
     — but verify after any refactor).
2. **Frontend "se fedt ud" (UI-only, 🟢, no data fabrication):** make the balance tile
   **animate/count-up between consecutive REAL `BalanceUpdated` values** (tween
   `old_balance → new_balance` over ~150–300 ms with an easing curve; flash/pulse on
   change). The displayed number must always start and land on values that came from the
   stream — the animation only interpolates the *visual* transition, never invents a
   balance. If two real events arrive close together, cancel the in-flight tween and retween
   to the newest value (no queue build-up).
3. Cheap interim until N‑1c lands: the wallet may also poll `GET /api/v1/balance`
   (authoritative RocksDB read, same source the SSE initial-balance uses,
   `streaming.rs:996‑1006`) as a fallback — but the **canonical** mechanism is the
   per-change SSE event above.

**Verify.** With a connected, actively-mining wallet: one `BalanceUpdated` per produced
block (post‑N‑1c), each within <1 s of the on-chain change; the UI tweens smoothly between
each pair of real values; **no** event ever carries `old_balance == new_balance` from a
heartbeat (those must not exist).

---

## Ordering / risk
- **E‑1 is independent and 🟢** — pure frontend (`sseManager.ts`), no consensus/balance/
  storage path, no Docker soak. **Ship it first**; it most likely also resolves E‑2's
  visible "zero blocks." Build the frontend per repo rules (vite, not `npm run` per repo
  convention; deploy to `dist-final` on Epsilon — the apex serves from there).
- **E‑2** = confirm-in-browser + possibly the same base-URL/timestamp Tier‑0 cleanup.
- **E‑3** node-side cadence couples to master **Tier‑1 / N‑1c** (already 🔴/🟠 with soak).
  **Product decision is made (Viktor 2026‑06‑15): true per-change events, no heartbeat;
  the "looks great" requirement is a 🟢 UI-only animation between real values.** The
  frontend tween (E‑3 item 2) can ship immediately and independently; the node-side cadence
  lands with N‑1c.

## Verify-loop (read-only)
```bash
# Public SSE (what the logged-out explorer SHOULD consume — proven to stream):
curl -sN http://89.149.241.126:8080/api/v1/events | head        # backend
curl -sN https://quillon.xyz/api/v1/events | head               # through q-flux (not buffered)
# Explorer initial block list source (proven 200 + data on :8080; confirm via proxy too):
curl -s  https://quillon.xyz/api/v1/blocks/recent?limit=5 | head -c 400
# Per-wallet balance fallback / SSE initial-balance source:
curl -s  http://89.149.241.126:8080/api/v1/balance?wallet_address=<qnk…>
```

*Provenance: all live reads 2026‑06‑15 ~07:25 UTC from this session; `file:line` cites
against Beta `/opt/orobit/shared/q-narwhalknight`. Builds on `MASTER-DIAGNOSIS.md` /
`REMEDIATION-codex.md` (do not re-derive the death-loop/peering/fork lane — it's covered
there). Outstanding from the master doc is unchanged.*
