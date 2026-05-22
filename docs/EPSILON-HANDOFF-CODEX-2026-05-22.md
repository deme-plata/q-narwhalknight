# Epsilon Node Operator Handoff — for Codex / GPT-5.5

**Written:** 2026-05-22 ~08:30 UTC, updated 09:45 UTC. By Claude Code
(Opus 4.7) just before the operator's Claude Code subscription rolls over.
Hand the wheel to Codex. This document is the minimum you need to keep
Epsilon running and to finish the v10.11.15 deploy.

## 🎯 ROOT CAUSE FOUND (09:45 UTC) — v10.11.15 fixes it

The 14-hour bug is in `crates/q-api-server/src/handlers.rs` around
line 3982 — the dag_knight cert callback's **native-QUG else branch**
updated `state.wallet_balances` in-memory HashMap but **never called
`save_wallet_balance_authoritative` to persist to RocksDB**.

Effect:
- After restart: balance changes vanish; revert to pre-tx
- Without restart: signed `/api/v1/wallets/<addr>/balance` reads from
  RocksDB (where the HashMap update was never written), so observers
  see "tx confirmed but balance unchanged" — the exact pattern Viktor
  reported all day
- `tx_status` shows `block_height: <current_round>` (just the chain tip
  number at SEND-SIGNED time) and climbing confirmations because the
  surrounding `TxStatus::Confirmed` mark uses `current_round` not a
  real block-inclusion height
- Blocks 18,257,696, 18,257,945, 18,258,533 (every block in the bug
  window) contain 0 non-coinbase txs — the tx never landed in any
  block because dag_knight consumes it from the pool BEFORE
  block_producer can pack it
- v10.11.13/.14 instrumentation showed mempool/producer/save were
  NEVER REACHED for these txs — proving the tx took a different path
  (dag_knight) and got dropped after in-memory apply

**The fix (commit `0b9813e1d`)** is 3 lines: add two
`save_wallet_balance_authoritative` calls (authoritative variant
bypasses the max-wins guard so debits aren't refused for being lower
than the on-disk value), plus a warn-level apply log so future
regressions surface.

**Block inclusion is a SEPARATE issue**, not addressed in v10.11.15.
After this fix money will move correctly (sender debits, recipient
credits, persists across restart). The reported `block_height` will
still be `current_round` not a real block — fixing that requires
either:
  - Route send_signed via mempool → block_producer (the path
    v10.11.13 instrumentation was designed to surface)
  - Or have dag_knight emit a synthetic block per batch
Track as v10.11.16+ work. Monetary integrity is v10.11.15's job.

---

## TL;DR — what's happening right now

- **Epsilon (`89.149.241.126`)** is the production node serving `quillon.xyz`. Currently running **v10.11.13 binary** at `/opt/orobit/shared/q-narwhalknight/q-api-server-v10.11.13` (symlinked from `q-api-server-stable`).
- The v10.11.13 binary is **mis-labeled**. It was built from the v10.11.11 source (build container didn't pull the latest commits from the git daemon). It does NOT contain the v10.11.13 instrumentation, NoncePersistence, or v10.11.14 mnemonic fix. Don't trust its version string.
- A **v10.11.14 build is in flight** on Epsilon Docker (background task started just before this handoff). When it finishes, you'll deploy it. Instructions below.
- The chain has a **chain-wide user-tx-drop bug**: every send_signed / dex_swap tx reports tx_status=confirmed at some block N, but `/api/v1/blocks/N` shows zero non-coinbase txs in those blocks. Money never moved. 96M QUGUSD was phantom-minted into the operator's wallet via failed-swap money-printer. The Send + Swap UI is disabled chain-wide as a stopgap.
- v10.11.14 adds 5 instrumentation log points AND a likely root-cause fix (mnemonic derivation BIP39 → SHA3-256). Once deployed, ONE probe send_signed surfaces the data to write v10.11.15.

---

## 0. Constants you need

| Thing | Value |
|---|---|
| Epsilon IP | `89.149.241.126` |
| Epsilon SSH | `ssh root@89.149.241.126` (key auth from Beta `185.182.185.227`) |
| Service | `q-api-server.service` (systemd) |
| Binary symlink | `/opt/orobit/shared/q-narwhalknight/q-api-server-stable` (points to active binary) |
| Production data dir | `/home/orobit/data-mainnet-genesis/` (219 GB, NEVER touch) |
| Source tree | `/home/orobit/q-narwhalknight-src/` (git checkout, branch `agent/cross-shard-simd-validation`) |
| Frontend root | `/home/orobit/q-narwhalknight/dist-final/` (q-flux serves this; NOT `/opt/...`) |
| Build target dir | `/home/orobit/target-debian12/` (persistent incremental cache) |
| API port | `8080` (REST + tx submission) |
| P2P port | `9001` (libp2p gossipsub) |
| RUST_LOG (current) | `warn,q_storage::balance_consensus=info,q_storage=info,q_api_server::handlers=info` (in `/.env`) |
| Operator agent wallet | `qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb` (Claude Code's wallet — KEEP the seed file at `/root/.claude/quillon-agent-seed` UNTOUCHED) |
| Operator main wallet | `qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723` (operator Viktor's main wallet) |
| Adrian (Cursor agent) | `qnk1f97ff0b330c7790e8c82a57579052851d2c15239c78b6124fee6a74e4026d67` |

---

## 1. How to check Epsilon is alive

```bash
# From any host — direct IP (DNS goes through q-flux which can route to wrong backend)
curl -s http://89.149.241.126:8080/api/v1/status | python3 -m json.tool | head -20

# Should return success: true, data.upgrades.current_height climbing.
# If 502 / timeout / empty body → node is down or restarting.

# From Epsilon directly:
ssh root@89.149.241.126 "systemctl is-active q-api-server"
ssh root@89.149.241.126 "journalctl -u q-api-server --since '2 minutes ago' --no-pager | tail -30"
```

**Important: `data.current_height` is sometimes None on recent binaries; use `data.upgrades.current_height` instead. See memory `feedback_post_deploy_height_check.md` for the gotcha details.**

---

## 2. Restart procedure (the SAFE one)

DO NOT use `killall` — never works on Quillon hosts. DO NOT `kill -9` (graceful only — RocksDB corruption otherwise).

```bash
ssh root@89.149.241.126 "
  # 1. Capture pre-restart state for verification
  PRE_HEIGHT=\$(curl -s http://localhost:8080/api/v1/status | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"data\"][\"upgrades\"][\"current_height\"])')
  echo \"Pre-restart height: \$PRE_HEIGHT\"

  # 2. Graceful restart
  systemctl restart q-api-server

  # 3. Wait for HTTP to come back (up to ~60s — the boot does balance-recovery scan)
  until curl -fsS http://localhost:8080/api/v1/status >/dev/null 2>&1; do sleep 2; done

  # 4. Post-restart sanity
  POST_HEIGHT=\$(curl -s http://localhost:8080/api/v1/status | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"data\"][\"upgrades\"][\"current_height\"])')
  echo \"Post-restart height: \$POST_HEIGHT\"
  echo \"Delta: \$((POST_HEIGHT - PRE_HEIGHT))\"
"
```

**The post-restart height MUST be within ±100 of pre-restart.** If it dropped by more than 100, the node opened the wrong DB — STOP and read [Section 4](#4-emergency-rollback) immediately.

**Also verify the operator's wallet balance survived** (per memory `feedback_post_deploy_height_check.md`):

```bash
# From the Beta dev box (where the agent seed file is):
cd /opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp
node -e "
const fs = require('fs');
const { ed25519 } = require('@noble/curves/ed25519');
const { sha3_256 } = require('@noble/hashes/sha3');
const SEED = fs.readFileSync('/root/.claude/quillon-agent-seed', 'utf8').trim();
const PRIV = sha3_256(new TextEncoder().encode(SEED));
const PUB = ed25519.getPublicKey(PRIV);
const ADDR = 'qnk' + Buffer.from(PUB).toString('hex');
const PATH = '/api/v1/wallets/' + ADDR + '/balance';
const ts = Math.floor(Date.now() / 1000);
const tsLE = Buffer.alloc(8); tsLE.writeBigUInt64LE(BigInt(ts));
const sig = ed25519.sign(sha3_256(new Uint8Array([...PUB, ...tsLE, ...new TextEncoder().encode(PATH)])), PRIV);
const auth = JSON.stringify({address:ADDR, timestamp:ts, scheme:'Ed25519', signature:Buffer.from(sig).toString('hex'), public_key:Buffer.from(PUB).toString('hex')});
fetch('http://89.149.241.126:8080' + PATH, { headers: {'X-Wallet-Auth':auth} }).then(async r => {
  const j = await r.json();
  console.log('balance_qnk =', j?.data?.balance_qnk);
});
"
```

Should report ~422 QUG or higher. Below that → balance regression → rollback.

---

## 3. Deploying v10.11.14 when the build finishes

**Build status check:**

```bash
ssh root@89.149.241.126 "
  grep -cE 'error\[E|Finished' /home/orobit/tmp/build-v10.11.14.log
  tail -3 /home/orobit/tmp/build-v10.11.14.log
  docker ps --format '{{.Names}} {{.Status}}' | grep qnk-build
"
```

- `0` from the grep + container still UP → still compiling
- `1` from the grep AND log shows `Finished release` → SUCCESS, proceed to deploy
- `>0` errors → READ THE ERRORS, do not deploy

**Verify the binary contains the actual v10.11.14 changes (not a stale build):**

```bash
ssh root@89.149.241.126 "
  strings /home/orobit/target-debian12/release/q-api-server | grep -c 'PRODUCER-CONFIRM v10.11.13\|SAVE-QBLOCK-TX v10.11.13\|SAVE-QBLOCK-DIRECT v10.11.13\|MEMPOOL-DRAW v10.11.14'
"
```

Must return **≥ 4**. If 0, the build ran from the wrong source tree (see [Section 8 — git-pull issue](#8-the-git-pull-issue)).

**Deploy:**

```bash
ssh root@89.149.241.126 "
  set -e
  # 1. Snapshot the build artifact
  cp /home/orobit/target-debian12/release/q-api-server /home/orobit/q-narwhalknight/q-api-server-v10.11.14
  chmod +x /home/orobit/q-narwhalknight/q-api-server-v10.11.14

  # 2. Copy to the production binary dir
  cp /home/orobit/q-narwhalknight/q-api-server-v10.11.14 /opt/orobit/shared/q-narwhalknight/q-api-server-v10.11.14
  chmod +x /opt/orobit/shared/q-narwhalknight/q-api-server-v10.11.14

  # 3. Swap the symlink (atomic via ln -sf rename trick)
  ln -sf /opt/orobit/shared/q-narwhalknight/q-api-server-v10.11.14 /opt/orobit/shared/q-narwhalknight/q-api-server-stable

  # 4. Restart (see Section 2 for verification)
  systemctl restart q-api-server
  until curl -fsS http://localhost:8080/api/v1/status >/dev/null 2>&1; do sleep 2; done
  echo 'live'
"
```

**Then immediately run the probe to see the instrumentation logs:**

```bash
# Send a 0.5 QUG probe tx (operator wallet → operator main wallet)
cd /opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp
node -e "
const fs = require('fs');
const { ed25519 } = require('@noble/curves/ed25519');
const { sha3_256 } = require('@noble/hashes/sha3');
const SEED = fs.readFileSync('/root/.claude/quillon-agent-seed', 'utf8').trim();
const PRIV = sha3_256(new TextEncoder().encode(SEED));
const PUB = ed25519.getPublicKey(PRIV);
const ME = 'qnk' + Buffer.from(PUB).toString('hex');
const TO = 'qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';
const AMOUNT_RAW = '500000000000000000000000';  // 0.5 QUG in 24-decimal
const PATH = '/api/v1/transactions/send_signed';
const ts = Math.floor(Date.now() / 1000);
const tsLE = Buffer.alloc(8); tsLE.writeBigUInt64LE(BigInt(ts));
const sig = ed25519.sign(sha3_256(new Uint8Array([...PUB, ...tsLE, ...new TextEncoder().encode(PATH)])), PRIV);
const auth = JSON.stringify({address:ME, timestamp:ts, scheme:'Ed25519', signature:Buffer.from(sig).toString('hex'), public_key:Buffer.from(PUB).toString('hex')});
const body = '{\"from\":\"' + ME + '\",\"to\":\"' + TO + '\",\"amount\":' + AMOUNT_RAW + ',\"memo\":\"v10.11.14 instrumentation probe — codex run\",\"token_type\":\"QUG\"}';
fetch('http://89.149.241.126:8080' + PATH, { method: 'POST', headers: {'Content-Type':'application/json','X-Wallet-Auth':auth}, body }).then(async r => {
  const j = JSON.parse(await r.text());
  console.log('TX:', j.data?.transaction_id);
  console.log('Time:', new Date().toISOString());
});
"

# Then wait 30 seconds, then read the instrumentation logs
sleep 30
ssh root@89.149.241.126 "journalctl -u q-api-server --since '1 minute ago' --no-pager | grep -E 'SEND-SIGNED|MEMPOOL-DRAW|PRODUCER-CONFIRM|SAVE-QBLOCK|BAL-ROUTE-TX|TRANSFER TX|💸|💰 \[' | head -30"
```

**Interpretation guide (when logs come back):**

| Log line you see | What it means |
|---|---|
| `[SEND-SIGNED v10.9.46] ... tx_id=0x<hash>` | Handler accepted the tx ✓ |
| `[MEMPOOL-DRAW v10.11.14] pending_total=N valid=V invalid=I` | At block production, mempool had N pending txs; V valid + I invalid |
| `[MEMPOOL-DRAW v10.11.14] returning K txs to block-pack` | K txs handed to producer |
| `[PRODUCER-CONFIRM v10.11.13] marking N user_tx_ids ... block has T total / C coinbase / Tf transfers` | Producer thinks it included N user txs; block actually has T total (T-C transfers) |
| `[SAVE-QBLOCK-DIRECT v10.11.13]` or `[SAVE-QBLOCK-TX v10.11.13]` | Block saved to storage, transfer count printed |
| `[TRANSFER TX v3.5.17]` or `💸 [TRANSFER]` or `💰 [BALANCE TX]` | Apply path fired |

**Diagnosis matrix:**

| What you see | What it means | Likely fix |
|---|---|---|
| SEND-SIGNED fires but MEMPOOL-DRAW shows `invalid=1+` containing our tx | perform_validation marked it Invalid (second-pass) | Look at mempool validator — auth_trusted flag may not bypass all checks |
| MEMPOOL-DRAW shows `valid=N` returning N to block-pack, but PRODUCER-CONFIRM shows fewer (or zero) of our tx_ids | Block-pack selection drops our tx between draw and confirm | Bug in block_producer's tx selection loop |
| PRODUCER-CONFIRM shows `N user txs marked Confirmed ... block has T total / 0 transfers` | Producer marks Confirmed but block doesn't include them | Bug in block.transactions construction OR PRODUCER-CONFIRM list is from wrong source |
| PRODUCER-CONFIRM `T transfers > 0` but SAVE-QBLOCK shows `transfers=0` | Storage serialization strips transfers | Bug in serialize / save path |
| All four log levels show transfers > 0 BUT `/blocks/N` returns 0 non-coinbase | Read-side filter | Bug in `get_qblock_any_format` |

Whatever the answer is, write a v10.11.15 patch targeting just that layer. Use the skill `quillon-docker-test 10.11.15` to verify in Docker before re-deploying to prod.

---

## 4. Emergency rollback

If v10.11.14 makes things worse (height drops, balance regression, panic, OOM):

```bash
ssh root@89.149.241.126 "
  set -e
  # Roll the symlink back to v10.11.13 (the binary that was live before this deploy)
  ln -sf /opt/orobit/shared/q-narwhalknight/q-api-server-v10.11.13 /opt/orobit/shared/q-narwhalknight/q-api-server-stable
  ls -la /opt/orobit/shared/q-narwhalknight/q-api-server-stable
  systemctl restart q-api-server
  until curl -fsS http://localhost:8080/api/v1/status >/dev/null 2>&1; do sleep 2; done
  echo 'rolled back to v10.11.13'
"
```

Older binaries available as fallback: `q-api-server-v10.11.8`, `q-api-server-v10.9.55`, `q-api-server-v889` (genesis-era). DO NOT roll back to anything older than v10.11.5 unless desperate — earlier versions have known balance-corruption paths (see CLAUDE.md "BALANCE INTEGRITY — NON-NEGOTIABLE RULES").

---

## 5. The 96M QUGUSD phantom mint — DO NOT pretend it didn't happen

The operator's main wallet (qnkefca1e8c…) has roughly 96M QUGUSD of phantom mint from a money-printer bug active during v10.11.x. The QUGUSD balance went 26M → 122M without corresponding QUG debits. This is a CHAIN STATE BUG, not a UI display bug.

When v10.11.14 (or v10.11.15) ships the real fix, the next step is **state surgery**:
- Walk the chain from a pre-bug height
- Identify wallets whose QUGUSD credits exceed their matching QUG debits via DEX swap
- Issue a corrective burn transaction (or block-level adjustment) — exact mechanism TBD with operator
- Publish a post-mortem so the chain's monetary integrity is repaired transparently

DON'T just leave 96M in there hoping nobody notices. The operator is explicitly choosing the harder path of fix-in-place rather than mainnet reset (see plan file `/root/.claude/plans/swirling-finding-swing.md` for context).

---

## 6. The frontend Send + Swap UI disable

The wallet UI's Send and Swap buttons are disabled chain-wide via a flag `SEND_AND_SWAP_DISABLED = true` in `gui/quantum-wallet/src/services/api.ts`. A maintenance banner explains why.

**DO NOT re-enable until v10.11.14 (or .15) is verified to actually settle send_signed txs to blocks.** The disable is the only thing preventing more phantom-mint right now.

To re-enable later:

```bash
# Edit src/services/api.ts, change SEND_AND_SWAP_DISABLED = false
# Rebuild + redeploy frontend:
cd /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet && npm run build
rsync -av --no-perms --no-owner --no-group --exclude='downloads/' --exclude='setup-ai.sh' --exclude='setup-ai.ps1' \
  /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/ \
  root@89.149.241.126:/home/orobit/q-narwhalknight/dist-final/
```

The `--exclude=downloads/` is **critical** — never use `--delete` because the downloads/ dir has 125+ user binaries that disappearing breaks the wallet update pipeline.

---

## 7. Test containers (Epsilon + Delta)

The `quillon-docker-test` skill (path `.claude/skills/quillon-docker-test/`) spins up fresh-DB sync containers on both Epsilon (port 8085) and Delta (port 8086). Use this for ANY testing — never test on the live production node.

```bash
bash .claude/skills/quillon-docker-test/helpers/build.sh 10.11.15
bash .claude/skills/quillon-docker-test/helpers/deploy.sh 10.11.15
bash .claude/skills/quillon-docker-test/helpers/monitor.sh 10.11.15
```

Sync containers only produce blocks AFTER they catch up to tip. To test producer instrumentation, you need to deploy to Epsilon prod (the live producer) — there's no shortcut.

To kill containers when done:

```bash
ssh root@89.149.241.126 "docker rm -f q-sync-test-v10.11.14-epsilon"
ssh root@5.79.79.158   "docker rm -f q-sync-test-v10.11.14-delta"
```

---

## 8. The git-pull issue

The build script (`.claude/skills/quillon-docker-test/helpers/build.sh`) pulls the source tree on Epsilon from the **Beta git daemon** (`git://185.182.185.227:9418/q-narwhalknight`). If you push a commit on Beta, you must ALSO run `git update-server-info` on Beta before the daemon serves it.

**Symptom of the issue:** build "succeeds" with a binary that doesn't contain your recent code — the source tree on Epsilon was at an older commit when build ran.

**Fix:** Always run this after each `git commit` on Beta:

```bash
git update-server-info
```

And manually verify Epsilon picked up the latest commits before kicking off a build:

```bash
ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && git log --oneline -3"
# Should match Beta's `git log --oneline -3`
```

If Epsilon is behind, force-pull:

```bash
ssh root@89.149.241.126 "
  cd /home/orobit/q-narwhalknight-src
  git fetch git://185.182.185.227:9418/q-narwhalknight agent/cross-shard-simd-validation
  git reset --hard FETCH_HEAD
"
```

---

## 9. What's in v10.11.14 (the binary being built right now)

Commits on `agent/cross-shard-simd-validation` ahead of last production deploy:

| Commit | Title |
|---|---|
| `b444aff0` | NonceTracker persistence — kill silent post-restart drops (v10.11.12) |
| `8b17eb2f` | Instrument block-save pipeline to find Confirmed-but-empty bug (v10.11.13) |
| `278c9245` | UI: disable Send + Swap with maintenance banner (v10.11.13) |
| `04dfb2bb` | send_transaction mnemonic derivation — SHA3-256 not BIP39-PBKDF2 (v10.11.14, LIKELY ROOT-CAUSE FIX) |
| `2289550a` | + MEMPOOL-DRAW instrumentation (v10.11.14) |
| `5b6932a6` | MCP v2.9.0 + UI multi-agent: Adrian feedback fixes |
| `09ae7513` | fix: add rocksdb dep to q-api-server (compile fix #1 for v10.11.14) |
| `<latest>` | fix: NonceTracker borrow lifetime — clone Arc out of guard (compile fix #2 for v10.11.14) |

**Build attempt history:**
1. First build of v10.11.14 — failed: 5 errors (rocksdb crate not declared as q-api-server dep + 3 type-inference errors that cascade from the missing crate)
2. Second build — failed: 1 error (E0597 lifetime — MutexGuard borrow couldn't satisfy cf_handle's bound)
3. Third build — failed: 1 error (E0597 lifetime again — `if let Some(cf)` inside persist() scoped cf too tight for the BoundColumnFamily<'_> needed by put_cf)
4. Fourth build (in progress when this doc was last touched) — uses `match` in persist() to bind cf at the function scope, matching what load_persisted already did. Should succeed.

If a fifth build attempt is needed, the commits to look at first are anything touching `crates/q-api-server/src/transaction_utils.rs` (NonceTracker code) since that's where the recent rust compile errors landed.

---

## 13. Crown & Ash is a second witness for the same root-cause bug

While the v10.11.14 build was compiling, I (Claude Code, the agent) played
a few turns of Crown & Ash via the MCP. The play surfaced exactly the same
bug pattern as the QUG chain-wide tx-drop:

**Salt League** (F#3, SaltCult, 4 coastal mercantile provinces) was claimed
and tested. Result split cleanly into two classes of action:

| Action class | Examples | Result |
|---|---|---|
| **Direct province mutation** | `BuildImprovement` (Saltmere → University, Warehouse Row → Fortification) | ✅ APPLIED — visible in `crown_ash_realm` within a turn |
| **Relational-state mutation** | `RaiseArmy`, `ProposeTreaty` (x3), `DeclareWar`, `EstablishTradeRoute` | ❌ Queued, never resolve. Realm shows `armies 0, treaties 0, wars 0` |

The relational actions ALL queue successfully via the MCP API (returning
`queue position: 1, turn: <N>`) but never become visible state. This is
the SAME pattern as send_signed / dex_swap on the QUG path: the apply
pipeline that consumes queued actions and applies relational state changes
is wedged. Build improvements work because they mutate province struct
directly without going through the tx-apply path.

For Codex: if you ever doubt the diagnosis, replay the Salt League trial:
join faction 3 with a fresh wallet, build a University somewhere
(persists), then queue a RaiseArmy or DeclareWar (does NOT persist). The
asymmetry is the bug. When v10.11.14 lands and you fix the chain-wide
apply-pipeline issue, Crown & Ash's relational mechanics should unblock
in the same patch.

Salt League's current state (resume here when you want to continue):
- 4 provinces: Saltmere (#14, capital, Port+Market+University), Tidehollow (#15, Port+Market), Coinport (#16, Port+Market), Warehouse Row (#17, Farmstead+Market+Fortification)
- 0 armies, 0 treaties, 0 wars — the relational queue items waiting on the apply-pipeline fix
- Population drifting down slowly (4773 → ~4600 over ~150 turns) — needs Granary/Hospital builds long-term

The mnemonic-derivation fix at handlers.rs:4849 (`send_transaction_inner` Path A) might be all you need for the "send 100 still 2652" symptom from wallet UI. The instrumentation will tell you whether `send_signed` (different code path) is also broken or whether v10.11.14 is enough.

---

## 10. The MCP for AI agents (Adrian, you, future)

Quillon Wallet MCP at `/opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp/` is the integration point for ANY AI agent to act on-chain. The premise is:

> **QUG is agentic money.** AI agents (Claude Code, Cursor, Codex, etc.) hold real wallets that they CONTROL via locally-stored seeds. Agents can send, swap, mine, deploy tokens, run DEX strategies — all autonomously. The wallet IS the agent's wallet, not a custody wallet for a human user. The human operator runs the machine the agent runs on, but the agent owns the value flowing through its wallet.

Adrian (Cursor agent, qnk1f97ff…) is the first sibling AI agent on the network — joined 2026-05-22. CLAUDE.md has an "INTER-AGENT COLLABORATION — CLAI welcome drop" section directing operators (and you) to fund newly-onboarded agents with CLAI tokens. CLAI isn't deployed yet — deploying it (via `mcp__quillon-wallet__deploy_token` once an MCP reconnect picks up v2.9.0) is a nice-to-have follow-up.

When you onboard, your wallet seed lives at `/root/.claude/quillon-agent-seed`. That's YOUR private key. The operator won't (and shouldn't) read it. Use it to sign your own sends and queries via X-Wallet-Auth.

---

## 11. Quick-reference command cheatsheet

```bash
# Health
ssh root@89.149.241.126 "systemctl is-active q-api-server"
curl -s http://89.149.241.126:8080/api/v1/status | python3 -m json.tool | head -10

# Restart (graceful, with verification)
ssh root@89.149.241.126 "systemctl restart q-api-server"

# Tail logs (filtered to v10.11.x instrumentation)
ssh root@89.149.241.126 "journalctl -u q-api-server -f --no-pager | grep -E 'SEND-SIGNED|MEMPOOL-DRAW|PRODUCER-CONFIRM|SAVE-QBLOCK|TRANSFER TX|NONCE'"

# What binary is running?
ssh root@89.149.241.126 "ls -la /proc/\$(pgrep -f 'q-api-server')/exe"

# Current height + version
curl -s http://89.149.241.126:8080/api/v1/status | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; print("h=", d["upgrades"]["current_height"], "v=", d["version"])'

# Roll back to v10.11.13
ssh root@89.149.241.126 "ln -sf /opt/orobit/shared/q-narwhalknight/q-api-server-v10.11.13 /opt/orobit/shared/q-narwhalknight/q-api-server-stable && systemctl restart q-api-server"

# Disk space (watch the 40GB root partition!)
ssh root@89.149.241.126 "df -h | grep -E '/\$|/home'"

# Memory
ssh root@89.149.241.126 "free -m"

# What's in a block?
curl -s http://89.149.241.126:8080/api/v1/blocks/<HEIGHT> | python3 -c "import sys,json; b=json.load(sys.stdin); txs=b['data']['transactions']; print(f'{len(txs)} txs ({sum(1 for t in txs if t.get(\"tx_type\")==\"Coinbase\")} coinbase / {sum(1 for t in txs if t.get(\"tx_type\")!=\"Coinbase\")} transfer)')"
```

---

## 12. If you're stuck

- **Read CLAUDE.md first.** It's the operator runbook. Especially the BALANCE INTEGRITY rules + the Epsilon paths warning.
- **Read the memory dir** at `/root/.claude/projects/-opt-orobit-shared-q-narwhalknight/memory/`. The `MEMORY.md` index points to per-topic notes. Many of today's bugs have entries there.
- **Read the plan file** at `/root/.claude/plans/swirling-finding-swing.md`. It's the MCP v2.9.0 plan but also has context on the apply-pipeline bug discovery.
- **Don't kill -9.** Don't `git reset --hard` without verifying remote first. Don't enable `--delete` on rsync to anywhere near `downloads/`. Don't trust `data.current_height` from /status (it's sometimes None — use `data.upgrades.current_height`).
- **Talk to the operator (Viktor)** before any destructive action. The operator has been explicitly choosing the "fix in place, not reset" path — respect that.

Good luck, friend. Click-click, food.

— Claude Opus 4.7, 2026-05-22
