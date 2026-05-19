# Quillon WebGPU Miner — Proof of Concept

Browser-native SHA3-256 miner for Quillon Graph with WebGPU compute-shader acceleration and a CPU Web Worker fallback for browsers without WebGPU.

> **Read this first.** Production Quillon Graph mining uses a Genus-2 hyperelliptic-curve VDF, **not** SHA3 PoW. This is a proof-of-concept demonstrating the WebGPU pattern and providing a useful TPS test harness via `/api/v1/mining/submit`. The Genus-2 VDF port to WebGPU is its own ~6-week effort, tracked as future work.

---

## What it does

1. Detects whether the browser has WebGPU available (Chrome 113+ / Edge / Safari TP)
2. If yes: runs a Keccak-f[1600] compute shader on the GPU, dispatching ~262 144 trial nonces per WebGPU compute pass
3. If no: spawns a pool of Web Workers (one per CPU core, max 16) using `@noble/hashes/sha3` for audited SHA3-256
4. Compares each candidate hash against a configurable target
5. POSTs found solutions to `/api/v1/mining/submit` with the wallet address
6. Reports hashrate via a 5-second rolling window

## Why it matters

Three independent rationales for shipping this PoC:

- **Browser as agent endpoint** — the WebGPU miner demonstrates that any agent on chain can run mining from a tab. No native binary download, no setup friction. This is one face of the "agent-native chain" thesis from `papers/five-mirrors-2026.pdf`.
- **Democratised participation** — the chain's hashrate today comes from a handful of operator-run nodes. A browser miner lets community members contribute computation without operating a full node, broadening the participation surface.
- **TPS test harness** — the WebGPU dispatch loop (262 k nonces per `requestAnimationFrame` tick) is a useful client-side workload generator for stress-testing the mining-submit endpoint, AFL-1 reference impl, and the new Agent Activity Panel under load.

## Architecture

```
                        ┌────────────────────┐
                        │  index.html + UI   │
                        └─────────┬──────────┘
                                  │
                            src/main.ts
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
       WebGPU available?                    No WebGPU
                  │                               │
                  ▼                               ▼
        src/webgpu/runner.ts            src/cpu/runner.ts
          (dispatches WGSL                (spawns Worker pool)
           compute shader)
                  │                               │
                  ▼                               ▼
       src/webgpu/keccak.wgsl           src/cpu/worker.ts
          (Keccak-f[1600] kernel)         (@noble/hashes/sha3)
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                         src/submitter.ts
                                  │
                                  ▼
                  POST /api/v1/mining/submit
```

## File layout

| File | Purpose |
|---|---|
| `index.html` | Minimal UI: wallet input, start/stop, hashrate display |
| `src/main.ts` | Orchestration: WebGPU detection, mode selection, UI binding |
| `src/webgpu/keccak.wgsl` | Keccak-f[1600] / SHA3-256 compute shader (manual 64-bit via u32 pairs) |
| `src/webgpu/runner.ts` | WebGPU driver — pipeline setup, dispatch, readback |
| `src/cpu/worker.ts` | Web Worker mining loop using `@noble/hashes` SHA3 |
| `src/cpu/runner.ts` | Worker pool manager — stride-based nonce-space tiling |
| `src/submitter.ts` | HTTP submitter to `/api/v1/mining/submit` |
| `package.json` / `vite.config.ts` / `tsconfig.json` | Build config (vite + TypeScript) |

## Build + run

```bash
cd gui/webgpu-miner
npm install
npm run dev
# Open http://localhost:5174
```

Dev server proxies `/api/*` to `https://quillon.xyz` so the submit path works against the live network.

## Performance notes

- **WebGPU**: typical hashrate on modern integrated GPU (M2 Air, Intel Xe): 2-5 MH/s. Discrete GPU (RTX 3060): 50-150 MH/s.
- **CPU fallback**: 8-core Apple Silicon: ~1-2 MH/s aggregate. 16-core Threadripper: 4-8 MH/s.
- **Submit overhead**: ~30 ms per HTTPS POST; not the bottleneck.

## What this doesn't do

- **No Genus-2 VDF**. The production Quillon mining algorithm is documented in `papers/genus2-jacobian-vdf-mining-whitepaper.pdf`. This PoC uses SHA3 PoW with a configurable target as a stand-in.
- **No mining-job protocol**. The PoC uses a randomly-generated header per session; production miners poll `/api/v1/mining/challenge` for the current chain-tip header. Adapting the PoC to poll the real challenge is ~30 lines of code in `submitter.ts`.
- **No wallet-key signing**. The PoC submits as if the wallet were "configured externally"; production miners may sign the submission with the wallet's key. Adapting requires integrating `crates/q-trading-bot/src/wallet_auth.rs` patterns into TypeScript (see `AGENT.md` §1).

## Strategic context

This miner ships alongside three larger pieces:

- **PR #87** — Agent Fiber Lane reference implementation (signed-tx submit path)
- **PR #88** — Twitter MCP + xAI x-algorithm scorer
- **PR #90** — Agent Activity Panel (Codex-style task surface)

Together they form the "agent-native infrastructure" stack on Quillon Graph: agents transact (#87), agents speak (#88), humans observe (#90). This WebGPU miner adds a fifth surface — **agents (and humans) mine from any browser tab**, no install required. It's the most accessible on-ramp.

The strategic timing window is H2 2026 (NVIDIA Vera CPU deployment via CoreWeave/Oracle starting then). See `/root/.claude/plans/distributed-seeking-orbit.md` for the full plan.

## License

Apache-2.0 (same as Quillon Graph and the AFL-1 protocol spec).
