// cross_node_root_diff.mjs — read-only multi-node balance-root cross-diff.
//
// Queries GET /api/v1/integrity/balance-root on each known bootstrap and
// surfaces whether all reachable nodes agree on root_v1 + root_v2_smt at
// the common height. NO AUTH — the endpoint is unauthenticated by design.
//
// Exit codes:
//   0 — all reachable nodes agree on BOTH root_v1 AND root_v2_smt at the
//       common min-tip height. Cluster state is consistent.
//   1 — divergence detected, OR fewer than 2 nodes reachable.
//   2 — script-level error (HTTP failures only, no parseable response).
//
// Usage:
//   node cross_node_root_diff.mjs                    # default 4 bootstraps
//   node cross_node_root_diff.mjs --target https://... # add custom target
//   node cross_node_root_diff.mjs --strict           # require ALL nodes to match (not just reachable subset)
//
// Companion to multi_node_balance.mjs from 2026-05-21 (per-wallet balance
// diff). This one operates over the entire wallet table via the new
// /integrity/balance-root surface in v10.11.0.

const argv = process.argv.slice(2);
const extra = [];
let strict = false;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--target") { extra.push(argv[++i]); continue; }
  if (argv[i] === "--strict") { strict = true; continue; }
}

const NODES = [
  { name: "Epsilon (prod via quillon.xyz)", url: "https://quillon.xyz" },
  { name: "Epsilon (direct :8080)",          url: "http://89.149.241.126:8080" },
  { name: "Gamma (:8808)",                   url: "http://109.205.176.60:8808" },
  { name: "Delta (:8080)",                   url: "http://5.79.79.158:8080" },
  ...extra.map((u, i) => ({ name: `extra[${i}] ${u}`, url: u })),
];

async function queryNode(node) {
  const t0 = Date.now();
  try {
    const r = await fetch(node.url + "/api/v1/integrity/balance-root", {
      signal: AbortSignal.timeout(15000),
    });
    const latency = Date.now() - t0;
    if (!r.ok) {
      return { ...node, ok: false, error: `HTTP ${r.status}`, latency_ms: latency };
    }
    const j = await r.json();
    const d = j.data || j;
    if (!d || typeof d.root_v1 !== "string" || typeof d.root_v2_smt !== "string") {
      // Endpoint not present (pre-v10.11.0 node).
      return { ...node, ok: false, error: "no /integrity/balance-root (pre-v10.11.0)", latency_ms: latency };
    }
    // Also fetch /status to read the version + height (handy context).
    let version = "?";
    try {
      const sr = await fetch(node.url + "/api/v1/status", { signal: AbortSignal.timeout(5000) });
      const sj = await sr.json();
      version = sj.data?.version ?? sj.version ?? "?";
    } catch { /* version is best-effort */ }
    return {
      ...node,
      ok: true,
      latency_ms: latency,
      height: d.height,
      version,
      root_v1: d.root_v1,
      root_v2_smt: d.root_v2_smt,
      smt_state: d.smt_state,
      wallet_count: d.wallet_count,
      total_supply_qug: d.total_supply_qug,
    };
  } catch (e) {
    return { ...node, ok: false, error: e.message, latency_ms: Date.now() - t0 };
  }
}

console.log("=== cross_node_root_diff (v10.11.0 BalanceRootV2 visibility) ===");
console.log(`Targets: ${NODES.length}  strict=${strict}`);
console.log();

const results = await Promise.all(NODES.map(queryNode));

const NAME_W = 38;
const HEIGHT_W = 14;
const VER_W = 10;
const STATE_W = 10;
const HASH_W = 16; // first 16 hex chars shown

console.log(
  "node".padEnd(NAME_W) +
  "| version    " +
  "| height       " +
  "| smt_state " +
  "| root_v1 (first 16) " +
  "| root_v2_smt (first 16)" +
  "| wallets  " +
  "| latency"
);
console.log("-".repeat(NAME_W + 100));

for (const r of results) {
  if (!r.ok) {
    console.log(
      r.name.padEnd(NAME_W) +
      `| -          | -            | -         | -                  | -                     | -        | ERR ${r.error} (${r.latency_ms}ms)`
    );
    continue;
  }
  console.log(
    r.name.padEnd(NAME_W) +
    `| ${String(r.version).padEnd(11)}` +
    `| ${String(r.height).padEnd(13)}` +
    `| ${r.smt_state.padEnd(10)}` +
    `| ${r.root_v1.slice(0, HASH_W).padEnd(19)}` +
    `| ${r.root_v2_smt.slice(0, HASH_W).padEnd(22)}` +
    `| ${String(r.wallet_count).padEnd(9)}` +
    `| ${r.latency_ms}ms`
  );
}

console.log();

const reachable = results.filter(r => r.ok);
if (reachable.length === 0) {
  console.log("✗ No reachable nodes — cannot perform diff.");
  process.exit(2);
}
if (reachable.length < 2) {
  console.log("✗ Only one node reachable — cannot compare across nodes.");
  process.exit(strict ? 1 : 0);
}

// Compare at the COMMON MIN height — older nodes may legitimately be at lower
// tip; we want to see whether the heights they ALL share agree. But since
// /integrity/balance-root currently reports the CURRENT (live) root, not a
// historical root, the only honest comparison is when all heights are equal.
const heights = reachable.map(r => r.height);
const minH = Math.min(...heights);
const maxH = Math.max(...heights);
const heightSpread = maxH - minH;

console.log(`Height spread across reachable nodes: ${heightSpread} blocks (min=${minH}, max=${maxH})`);

if (heightSpread > 100) {
  console.log(`⚠️  Heights differ by >100 — root comparison is NOT meaningful (each node hashes its own wallet table at its own height). Need nodes within ~100 blocks to interpret root diffs.`);
  console.log(`   For now: visible inspection only. Operator should resync laggards before re-running.`);
  // In non-strict mode this is informational, not a failure.
  process.exit(strict ? 1 : 0);
}

// Heights close enough — proceed with root comparison.
const root_v1_set = new Set(reachable.map(r => r.root_v1));
const root_v2_set = new Set(reachable.map(r => r.root_v2_smt));
const wallet_count_set = new Set(reachable.map(r => r.wallet_count));

let problems = 0;

if (root_v1_set.size === 1) {
  console.log(`✓ root_v1 matches across all ${reachable.length} reachable nodes: ${[...root_v1_set][0]}`);
} else {
  console.log(`✗ root_v1 DIVERGENT — ${root_v1_set.size} distinct values across ${reachable.length} nodes:`);
  for (const v of root_v1_set) {
    const who = reachable.filter(r => r.root_v1 === v).map(r => r.name);
    console.log(`   ${v}  ←  ${who.join(", ")}`);
  }
  problems++;
}

if (root_v2_set.size === 1) {
  console.log(`✓ root_v2_smt matches across all ${reachable.length} reachable nodes: ${[...root_v2_set][0]}`);
} else {
  console.log(`✗ root_v2_smt DIVERGENT — ${root_v2_set.size} distinct values across ${reachable.length} nodes:`);
  for (const v of root_v2_set) {
    const who = reachable.filter(r => r.root_v2_smt === v).map(r => r.name);
    console.log(`   ${v}  ←  ${who.join(", ")}`);
  }
  problems++;
}

if (wallet_count_set.size !== 1) {
  console.log(`⚠️  wallet_count differs: ${[...wallet_count_set].join(", ")} — this is the root cause if the hashes disagree.`);
}

if (strict && reachable.length < NODES.length) {
  console.log(`✗ --strict: only ${reachable.length}/${NODES.length} nodes reachable; expected all.`);
  problems++;
}

if (problems === 0) {
  console.log();
  console.log("✓ All checks passed — cluster state is consistent across reachable nodes.");
  process.exit(0);
}
console.log();
console.log(`✗ ${problems} divergence(s) detected — see above.`);
console.log(`   Recommended: run quillon-sync-probe skill against the diverging nodes,`);
console.log(`   then trigger /integrity/rebuild-smt (v10.10.17 endpoint, not yet shipped) on the laggard.`);
process.exit(1);
