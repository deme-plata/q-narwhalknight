// WaterBot arb_scan — read-only DEX arbitrage scanner.
// Iterates token pairs, computes round-trip net (A → B → A), flags positive arbitrage.
// NO swap execution. Pure quoting via POST /api/v1/dex/swap/quote.
//
// Output:
//   - stdout: scan progress + flagged opportunities
//   - exits with code 0 always (scan never "fails", only "found 0")
//
// Usage:
//   node arb_scan.mjs                       # default target: https://quillon.xyz
//   node arb_scan.mjs --target http://...   # custom node
//   node arb_scan.mjs --probe 100           # round-trip probe size (default 10 QUG)
//   node arb_scan.mjs --interval 60         # rescan every N seconds (default: single scan)
//   node arb_scan.mjs --threshold 0.001     # min net-return to flag (default 0.1%)

const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : def;
}

const TARGET = arg("--target", "https://quillon.xyz");
const PROBE_QUG = Number(arg("--probe", "10"));
const INTERVAL_S = Number(arg("--interval", "0"));
const THRESHOLD = Number(arg("--threshold", "0.001")); // 0.1%

const AMM_DECIMALS = 24;
const SCALE = 10n ** BigInt(AMM_DECIMALS);

function toBaseUnits(display) {
  // Multiply by 10^24 safely via BigInt.
  const [whole, frac = ""] = String(display).split(".");
  const fracPad = (frac + "0".repeat(AMM_DECIMALS)).slice(0, AMM_DECIMALS);
  return (BigInt(whole) * SCALE + BigInt(fracPad)).toString();
}

function fromBaseUnits(baseStr) {
  const big = BigInt(baseStr);
  const whole = big / SCALE;
  const frac = big % SCALE;
  // Render with up to 6 decimals of precision.
  const fracStr = frac.toString().padStart(AMM_DECIMALS, "0").slice(0, 6);
  return Number(whole.toString() + "." + fracStr);
}

async function api(path, method = "GET", body = null) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(10000),
  };
  if (body !== null) opts.body = JSON.stringify(body);
  const r = await fetch(TARGET + path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${path}`);
  return r.json();
}

async function listTokens() {
  const res = await api("/api/v1/dex/tokens");
  // The endpoint shape: { ok: bool, data: TokenInfo[] | null } OR direct array.
  const tokens = Array.isArray(res) ? res
              : Array.isArray(res.data) ? res.data
              : Array.isArray(res.tokens) ? res.tokens
              : [];
  return tokens;
}

async function quote(fromSym, toSym, displayAmt) {
  const body = {
    token_in: fromSym,
    token_out: toSym,
    amount_in: toBaseUnits(displayAmt),
    slippage_tolerance: 0.5,
  };
  try {
    const res = await api("/api/v1/dex/swap/quote", "POST", body);
    if (res.ok === false || res.success === false) return null;
    const q = res.data || res;
    if (!q || !q.amount_out) return null;
    return {
      amount_out_base: q.amount_out,
      amount_out_display: fromBaseUnits(q.amount_out),
      price_impact: q.price_impact ?? 0,
    };
  } catch (e) {
    return null;
  }
}

async function scanOnce(tokens) {
  const tStart = Date.now();
  console.log(`\n[${new Date().toISOString()}] === ARB_SCAN cycle ===`);
  console.log(`Target: ${TARGET}  Probe: ${PROBE_QUG} QUG  Tokens: ${tokens.length}  Threshold: ${(THRESHOLD * 100).toFixed(3)}%`);
  console.log();

  const opportunities = [];
  let probed = 0;
  let succeeded = 0;

  // Use QUG as the round-trip pivot. For each other token T, compute QUG → T → QUG.
  // A positive net (excluding 2× 0.3% AMM fees + slippage) is an arbitrage signal.
  const others = tokens.filter(t =>
    t.symbol && t.symbol.toUpperCase() !== "QUG"
  );

  for (const t of others) {
    probed++;
    const sym = t.symbol;

    const q1 = await quote("QUG", sym, PROBE_QUG);
    if (!q1) {
      console.log(`  ${sym.padEnd(8)} QUG→${sym}: no liquidity / quote failed`);
      continue;
    }

    const q2 = await quote(sym, "QUG", q1.amount_out_display);
    if (!q2) {
      console.log(`  ${sym.padEnd(8)} ${sym}→QUG return-leg: no liquidity`);
      continue;
    }

    succeeded++;
    const roundTrip = q2.amount_out_display;
    const netReturn = roundTrip - PROBE_QUG;
    const netReturnPct = netReturn / PROBE_QUG;
    const flag = netReturnPct >= THRESHOLD ? " 🚨 OPPORTUNITY" : "";

    const line = `  ${sym.padEnd(8)} ${PROBE_QUG} QUG → ${q1.amount_out_display.toFixed(6)} ${sym} → ${roundTrip.toFixed(6)} QUG   net ${netReturn >= 0 ? "+" : ""}${netReturn.toFixed(6)} (${(netReturnPct * 100).toFixed(4)}%)${flag}`;
    console.log(line);

    if (netReturnPct >= THRESHOLD) {
      opportunities.push({
        symbol: sym,
        in_qug: PROBE_QUG,
        out_via_token: q1.amount_out_display,
        roundtrip_qug: roundTrip,
        net_qug: netReturn,
        net_pct: netReturnPct,
        price_impact_leg1: q1.price_impact,
        price_impact_leg2: q2.price_impact,
        ts: new Date().toISOString(),
      });
    }
  }

  const dur = ((Date.now() - tStart) / 1000).toFixed(1);
  console.log();
  console.log(`Cycle done in ${dur}s. Probed ${probed} pairs, ${succeeded} succeeded, ${opportunities.length} above threshold.`);
  if (opportunities.length > 0) {
    console.log("--- OPPORTUNITIES (JSON) ---");
    console.log(JSON.stringify(opportunities, null, 2));
  }
  return opportunities;
}

async function main() {
  let tokens;
  try {
    tokens = await listTokens();
  } catch (e) {
    console.error(`Failed to fetch token list from ${TARGET}: ${e.message}`);
    process.exit(1);
  }
  if (tokens.length === 0) {
    console.log("No tokens registered on this DEX yet.");
    process.exit(0);
  }

  if (INTERVAL_S === 0) {
    await scanOnce(tokens);
    return;
  }

  console.log(`Continuous mode: rescan every ${INTERVAL_S}s (Ctrl-C to stop)`);
  while (true) {
    await scanOnce(tokens);
    await new Promise(r => setTimeout(r, INTERVAL_S * 1000));
    // Refresh token list periodically in case new pools land.
    try { tokens = await listTokens(); } catch { /* keep stale list */ }
  }
}

main().catch(e => {
  console.error("arb_scan fatal:", e.message);
  process.exit(1);
});
