#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# run_benchmarks.sh — Unified benchmark harness for Q-NarwhalKnight
# Issue #035: Reproducible benchmarking suite
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run all benchmarks
#   ./scripts/run_benchmarks.sh --quick      # Quick subset (2-3 min)
#   ./scripts/run_benchmarks.sh --json       # Output JSON results
#   ./scripts/run_benchmarks.sh --suite NAME # Run specific suite
#
# Outputs:
#   target/benchmark-results/YYYY-MM-DD_HHMMSS/
#     ├── summary.json       # Machine-readable results
#     ├── summary.txt        # Human-readable summary
#     └── criterion/         # Criterion HTML reports
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ── Configuration ──
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
RESULTS_DIR="target/benchmark-results/$TIMESTAMP"
JSON_OUTPUT=false
QUICK_MODE=false
SUITE_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --json) JSON_OUTPUT=true; shift ;;
        --quick) QUICK_MODE=true; shift ;;
        --suite) SUITE_FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ── System info ──
echo "═══════════════════════════════════════════════════════════"
echo "  Q-NarwhalKnight Benchmark Suite"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "System:"
echo "  Hostname: $(hostname)"
echo "  Kernel:   $(uname -r)"
echo "  CPUs:     $(nproc)"
echo "  RAM:      $(free -h | awk '/Mem:/{print $2}')"
echo "  Rust:     $(rustc --version 2>/dev/null || echo 'not found')"
echo ""

# Save system info to results
cat > "$RESULTS_DIR/system_info.json" << SYSEOF
{
  "hostname": "$(hostname)",
  "kernel": "$(uname -r)",
  "cpus": $(nproc),
  "ram_bytes": $(free -b | awk '/Mem:/{print $2}'),
  "rust_version": "$(rustc --version 2>/dev/null || echo 'unknown')",
  "timestamp": "$TIMESTAMP"
}
SYSEOF

# ── Benchmark suites ──
# Each suite is a (name, package, bench_name) tuple.
# Criterion benchmarks output to target/criterion/.

declare -A SUITE_RESULTS
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

run_suite() {
    local name="$1"
    local package="$2"
    local bench="$3"
    local timeout="${4:-600}"

    if [[ -n "$SUITE_FILTER" && "$name" != "$SUITE_FILTER" ]]; then
        return 0
    fi

    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    echo "──────────────────────────────────────────────────────"
    echo "  [$TOTAL_SUITES] $name"
    echo "  Package: $package | Bench: $bench"
    echo "──────────────────────────────────────────────────────"

    local start_time=$(date +%s)
    if timeout "$timeout" cargo bench --package "$package" --bench "$bench" -- --output-format=bencher 2>&1 | tee "$RESULTS_DIR/${name}.txt"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "  Result: PASS (${duration}s)"
        SUITE_RESULTS[$name]="pass:${duration}s"
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "  Result: FAIL (${duration}s)"
        SUITE_RESULTS[$name]="fail:${duration}s"
        FAILED_SUITES=$((FAILED_SUITES + 1))
    fi
    echo ""
}

# ── Run benchmark suites ──

if $QUICK_MODE; then
    echo "Quick mode: running core suites only"
    echo ""
    run_suite "crypto"         "q-api-server" "crypto_performance"      120
    run_suite "storage"        "q-api-server" "storage_performance"     120
    run_suite "vm"             "q-vm"         "vm_benchmarks"           120
else
    # Core performance
    run_suite "crypto"         "q-api-server" "crypto_performance"      300
    run_suite "storage"        "q-api-server" "storage_performance"     300
    run_suite "quantum_rng"    "q-api-server" "quantum_rng_performance" 300
    run_suite "zk_performance" "q-api-server" "zk_performance"          300
    run_suite "phase1_pq"      "q-api-server" "phase1_performance"      300

    # Subsystem benchmarks
    run_suite "vm"             "q-vm"               "vm_benchmarks"           300
    run_suite "simd_crypto"    "q-crypto-simd"       "simd_crypto_benchmarks"  300
    run_suite "precision"      "q-precision"          "precision_bench"         300
    run_suite "mixing"         "q-quantum-mixing"     "mixing_performance"      300
    run_suite "sharding"       "q-sharding"           "shard_performance"       300
    run_suite "temporal_shield" "q-temporal-shield"   "temporal_shield"         300
    run_suite "cache"          "q-cache"              "cache_performance"       300
    run_suite "lattice_guard"  "q-lattice-guard"      "lattice_guard_bench"     300
    run_suite "recursive_proofs" "q-recursive-proofs" "recursive_proof_bench"   300

    # q-flux
    run_suite "flux"           "q-flux"               "flux_bench"              300
    run_suite "queue"          "q-queue"               "queue_bench"             300

    # Platform-specific (Linux only)
    if [[ "$(uname)" == "Linux" ]]; then
        run_suite "kernel_io" "q-kernel-io" "kernel_io_benchmarks" 300
        run_suite "consensus" "q-benchmarks" "consensus_performance" 300
        run_suite "tps"       "q-benchmarks" "tps_benchmark"         300
        run_suite "network"   "q-benchmarks" "network_benchmark"     300
        run_suite "memory"    "q-benchmarks" "memory_benchmark"      300
        run_suite "zk_snark"  "q-benchmarks" "zk_snark_benchmark"    300
    fi
fi

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  BENCHMARK SUMMARY"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Total:  $TOTAL_SUITES suites"
echo "  Passed: $PASSED_SUITES"
echo "  Failed: $FAILED_SUITES"
echo ""
echo "  Results: $RESULTS_DIR/"
echo ""

# Generate JSON summary
{
echo "{"
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"total\": $TOTAL_SUITES,"
echo "  \"passed\": $PASSED_SUITES,"
echo "  \"failed\": $FAILED_SUITES,"
echo "  \"suites\": {"
local first=true
for name in "${!SUITE_RESULTS[@]}"; do
    if $first; then first=false; else echo ","; fi
    local result="${SUITE_RESULTS[$name]}"
    local status="${result%%:*}"
    local duration="${result#*:}"
    printf "    \"%s\": {\"status\": \"%s\", \"duration\": \"%s\"}" "$name" "$status" "$duration"
done
echo ""
echo "  }"
echo "}"
} > "$RESULTS_DIR/summary.json"

if $JSON_OUTPUT; then
    cat "$RESULTS_DIR/summary.json"
fi

# Copy Criterion reports if they exist
if [[ -d "target/criterion" ]]; then
    cp -r target/criterion "$RESULTS_DIR/criterion" 2>/dev/null || true
fi

echo "Done. View results at: $RESULTS_DIR/"
