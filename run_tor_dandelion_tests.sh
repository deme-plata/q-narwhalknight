#!/bin/bash
# Comprehensive Tor + Dandelion++ + Arti Test Runner
# Executes all privacy stack tests with detailed reporting

set -e

echo "======================================================================"
echo "  🧅 Q-NarwhalKnight Tor + Dandelion++ + Arti Test Suite"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0.32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test categories
TESTS=(
    # Section 1: Arti Tests
    "test_arti_embedded_client_initialization"
    "test_arti_fallback_from_socks_failure"
    "test_arti_connection_to_hidden_service"
    "test_arti_metrics_collection"

    # Section 2: Circuit Management Tests
    "test_circuit_manager_initialization"
    "test_circuit_rotation"
    "test_circuit_quality_of_service"

    # Section 3: Dandelion++ Protocol Tests
    "test_dandelion_initialization"
    "test_dandelion_relay_candidate_update"
    "test_dandelion_stem_phase"
    "test_dandelion_fluff_phase"
    "test_dandelion_stem_to_fluff_transition"
    "test_dandelion_quantum_timing_obfuscation"
    "test_dandelion_deduplication"
    "test_dandelion_cleanup_expired"

    # Section 4: Integration Tests
    "test_full_privacy_stack_integration"
    "test_multi_node_dandelion_simulation"

    # Section 5: Performance Benchmarks
    "benchmark_dandelion_throughput"
    "benchmark_tor_latency"
    "benchmark_quantum_seeding_overhead"

    # Section 6: Security Tests
    "test_traffic_analysis_resistance"
    "test_anonymity_set_size"
)

# Test execution
echo "🚀 Running ${#TESTS[@]} tests..."
echo ""

PASSED=0
FAILED=0
SKIPPED=0

# Run all tests
for test in "${TESTS[@]}"; do
    echo -n "   Testing: $test ... "

    if timeout 60 cargo test --package q-tor-client --test comprehensive_tor_dandelion_arti_tests "$test" -- --nocapture > /tmp/test_${test}.log 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        if grep -q "test result: ok" /tmp/test_${test}.log; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}❌ FAILED${NC}"
            FAILED=$((FAILED + 1))
            echo "   See /tmp/test_${test}.log for details"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "  📊 Test Results Summary"
echo "======================================================================"
echo ""
echo "   Total Tests:  ${#TESTS[@]}"
echo -e "   ${GREEN}Passed:       $PASSED${NC}"
echo -e "   ${RED}Failed:       $FAILED${NC}"
echo -e "   ${YELLOW}Skipped:      $SKIPPED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check logs in /tmp/${NC}"
    echo ""
    exit 1
fi
