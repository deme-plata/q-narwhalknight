#!/bin/bash

# Q-NarwhalKnight VM Test Runner Script
# Comprehensive testing suite for the virtual machine

set -e

echo "🧪 Q-NarwhalKnight Virtual Machine Test Suite"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "\n${BLUE}🔍 Running: $test_name${NC}"
    echo "Command: $test_command"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

echo -e "\n${YELLOW}📋 Running Unit Tests${NC}"
echo "===================="

run_test "VM State Management Tests" "cargo test --package q-vm --test vm_tests test_vm_state_management"
run_test "Contract State Tests" "cargo test --package q-vm --test vm_tests test_contract_state_management"
run_test "State Persistence Tests" "cargo test --package q-vm --test vm_tests test_state_persistence"
run_test "State Root Calculation Tests" "cargo test --package q-vm --test vm_tests test_state_root_calculation"
run_test "VM Error Handling Tests" "cargo test --package q-vm --test vm_tests test_vm_error_handling"
run_test "Concurrent State Access Tests" "cargo test --package q-vm --test vm_tests test_concurrent_state_access"
run_test "Large State Operations Tests" "cargo test --package q-vm --test vm_tests test_large_state_operations"
run_test "VM Resource Tracking Tests" "cargo test --package q-vm --test vm_tests test_vm_resource_tracking"
run_test "VM Performance Benchmark" "cargo test --package q-vm --test vm_tests benchmark_vm_performance"
run_test "Full VM Integration Tests" "cargo test --package q-vm --test vm_tests test_full_vm_integration"
run_test "VM Error Scenarios Tests" "cargo test --package q-vm --test vm_tests test_vm_error_scenarios"

echo -e "\n${YELLOW}🔗 Running Integration Tests${NC}"
echo "============================="

run_test "VM-DAG Integration" "cargo test --package q-vm --test integration_tests test_vm_dag_knight_integration"
run_test "VM-Quantum Crypto Integration" "cargo test --package q-vm --test integration_tests test_vm_quantum_crypto_integration"
run_test "VM-Robot Control Integration" "cargo test --package q-vm --test integration_tests test_vm_robot_control_integration"
run_test "VM Persistence Integration" "cargo test --package q-vm --test integration_tests test_vm_persistence_integration"
run_test "VM Concurrent Robot Operations" "cargo test --package q-vm --test integration_tests test_vm_concurrent_robot_operations"
run_test "VM Complex Smart Contracts" "cargo test --package q-vm --test integration_tests test_vm_complex_smart_contracts"
run_test "End-to-End Transaction Processing" "cargo test --package q-vm --test integration_tests test_end_to_end_transaction_processing"

echo -e "\n${YELLOW}⚡ Running Performance Benchmarks${NC}"
echo "================================="

echo -e "${BLUE}📊 Note: Benchmarks take longer to run and provide performance metrics${NC}"
run_test "VM Performance Benchmarks" "cargo bench --package q-vm --bench vm_benchmarks -- --test"

echo -e "\n${YELLOW}🔍 Running Code Quality Checks${NC}"
echo "==============================="

run_test "Clippy Lints" "cargo clippy --package q-vm -- -D warnings"
run_test "Format Check" "cargo fmt --package q-vm -- --check"
run_test "Doc Tests" "cargo test --package q-vm --doc"

echo -e "\n${YELLOW}📊 Test Results Summary${NC}"
echo "======================"
echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! VM is ready for deployment.${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review and fix issues.${NC}"
    exit 1
fi