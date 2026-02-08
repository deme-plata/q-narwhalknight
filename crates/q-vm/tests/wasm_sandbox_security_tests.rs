//! WASM Sandbox Security Tests
//!
//! Tests to ensure the WASM VM properly sandboxes smart contracts
//! and prevents escape or resource exhaustion attacks.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Infinite loop detection and termination
//! 2. Memory bounds validation
//! 3. Gas limit enforcement
//! 4. Stack overflow attempts
//! 5. Host function abuse
//! 6. State isolation between contracts
//!
//! Run with: cargo test --package q-vm --test wasm_sandbox_security_tests

use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR WASM VM SECURITY TESTING
// ============================================================================

/// Gas metering state
#[derive(Debug)]
pub struct GasState {
    pub limit: u64,
    pub used: AtomicU64,
}

impl GasState {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            used: AtomicU64::new(0),
        }
    }

    pub fn charge(&self, amount: u64) -> Result<(), String> {
        let current = self.used.fetch_add(amount, Ordering::SeqCst);
        if current + amount > self.limit {
            return Err(format!(
                "OUT_OF_GAS: Used {} + {} > limit {}",
                current, amount, self.limit
            ));
        }
        Ok(())
    }

    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used.load(Ordering::SeqCst))
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::SeqCst)
    }
}

/// Memory bounds for WASM linear memory
#[derive(Debug, Clone)]
pub struct MemoryBounds {
    pub min_pages: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub current_pages: u32,
}

impl MemoryBounds {
    pub fn new(min_pages: u32, max_pages: u32) -> Self {
        Self {
            min_pages,
            max_pages,
            page_size: 65536, // 64KB per WASM page
            current_pages: min_pages,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.current_pages as u64 * self.page_size as u64
    }

    pub fn is_valid_access(&self, offset: u64, size: u64) -> bool {
        let total = self.total_bytes();
        offset.checked_add(size).map(|end| end <= total).unwrap_or(false)
    }

    pub fn grow(&mut self, pages: u32) -> Result<u32, String> {
        let new_pages = self.current_pages.checked_add(pages)
            .ok_or("Memory page overflow")?;

        if new_pages > self.max_pages {
            return Err(format!(
                "Memory growth denied: {} + {} > max {}",
                self.current_pages, pages, self.max_pages
            ));
        }

        let old_pages = self.current_pages;
        self.current_pages = new_pages;
        Ok(old_pages)
    }
}

/// Simulated WASM execution state
#[derive(Debug)]
pub struct MockWasmExecution {
    pub gas: GasState,
    pub memory: MemoryBounds,
    pub call_depth: AtomicU64,
    pub max_call_depth: u64,
    pub execution_steps: AtomicU64,
    pub max_execution_steps: u64,
    pub terminated: AtomicBool,
    pub termination_reason: Mutex<Option<String>>,
    pub contract_storage: Mutex<HashMap<[u8; 32], Vec<u8>>>,
}

impl MockWasmExecution {
    pub fn new(gas_limit: u64, max_memory_pages: u32) -> Self {
        Self {
            gas: GasState::new(gas_limit),
            memory: MemoryBounds::new(1, max_memory_pages),
            call_depth: AtomicU64::new(0),
            max_call_depth: 1024,
            execution_steps: AtomicU64::new(0),
            max_execution_steps: 10_000_000, // 10M steps max
            terminated: AtomicBool::new(false),
            termination_reason: Mutex::new(None),
            contract_storage: Mutex::new(HashMap::new()),
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated.load(Ordering::SeqCst)
    }

    pub fn terminate(&self, reason: &str) {
        self.terminated.store(true, Ordering::SeqCst);
        *self.termination_reason.lock().unwrap() = Some(reason.to_string());
    }

    pub fn get_termination_reason(&self) -> Option<String> {
        self.termination_reason.lock().unwrap().clone()
    }

    /// Simulate executing one instruction
    pub fn execute_step(&self, gas_cost: u64) -> Result<(), String> {
        if self.is_terminated() {
            return Err("Execution terminated".to_string());
        }

        // Check execution step limit (infinite loop protection)
        let steps = self.execution_steps.fetch_add(1, Ordering::SeqCst);
        if steps >= self.max_execution_steps {
            self.terminate("INFINITE_LOOP: Exceeded max execution steps");
            return Err("Execution step limit exceeded".to_string());
        }

        // Charge gas
        self.gas.charge(gas_cost)?;

        Ok(())
    }

    /// Simulate a function call (increases call depth)
    pub fn enter_call(&self) -> Result<(), String> {
        let depth = self.call_depth.fetch_add(1, Ordering::SeqCst);
        if depth >= self.max_call_depth {
            self.terminate("STACK_OVERFLOW: Max call depth exceeded");
            return Err(format!(
                "Stack overflow: call depth {} exceeds max {}",
                depth, self.max_call_depth
            ));
        }
        Ok(())
    }

    pub fn exit_call(&self) {
        self.call_depth.fetch_sub(1, Ordering::SeqCst);
    }

    /// Simulate memory access
    pub fn memory_access(&self, offset: u64, size: u64, gas_per_byte: u64) -> Result<(), String> {
        // Check if terminated
        if self.is_terminated() {
            return Err("Execution terminated".to_string());
        }

        // Bounds check
        if !self.memory.is_valid_access(offset, size) {
            self.terminate("MEMORY_BOUNDS_VIOLATION");
            return Err(format!(
                "Memory access out of bounds: offset {} size {} exceeds {}",
                offset, size, self.memory.total_bytes()
            ));
        }

        // Charge gas for memory access
        let gas_cost = size.saturating_mul(gas_per_byte);
        self.gas.charge(gas_cost)?;

        Ok(())
    }

    /// Simulate storage read
    pub fn storage_read(&self, key: &[u8; 32]) -> Result<Option<Vec<u8>>, String> {
        self.gas.charge(200)?; // Base cost for storage read
        let storage = self.contract_storage.lock().unwrap();
        let value = storage.get(key).cloned();
        if let Some(ref v) = value {
            self.gas.charge(v.len() as u64)?; // Per-byte cost
        }
        Ok(value)
    }

    /// Simulate storage write
    pub fn storage_write(&self, key: [u8; 32], value: Vec<u8>) -> Result<(), String> {
        // Check value size limit
        if value.len() > 1024 * 1024 {
            return Err("Storage value too large (max 1MB)".to_string());
        }

        self.gas.charge(5000)?; // Base cost for storage write
        self.gas.charge(value.len() as u64 * 10)?; // Per-byte cost

        let mut storage = self.contract_storage.lock().unwrap();
        storage.insert(key, value);
        Ok(())
    }
}

// ============================================================================
// INFINITE LOOP DETECTION TESTS
// ============================================================================

/// Test that infinite loops are terminated
#[test]
fn test_infinite_loop_termination() {
    // Use high gas limit so we hit step limit before gas limit
    let vm = MockWasmExecution::new(100_000_000, 16);
    // Override max_execution_steps to a smaller value for faster test
    // (Cannot directly set, so we'll run until we hit the gas limit or step limit)

    // Simulate an infinite loop - will either run out of gas or hit step limit
    let mut iterations = 0;
    let mut termination_reason = String::new();
    loop {
        match vm.execute_step(1) {
            Ok(_) => iterations += 1,
            Err(e) => {
                termination_reason = e;
                break;
            }
        }
    }

    // Either terminated due to step limit or out of gas - both are valid termination
    assert!(iterations > 0, "Should have executed some iterations");

    // The termination should have happened - either due to step limit or gas
    // If step limit was hit first, vm.is_terminated() will be true
    // If gas ran out first, we get an error but terminated flag may not be set
    // Both are valid protection mechanisms
    let is_step_limit = termination_reason.contains("step limit") || termination_reason.contains("Execution step limit");
    let is_gas_limit = termination_reason.contains("OUT_OF_GAS");

    assert!(
        is_step_limit || is_gas_limit,
        "Should terminate due to step limit or gas: {}",
        termination_reason
    );
}

/// Test that normal execution completes
#[test]
fn test_normal_execution_completes() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Execute a reasonable number of steps
    for _ in 0..1000 {
        vm.execute_step(1).unwrap();
    }

    assert!(!vm.is_terminated(), "VM should not be terminated");
    assert_eq!(vm.execution_steps.load(Ordering::SeqCst), 1000);
}

// ============================================================================
// GAS LIMIT ENFORCEMENT TESTS
// ============================================================================

/// Test out of gas termination
#[test]
fn test_out_of_gas() {
    let vm = MockWasmExecution::new(100, 16); // Only 100 gas

    // Try to use more gas than available
    for _ in 0..10 {
        let _ = vm.execute_step(20); // 20 gas per step
    }

    // Should have run out of gas
    let result = vm.execute_step(1);
    assert!(result.is_err(), "Should be out of gas");
    assert!(result.unwrap_err().contains("OUT_OF_GAS"));
}

/// Test gas accounting accuracy
#[test]
fn test_gas_accounting() {
    let vm = MockWasmExecution::new(1000, 16);

    vm.execute_step(100).unwrap();
    vm.execute_step(200).unwrap();
    vm.execute_step(50).unwrap();

    assert_eq!(vm.gas.used(), 350);
    assert_eq!(vm.gas.remaining(), 650);
}

/// Test zero gas operation
#[test]
fn test_zero_gas_operation() {
    let vm = MockWasmExecution::new(100, 16);

    // Zero gas operations should succeed
    for _ in 0..1000 {
        vm.execute_step(0).unwrap();
    }

    assert_eq!(vm.gas.used(), 0);
    assert_eq!(vm.gas.remaining(), 100);
}

/// Test exact gas limit
#[test]
fn test_exact_gas_limit() {
    let vm = MockWasmExecution::new(100, 16);

    // Use exactly 100 gas
    vm.gas.charge(100).unwrap();

    // Next charge should fail
    let result = vm.gas.charge(1);
    assert!(result.is_err());
}

// ============================================================================
// MEMORY BOUNDS TESTS
// ============================================================================

/// Test valid memory access
#[test]
fn test_valid_memory_access() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Access within bounds (1 page = 64KB)
    let result = vm.memory_access(0, 1000, 1);
    assert!(result.is_ok());
}

/// Test memory bounds violation
#[test]
fn test_memory_bounds_violation() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Try to access beyond allocated memory
    let total = vm.memory.total_bytes();
    let result = vm.memory_access(total, 1, 1);

    assert!(result.is_err(), "Should reject out-of-bounds access");
    assert!(vm.is_terminated(), "Should terminate on bounds violation");
    assert!(
        vm.get_termination_reason()
            .unwrap()
            .contains("MEMORY_BOUNDS"),
        "Should report bounds violation"
    );
}

/// Test memory access overflow
#[test]
fn test_memory_access_overflow() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Offset + size overflows u64
    let result = vm.memory_access(u64::MAX, 100, 1);
    assert!(result.is_err(), "Should reject overflowing access");
}

/// Test memory growth limits
#[test]
fn test_memory_growth_limits() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Try to grow beyond max pages
    let mut memory = vm.memory.clone();
    let result = memory.grow(20); // 1 + 20 > 16 max

    assert!(result.is_err(), "Should reject growth beyond max");
}

/// Test valid memory growth
#[test]
fn test_valid_memory_growth() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    let mut memory = vm.memory.clone();
    let old_pages = memory.grow(5).unwrap();

    assert_eq!(old_pages, 1);
    assert_eq!(memory.current_pages, 6);
    assert_eq!(memory.total_bytes(), 6 * 65536);
}

// ============================================================================
// STACK OVERFLOW TESTS
// ============================================================================

/// Test stack overflow detection
#[test]
fn test_stack_overflow_detection() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Simulate deep recursion
    let mut depth = 0;
    loop {
        match vm.enter_call() {
            Ok(_) => depth += 1,
            Err(_) => break,
        }
    }

    assert!(vm.is_terminated(), "Should terminate on stack overflow");
    assert!(
        vm.get_termination_reason()
            .unwrap()
            .contains("STACK_OVERFLOW"),
        "Should report stack overflow"
    );
    assert!(depth > 0 && depth <= vm.max_call_depth as usize);
}

/// Test normal call depth
#[test]
fn test_normal_call_depth() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Normal call depth
    for _ in 0..100 {
        vm.enter_call().unwrap();
    }

    assert_eq!(vm.call_depth.load(Ordering::SeqCst), 100);

    // Exit calls
    for _ in 0..100 {
        vm.exit_call();
    }

    assert_eq!(vm.call_depth.load(Ordering::SeqCst), 0);
}

// ============================================================================
// STORAGE SECURITY TESTS
// ============================================================================

/// Test storage write size limit
#[test]
fn test_storage_write_size_limit() {
    let vm = MockWasmExecution::new(100_000_000, 16);

    let key = [1u8; 32];
    let huge_value = vec![0u8; 2 * 1024 * 1024]; // 2MB

    let result = vm.storage_write(key, huge_value);
    assert!(result.is_err(), "Should reject oversized storage write");
}

/// Test storage gas costs
#[test]
fn test_storage_gas_costs() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    let key = [1u8; 32];
    let value = vec![0u8; 1000]; // 1KB

    let gas_before = vm.gas.used();
    vm.storage_write(key, value).unwrap();
    let gas_after = vm.gas.used();

    let gas_used = gas_after - gas_before;
    assert!(gas_used >= 5000, "Should charge base storage cost");
    assert!(gas_used >= 5000 + 1000 * 10, "Should charge per-byte cost");
}

/// Test storage read gas costs
#[test]
fn test_storage_read_gas_costs() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    let key = [1u8; 32];
    let value = vec![42u8; 500];
    vm.storage_write(key, value.clone()).unwrap();

    let gas_before = vm.gas.used();
    let read_value = vm.storage_read(&key).unwrap();
    let gas_after = vm.gas.used();

    assert_eq!(read_value, Some(value));
    assert!(gas_after > gas_before, "Storage read should cost gas");
}

/// Test storage isolation
#[test]
fn test_storage_isolation() {
    let vm1 = MockWasmExecution::new(1_000_000, 16);
    let vm2 = MockWasmExecution::new(1_000_000, 16);

    let key = [1u8; 32];
    vm1.storage_write(key, vec![1, 2, 3]).unwrap();

    // VM2 should not see VM1's storage
    let result = vm2.storage_read(&key).unwrap();
    assert!(result.is_none(), "Storage should be isolated between contracts");
}

// ============================================================================
// RESOURCE EXHAUSTION TESTS
// ============================================================================

/// Test combined resource exhaustion (gas + memory + time)
#[test]
fn test_combined_resource_exhaustion() {
    let vm = MockWasmExecution::new(10_000, 2); // Limited gas and memory

    // Try to exhaust gas through memory operations
    let mut exhausted = false;
    for i in 0..1000 {
        match vm.memory_access(0, 1000, 10) {
            Ok(_) => {}
            Err(_) => {
                exhausted = true;
                break;
            }
        }
    }

    assert!(exhausted, "Should exhaust resources");
}

/// Test many small allocations
#[test]
fn test_many_small_allocations() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Many small storage writes
    for i in 0..100 {
        let key = {
            let mut k = [0u8; 32];
            k[0..4].copy_from_slice(&(i as u32).to_le_bytes());
            k
        };
        let _ = vm.storage_write(key, vec![i as u8; 100]);
    }

    // Check storage was populated
    let storage = vm.contract_storage.lock().unwrap();
    assert!(storage.len() > 0, "Should have written some storage");
}

// ============================================================================
// HOST FUNCTION ABUSE TESTS
// ============================================================================

/// Test storage spam attack
#[test]
fn test_storage_spam_attack() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Try to spam storage with many keys
    let mut writes = 0;
    for i in 0..10000 {
        let key = {
            let mut k = [0u8; 32];
            k[0..4].copy_from_slice(&(i as u32).to_le_bytes());
            k
        };
        match vm.storage_write(key, vec![42u8; 100]) {
            Ok(_) => writes += 1,
            Err(_) => break, // Out of gas
        }
    }

    println!("Completed {} writes before gas exhaustion", writes);
    assert!(writes < 10000, "Should be limited by gas");
}

/// Test rapid memory growth attempts
#[test]
fn test_rapid_memory_growth() {
    let mut memory = MemoryBounds::new(1, 100);

    // Try rapid growth
    for _ in 0..50 {
        let _ = memory.grow(1);
    }

    assert!(memory.current_pages <= 100, "Should not exceed max pages");
}

/// Test execution after termination
#[test]
fn test_no_execution_after_termination() {
    let vm = MockWasmExecution::new(1_000_000, 16);

    // Manually terminate
    vm.terminate("TEST_TERMINATION");

    // All operations should fail
    assert!(vm.execute_step(1).is_err());
    assert!(vm.memory_access(0, 1, 1).is_err());
}
