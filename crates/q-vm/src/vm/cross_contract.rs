/// Cross-Contract Call Implementation
///
/// Enables smart contracts to call functions in other deployed contracts,
/// creating composable DeFi protocols and complex inter-contract interactions.

use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vm::VmError;
use crate::vm::ultra_performance_bridge::StateDB;

use wasmer::{Store, Module, Instance, imports, Value, Function, FunctionEnv, FunctionEnvMut, Memory};

/// Call frame for tracking cross-contract call stack
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub contract_address: String,
    pub function_name: String,
    pub caller: String,
    pub depth: usize,
    pub gas_used: u64,
    pub gas_limit: u64,
}

/// Reentrancy protection guard
pub struct ReentrancyGuard {
    active_calls: Arc<DashMap<String, bool>>,
}

impl ReentrancyGuard {
    pub fn new() -> Self {
        Self {
            active_calls: Arc::new(DashMap::new()),
        }
    }

    /// Check if contract is already executing and lock it
    pub fn check_and_lock(&self, contract_address: &str) -> Result<(), VmError> {
        if self.active_calls.contains_key(contract_address) {
            return Err(VmError::ExecutionError(
                format!("Reentrancy detected: Contract {} is already executing", contract_address)
            ));
        }

        self.active_calls.insert(contract_address.to_string(), true);
        Ok(())
    }

    /// Unlock contract after execution
    pub fn unlock(&self, contract_address: &str) {
        self.active_calls.remove(contract_address);
    }
}

/// Gas costs for cross-contract operations
#[derive(Debug, Clone)]
pub struct CrossContractGasCosts {
    /// Base cost for cross-contract call
    pub call_base: u64,

    /// Per-byte cost for arguments
    pub call_arg_per_byte: u64,

    /// Per-byte cost for return data
    pub call_return_per_byte: u64,

    /// Per-depth gas penalty
    pub call_depth_penalty: u64,

    /// Module compilation (first time)
    pub module_compile: u64,

    /// Module cached (from cache)
    pub module_cached: u64,
}

impl Default for CrossContractGasCosts {
    fn default() -> Self {
        Self {
            call_base: 10_000,
            call_arg_per_byte: 100,
            call_return_per_byte: 100,
            call_depth_penalty: 1_000,
            module_compile: 100_000,
            module_cached: 1_000,
        }
    }
}

/// Cross-contract call result
#[derive(Debug, Clone)]
pub struct CrossContractResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
}

/// Cross-contract call handler
pub struct CrossContractCallHandler {
    /// State database reference
    state_db: Arc<StateDB>,

    /// Call stack for tracking depth
    call_stack: Arc<Mutex<Vec<CallFrame>>>,

    /// Reentrancy protection
    reentrancy_guard: ReentrancyGuard,

    /// Gas cost configuration
    gas_costs: CrossContractGasCosts,

    /// Maximum call depth (default: 16)
    max_call_depth: usize,

    /// Gas usage tracker
    gas_tracker: Arc<AtomicU64>,
}

impl CrossContractCallHandler {
    /// Create new cross-contract call handler
    pub fn new(state_db: Arc<StateDB>) -> Self {
        Self {
            state_db,
            call_stack: Arc::new(Mutex::new(Vec::new())),
            reentrancy_guard: ReentrancyGuard::new(),
            gas_costs: CrossContractGasCosts::default(),
            max_call_depth: 16,
            gas_tracker: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Configure gas costs
    pub fn with_gas_costs(mut self, costs: CrossContractGasCosts) -> Self {
        self.gas_costs = costs;
        self
    }

    /// Configure maximum call depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_call_depth = max_depth;
        self
    }

    /// Calculate gas cost for cross-contract call
    fn calculate_gas_cost(
        &self,
        args_len: usize,
        return_len: usize,
        call_depth: usize,
        module_cached: bool,
    ) -> u64 {
        let mut total_gas = self.gas_costs.call_base;

        // Argument cost
        total_gas += (args_len as u64) * self.gas_costs.call_arg_per_byte;

        // Return data cost
        total_gas += (return_len as u64) * self.gas_costs.call_return_per_byte;

        // Depth penalty
        total_gas += (call_depth as u64) * self.gas_costs.call_depth_penalty;

        // Module compilation cost
        total_gas += if module_cached {
            self.gas_costs.module_cached
        } else {
            self.gas_costs.module_compile
        };

        total_gas
    }

    /// Propagate gas limit (63/64 rule from Ethereum)
    fn propagate_gas_limit(
        &self,
        caller_gas_remaining: u64,
        call_gas_requested: u64,
    ) -> Result<u64, VmError> {
        // Callee gets min(requested, 63/64 of caller's remaining gas)
        let max_callee_gas = (caller_gas_remaining * 63) / 64;

        if call_gas_requested > max_callee_gas {
            Ok(max_callee_gas)
        } else {
            Ok(call_gas_requested)
        }
    }

    /// Execute cross-contract call
    pub async fn execute_cross_contract_call(
        &self,
        caller_address: &str,
        target_address: &str,
        function_name: &str,
        args: &[u8],
        caller_gas_remaining: u64,
        call_gas_requested: u64,
    ) -> Result<CrossContractResult, VmError> {
        // 1. Check call depth
        let call_stack = self.call_stack.lock().await;
        let current_depth = call_stack.len();

        if current_depth >= self.max_call_depth {
            return Err(VmError::ExecutionError(
                format!("Maximum call depth {} exceeded", self.max_call_depth)
            ));
        }
        drop(call_stack);

        // 2. Reentrancy check
        self.reentrancy_guard.check_and_lock(target_address)?;

        // 3. Calculate gas limit for callee
        let callee_gas_limit = self.propagate_gas_limit(
            caller_gas_remaining,
            call_gas_requested,
        )?;

        // 4. Load target contract bytecode
        let target_bytecode = self.state_db.get_contract(target_address)
            .map_err(|e| VmError::ExecutionError(
                format!("Failed to load contract {}: {}", target_address, e)
            ))?;

        // 5. Check module cache
        let bytecode_hash = Self::hash_bytecode(&target_bytecode);
        let module_cached = self.state_db.module_cache.contains_key(&bytecode_hash);

        let module = if let Some(cached_module) = self.state_db.module_cache.get(&bytecode_hash) {
            cached_module.clone()
        } else {
            // Compile and cache
            let mut store = Store::default();
            let compiled = Arc::new(Module::new(&store, &target_bytecode)
                .map_err(|e| VmError::ExecutionError(
                    format!("Failed to compile contract {}: {}", target_address, e)
                ))?);

            self.state_db.module_cache.insert(bytecode_hash, compiled.clone());
            compiled
        };

        // 6. Calculate gas overhead
        let gas_overhead = self.calculate_gas_cost(
            args.len(),
            0, // Return length unknown at this point
            current_depth + 1,
            module_cached,
        );

        // 7. Setup WASM environment
        let mut store = Store::default();
        let env = WasmEnv {
            state_db: self.state_db.clone(),
            memory: None,
        };
        let function_env = FunctionEnv::new(&mut store, env);

        // 8. Create imports
        let import_object = imports! {
            "env" => {
                "read_state" => Function::new_typed_with_env(&mut store, &function_env, Self::read_state_host),
                "write_state" => Function::new_typed_with_env(&mut store, &function_env, Self::write_state_host),
            }
        };

        // 9. Instantiate WASM module
        let instance = Instance::new(&mut store, &*module, &import_object)
            .map_err(|e| VmError::ExecutionError(
                format!("Failed to instantiate contract {}: {}", target_address, e)
            ))?;

        // 10. Get exported memory
        let memory = instance.exports.get_memory("memory")
            .map_err(|e| VmError::ExecutionError(
                format!("Contract {} has no exported memory: {}", target_address, e)
            ))?;

        // 11. Get exported function
        let wasm_function = instance.exports.get_function(function_name)
            .map_err(|e| VmError::ExecutionError(
                format!("Function {} not found in contract {}: {}", function_name, target_address, e)
            ))?;

        // 12. Prepare arguments (write to WASM memory at address 1000)
        let args_ptr = 1000i32;
        memory.view(&store).write(args_ptr as u64, args)
            .map_err(|e| VmError::ExecutionError(
                format!("Failed to write args to WASM memory: {}", e)
            ))?;

        // 13. Push call frame
        let call_frame = CallFrame {
            contract_address: target_address.to_string(),
            function_name: function_name.to_string(),
            caller: caller_address.to_string(),
            depth: current_depth + 1,
            gas_used: gas_overhead,
            gas_limit: callee_gas_limit,
        };

        let mut call_stack = self.call_stack.lock().await;
        call_stack.push(call_frame);
        drop(call_stack);

        // 14. Execute function
        let result = wasm_function.call(&mut store, &[Value::I32(args_ptr), Value::I32(args.len() as i32)])
            .map_err(|e| {
                // Cleanup on error
                self.reentrancy_guard.unlock(target_address);
                VmError::ExecutionError(
                    format!("WASM execution failed in {}: {}", target_address, e)
                )
            })?;

        // 15. Extract return value
        let return_value = match &result[..] {
            [Value::I32(val)] => *val,
            _ => 0,
        };

        // 16. Read return data from WASM memory (assume written at address 2000)
        let return_data_ptr = 2000u64;
        let return_data_len = 32; // Fixed size for now
        let mut return_data = vec![0u8; return_data_len];
        memory.view(&store).read(return_data_ptr, &mut return_data)
            .map_err(|e| VmError::ExecutionError(
                format!("Failed to read return data: {}", e)
            ))?;

        // 17. Calculate total gas
        let return_gas_cost = (return_data.len() as u64) * self.gas_costs.call_return_per_byte;
        let total_gas = gas_overhead + return_gas_cost;

        // 18. Pop call frame
        let mut call_stack = self.call_stack.lock().await;
        call_stack.pop();
        drop(call_stack);

        // 19. Unlock reentrancy guard
        self.reentrancy_guard.unlock(target_address);

        // 20. Update gas tracker
        self.gas_tracker.fetch_add(total_gas, Ordering::SeqCst);

        Ok(CrossContractResult {
            success: return_value != 0,
            return_data,
            gas_used: total_gas,
        })
    }

    /// Get current call depth
    pub async fn get_call_depth(&self) -> usize {
        self.call_stack.lock().await.len()
    }

    /// Get total gas used across all cross-contract calls
    pub fn get_total_gas_used(&self) -> u64 {
        self.gas_tracker.load(Ordering::SeqCst)
    }

    /// Hash bytecode for caching
    fn hash_bytecode(bytecode: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        bytecode.hash(&mut hasher);
        hasher.finish()
    }

    // Host function implementations
    fn read_state_host(mut env: FunctionEnvMut<WasmEnv>, key_ptr: i32, key_len: i32, value_ptr: i32) -> i32 {
        let (data, store) = env.data_and_store_mut();
        let memory = data.memory.as_ref().unwrap();

        // Read key from WASM memory
        let mut key = vec![0u8; key_len as usize];
        if memory.view(&store).read(key_ptr as u64, &mut key).is_err() {
            return 0;
        }

        // Read from state
        if let Some(value) = data.state_db.read_state(&key) {
            // Write value to WASM memory
            if memory.view(&store).write(value_ptr as u64, &value).is_ok() {
                return 1;
            }
        }

        0
    }

    fn write_state_host(mut env: FunctionEnvMut<WasmEnv>, key_ptr: i32, key_len: i32, value_ptr: i32, value_len: i32) -> i32 {
        let (data, store) = env.data_and_store_mut();
        let memory = data.memory.as_ref().unwrap();

        // Read key and value from WASM memory
        let mut key = vec![0u8; key_len as usize];
        let mut value = vec![0u8; value_len as usize];

        if memory.view(&store).read(key_ptr as u64, &mut key).is_err() {
            return 0;
        }

        if memory.view(&store).read(value_ptr as u64, &mut value).is_err() {
            return 0;
        }

        // Write to state
        data.state_db.write_state(key, value);
        1
    }
}

/// WASM environment for host functions
#[derive(Clone)]
pub struct WasmEnv {
    pub state_db: Arc<StateDB>,
    pub memory: Option<Memory>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_call_depth_limit() {
        let state_db = Arc::new(StateDB::new());
        let handler = CrossContractCallHandler::new(state_db).with_max_depth(3);

        // Simulate deep call stack
        for i in 0..3 {
            let mut stack = handler.call_stack.lock().await;
            stack.push(CallFrame {
                contract_address: format!("0x{}", i),
                function_name: "test".to_string(),
                caller: "0xCaller".to_string(),
                depth: i,
                gas_used: 10_000,
                gas_limit: 100_000,
            });
        }

        // This should fail due to max depth
        let result = handler.execute_cross_contract_call(
            "0xCaller",
            "0xTarget",
            "test",
            &[],
            1_000_000,
            100_000,
        ).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Maximum call depth"));
    }

    #[tokio::test]
    async fn test_reentrancy_protection() {
        let state_db = Arc::new(StateDB::new());
        let handler = CrossContractCallHandler::new(state_db);

        // Lock contract
        handler.reentrancy_guard.check_and_lock("0xContract").unwrap();

        // Second lock should fail
        let result = handler.reentrancy_guard.check_and_lock("0xContract");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Reentrancy detected"));

        // Unlock
        handler.reentrancy_guard.unlock("0xContract");

        // Now should succeed
        assert!(handler.reentrancy_guard.check_and_lock("0xContract").is_ok());
    }

    #[test]
    fn test_gas_calculation() {
        let state_db = Arc::new(StateDB::new());
        let handler = CrossContractCallHandler::new(state_db);

        // Test gas calculation
        let gas = handler.calculate_gas_cost(
            100,  // args_len
            50,   // return_len
            2,    // call_depth
            false, // module_cached
        );

        // Base (10k) + args (100*100=10k) + return (50*100=5k) + depth (2*1k=2k) + compile (100k) = 127k
        assert_eq!(gas, 127_000);

        // Test with cached module
        let gas_cached = handler.calculate_gas_cost(
            100,  // args_len
            50,   // return_len
            2,    // call_depth
            true, // module_cached
        );

        // Base (10k) + args (10k) + return (5k) + depth (2k) + cached (1k) = 28k
        assert_eq!(gas_cached, 28_000);
    }

    #[test]
    fn test_gas_propagation() {
        let state_db = Arc::new(StateDB::new());
        let handler = CrossContractCallHandler::new(state_db);

        // Test 63/64 rule
        let caller_gas = 1_000_000;
        let requested_gas = 500_000;

        let propagated = handler.propagate_gas_limit(caller_gas, requested_gas).unwrap();

        // Should get requested amount (500k < 63/64 * 1M = 984,375)
        assert_eq!(propagated, 500_000);

        // Test with excessive request
        let excessive_request = 1_000_000;
        let propagated = handler.propagate_gas_limit(caller_gas, excessive_request).unwrap();

        // Should get capped at 63/64 of caller's gas
        assert_eq!(propagated, (caller_gas * 63) / 64);
    }
}
