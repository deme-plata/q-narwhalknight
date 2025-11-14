# Data Parallelism Testing Plan - Phase 5

**Date**: 2025-01-12
**Session**: Safe Batched Sync v1.0.2 Branch
**Status**: 🧪 **PHASE 5 TESTING PLAN**

---

## 🎯 TESTING OBJECTIVES

1. **Verify functionality** - System works end-to-end
2. **Measure performance** - Confirm linear scaling (N nodes = N× throughput)
3. **Test reliability** - Failover, error handling, edge cases
4. **Validate load balancing** - Requests distributed evenly
5. **Confirm backwards compatibility** - Old endpoints still work

---

## 🧪 TEST SEQUENCE

### **Test 1: Compilation Verification** ✅

**Status**: COMPLETE
**Result**: Compiles successfully in 3m 19s with 0 errors

---

### **Test 2: Single-Node Baseline** (15 min)

**Goal**: Establish baseline performance with 1 worker

**Setup**:
```bash
# Build release binary
timeout 36000 cargo build --release --package q-api-server

# Start API server with full-model engine
Q_DB_PATH=./data-test-single \
Q_P2P_PORT=9001 \
./target/release/q-api-server --port 8001 --node-id node1
```

**Test Steps**:
1. Wait for model to load
2. Create chat session
3. Send test request to `/api/chat/:id/stream-distributed`
4. Verify:
   - ✅ Tokens stream in real-time
   - ✅ SSE events received (started, token, complete)
   - ✅ Chat message saved to database
5. Measure:
   - First token latency
   - Total tokens generated
   - Tokens per second
   - Total time

**Expected Result**:
```
Baseline Performance:
- First token: ~2 seconds
- Throughput: 0.7-0.8 tokens/sec
- Latency: ~1280ms/token
- Total time for 50 tokens: ~64 seconds
```

**Success Criteria**:
- ✅ Request completes successfully
- ✅ All tokens received
- ✅ Performance matches single-node expectations

---

### **Test 3: Worker Initialization** (30 min)

**Goal**: Verify worker can load full model and register

**Setup**:
```bash
# Terminal 1: Start coordinator node
Q_DB_PATH=./data-coordinator \
Q_P2P_PORT=9001 \
./target/release/q-api-server --port 8001 --node-id coordinator

# Terminal 2: Start worker node with full-model engine
Q_DB_PATH=./data-worker1 \
Q_P2P_PORT=9002 \
./target/release/q-api-server --port 8002 --node-id worker1 \
  --bootstrap-peer "/ip4/127.0.0.1/tcp/9001/p2p/<coordinator-peer-id>"
```

**Test Steps**:
1. Verify worker loads full-model engine
2. Verify worker registers with coordinator
3. Check coordinator sees 1 available worker
4. Send test request
5. Verify request routes to worker1

**Expected Result**:
```
Coordinator logs:
🌐 Registered AI worker: worker1
✅ Data parallel request routed to worker worker1

Worker logs:
🚀 Initializing full-model MistralRsEngine for data parallelism
🎯 Worker worker1 received TARGETED inference request
```

**Success Criteria**:
- ✅ Worker registers successfully
- ✅ Request routes to correct worker
- ✅ Worker skips non-targeted requests

---

### **Test 4: 2-Node Data Parallel** (30 min)

**Goal**: Verify 2× throughput with 2 workers

**Setup**:
```bash
# Terminal 1: Coordinator
Q_DB_PATH=./data-coord ./target/release/q-api-server --port 8001 --node-id coordinator

# Terminal 2: Worker 1
Q_DB_PATH=./data-w1 ./target/release/q-api-server --port 8002 --node-id worker1 --bootstrap-peer <coordinator>

# Terminal 3: Worker 2
Q_DB_PATH=./data-w2 ./target/release/q-api-server --port 8003 --node-id worker2 --bootstrap-peer <coordinator>
```

**Test Steps**:
1. Send 2 concurrent requests
2. Verify each routes to different worker
3. Measure aggregate throughput
4. Compare to single-node baseline

**Test Script**:
```bash
#!/bin/bash
# Send 2 concurrent requests
curl -X POST http://localhost:8001/api/chat/chat1/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Request 1"}' &

curl -X POST http://localhost:8001/api/chat/chat2/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Request 2"}' &

wait
```

**Expected Result**:
```
2-Node Performance:
- Request 1 → worker1: 0.78 tok/s
- Request 2 → worker2: 0.78 tok/s
- Aggregate throughput: 1.56 tok/s (2× improvement! ✅)
- Per-user latency: ~1280ms/token (UNCHANGED! ✅)
```

**Success Criteria**:
- ✅ Requests route to different workers
- ✅ Aggregate throughput = 2× baseline
- ✅ Per-user latency unchanged

---

### **Test 5: 4-Node Cluster** (45 min)

**Goal**: Verify 4× throughput with 4 workers

**Setup**:
```bash
# Start coordinator + 4 workers (similar to Test 4)
```

**Test Steps**:
1. Send 10 concurrent requests
2. Verify load balancing (each worker gets ~2-3 requests)
3. Measure aggregate throughput
4. Monitor system resources

**Test Script**:
```bash
#!/bin/bash
for i in {1..10}; do
  curl -X POST http://localhost:8001/api/chat/chat$i/stream-distributed \
    -H "Content-Type: application/json" \
    -d "{\"content\":\"Request $i\"}" &
done
wait
```

**Expected Result**:
```
4-Node Performance:
- Worker 1: 2 requests, 1.56 tok/s
- Worker 2: 3 requests, 2.34 tok/s
- Worker 3: 2 requests, 1.56 tok/s
- Worker 4: 3 requests, 2.34 tok/s
- Aggregate throughput: ~3.12 tok/s (4× improvement! ✅)
- Memory: 4 × 4.4GB = 17.6GB
```

**Success Criteria**:
- ✅ All 10 requests complete successfully
- ✅ Load balanced across 4 workers
- ✅ Aggregate throughput = 4× baseline
- ✅ No errors or timeouts

---

### **Test 6: Load Balancing Strategies** (30 min)

**Goal**: Verify load balancer selects optimal worker

**Test Scenarios**:

#### **A. Least-Loaded Selection**
```
State:
- Worker 1: 0 active requests
- Worker 2: 2 active requests
- Worker 3: 1 active request

Expected: New request → Worker 1 (least loaded)
```

#### **B. Round-Robin (if implemented)**
```
Requests 1,2,3,4 → Workers 1,2,3,4 in order
```

#### **C. Capability-Aware (future)**
```
GPU worker preferred over CPU worker
```

**Test Steps**:
1. Manually create load imbalance
2. Send new request
3. Verify correct worker selected
4. Check coordinator logs

**Success Criteria**:
- ✅ Least-loaded worker selected
- ✅ No worker overloaded while others idle

---

### **Test 7: Failover & Timeout** (30 min)

**Goal**: Verify system handles worker failures gracefully

**Test Scenarios**:

#### **A. Worker Crash Mid-Inference**
```bash
# Start 4 workers, send request to worker2
# Kill worker2 during generation
kill -9 <worker2-pid>

# Expected: 3-second timeout, error returned to client
```

#### **B. Worker Unresponsive (No InferenceStarted)**
```
# Worker receives request but doesn't send InferenceStarted
# Expected: Coordinator times out after 3 seconds
```

#### **C. Worker Disappears After Started**
```
# Worker sends InferenceStarted, then crashes
# Expected: Client receives partial tokens, then error event
```

**Test Steps**:
1. Start cluster with 4 workers
2. Send request
3. Kill target worker at various stages
4. Verify error handling

**Expected Behavior**:
```
Coordinator logs:
⚠️  Worker worker2 timeout (no InferenceStarted after 3s)
❌ Removing pending request abc123

Client receives:
event: error
data: {"code":"worker_timeout","message":"Worker did not respond"}
```

**Success Criteria**:
- ✅ Timeout detected within 3-4 seconds
- ✅ Error event sent to client
- ✅ Request cleaned up (no memory leak)
- ✅ System continues working with remaining workers

---

### **Test 8: Edge Cases** (30 min)

**Goal**: Verify system handles edge cases

**Test Cases**:

#### **A. No Workers Available**
```bash
# Send request with 0 workers registered
# Expected: Error "No available workers"
```

#### **B. Worker Receives Non-Targeted Request**
```bash
# Send request targeted at worker1
# Verify worker2 skips it silently
```

#### **C. Duplicate Token Index**
```bash
# Worker sends same token_index twice (gossipsub duplicate)
# Expected: Coordinator drops duplicate
```

#### **D. Out-of-Order Tokens**
```bash
# Tokens arrive: 0,1,3,2,4
# Expected: Coordinator enforces order, drops 2
```

#### **E. Very Long Response (500 tokens)**
```bash
# Request max_tokens=500
# Expected: All 500 tokens stream correctly
```

**Success Criteria**:
- ✅ All edge cases handled gracefully
- ✅ No panics or crashes
- ✅ Appropriate error messages

---

### **Test 9: Performance Benchmarks** (1 hour)

**Goal**: Measure and document performance metrics

**Metrics to Capture**:

1. **Single-Node Baseline**:
   - First token latency (ms)
   - Tokens per second
   - Total time for 50/100/150 tokens

2. **Multi-Node Scaling**:
   - 2 nodes: Aggregate throughput
   - 4 nodes: Aggregate throughput
   - 8 nodes: Aggregate throughput (if resources available)

3. **Resource Usage**:
   - Memory per worker
   - CPU usage per worker
   - Network bandwidth

4. **Latency Breakdown**:
   - Request → Coordinator: <1ms (local)
   - Coordinator → Worker: <50ms (gossipsub)
   - Worker → First token: ~2000ms (model)
   - Token → Token: ~1280ms (generation)

**Benchmarking Script**:
```bash
#!/bin/bash
# benchmark.sh - Measure data parallel performance

echo "=== Data Parallelism Benchmark ==="
echo "Date: $(date)"
echo ""

# Test 1: Single-node baseline
echo "Test 1: Single-node baseline"
time curl -X POST http://localhost:8001/api/chat/test1/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Generate 50 tokens"}' > /dev/null

# Test 2: 2 concurrent requests (2 nodes)
echo "Test 2: 2-node cluster"
time (
  curl -X POST http://localhost:8001/api/chat/test2a/stream-distributed \
    -H "Content-Type: application/json" \
    -d '{"content":"Request A"}' > /dev/null &
  curl -X POST http://localhost:8001/api/chat/test2b/stream-distributed \
    -H "Content-Type: application/json" \
    -d '{"content":"Request B"}' > /dev/null &
  wait
)

# Test 3: 4 concurrent requests (4 nodes)
echo "Test 3: 4-node cluster"
time (
  for i in {1..4}; do
    curl -X POST http://localhost:8001/api/chat/test3$i/stream-distributed \
      -H "Content-Type: application/json" \
      -d "{\"content\":\"Request $i\"}" > /dev/null &
  done
  wait
)
```

**Success Criteria**:
- ✅ Linear scaling verified (within 10% margin)
- ✅ Performance documented for reference
- ✅ Bottlenecks identified (if any)

---

### **Test 10: Backwards Compatibility** (15 min)

**Goal**: Verify old endpoints still work

**Test Steps**:
1. Send request to old `/api/chat/:id/stream` endpoint
2. Verify pipeline parallelism still works
3. Send request to `/api/chat/:id/message` (non-streaming)
4. Verify synchronous API still works

**Success Criteria**:
- ✅ Old endpoints unchanged
- ✅ Both pipeline and data parallel coexist
- ✅ No breaking changes

---

## 📊 TESTING METRICS

| Test | Duration | Status | Result |
|------|----------|--------|--------|
| 1. Compilation | 3m 19s | ✅ PASS | 0 errors |
| 2. Single-node | 15 min | ⏳ | TBD |
| 3. Worker init | 30 min | ⏳ | TBD |
| 4. 2-node cluster | 30 min | ⏳ | TBD |
| 5. 4-node cluster | 45 min | ⏳ | TBD |
| 6. Load balancing | 30 min | ⏳ | TBD |
| 7. Failover | 30 min | ⏳ | TBD |
| 8. Edge cases | 30 min | ⏳ | TBD |
| 9. Benchmarks | 1 hour | ⏳ | TBD |
| 10. Backwards compat | 15 min | ⏳ | TBD |
| **TOTAL** | **~5 hours** | **10% DONE** | **1/10 PASS** |

---

## 🐛 KNOWN ISSUES TO WATCH

1. **Worker registration timing**
   - Workers may need time to register before first request
   - Solution: Wait 5-10 seconds after startup

2. **Model loading time**
   - Full model takes ~30-60 seconds to load
   - Solution: Check logs for "✅ Engine initialized"

3. **Gossipsub message order**
   - Tokens may arrive out of order
   - Solution: Token index enforcement (already implemented)

4. **3-Second timeout may be too short**
   - Cold start takes longer
   - Solution: May need to increase timeout to 10s

---

## 🚀 DEPLOYMENT CHECKLIST

Before deploying to production:

- [ ] All 10 tests pass
- [ ] Performance meets expectations (linear scaling)
- [ ] No memory leaks (monitor for 1+ hours)
- [ ] Error handling tested thoroughly
- [ ] Documentation complete
- [ ] Frontend integration tested
- [ ] Monitoring/metrics in place

---

## 📝 NEXT STEPS

1. **Immediate** (Today):
   - Run Test 2: Single-node baseline
   - Document baseline performance
   - Run Test 3: Worker initialization

2. **Tomorrow**:
   - Run Tests 4-7: Multi-node clusters
   - Measure scaling performance
   - Test failover scenarios

3. **This Week**:
   - Complete all 10 tests
   - Fix any issues found
   - Deploy to staging environment

---

**Ready to start testing! Let's verify that linear scaling! 📈**

**Generated**: 2025-01-12
**Branch**: feature/safe-batched-sync-v1.0.2
**Status**: 🧪 **TESTING PLAN COMPLETE - READY TO EXECUTE**
