# Data Parallelism Worker Verification Guide

**Date**: 2025-01-12
**Version**: v1.0
**Purpose**: How to verify compute happens on connected peers

---

## 🎯 VERIFICATION OVERVIEW

This guide shows you how to verify that AI inference is actually happening on **different workers** when you have multiple peers connected. Without verification, you can't tell if the system is actually using data parallelism or just running everything on one node!

---

## 🔍 VERIFICATION METHODS

### **Method 1: Server Console Logs** ⭐ **MOST RELIABLE**

Each worker prints highly visible box logs when it accepts and completes inference requests.

#### **What to Look For**:

When a worker receives and processes a request, you'll see:

```
╔═══════════════════════════════════════════════════════════════╗
║ 🎯 DATA PARALLEL INFERENCE REQUEST ACCEPTED                 ║
╠═══════════════════════════════════════════════════════════════╣
║ Worker:      worker-1 (THIS NODE)                             ║
║ Request ID:  550e8400-e29b-41d4-a716-446655440000              ║
║ Prompt:      Hello, how are you today?                        ║
║ Max tokens:  150                                               ║
║ Temperature: 0.7                                               ║
╚═══════════════════════════════════════════════════════════════╝

🧠 Worker generating 150 tokens with streaming...
... (tokens stream in real-time) ...

╔═══════════════════════════════════════════════════════════════╗
║ ✅ DATA PARALLEL INFERENCE COMPLETED                        ║
╠═══════════════════════════════════════════════════════════════╣
║ Worker:      worker-1 (THIS NODE)                             ║
║ Request ID:  550e8400-e29b-41d4-a716-446655440000              ║
║ Tokens:      50 generated                                      ║
║ Time:        64000ms                                           ║
║ Throughput:  0.78 tokens/sec                                   ║
║ Generated:   Hello there! I'm doing great, thank you for as...║
╚═══════════════════════════════════════════════════════════════╝
```

#### **How to Test**:

1. **Start 2 nodes** (coordinator + worker):

```bash
# Terminal 1: Coordinator
Q_DB_PATH=./data-coordinator \
Q_P2P_PORT=9001 \
RUST_LOG=info \
./target/release/q-api-server --port 8001 --node-id coordinator

# Terminal 2: Worker
Q_DB_PATH=./data-worker1 \
Q_P2P_PORT=9002 \
RUST_LOG=info \
./target/release/q-api-server --port 8002 --node-id worker1 \
  --bootstrap-peer "/ip4/127.0.0.1/tcp/9001/p2p/<coordinator-peer-id>"
```

2. **Send 2 concurrent requests** from browser or curl:

```bash
# Request 1
curl -X POST http://localhost:8001/api/chat/chat1/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello from user 1"}' &

# Request 2
curl -X POST http://localhost:8001/api/chat/chat2/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello from user 2"}' &
```

3. **Watch both terminal windows**:
   - Terminal 1 should show: Worker: coordinator (THIS NODE) for request 1
   - Terminal 2 should show: Worker: worker1 (THIS NODE) for request 2

**✅ SUCCESS**: If you see the boxes appear in **different terminals**, that proves different workers are processing requests!

---

### **Method 2: API Endpoint** `/api/chat/workers`

Query the API to see which workers are connected and online.

#### **Request**:

```bash
curl http://localhost:8001/api/chat/workers | jq
```

#### **Response**:

```json
{
  "success": true,
  "data": {
    "workers": [
      {
        "node_id": "coordinator",
        "peer_id": "12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN",
        "active_requests": 0,
        "capability": "CPU",
        "status": "online"
      },
      {
        "node_id": "worker1",
        "peer_id": "12D3KooWMB6x1ZpXxYzNd4R8sGJ7nQ5vZ1kL9tA2fK3pW8cH4mN7",
        "active_requests": 1,
        "capability": "CPU",
        "status": "online"
      }
    ],
    "total_workers": 2,
    "coordinator_node_id": "coordinator"
  },
  "timestamp": 1705071234
}
```

#### **What to Check**:

- ✅ `total_workers` should be > 1 (multiple workers connected)
- ✅ Each worker should have status "online"
- ✅ `active_requests` shows load distribution

---

### **Method 3: Frontend Browser Console**

The frontend logs which worker handled each request to the browser console.

#### **How to Check**:

1. Open browser to https://quillon.xyz
2. Navigate to AI Chat screen
3. Open Browser Console (F12)
4. Send a message
5. Look for logs:

```javascript
🌊 Data parallel stream started on worker: worker1
✅ Complete: 50 tokens in 64000ms
   Throughput: 0.78 tok/s
   Worker: worker1, Mode: data_parallel
```

#### **With Multiple Requests**:

Send 2-3 messages quickly and watch the console logs. You should see different `worker:` values:

```
🌊 Data parallel stream started on worker: coordinator
✅ Complete: 50 tokens... Worker: coordinator, Mode: data_parallel

🌊 Data parallel stream started on worker: worker1
✅ Complete: 50 tokens... Worker: worker1, Mode: data_parallel

🌊 Data parallel stream started on worker: coordinator
✅ Complete: 50 tokens... Worker: coordinator, Mode: data_parallel
```

**✅ SUCCESS**: Different messages handled by different workers = load balancing working!

---

## 🧪 TESTING SCENARIOS

### **Scenario 1: Single Node (Baseline)**

**Setup**: Run only the coordinator (no workers)

**Expected**:
- Coordinator processes all requests itself
- Console shows: Worker: coordinator (THIS NODE)
- `/api/chat/workers` shows: total_workers: 1

**Verification**:
```bash
# Start coordinator only
Q_DB_PATH=./data-single RUST_LOG=info ./target/release/q-api-server --port 8001 --node-id coordinator

# Check workers
curl http://localhost:8001/api/chat/workers | jq '.data.total_workers'
# Expected: 1

# Send request
curl -X POST http://localhost:8001/api/chat/test/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Test message"}'

# Check console - should see box with "Worker: coordinator (THIS NODE)"
```

---

### **Scenario 2: Two-Node Cluster (2× Scaling)**

**Setup**: Coordinator + 1 worker

**Expected**:
- 2 concurrent requests go to different nodes
- Console shows boxes in BOTH terminal windows
- `/api/chat/workers` shows: total_workers: 2

**Verification**:
```bash
# Terminal 1: Coordinator
Q_DB_PATH=./data-coord RUST_LOG=info ./target/release/q-api-server --port 8001 --node-id coordinator

# Terminal 2: Worker
Q_DB_PATH=./data-w1 RUST_LOG=info ./target/release/q-api-server --port 8002 --node-id worker1 \
  --bootstrap-peer "/ip4/127.0.0.1/tcp/9001/p2p/<peer-id>"

# Check workers
curl http://localhost:8001/api/chat/workers | jq '.data.total_workers'
# Expected: 2

# Send 2 concurrent requests
curl -X POST http://localhost:8001/api/chat/test1/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Request 1"}' &

curl -X POST http://localhost:8001/api/chat/test2/stream-distributed \
  -H "Content-Type: application/json" \
  -d '{"content":"Request 2"}' &

wait

# Verify: Terminal 1 shows ONE box, Terminal 2 shows ONE box (different requests)
```

---

### **Scenario 3: Four-Node Cluster (4× Scaling)**

**Setup**: Coordinator + 3 workers

**Expected**:
- 4 concurrent requests distributed across all nodes
- Console shows boxes in ALL FOUR terminal windows
- `/api/chat/workers` shows: total_workers: 4

**Verification**:
```bash
# Start 4 nodes (coordinator + 3 workers)
# ... (similar to Scenario 2 but with more workers)

# Check workers
curl http://localhost:8001/api/chat/workers | jq '.data.workers[].node_id'
# Expected: ["coordinator", "worker1", "worker2", "worker3"]

# Send 10 concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8001/api/chat/test$i/stream-distributed \
    -H "Content-Type: application/json" \
    -d "{\"content\":\"Request $i\"}" &
done
wait

# Verify: All 4 terminals show boxes (load distributed)
```

---

## 📊 LOAD BALANCING VERIFICATION

The coordinator uses a **least-loaded strategy**. To verify:

### **Test Load Balancing**:

1. Start 3 workers
2. Send request 1 → Should go to coordinator (0 active requests)
3. While request 1 is still processing, send request 2 → Should go to worker1 (0 active requests)
4. Send request 3 → Should go to worker2 (0 active requests)

### **Check with API**:

```bash
# While requests are processing
curl http://localhost:8001/api/chat/workers | jq '.data.workers[] | {node_id, active_requests}'
```

**Expected Output** (while 3 requests running):
```json
{"node_id": "coordinator", "active_requests": 1}
{"node_id": "worker1", "active_requests": 1}
{"node_id": "worker2", "active_requests": 1}
```

---

## ⚠️ COMMON ISSUES

### **Issue 1: All Requests Go to Coordinator**

**Symptom**: Only coordinator terminal shows boxes, worker terminals silent

**Cause**: Workers not registering properly with coordinator

**Fix**:
1. Check worker logs for "Registered AI worker: worker1" message
2. Verify bootstrap peer ID is correct
3. Check network connectivity between nodes

### **Issue 2: Workers Show 0 Total**

**Symptom**: `/api/chat/workers` returns `total_workers: 1` despite multiple nodes running

**Cause**: Workers haven't sent heartbeat yet

**Fix**:
1. Wait 10 seconds after starting workers
2. Check worker logs for gossipsub connection
3. Verify --bootstrap-peer flag is set correctly

### **Issue 3: No Box Logs Appear**

**Symptom**: No colorful box logs in any terminal

**Cause**: Log level not set to INFO

**Fix**:
```bash
# Set RUST_LOG=info
RUST_LOG=info ./target/release/q-api-server ...
```

### **Issue 4: "target_node_id mismatch" Warnings**

**Symptom**: Worker logs show "Skipping TargetedInferenceRequest (target=coordinator, me=worker1)"

**Cause**: **THIS IS NORMAL!** It means the worker correctly skipped a request meant for another node

**Expected Behavior**: You should see BOTH:
- ℹ️  Skipping messages (for other workers' requests)
- 🎯 ACCEPTED boxes (for THIS worker's requests)

---

## 🎯 SUCCESS CRITERIA

You've successfully verified data parallelism when:

✅ **Console Verification**:
- Boxes appear in MULTIPLE terminal windows (not just one)
- Different worker node_ids in the boxes
- Boxes appear concurrently when sending multiple requests

✅ **API Verification**:
- `/api/chat/workers` shows `total_workers > 1`
- All workers show status "online"
- `active_requests` distributed across workers

✅ **Frontend Verification**:
- Browser console shows different `worker:` values for consecutive requests
- Throughput stats make sense (should see variation between workers)

✅ **Performance Verification**:
- 2 concurrent users → 2 workers active simultaneously
- Per-user latency unchanged (~1280ms/token)
- Aggregate throughput = N × single-node throughput

---

## 📝 TESTING CHECKLIST

Before claiming data parallelism works:

- [ ] Single-node baseline test passes
- [ ] 2-node cluster shows boxes in BOTH terminals
- [ ] `/api/chat/workers` returns correct count
- [ ] Browser console logs show different workers
- [ ] Concurrent requests go to different nodes
- [ ] Load balancing distributes requests evenly
- [ ] No "all requests to coordinator" issue
- [ ] Per-user latency unchanged
- [ ] Aggregate throughput = N × baseline

---

## 🚀 PRODUCTION VERIFICATION

For production deployment, verify:

1. **Multi-Node Cluster Active**:
   ```bash
   curl https://quillon.xyz/api/chat/workers | jq '.data.total_workers'
   # Should be > 1
   ```

2. **Metrics Show Distribution**:
   ```bash
   curl https://quillon.xyz/api/chat/metrics | jq '.data.distributed.average_nodes_per_request'
   # Should be > 1.0 (indicates multiple nodes participating)
   ```

3. **Console Logs Visible** (via systemctl status):
   ```bash
   sudo journalctl -u q-api-server -f | grep "DATA PARALLEL"
   # Should show boxes appearing on different nodes
   ```

---

## 💡 PRO TIPS

### **Tip 1: Use tmux/screen for Multiple Terminals**

```bash
# Create tmux session with 4 panes
tmux new-session -s ai-cluster \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; \
  split-window -v

# Now you can see all 4 node logs simultaneously!
```

### **Tip 2: Color-Code Your Nodes**

Add emoji to node IDs for easy visual identification:

```bash
--node-id "🔷coordinator"
--node-id "🟢worker1"
--node-id "🟡worker2"
--node-id "🔴worker3"
```

### **Tip 3: Watch Workers in Real-Time**

```bash
watch -n 1 'curl -s http://localhost:8001/api/chat/workers | jq ".data.workers[] | {node_id, active_requests}"'
```

### **Tip 4: Verify Load Balancing with Loop**

```bash
#!/bin/bash
# Send 20 requests and track which worker handled each
for i in {1..20}; do
  curl -X POST http://localhost:8001/api/chat/test$i/stream-distributed \
    -H "Content-Type: application/json" \
    -d "{\"content\":\"Request $i\"}" 2>&1 | \
    grep -oP 'worker_node":"\\K[^"]+' &
  sleep 0.5
done
wait | sort | uniq -c
# Should show even distribution:
#   10 coordinator
#   10 worker1
```

---

## 🎉 CONCLUSION

With these verification methods, you can **prove** that data parallelism is working:

1. **Visual Proof**: Colorful boxes in multiple terminals
2. **API Proof**: `/api/chat/workers` shows multiple nodes
3. **Frontend Proof**: Browser console shows different workers
4. **Performance Proof**: Aggregate throughput = N × baseline

**No more guessing - you can see exactly which worker processed each request!**

---

**Generated**: 2025-01-12
**Version**: v1.0
**Status**: ✅ **READY FOR VERIFICATION**
