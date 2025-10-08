# 🚀 Real 5-Node TPS Benchmark Test with Quantum Transport

## Test Objective
Measure real-world transactions per second (TPS) across 5 nodes with REAL quantum transport (Kyber1024 + Dilithium5).

## Test Configuration

### Network Topology
```
Node 1 (Bootstrap) ←→ Node 2
       ↓                 ↓
     Node 3 ←→ Node 4 ←→ Node 5
```

### Node Specifications
- **Quantum Transport**: Kyber1024 + Dilithium5 (Phase 1)
- **Consensus**: DAG-Knight BFT
- **Network**: libp2p with TCP transport
- **Storage**: RocksDB persistent storage
- **Ports**: 8081-8085 (API), 7001-7005 (P2P)

### Performance Targets
- **Baseline TPS**: 1,000 TPS (without quantum overhead)
- **Quantum TPS**: 800+ TPS (with quantum handshakes)
- **Latency**: <300ms end-to-end with quantum encryption
- **Finality**: <3 seconds

## Test Phases

### Phase 1: Network Initialization (30s)
- Start all 5 nodes
- Establish libp2p connections
- Bootstrap discovery complete
- All peers connected

### Phase 2: Quantum Handshake Establishment (60s)
- Submit initial transactions to trigger handshakes
- Kyber1024 key exchanges complete across all peers
- Dilithium5 signature verification
- All quantum channels established

### Phase 3: Warm-up Load (30s)
- 100 TPS sustained load
- Verify consensus processing
- Check all nodes synchronized

### Phase 4: TPS Ramp-Up (180s)
- Ramp from 100 → 500 → 1000 → 1500 TPS
- Measure throughput at each level
- Monitor quantum transport overhead
- Track memory and CPU usage

### Phase 5: Sustained Peak Load (300s)
- Maximum achievable TPS for 5 minutes
- Monitor for degradation
- Track consensus finality times
- Measure quantum handshake performance

### Phase 6: Cool-Down Analysis (60s)
- Reduce load gradually
- Verify all transactions finalized
- Check node synchronization
- Generate performance report

## Metrics to Capture

### Transaction Metrics
- Transactions submitted per second
- Transactions processed per second
- Transaction confirmation latency
- Transaction finality time
- Failed/dropped transactions

### Quantum Transport Metrics
- Quantum handshakes per second
- Average handshake time (target <50ms)
- Kyber1024 key exchange latency
- Dilithium5 signature verification time
- Quantum channel reuse rate

### Consensus Metrics
- Vertices created per second
- Consensus rounds per second
- Block finality time
- Fork rate (should be 0 with DAG-Knight)

### System Metrics
- CPU usage per node
- Memory usage per node
- Network bandwidth usage
- Disk I/O operations
- Database size growth

## Expected Results

### Without Quantum Transport (Baseline)
- TPS: 1,000-1,500
- Latency: 50-100ms
- Finality: 1-2 seconds

### With Quantum Transport (Phase 1)
- TPS: 800-1,200 (20% overhead acceptable)
- Latency: 100-300ms (quantum handshake overhead)
- Finality: 2-3 seconds
- First-message handshake: <50ms
- Subsequent messages: <10ms overhead

## Success Criteria

✅ All 5 nodes remain synchronized throughout test
✅ No consensus forks or disagreements
✅ Quantum handshakes complete in <50ms
✅ Sustained TPS ≥ 800 with quantum transport
✅ Transaction finality < 3 seconds
✅ Zero transaction loss
✅ All quantum channels established successfully
✅ No memory leaks or resource exhaustion

## Test Execution Commands

See `run_5_node_tps_benchmark.sh` for automated test execution.