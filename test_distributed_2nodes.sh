#!/bin/bash
# Quick 2-Node Distributed TPS Test

echo "================================================================================"
echo "🌟 DISTRIBUTED 2-NODE TPS BENCHMARK"
echo "================================================================================"
echo "Testing nodes: 9100, 9102"
echo ""

# Test concurrent load to both nodes
echo "⚡ Submitting 5 batches of 10K transactions to each node..."
echo ""

# Create simple Python test client
cat > /tmp/distributed_2node_test.py <<'PYTHON'
import requests
import time
import hashlib
import struct
from datetime import datetime
import concurrent.futures
import msgpack

def create_address(seed):
    return hashlib.sha256(seed.encode()).digest()

def create_signature(data):
    hash1 = hashlib.sha256(data.encode()).digest()
    hash2 = hashlib.sha256(hash1).digest()
    return hash1 + hash2

def create_transaction(index):
    from_addr = create_address(f"from_{index}")
    to_addr = create_address(f"to_{index}")
    tx_id = create_address(f"tx_{index}_{index}")
    signature = create_signature(f"sign_{index}")

    return {
        'id': tx_id,
        'from': from_addr,
        'to': to_addr,
        'amount': 1000 + index,
        'fee': 10,
        'nonce': index,
        'signature': signature,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'data': b''
    }

def submit_to_node(node_id, port, num_batches, batch_size):
    url = f"http://localhost:{port}/api/v1/binary/batch"
    total_tx = 0
    start = time.time()

    for batch_num in range(num_batches):
        base_idx = node_id * 1_000_000 + batch_num * batch_size
        transactions = [create_transaction(base_idx + i) for i in range(batch_size)]

        batch = {'transactions': transactions}
        packed = msgpack.packb(batch)

        try:
            resp = requests.post(url, data=packed, headers={'Content-Type': 'application/msgpack'}, timeout=120)
            if resp.status_code == 200:
                total_tx += batch_size
            else:
                print(f"  ⚠️  Node {node_id} batch {batch_num} failed: {resp.status_code}")
        except Exception as e:
            print(f"  ❌ Node {node_id} batch {batch_num} error: {e}")

    elapsed = time.time() - start
    tps = total_tx / elapsed if elapsed > 0 else 0

    return total_tx, tps

def main():
    nodes = [(0, 9100), (2, 9102)]
    batches_per_node = 5
    batch_size = 10_000

    print(f"📋 Configuration:")
    print(f"  Nodes:              {len(nodes)}")
    print(f"  Batches/Node:       {batches_per_node}")
    print(f"  Batch Size:         {batch_size} transactions")
    print(f"  Total Transactions: {len(nodes) * batches_per_node * batch_size}")
    print()

    global_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {}
        for node_id, port in nodes:
            future = executor.submit(submit_to_node, node_id, port, batches_per_node, batch_size)
            futures[future] = (node_id, port)
            print(f"  🔹 Submitting to Node {node_id} (port {port})...")

        results = []
        for future in concurrent.futures.as_completed(futures):
            node_id, port = futures[future]
            try:
                sent, tps = future.result()
                print(f"  ✅ Node {node_id} completed: {sent} tx at {tps:.0f} TPS")
                results.append((sent, tps))
            except Exception as e:
                print(f"  ❌ Node {node_id} failed: {e}")

    total_elapsed = time.time() - global_start
    total_tx = sum(r[0] for r in results)
    aggregate_tps = total_tx / total_elapsed if total_elapsed > 0 else 0

    print()
    print("="*80)
    print("📊 DISTRIBUTED BENCHMARK RESULTS")
    print("="*80)
    print(f"Total Transactions:    {total_tx}")
    print(f"Total Time:            {total_elapsed:.2f}s")
    print(f"Aggregate TPS:         {aggregate_tps:.0f}")
    print()

    if results:
        avg_tps = sum(r[1] for r in results) / len(results)
        min_tps = min(r[1] for r in results)
        max_tps = max(r[1] for r in results)

        print("Node Performance:")
        print(f"  Average TPS/Node:    {avg_tps:.0f}")
        print(f"  Min TPS/Node:        {min_tps:.0f}")
        print(f"  Max TPS/Node:        {max_tps:.0f}")
        print()

    target_1m = 1_000_000.0
    percent_target = (aggregate_tps / target_1m) * 100.0

    print("🎯 PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Target (1M TPS):       {target_1m:>10.0f} TPS")
    print(f"Actual (Measured):     {aggregate_tps:>10.0f} TPS")
    print(f"Percent of Target:     {percent_target:>10.1f}%")
    print()

    if aggregate_tps >= target_1m:
        print("🎉🎉🎉 ACHIEVED 1M+ TPS WITH DISTRIBUTED NODES! 🎉🎉🎉")
    elif aggregate_tps >= target_1m * 0.5:
        print(f"📈 ACHIEVED {percent_target:.0f}% OF 1M TPS TARGET!")
    else:
        print(f"📊 Distributed performance: {percent_target:.1f}% of 1M TPS target")

    print()
    print("="*80)
    print("✅ DISTRIBUTED BENCHMARK COMPLETE!")
    print("🚀 Q-NarwhalKnight Quantum-Enhanced DAG-BFT Consensus")
    print("="*80)

if __name__ == "__main__":
    main()
PYTHON

python3 /tmp/distributed_2node_test.py
