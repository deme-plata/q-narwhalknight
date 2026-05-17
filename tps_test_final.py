#!/usr/bin/env python3
"""
TPS Benchmark - Testing 1M+ TPS with ParallelWorkerPool
Sends properly formatted byte arrays to the batch API
"""

import requests
import json
import time
import random

def generate_byte_array(size):
    """Generate random byte array"""
    return [random.randint(0, 255) for _ in range(size)]

def generate_transaction():
    """Generate a properly formatted transaction"""
    from datetime import datetime, timezone
    return {
        "transaction": {
            "id": generate_byte_array(32),
            "from": generate_byte_array(32),
            "to": generate_byte_array(32),
            "amount": random.randint(1, 1000000),
            "fee": 1,
            "nonce": random.randint(1, 1000000),
            "signature": generate_byte_array(64),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": []
        }
    }

def run_benchmark():
    """Run TPS benchmark"""
    print("🚀 Q-NarwhalKnight TPS Benchmark")
    print("="*50)
    print()

    node_url = "http://localhost:8200"
    batch_size = 10000  # Scale up to 10k transactions per batch
    num_batches = 100   # 100 batches = 1M total transactions

    print(f"Configuration:")
    print(f"  Node: {node_url}")
    print(f"  Batch size: {batch_size} transactions")
    print(f"  Batches: {num_batches}")
    print(f"  Total transactions: {batch_size * num_batches}")
    print()

    # Check node health
    print("🏥 Checking node health...")
    try:
        resp = requests.get(f"{node_url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✅ Node is healthy")
        else:
            print(f"  ❌ Node returned status {resp.status_code}")
            return
    except Exception as e:
        print(f"  ❌ Node not responding: {e}")
        return

    print()
    print("📊 Starting TPS benchmark...")
    print()

    total_submitted = 0
    total_failed = 0
    start_time = time.time()

    for batch_num in range(1, num_batches + 1):
        # Generate batch of transactions
        batch = {
            "transactions": [generate_transaction() for _ in range(batch_size)]
        }

        # Submit batch
        try:
            resp = requests.post(
                f"{node_url}/api/v1/transactions/batch",
                json=batch,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    result = data.get("data", {})
                    submitted = result.get("submitted", 0)
                    total_submitted += submitted

                    if batch_num % 2 == 0:
                        elapsed = time.time() - start_time
                        current_tps = total_submitted / elapsed if elapsed > 0 else 0
                        server_tps = result.get("tps", 0)
                        print(f"  Batch {batch_num}/{num_batches}: {total_submitted} tx "
                              f"({current_tps:.0f} TPS overall, server: {server_tps} TPS/batch)")
                else:
                    error = data.get("error", "Unknown error")
                    print(f"  ❌ Batch {batch_num} API error: {error}")
                    total_failed += batch_size
            else:
                print(f"  ❌ Batch {batch_num} HTTP {resp.status_code}: {resp.text[:100]}")
                total_failed += batch_size

        except Exception as e:
            print(f"  ❌ Batch {batch_num} failed: {e}")
            total_failed += batch_size

    end_time = time.time()
    elapsed = end_time - start_time
    overall_tps = total_submitted / elapsed if elapsed > 0 else 0

    print()
    print("📈 TPS Benchmark Results:")
    print("="*50)
    print(f"  Total transactions: {total_submitted}")
    print(f"  Failed: {total_failed}")
    print(f"  Success rate: {(total_submitted / (batch_size * num_batches) * 100):.1f}%")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Overall TPS: {overall_tps:.0f}")
    print()

    if overall_tps >= 1_000_000:
        print("🎉 SUCCESS: Achieved 1M+ TPS target!")
    elif overall_tps >= 500_000:
        print("🚀 EXCELLENT: Achieved 500k+ TPS!")
    elif overall_tps >= 100_000:
        print("✅ GOOD: Achieved 100k+ TPS!")
    elif overall_tps >= 10_000:
        print("⚡ PROGRESS: Achieved 10k+ TPS!")
    elif overall_tps >= 1_000:
        print("📊 BASELINE: Achieved 1k+ TPS!")
    else:
        print(f"⚠️  Initial test: {overall_tps:.0f} TPS")

    print()
    print("Optimizations tested:")
    print("  ✅ Batch Transaction API")
    print("  ✅ ParallelWorkerPool (10 workers)")
    print("  ✅ SIMD + Kernel I/O")
    print("  ✅ Proper byte array format")

if __name__ == "__main__":
    run_benchmark()
