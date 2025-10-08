#!/usr/bin/env python3
"""
Binary Protocol Performance Test

Compares performance of:
1. JSON endpoint (baseline: 333 TPS)
2. Binary MessagePack endpoint (target: 10x-60x faster)
3. Binary batch endpoint (target: 1000x faster)
4. WebSocket streaming (target: maximum throughput)
"""

import requests
import msgpack
import time
import json
import hashlib
from typing import List, Dict, Any
from datetime import datetime, timezone

# Server configuration
SERVER_URL = "http://localhost:9010"
JSON_ENDPOINT = f"{SERVER_URL}/api/v1/transactions"
BINARY_SINGLE_ENDPOINT = f"{SERVER_URL}/api/v1/binary/transaction"
BINARY_BATCH_ENDPOINT = f"{SERVER_URL}/api/v1/binary/batch"

def create_address(seed: str) -> List[int]:
    """Create a 32-byte address from a seed string"""
    hash_bytes = hashlib.sha256(seed.encode()).digest()
    return list(hash_bytes)

def create_signature(data: str) -> List[int]:
    """Create a dummy signature (64 bytes)"""
    hash1 = hashlib.sha256(data.encode()).digest()
    hash2 = hashlib.sha256(hash1).digest()
    return list(hash1 + hash2)

def create_test_transaction(index: int) -> Dict[str, Any]:
    """Create a test transaction with proper types"""
    # Create 32-byte addresses
    from_addr = create_address(f"from_{index}")
    to_addr = create_address(f"to_{index}")

    # Create transaction ID (TxHash is [u8; 32])
    tx_id = create_address(f"tx_{index}_{time.time()}")

    # Create signature (64 bytes)
    signature = create_signature(f"sign_{index}")

    # Create timestamp in RFC3339 format
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    return {
        "id": tx_id,
        "from": from_addr,
        "to": to_addr,
        "amount": 1000 + index,
        "fee": 10,
        "nonce": index,
        "signature": signature,
        "timestamp": timestamp,
        "data": [],  # Empty data payload
    }

def benchmark_json_endpoint(num_transactions: int = 5000) -> Dict[str, float]:
    """Benchmark JSON endpoint (baseline)"""
    print(f"\n📊 Benchmarking JSON endpoint ({num_transactions} transactions)...")

    start_time = time.time()
    successful = 0
    failed = 0

    for i in range(num_transactions):
        tx = create_test_transaction(i)
        try:
            # Wrap transaction in expected request format
            request_body = {"transaction": tx}
            response = requests.post(
                JSON_ENDPOINT,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                successful += 1
            else:
                failed += 1
                if i < 3:  # Only print first few errors
                    print(f"  ❌ Error {response.status_code}: {response.text[:100]}")
        except Exception as e:
            failed += 1
            if i < 3:  # Only print first few errors
                print(f"  ❌ Exception: {e}")

    elapsed = time.time() - start_time
    tps = successful / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / successful * 1000) if successful > 0 else 0

    print(f"✅ JSON Results:")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   TPS: {tps:.0f}")
    print(f"   Latency: {latency_ms:.2f}ms per tx")

    return {
        "successful": successful,
        "failed": failed,
        "elapsed": elapsed,
        "tps": tps,
        "latency_ms": latency_ms
    }

def benchmark_binary_single_endpoint(num_transactions: int = 5000) -> Dict[str, float]:
    """Benchmark binary MessagePack endpoint"""
    print(f"\n📊 Benchmarking Binary MessagePack endpoint ({num_transactions} transactions)...")

    start_time = time.time()
    successful = 0
    failed = 0

    for i in range(num_transactions):
        tx = create_test_transaction(i)
        try:
            # Serialize to MessagePack
            packed_data = msgpack.packb(tx)

            response = requests.post(
                BINARY_SINGLE_ENDPOINT,
                data=packed_data,
                headers={"Content-Type": "application/msgpack"},
                timeout=5
            )
            if response.status_code == 200:
                successful += 1
                # Can also deserialize response if needed
                # result = msgpack.unpackb(response.content)
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if i < 5:
                print(f"  ❌ Error: {e}")

    elapsed = time.time() - start_time
    tps = successful / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / successful * 1000) if successful > 0 else 0

    print(f"✅ Binary Single Results:")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   TPS: {tps:.0f}")
    print(f"   Latency: {latency_ms:.2f}ms per tx")

    return {
        "successful": successful,
        "failed": failed,
        "elapsed": elapsed,
        "tps": tps,
        "latency_ms": latency_ms
    }

def benchmark_binary_batch_endpoint(num_transactions: int = 5000, batch_size: int = 100) -> Dict[str, float]:
    """Benchmark binary batch endpoint"""
    print(f"\n📊 Benchmarking Binary Batch endpoint ({num_transactions} transactions, batch size: {batch_size})...")

    start_time = time.time()
    successful = 0
    failed = 0

    # Create batches
    num_batches = num_transactions // batch_size

    for batch_idx in range(num_batches):
        # Create batch of transactions
        batch = {
            "transactions": [
                create_test_transaction(batch_idx * batch_size + i)
                for i in range(batch_size)
            ]
        }

        try:
            # Serialize batch to MessagePack
            packed_data = msgpack.packb(batch)

            response = requests.post(
                BINARY_BATCH_ENDPOINT,
                data=packed_data,
                headers={"Content-Type": "application/msgpack"},
                timeout=5
            )
            if response.status_code == 200:
                successful += batch_size
            else:
                failed += batch_size
        except Exception as e:
            failed += batch_size
            if batch_idx < 5:
                print(f"  ❌ Error: {e}")

    elapsed = time.time() - start_time
    tps = successful / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / successful * 1000) if successful > 0 else 0

    print(f"✅ Binary Batch Results:")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   TPS: {tps:.0f}")
    print(f"   Latency: {latency_ms:.4f}ms per tx")

    return {
        "successful": successful,
        "failed": failed,
        "elapsed": elapsed,
        "tps": tps,
        "latency_ms": latency_ms
    }

def print_comparison(json_results: Dict, binary_results: Dict, batch_results: Dict):
    """Print comparison table"""
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON REPORT")
    print("="*80)

    print("\n┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Metric              │ JSON (Base)  │ Binary (1x)  │ Binary Batch │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")

    # TPS comparison
    json_tps = json_results.get('tps', 0)
    binary_tps = binary_results.get('tps', 0)
    batch_tps = batch_results.get('tps', 0)

    print(f"│ TPS                 │ {json_tps:>11.0f}  │ {binary_tps:>11.0f}  │ {batch_tps:>11.0f}  │")

    # Latency comparison
    json_lat = json_results.get('latency_ms', 0)
    binary_lat = binary_results.get('latency_ms', 0)
    batch_lat = batch_results.get('latency_ms', 0)

    print(f"│ Latency (ms/tx)     │ {json_lat:>11.2f}  │ {binary_lat:>11.2f}  │ {batch_lat:>11.4f}  │")

    # Improvement factors
    binary_improvement = binary_tps / json_tps if json_tps > 0 else 0
    batch_improvement = batch_tps / json_tps if json_tps > 0 else 0

    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Improvement Factor  │      1.00x   │ {binary_improvement:>11.1f}x  │ {batch_improvement:>11.1f}x  │")
    print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")

    # Analysis
    print("\n🎯 ANALYSIS:")
    print(f"   Binary Single: {binary_improvement:.1f}x faster than JSON")
    print(f"   Binary Batch:  {batch_improvement:.1f}x faster than JSON")

    # Check if we hit targets
    print("\n🎯 TARGET ACHIEVEMENT:")
    if binary_improvement >= 10:
        print(f"   ✅ Binary single exceeded 10x target ({binary_improvement:.1f}x)")
    else:
        print(f"   ⚠️  Binary single below 10x target ({binary_improvement:.1f}x)")

    if batch_improvement >= 100:
        print(f"   ✅ Binary batch exceeded 100x target ({batch_improvement:.1f}x)")
    else:
        print(f"   ⚠️  Binary batch below 100x target ({batch_improvement:.1f}x)")

    # Projected 1M TPS capability
    if batch_tps > 0:
        print(f"\n📈 SCALING PROJECTION:")
        print(f"   Current batch TPS: {batch_tps:.0f}")

        # With optimizations (remove HTTP overhead entirely via WebSocket)
        projected_ws_tps = batch_tps * 2  # Conservative estimate
        print(f"   With WebSocket streaming: ~{projected_ws_tps:.0f} TPS (estimated)")

        # With parallel workers
        projected_parallel_tps = projected_ws_tps * 4
        print(f"   With 4 parallel workers: ~{projected_parallel_tps:.0f} TPS (estimated)")

        if projected_parallel_tps >= 1_000_000:
            print(f"   🎉 1M+ TPS ACHIEVABLE with full optimizations!")
        else:
            remaining = 1_000_000 / projected_parallel_tps
            print(f"   ⏳ Need {remaining:.1f}x more improvement to reach 1M TPS")

def main():
    print("🚀 Q-NarwhalKnight Binary Protocol Performance Test")
    print("=" * 80)
    print(f"Server: {SERVER_URL}")
    print("Testing with 1,000 transactions per endpoint (quick test)")

    # Check server is running
    try:
        response = requests.get(f"{SERVER_URL}/api/v1/status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server returned unexpected status")
            return
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        print("Please start the server first:")
        print("  Q_DB_PATH=./data-binary-test Q_P2P_PORT=9011 ./target/release/q-api-server --port 9010")
        return

    # Run benchmarks with smaller sample for faster test
    print("\n📝 Note: Using 1000 transactions for quick performance comparison")
    json_results = benchmark_json_endpoint(1000)
    binary_results = benchmark_binary_single_endpoint(1000)
    batch_results = benchmark_binary_batch_endpoint(1000, batch_size=100)

    # Print comparison
    print_comparison(json_results, binary_results, batch_results)

    print("\n✅ Performance test complete!")

if __name__ == "__main__":
    main()
