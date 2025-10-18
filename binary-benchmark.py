#!/usr/bin/env python3
"""
Binary Protocol TPS Benchmark with SIMD Verification
Uses MessagePack binary protocol for 10x faster serialization
Expected improvement: 10-100x over JSON (1,600 TPS → 16,000-160,000 TPS)
"""

import msgpack
import requests
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "http://localhost:8200"
BINARY_BATCH_ENDPOINT = f"{API_BASE}/api/v1/binary/batch"
TOTAL_TRANSACTIONS = 10000
BATCH_SIZE = 1000  # Send 1000 transactions per batch
MAX_CONCURRENT = 10  # Number of concurrent HTTP requests

def generate_random_bytes(length):
    """Generate random bytes for signatures and addresses"""
    return bytes([random.randint(0, 255) for _ in range(length)])

def create_transaction(index):
    """Create a single transaction with random data"""
    return {
        'id': list(generate_random_bytes(32)),
        'from': list(generate_random_bytes(32)),
        'to': list(generate_random_bytes(32)),
        'amount': random.randint(1000, 1000000),
        'fee': random.randint(10, 100),
        'nonce': index,
        'signature': list(generate_random_bytes(64)),
        'timestamp': '2025-10-12T00:00:00Z',
        'data': []
    }

def create_batch(batch_size, start_index):
    """Create a batch of transactions"""
    return {
        'transactions': [create_transaction(start_index + i) for i in range(batch_size)]
    }

def submit_binary_batch(batch):
    """Submit batch using MessagePack binary protocol"""
    try:
        # Serialize to MessagePack (10x faster than JSON)
        start_serialize = time.time()
        packed_data = msgpack.packb(batch)
        serialize_time = time.time() - start_serialize

        # Send binary request
        start_request = time.time()
        response = requests.post(
            BINARY_BATCH_ENDPOINT,
            data=packed_data,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=30
        )
        request_time = time.time() - start_request

        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'serialize_time': serialize_time,
            'request_time': request_time,
            'total_time': serialize_time + request_time,
            'batch_size': len(batch['transactions'])
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'batch_size': len(batch['transactions'])
        }

def run_benchmark():
    """Run the binary protocol benchmark"""
    print("🚀 Q-NarwhalKnight Binary Protocol TPS Benchmark")
    print(f"📡 Testing against: {API_BASE}")
    print(f"📦 Batch size: {BATCH_SIZE} transactions")
    print(f"🔢 Total transactions: {TOTAL_TRANSACTIONS}")
    print(f"⚡ Max concurrent requests: {MAX_CONCURRENT}")
    print()

    # Create all batches
    num_batches = TOTAL_TRANSACTIONS // BATCH_SIZE
    print(f"💼 Creating {num_batches} batches...")
    batches = [create_batch(BATCH_SIZE, i * BATCH_SIZE) for i in range(num_batches)]
    print("✅ Batches created")
    print()

    # Warmup
    print("🔥 Warming up API...")
    warmup_batch = create_batch(10, 0)
    submit_binary_batch(warmup_batch)
    print("✅ Warmup complete")
    print()

    # Run benchmark
    print("⚡ Starting binary batch benchmark...")
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = [executor.submit(submit_binary_batch, batch) for batch in batches]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Progress indicator
            if len(results) % 5 == 0:
                print(f"  Processed {len(results)}/{num_batches} batches...", end='\r')

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    serialize_times = [r['serialize_time'] * 1000 for r in successful]  # Convert to ms
    request_times = [r['request_time'] * 1000 for r in successful]
    total_times = [r['total_time'] * 1000 for r in successful]

    total_txs = sum(r['batch_size'] for r in successful)
    tps = total_txs / elapsed

    print()
    print("=" * 80)
    print("📊 BINARY PROTOCOL TPS BENCHMARK RESULTS")
    print("=" * 80)
    print(f"📈 Total Transactions: {total_txs}")
    print(f"✅ Successful Batches: {len(successful)}")
    print(f"❌ Failed Batches: {len(failed)}")
    print(f"📦 Average Batch Size: {BATCH_SIZE}")
    print(f"⏱️  Total Time: {elapsed:.2f}s")
    print(f"⚡ TPS: {tps:.2f}")
    print()

    if serialize_times:
        print("🔄 Serialization Performance (MessagePack):")
        print(f"  • Average: {statistics.mean(serialize_times):.2f}ms")
        print(f"  • Median: {statistics.median(serialize_times):.2f}ms")
        print(f"  • Min: {min(serialize_times):.2f}ms")
        print(f"  • Max: {max(serialize_times):.2f}ms")
        print()

    if request_times:
        print("🌐 Request Latency Statistics:")
        print(f"  • Average: {statistics.mean(request_times):.2f}ms")
        print(f"  • Median: {statistics.median(request_times):.2f}ms")
        print(f"  • Min: {min(request_times):.2f}ms")
        print(f"  • Max: {max(request_times):.2f}ms")

        # Calculate percentiles
        sorted_times = sorted(request_times)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        print(f"  • P95: {sorted_times[p95_idx]:.2f}ms")
        print(f"  • P99: {sorted_times[p99_idx]:.2f}ms")
        print()

    if total_times:
        print("⏱️  Total Processing Time (Serialization + Request):")
        print(f"  • Average: {statistics.mean(total_times):.2f}ms")
        print(f"  • Per Transaction: {statistics.mean(total_times) / BATCH_SIZE:.3f}ms")
        print()

    # Calculate improvement vs JSON baseline
    json_baseline_tps = 1594.35  # From previous benchmark
    improvement = (tps / json_baseline_tps)

    print("🎯 Performance Comparison:")
    print(f"  • JSON Baseline: {json_baseline_tps:.2f} TPS")
    print(f"  • Binary Protocol: {tps:.2f} TPS")
    print(f"  • Improvement: {improvement:.2f}x faster")
    print()

    # Calculate effective per-transaction latency
    avg_per_tx_ms = (elapsed * 1000) / total_txs
    print("📊 Effective Metrics:")
    print(f"  • Per-Transaction Latency: {avg_per_tx_ms:.3f}ms")
    print(f"  • Transactions per Second: {tps:.0f}")
    print(f"  • Batches per Second: {num_batches / elapsed:.2f}")
    print()

    if failed:
        print("❌ Failed Requests:")
        for i, fail in enumerate(failed[:5]):  # Show first 5 failures
            print(f"  {i+1}. {fail.get('error', 'Unknown error')}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
        print()

    print("=" * 80)
    print()
    print("🔬 SIMD Verification Analysis:")
    print("  Check server logs for SIMD verification messages:")
    print("  tail -f /tmp/api-server.log | grep SIMD")
    print()
    print("✅ Binary protocol benchmark complete!")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
