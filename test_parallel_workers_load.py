#!/usr/bin/env python3
"""
Parallel Workers Load Test - Multi-threaded WebSocket Streaming

This test spawns multiple concurrent WebSocket clients to measure
the actual throughput improvement from 16 parallel workers.

Expected: 16x improvement over single worker (21,817 → 349,072 TPS)
"""

import asyncio
import websockets
import msgpack
import time
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

SERVER_URL = "ws://localhost:9050/api/v1/binary/stream"

def create_address(seed: str) -> List[int]:
    """Create a 32-byte address from a seed string"""
    hash_bytes = hashlib.sha256(seed.encode()).digest()
    return list(hash_bytes)

def create_signature(data: str) -> List[int]:
    """Create a dummy signature (64 bytes)"""
    hash1 = hashlib.sha256(data.encode()).digest()
    hash2 = hashlib.sha256(hash1).digest()
    return list(hash1 + hash2)

def create_test_transaction(index: int, client_id: int) -> Dict[str, Any]:
    """Create a test transaction with proper types"""
    from_addr = create_address(f"from_{client_id}_{index}")
    to_addr = create_address(f"to_{client_id}_{index}")
    tx_id = create_address(f"tx_{client_id}_{index}_{time.time()}")
    signature = create_signature(f"sign_{client_id}_{index}")
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
        "data": [],
    }

async def worker_client(client_id: int, num_transactions: int):
    """Single WebSocket client worker"""
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            start_time = time.time()
            sent = 0

            for i in range(num_transactions):
                tx = create_test_transaction(i, client_id)
                packed = msgpack.packb(tx)
                await websocket.send(packed)
                sent += 1

            elapsed = time.time() - start_time
            tps = sent / elapsed if elapsed > 0 else 0

            return {
                'client_id': client_id,
                'sent': sent,
                'elapsed': elapsed,
                'tps': tps
            }
    except Exception as e:
        print(f"  ❌ Client {client_id} error: {e}")
        return None

async def run_load_test(num_clients: int, transactions_per_client: int):
    """Run parallel load test with multiple clients"""
    print(f"🚀 Parallel Workers Load Test")
    print(f"=" * 80)
    print(f"Concurrent Clients: {num_clients}")
    print(f"Transactions/Client: {transactions_per_client}")
    print(f"Total Transactions: {num_clients * transactions_per_client:,}")
    print(f"Target Server: {SERVER_URL}")
    print()

    print(f"⚡ Starting {num_clients} concurrent WebSocket clients...")
    start_time = time.time()

    # Run all clients concurrently
    tasks = [worker_client(i, transactions_per_client) for i in range(num_clients)]
    results = await asyncio.gather(*tasks)

    total_elapsed = time.time() - start_time

    # Filter successful results
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)

    if not successful:
        print("❌ All clients failed!")
        return

    print()
    print(f"✅ Load Test Complete!")
    print()

    # Aggregate statistics
    total_sent = sum(r['sent'] for r in successful)
    total_time = total_elapsed
    aggregate_tps = total_sent / total_time if total_time > 0 else 0
    avg_client_tps = sum(r['tps'] for r in successful) / len(successful)
    min_client_tps = min(r['tps'] for r in successful)
    max_client_tps = max(r['tps'] for r in successful)

    print("📊 PARALLEL WORKERS LOAD TEST RESULTS")
    print("=" * 80)
    print(f"Successful Clients:    {len(successful)}/{num_clients}")
    print(f"Failed Clients:        {failed}")
    print(f"Total Transactions:    {total_sent:,}")
    print(f"Total Time:            {total_time:.2f}s")
    print(f"Aggregate TPS:         {aggregate_tps:,.0f}")
    print()
    print("Client Performance:")
    print(f"  Average TPS/Client:  {avg_client_tps:,.0f}")
    print(f"  Min TPS/Client:      {min_client_tps:,.0f}")
    print(f"  Max TPS/Client:      {max_client_tps:,.0f}")
    print()

    # Compare to baseline
    baseline_single_worker = 21817  # WebSocket streaming baseline
    baseline_16_workers_expected = 349072  # 16x projected

    improvement_over_baseline = aggregate_tps / baseline_single_worker
    percent_of_target = (aggregate_tps / baseline_16_workers_expected) * 100

    print("🎯 PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Baseline (Single Worker):      {baseline_single_worker:>10,} TPS")
    print(f"Projected (16 Workers):        {baseline_16_workers_expected:>10,} TPS")
    print(f"Actual (Measured):             {aggregate_tps:>10,.0f} TPS")
    print()
    print(f"Improvement over baseline:     {improvement_over_baseline:>10.1f}x")
    print(f"Percent of 16x target:         {percent_of_target:>10.1f}%")
    print()

    if aggregate_tps >= baseline_16_workers_expected:
        print("🎉 EXCEEDED 16x TARGET! Parallel workers delivering full performance!")
    elif aggregate_tps >= baseline_16_workers_expected * 0.8:
        print("✅ ACHIEVED 80%+ OF TARGET! Parallel workers working well!")
    elif aggregate_tps >= baseline_16_workers_expected * 0.5:
        print("⚡ ACHIEVED 50%+ OF TARGET! Parallel workers active, room for optimization!")
    elif aggregate_tps >= baseline_single_worker * 2:
        print("📈 ACHIEVED 2x+ IMPROVEMENT! Parallel workers providing speedup!")
    else:
        print(f"⚠️  Below 2x improvement - May be client-limited or server bottleneck")

    print()
    print("💡 ANALYSIS:")

    # Check if client-limited
    if avg_client_tps < 10000:
        print("  • Client throughput is low (<10K TPS/client)")
        print("  • This may be Python asyncio limitation, not server limit")
        print("  • Try more concurrent clients to saturate server")

    # Check if workers are being utilized
    if aggregate_tps > baseline_single_worker * 4:
        print(f"  • Aggregate TPS ({aggregate_tps:,.0f}) >> baseline ({baseline_single_worker:,})")
        print("  • Multiple workers are being utilized!")

    # Check consistency
    variance = (max_client_tps - min_client_tps) / avg_client_tps if avg_client_tps > 0 else 0
    if variance < 0.2:
        print(f"  • Low variance ({variance:.1%}) - consistent performance across clients")
    else:
        print(f"  • High variance ({variance:.1%}) - some clients slower than others")

    print()
    print("📈 NEXT STEPS:")
    if aggregate_tps < baseline_16_workers_expected * 0.8:
        print("  1. Increase concurrent clients to fully saturate 16 workers")
        print("  2. Check server CPU utilization (should be high across multiple cores)")
        print("  3. Monitor worker statistics in server logs")
        print(f"  4. Try with {num_clients * 2} clients for higher load")

    return {
        'clients': num_clients,
        'total_sent': total_sent,
        'total_time': total_time,
        'aggregate_tps': aggregate_tps,
        'improvement': improvement_over_baseline,
        'percent_of_target': percent_of_target
    }

async def main():
    print("🌟 Q-NarwhalKnight Parallel Workers Load Test")
    print()

    # Test configurations
    tests = [
        (4, 5000),   # 4 clients, 5K tx each = 20K total
        (8, 5000),   # 8 clients, 5K tx each = 40K total
        (16, 5000),  # 16 clients, 5K tx each = 80K total
    ]

    all_results = []

    for num_clients, txs_per_client in tests:
        result = await run_load_test(num_clients, txs_per_client)
        if result:
            all_results.append(result)
        print()
        print("─" * 80)
        print()
        await asyncio.sleep(2)  # Brief pause between tests

    # Final summary
    if all_results:
        print("📊 FINAL SUMMARY - PARALLEL WORKERS PERFORMANCE")
        print("=" * 80)
        print(f"{'Clients':<10} {'Total TPS':<15} {'vs Baseline':<15} {'% of 16x Target':<20}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['clients']:<10} {r['aggregate_tps']:>10,.0f}     "
                  f"{r['improvement']:>6.1f}x         {r['percent_of_target']:>6.1f}%")
        print()

        best = max(all_results, key=lambda x: x['aggregate_tps'])
        print(f"🏆 Best Performance: {best['aggregate_tps']:,.0f} TPS with {best['clients']} concurrent clients")
        print(f"   Improvement: {best['improvement']:.1f}x over single worker baseline")
        print(f"   Achievement: {best['percent_of_target']:.1f}% of 16x target (349,072 TPS)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
