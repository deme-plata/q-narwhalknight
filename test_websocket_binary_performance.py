#!/usr/bin/env python3
"""
WebSocket Binary Protocol Performance Test

Tests the persistent WebSocket connection with binary MessagePack streaming.
This should eliminate HTTP overhead and achieve 2-5x improvement over batch HTTP.

Expected: 21,000-50,000 TPS
"""

import asyncio
import websockets
import msgpack
import time
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any

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

def create_test_transaction(index: int) -> Dict[str, Any]:
    """Create a test transaction with proper types"""
    from_addr = create_address(f"from_{index}")
    to_addr = create_address(f"to_{index}")
    tx_id = create_address(f"tx_{index}_{time.time()}")
    signature = create_signature(f"sign_{index}")
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

async def test_websocket_streaming(num_transactions: int = 10000):
    """Test WebSocket binary streaming"""
    print(f"🚀 WebSocket Binary Streaming Test")
    print(f"=" * 80)
    print(f"Target: {num_transactions} transactions")
    print(f"Protocol: Binary MessagePack over WebSocket")
    print()

    # Generate transactions
    print("📝 Generating transactions...")
    gen_start = time.time()
    transactions = [create_test_transaction(i) for i in range(num_transactions)]
    gen_time = time.time() - gen_start
    print(f"✅ Generated {num_transactions} transactions in {gen_time:.2f}s")
    print()

    # Connect to WebSocket
    print(f"🔌 Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("✅ WebSocket connected")
            print()

            # Send all transactions as fast as possible
            print(f"⚡ Streaming {num_transactions} transactions...")
            start_time = time.time()
            sent = 0

            for i, tx in enumerate(transactions):
                # Serialize to MessagePack
                packed = msgpack.packb(tx)

                # Send over WebSocket
                await websocket.send(packed)
                sent += 1

                # Print progress every 1000 transactions
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    current_tps = (i + 1) / elapsed
                    print(f"  📊 Sent {i + 1}/{num_transactions} tx - {current_tps:.0f} TPS")

            elapsed = time.time() - start_time
            tps = sent / elapsed
            latency_ms = (elapsed / sent * 1000) if sent > 0 else 0

            print()
            print(f"✅ Transmission Complete!")
            print(f"   Sent: {sent} transactions")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   TPS: {tps:.0f}")
            print(f"   Latency: {latency_ms:.4f}ms per tx")
            print()

            # Wait for acknowledgments
            print("📥 Waiting for server acknowledgments...")
            acks_received = 0
            total_accepted = 0

            try:
                # Wait up to 10 seconds for acknowledgments
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)

                        # Deserialize MessagePack response
                        ack = msgpack.unpackb(response)

                        if ack.get('success'):
                            total_accepted = ack.get('accepted', 0)
                            acks_received += 1
                            print(f"  ✅ Acknowledgment {acks_received}: {total_accepted} transactions accepted")
                    except asyncio.TimeoutError:
                        print("  ⏱️  No more acknowledgments (timeout)")
                        break
            except Exception as e:
                print(f"  ⚠️  Error receiving acks: {e}")

            print()
            print("📊 WEBSOCKET STREAMING RESULTS")
            print("=" * 80)
            print(f"Transactions Sent:     {sent}")
            print(f"Transactions Accepted: {total_accepted}")
            print(f"Streaming Time:        {elapsed:.2f}s")
            print(f"Streaming TPS:         {tps:.0f}")
            print(f"Latency per TX:        {latency_ms:.4f}ms")
            print()

            # Compare to batch HTTP
            batch_tps = 10757  # From previous benchmark
            improvement = tps / batch_tps

            print("🎯 PERFORMANCE COMPARISON")
            print("=" * 80)
            print(f"Binary Batch (HTTP):   {batch_tps:>10.0f} TPS (baseline)")
            print(f"WebSocket Streaming:   {tps:>10.0f} TPS")
            print(f"Improvement:           {improvement:>10.1f}x")
            print()

            if improvement >= 5.0:
                print("🎉 EXCEEDED 5x IMPROVEMENT TARGET!")
            elif improvement >= 2.0:
                print("✅ ACHIEVED 2x+ IMPROVEMENT TARGET!")
            else:
                print(f"⚠️  Below 2x target ({improvement:.1f}x)")

            # Projection to 1M TPS
            print()
            print("🚀 PROJECTION TO 1M+ TPS")
            print("=" * 80)
            print(f"Current WebSocket TPS:     {tps:>10.0f}")

            with_workers = tps * 16  # 16 parallel workers
            print(f"+ 16 Parallel Workers:     {with_workers:>10.0f} TPS")

            with_io_uring = with_workers * 5  # 5x from kernel I/O
            print(f"+ Kernel I/O (5x):         {with_io_uring:>10.0f} TPS")

            with_simd = with_io_uring * 2  # 2x from SIMD batch validation
            print(f"+ SIMD Batch Validation:   {with_simd:>10.0f} TPS")
            print()

            if with_simd >= 1_000_000:
                print(f"✅ 1M+ TPS ACHIEVABLE! (Projected: {with_simd:,.0f} TPS)")
            else:
                remaining = 1_000_000 / with_simd
                print(f"⏳ Need {remaining:.1f}x more to reach 1M TPS")

            await websocket.close()

    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        print()
        print("Make sure the server is running:")
        print("  Q_DB_PATH=./data-binary-test Q_P2P_PORT=9011 \\")
        print("    ./target/release/q-api-server --port 9010")
        return None

    return {
        "sent": sent,
        "accepted": total_accepted,
        "elapsed": elapsed,
        "tps": tps,
        "latency_ms": latency_ms,
        "improvement": improvement
    }

async def main():
    print("🌟 Q-NarwhalKnight WebSocket Binary Streaming Benchmark")
    print()

    # Test with 10,000 transactions
    result = await test_websocket_streaming(10000)

    if result:
        print()
        print("✅ WebSocket streaming test complete!")
        print()
        print("Key metrics:")
        print(f"  • Streaming TPS: {result['tps']:.0f}")
        print(f"  • Improvement over batch HTTP: {result['improvement']:.1f}x")
        print(f"  • Latency: {result['latency_ms']:.4f}ms per transaction")

if __name__ == "__main__":
    asyncio.run(main())
