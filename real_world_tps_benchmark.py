#!/usr/bin/env python3
"""
Real-World TPS Benchmark for Q-NarwhalKnight
Tests actual production system at quillon.xyz with real transactions
"""

import asyncio
import aiohttp
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple
import statistics

# Production API endpoint
API_BASE = "https://quillon.xyz"

class Wallet:
    """Simple wallet with address generation"""
    def __init__(self, seed: str):
        self.seed = seed
        # Generate deterministic address from seed
        self.address = hashlib.sha256(seed.encode()).hexdigest()[:40]
        self.balance = 0

    def __repr__(self):
        return f"Wallet({self.address[:12]}... balance={self.balance})"

class TPSBenchmark:
    def __init__(self, num_wallets: int = 100):
        self.num_wallets = num_wallets
        self.wallets: List[Wallet] = []
        self.session: aiohttp.ClientSession = None
        self.results = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "start_time": None,
            "end_time": None,
            "tps": 0,
            "latencies": []
        }

    async def setup(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(ssl=False, limit=1000)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        print(f"✅ Session initialized with connection pool")

    async def cleanup(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def check_health(self) -> bool:
        """Check API health"""
        try:
            async with self.session.get(f"{API_BASE}/health") as resp:
                data = await resp.json()
                print(f"✅ API Health: {data}")
                return data.get("success", False)
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    async def create_wallets(self, count: int):
        """Create wallet addresses"""
        print(f"\n🔑 Creating {count} wallets...")
        for i in range(count):
            seed = f"tps_benchmark_wallet_{i}_{datetime.now().timestamp()}"
            wallet = Wallet(seed)
            self.wallets.append(wallet)
        print(f"✅ Created {len(self.wallets)} wallets")
        for i, w in enumerate(self.wallets[:5]):
            print(f"  Wallet {i+1}: {w.address}")
        if len(self.wallets) > 5:
            print(f"  ... and {len(self.wallets) - 5} more")

    async def request_faucet(self, wallet: Wallet) -> bool:
        """Request faucet funds for a wallet"""
        try:
            payload = {"address": wallet.address}
            async with self.session.post(
                f"{API_BASE}/api/faucet",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        wallet.balance = 100  # Faucet gives 100 QNK
                        return True
                return False
        except Exception as e:
            print(f"❌ Faucet request failed for {wallet.address[:12]}: {e}")
            return False

    async def fund_wallets(self):
        """Fund all wallets via faucet"""
        print(f"\n💰 Requesting faucet funds for {len(self.wallets)} wallets...")

        # Fund in batches to avoid overwhelming the API
        batch_size = 10
        funded = 0

        for i in range(0, len(self.wallets), batch_size):
            batch = self.wallets[i:i+batch_size]
            tasks = [self.request_faucet(w) for w in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            funded += sum(1 for r in results if r is True)
            print(f"  Funded {funded}/{len(self.wallets)} wallets...")

            # Rate limit
            await asyncio.sleep(0.5)

        print(f"✅ Successfully funded {funded} wallets")
        return funded

    async def get_balance(self, address: str) -> float:
        """Get wallet balance"""
        try:
            async with self.session.get(f"{API_BASE}/api/wallet/balance/{address}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        return float(data.get("data", {}).get("balance", 0))
        except:
            pass
        return 0

    async def send_transaction(self, from_wallet: Wallet, to_wallet: Wallet, amount: float) -> Tuple[bool, float]:
        """Send a transaction and measure latency"""
        start = time.time()

        try:
            payload = {
                "from": from_wallet.address,
                "to": to_wallet.address,
                "amount": amount,
                "timestamp": datetime.now().isoformat()
            }

            async with self.session.post(
                f"{API_BASE}/api/transactions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                latency = time.time() - start

                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        from_wallet.balance -= amount
                        to_wallet.balance += amount
                        return True, latency

                return False, latency
        except Exception as e:
            latency = time.time() - start
            return False, latency

    async def execute_transaction_wave(self, num_transactions: int) -> Dict:
        """Execute a wave of transactions concurrently"""
        print(f"\n🚀 Executing {num_transactions} transactions...")

        tasks = []
        for i in range(num_transactions):
            # Select random sender and receiver
            from_idx = i % len(self.wallets)
            to_idx = (i + 1) % len(self.wallets)

            from_wallet = self.wallets[from_idx]
            to_wallet = self.wallets[to_idx]

            # Send small amounts
            amount = 0.1

            tasks.append(self.send_transaction(from_wallet, to_wallet, amount))

        # Execute all transactions concurrently
        self.results["start_time"] = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.results["end_time"] = time.time()

        # Process results
        successful = 0
        failed = 0
        latencies = []

        for result in results:
            if isinstance(result, tuple):
                success, latency = result
                if success:
                    successful += 1
                    latencies.append(latency)
                else:
                    failed += 1
            else:
                failed += 1

        elapsed = self.results["end_time"] - self.results["start_time"]
        tps = successful / elapsed if elapsed > 0 else 0

        self.results["total_transactions"] += num_transactions
        self.results["successful_transactions"] += successful
        self.results["failed_transactions"] += failed
        self.results["tps"] = tps
        self.results["latencies"].extend(latencies)

        return {
            "total": num_transactions,
            "successful": successful,
            "failed": failed,
            "elapsed": elapsed,
            "tps": tps,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "median_latency": statistics.median(latencies) if latencies else 0,
            "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0,
            "p99_latency": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0
        }

    async def run_benchmark(self, transaction_count: int):
        """Run complete TPS benchmark"""
        print("="*80)
        print("🌊 Q-NARWHALKNIGHT REAL-WORLD TPS BENCHMARK")
        print("="*80)
        print(f"📍 Endpoint: {API_BASE}")
        print(f"🔢 Wallets: {self.num_wallets}")
        print(f"💸 Transactions: {transaction_count}")
        print("="*80)

        # Setup
        await self.setup()

        # Check health
        if not await self.check_health():
            print("❌ API not healthy, aborting")
            return

        # Create wallets
        await self.create_wallets(self.num_wallets)

        # Fund wallets
        funded = await self.fund_wallets()
        if funded < self.num_wallets * 0.5:  # Need at least 50% funded
            print(f"⚠️  Only {funded} wallets funded, continuing anyway...")

        # Execute transactions in waves
        wave_size = min(1000, transaction_count)
        waves = (transaction_count + wave_size - 1) // wave_size

        print(f"\n📊 Executing {waves} waves of {wave_size} transactions each...")

        all_wave_results = []
        for wave in range(waves):
            wave_txs = min(wave_size, transaction_count - wave * wave_size)
            print(f"\n--- Wave {wave+1}/{waves} ({wave_txs} transactions) ---")

            wave_result = await self.execute_transaction_wave(wave_txs)
            all_wave_results.append(wave_result)

            print(f"  ✅ Success: {wave_result['successful']}/{wave_result['total']}")
            print(f"  ⚡ TPS: {wave_result['tps']:.2f}")
            print(f"  ⏱️  Avg Latency: {wave_result['avg_latency']*1000:.2f}ms")
            print(f"  📈 P95 Latency: {wave_result['p95_latency']*1000:.2f}ms")

            # Brief pause between waves
            if wave < waves - 1:
                await asyncio.sleep(1)

        # Final statistics
        print("\n" + "="*80)
        print("📊 FINAL BENCHMARK RESULTS")
        print("="*80)

        total_elapsed = sum(w["elapsed"] for w in all_wave_results)
        total_successful = sum(w["successful"] for w in all_wave_results)
        total_failed = sum(w["failed"] for w in all_wave_results)
        overall_tps = total_successful / total_elapsed if total_elapsed > 0 else 0

        all_latencies = self.results["latencies"]

        print(f"📈 Total Transactions: {transaction_count}")
        print(f"✅ Successful: {total_successful}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏱️  Total Time: {total_elapsed:.2f}s")
        print(f"⚡ OVERALL TPS: {overall_tps:.2f}")
        print(f"\n🕐 Latency Statistics:")
        if all_latencies:
            print(f"  • Average: {statistics.mean(all_latencies)*1000:.2f}ms")
            print(f"  • Median: {statistics.median(all_latencies)*1000:.2f}ms")
            print(f"  • Min: {min(all_latencies)*1000:.2f}ms")
            print(f"  • Max: {max(all_latencies)*1000:.2f}ms")
            if len(all_latencies) >= 20:
                print(f"  • P95: {statistics.quantiles(all_latencies, n=20)[18]*1000:.2f}ms")
            if len(all_latencies) >= 100:
                print(f"  • P99: {statistics.quantiles(all_latencies, n=100)[98]*1000:.2f}ms")

        print("="*80)

        # Save results
        report = {
            "benchmark_type": "real_world_tps",
            "endpoint": API_BASE,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "num_wallets": self.num_wallets,
                "total_transactions": transaction_count
            },
            "results": {
                "total_transactions": transaction_count,
                "successful": total_successful,
                "failed": total_failed,
                "elapsed_seconds": total_elapsed,
                "tps": overall_tps,
                "latency_stats": {
                    "avg_ms": statistics.mean(all_latencies)*1000 if all_latencies else 0,
                    "median_ms": statistics.median(all_latencies)*1000 if all_latencies else 0,
                    "min_ms": min(all_latencies)*1000 if all_latencies else 0,
                    "max_ms": max(all_latencies)*1000 if all_latencies else 0,
                }
            },
            "wave_results": all_wave_results
        }

        report_file = f"real_world_tps_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Full report saved to: {report_file}")

        await self.cleanup()

async def main():
    """Main benchmark execution"""
    # Start small, then scale up
    benchmarks = [
        {"wallets": 10, "transactions": 100, "name": "Warmup"},
        {"wallets": 50, "transactions": 1000, "name": "Small Scale"},
        {"wallets": 100, "transactions": 5000, "name": "Medium Scale"},
        # Uncomment for larger tests:
        # {"wallets": 500, "transactions": 10000, "name": "Large Scale"},
        # {"wallets": 1000, "transactions": 50000, "name": "Very Large Scale"},
        # {"wallets": 5000, "transactions": 100000, "name": "Massive Scale"},
    ]

    for config in benchmarks:
        print(f"\n\n{'='*80}")
        print(f"🎯 Running {config['name']} Benchmark")
        print(f"{'='*80}\n")

        benchmark = TPSBenchmark(num_wallets=config["wallets"])
        await benchmark.run_benchmark(config["transactions"])

        # Pause between benchmarks
        print(f"\n⏸️  Pausing 5 seconds before next benchmark...\n")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
