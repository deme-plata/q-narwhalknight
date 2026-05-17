#!/usr/bin/env python3
"""
Real Wallet TPS Benchmark - Using Actual Web Wallet Flow
Tests performance using the same APIs that the web wallet uses
"""

import requests
import json
import time
import random
from datetime import datetime, timezone
from typing import List, Dict

class RealWalletBenchmark:
    def __init__(self, node_url: str = "http://localhost:8200"):
        self.node_url = node_url
        self.session = requests.Session()
        self.wallets = []

    def create_wallet(self, name: str, password: str) -> Dict:
        """Create wallet using actual API endpoint"""
        response = self.session.post(
            f"{self.node_url}/api/v1/wallets",
            json={
                "name": name,
                "password": password
            }
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                wallet = data.get("data")
                print(f"  ✅ Created wallet: {wallet['name']} (address: {wallet['address'][:16]}...)")
                return wallet

        raise Exception(f"Failed to create wallet: {response.text}")

    def get_wallet_balance(self, wallet_id: str) -> int:
        """Get wallet balance using actual API"""
        response = self.session.get(f"{self.node_url}/api/v1/wallets/{wallet_id}/balance")

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("data", {}).get("balance", 0)

        return 0

    def faucet_request(self, address: str) -> bool:
        """Request coins from faucet using actual API"""
        response = self.session.post(
            f"{self.node_url}/api/v1/faucet",
            json={"address": address}
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("success", False)

        return False

    def create_transaction(self, from_wallet: Dict, to_address: str, amount: int) -> Dict:
        """Create transaction using actual wallet API"""
        response = self.session.post(
            f"{self.node_url}/api/v1/wallets/{from_wallet['id']}/transactions",
            json={
                "to": to_address,
                "amount": amount,
                "fee": 1
            }
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("data")

        raise Exception(f"Failed to create transaction: {response.text}")

    def submit_transaction_batch(self, transactions: List[Dict]) -> Dict:
        """Submit batch of transactions using batch API"""
        batch_request = {
            "transactions": transactions
        }

        response = self.session.post(
            f"{self.node_url}/api/v1/transactions/batch",
            json=batch_request,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Batch submission failed: {response.status_code} - {response.text}")

def run_real_wallet_benchmark():
    print("🚀 Real Wallet TPS Benchmark")
    print("=" * 60)
    print()
    print("Using actual web wallet APIs to test real-world performance")
    print()

    benchmark = RealWalletBenchmark()

    # Check node health
    print("🏥 Checking node health...")
    try:
        resp = requests.get(f"{benchmark.node_url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✅ Node is healthy")
        else:
            print(f"  ❌ Node returned status {resp.status_code}")
            return
    except Exception as e:
        print(f"  ❌ Node not responding: {e}")
        return

    print()

    # Create test wallets using real API
    print("👛 Creating test wallets using /api/v1/wallets...")
    sender_wallet = benchmark.create_wallet("benchmark-sender", "test123")
    receiver_wallets = []

    for i in range(5):
        wallet = benchmark.create_wallet(f"benchmark-receiver-{i}", "test123")
        receiver_wallets.append(wallet)

    print()

    # Fund sender wallet using faucet
    print("💰 Funding sender wallet from faucet...")
    for _ in range(3):  # Request multiple times to get enough balance
        if benchmark.faucet_request(sender_wallet['address']):
            print(f"  ✅ Faucet request successful")
        time.sleep(0.5)

    # Check balance
    balance = benchmark.get_wallet_balance(sender_wallet['id'])
    print(f"  💵 Sender balance: {balance} coins")
    print()

    if balance < 1000:
        print("⚠️  Warning: Low balance, may affect benchmark results")
        print()

    # Generate real transactions using wallet API
    print("📝 Creating real transactions using wallet API...")
    batch_size = 100
    num_batches = 10

    all_transactions = []
    start_gen = time.time()

    for batch_num in range(num_batches):
        for i in range(batch_size):
            receiver = receiver_wallets[i % len(receiver_wallets)]

            try:
                # Create transaction using real wallet endpoint
                # This generates proper signatures using the wallet's private key
                tx_data = benchmark.create_transaction(
                    sender_wallet,
                    receiver['address'],
                    10  # Small amount
                )

                # Format for batch API
                all_transactions.append({
                    "transaction": tx_data
                })

                if len(all_transactions) % 100 == 0:
                    print(f"  Generated {len(all_transactions)} real signed transactions...")

            except Exception as e:
                print(f"  ⚠️  Transaction creation failed: {e}")
                break

    gen_time = time.time() - start_gen
    print(f"  ✅ Generated {len(all_transactions)} real transactions in {gen_time:.2f}s")
    print(f"  📊 Generation rate: {len(all_transactions)/gen_time:.0f} tx/sec")
    print()

    # Submit in batches
    print("🚀 Submitting transaction batches to node...")
    print()

    total_submitted = 0
    total_failed = 0
    batch_results = []

    start_bench = time.time()

    for batch_num in range(0, len(all_transactions), batch_size):
        batch = all_transactions[batch_num:batch_num + batch_size]

        try:
            result = benchmark.submit_transaction_batch(batch)

            if result.get("success"):
                data = result.get("data", {})
                submitted = data.get("submitted", 0)
                failed = data.get("failed", 0)
                tps = data.get("tps", 0)

                total_submitted += submitted
                total_failed += failed
                batch_results.append({
                    "batch": batch_num // batch_size + 1,
                    "submitted": submitted,
                    "tps": tps
                })

                print(f"  Batch {batch_num // batch_size + 1}: {submitted} submitted, {tps} TPS")
            else:
                print(f"  ❌ Batch {batch_num // batch_size + 1} failed: {result.get('error')}")
                total_failed += len(batch)

        except Exception as e:
            print(f"  ❌ Batch {batch_num // batch_size + 1} error: {e}")
            total_failed += len(batch)

    end_bench = time.time()
    total_time = end_bench - start_bench
    overall_tps = total_submitted / total_time if total_time > 0 else 0

    print()
    print("📈 Real Wallet Benchmark Results:")
    print("=" * 60)
    print(f"  Total transactions generated: {len(all_transactions)}")
    print(f"  Total submitted: {total_submitted}")
    print(f"  Total failed: {total_failed}")
    print(f"  Success rate: {(total_submitted / len(all_transactions) * 100):.1f}%")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall TPS: {overall_tps:.0f}")
    print()

    if batch_results:
        avg_batch_tps = sum(r['tps'] for r in batch_results) / len(batch_results)
        max_batch_tps = max(r['tps'] for r in batch_results)
        print(f"  Average batch TPS: {avg_batch_tps:.0f}")
        print(f"  Peak batch TPS: {max_batch_tps:.0f}")
        print()

    print("✅ Real-world performance metrics:")
    print("  - Used actual wallet creation API")
    print("  - Used actual faucet API")
    print("  - Used actual transaction signing API")
    print("  - Used actual batch submission API")
    print("  - All signatures generated by real wallet private keys")
    print("  - Matches exact web wallet user flow")
    print()

if __name__ == "__main__":
    run_real_wallet_benchmark()
