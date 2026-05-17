#!/usr/bin/env python3
"""
Q-NarwhalKnight 5-Node Performance Testing Suite
Measures MB/s throughput and TPS (Transactions Per Second)
"""

import json
import time
import random
import asyncio
import aiohttp
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
import subprocess
import sys

class PerformanceTest:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.results = {
            'transactions_sent': 0,
            'transactions_confirmed': 0,
            'bytes_transferred': 0,
            'start_time': None,
            'end_time': None,
            'tps_samples': [],
            'mbps_samples': [],
            'latency_samples': [],
            'errors': 0
        }
    
    def generate_transaction(self, size_bytes: int = 1024) -> Dict:
        """Generate a transaction of specified size"""
        # Base transaction structure
        tx = {
            'id': f'tx_{int(time.time() * 1000000)}_{random.randint(0, 999999)}',
            'from': f'node_{random.randint(1, 5)}',
            'to': f'node_{random.randint(1, 5)}',
            'amount': random.randint(1, 1000),
            'timestamp': time.time(),
            'nonce': random.randint(0, 2**32),
        }
        
        # Add padding to reach desired size
        current_size = len(json.dumps(tx).encode())
        if size_bytes > current_size:
            padding_size = size_bytes - current_size - 20  # Account for JSON overhead
            tx['data'] = 'x' * max(0, padding_size)
        
        return tx
    
    async def send_transaction(self, session: aiohttp.ClientSession, node_url: str, tx: Dict) -> Tuple[bool, float, int]:
        """Send transaction and return (success, latency, bytes)"""
        start_time = time.time()
        tx_json = json.dumps(tx)
        tx_bytes = len(tx_json.encode())
        
        try:
            async with session.post(
                f'{node_url}/api/v1/transactions',
                json=tx,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                latency = time.time() - start_time
                success = response.status == 200
                return success, latency, tx_bytes
        except Exception as e:
            return False, 0, 0
    
    async def send_batch(self, session: aiohttp.ClientSession, tx_count: int, tx_size: int) -> Dict:
        """Send a batch of transactions"""
        batch_results = {
            'sent': 0,
            'confirmed': 0,
            'bytes': 0,
            'latencies': [],
            'start_time': time.time()
        }
        
        tasks = []
        for _ in range(tx_count):
            tx = self.generate_transaction(tx_size)
            node_url = random.choice(self.nodes)
            tasks.append(self.send_transaction(session, node_url, tx))
        
        results = await asyncio.gather(*tasks)
        
        for success, latency, tx_bytes in results:
            batch_results['sent'] += 1
            if success:
                batch_results['confirmed'] += 1
                batch_results['bytes'] += tx_bytes
                if latency > 0:
                    batch_results['latencies'].append(latency)
            else:
                self.results['errors'] += 1
        
        batch_results['duration'] = time.time() - batch_results['start_time']
        return batch_results
    
    async def run_performance_test(self, 
                                 duration_seconds: int = 60,
                                 batch_size: int = 100,
                                 tx_size: int = 1024,
                                 batch_delay: float = 0.1):
        """Run the main performance test"""
        print(f"\n🚀 Starting Performance Test")
        print(f"   Duration: {duration_seconds} seconds")
        print(f"   Batch size: {batch_size} transactions")
        print(f"   Transaction size: {tx_size} bytes")
        print(f"   Batch delay: {batch_delay} seconds")
        print(f"   Target nodes: {len(self.nodes)}")
        print("-" * 60)
        
        self.results['start_time'] = time.time()
        
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + duration_seconds
            batch_num = 0
            
            while time.time() < end_time:
                batch_num += 1
                print(f"\n📦 Batch {batch_num}:", end='')
                
                batch_results = await self.send_batch(session, batch_size, tx_size)
                
                # Update totals
                self.results['transactions_sent'] += batch_results['sent']
                self.results['transactions_confirmed'] += batch_results['confirmed']
                self.results['bytes_transferred'] += batch_results['bytes']
                
                # Calculate batch metrics
                if batch_results['duration'] > 0:
                    batch_tps = batch_results['confirmed'] / batch_results['duration']
                    batch_mbps = (batch_results['bytes'] / (1024 * 1024)) / batch_results['duration']
                    
                    self.results['tps_samples'].append(batch_tps)
                    self.results['mbps_samples'].append(batch_mbps)
                    
                    if batch_results['latencies']:
                        avg_latency = statistics.mean(batch_results['latencies'])
                        self.results['latency_samples'].extend(batch_results['latencies'])
                        
                        print(f" ✅ {batch_results['confirmed']}/{batch_results['sent']} | "
                              f"TPS: {batch_tps:.1f} | "
                              f"MB/s: {batch_mbps:.3f} | "
                              f"Latency: {avg_latency*1000:.1f}ms")
                    else:
                        print(f" ❌ Failed")
                
                # Small delay between batches
                await asyncio.sleep(batch_delay)
        
        self.results['end_time'] = time.time()
    
    def generate_report(self) -> str:
        """Generate performance report"""
        duration = self.results['end_time'] - self.results['start_time']
        
        # Overall metrics
        total_tps = self.results['transactions_confirmed'] / duration
        total_mbps = (self.results['bytes_transferred'] / (1024 * 1024)) / duration
        success_rate = (self.results['transactions_confirmed'] / 
                       self.results['transactions_sent'] * 100) if self.results['transactions_sent'] > 0 else 0
        
        # Statistical analysis
        tps_stats = {
            'avg': statistics.mean(self.results['tps_samples']) if self.results['tps_samples'] else 0,
            'median': statistics.median(self.results['tps_samples']) if self.results['tps_samples'] else 0,
            'stdev': statistics.stdev(self.results['tps_samples']) if len(self.results['tps_samples']) > 1 else 0,
            'max': max(self.results['tps_samples']) if self.results['tps_samples'] else 0,
            'min': min(self.results['tps_samples']) if self.results['tps_samples'] else 0
        }
        
        mbps_stats = {
            'avg': statistics.mean(self.results['mbps_samples']) if self.results['mbps_samples'] else 0,
            'median': statistics.median(self.results['mbps_samples']) if self.results['mbps_samples'] else 0,
            'stdev': statistics.stdev(self.results['mbps_samples']) if len(self.results['mbps_samples']) > 1 else 0,
            'max': max(self.results['mbps_samples']) if self.results['mbps_samples'] else 0,
            'min': min(self.results['mbps_samples']) if self.results['mbps_samples'] else 0
        }
        
        latency_stats = {
            'avg': statistics.mean(self.results['latency_samples']) * 1000 if self.results['latency_samples'] else 0,
            'median': statistics.median(self.results['latency_samples']) * 1000 if self.results['latency_samples'] else 0,
            'p95': statistics.quantiles(self.results['latency_samples'], n=20)[18] * 1000 if len(self.results['latency_samples']) > 20 else 0,
            'p99': statistics.quantiles(self.results['latency_samples'], n=100)[98] * 1000 if len(self.results['latency_samples']) > 100 else 0
        }
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         Q-NarwhalKnight Performance Test Results              ║
╚══════════════════════════════════════════════════════════════╝

📊 TEST SUMMARY
├─ Duration: {duration:.2f} seconds
├─ Total Transactions Sent: {self.results['transactions_sent']:,}
├─ Total Transactions Confirmed: {self.results['transactions_confirmed']:,}
├─ Success Rate: {success_rate:.2f}%
├─ Total Data Transferred: {self.results['bytes_transferred'] / (1024*1024):.2f} MB
└─ Errors: {self.results['errors']}

⚡ THROUGHPUT METRICS
├─ Overall TPS: {total_tps:.2f} transactions/second
├─ Overall Throughput: {total_mbps:.3f} MB/s
│
├─ TPS Statistics:
│  ├─ Average: {tps_stats['avg']:.2f}
│  ├─ Median: {tps_stats['median']:.2f}
│  ├─ Std Dev: {tps_stats['stdev']:.2f}
│  ├─ Maximum: {tps_stats['max']:.2f}
│  └─ Minimum: {tps_stats['min']:.2f}
│
└─ Throughput Statistics (MB/s):
   ├─ Average: {mbps_stats['avg']:.3f}
   ├─ Median: {mbps_stats['median']:.3f}
   ├─ Std Dev: {mbps_stats['stdev']:.3f}
   ├─ Maximum: {mbps_stats['max']:.3f}
   └─ Minimum: {mbps_stats['min']:.3f}

⏱️  LATENCY METRICS (milliseconds)
├─ Average: {latency_stats['avg']:.2f} ms
├─ Median: {latency_stats['median']:.2f} ms
├─ P95: {latency_stats['p95']:.2f} ms
└─ P99: {latency_stats['p99']:.2f} ms

🎯 PERFORMANCE GRADE
"""
        
        # Performance grading
        if total_tps > 10000:
            grade = "A+ (Excellent)"
        elif total_tps > 5000:
            grade = "A (Very Good)"
        elif total_tps > 1000:
            grade = "B (Good)"
        elif total_tps > 500:
            grade = "C (Acceptable)"
        else:
            grade = "D (Needs Improvement)"
        
        report += f"└─ Grade: {grade} based on {total_tps:.0f} TPS\n"
        
        return report

async def setup_docker_network():
    """Set up the 5-node Docker network"""
    print("🐳 Setting up Docker network...")
    
    # Clean up existing containers
    subprocess.run("docker rm -f qnk-perf-{1..5} 2>/dev/null", shell=True)
    subprocess.run("docker network create qnk-perf-net --subnet 172.22.0.0/24 2>/dev/null", shell=True)
    
    nodes = []
    for i in range(1, 6):
        print(f"   Starting node {i}...")
        port = 8080 + i
        cmd = f"""
        docker run -d \
            --name qnk-perf-{i} \
            --network qnk-perf-net \
            --ip 172.22.0.1{i} \
            -p {port}:8080 \
            python:3.9-alpine \
            sh -c "pip install aiohttp && python -m aiohttp.web -H 0.0.0.0 -P 8080"
        """
        subprocess.run(cmd, shell=True, capture_output=True)
        nodes.append(f"http://localhost:{port}")
    
    # Wait for nodes to be ready
    print("   Waiting for nodes to initialize...")
    time.sleep(5)
    
    return nodes

async def main():
    # Parse command line arguments
    duration = 60  # Default 60 seconds
    batch_size = 100  # Default 100 tx per batch
    tx_size = 1024  # Default 1KB per transaction
    
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        tx_size = int(sys.argv[3])
    
    print("=" * 60)
    print("Q-NarwhalKnight 5-Node Performance Test")
    print("=" * 60)
    
    # Set up network
    nodes = await setup_docker_network()
    
    # Run performance test
    tester = PerformanceTest(nodes)
    await tester.run_performance_test(
        duration_seconds=duration,
        batch_size=batch_size,
        tx_size=tx_size,
        batch_delay=0.1
    )
    
    # Generate and print report
    report = tester.generate_report()
    print(report)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_report_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {filename}")
    
    # Clean up
    print("\n🧹 Cleaning up Docker containers...")
    subprocess.run("docker rm -f qnk-perf-{1..5} 2>/dev/null", shell=True)
    subprocess.run("docker network rm qnk-perf-net 2>/dev/null", shell=True)

if __name__ == "__main__":
    asyncio.run(main())