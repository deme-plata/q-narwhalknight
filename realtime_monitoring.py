#!/usr/bin/env python3
"""
🔬 Real-time Q-NarwhalKnight Network Monitoring for 50-Node Test
Monitors DNS-phantom bridge activity, consensus performance, and scaling metrics
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor

class NetworkMonitor:
    def __init__(self, node_count=50, base_port=8081):
        self.node_count = node_count
        self.base_port = base_port
        self.metrics = {
            'phantom_discoveries': 0,
            'bridge_connections': 0,
            'tor_circuits': 0,
            'consensus_ready': 0,
            'active_nodes': 0,
            'total_transactions': 0,
            'avg_tps': 0,
            'network_latency': 0,
            'consensus_rounds': 0
        }
        self.node_metrics = {}
        
    async def check_node_health(self, session, node_id):
        """Check individual node health and metrics"""
        port = self.base_port + node_id - 1
        try:
            # Try API endpoint first
            async with session.get(f"http://localhost:{port}/api/v1/health", 
                                 timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status == 200:
                    return await self.extract_node_metrics(node_id)
        except:
            pass
        
        # Fall back to docker logs analysis
        return await self.extract_node_metrics(node_id)
    
    async def extract_node_metrics(self, node_id):
        """Extract metrics from node logs"""
        try:
            container_name = f"qnk-node-{node_id}"
            
            # Get recent logs
            proc = await asyncio.create_subprocess_exec(
                'docker', 'logs', '--tail', '100', container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            logs = stdout.decode() + stderr.decode()
            
            # Extract metrics using regex
            metrics = {
                'node_id': node_id,
                'is_healthy': 'API server listening' in logs,
                'phantom_discoveries': len(re.findall(r'phantom discovered peer', logs)),
                'bridge_connections': len(re.findall(r'P2P connection established', logs)),
                'tor_circuits': len(re.findall(r'Tor circuit established', logs)),
                'consensus_ready': 'Consensus network ready' in logs,
                'transactions': len(re.findall(r'transaction processed', logs)),
                'consensus_rounds': len(re.findall(r'consensus round completed', logs)),
                'last_activity': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            return {
                'node_id': node_id,
                'is_healthy': False,
                'error': str(e),
                'last_activity': datetime.now().isoformat()
            }
    
    async def collect_network_metrics(self):
        """Collect metrics from all nodes"""
        print(f"🔍 Scanning {self.node_count} nodes for metrics...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_node_health(session, node_id) 
                    for node_id in range(1, self.node_count + 1)]
            
            node_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # Aggregate metrics
        self.metrics = {
            'phantom_discoveries': 0,
            'bridge_connections': 0,
            'tor_circuits': 0,
            'consensus_ready': 0,
            'active_nodes': 0,
            'total_transactions': 0,
            'consensus_rounds': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        healthy_nodes = []
        
        for result in node_results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                self.node_metrics[result['node_id']] = result
                
                if result.get('is_healthy', False):
                    self.metrics['active_nodes'] += 1
                    healthy_nodes.append(result['node_id'])
                
                self.metrics['phantom_discoveries'] += result.get('phantom_discoveries', 0)
                self.metrics['bridge_connections'] += result.get('bridge_connections', 0)
                self.metrics['tor_circuits'] += result.get('tor_circuits', 0)
                self.metrics['total_transactions'] += result.get('transactions', 0)
                self.metrics['consensus_rounds'] += result.get('consensus_rounds', 0)
                
                if result.get('consensus_ready', False):
                    self.metrics['consensus_ready'] += 1
        
        # Calculate derived metrics
        if self.metrics['active_nodes'] > 0:
            self.metrics['avg_discoveries_per_node'] = round(
                self.metrics['phantom_discoveries'] / self.metrics['active_nodes'], 2)
            self.metrics['avg_bridges_per_node'] = round(
                self.metrics['bridge_connections'] / self.metrics['active_nodes'], 2)
            self.metrics['bridge_success_rate'] = round(
                (self.metrics['bridge_connections'] / max(1, self.metrics['phantom_discoveries'])) * 100, 2)
        
        return healthy_nodes
    
    def print_dashboard(self):
        """Print real-time dashboard"""
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print("=" * 80)
        print(f"🔬 Q-NARWHALKNIGHT 50-NODE REAL-TIME MONITORING DASHBOARD")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Network Status
        print(f"\n🌐 NETWORK STATUS:")
        print(f"   Active Nodes: {self.metrics['active_nodes']}/50")
        print(f"   Consensus Ready: {self.metrics['consensus_ready']}/50")
        consensus_threshold = 34
        consensus_met = "✅ MET" if self.metrics['consensus_ready'] >= consensus_threshold else "❌ NOT MET"
        print(f"   Byzantine Threshold (34): {consensus_met}")
        
        # DNS-Phantom Bridge Metrics
        print(f"\n👻 DNS-PHANTOM BRIDGE PERFORMANCE:")
        print(f"   Total Phantom Discoveries: {self.metrics['phantom_discoveries']}")
        print(f"   Total Bridge Connections: {self.metrics['bridge_connections']}")
        print(f"   Bridge Success Rate: {self.metrics.get('bridge_success_rate', 0)}%")
        print(f"   Avg Discoveries/Node: {self.metrics.get('avg_discoveries_per_node', 0)}")
        print(f"   Avg Bridges/Node: {self.metrics.get('avg_bridges_per_node', 0)}")
        
        # Tor Network Metrics
        print(f"\n🧅 TOR NETWORK STATUS:")
        print(f"   Total Tor Circuits: {self.metrics['tor_circuits']}")
        print(f"   Avg Circuits/Node: {round(self.metrics['tor_circuits'] / max(1, self.metrics['active_nodes']), 1)}")
        
        # Consensus Performance
        print(f"\n🏛️ CONSENSUS PERFORMANCE:")
        print(f"   Total Consensus Rounds: {self.metrics['consensus_rounds']}")
        print(f"   Total Transactions: {self.metrics['total_transactions']}")
        print(f"   Avg Rounds/Node: {round(self.metrics['consensus_rounds'] / max(1, self.metrics['active_nodes']), 1)}")
        
        # Top Performing Nodes
        print(f"\n⭐ TOP PERFORMING NODES:")
        sorted_nodes = sorted(self.node_metrics.items(), 
                            key=lambda x: x[1].get('bridge_connections', 0), reverse=True)
        for i, (node_id, metrics) in enumerate(sorted_nodes[:5]):
            if metrics.get('is_healthy', False):
                discoveries = metrics.get('phantom_discoveries', 0)
                bridges = metrics.get('bridge_connections', 0) 
                print(f"   #{i+1} Node-{node_id}: {discoveries} discoveries → {bridges} bridges")
        
        # System Health
        try:
            # Get Docker stats
            proc = subprocess.run(['docker', 'stats', '--no-stream', '--format', 
                                 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'],
                                capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                lines = proc.stdout.strip().split('\n')[1:]  # Skip header
                qnk_lines = [line for line in lines if 'qnk-node-' in line][:5]
                
                if qnk_lines:
                    print(f"\n💻 SYSTEM RESOURCES (Sample):")
                    for line in qnk_lines:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            print(f"   {parts[0]}: CPU {parts[1]}, Memory {parts[2]}")
        except:
            pass
        
        print(f"\n📊 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
    
    async def monitor_loop(self, interval=10):
        """Main monitoring loop"""
        print("🚀 Starting Q-NarwhalKnight Network Monitor...")
        print(f"📡 Monitoring {self.node_count} nodes every {interval}s")
        
        while True:
            try:
                healthy_nodes = await self.collect_network_metrics()
                self.print_dashboard()
                
                # Save metrics to file for later analysis
                with open('/tmp/network_metrics.json', 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'summary': self.metrics,
                        'nodes': self.node_metrics
                    }, f, indent=2)
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Monitoring error: {e}")
                await asyncio.sleep(5)

async def main():
    node_count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    monitor = NetworkMonitor(node_count=node_count)
    await monitor.monitor_loop(interval=interval)

if __name__ == "__main__":
    asyncio.run(main())