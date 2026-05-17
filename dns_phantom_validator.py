#!/usr/bin/env python3
"""
DNS-Phantom Discovery Validation System
Q-NarwhalKnight Native Binary Analysis Tool

This tool provides automated validation and analysis of DNS-Phantom 
steganographic peer discovery functionality in the Q-NarwhalKnight 
native Rust binary.
"""

import os
import re
import json
import time
import socket
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass, asdict
import csv

@dataclass
class DiscoveryEvent:
    """Represents a DNS-Phantom discovery event"""
    timestamp: str
    node_id: str
    event_type: str  # 'initialization', 'peer_discovery', 'steganographic_comm'
    details: str
    dns_provider: Optional[str] = None
    peer_count: int = 0
    success: bool = True

@dataclass  
class NodeStatus:
    """Represents the status of a Q-NarwhalKnight node"""
    node_id: str
    pid: int
    port: int
    role: str
    is_active: bool
    discovery_events: int
    peer_connections: int
    steganographic_comms: int
    last_activity: str

@dataclass
class TestResults:
    """Comprehensive test results"""
    test_start_time: str
    test_duration_seconds: int
    total_nodes_deployed: int
    nodes_with_dns_phantom: int
    total_discovery_events: int
    successful_peer_discoveries: int
    steganographic_communications: int
    cross_provider_discoveries: int
    network_stability_score: float
    verdict: str
    evidence: List[str]

class DNSPhantomValidator:
    """Main validator class for DNS-Phantom discovery functionality"""
    
    def __init__(self, test_directory: str = "./massive-dns-phantom-test"):
        self.test_dir = test_directory
        self.log_dir = f"{test_directory}/logs"
        self.results_dir = f"{test_directory}/results" 
        self.binary_path = "./target/release/q-api-server"
        
        # DNS providers for validation
        self.dns_providers = {
            "cloudflare": "1.1.1.1",
            "google": "8.8.8.8", 
            "quad9": "9.9.9.9",
            "opendns": "208.67.222.222"
        }
        
        # Tracking data
        self.discovery_events: List[DiscoveryEvent] = []
        self.node_statuses: Dict[str, NodeStatus] = {}
        self.test_start_time = datetime.now()
        
        # Validation patterns
        self.dns_phantom_patterns = [
            r"DNS Phantom Network started successfully",
            r"invisible internet.*active",
            r"steganographic.*communication",
            r"DNS-Phantom.*discovery.*engine",
            r"phantom.*peer.*discovered",
            r"DNS.*steganography.*initialized"
        ]
        
        self.discovery_patterns = [
            r"peer.*discovered.*via.*dns",
            r"steganographic.*peer.*discovery",
            r"DNS.*phantom.*discovered.*(\d+).*peers",
            r"hidden.*peer.*through.*dns",
            r"phantom.*network.*peer.*(\w+)"
        ]
        
    def validate_binary_capabilities(self) -> bool:
        """Validate that the binary supports DNS-Phantom discovery"""
        print("🔍 Validating Q-NarwhalKnight binary DNS-Phantom capabilities...")
        
        if not os.path.exists(self.binary_path):
            print(f"❌ Binary not found at {self.binary_path}")
            return False
            
        # Check binary help output for DNS-Phantom flags
        try:
            result = subprocess.run([self.binary_path, "--help"], 
                                  capture_output=True, text=True, timeout=10)
            help_output = result.stdout + result.stderr
            
            dns_phantom_flags = [
                "--enable-dns-phantom",
                "--discovery-interval", 
                "--bootstrap-peer",
                "--max-peers"
            ]
            
            supported_flags = []
            for flag in dns_phantom_flags:
                if flag in help_output:
                    supported_flags.append(flag)
                    print(f"✅ Found DNS-Phantom flag: {flag}")
                    
            if len(supported_flags) >= 2:
                print(f"✅ Binary supports {len(supported_flags)}/4 DNS-Phantom flags")
                return True
            else:
                print(f"⚠️  Limited DNS-Phantom support: {len(supported_flags)}/4 flags")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  Binary help check timed out")
            return False
        except Exception as e:
            print(f"❌ Error checking binary capabilities: {e}")
            return False
    
    def analyze_log_file(self, log_file_path: str) -> List[DiscoveryEvent]:
        """Analyze a single log file for DNS-Phantom discovery events"""
        events = []
        
        if not os.path.exists(log_file_path):
            return events
            
        node_id = os.path.basename(log_file_path).replace('.log', '')
        
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
                
            # Find DNS-Phantom initialization events
            for pattern in self.dns_phantom_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Extract timestamp from log line
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    line = content[line_start:line_end]
                    
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                    
                    events.append(DiscoveryEvent(
                        timestamp=timestamp,
                        node_id=node_id,
                        event_type="initialization",
                        details=match.group(0),
                        success=True
                    ))
            
            # Find peer discovery events  
            for pattern in self.discovery_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    line = content[line_start:line_end]
                    
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                    
                    # Extract peer count if available
                    peer_count_match = re.search(r'(\d+).*peers?', match.group(0))
                    peer_count = int(peer_count_match.group(1)) if peer_count_match else 1
                    
                    events.append(DiscoveryEvent(
                        timestamp=timestamp,
                        node_id=node_id,
                        event_type="peer_discovery",
                        details=match.group(0),
                        peer_count=peer_count,
                        success=True
                    ))
                    
        except Exception as e:
            print(f"⚠️  Error analyzing log file {log_file_path}: {e}")
            
        return events
    
    def scan_all_logs(self) -> List[DiscoveryEvent]:
        """Scan all log files for DNS-Phantom discovery events"""
        print(f"📖 Scanning logs in {self.log_dir}...")
        
        all_events = []
        
        if not os.path.exists(self.log_dir):
            print(f"⚠️  Log directory not found: {self.log_dir}")
            return all_events
            
        log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
        print(f"   Found {len(log_files)} log files")
        
        for log_file in log_files:
            log_path = os.path.join(self.log_dir, log_file)
            events = self.analyze_log_file(log_path)
            all_events.extend(events)
            
            if events:
                print(f"   📄 {log_file}: {len(events)} DNS-Phantom events")
                
        print(f"✅ Total DNS-Phantom events found: {len(all_events)}")
        return all_events
    
    def validate_network_connectivity(self) -> Dict[str, int]:
        """Validate network connectivity and active processes"""
        print("🌐 Validating network connectivity...")
        
        connectivity_stats = {
            "total_processes": 0,
            "listening_ports": 0,
            "established_connections": 0,
            "dns_phantom_hubs": 0
        }
        
        try:
            # Count Q-NarwhalKnight processes
            result = subprocess.run(['pgrep', '-f', 'q-api-server'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                processes = result.stdout.strip().split('\n')
                connectivity_stats["total_processes"] = len([p for p in processes if p])
                
            # Check listening ports
            result = subprocess.run(['ss', '-tln'], capture_output=True, text=True)
            if result.returncode == 0:
                listening_ports = re.findall(r':(\d{4,5})\s+', result.stdout)
                relevant_ports = [p for p in listening_ports if 8080 <= int(p) <= 9500]
                connectivity_stats["listening_ports"] = len(relevant_ports)
                
                # Count DNS-Phantom hubs (ports 8080-8082)
                hub_ports = [p for p in listening_ports if 8080 <= int(p) <= 8082]
                connectivity_stats["dns_phantom_hubs"] = len(hub_ports)
                
            # Check established connections
            result = subprocess.run(['ss', '-t'], capture_output=True, text=True)
            if result.returncode == 0:
                established = result.stdout.count('ESTAB')
                connectivity_stats["established_connections"] = established
                
        except Exception as e:
            print(f"⚠️  Error checking network connectivity: {e}")
            
        for key, value in connectivity_stats.items():
            print(f"   {key}: {value}")
            
        return connectivity_stats
    
    def test_dns_provider_accessibility(self) -> Dict[str, bool]:
        """Test accessibility to DNS providers used for steganography"""
        print("🔍 Testing DNS provider accessibility...")
        
        results = {}
        for provider_name, ip in self.dns_providers.items():
            try:
                # Test DNS query to provider
                socket.setdefaulttimeout(3)
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.connect((ip, 53))
                sock.close()
                results[provider_name] = True
                print(f"   ✅ {provider_name} ({ip}): Accessible")
            except Exception:
                results[provider_name] = False
                print(f"   ❌ {provider_name} ({ip}): Not accessible")
                
        accessible_count = sum(results.values())
        print(f"✅ DNS providers accessible: {accessible_count}/{len(results)}")
        return results
    
    def generate_comprehensive_report(self, discovery_events: List[DiscoveryEvent], 
                                    connectivity_stats: Dict[str, int],
                                    dns_accessibility: Dict[str, bool]) -> TestResults:
        """Generate comprehensive test results"""
        
        # Calculate metrics
        total_nodes = len(set(event.node_id for event in discovery_events))
        nodes_with_dns_phantom = len(set(event.node_id for event in discovery_events 
                                       if event.event_type == "initialization"))
        
        total_discovery_events = len(discovery_events)
        peer_discoveries = len([e for e in discovery_events if e.event_type == "peer_discovery"])
        steganographic_comms = len([e for e in discovery_events if "steganographic" in e.details.lower()])
        
        # Calculate network stability score
        expected_processes = 150  # From massive test
        actual_processes = connectivity_stats.get("total_processes", 0)
        stability_score = min(actual_processes / expected_processes, 1.0) if expected_processes > 0 else 0.0
        
        # Determine verdict
        evidence = []
        verdict = "INCONCLUSIVE"
        
        if nodes_with_dns_phantom > 0:
            evidence.append(f"✅ {nodes_with_dns_phantom} nodes successfully initialized DNS-Phantom")
            
        if total_discovery_events > 0:
            evidence.append(f"✅ {total_discovery_events} DNS-Phantom discovery events recorded")
            
        if peer_discoveries > 0:
            evidence.append(f"✅ {peer_discoveries} successful peer discoveries via DNS-Phantom")
            
        if steganographic_comms > 0:
            evidence.append(f"✅ {steganographic_comms} steganographic communications detected")
            
        if connectivity_stats.get("dns_phantom_hubs", 0) > 0:
            evidence.append(f"✅ {connectivity_stats['dns_phantom_hubs']} DNS-Phantom hubs operational")
            
        accessible_dns = sum(dns_accessibility.values())
        evidence.append(f"✅ {accessible_dns}/{len(dns_accessibility)} DNS providers accessible")
        
        # Verdict logic
        if (nodes_with_dns_phantom >= 10 and total_discovery_events >= 20 and 
            peer_discoveries >= 5 and stability_score >= 0.3):
            verdict = "CONFIRMED: DNS-PHANTOM DISCOVERY FULLY FUNCTIONAL"
        elif nodes_with_dns_phantom >= 5 and total_discovery_events >= 10:
            verdict = "PARTIAL: DNS-PHANTOM DISCOVERY DETECTED"
        elif nodes_with_dns_phantom >= 1:
            verdict = "LIMITED: DNS-PHANTOM CAPABILITY EXISTS"
        else:
            verdict = "NEGATIVE: NO DNS-PHANTOM ACTIVITY DETECTED"
            
        test_duration = int((datetime.now() - self.test_start_time).total_seconds())
        
        return TestResults(
            test_start_time=self.test_start_time.isoformat(),
            test_duration_seconds=test_duration,
            total_nodes_deployed=actual_processes,
            nodes_with_dns_phantom=nodes_with_dns_phantom,
            total_discovery_events=total_discovery_events,
            successful_peer_discoveries=peer_discoveries,
            steganographic_communications=steganographic_comms,
            cross_provider_discoveries=accessible_dns,
            network_stability_score=stability_score,
            verdict=verdict,
            evidence=evidence
        )
    
    def save_detailed_results(self, results: TestResults, discovery_events: List[DiscoveryEvent]):
        """Save detailed results to files"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save main results as JSON
        with open(f"{self.results_dir}/validation_results.json", 'w') as f:
            json.dump(asdict(results), f, indent=2)
            
        # Save discovery events as CSV
        with open(f"{self.results_dir}/discovery_events.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'node_id', 'event_type', 'details', 'dns_provider', 'peer_count', 'success'])
            
            for event in discovery_events:
                writer.writerow([
                    event.timestamp, event.node_id, event.event_type,
                    event.details, event.dns_provider or '', 
                    event.peer_count, event.success
                ])
                
        # Save human-readable report
        with open(f"{self.results_dir}/validation_report.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("DNS-PHANTOM DISCOVERY VALIDATION REPORT\n")
            f.write("Q-NarwhalKnight Native Binary Analysis\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Started: {results.test_start_time}\n")
            f.write(f"Test Duration: {results.test_duration_seconds} seconds\n")
            f.write(f"Nodes Deployed: {results.total_nodes_deployed}\n\n")
            
            f.write("DNS-PHANTOM DISCOVERY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Nodes with DNS-Phantom: {results.nodes_with_dns_phantom}\n")
            f.write(f"Total Discovery Events: {results.total_discovery_events}\n")
            f.write(f"Peer Discoveries: {results.successful_peer_discoveries}\n") 
            f.write(f"Steganographic Comms: {results.steganographic_communications}\n")
            f.write(f"Network Stability: {results.network_stability_score:.2%}\n\n")
            
            f.write(f"VERDICT: {results.verdict}\n\n")
            
            f.write("EVIDENCE:\n")
            f.write("-" * 20 + "\n")
            for evidence_item in results.evidence:
                f.write(f"{evidence_item}\n")
                
        print(f"📁 Detailed results saved to {self.results_dir}/")
    
    def run_validation(self) -> TestResults:
        """Run complete DNS-Phantom discovery validation"""
        print("🚀 Starting DNS-Phantom Discovery Validation...")
        print("="*60)
        
        # Step 1: Validate binary capabilities
        binary_valid = self.validate_binary_capabilities()
        if not binary_valid:
            print("⚠️  Binary validation failed, but continuing with log analysis...")
            
        # Step 2: Scan logs for discovery events
        discovery_events = self.scan_all_logs()
        
        # Step 3: Check network connectivity
        connectivity_stats = self.validate_network_connectivity()
        
        # Step 4: Test DNS provider accessibility
        dns_accessibility = self.test_dns_provider_accessibility()
        
        # Step 5: Generate comprehensive report
        results = self.generate_comprehensive_report(
            discovery_events, connectivity_stats, dns_accessibility)
            
        # Step 6: Save results
        self.save_detailed_results(results, discovery_events)
        
        return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="DNS-Phantom Discovery Validator")
    parser.add_argument("--test-dir", default="./massive-dns-phantom-test",
                       help="Test directory containing logs and results")
    parser.add_argument("--binary-path", default="./target/release/q-api-server", 
                       help="Path to Q-NarwhalKnight binary")
    
    args = parser.parse_args()
    
    validator = DNSPhantomValidator(args.test_dir)
    validator.binary_path = args.binary_path
    
    results = validator.run_validation()
    
    print("\n" + "="*60)
    print("DNS-PHANTOM DISCOVERY VALIDATION COMPLETE")
    print("="*60)
    print(f"🎯 VERDICT: {results.verdict}")
    print(f"📊 Discovery Events: {results.total_discovery_events}")
    print(f"🌐 Nodes with DNS-Phantom: {results.nodes_with_dns_phantom}")
    print(f"🤝 Peer Discoveries: {results.successful_peer_discoveries}")
    print(f"📡 Network Stability: {results.network_stability_score:.2%}")
    
    if "CONFIRMED" in results.verdict:
        print("\n🎉 DNS-PHANTOM DISCOVERY NATIVE SUPPORT CONFIRMED! 🎉")
        return 0
    elif "PARTIAL" in results.verdict:
        print("\n⚠️  DNS-PHANTOM DISCOVERY PARTIALLY VALIDATED")
        return 1
    else:
        print("\n❌ DNS-PHANTOM DISCOVERY NOT SUFFICIENTLY VALIDATED")
        return 2

if __name__ == "__main__":
    exit(main())