#!/usr/bin/env python3
"""
Test and log actual node-to-node connections through Tor DHT
Provides solid evidence of peer-to-peer communication
"""

import socket
import threading
import time
import json
import subprocess
from datetime import datetime

class TorDHTNode:
    def __init__(self, name, local_port):
        self.name = name
        self.local_port = local_port
        self.onion_address = None
        self.control_socket = None
        self.connections_log = []
        self.messages_received = []
        self.messages_sent = []
        
    def create_onion_service(self):
        """Create real onion service and log it"""
        try:
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.connect(('127.0.0.1', 9051))
            
            # Authenticate
            self.control_socket.send(b"AUTHENTICATE\r\n")
            auth_response = self.control_socket.recv(1024).decode()
            
            if "250 OK" in auth_response:
                # Create onion service
                create_cmd = f"ADD_ONION NEW:BEST Port=80,127.0.0.1:{self.local_port}\r\n"
                self.control_socket.send(create_cmd.encode())
                create_response = self.control_socket.recv(2048).decode()
                
                # Parse onion address
                for line in create_response.split('\n'):
                    if line.startswith("250-ServiceID="):
                        service_id = line.replace("250-ServiceID=", "").strip()
                        self.onion_address = f"{service_id}.onion"
                        
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "event": "ONION_SERVICE_CREATED",
                            "node": self.name,
                            "onion_address": self.onion_address,
                            "port": self.local_port,
                            "tor_response": create_response.strip()
                        }
                        self.connections_log.append(log_entry)
                        return True
        except Exception as e:
            print(f"Error creating onion service for {self.name}: {e}")
        return False
    
    def start_dht_server(self):
        """Start DHT server and log all connections"""
        def server_thread():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', self.local_port))
            server.listen(5)
            
            while True:
                try:
                    client, addr = server.accept()
                    
                    # Log incoming connection
                    connection_log = {
                        "timestamp": datetime.now().isoformat(),
                        "event": "INCOMING_CONNECTION",
                        "node": self.name,
                        "from_address": str(addr),
                        "local_port": self.local_port
                    }
                    self.connections_log.append(connection_log)
                    
                    # Receive message
                    data = client.recv(4096).decode()
                    
                    # Parse and log message
                    if data:
                        message_log = {
                            "timestamp": datetime.now().isoformat(),
                            "event": "MESSAGE_RECEIVED",
                            "node": self.name,
                            "message_size": len(data),
                            "message_preview": data[:200]
                        }
                        self.messages_received.append(message_log)
                        
                        # Send response
                        response = {
                            "node": self.name,
                            "onion": self.onion_address,
                            "status": "active",
                            "timestamp": datetime.now().isoformat(),
                            "peers_known": len(self.connections_log)
                        }
                        
                        response_data = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{json.dumps(response)}"
                        client.send(response_data.encode())
                        
                        response_log = {
                            "timestamp": datetime.now().isoformat(),
                            "event": "RESPONSE_SENT",
                            "node": self.name,
                            "response": response
                        }
                        self.messages_sent.append(response_log)
                    
                    client.close()
                    
                except Exception as e:
                    pass
        
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
    
    def connect_to_peer(self, peer_onion, peer_name):
        """Connect to peer through Tor and log everything"""
        try:
            # Use curl through Tor SOCKS proxy
            cmd = [
                'curl',
                '--socks5-hostname', '127.0.0.1:9050',
                '--max-time', '30',
                f'http://{peer_onion}/',
                '-H', f'X-From-Node: {self.name}',
                '-H', f'X-From-Onion: {self.onion_address}',
                '-v'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            
            connection_result = {
                "timestamp": datetime.now().isoformat(),
                "event": "PEER_CONNECTION_ATTEMPT",
                "from_node": self.name,
                "to_node": peer_name,
                "to_onion": peer_onion,
                "success": result.returncode == 0,
                "response": result.stdout[:500] if result.stdout else None,
                "connection_details": result.stderr[:1000] if result.stderr else None
            }
            
            self.connections_log.append(connection_result)
            
            return result.returncode == 0
            
        except Exception as e:
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "event": "CONNECTION_ERROR",
                "from_node": self.name,
                "to_node": peer_name,
                "error": str(e)
            }
            self.connections_log.append(error_log)
            return False
    
    def cleanup(self):
        """Clean up onion service"""
        if self.control_socket and self.onion_address:
            try:
                service_id = self.onion_address.replace('.onion', '')
                cleanup_cmd = f"DEL_ONION {service_id}\r\n"
                self.control_socket.send(cleanup_cmd.encode())
                self.control_socket.close()
            except:
                pass

def main():
    print("🔬 Q-NARWHALKNIGHT NODE-TO-NODE CONNECTION EVIDENCE TEST")
    print("=" * 70)
    print("Gathering solid evidence of actual peer-to-peer connections")
    print()
    
    # Create three validator nodes
    print("📍 Phase 1: Creating validator nodes with real .onion addresses...")
    
    node_alpha = TorDHTNode("validator-alpha", 8091)
    node_beta = TorDHTNode("validator-beta", 8092)
    node_gamma = TorDHTNode("validator-gamma", 8093)
    
    # Create onion services
    if not node_alpha.create_onion_service():
        print("❌ Failed to create onion service for alpha")
        return
    print(f"✅ Alpha: {node_alpha.onion_address}")
    
    if not node_beta.create_onion_service():
        print("❌ Failed to create onion service for beta")
        return
    print(f"✅ Beta: {node_beta.onion_address}")
    
    if not node_gamma.create_onion_service():
        print("❌ Failed to create onion service for gamma")
        return
    print(f"✅ Gamma: {node_gamma.onion_address}")
    
    # Start DHT servers
    print("\n📍 Phase 2: Starting DHT services on all nodes...")
    node_alpha.start_dht_server()
    print(f"✅ Alpha DHT server running on port {node_alpha.local_port}")
    
    node_beta.start_dht_server()
    print(f"✅ Beta DHT server running on port {node_beta.local_port}")
    
    node_gamma.start_dht_server()
    print(f"✅ Gamma DHT server running on port {node_gamma.local_port}")
    
    # Wait for services to stabilize
    print("\n⏳ Waiting for Tor network propagation (20 seconds)...")
    time.sleep(20)
    
    # Test connections
    print("\n📍 Phase 3: Testing peer-to-peer connections through Tor...")
    
    print("\n🔗 Alpha → Beta connection:")
    alpha_to_beta = node_alpha.connect_to_peer(node_beta.onion_address, "validator-beta")
    print(f"   Result: {'✅ SUCCESS' if alpha_to_beta else '❌ FAILED'}")
    
    print("\n🔗 Beta → Gamma connection:")
    beta_to_gamma = node_beta.connect_to_peer(node_gamma.onion_address, "validator-gamma")
    print(f"   Result: {'✅ SUCCESS' if beta_to_gamma else '❌ FAILED'}")
    
    print("\n🔗 Gamma → Alpha connection:")
    gamma_to_alpha = node_gamma.connect_to_peer(node_alpha.onion_address, "validator-alpha")
    print(f"   Result: {'✅ SUCCESS' if gamma_to_alpha else '❌ FAILED'}")
    
    # Give time for all messages to be processed
    time.sleep(2)
    
    # Generate evidence report
    print("\n" + "=" * 70)
    print("📊 EVIDENCE REPORT - NODE-TO-NODE CONNECTIONS")
    print("=" * 70)
    
    evidence = {
        "test_timestamp": datetime.now().isoformat(),
        "test_type": "Q-NarwhalKnight Tor DHT P2P Connection Test",
        "nodes": {
            "alpha": {
                "onion_address": node_alpha.onion_address,
                "port": node_alpha.local_port,
                "connections_made": len([l for l in node_alpha.connections_log if l["event"] == "PEER_CONNECTION_ATTEMPT"]),
                "messages_received": len(node_alpha.messages_received),
                "messages_sent": len(node_alpha.messages_sent)
            },
            "beta": {
                "onion_address": node_beta.onion_address,
                "port": node_beta.local_port,
                "connections_made": len([l for l in node_beta.connections_log if l["event"] == "PEER_CONNECTION_ATTEMPT"]),
                "messages_received": len(node_beta.messages_received),
                "messages_sent": len(node_beta.messages_sent)
            },
            "gamma": {
                "onion_address": node_gamma.onion_address,
                "port": node_gamma.local_port,
                "connections_made": len([l for l in node_gamma.connections_log if l["event"] == "PEER_CONNECTION_ATTEMPT"]),
                "messages_received": len(node_gamma.messages_received),
                "messages_sent": len(node_gamma.messages_sent)
            }
        },
        "connection_logs": {
            "alpha": node_alpha.connections_log,
            "beta": node_beta.connections_log,
            "gamma": node_gamma.connections_log
        }
    }
    
    # Save evidence to file
    with open('node_to_node_connection_evidence.json', 'w') as f:
        json.dump(evidence, f, indent=2)
    
    print("\n🎯 EVIDENCE SUMMARY:")
    print(f"✅ Onion addresses created: 3")
    print(f"   • Alpha: {node_alpha.onion_address}")
    print(f"   • Beta: {node_beta.onion_address}")
    print(f"   • Gamma: {node_gamma.onion_address}")
    
    total_connections = sum([
        alpha_to_beta,
        beta_to_gamma,
        gamma_to_alpha
    ])
    
    print(f"\n✅ Successful P2P connections: {total_connections}/3")
    
    total_messages = (
        len(node_alpha.messages_received) + 
        len(node_beta.messages_received) + 
        len(node_gamma.messages_received)
    )
    
    print(f"✅ Messages exchanged: {total_messages}")
    
    print(f"\n📁 Full evidence saved to: node_to_node_connection_evidence.json")
    
    # Show sample connection log
    print("\n📋 SAMPLE CONNECTION LOG (Alpha → Beta):")
    for log in node_alpha.connections_log:
        if log["event"] == "PEER_CONNECTION_ATTEMPT" and "beta" in str(log.get("to_node", "")):
            print(f"   Timestamp: {log['timestamp']}")
            print(f"   From: {log['from_node']}")
            print(f"   To: {log['to_node']}")
            print(f"   Success: {log['success']}")
            if log.get('response'):
                print(f"   Response received: {log['response'][:100]}...")
            break
    
    # Cleanup
    print("\n🧹 Cleaning up onion services...")
    node_alpha.cleanup()
    node_beta.cleanup()
    node_gamma.cleanup()
    
    print("\n✅ TEST COMPLETE - Evidence gathered successfully")

if __name__ == "__main__":
    main()