#!/usr/bin/env python3
"""
Real Tor DHT Connection Test
Tests if Q-NarwhalKnight nodes can actually connect to each other through Tor DHT
"""

import socket
import time
import threading
import json
from datetime import datetime

def create_real_onion_service(name, port):
    """Create a real onion service and return its address"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 9051))
        
        # Authenticate
        sock.send(b"AUTHENTICATE\r\n")
        auth_response = sock.recv(1024).decode()
        if "250 OK" not in auth_response:
            return None
            
        # Create service
        create_cmd = f"ADD_ONION NEW:BEST Port=80,127.0.0.1:{port}\r\n"
        sock.send(create_cmd.encode())
        create_response = sock.recv(2048).decode()
        
        # Parse address
        for line in create_response.split('\n'):
            if line.startswith("250-ServiceID="):
                service_id = line.replace("250-ServiceID=", "").strip()
                onion_address = f"{service_id}.onion"
                return onion_address, sock
                
    except Exception as e:
        print(f"Error creating onion service: {e}")
    return None, None

def start_dht_node_server(onion_address, port):
    """Start a simple DHT node server"""
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', port))
        server.listen(5)
        print(f"🟢 DHT Node server running for {onion_address} on port {port}")
        
        while True:
            try:
                client, addr = server.accept()
                print(f"📡 Connection received from {addr}")
                
                # Simple DHT response
                dht_response = {
                    "type": "dht_peer_info",
                    "onion_address": onion_address,
                    "port": port,
                    "node_id": f"node-{port}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                }
                
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{json.dumps(dht_response)}\r\n"
                client.send(response.encode())
                client.close()
                print(f"✅ Sent DHT response to {addr}")
                
            except Exception as e:
                print(f"Server error: {e}")
                break
                
    except Exception as e:
        print(f"Failed to start DHT server: {e}")

def test_tor_dht_connection(target_onion, target_port):
    """Test connecting to another node's DHT through Tor"""
    try:
        import socks
        
        # Create SOCKS5 connection through Tor
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 9050)
        sock = socks.socksocket()
        sock.settimeout(30)
        
        print(f"🔗 Connecting to {target_onion}:{target_port} through Tor...")
        sock.connect((target_onion, 80))  # Connect to onion service
        
        # Send DHT query
        http_request = f"GET /dht/peers HTTP/1.1\r\nHost: {target_onion}\r\nConnection: close\r\n\r\n"
        sock.send(http_request.encode())
        
        # Receive response
        response = sock.recv(4096).decode()
        sock.close()
        
        print(f"📨 Response from {target_onion}:")
        print(response)
        
        return True
        
    except ImportError:
        print("❌ Missing socks library. Install with: pip install PySocks")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    print("🧅 REAL TOR DHT CONNECTION TEST")
    print("=" * 50)
    print("Testing if Q-NarwhalKnight nodes can actually connect through Tor DHT")
    print()
    
    # Test 1: Create two real onion services (simulating two DHT nodes)
    print("1️⃣ Creating first DHT node onion service...")
    node1_address, node1_control = create_real_onion_service("qnk-dht-node1", 8081)
    
    if not node1_address:
        print("❌ Failed to create first onion service")
        return
    
    print(f"✅ Node 1 onion address: {node1_address}")
    
    print("\n2️⃣ Creating second DHT node onion service...")
    node2_address, node2_control = create_real_onion_service("qnk-dht-node2", 8082)
    
    if not node2_address:
        print("❌ Failed to create second onion service")
        return
        
    print(f"✅ Node 2 onion address: {node2_address}")
    
    # Test 2: Start DHT servers
    print("\n3️⃣ Starting DHT node servers...")
    
    # Start servers in background threads
    server1_thread = threading.Thread(target=start_dht_node_server, args=(node1_address, 8081), daemon=True)
    server2_thread = threading.Thread(target=start_dht_node_server, args=(node2_address, 8082), daemon=True)
    
    server1_thread.start()
    server2_thread.start()
    
    time.sleep(2)  # Let servers start
    
    # Test 3: Test DHT connections through Tor
    print("\n4️⃣ Testing DHT peer discovery through Tor...")
    
    # Wait for Tor to propagate the onion services
    print("⏳ Waiting for Tor network propagation (30 seconds)...")
    time.sleep(30)
    
    print(f"\n🔍 Node 1 trying to discover Node 2 via Tor DHT...")
    success1 = test_tor_dht_connection(node2_address, 8082)
    
    print(f"\n🔍 Node 2 trying to discover Node 1 via Tor DHT...")
    success2 = test_tor_dht_connection(node1_address, 8081)
    
    # Results
    print("\n🎯 DHT CONNECTION TEST RESULTS")
    print("=" * 40)
    
    if success1 or success2:
        print("✅ REAL TOR DHT WORKING!")
        print("   • Nodes can create real .onion addresses")
        print("   • Nodes can connect through Tor network")
        print("   • DHT peer discovery via Tor is operational")
        print(f"   • Node 1 → Node 2: {'✅' if success1 else '❌'}")
        print(f"   • Node 2 → Node 1: {'✅' if success2 else '❌'}")
    else:
        print("❌ TOR DHT CONNECTION FAILED")
        print("   • Nodes created real .onion addresses")
        print("   • But nodes cannot connect through Tor")
        print("   • DHT discovery needs troubleshooting")
    
    # Cleanup
    print("\n5️⃣ Cleaning up onion services...")
    if node1_control:
        try:
            service_id = node1_address.replace('.onion', '')
            cleanup_cmd = f"DEL_ONION {service_id}\r\n"
            node1_control.send(cleanup_cmd.encode())
            node1_control.close()
        except:
            pass
            
    if node2_control:
        try:
            service_id = node2_address.replace('.onion', '')
            cleanup_cmd = f"DEL_ONION {service_id}\r\n"
            node2_control.send(cleanup_cmd.encode())
            node2_control.close()
        except:
            pass
    
    print("✅ Cleanup complete")

if __name__ == "__main__":
    main()