#!/usr/bin/env python3
"""
Simple Alpha Node Test Client for Server Beta Connection
Tests automatic DNS-Phantom discovery simulation
"""
import socket
import json
import time
import threading

def test_alpha_node_connection(node_id, beta_address="185.182.185.227", beta_port=8081):
    """Test Alpha node connection to Server Beta"""
    print(f"🚀 SERVER ALPHA {node_id}: Testing connection to Server Beta")
    
    try:
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        print(f"🔗 SERVER ALPHA {node_id}: Connecting to {beta_address}:{beta_port}")
        sock.connect((beta_address, beta_port))
        
        # Send JSON handshake
        handshake = {
            "node_id": f"alpha-test-{node_id}",
            "server": "alpha", 
            "message": f"Hello Server Beta from Alpha Test Node {node_id}",
            "discovery_method": "dns-phantom-simulation",
            "capabilities": ["consensus", "mempool", "state_sync"],
            "version": "0.1.0"
        }
        
        handshake_str = json.dumps(handshake) + "\n"
        sock.send(handshake_str.encode())
        
        print(f"📨 SERVER ALPHA {node_id}: Sent JSON handshake")
        
        # Receive Server Beta response
        response = sock.recv(1024).decode().strip()
        print(f"📬 SERVER ALPHA {node_id}: Server Beta response: {response}")
        
        # Parse response
        try:
            beta_response = json.loads(response)
            if beta_response.get("status") == "connected":
                print(f"🎉 SERVER ALPHA {node_id}: Successfully connected to Server Beta mesh!")
                print(f"📊 Total peers reported by Server Beta: {beta_response.get('total_peers', 'unknown')}")
                return True
            else:
                print(f"⚠️ SERVER ALPHA {node_id}: Connection status unclear: {response}")
                return False
        except json.JSONDecodeError:
            print(f"⚠️ SERVER ALPHA {node_id}: Invalid JSON response from Server Beta")
            return False
            
    except Exception as e:
        print(f"❌ SERVER ALPHA {node_id}: Connection failed: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def test_multiple_alpha_nodes():
    """Test multiple Alpha nodes connecting to Server Beta"""
    print("🎯 Starting multi-Alpha node connection test")
    
    # Test 3 Alpha nodes in parallel
    threads = []
    results = {}
    
    def node_test(node_id):
        results[node_id] = test_alpha_node_connection(node_id)
        time.sleep(1)  # Small delay between connections
    
    for i in range(1, 4):
        node_id = f"node-{i}"
        thread = threading.Thread(target=node_test, args=(node_id,))
        thread.start()
        threads.append(thread)
        time.sleep(2)  # Stagger connection attempts
    
    # Wait for all tests to complete
    for thread in threads:
        thread.join()
    
    # Report results
    print("\n🌟 CONNECTION TEST RESULTS:")
    successful_connections = 0
    for node_id, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  Alpha {node_id}: {status}")
        if success:
            successful_connections += 1
    
    print(f"\n📊 SUMMARY: {successful_connections}/{len(results)} Alpha nodes connected successfully")
    
    if successful_connections > 0:
        print("🎉 Cross-server Alpha-Beta mesh is OPERATIONAL!")
    else:
        print("❌ No successful connections - check Server Beta status")

if __name__ == "__main__":
    test_multiple_alpha_nodes()