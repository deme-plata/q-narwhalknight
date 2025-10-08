
import socket
import json
import time
import threading
import sys

def simulate_dns_discovery():
    """Simulate DNS-Phantom discovery process"""
    domains = [
        "beta-coordinator.qnarwhal-mesh",
        "dns-phantom-hub.qnarwhal-mesh", 
        "discovery.q-narwhal.local"
    ]
    
    for domain in domains:
        try:
            # This would trigger DNS queries that DNS-Phantom detects
            socket.gethostbyname_ex(domain)
            print(f"🔍 DNS query sent for {domain}")
        except:
            pass
        time.sleep(1)

def connect_to_beta():
    """Try to connect to Beta coordinator after discovery"""
    beta_addresses = [
        ("beta-coordinator", 8081),
        ("172.20.0.20", 8081),  # Docker internal IP
        ("localhost", 8181)     # Host mapped port
    ]
    
    for addr, port in beta_addresses:
        try:
            print(f"🚀 Alpha trying to connect to {addr}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((addr, port))
            
            # Send handshake
            handshake = {
                "node_id": f"alpha-container-{sys.argv[1] if len(sys.argv) > 1 else '1'}",
                "server": "alpha",
                "message": "Auto-discovery container test",
                "discovery_method": "dns_phantom_simulation"
            }
            
            sock.send((json.dumps(handshake) + "\n").encode())
            response = sock.recv(1024).decode()
            print(f"📬 Response from Beta: {response}")
            
            if "Connection Successful" in response:
                print(f"✅ Successfully connected to Beta at {addr}:{port}")
                sock.close()
                return True
            
            sock.close()
        except Exception as e:
            print(f"⚠️ Failed to connect to {addr}:{port}: {e}")
    
    return False

if __name__ == "__main__":
    node_id = sys.argv[1] if len(sys.argv) > 1 else "alpha-1"
    print(f"🚀 Alpha Node {node_id} starting auto-discovery...")
    
    # Simulate DNS discovery
    simulate_dns_discovery()
    
    # Try to connect to Beta
    time.sleep(2)
    if connect_to_beta():
        print(f"🎉 Alpha Node {node_id}: Auto-discovery successful!")
    else:
        print(f"❌ Alpha Node {node_id}: Auto-discovery failed")
        
    # Keep container running for observation
    time.sleep(30)
