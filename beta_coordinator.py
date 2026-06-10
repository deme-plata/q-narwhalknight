
import socket
import json
import threading
import time

def handle_client(client_socket, address):
    """Handle incoming Alpha node connections"""
    try:
        data = client_socket.recv(1024).decode().strip()
        print(f"📨 Received from {address}: {data}")
        
        try:
            request = json.loads(data)
            node_id = request.get("node_id", "unknown")
            print(f"🎯 Alpha node {node_id} connected via auto-discovery!")
            
            # Send success response
            response = "🎯 Q-NarwhalKnight Server Beta P2P Bridge - Connection Successful!\n"
            response += json.dumps({
                "status": "connected",
                "server": "beta",
                "peer_id": f"alpha-peer-{hash(node_id) % 10000}",
                "total_peers": 1
            })
            
            client_socket.send(response.encode())
            
        except json.JSONDecodeError:
            response = "🎯 Q-NarwhalKnight Server Beta P2P Bridge - Connection Successful!"
            client_socket.send(response.encode())
            
    except Exception as e:
        print(f"❌ Error handling client {address}: {e}")
    finally:
        client_socket.close()

def start_beta_server():
    """Start Beta coordinator server"""
    print("🤝 Beta Coordinator starting P2P bridge...")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", 8081))
    server_socket.listen(10)
    
    print("✅ Beta P2P bridge listening on port 8081")
    
    while True:
        client_socket, address = server_socket.accept()
        print(f"🔗 New connection from {address}")
        
        client_thread = threading.Thread(
            target=handle_client,
            args=(client_socket, address)
        )
        client_thread.daemon = True
        client_thread.start()

if __name__ == "__main__":
    start_beta_server()
