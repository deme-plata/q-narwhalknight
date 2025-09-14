#!/usr/bin/env python3
"""
Simple Python Faucet Server for Q-NarwhalKnight
Temporary solution while the main API server has SSL linking issues
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
from urllib.parse import urlparse, parse_qs

class FaucetHandler(BaseHTTPRequestHandler):
    # In-memory balance store
    balances = {}
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/v1/node/status':
            self.send_node_status()
        else:
            self.send_404()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/v1/faucet':
            self.handle_faucet()
        elif self.path == '/api/v1/transactions/send':
            self.handle_transaction()
        else:
            self.send_404()
    
    def send_node_status(self):
        """Send node status response"""
        # Simulate getting the balance for the node
        node_balance = self.balances.get('default_wallet', 0)
        
        response = {
            "success": True,
            "data": {
                "node_id": "q1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7",
                "current_round": 12345,
                "current_height": 98765,
                "connected_peers": 8,
                "tx_pool_size": 42,
                "is_validator": True,
                "uptime_seconds": 86400,
                "uptime_formatted": "1 day",
                "network_health": "healthy",
                "consensus_status": "active",
                "last_block_time": int(time.time()),
                "tps_current": 1547.3,
                "tps_average": 1423.7,
                "balance": node_balance
            },
            "error": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        self.send_json_response(response)
    
    def handle_faucet(self):
        """Handle faucet requests"""
        wallet_id = 'default_wallet'
        
        # Check if already has tokens
        current_balance = self.balances.get(wallet_id, 0)
        
        if current_balance > 0:
            response = {
                "success": False,
                "data": None,
                "error": "Wallet already has tokens. Faucet can only be used once per wallet.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        else:
            # Give 1000 QNK
            faucet_amount = 100000000000  # 1000 QNK with 8 decimal places
            self.balances[wallet_id] = faucet_amount
            
            response = {
                "success": True,
                "data": {
                    "amount_qnk": 1000,
                    "amount_units": faucet_amount,
                    "wallet_id": wallet_id,
                    "message": "Faucet tokens distributed successfully!"
                },
                "error": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        
        self.send_json_response(response)
        print(f"Faucet request processed. New balance: {self.balances.get(wallet_id, 0)}")
    
    def handle_transaction(self):
        """Handle transaction sending"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            transaction_data = json.loads(post_data.decode('utf-8'))
            
            # Simulate transaction processing
            import hashlib
            import random
            
            tx_hash = hashlib.sha256(f"{time.time()}{random.random()}".encode()).hexdigest()
            
            # Update balances
            amount = transaction_data.get('amount', 0)
            sender_balance = self.balances.get('default_wallet', 0)
            
            if sender_balance >= amount:
                # Deduct from sender
                self.balances['default_wallet'] = sender_balance - amount
                print(f"Transaction processed. New sender balance: {self.balances['default_wallet']}")
            
            response = {
                "success": True,
                "data": {
                    "transaction_hash": tx_hash,
                    "stark_proof": {
                        "proof_system": "STARK-256",
                        "proving_time_ms": random.randint(45, 89),
                        "verification_time_ms": random.randint(2, 8),
                        "proof_size_bytes": random.randint(1024, 2048),
                        "security_level": 128,
                        "field": "BabyBear",
                        "hash_function": "Blake3",
                        "fri_queries": 27
                    }
                },
                "error": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            response = {
                "success": False,
                "data": None,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            self.send_json_response(response, status_code=500)
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response with CORS headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def send_404(self):
        """Send 404 response"""
        self.send_response(404)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b'404 - Not Found')
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

def run_faucet_server():
    """Run the faucet server"""
    server_address = ('0.0.0.0', 3030)
    httpd = HTTPServer(server_address, FaucetHandler)
    print(f"🚰 Q-NarwhalKnight Faucet Server running on http://localhost:3030")
    print("📡 Endpoints available:")
    print("  GET  /api/v1/node/status - Node status with balance")
    print("  POST /api/v1/faucet - Request test tokens") 
    print("  POST /api/v1/transactions/send - Send transactions")
    print("🌟 Ready to serve quantum wallets!")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Faucet server shutting down...")
        httpd.shutdown()

if __name__ == "__main__":
    run_faucet_server()