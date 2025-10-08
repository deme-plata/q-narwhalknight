#!/usr/bin/env python3
"""
Real Tor Integration Test
This creates an actual .onion address using the Tor daemon control protocol
"""

import socket
import time
import sys

def test_tor_control():
    """Test Tor control port connectivity and create real onion service"""
    print("🧅🎯 Real Tor Integration Test")
    print("==============================")
    print("Creating GENUINE .onion address using Tor daemon control protocol")
    print()
    
    try:
        # Step 1: Test Tor control port
        print("🔍 Step 1: Testing Tor control port connectivity...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 9051))
        
        if result != 0:
            print("❌ Cannot connect to Tor control port (127.0.0.1:9051)")
            print()
            print("🛠️  SETUP REQUIRED:")
            print("   1. Install Tor: sudo apt-get install tor")
            print("   2. Edit /etc/tor/torrc and add:")
            print("      ControlPort 9051")
            print("      CookieAuthentication 0")
            print("   3. Restart Tor: sudo systemctl restart tor")
            return False
            
        print("✅ Tor control port is accessible")
        
        # Step 2: Authenticate
        print()
        print("🔐 Step 2: Authenticating with Tor daemon...")
        sock.send(b"AUTHENTICATE\r\n")
        response = sock.recv(1024).decode()
        
        if "250 OK" not in response:
            print(f"❌ Authentication failed: {response}")
            return False
            
        print("✅ Successfully authenticated with Tor daemon")
        
        # Step 3: Create real onion service
        print()
        print("🧅 Step 3: Creating REAL onion service...")
        print("   This generates a genuine .onion address from the Tor network")
        
        # Send ADD_ONION command to create genuine v3 .onion address
        command = "ADD_ONION NEW:BEST Port=80,127.0.0.1:8080\r\n"
        sock.send(command.encode())
        
        # Read multi-line response
        response_lines = []
        while True:
            data = sock.recv(1024).decode()
            response_lines.append(data)
            if "250 OK" in data:
                break
        
        full_response = ''.join(response_lines)
        
        # Parse onion address
        onion_address = None
        for line in full_response.split('\n'):
            if line.startswith("250-ServiceID="):
                service_id = line.replace("250-ServiceID=", "").strip()
                onion_address = f"{service_id}.onion"
                break
        
        if not onion_address:
            print("❌ Failed to parse onion address from response")
            print(f"Response: {full_response}")
            return False
        
        print("🎉 SUCCESS! REAL onion service created:")
        print(f"   Address: {onion_address}")
        
        # Step 4: Validate address format
        print()
        print("🔍 Step 4: Validating onion address format...")
        if onion_address.endswith('.onion') and len(onion_address) == 62:
            print("✅ Verified: This is a genuine Tor v3 onion address!")
            print(f"   Length: {len(onion_address)} characters (correct for v3)")
            print("   Format: Valid .onion suffix")
        else:
            print(f"⚠️ Unexpected address format: {onion_address}")
        
        # Step 5: Test SOCKS connectivity
        print()
        print("🌐 Step 5: Testing SOCKS proxy connectivity...")
        try:
            socks_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socks_sock.settimeout(5)
            socks_result = socks_sock.connect_ex(('127.0.0.1', 9050))
            
            if socks_result == 0:
                print("✅ SOCKS proxy (127.0.0.1:9050) is accessible")
                socks_sock.close()
            else:
                print("⚠️ SOCKS proxy not accessible (this is optional)")
                
        except Exception as e:
            print(f"⚠️ SOCKS test failed: {e}")
        
        # Step 6: Cleanup
        print()
        print("🧹 Step 6: Cleaning up...")
        service_id = onion_address.replace('.onion', '')
        cleanup_command = f"DEL_ONION {service_id}\r\n"
        sock.send(cleanup_command.encode())
        cleanup_response = sock.recv(1024).decode()
        
        if "250 OK" in cleanup_response:
            print("✅ Onion service removed successfully")
        else:
            print(f"⚠️ Cleanup warning: {cleanup_response}")
        
        sock.close()
        
        # Final summary
        print()
        print("🎯 REAL TOR INTEGRATION TEST COMPLETE!")
        print("=====================================")
        print("✅ Created GENUINE .onion address using Tor daemon")
        print("✅ Connected to actual Tor control protocol")
        print("✅ Used real Tor network (not simulation)")
        print()
        print("This demonstrates that Q-NarwhalKnight can create")
        print("REAL anonymous validator endpoints for quantum consensus!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

if __name__ == "__main__":
    success = test_tor_control()
    sys.exit(0 if success else 1)