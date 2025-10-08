#!/usr/bin/env python3
"""
Manual test script for faucet protection mechanisms
Tests the enhanced faucet system with daily limits, IP rate limiting, and abuse detection
"""

import requests
import json
import time
import threading
from datetime import datetime

API_URL = "http://127.0.0.1:8080"
FAUCET_ENDPOINT = f"{API_URL}/api/v1/faucet"

def test_faucet_request(address, amount=100):
    """Make a single faucet request"""
    payload = {
        "address": address,
        "amount": amount
    }
    
    try:
        response = requests.post(FAUCET_ENDPOINT, json=payload, timeout=10)
        return {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "timestamp": datetime.now().isoformat()
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 0,
            "response": str(e),
            "timestamp": datetime.now().isoformat()
        }

def test_daily_limit():
    """Test that faucet stops after 1000 QNK daily limit"""
    print("🧪 Testing Daily Limit (1000 QNK)")
    print("=" * 50)
    
    # Try to request exactly 1000 QNK
    result = test_faucet_request("test_daily_limit_address", 1000)
    print(f"Request 1000 QNK: {result}")
    
    # Try to request 1 more QNK (should fail)
    result2 = test_faucet_request("test_daily_limit_address2", 1)
    print(f"Request 1 more QNK: {result2}")
    
    print()

def test_ip_rate_limiting():
    """Test IP-based rate limiting (10 requests per hour)"""
    print("🧪 Testing IP Rate Limiting (10 requests/hour)")
    print("=" * 50)
    
    successful_requests = 0
    
    for i in range(12):  # Try 12 requests to exceed limit of 10
        result = test_faucet_request(f"test_ip_rate_limit_{i}", 10)
        print(f"Request {i+1}: Status {result['status_code']}")
        
        if result['status_code'] == 200:
            successful_requests += 1
        
        time.sleep(0.1)  # Small delay between requests
    
    print(f"Successful requests: {successful_requests}/12 (should be ≤10)")
    print()

def test_address_cooldown():
    """Test address-based cooldown (24 hours per address)"""
    print("🧪 Testing Address Cooldown (24 hours per address)")
    print("=" * 50)
    
    address = "test_address_cooldown"
    
    # First request should succeed
    result1 = test_faucet_request(address, 50)
    print(f"First request: {result1}")
    
    # Second request to same address should fail
    result2 = test_faucet_request(address, 50)
    print(f"Second request (same address): {result2}")
    
    print()

def test_abuse_detection():
    """Test scripted abuse detection"""
    print("🧪 Testing Scripted Abuse Detection")
    print("=" * 50)
    
    def make_rapid_requests():
        """Simulate rapid scripted requests"""
        for i in range(20):
            # Sequential addresses (pattern detection)
            address = f"1A2B3C4D5E6F{i:04d}"
            result = test_faucet_request(address, 25)
            print(f"Rapid request {i+1}: Status {result['status_code']} - {address}")
            time.sleep(0.05)  # Very rapid requests
    
    make_rapid_requests()
    print()

def test_normal_usage():
    """Test normal legitimate usage"""
    print("🧪 Testing Normal Legitimate Usage")
    print("=" * 50)
    
    # Different addresses, reasonable amounts, normal timing
    legitimate_addresses = [
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
        "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy", 
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    ]
    
    for address in legitimate_addresses:
        result = test_faucet_request(address, 100)
        print(f"Legitimate request to {address[:20]}...: {result}")
        time.sleep(2)  # Normal human-like delay
    
    print()

def test_server_health():
    """Test if server is responding"""
    print("🧪 Testing Server Health")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_URL}/api/v1/status", timeout=5)
        print(f"Server health check: Status {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
    except requests.exceptions.RequestException as e:
        print(f"Server health check failed: {e}")
        return False
    
    print()
    return False

def main():
    """Run all faucet protection tests"""
    print("🚀 Q-NarwhalKnight Faucet Protection Test Suite")
    print("=" * 60)
    print(f"Testing API at: {API_URL}")
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Check if server is running
    if not test_server_health():
        print("❌ Server is not responding. Please start q-api-server first.")
        return
    
    # Run all tests
    test_daily_limit()
    test_ip_rate_limiting()
    test_address_cooldown()
    test_abuse_detection()
    test_normal_usage()
    
    print("✅ All faucet protection tests completed!")
    print("📊 Check the server logs for detailed protection system behavior")

if __name__ == "__main__":
    main()