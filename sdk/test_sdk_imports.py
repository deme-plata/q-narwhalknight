#!/usr/bin/env python3
"""
Test script to verify Q-NarwhalKnight PaaS SDK installation
"""

print("=" * 60)
print("Q-NarwhalKnight PaaS SDK - Installation Test")
print("=" * 60)

# Test Python SDK
print("\n✅ Testing Python SDK...")
try:
    from q_paas import QNarwhalKnightPaaSClient, BitcoinWallet, PrivacyLevel
    print("   ✓ Successfully imported QNarwhalKnightPaaSClient")
    print("   ✓ Successfully imported BitcoinWallet")
    print("   ✓ Successfully imported PrivacyLevel")

    # Test instantiation (without API call)
    client = QNarwhalKnightPaaSClient(
        api_key="test_key",
        base_url="https://quillon.xyz"
    )
    print("   ✓ Client instantiation successful")
    print(f"   ✓ Client base URL: {client.base_url}")

except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("PYTHON SDK: ✅ ALL TESTS PASSED")
print("=" * 60)
print("\nYou can now use the SDK:")
print("  pip install -e /opt/orobit/shared/q-narwhalknight/sdk/python")
print("  from q_paas import QNarwhalKnightPaaSClient")
print("\nDocumentation: https://quillon.xyz/docs")
