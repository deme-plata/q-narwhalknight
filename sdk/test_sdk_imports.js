#!/usr/bin/env node
/**
 * Test script to verify Q-NarwhalKnight PaaS SDK installation
 */

console.log("=".repeat(60));
console.log("Q-NarwhalKnight PaaS SDK - Installation Test");
console.log("=".repeat(60));

// Test JavaScript SDK
console.log("\n✅ Testing JavaScript SDK...");

try {
  const sdk = require('./javascript/index.js');

  console.log("   ✓ Successfully imported SDK");
  console.log(`   ✓ SDK Version: ${sdk.version}`);

  // Check all exports
  const expectedExports = [
    'QNarwhalKnightPaaSClient',
    'EthereumWallet',
    'PrivacyLevel',
    'SolanaWallet',
    'SolanaPaaSClient',
    'getClient'
  ];

  for (const exp of expectedExports) {
    if (sdk[exp]) {
      console.log(`   ✓ Successfully imported ${exp}`);
    } else {
      console.log(`   ✗ Missing export: ${exp}`);
      process.exit(1);
    }
  }

  // Test getClient utility
  const EthClient = sdk.getClient('ethereum');
  console.log("   ✓ getClient('ethereum') works");

  const SolClient = sdk.getClient('solana');
  console.log("   ✓ getClient('solana') works");

} catch (error) {
  console.log(`   ✗ Import failed: ${error.message}`);
  process.exit(1);
}

console.log("\n" + "=".repeat(60));
console.log("JAVASCRIPT SDK: ✅ ALL TESTS PASSED");
console.log("=".repeat(60));
console.log("\nYou can now use the SDK:");
console.log("  npm install /opt/orobit/shared/q-narwhalknight/sdk/javascript");
console.log("  const { QNarwhalKnightPaaSClient } = require('@q-narwhalknight/paas-sdk');");
console.log("\nDocumentation: https://quillon.xyz/docs");
