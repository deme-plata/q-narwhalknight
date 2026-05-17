/**
 * Q-NarwhalKnight Privacy-as-a-Service SDK
 * Production-ready Ethereum and Solana integration
 *
 * @module @q-narwhalknight/paas-sdk
 * @version 4.0.0
 */

// Export Ethereum SDK
const ethereumSdk = require('./q_paas_ethereum_production.js');

// Export Solana SDK
const solanaSdk = require('./q_paas_solana_production.js');

// Re-export all classes and enums
module.exports = {
  // Ethereum exports
  QNarwhalKnightPaaSClient: ethereumSdk.QNarwhalKnightPaaSClient || class QNarwhalKnightPaaSClient {},
  EthereumWallet: ethereumSdk.EthereumWallet || class EthereumWallet {},
  PrivacyLevel: ethereumSdk.PrivacyLevel || {
    STANDARD: 'standard',
    ENHANCED: 'enhanced',
    MAXIMUM: 'maximum'
  },

  // Solana exports
  SolanaWallet: solanaSdk.SolanaWallet || class SolanaWallet {},
  SolanaPaaSClient: solanaSdk.SolanaPaaSClient || class SolanaPaaSClient {},

  // Version info
  version: '4.0.0',

  // Utility to detect which blockchain SDK to use
  getClient: (chain) => {
    switch(chain.toLowerCase()) {
      case 'ethereum':
      case 'eth':
        return ethereumSdk.QNarwhalKnightPaaSClient;
      case 'solana':
      case 'sol':
        return solanaSdk.SolanaPaaSClient;
      default:
        throw new Error(`Unsupported chain: ${chain}`);
    }
  }
};
