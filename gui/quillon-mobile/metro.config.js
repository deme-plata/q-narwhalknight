const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const config = getDefaultConfig(__dirname);

const emptyModule = path.resolve(__dirname, 'empty-module.js');

// Shim Node.js built-ins that crystals-kyber / dilithium-crystals require.
// These modules are unreachable at runtime (pqCrypto.ts catches the import
// failure and falls back to the simulated backend), but Metro resolves them
// statically so they must point to something.
config.resolver.extraNodeModules = {
  ...config.resolver.extraNodeModules,
  crypto: path.resolve(__dirname, 'crypto-shim.js'),
  fs: emptyModule,
  path: emptyModule,
  stream: emptyModule,
  buffer: emptyModule,
};

module.exports = config;
