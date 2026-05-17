/**
 * Minimal shim for Node.js 'crypto' module in React Native.
 *
 * crystals-kyber only uses crypto.webcrypto.getRandomValues which is
 * already polyfilled by react-native-get-random-values.
 * This shim wires it through so require('crypto') doesn't crash Metro.
 */

module.exports = {
  webcrypto: {
    getRandomValues: (buf) => {
      // globalThis.crypto is provided by react-native-get-random-values polyfill
      if (globalThis.crypto && globalThis.crypto.getRandomValues) {
        return globalThis.crypto.getRandomValues(buf);
      }
      // Fallback: fill with Math.random (insecure — only used if polyfill missing)
      for (let i = 0; i < buf.length; i++) {
        buf[i] = Math.floor(Math.random() * 256);
      }
      return buf;
    },
  },
  // Stubs for any other crypto methods that might be probed
  randomBytes: (size) => {
    const buf = new Uint8Array(size);
    if (globalThis.crypto && globalThis.crypto.getRandomValues) {
      globalThis.crypto.getRandomValues(buf);
    }
    return buf;
  },
};
