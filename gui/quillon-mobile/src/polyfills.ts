/**
 * Polyfills for React Native (Hermes engine).
 *
 * MUST be imported before ANY crypto library.
 * Hermes lacks Web Crypto API — we polyfill the pieces noble-hashes/ed25519 need.
 */

// 1. crypto.getRandomValues — needed by @scure/bip39, PQ simulated backend
import 'react-native-get-random-values';

// 2. @noble/ed25519 SHA-512 setup — Hermes lacks crypto.subtle
//    We provide SHA-512 from @noble/hashes instead of WebCrypto.
import { sha512 } from '@noble/hashes/sha2.js';
import { etc, hashes } from '@noble/ed25519';

// Sync SHA-512 (used by getPublicKey, sign, verify)
hashes.sha512 = (message: Uint8Array) => sha512(etc.concatBytes(message));
// Async SHA-512 (used by getPublicKeyAsync, signAsync, verifyAsync)
hashes.sha512Async = async (message: Uint8Array) => sha512(etc.concatBytes(message));
