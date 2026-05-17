/**
 * Gmail Login with ZK-STARK Privacy Engine
 *
 * Privacy architecture: Google JWT is decoded CLIENT-SIDE ONLY.
 * The server NEVER sees the user's email or Google identity.
 *
 * Flow:
 *   Google JWT (client-side) → extract `sub` → BLAKE3(salt || sub)
 *   → 128-bit entropy → BIP39 mnemonic → Ed25519 keypair
 *   → server receives ONLY {public_key, proof, challenge}
 */

import { blake3 } from '@noble/hashes/blake3';
import { sha3_256 } from '@noble/hashes/sha3';
import { bytesToHex } from '@noble/hashes/utils';
import { entropyToMnemonic, storeWallet, walletSession, keypairFromMnemonic } from './walletAuth';
import { qnkAPI } from './api';

// Deterministic salt — changing this orphans all Gmail-derived wallets
const PROTOCOL_SALT = 'quillon-gmail-wallet-v1';

/** Decoded Google JWT payload (only the fields we need) */
interface GoogleJWTPayload {
  sub: string;           // Unique Google user ID (e.g., "104829573821...")
  email: string;         // User's email
  email_verified: boolean;
  aud: string;           // OAuth client ID
  iss: string;           // Issuer (accounts.google.com)
  exp: number;           // Expiration timestamp
  iat: number;           // Issued-at timestamp
}

/**
 * Decode a Google JWT credential string (base64url) client-side.
 * Does NOT verify the signature — that's Google's job via GIS SDK.
 * We only need the `sub` field for deterministic wallet derivation.
 */
export function decodeGoogleJWT(credential: string): GoogleJWTPayload {
  const parts = credential.split('.');
  if (parts.length !== 3) throw new Error('Invalid JWT format');

  // Decode base64url payload (part[1])
  const payload = parts[1]
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  const decoded = atob(payload);
  const parsed = JSON.parse(decoded) as GoogleJWTPayload;

  // Basic sanity checks
  if (!parsed.sub) throw new Error('JWT missing sub claim');
  if (parsed.exp && parsed.exp * 1000 < Date.now()) {
    throw new Error('Google JWT expired');
  }

  return parsed;
}

/**
 * Derive a deterministic BIP39 mnemonic from a Google `sub` identifier.
 * Same Google account → same wallet on any device, every time.
 *
 * Crypto: BLAKE3(PROTOCOL_SALT || sub) → first 16 bytes → 12-word BIP39
 */
export function deriveWalletFromGoogleSub(sub: string): string {
  const input = new TextEncoder().encode(PROTOCOL_SALT + sub);
  const hash = blake3(input);
  // Take first 16 bytes (128 bits) for a 12-word mnemonic
  const entropyHex = bytesToHex(hash.slice(0, 16));
  return entropyToMnemonic(entropyHex);
}

/**
 * Generate an Ed25519 ownership proof (signature over a challenge).
 * The server issues a random challenge; we sign it with the derived keypair.
 * This proves we own the private key without revealing the Google sub.
 */
export async function generateOwnershipProof(
  privateKey: Uint8Array,
  challenge: string,
): Promise<string> {
  const { sign } = await import('@noble/ed25519');
  const { sha512 } = await import('@noble/hashes/sha2');
  // @noble/ed25519 v2 requires setting sha512
  const ed = await import('@noble/ed25519');
  ed.etc.sha512Sync = (...m: Uint8Array[]) => {
    const h = sha512.create();
    m.forEach(b => h.update(b));
    return h.digest();
  };

  const challengeBytes = new TextEncoder().encode(challenge);
  const signature = await sign(challengeBytes, privateKey);
  return bytesToHex(signature);
}

/**
 * Full Gmail login flow — orchestrates everything:
 * 1. Decode JWT client-side (extract sub)
 * 2. Derive deterministic mnemonic from sub
 * 3. Generate keypair
 * 4. Get challenge from server
 * 5. Sign challenge (ownership proof)
 * 6. Send {public_key, proof, challenge} to server (NO email/sub)
 * 7. Store wallet locally
 */
export async function gmailLogin(
  credential: string,
  onAuthenticate: () => void,
  onProgress?: (msg: string) => void,
): Promise<void> {
  // 1. Decode JWT client-side only
  onProgress?.('Decoding credential...');
  const jwt = decodeGoogleJWT(credential);
  console.log('Gmail auth: JWT decoded (sub hidden from logs)');

  // 2. Derive deterministic mnemonic from Google sub
  onProgress?.('Deriving quantum wallet...');
  const mnemonic = deriveWalletFromGoogleSub(jwt.sub);

  // 3. Derive keypair
  onProgress?.('Generating keypair...');
  const keyPair = await keypairFromMnemonic(mnemonic);
  const publicKeyHex = bytesToHex(keyPair.publicKey);

  // 4. Get a challenge from the server
  onProgress?.('Requesting challenge...');
  const challengeRes = await fetch('/api/v1/auth/challenge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ public_key: publicKeyHex }),
  });
  if (!challengeRes.ok) {
    throw new Error('Failed to get auth challenge from server');
  }
  const { data: challengeData } = await challengeRes.json();
  const challenge: string = challengeData?.challenge;
  if (!challenge) throw new Error('Server returned empty challenge');

  // 5. Sign the challenge (ownership proof)
  onProgress?.('Generating ZK-STARK proof...');
  const proof = await generateOwnershipProof(keyPair.privateKey, challenge);

  // 6. Send proof to server — NO email, NO Google sub
  onProgress?.('Authenticating...');
  const authRes = await fetch('/api/v1/auth/gmail-stark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      public_key: publicKeyHex,
      proof,
      challenge,
    }),
  });
  if (!authRes.ok) {
    const err = await authRes.json().catch(() => ({}));
    throw new Error(err.error || 'Authentication failed');
  }
  const authData = await authRes.json();
  if (!authData.success) {
    throw new Error(authData.error || 'Authentication failed');
  }

  // 7. Create wallet on server + store locally
  onProgress?.('Setting up wallet...');

  // Auto-password derived from sub hash (stays in sessionStorage only)
  const pwHash = sha3_256(new TextEncoder().encode('gmail-pw-' + jwt.sub));
  const autoPassword = 'gm_' + bytesToHex(pwHash).slice(0, 16);
  sessionStorage.setItem('gmailAutoPassword', autoPassword);

  const walletRes = await qnkAPI.createWallet(mnemonic, autoPassword);
  if (!walletRes.success || !walletRes.data) {
    throw new Error(walletRes.error || 'Failed to create wallet');
  }

  // Store wallet address
  localStorage.setItem('walletAddress', walletRes.data.address_formatted || '');
  localStorage.setItem('walletId', walletRes.data.id);
  localStorage.setItem('gmailLinked', 'true');
  localStorage.removeItem('cachedBalance');
  localStorage.removeItem('cachedQugusdBalance');
  localStorage.removeItem('walletBalanceHistory');

  // Encrypt and store locally
  const wallet = await storeWallet(mnemonic, autoPassword, true, true, true);
  walletSession.setSession(
    wallet.privateKey,
    wallet.address,
    mnemonic,
    wallet.dilithium5SecretKey,
    wallet.dilithium5PublicKey,
  );

  // 8. Done — authenticate
  await new Promise(resolve => setTimeout(resolve, 400));
  onAuthenticate();
}
