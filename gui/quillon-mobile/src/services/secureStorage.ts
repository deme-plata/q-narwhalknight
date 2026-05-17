/**
 * Secure storage wrapper around expo-secure-store.
 *
 * Used for persisting sensitive data like mnemonics and auth tokens
 * with hardware-backed encryption (Keychain on iOS, Keystore on Android).
 */

import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

const KEYS = {
  MNEMONIC: 'qnk_mnemonic',
  AUTH_TOKEN: 'qnk_auth_token',
  PIN_HASH: 'qnk_pin_hash',
  WALLET_ADDRESS: 'qnk_wallet_address',
  AUTO_LOCK_MINUTES: 'qnk_auto_lock_minutes',
  BIOMETRIC_ENABLED: 'qnk_biometric_enabled',
  LAST_ACTIVE: 'qnk_last_active',
  PIN_LOCKOUT: 'qnk_pin_lockout',
  // Post-quantum keys (hex-encoded, encrypted by Keystore)
  DILITHIUM5_PUBLIC_KEY: 'qnk_dilithium5_pk',
  DILITHIUM5_SECRET_KEY: 'qnk_dilithium5_sk',
  KYBER1024_PUBLIC_KEY: 'qnk_kyber1024_pk',
  KYBER1024_SECRET_KEY: 'qnk_kyber1024_sk',
  PQ_BACKEND_TYPE: 'qnk_pq_backend_type',
  // OAuth2 tokens (for server vault login)
  OAUTH_ACCESS_TOKEN: 'qnk_oauth_access',
  OAUTH_REFRESH_TOKEN: 'qnk_oauth_refresh',
  OAUTH_EXPIRES_AT: 'qnk_oauth_expires',
} as const;

// iOS: AFTER_FIRST_UNLOCK persists across reboots and is available when
// the device has been unlocked at least once since boot. This is more
// reliable than WHEN_UNLOCKED_THIS_DEVICE_ONLY which can lose data in
// Expo Go on some devices.
// Android: keychainAccessible is ignored — EncryptedSharedPreferences is
// always used and data persists across app restarts.
const SECURE_OPTIONS: SecureStore.SecureStoreOptions = Platform.select({
  ios: { keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK },
  default: {}, // Android ignores keychainAccessible
});

// ---------- Mnemonic ----------

export async function saveMnemonic(mnemonic: string): Promise<void> {
  await SecureStore.setItemAsync(KEYS.MNEMONIC, mnemonic, SECURE_OPTIONS);
}

export async function getMnemonic(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.MNEMONIC, SECURE_OPTIONS);
}

export async function deleteMnemonic(): Promise<void> {
  await SecureStore.deleteItemAsync(KEYS.MNEMONIC, SECURE_OPTIONS);
}

export async function hasMnemonic(): Promise<boolean> {
  const value = await getMnemonic();
  return value !== null && value.length > 0;
}

// ---------- Auth Token ----------

export async function saveAuthToken(token: string): Promise<void> {
  await SecureStore.setItemAsync(KEYS.AUTH_TOKEN, token, SECURE_OPTIONS);
}

export async function getAuthToken(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.AUTH_TOKEN, SECURE_OPTIONS);
}

export async function deleteAuthToken(): Promise<void> {
  await SecureStore.deleteItemAsync(KEYS.AUTH_TOKEN, SECURE_OPTIONS);
}

// ---------- PIN ----------

export async function savePinHash(hash: string): Promise<void> {
  await SecureStore.setItemAsync(KEYS.PIN_HASH, hash, SECURE_OPTIONS);
}

export async function getPinHash(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.PIN_HASH, SECURE_OPTIONS);
}

export async function hasPinSet(): Promise<boolean> {
  const hash = await getPinHash();
  return hash !== null && hash.length > 0;
}

// ---------- Wallet Address (cached, non-sensitive) ----------

export async function saveWalletAddress(address: string): Promise<void> {
  await SecureStore.setItemAsync(KEYS.WALLET_ADDRESS, address);
}

export async function getWalletAddress(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.WALLET_ADDRESS);
}

// ---------- Settings ----------

export async function saveBiometricEnabled(enabled: boolean): Promise<void> {
  await SecureStore.setItemAsync(KEYS.BIOMETRIC_ENABLED, enabled ? 'true' : 'false');
}

export async function isBiometricEnabled(): Promise<boolean> {
  const value = await SecureStore.getItemAsync(KEYS.BIOMETRIC_ENABLED);
  return value === 'true';
}

export async function saveAutoLockMinutes(minutes: number): Promise<void> {
  await SecureStore.setItemAsync(KEYS.AUTO_LOCK_MINUTES, minutes.toString());
}

export async function getAutoLockMinutes(): Promise<number> {
  const value = await SecureStore.getItemAsync(KEYS.AUTO_LOCK_MINUTES);
  return value ? parseInt(value, 10) : 5; // Default 5 minutes
}

export async function saveLastActiveTime(): Promise<void> {
  await SecureStore.setItemAsync(KEYS.LAST_ACTIVE, Date.now().toString());
}

export async function getLastActiveTime(): Promise<number> {
  const value = await SecureStore.getItemAsync(KEYS.LAST_ACTIVE);
  return value ? parseInt(value, 10) : 0;
}

// ---------- PIN Lockout (persists across app restarts) ----------

export interface PinLockoutState {
  failedAttempts: number;
  lockedUntil: number; // Unix timestamp ms, 0 = not locked
}

export async function savePinLockout(state: PinLockoutState): Promise<void> {
  await SecureStore.setItemAsync(KEYS.PIN_LOCKOUT, JSON.stringify(state));
}

export async function getPinLockout(): Promise<PinLockoutState> {
  const value = await SecureStore.getItemAsync(KEYS.PIN_LOCKOUT);
  if (!value) return { failedAttempts: 0, lockedUntil: 0 };
  try {
    return JSON.parse(value) as PinLockoutState;
  } catch {
    return { failedAttempts: 0, lockedUntil: 0 };
  }
}

export async function clearPinLockout(): Promise<void> {
  await SecureStore.deleteItemAsync(KEYS.PIN_LOCKOUT);
}

// ---------- Post-Quantum Keys ----------

export async function saveDilithium5Keys(
  publicKeyHex: string,
  secretKeyHex: string
): Promise<void> {
  await SecureStore.setItemAsync(KEYS.DILITHIUM5_PUBLIC_KEY, publicKeyHex, SECURE_OPTIONS);
  await SecureStore.setItemAsync(KEYS.DILITHIUM5_SECRET_KEY, secretKeyHex, SECURE_OPTIONS);
}

export async function getDilithium5PublicKey(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.DILITHIUM5_PUBLIC_KEY, SECURE_OPTIONS);
}

export async function getDilithium5SecretKey(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.DILITHIUM5_SECRET_KEY, SECURE_OPTIONS);
}

export async function saveKyber1024Keys(
  publicKeyHex: string,
  secretKeyHex: string
): Promise<void> {
  await SecureStore.setItemAsync(KEYS.KYBER1024_PUBLIC_KEY, publicKeyHex, SECURE_OPTIONS);
  await SecureStore.setItemAsync(KEYS.KYBER1024_SECRET_KEY, secretKeyHex, SECURE_OPTIONS);
}

export async function getKyber1024PublicKey(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.KYBER1024_PUBLIC_KEY, SECURE_OPTIONS);
}

export async function savePQBackendType(backendType: string): Promise<void> {
  await SecureStore.setItemAsync(KEYS.PQ_BACKEND_TYPE, backendType);
}

export async function getPQBackendType(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.PQ_BACKEND_TYPE);
}

export async function hasPQKeys(): Promise<boolean> {
  const pk = await getDilithium5PublicKey();
  return pk !== null && pk.length > 0;
}

// ---------- OAuth2 Tokens ----------

export async function saveOAuthTokens(
  accessToken: string,
  refreshToken: string,
  expiresAt: number,
): Promise<void> {
  await SecureStore.setItemAsync(KEYS.OAUTH_ACCESS_TOKEN, accessToken, SECURE_OPTIONS);
  await SecureStore.setItemAsync(KEYS.OAUTH_REFRESH_TOKEN, refreshToken, SECURE_OPTIONS);
  await SecureStore.setItemAsync(KEYS.OAUTH_EXPIRES_AT, expiresAt.toString());
}

export async function getOAuthAccessToken(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.OAUTH_ACCESS_TOKEN, SECURE_OPTIONS);
}

export async function getOAuthRefreshToken(): Promise<string | null> {
  return SecureStore.getItemAsync(KEYS.OAUTH_REFRESH_TOKEN, SECURE_OPTIONS);
}

export async function hasOAuthSession(): Promise<boolean> {
  const token = await getOAuthAccessToken();
  return token !== null && token.length > 0;
}

export async function clearOAuthTokens(): Promise<void> {
  await SecureStore.deleteItemAsync(KEYS.OAUTH_ACCESS_TOKEN).catch(() => {});
  await SecureStore.deleteItemAsync(KEYS.OAUTH_REFRESH_TOKEN).catch(() => {});
  await SecureStore.deleteItemAsync(KEYS.OAUTH_EXPIRES_AT).catch(() => {});
}

// ---------- Full Wipe ----------

export async function wipeAllSecureData(): Promise<void> {
  const allKeys = Object.values(KEYS);
  await Promise.all(allKeys.map((key) => SecureStore.deleteItemAsync(key).catch(() => {})));
}

/**
 * Check if secure storage is available on this device.
 */
export async function isSecureStorageAvailable(): Promise<boolean> {
  if (Platform.OS === 'web') return false;
  try {
    const testKey = '__qnk_test__';
    await SecureStore.setItemAsync(testKey, 'test');
    await SecureStore.deleteItemAsync(testKey);
    return true;
  } catch {
    return false;
  }
}
