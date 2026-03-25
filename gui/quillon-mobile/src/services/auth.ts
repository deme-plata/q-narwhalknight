/**
 * Authentication service handling biometric auth, PIN fallback, and auto-lock.
 */

import * as LocalAuthentication from 'expo-local-authentication';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex } from '@noble/hashes/utils.js';
import {
  savePinHash,
  getPinHash,
  hasPinSet,
  isBiometricEnabled,
  saveBiometricEnabled,
  getAutoLockMinutes,
  saveLastActiveTime,
  getLastActiveTime,
  savePinLockout,
  getPinLockout,
  clearPinLockout,
} from './secureStorage';

// ---------- Biometric ----------

export interface BiometricCapability {
  available: boolean;
  biometricTypes: LocalAuthentication.AuthenticationType[];
  enrolled: boolean;
}

/**
 * Check device biometric capabilities.
 */
export async function checkBiometrics(): Promise<BiometricCapability> {
  const available = await LocalAuthentication.hasHardwareAsync();
  const enrolled = await LocalAuthentication.isEnrolledAsync();
  const types = await LocalAuthentication.supportedAuthenticationTypesAsync();

  return { available, enrolled, biometricTypes: types };
}

/**
 * Authenticate with biometrics (Face ID / fingerprint).
 */
export async function authenticateWithBiometrics(
  promptMessage: string = 'Unlock Quillon Wallet'
): Promise<boolean> {
  const isEnabled = await isBiometricEnabled();
  if (!isEnabled) return false;

  const result = await LocalAuthentication.authenticateAsync({
    promptMessage,
    cancelLabel: 'Use PIN',
    disableDeviceFallback: true,
    fallbackLabel: 'Use PIN',
  });

  return result.success;
}

/**
 * Enable or disable biometric authentication.
 */
export async function setBiometricAuth(enabled: boolean): Promise<void> {
  if (enabled) {
    const { available, enrolled } = await checkBiometrics();
    if (!available || !enrolled) {
      throw new Error('Biometric authentication is not available on this device');
    }
  }
  await saveBiometricEnabled(enabled);
}

// ---------- PIN ----------

/**
 * Hash a PIN for secure storage.
 */
function hashPin(pin: string): string {
  const encoded = new TextEncoder().encode(`qnk:pin:${pin}`);
  return bytesToHex(sha3_256(encoded));
}

/**
 * Set a new PIN.
 */
export async function setPin(pin: string): Promise<void> {
  if (pin.length < 4 || pin.length > 8) {
    throw new Error('PIN must be 4-8 digits');
  }
  if (!/^\d+$/.test(pin)) {
    throw new Error('PIN must contain only digits');
  }
  const hash = hashPin(pin);
  await savePinHash(hash);
}

// PIN lockout thresholds: attempts -> lockout duration
const LOCKOUT_SCHEDULE: [number, number][] = [
  [3, 60_000],       // 3 failures = 60s lockout
  [6, 5 * 60_000],   // 6 failures = 5min lockout
  [9, 30 * 60_000],  // 9 failures = 30min lockout
];

/**
 * Check if PIN is currently locked out.
 * Returns remaining lockout ms, or 0 if not locked.
 */
export async function getPinLockoutRemaining(): Promise<number> {
  const state = await getPinLockout();
  if (state.lockedUntil <= 0) return 0;
  const remaining = state.lockedUntil - Date.now();
  return remaining > 0 ? remaining : 0;
}

/**
 * Verify a PIN against the stored hash.
 * Tracks failed attempts in persistent storage (survives app restart).
 * Returns { success, lockoutMs } where lockoutMs > 0 if now locked.
 */
export async function verifyPin(pin: string): Promise<{ success: boolean; lockoutMs: number }> {
  // Check if currently locked out
  const remaining = await getPinLockoutRemaining();
  if (remaining > 0) {
    return { success: false, lockoutMs: remaining };
  }

  const stored = await getPinHash();
  if (!stored) return { success: false, lockoutMs: 0 };

  const hash = hashPin(pin);
  if (hash === stored) {
    // Successful — clear lockout state
    await clearPinLockout();
    return { success: true, lockoutMs: 0 };
  }

  // Failed — increment and possibly lock
  const state = await getPinLockout();
  state.failedAttempts++;

  let lockoutMs = 0;
  for (const [threshold, duration] of LOCKOUT_SCHEDULE) {
    if (state.failedAttempts >= threshold) {
      lockoutMs = duration;
    }
  }

  if (lockoutMs > 0) {
    state.lockedUntil = Date.now() + lockoutMs;
  }

  await savePinLockout(state);
  return { success: false, lockoutMs };
}

/**
 * Check if the user has set up any authentication method.
 */
export async function hasAuthSetup(): Promise<boolean> {
  const pin = await hasPinSet();
  const bio = await isBiometricEnabled();
  return pin || bio;
}

// ---------- Auto-Lock ----------

/**
 * Record user activity to reset the auto-lock timer.
 */
export async function recordActivity(): Promise<void> {
  await saveLastActiveTime();
}

/**
 * Check if the wallet should be locked due to inactivity.
 */
export async function shouldAutoLock(): Promise<boolean> {
  const autoLockMinutes = await getAutoLockMinutes();
  if (autoLockMinutes <= 0) return false; // Auto-lock disabled

  const lastActive = await getLastActiveTime();
  if (lastActive === 0) return true; // Never been active

  const elapsed = Date.now() - lastActive;
  const thresholdMs = autoLockMinutes * 60 * 1000;
  return elapsed > thresholdMs;
}

/**
 * Full authentication flow: try biometrics first, then fall back to PIN.
 */
export async function authenticate(
  promptMessage?: string
): Promise<{ success: boolean; method: 'biometric' | 'pin' | 'none' }> {
  // Try biometrics first
  const bioEnabled = await isBiometricEnabled();
  if (bioEnabled) {
    const bioSuccess = await authenticateWithBiometrics(promptMessage);
    if (bioSuccess) {
      await recordActivity();
      return { success: true, method: 'biometric' };
    }
  }

  // Fall back to PIN (UI must handle PIN entry)
  return { success: false, method: 'pin' };
}
