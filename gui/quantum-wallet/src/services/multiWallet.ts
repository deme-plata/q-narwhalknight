// multiWallet.ts — per-address credential vault for the multi-wallet drawer.
//
// WHY THIS EXISTS
// ---------------
// The wallet grew two credential systems stacked on top of each other:
//
//   LEGACY (single-wallet)          CURRENT (what actually signs)
//   ─────────────────────           ──────────────────────────────────────
//   localStorage.walletAddress      sessionStorage.walletSession
//   localStorage.walletSeed           → { privateKey, address, mnemonic }
//                                   localStorage.walletEncryptedMnemonic
//                                   localStorage.walletEncryptedKey
//                                   localStorage.walletPasswordHash
//                                   localStorage.walletEncrypted{Aegis,SQIsign,Dilithium5}Key
//
// `MultiWalletDrawer` used to switch wallets by writing ONLY the two legacy
// keys and reloading. The CURRENT credentials never moved, which produced
// three bugs, escalating in severity:
//
//   1. After a "switch", walletSession still held the PREVIOUS wallet's
//      private key + address, so every signed API call (send, dex_swap)
//      silently operated on the old wallet while the UI displayed the new
//      address.
//   2. Drawer-created wallets were raw `crypto.getRandomValues()` entropy —
//      no BIP39 mnemonic, no password encryption, never registered with
//      walletSession. They could never sign, and clearing the browser lost
//      them permanently.
//   3. THE DESTRUCTIVE ONE: `walletAddress` now pointed at wallet B while the
//      encrypted blobs still belonged to wallet A. On the next login with A's
//      mnemonic, LoginScreen's "address mismatch → different wallet" branch
//      fired and DELETED walletEncryptedMnemonic / walletEncryptedKey /
//      walletPasswordHash. The user was logged out of their main wallet and
//      had to re-import from seed phrase.
//
// THE FIX
// -------
// Treat the global credential keys as a *window* onto exactly one wallet, and
// keep the full set for every wallet in a per-address vault:
//
//   quillon:wallet:<address>  →  { version, address, savedAt, creds: {...} }
//
// Switching is then: snapshot(current) → restore(target) → loadWallet(password)
// → walletSession.setSession(...). Nothing is ever deleted, so a wallet can
// never be orphaned by a switch.

/**
 * Every localStorage key that belongs to ONE specific wallet.
 *
 * Sourced from `storeWallet()` / `loadWallet()` in walletAuth.ts plus the
 * removal list in App.tsx's handleLogout — if a key is per-wallet and either
 * of those touches it, it must be here. Missing a key means that credential
 * leaks across a switch and gets decrypted with the wrong password.
 *
 * NOTE: `walletAddress` is deliberately NOT in this list. It is the pointer
 * that selects which wallet is active, not part of the credential payload.
 */
const CREDENTIAL_KEYS = [
  'walletEncryptedKey',
  'walletEncryptedMnemonic',
  'walletPasswordHash',
  'walletPublicKey',
  'walletId',
  'walletEncryptedAegisKey',
  'walletAegisPublicKey',
  'walletEncryptedSQIsignKey',
  'walletSQIsignPublicKey',
  'walletEncryptedDilithium5Key',
  'walletDilithium5PublicKey',
] as const;

/**
 * Balance/history caches keyed to the active wallet. These are NOT credentials
 * — they must be dropped on every switch so wallet B never renders wallet A's
 * cached balance during the reload window.
 */
const CACHE_KEYS = [
  'cachedBalance',
  'cachedQugusdBalance',
  'walletBalanceHistory',
  'dexLockedBalance',
  'dexCooldownUntil',
  'protectedTokenBalances',
  'customTokensCache',
  'customTokensCooldownUntil',
  'qnk_balance_long_v1',
] as const;

export type WalletTemplateId = 'savings' | 'trading' | 'mining' | 'agent' | 'faucet';

export interface WalletEntry {
  address: string;
  name: string;
  template: WalletTemplateId | 'main';
  createdAt: string;
}

interface VaultBlob {
  version: 1;
  address: string;
  savedAt: string;
  creds: Record<string, string>;
}

const VAULT_PREFIX = 'quillon:wallet:';
const WALLET_LIST_KEY = 'quillon:wallets';

function vaultKey(address: string): string {
  return `${VAULT_PREFIX}${address}`;
}

// ---------------------------------------------------------------------------
// Vault: snapshot / restore
// ---------------------------------------------------------------------------

/**
 * Copy the currently-active global credential keys into `address`'s vault slot.
 *
 * Call this BEFORE anything overwrites the globals — switching wallets,
 * creating a wallet, or importing a different one in LoginScreen. It is safe to
 * call repeatedly.
 *
 * Returns false (and writes nothing) when the globals hold no usable
 * credentials, so we never persist an empty blob over a good one.
 */
export function snapshotCredentials(address: string): boolean {
  if (!address) return false;

  const creds: Record<string, string> = {};
  for (const key of CREDENTIAL_KEYS) {
    const value = localStorage.getItem(key);
    if (value !== null) creds[key] = value;
  }

  // A wallet is only recoverable if we hold the encrypted key or mnemonic.
  // Anything less is a partial state we refuse to persist.
  if (!creds.walletEncryptedKey && !creds.walletEncryptedMnemonic) {
    console.warn(`[multiWallet] Refusing to snapshot ${address.slice(0, 12)}… — no encrypted key or mnemonic in globals`);
    return false;
  }

  const blob: VaultBlob = {
    version: 1,
    address,
    savedAt: new Date().toISOString(),
    creds,
  };

  try {
    localStorage.setItem(vaultKey(address), JSON.stringify(blob));
    console.log(`[multiWallet] Snapshotted ${Object.keys(creds).length} credential keys for ${address.slice(0, 12)}…`);
    return true;
  } catch (e) {
    console.error('[multiWallet] Snapshot failed (quota?):', e);
    return false;
  }
}

/**
 * Load `address`'s vault slot into the global credential keys, making it the
 * wallet that `loadWallet()` / `recoverMnemonic()` will operate on.
 *
 * Clears every credential key first, so a key the target wallet does not have
 * (e.g. no AEGIS-QL) cannot linger from the previous wallet and get decrypted
 * with the wrong password.
 *
 * Does NOT set `walletAddress` and does NOT touch walletSession — the caller
 * does both, after it has successfully decrypted with the password.
 */
export function restoreCredentials(address: string): boolean {
  if (!address) return false;

  const raw = localStorage.getItem(vaultKey(address));
  if (!raw) {
    console.warn(`[multiWallet] No vault entry for ${address.slice(0, 12)}…`);
    return false;
  }

  let blob: VaultBlob;
  try {
    blob = JSON.parse(raw);
  } catch {
    console.error(`[multiWallet] Vault entry for ${address.slice(0, 12)}… is corrupt JSON`);
    return false;
  }

  if (blob.version !== 1 || blob.address !== address || !blob.creds) {
    console.error('[multiWallet] Vault entry failed validation', { version: blob.version, address: blob.address });
    return false;
  }

  // Wipe the window, then paint the target wallet into it.
  for (const key of CREDENTIAL_KEYS) localStorage.removeItem(key);
  for (const [key, value] of Object.entries(blob.creds)) {
    localStorage.setItem(key, value);
  }

  console.log(`[multiWallet] Restored ${Object.keys(blob.creds).length} credential keys for ${address.slice(0, 12)}…`);
  return true;
}

/** True when we hold enough locally to unlock `address` with its password. */
export function hasCredentials(address: string): boolean {
  if (!address) return false;
  const raw = localStorage.getItem(vaultKey(address));
  if (!raw) return false;
  try {
    const blob: VaultBlob = JSON.parse(raw);
    return !!(blob?.creds?.walletEncryptedKey || blob?.creds?.walletEncryptedMnemonic);
  } catch {
    return false;
  }
}

/** Drop the caches that belong to the outgoing wallet. */
export function clearWalletCaches(): void {
  for (const key of CACHE_KEYS) localStorage.removeItem(key);
}

// ---------------------------------------------------------------------------
// Wallet list
// ---------------------------------------------------------------------------

/** Read the drawer's wallet list. Never throws. */
export function loadWalletList(): WalletEntry[] {
  try {
    const raw = localStorage.getItem(WALLET_LIST_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch { /* corrupt — fall through to the migration path */ }
  return [];
}

export function saveWalletList(wallets: WalletEntry[]): void {
  try {
    localStorage.setItem(WALLET_LIST_KEY, JSON.stringify(wallets));
  } catch (e) {
    console.error('[multiWallet] Failed to persist wallet list:', e);
  }
}

export function findWalletEntry(address: string): WalletEntry | undefined {
  return loadWalletList().find(w => w.address === address);
}

// ---------------------------------------------------------------------------
// Migration
// ---------------------------------------------------------------------------

/**
 * Bring pre-vault sessions up to date. Idempotent — safe on every app boot and
 * every drawer open.
 *
 * 1. If the active wallet has credentials in the globals but no vault slot,
 *    snapshot it. THIS IS THE LINE THAT STOPS THE MAIN WALLET BEING ORPHANED:
 *    users who already created a drawer wallet under the old code have a
 *    `walletAddress` pointing at a seed-only wallet while the globals still
 *    hold the main wallet's credentials.
 * 2. Ensure the active wallet appears in the drawer's list.
 *
 * Returns the address it vaulted, or null if there was nothing to do.
 */
export function migrateLegacyWallet(): string | null {
  const activeAddress = localStorage.getItem('walletAddress') || '';
  const hasGlobalCreds = !!(
    localStorage.getItem('walletEncryptedKey') || localStorage.getItem('walletEncryptedMnemonic')
  );

  let vaulted: string | null = null;

  if (activeAddress && hasGlobalCreds && !hasCredentials(activeAddress)) {
    // The globals may in fact belong to a DIFFERENT wallet than walletAddress
    // points at (exactly the corrupt state the old drawer produced). We cannot
    // tell which without the password, so we vault them under the address
    // currently on file — the same address LoginScreen would compare against.
    if (snapshotCredentials(activeAddress)) {
      vaulted = activeAddress;
      console.log('[multiWallet] Migrated pre-vault wallet into the credential vault');
    }
  }

  const list = loadWalletList();
  if (activeAddress && !list.some(w => w.address === activeAddress)) {
    list.push({
      address: activeAddress,
      name: 'Primary',
      template: 'main',
      createdAt: new Date().toISOString(),
    });
    saveWalletList(list);
  }

  return vaulted;
}
