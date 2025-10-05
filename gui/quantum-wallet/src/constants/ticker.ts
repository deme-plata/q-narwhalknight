/**
 * Ticker and Branding Constants for Quillon Graph
 *
 * Official Name: Quillon Graph
 * Codename: Q-NarwhalKnight (internal only)
 * Ticker: QUG
 */

/** Official ticker symbol displayed to users */
export const TICKER_SYMBOL = 'QUG';

/** Legacy ticker symbol (for backward compatibility) */
export const LEGACY_TICKER = 'QNK';

/** Official project display name */
export const DISPLAY_NAME = 'Quillon Graph';

/** Address prefix for new addresses */
export const ADDRESS_PREFIX = 'qug';

/** Legacy address prefixes (still accepted) */
export const LEGACY_PREFIXES = ['qnk'];

/** Satoshis per coin */
export const SATOSHIS_PER_COIN = 100_000_000;

/**
 * Normalize address by removing any valid prefix
 */
export function normalizeAddress(address: string): string {
  const addressLower = address.toLowerCase();

  // Try new prefix
  if (addressLower.startsWith(ADDRESS_PREFIX)) {
    return addressLower.slice(ADDRESS_PREFIX.length);
  }

  // Try legacy prefixes
  for (const prefix of LEGACY_PREFIXES) {
    if (addressLower.startsWith(prefix)) {
      return addressLower.slice(prefix.length);
    }
  }

  // No prefix, return as-is
  return address;
}

/**
 * Add official prefix to normalized address
 */
export function addAddressPrefix(normalizedAddress: string): string {
  return `${ADDRESS_PREFIX}${normalizedAddress}`;
}

/**
 * Format balance with ticker symbol
 */
export function formatBalance(satoshis: number): string {
  const coins = satoshis / SATOSHIS_PER_COIN;
  return `${coins.toFixed(8)} ${TICKER_SYMBOL}`;
}

/**
 * Convert satoshis to decimal coins
 */
export function satoshisToCoins(satoshis: number): number {
  return satoshis / SATOSHIS_PER_COIN;
}

/**
 * Convert decimal coins to satoshis
 */
export function coinsToSatoshis(coins: number): number {
  return Math.floor(coins * SATOSHIS_PER_COIN);
}
