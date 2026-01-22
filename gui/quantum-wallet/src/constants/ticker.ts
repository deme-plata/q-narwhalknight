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

/**
 * Base units per coin (QUG uses 24 decimals: 1 QUG = 10^24 base units)
 * v3.0.6-beta: Updated for u128 migration
 * NOTE: JavaScript cannot accurately represent 10^24, so we use 1e24 for division
 * When displaying, prefer the pre-formatted `balance` string from API over `balance_base_units`
 */
export const SATOSHIS_PER_COIN = 1e24;

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
 * v3.0.6-beta: Updated to show up to 16 decimal places for small amounts
 */
export function formatBalance(satoshis: number): string {
  const coins = satoshis / SATOSHIS_PER_COIN;
  // For very small amounts, show more decimals
  if (coins > 0 && coins < 0.00000001) {
    return `${coins.toFixed(16)} ${TICKER_SYMBOL}`;
  }
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
