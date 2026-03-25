/**
 * Format a raw balance (in smallest unit) to a human-readable string.
 * QUG uses 24 decimal places internally, but we display with reasonable precision.
 */

const DECIMALS = 24;
const MAX_DISPLAY_DECIMALS = 8;

/**
 * Convert a raw balance string (24-decimal) to a display number.
 * E.g. "1000000000000000000000000" => 1.0
 */
export function rawToDisplay(raw: string | bigint, decimals: number = DECIMALS): number {
  const rawBigInt = typeof raw === 'string' ? BigInt(raw) : raw;
  const divisor = BigInt(10) ** BigInt(decimals);
  const intPart = rawBigInt / divisor;
  const fracPart = rawBigInt % divisor;

  if (fracPart === BigInt(0)) {
    return Number(intPart);
  }

  const fracStr = fracPart.toString().padStart(decimals, '0').slice(0, MAX_DISPLAY_DECIMALS);
  return parseFloat(`${intPart}.${fracStr}`);
}

/**
 * Format a number with locale-aware thousands separators and fixed decimal places.
 */
export function formatNumber(
  value: number,
  options: {
    minDecimals?: number;
    maxDecimals?: number;
    locale?: string;
    compact?: boolean;
  } = {}
): string {
  const { minDecimals = 2, maxDecimals = 8, locale = 'en-US', compact = false } = options;

  if (compact && Math.abs(value) >= 1_000_000) {
    return new Intl.NumberFormat(locale, {
      notation: 'compact',
      maximumFractionDigits: 2,
    }).format(value);
  }

  if (compact && Math.abs(value) >= 1_000) {
    return new Intl.NumberFormat(locale, {
      notation: 'compact',
      maximumFractionDigits: 2,
    }).format(value);
  }

  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: minDecimals,
    maximumFractionDigits: maxDecimals,
  }).format(value);
}

/**
 * Format a balance for display with currency symbol.
 */
export function formatBalance(
  raw: string | bigint | number,
  symbol: string = 'QUG',
  decimals: number = DECIMALS
): string {
  let value: number;

  if (typeof raw === 'number') {
    value = raw;
  } else if (typeof raw === 'string' && raw.includes('.')) {
    // Already a display-formatted decimal string (e.g. "5.0" from swap history)
    value = parseFloat(raw) || 0;
  } else {
    value = rawToDisplay(raw, decimals);
  }

  const formatted = formatNumber(value, {
    minDecimals: 2,
    maxDecimals: value < 1 ? 8 : 4,
    compact: value >= 100_000,
  });

  return `${formatted} ${symbol}`;
}

/**
 * Format a fiat currency value.
 */
export function formatFiat(value: number, currency: string = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Abbreviate a long number string for compact display.
 */
export function abbreviateNumber(num: number): string {
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  return num.toFixed(2);
}
