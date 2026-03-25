/**
 * Balance sanitization utilities.
 *
 * Quillon has a 21M max supply. Any balance exceeding this is invalid
 * and likely a display/sync bug. We clamp and warn rather than show
 * misleading numbers.
 */

/** Maximum sane QUG supply: 21,000,000 */
export const MAX_SANE_BALANCE = 21_000_000;

/** Maximum sane balance in raw (24-decimal) form */
export const MAX_SANE_BALANCE_RAW = BigInt('21000000') * BigInt(10) ** BigInt(24);

/**
 * Sanitize a display balance (number). Clamp to MAX_SANE_BALANCE.
 * Returns the clamped value and a flag indicating if clamping occurred.
 */
export function sanitizeBalance(value: number): {
  value: number;
  clamped: boolean;
  original: number;
} {
  if (!Number.isFinite(value) || value < 0) {
    return { value: 0, clamped: true, original: value };
  }

  if (value > MAX_SANE_BALANCE) {
    console.warn(
      `[sanitizeBalance] Balance ${value} exceeds MAX_SANE_BALANCE (${MAX_SANE_BALANCE}). Clamping.`
    );
    return { value: MAX_SANE_BALANCE, clamped: true, original: value };
  }

  return { value, clamped: false, original: value };
}

/**
 * Sanitize a raw balance string. Returns "0" for any negative or NaN input.
 */
export function sanitizeRawBalance(raw: string): string {
  try {
    const bi = BigInt(raw);
    if (bi < BigInt(0)) return '0';
    if (bi > MAX_SANE_BALANCE_RAW) {
      console.warn(
        `[sanitizeRawBalance] Raw balance exceeds max supply. Clamping.`
      );
      return MAX_SANE_BALANCE_RAW.toString();
    }
    return bi.toString();
  } catch {
    return '0';
  }
}

/**
 * Check if a token balance map contains any suspicious values.
 */
export function auditBalances(
  balances: Record<string, number>
): { clean: boolean; warnings: string[] } {
  const warnings: string[] = [];

  for (const [token, balance] of Object.entries(balances)) {
    if (balance < 0) {
      warnings.push(`Negative balance for ${token}: ${balance}`);
    }
    if (!Number.isFinite(balance)) {
      warnings.push(`Non-finite balance for ${token}: ${balance}`);
    }
    if (token === 'QUG' && balance > MAX_SANE_BALANCE) {
      warnings.push(`QUG balance ${balance} exceeds max supply ${MAX_SANE_BALANCE}`);
    }
  }

  return { clean: warnings.length === 0, warnings };
}
