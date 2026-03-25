/**
 * Quillon address validation utilities.
 *
 * A valid QNK address is:
 *   - Prefix "qnk" followed by exactly 64 lowercase hex characters
 *   - Total length: 67 characters
 */

const QNK_ADDRESS_REGEX = /^qnk[0-9a-f]{64}$/;
const QNK_PREFIX = 'qnk';
const QNK_ADDRESS_LENGTH = 67; // 3 (prefix) + 64 (hex pubkey)

/**
 * Validate a QNK address.
 * Returns true if the address is syntactically valid.
 */
export function isValidAddress(address: string): boolean {
  if (!address || typeof address !== 'string') return false;
  if (address.length !== QNK_ADDRESS_LENGTH) return false;
  return QNK_ADDRESS_REGEX.test(address);
}

/**
 * Validate an address and return a descriptive error message if invalid.
 */
export function validateAddress(address: string): { valid: boolean; error?: string } {
  if (!address || typeof address !== 'string') {
    return { valid: false, error: 'Address is required' };
  }

  const trimmed = address.trim().toLowerCase();

  if (trimmed.length === 0) {
    return { valid: false, error: 'Address is required' };
  }

  if (!trimmed.startsWith(QNK_PREFIX)) {
    return { valid: false, error: 'Address must start with "qnk"' };
  }

  if (trimmed.length !== QNK_ADDRESS_LENGTH) {
    return {
      valid: false,
      error: `Address must be ${QNK_ADDRESS_LENGTH} characters (got ${trimmed.length})`,
    };
  }

  const hexPart = trimmed.slice(QNK_PREFIX.length);
  if (!/^[0-9a-f]+$/.test(hexPart)) {
    return { valid: false, error: 'Address contains invalid characters' };
  }

  return { valid: true };
}

/**
 * Truncate an address for display. E.g. "qnkabcdef...12345678"
 */
export function truncateAddress(address: string, chars: number = 8): string {
  if (!address) return '';
  if (address.length <= chars * 2 + 3) return address;
  return `${address.slice(0, QNK_PREFIX.length + chars)}...${address.slice(-chars)}`;
}

/**
 * Parse a QNK URI (qnk://address?amount=X&label=Y)
 */
export function parseQnkUri(uri: string): {
  address: string;
  amount?: string;
  label?: string;
} | null {
  try {
    let cleaned = uri;
    if (cleaned.startsWith('qnk://')) {
      cleaned = cleaned.slice(6);
    } else if (cleaned.startsWith('qnk:')) {
      cleaned = cleaned.slice(4);
    }

    const [addressPart, queryPart] = cleaned.split('?');
    if (!isValidAddress(addressPart)) return null;

    const result: { address: string; amount?: string; label?: string } = {
      address: addressPart,
    };

    if (queryPart) {
      const params = new URLSearchParams(queryPart);
      const amount = params.get('amount');
      const label = params.get('label');
      if (amount) result.amount = amount;
      if (label) result.label = label;
    }

    return result;
  } catch {
    return null;
  }
}
