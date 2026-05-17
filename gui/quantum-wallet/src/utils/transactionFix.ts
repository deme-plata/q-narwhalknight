// Transaction Validation Fix
// QUG uses 9 decimals (1 QUG = 1,000,000,000 base units)

export const QNK_UNIT_MULTIPLIER = 1000000000;

/**
 * Check if an amount has been unit-converted
 */
export function isUnitConverted(amount: number): boolean {
  // If amount is greater than 1,000,000 and is close to a whole number when divided by the multiplier
  if (amount > 1000000) {
    const possibleQnk = amount / QNK_UNIT_MULTIPLIER;
    // Check if it looks like a reasonable QNK amount (between 0.001 and 10000)
    return possibleQnk >= 0.001 && possibleQnk <= 10000;
  }
  return false;
}

/**
 * Fix an amount that has been unit-converted
 */
export function fixAmount(amount: number): number {
  if (isUnitConverted(amount)) {
    const fixed = amount / QNK_UNIT_MULTIPLIER;
    console.log(`🔧 Fixed unit conversion: ${amount} -> ${fixed} QNK`);
    return fixed;
  }
  return amount;
}

/**
 * Validate balance with unit conversion fix
 */
export function validateBalance(
  required: number,
  available: number,
  skipValidation: boolean = false
): { valid: boolean; error?: string; fixedRequired?: number; fixedAvailable?: number } {
  // Fix amounts if they've been unit-converted
  const fixedRequired = fixAmount(required);
  const fixedAvailable = fixAmount(available);
  
  console.log('💰 Balance Validation:', {
    originalRequired: required,
    originalAvailable: available,
    fixedRequired,
    fixedAvailable,
    skipValidation
  });
  
  // If skipValidation is true, always return valid
  if (skipValidation) {
    console.log('⚠️ Skipping balance validation (debug mode)');
    return { valid: true, fixedRequired, fixedAvailable };
  }
  
  // Check if balance is sufficient (zero balance = insufficient)
  if (fixedAvailable === 0) {
    return {
      valid: false,
      error: 'No balance. Earn QUG through mining.',
      fixedRequired,
      fixedAvailable,
    };
  }

  if (fixedAvailable < fixedRequired) {
    return {
      valid: false,
      error: `Insufficient balance. Required: ${fixedRequired.toFixed(8)} QNK, Available: ${fixedAvailable.toFixed(8)} QNK`,
      fixedRequired,
      fixedAvailable
    };
  }
  
  return { valid: true, fixedRequired, fixedAvailable };
}

/**
 * Override global Error to catch balance validation errors
 */
export function installGlobalErrorFix(): void {
  const OriginalError = window.Error;
  
  // @ts-ignore - We're intentionally overriding the Error constructor
  window.Error = function(message: string, ...args: any[]) {
    // Check if this is a balance error with unit-converted amounts
    if (message && message.includes('Insufficient balance')) {
      const match = message.match(/Required:\s*(\d+),\s*Available:\s*(\d+)/);
      if (match) {
        const required = parseInt(match[1]);
        const available = parseInt(match[2]);
        
        if (isUnitConverted(required)) {
          const fixedRequired = fixAmount(required);
          const fixedAvailable = fixAmount(available);
          
          console.warn(`⚠️ Caught unit conversion error:`, {
            original: message,
            fixedRequired,
            fixedAvailable
          });
          
          // Return a fixed error message
          message = `Insufficient balance. Required: ${fixedRequired.toFixed(8)} QNK, Available: ${fixedAvailable.toFixed(8)} QNK`;
        }
      }
    }
    
    return new OriginalError(message, ...args);
  };
  
  // Copy prototype
  Object.setPrototypeOf(window.Error, OriginalError);
  Object.setPrototypeOf(window.Error.prototype, OriginalError.prototype);
  
  console.log('✅ Global error fix installed');
}

// Auto-install the fix when this module is imported
if (typeof window !== 'undefined') {
  installGlobalErrorFix();
  console.log('🔧 Transaction validation fix loaded');
}