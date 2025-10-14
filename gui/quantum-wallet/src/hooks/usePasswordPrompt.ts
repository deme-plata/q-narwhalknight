/**
 * Password Prompt Hook
 * Provides a function to prompt for password and recover mnemonic from encrypted storage
 */

import { recoverMnemonic } from '../services/walletAuth';

/**
 * Prompt for password using browser's built-in prompt
 * In production, this should use a proper modal component
 */
export function usePasswordPrompt() {
  const promptForPassword = async (message?: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      // Use browser prompt for now
      // TODO: Replace with proper password modal component
      const password = window.prompt(message || 'Enter your wallet password to continue:');

      if (password === null) {
        reject(new Error('Password prompt cancelled'));
      } else if (password === '') {
        reject(new Error('Password cannot be empty'));
      } else {
        resolve(password);
      }
    });
  };

  const recoverMnemonicWithPrompt = async (): Promise<string> => {
    const password = await promptForPassword('Session expired. Enter your password to restore access:');

    try {
      const mnemonic = await recoverMnemonic(password);
      // SECURITY: Do NOT store plaintext mnemonic - keep it encrypted only
      console.log('✅ Mnemonic recovered from encrypted storage (not stored in plaintext)');
      return mnemonic;
    } catch (error) {
      throw new Error('Failed to decrypt mnemonic. Please check your password.');
    }
  };

  return {
    promptForPassword,
    recoverMnemonicWithPrompt,
  };
}
