/**
 * Password Prompt Hook
 * v3.6.12-beta: Updated to use proper modal context instead of browser window.prompt
 * Provides a function to prompt for password and recover mnemonic from encrypted storage
 */

import { usePasswordModal } from '../contexts/PasswordModalContext';
import { recoverMnemonic } from '../services/walletAuth';

/**
 * Hook to prompt for password using the custom modal
 * This replaces the old browser window.prompt implementation
 */
export function usePasswordPrompt() {
  const { requestPassword } = usePasswordModal();

  const promptForPassword = async (message?: string): Promise<string> => {
    try {
      const password = await requestPassword({
        title: 'Unlock Wallet',
        message: message || 'Enter your wallet password to continue',
      });

      if (!password) {
        throw new Error('Password cannot be empty');
      }

      return password;
    } catch (error) {
      throw new Error('Password prompt cancelled');
    }
  };

  const recoverMnemonicWithPrompt = async (): Promise<string> => {
    const password = await promptForPassword('Session expired. Enter your password to restore access.');

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
