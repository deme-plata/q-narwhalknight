import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import SessionTimeoutModal from '../components/SessionTimeoutModal';
import { recoverMnemonic, walletSession, keypairFromMnemonic } from '../services/walletAuth';

interface SessionTimeoutContextType {
  requestPassword: () => Promise<string>;
}

const SessionTimeoutContext = createContext<SessionTimeoutContextType | null>(null);

// Global reference to the password request function
let globalPasswordRequester: (() => Promise<string>) | null = null;

export const useSessionTimeout = () => {
  const context = useContext(SessionTimeoutContext);
  if (!context) {
    throw new Error('useSessionTimeout must be used within SessionTimeoutProvider');
  }
  return context;
};

/**
 * Get the global password requester function
 * This allows non-React code (like api.ts) to request passwords
 */
export const getGlobalPasswordRequester = (): (() => Promise<string>) | null => {
  return globalPasswordRequester;
};

export const SessionTimeoutProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState<string>('');
  const [resolver, setResolver] = useState<{
    resolve: (password: string) => void;
    reject: (error: Error) => void;
  } | null>(null);

  const requestPassword = useCallback((): Promise<string> => {
    return new Promise((resolve, reject) => {
      setIsOpen(true);
      setError('');
      setResolver({ resolve, reject });
    });
  }, []);

  // Register global password requester on mount
  useEffect(() => {
    globalPasswordRequester = requestPassword;
    return () => {
      globalPasswordRequester = null;
    };
  }, [requestPassword]);

  const handleSubmit = useCallback(async (password: string) => {
    try {
      // Attempt to decrypt mnemonic with provided password
      const mnemonic = await recoverMnemonic(password);

      // Restore session
      const keyPair = await keypairFromMnemonic(mnemonic);
      walletSession.setSession(keyPair.privateKey, keyPair.address);

      // SECURITY: Do NOT store plaintext mnemonic - keep it encrypted only
      console.log('✅ Session restored - mnemonic recovered from encrypted storage (not stored in plaintext)');

      // Resolve promise with the mnemonic
      if (resolver) {
        resolver.resolve(mnemonic);
        setResolver(null);
      }

      // Close modal
      setIsOpen(false);
      setError('');
    } catch (err) {
      // Show error in modal
      setError('Incorrect password. Please try again.');
      console.error('Failed to decrypt wallet:', err);
    }
  }, [resolver]);

  const handleCancel = useCallback(() => {
    if (resolver) {
      resolver.reject(new Error('Password request cancelled by user'));
      setResolver(null);
    }
    setIsOpen(false);
    setError('');
  }, [resolver]);

  return (
    <SessionTimeoutContext.Provider value={{ requestPassword }}>
      {children}
      <SessionTimeoutModal
        isOpen={isOpen}
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        error={error}
      />
    </SessionTimeoutContext.Provider>
  );
};

export default SessionTimeoutContext;
