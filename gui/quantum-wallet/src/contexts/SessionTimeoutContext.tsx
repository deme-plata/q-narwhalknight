import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import SessionTimeoutModal from '../components/SessionTimeoutModal';
import { recoverMnemonic, walletSession, keypairFromMnemonic, loadWallet } from '../services/walletAuth';

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

      // v3.7.4: Load full wallet to get Dilithium5 keys as well
      // The loadWallet function decrypts and loads Ed25519, SQIsign, and Dilithium5 keys
      let dilithium5SecretKey: Uint8Array | undefined;
      let dilithium5PublicKey: Uint8Array | undefined;

      try {
        const fullWallet = await loadWallet(password);
        dilithium5SecretKey = fullWallet.dilithium5SecretKey;
        dilithium5PublicKey = fullWallet.dilithium5PublicKey;
        if (dilithium5SecretKey) {
          console.log('✅ Loaded Dilithium5 post-quantum keys from encrypted storage');
        }
      } catch (walletError) {
        console.warn('⚠️ Could not load Dilithium5 keys, using Ed25519 only:', walletError);
      }

      // Restore session with Ed25519 keys (always) and Dilithium5 keys (if available)
      const keyPair = await keypairFromMnemonic(mnemonic);
      // Pass mnemonic to session for "Never expire" convenience (stored only if timeout is "never")
      // v3.7.4: Also include Dilithium5 keys for post-quantum P2P transactions
      walletSession.setSession(
        keyPair.privateKey,
        keyPair.address,
        mnemonic,
        dilithium5SecretKey,
        dilithium5PublicKey
      );

      // SECURITY: Mnemonic is only stored in sessionStorage if "Never expire" is enabled
      console.log('✅ Session restored with mnemonic for "Never expire" convenience');

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
