import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import PasswordModal from '../components/PasswordModal';
import { setPasswordPrompt } from '../services/api';

interface PasswordModalContextType {
  requestPassword: (options?: PasswordRequestOptions) => Promise<string>;
}

interface PasswordRequestOptions {
  title?: string;
  message?: string;
}

interface PasswordModalState {
  isOpen: boolean;
  title: string;
  message: string;
  error: string;
  resolve: ((password: string) => void) | null;
  reject: ((reason?: any) => void) | null;
}

const PasswordModalContext = createContext<PasswordModalContextType | null>(null);

export const usePasswordModal = () => {
  const context = useContext(PasswordModalContext);
  if (!context) {
    throw new Error('usePasswordModal must be used within PasswordModalProvider');
  }
  return context;
};

export const PasswordModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [modalState, setModalState] = useState<PasswordModalState>({
    isOpen: false,
    title: 'Unlock Wallet',
    message: 'Enter your wallet password to continue',
    error: '',
    resolve: null,
    reject: null,
  });

  const requestPassword = useCallback((options?: PasswordRequestOptions): Promise<string> => {
    return new Promise((resolve, reject) => {
      setModalState({
        isOpen: true,
        title: options?.title || 'Unlock Wallet',
        message: options?.message || 'Enter your wallet password to continue',
        error: '',
        resolve,
        reject,
      });
    });
  }, []);

  // Register the password prompt with the API service
  useEffect(() => {
    setPasswordPrompt(requestPassword);
    return () => {
      setPasswordPrompt(null);
    };
  }, [requestPassword]);

  const handleSubmit = useCallback((password: string) => {
    if (modalState.resolve) {
      modalState.resolve(password);
      setModalState((prev) => ({
        ...prev,
        isOpen: false,
        error: '',
        resolve: null,
        reject: null,
      }));
    }
  }, [modalState.resolve]);

  const handleCancel = useCallback(() => {
    if (modalState.reject) {
      modalState.reject(new Error('Password request cancelled by user'));
      setModalState((prev) => ({
        ...prev,
        isOpen: false,
        error: '',
        resolve: null,
        reject: null,
      }));
    }
  }, [modalState.reject]);

  return (
    <PasswordModalContext.Provider value={{ requestPassword }}>
      {children}
      <PasswordModal
        isOpen={modalState.isOpen}
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        title={modalState.title}
        message={modalState.message}
        error={modalState.error}
      />
    </PasswordModalContext.Provider>
  );
};

export default PasswordModalContext;
