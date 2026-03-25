import { useEffect, useRef, useState, useCallback } from 'react';
import { AppState, AppStateStatus } from 'react-native';
import { useWalletStore } from '../stores/walletStore';
import { shouldAutoLock, recordActivity, hasAuthSetup } from '../services/auth';

/**
 * Hook that manages authentication state and auto-lock behavior.
 *
 * Key behavior: If the user has NOT set up any auth (no PIN, no biometrics),
 * we auto-unlock — never show the lock screen for a fresh wallet that has
 * no security configured yet. The user can enable PIN/biometrics in Settings.
 */
export function useAuth() {
  const isLoggedIn = useWalletStore((s) => s.isLoggedIn);
  const isLocked = useWalletStore((s) => s.isLocked);
  const lock = useWalletStore((s) => s.lock);
  const unlock = useWalletStore((s) => s.unlock);
  const appStateRef = useRef<AppStateStatus>(AppState.currentState);
  const [authConfigured, setAuthConfigured] = useState<boolean | null>(null);

  // On mount + when login state changes, check if auth is configured
  useEffect(() => {
    if (!isLoggedIn) {
      setAuthConfigured(null);
      return;
    }

    hasAuthSetup().then((configured) => {
      setAuthConfigured(configured);

      // Auto-unlock if no auth method is set up (fresh wallet)
      if (!configured && isLocked) {
        unlock();
      }
    });
  }, [isLoggedIn, isLocked, unlock]);

  // Record activity on each interaction
  const touch = useCallback(async () => {
    if (isLoggedIn && !isLocked) {
      await recordActivity();
    }
  }, [isLoggedIn, isLocked]);

  // Auto-lock on app state change (only if auth is configured)
  useEffect(() => {
    const subscription = AppState.addEventListener('change', async (nextState) => {
      if (
        appStateRef.current.match(/inactive|background/) &&
        nextState === 'active'
      ) {
        if (isLoggedIn) {
          const configured = await hasAuthSetup();
          if (configured) {
            const shouldLock = await shouldAutoLock();
            if (shouldLock) {
              lock();
            }
          }
        }
      } else if (nextState === 'background') {
        await recordActivity();
      }
      appStateRef.current = nextState;
    });

    return () => subscription.remove();
  }, [isLoggedIn, lock]);

  // Only require auth if user has actually configured a PIN or biometrics
  const needsAuth = isLoggedIn && isLocked && authConfigured === true;

  return {
    isLoggedIn,
    isLocked,
    touch,
    needsAuth,
    needsOnboarding: !isLoggedIn,
  };
}
