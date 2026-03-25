import { create } from 'zustand';
import {
  saveAutoLockMinutes,
  getAutoLockMinutes,
  saveBiometricEnabled,
  isBiometricEnabled,
} from '../services/secureStorage';

type ThemeMode = 'dark' | 'light' | 'system';
type Currency = 'USD' | 'EUR' | 'GBP' | 'BTC' | 'ETH';

interface SettingsState {
  themeMode: ThemeMode;
  currency: Currency;
  autoLockMinutes: number;
  biometricEnabled: boolean;
  notificationsEnabled: boolean;
  hideBalances: boolean;

  // Actions
  loadSettings: () => Promise<void>;
  setThemeMode: (mode: ThemeMode) => void;
  setCurrency: (currency: Currency) => void;
  setAutoLockMinutes: (minutes: number) => Promise<void>;
  setBiometricEnabled: (enabled: boolean) => Promise<void>;
  setNotificationsEnabled: (enabled: boolean) => void;
  setHideBalances: (hide: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  themeMode: 'dark',
  currency: 'USD',
  autoLockMinutes: 5,
  biometricEnabled: false,
  notificationsEnabled: true,
  hideBalances: false,

  loadSettings: async () => {
    try {
      const [lockMinutes, bioEnabled] = await Promise.all([
        getAutoLockMinutes(),
        isBiometricEnabled(),
      ]);

      set({
        autoLockMinutes: lockMinutes,
        biometricEnabled: bioEnabled,
      });
    } catch (error) {
      console.error('[SettingsStore] Load error:', error);
    }
  },

  setThemeMode: (mode) => set({ themeMode: mode }),

  setCurrency: (currency) => set({ currency }),

  setAutoLockMinutes: async (minutes) => {
    await saveAutoLockMinutes(minutes);
    set({ autoLockMinutes: minutes });
  },

  setBiometricEnabled: async (enabled) => {
    await saveBiometricEnabled(enabled);
    set({ biometricEnabled: enabled });
  },

  setNotificationsEnabled: (enabled) => set({ notificationsEnabled: enabled }),

  setHideBalances: (hide) => set({ hideBalances: hide }),
}));
