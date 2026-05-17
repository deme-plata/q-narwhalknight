import { create } from 'zustand';
import * as api from '../services/api';
import { sseManager } from '../services/sse';
import {
  saveMnemonic,
  getMnemonic,
  deleteMnemonic,
  hasMnemonic,
  saveWalletAddress,
  getWalletAddress,
  saveAuthToken,
  saveDilithium5Keys,
  saveKyber1024Keys,
  savePQBackendType,
  wipeAllSecureData,
  saveOAuthTokens,
  getOAuthAccessToken,
  hasOAuthSession,
} from '../services/secureStorage';
import type { OAuthTokens, OAuthWalletInfo } from '../services/oauth';
import {
  generateMnemonic,
  validateMnemonicPhrase,
  deriveAddress,
  deriveAllKeys,
  createAuthChallenge,
} from '../services/wallet';
import { sanitizeBalance, MAX_SANE_BALANCE } from '../utils/sanitizeBalance';
import { bytesToHex } from '@noble/hashes/utils.js';

export interface TokenBalance {
  symbol: string;
  name: string;
  balance: number;
  rawBalance: string;
  decimals: number;
  priceUsd: number;
  valueUsd: number;
}

interface WalletState {
  // Auth state
  isLoggedIn: boolean;
  isLocked: boolean;
  isLoading: boolean;

  // Wallet data
  address: string | null;
  qugBalance: number;
  tokenBalances: TokenBalance[];
  nonce: number;

  // Transaction history
  transactions: api.Transaction[];
  txPage: number;
  txTotal: number;
  txLoading: boolean;

  // Actions
  initialize: () => Promise<void>;
  createWallet: () => Promise<string>;
  importWallet: (mnemonic: string) => Promise<void>;
  loginWithOAuth: (tokens: OAuthTokens, walletInfo: OAuthWalletInfo | null) => Promise<void>;
  loginVault: () => Promise<void>;
  logout: () => Promise<void>;
  lock: () => void;
  unlock: () => void;
  refreshBalance: () => Promise<void>;
  refreshHistory: (page?: number) => Promise<void>;
  setBalance: (balance: number, tokens: TokenBalance[]) => void;
}

// Guard against concurrent refreshBalance calls + allow cancellation
let _refreshAbort: AbortController | null = null;
let _refreshInFlight = false;

// Guard against concurrent refreshHistory calls + allow cancellation
let _historyAbort: AbortController | null = null;
let _historyInFlight = false;

export const useWalletStore = create<WalletState>((set, get) => ({
  isLoggedIn: false,
  isLocked: true,
  isLoading: true,
  address: null,
  qugBalance: 0,
  tokenBalances: [],
  nonce: 0,
  transactions: [],
  txPage: 1,
  txTotal: 0,
  txLoading: false,

  initialize: async () => {
    try {
      set({ isLoading: true });
      console.log('[WalletStore] Initializing — reading SecureStore...');

      const hasWallet = await hasMnemonic();
      console.log('[WalletStore] hasMnemonic:', hasWallet);

      if (hasWallet) {
        const address = await getWalletAddress();
        console.log('[WalletStore] Restored wallet session for', address?.slice(0, 16));
        set({ address, isLoggedIn: true, isLocked: true, isLoading: false });

        // Connect SSE and start refreshing immediately (auto-unlock will fire
        // via useAuth if no PIN/biometric is configured)
        if (address) {
          sseManager.connect(address);
          setTimeout(() => {
            get().refreshBalance();
            get().refreshHistory(1);
          }, 500);
        }
      } else {
        // Check for OAuth session (server vault login without mnemonic)
        const hasOAuth = await hasOAuthSession();
        console.log('[WalletStore] hasOAuthSession:', hasOAuth);
        if (hasOAuth) {
          const address = await getWalletAddress();
          const token = await getOAuthAccessToken();
          if (token) {
            api.setAuthToken(token);
            console.log('[WalletStore] Restored OAuth session for', address?.slice(0, 16));
          }
          set({ address, isLoggedIn: true, isLocked: false, isLoading: false });
          // Connect SSE and refresh balance for OAuth sessions
          if (address && address !== 'oauth-session') {
            sseManager.connect(address);
            setTimeout(() => get().refreshBalance(), 500);
          }
        } else {
          console.log('[WalletStore] No saved wallet found — showing login');
          set({ isLoggedIn: false, isLocked: false, isLoading: false });
        }
      }
    } catch (error) {
      console.error('[WalletStore] Initialize error:', error);
      set({ isLoggedIn: false, isLocked: false, isLoading: false });
    }
  },

  createWallet: async () => {
    const mnemonic = generateMnemonic();
    const keys = await deriveAllKeys(mnemonic);

    await saveMnemonic(mnemonic);
    await saveWalletAddress(keys.address);

    // Store PQ keys in secure storage (hex-encoded)
    if (keys.dilithium5PublicKey) {
      await saveDilithium5Keys(bytesToHex(keys.dilithium5PublicKey), '');
    }
    if (keys.kyber1024PublicKey) {
      await saveKyber1024Keys(bytesToHex(keys.kyber1024PublicKey), '');
    }
    if (keys.pqBackendType) {
      await savePQBackendType(keys.pqBackendType);
    }

    set({ address: keys.address, isLoggedIn: true, isLocked: false });
    sseManager.connect(keys.address);

    return mnemonic;
  },

  importWallet: async (mnemonic: string) => {
    if (!validateMnemonicPhrase(mnemonic)) {
      throw new Error('Invalid mnemonic phrase');
    }

    const keys = await deriveAllKeys(mnemonic);

    await saveMnemonic(mnemonic);
    await saveWalletAddress(keys.address);

    // Store PQ keys in secure storage (hex-encoded)
    if (keys.dilithium5PublicKey) {
      await saveDilithium5Keys(bytesToHex(keys.dilithium5PublicKey), '');
    }
    if (keys.kyber1024PublicKey) {
      await saveKyber1024Keys(bytesToHex(keys.kyber1024PublicKey), '');
    }
    if (keys.pqBackendType) {
      await savePQBackendType(keys.pqBackendType);
    }

    set({ address: keys.address, isLoggedIn: true, isLocked: false });
    sseManager.connect(keys.address);
  },

  loginWithOAuth: async (tokens: OAuthTokens, walletInfo: OAuthWalletInfo | null) => {
    // Persist tokens
    await saveOAuthTokens(tokens.accessToken, tokens.refreshToken, tokens.expiresAt);

    // Set auth token for API calls
    api.setAuthToken(tokens.accessToken);
    await saveAuthToken(tokens.accessToken);

    // Use wallet address from userinfo, or a placeholder
    const address = walletInfo?.walletAddress || 'oauth-session';
    await saveWalletAddress(address);

    console.log('[WalletStore] OAuth login complete, address:', address);
    set({ address, isLoggedIn: true, isLocked: false });

    if (address !== 'oauth-session') {
      sseManager.connect(address);
      // Trigger immediate balance refresh so user sees their balance right away
      setTimeout(() => get().refreshBalance(), 500);
    }
  },

  loginVault: async () => {
    const mnemonic = await getMnemonic();
    if (!mnemonic) throw new Error('No wallet found');

    const challenge = await createAuthChallenge(mnemonic);
    const result = await api.loginVault(challenge);

    await saveAuthToken(result.token);
    api.setAuthToken(result.token);
  },

  logout: async () => {
    // Cancel any in-flight requests before tearing down
    _refreshAbort?.abort();
    _refreshAbort = null;
    _refreshInFlight = false;
    _historyAbort?.abort();
    _historyAbort = null;
    _historyInFlight = false;

    sseManager.disconnect();
    api.setAuthToken(null);
    await wipeAllSecureData();

    set({
      isLoggedIn: false,
      isLocked: false,
      address: null,
      qugBalance: 0,
      tokenBalances: [],
      nonce: 0,
      transactions: [],
      txPage: 1,
      txTotal: 0,
    });
  },

  lock: () => {
    set({ isLocked: true });
  },

  unlock: () => {
    set({ isLocked: false });
    const { address } = get();
    if (address) {
      sseManager.connect(address);
    }
  },

  refreshBalance: async () => {
    const { address } = get();
    if (!address) return;

    // Skip if another refresh is already in flight
    if (_refreshInFlight) {
      console.log('[WalletStore] Balance refresh already in flight, skipping');
      return;
    }

    // Cancel any stale request from a previous call
    _refreshAbort?.abort();
    _refreshAbort = new AbortController();
    const signal = _refreshAbort.signal;
    _refreshInFlight = true;

    try {
      // Fetch multi-token balances (QUG, QUGUSD, custom tokens) in one call
      const multiToken = await api.getMultiTokenBalance(address, signal);

      // Don't update store if this request was cancelled while in flight
      if (signal.aborted) return;

      const qugEntry = multiToken.tokens['QUG'];
      const qugNum = qugEntry ? parseFloat(qugEntry.balance) : 0;
      const { value: safeBalance } = sanitizeBalance(qugNum);

      const tokens: TokenBalance[] = [];
      for (const [symbol, entry] of Object.entries(multiToken.tokens)) {
        if (symbol === 'QUG') continue; // QUG is shown separately as qugBalance
        tokens.push({
          symbol,
          name: entry.name ?? symbol,
          balance: parseFloat(entry.balance) || 0,
          rawBalance: entry.balance_base_units?.toString() ?? '0',
          decimals: entry.decimals ?? 24,
          priceUsd: entry.usd_value && parseFloat(entry.balance) > 0
            ? entry.usd_value / parseFloat(entry.balance)
            : 0,
          valueUsd: entry.usd_value ?? 0,
        });
      }

      set({
        qugBalance: safeBalance,
        tokenBalances: tokens,
        nonce: 0,
      });
    } catch (error) {
      // Suppress AbortError — it's an intentional cancellation, not a real failure
      if (error instanceof Error && error.name === 'AbortError') return;
      console.error('[WalletStore] Balance refresh error:', error);
    } finally {
      _refreshInFlight = false;
    }
  },

  refreshHistory: async (page: number = 1) => {
    const { address } = get();
    if (!address) return;

    // Skip if another history refresh is already in flight
    if (_historyInFlight) {
      console.log('[WalletStore] History refresh already in flight, skipping');
      return;
    }

    // Cancel any stale request from a previous call
    _historyAbort?.abort();
    _historyAbort = new AbortController();
    const signal = _historyAbort.signal;
    _historyInFlight = true;

    try {
      set({ txLoading: true });
      const data = await api.getHistory(address, page, 20, signal);

      // Don't update store if this request was cancelled while in flight
      if (signal.aborted) {
        set({ txLoading: false });
        return;
      }

      set({
        transactions: page === 1 ? data.transactions : [...get().transactions, ...data.transactions],
        txPage: data.page,
        txTotal: data.total,
        txLoading: false,
      });
    } catch (error) {
      // Suppress AbortError — it's an intentional cancellation, not a real failure
      if (error instanceof Error && error.name === 'AbortError') {
        set({ txLoading: false });
        return;
      }
      console.error('[WalletStore] History refresh error:', error);
      set({ txLoading: false });
    } finally {
      _historyInFlight = false;
    }
  },

  setBalance: (balance: number, tokens: TokenBalance[]) => {
    const { value: safeBalance } = sanitizeBalance(balance);
    set({ qugBalance: safeBalance, tokenBalances: tokens });
  },
}));
