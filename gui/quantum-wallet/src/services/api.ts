// Q-NarwhalKnight API Service
// Handles all communication with the quantum consensus node
// v1.0.53: Added automatic port discovery when default port is unavailable

import { generateAuthHeader, walletSession, loadWallet, keypairFromMnemonic, recoverMnemonic } from './walletAuth';
import { discoverNode, getDiscoveredNodeUrl, onNodeDiscovered } from './nodeDiscovery';

// Get API base URL from localStorage (set by network selector or auto-discovery) or use default
const getApiBaseUrl = () => {
  const storedBaseURL = localStorage.getItem('apiBaseURL');
  if (storedBaseURL) {
    return storedBaseURL + '/api';
  }
  return import.meta.env.VITE_API_URL || '/api';
};

let API_BASE_URL = getApiBaseUrl();

// v1.0.53: Initialize node discovery on module load
// This runs once when the API module is first imported
(async () => {
  console.log('🔍 [API] Initializing node auto-discovery...');
  const result = await discoverNode({
    startPort: 8080,
    maxAttempts: 10,
    timeout: 2000,
  });

  if (result.success && result.url) {
    const newBaseUrl = result.url + '/api';
    if (newBaseUrl !== API_BASE_URL) {
      console.log(`✅ [API] Auto-discovered node at ${result.url}, updating API base URL`);
      API_BASE_URL = newBaseUrl;
    }
  } else {
    console.warn('⚠️ [API] Node auto-discovery failed, using default URL:', API_BASE_URL);
  }
})();

// v1.0.53: Listen for node discovery events (in case discovery happens after initial load)
if (typeof window !== 'undefined') {
  onNodeDiscovered((result) => {
    if (result.success && result.url) {
      const newBaseUrl = result.url + '/api';
      console.log(`🔄 [API] Node discovered event received, updating to: ${newBaseUrl}`);
      API_BASE_URL = newBaseUrl;
    }
  });
}

// ============================================
// REQUEST THROTTLING & DEBOUNCING UTILITIES
// ============================================

/**
 * Throttle function - Limits function execution to once per interval
 * Prevents API request storms by enforcing minimum time between calls
 */
function throttle<T extends (...args: any[]) => any>(
  func: T,
  limitMs: number
): (...args: Parameters<T>) => ReturnType<T> | undefined {
  let inThrottle = false;
  let lastResult: ReturnType<T> | undefined;

  return function(this: any, ...args: Parameters<T>): ReturnType<T> | undefined {
    if (!inThrottle) {
      inThrottle = true;
      lastResult = func.apply(this, args);
      setTimeout(() => (inThrottle = false), limitMs);
      return lastResult;
    }
    console.log(`⏱️ [THROTTLE] Request throttled, minimum ${limitMs}ms between calls`);
    return lastResult;
  };
}

/**
 * Debounce function - Delays function execution until after wait time has elapsed
 * since the last time it was invoked. Prevents rapid-fire requests.
 */
function debounce<T extends (...args: any[]) => any>(
  func: T,
  waitMs: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function(this: any, ...args: Parameters<T>): void {
    const later = () => {
      timeout = null;
      func.apply(this, args);
    };

    if (timeout) {
      console.log(`⏱️ [DEBOUNCE] Request debounced, waiting ${waitMs}ms...`);
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, waitMs);
  };
}

/**
 * Request rate limiter - Tracks concurrent requests and enforces limits
 */
class RequestRateLimiter {
  private activeRequests = 0;
  private readonly maxConcurrent: number;

  constructor(maxConcurrent = 5) {
    this.maxConcurrent = maxConcurrent;
  }

  async acquire(): Promise<void> {
    while (this.activeRequests >= this.maxConcurrent) {
      console.log(`⏸️ [RATE LIMIT] Max concurrent requests reached (${this.maxConcurrent}), waiting...`);
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    this.activeRequests++;
  }

  release(): void {
    this.activeRequests--;
  }

  getActiveCount(): number {
    return this.activeRequests;
  }
}

const rateLimiter = new RequestRateLimiter(10); // Max 10 concurrent requests (balanced for performance)

// Global password prompt function - will be set by PasswordModalProvider
let globalPasswordPrompt: (() => Promise<string>) | null = null;

/**
 * Set the global password prompt function
 * This is called by the PasswordModalProvider when it's initialized
 */
export function setPasswordPrompt(promptFn: (() => Promise<string>) | null) {
  globalPasswordPrompt = promptFn;
  console.log('Password prompt registered:', !!promptFn);
}

export interface MnemonicResponse {
  mnemonic: string;
  words: string[];
  entropy: string;
  word_count: number;
  entropy_bits: number;
  language: string;
  standard: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
  timestamp: string;
}

// v3.4.0-beta: Fee estimation response from /api/v1/transactions/estimate-fee
export interface FeeEstimate {
  min_fee: string;          // Minimum fee in atomic units (as string for large numbers)
  recommended_fee: string;  // Recommended fee based on priority
  max_fee: string;          // Maximum reasonable fee
  recommended_fee_qug: number; // Human-readable fee in QUG
  gas_units: string;        // Gas units required
  congestion: number;       // Network congestion (0.0 - 1.0)
  tx_type: string;          // Transaction type parsed
}

// v3.4.0-beta: Fee reduction info for UI display
export const FEE_REDUCTION_ACTIVATION_HEIGHT = 350000;
export const CURRENT_MIN_FEE_QUG = 0.00021;  // Legacy fee before activation
export const NEW_MIN_FEE_QUG = 0.000021;     // Reduced fee after activation (10x cheaper)

export interface NodeStatus {
  node_id: string;
  current_round: number;
  current_height: number;
  connected_peers: number;
  tx_pool_size: number;
  is_validator: boolean;
  uptime_seconds: number;
  uptime_formatted: string;
  network_health: string;
  consensus_status: string;
  last_block_time: number;
  tps_current: number;
  tps_average: number;
  balance: number;
  performance?: {
    max_theoretical_tps: number;
    optimization_level: string;
    simd_crypto_enabled: boolean;
    kernel_io_enabled: boolean;
  };
}

export interface NetworkSupply {
  max_supply: number;
  max_supply_formatted: string;
  total_mined: number;
  total_mined_formatted: string;
  total_mined_base_units: number;
  remaining_supply: number;
  remaining_supply_formatted: string;
  circulating_percentage: number;
  circulating_percentage_formatted: string;
  network_hashrate: number;
  network_hashrate_formatted: string;
  block_reward: number;
  block_reward_formatted: string;
  current_height: number;
  connected_miners: number;
  holders: number;  // v2.3.8-beta: Real holder count from blockchain
  holders_formatted: string;
  timestamp: string;
}

// v2.3.8-beta: QUGUSD Stablecoin Vault Stats (real CDP data)
export interface VaultStats {
  total_qug_locked: number;     // QUG locked as collateral (base units)
  total_qugusd_minted: number;  // Total QUGUSD in circulation (base units)
  qug_price_usd: number;        // Current QUG price in USD
  global_collateral_ratio: number; // Collateralization ratio
  num_positions: number;        // Number of CDP positions (holders count)
  last_price_update: number;    // Unix timestamp of last price update
}

// Hashpower-weighted cryptographic security metrics (v1.3.0-beta)
export interface HashpowerSecurityData {
  version: string;
  feature: string;
  metrics: {
    blocks_processed: number;
    security_bits: number;
    security_tier: string;
    vdf_difficulty: number;
    beacon_epoch: number;
    network_hashrate: number;
    cumulative_work: string;
  };
  security_guarantees: {
    collision_resistance: string;
    preimage_resistance: string;
    double_spend_cost_usd: string;
    '51_percent_attack_cost': string;
  };
  components: {
    cumulative_work_security: boolean;
    adaptive_vdf_complexity: boolean;
    mining_randomness_beacon: boolean;
  };
}

export interface WalletData {
  id: string;
  address: number[];
  address_formatted?: string;
  public_key: number[];
  balance: number;
  nonce: number;
  created_at: string;
}

class QNarwhalKnightAPI {
  private _baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this._baseURL = baseURL;
  }

  // v1.0.53: Getter that always returns the latest discovered URL
  private get baseURL(): string {
    // If we were initialized with a custom URL, use that
    // Otherwise, always use the latest discovered URL
    if (this._baseURL !== '/api' && !this._baseURL.includes(':8080')) {
      return this._baseURL;
    }
    return API_BASE_URL;
  }

  // v1.0.53: Method to update base URL (called when node is discovered)
  public setBaseURL(url: string): void {
    this._baseURL = url;
    console.log(`🔄 [API] Base URL updated to: ${url}`);
  }

  // v1.0.53: Get current base URL (for debugging)
  public getBaseURL(): string {
    return this.baseURL;
  }

  private async request<T>(endpoint: string, options?: RequestInit, retries = 3): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;

    // Acquire rate limit token before making request
    await rateLimiter.acquire();

    try {
      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          // Exponential backoff: 100ms, 300ms, 900ms
          if (attempt > 0) {
            const backoffMs = Math.min(100 * Math.pow(3, attempt - 1), 1000);
            console.log(`⏳ [RETRY ${attempt}/${retries}] Waiting ${backoffMs}ms before retry...`);
            await new Promise(resolve => setTimeout(resolve, backoffMs));
          }

          console.log(`🌐 [API REQUEST] ${options?.method || 'GET'} ${endpoint} (attempt ${attempt + 1}/${retries + 1})`);

          const response = await fetch(url, {
            ...options,
            headers: {
              'Content-Type': 'application/json',
              ...options?.headers,
            },
          });

          // Handle rate limiting with exponential backoff (DISABLED - no rate limiting)
          if (response.status === 429) {
            // Rate limiting is disabled on backend - this should never happen
            console.warn(`⚠️ Unexpected 429 response (rate limiting is disabled on backend)`);
            throw new Error('Unexpected rate limit response. Please contact support.');
          }

          if (!response.ok) {
            // Try to get error details from response body (backend sends JSON ApiResponse)
            let errorMessage = `HTTP error! status: ${response.status}`;
            try {
              const contentType = response.headers.get('content-type');
              if (contentType && contentType.includes('application/json')) {
                const errorBody = await response.json();
                if (errorBody && errorBody.error) {
                  errorMessage = errorBody.error;
                } else if (errorBody && errorBody.message) {
                  errorMessage = errorBody.message;
                } else {
                  errorMessage = JSON.stringify(errorBody);
                }
              } else {
                const errorText = await response.text();
                if (errorText) {
                  // v3.0.7-beta: Filter out HTML error pages (like nginx 404/502)
                  // Don't display raw HTML to users
                  if (errorText.includes('<html') || errorText.includes('<!DOCTYPE') || errorText.includes('<body')) {
                    // v2.3.0: Provide user-friendly error messages based on status code
                    if (response.status === 502 || response.status === 503 || response.status === 504) {
                      errorMessage = 'The server is temporarily unavailable or processing heavy load. Please wait a moment and try again.';
                    } else if (response.status === 404) {
                      errorMessage = 'The requested resource was not found. Please refresh the page and try again.';
                    } else {
                      errorMessage = `Server error (${response.status}). Please try again in a few moments.`;
                    }
                    console.warn('⚠️ Received HTML error page:', errorText.substring(0, 200));
                  } else {
                    errorMessage = errorText;
                  }
                }
              }
            } catch (e) {
              console.warn('Failed to parse error response:', e);
            }
            throw new Error(errorMessage);
          }

          const result = await response.json();
          console.log(`✅ [API SUCCESS] ${options?.method || 'GET'} ${endpoint}`);
          return result;
        } catch (error) {
          // Check for browser resource exhaustion
          const isResourceError = error instanceof TypeError &&
            (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'));

          if (isResourceError) {
            console.error(`⚠️ [RESOURCE EXHAUSTION] Browser network resources exhausted. Backing off...`);
            // Force a longer backoff for resource errors
            const longBackoffMs = Math.min(2000 * Math.pow(2, attempt), 10000);
            await new Promise(resolve => setTimeout(resolve, longBackoffMs));
          }

          if (attempt === retries) {
            console.error(`❌ [API FAILED] ${options?.method || 'GET'} ${endpoint} after ${retries + 1} attempts:`, error);
            return {
              success: false,
              data: null,
              error: error instanceof Error ? error.message : 'Unknown error',
              timestamp: new Date().toISOString(),
            };
          }
        }
      }

      // Should never reach here, but TypeScript needs it
      return {
        success: false,
        data: null,
        error: 'Maximum retries exceeded',
        timestamp: new Date().toISOString(),
      };
    } finally {
      // Always release rate limit token
      rateLimiter.release();
      console.log(`📊 [RATE LIMITER] Active requests: ${rateLimiter.getActiveCount()}/10`);
    }
  }

  /**
   * Authenticated request - automatically signs with wallet private key
   * Requires wallet to be unlocked in session or uses stored mnemonic
   */
  private async authenticatedRequest<T>(
    endpoint: string,
    options?: RequestInit,
    passwordPrompt?: () => Promise<string>
  ): Promise<ApiResponse<T>> {
    try {
      // Check if wallet session is active
      let session = walletSession.getSession();
      let password: string | null = null; // Declare at function scope for AEGIS-QL key loading

      console.log('🔐 [AUTH DEBUG] authenticatedRequest called for endpoint:', endpoint);
      console.log('🔐 [AUTH DEBUG] Session exists:', !!session);
      console.log('🔐 [AUTH DEBUG] globalPasswordPrompt available:', !!globalPasswordPrompt);

      // If no active session, try to restore or decrypt wallet
      if (!session) {
        const encryptedKey = localStorage.getItem('walletEncryptedKey');
        const sessionTimeout = localStorage.getItem('walletSessionTimeout') || 'never';

        console.log('🔐 [AUTH DEBUG] No session, encrypted key exists:', !!encryptedKey);
        console.log('🔐 [AUTH DEBUG] Session timeout setting:', sessionTimeout);

        if (!encryptedKey) {
          // No encrypted wallet found
          console.error('🔐 [AUTH DEBUG] No encrypted wallet found');
          return {
            success: false,
            data: null,
            error: 'No encrypted wallet found. Please log in with your mnemonic phrase and password.',
            timestamp: new Date().toISOString(),
          };
        }

        // Check if "Never expire" is set - try to restore from stored session without password
        if (sessionTimeout === 'never') {
          try {
            // Try to get stored session data directly from sessionStorage
            const storedSession = sessionStorage.getItem('walletSession');
            if (storedSession) {
              const data = JSON.parse(storedSession);
              if (data.mnemonic) {
                console.log('🔐 [AUTH DEBUG] "Never expire" enabled - restoring session from stored mnemonic');
                const keyPair = await keypairFromMnemonic(data.mnemonic);
                walletSession.setSession(keyPair.privateKey, keyPair.address, data.mnemonic);
                session = { privateKey: keyPair.privateKey, address: keyPair.address, mnemonic: data.mnemonic };
                console.log('✅ [AUTH DEBUG] Session auto-restored for "Never expire" user');
              }
            }
          } catch (restoreError) {
            console.warn('🔐 [AUTH DEBUG] Failed to auto-restore session:', restoreError);
            // Fall through to password prompt
          }
        }

        // If still no session, need password
        if (!session) {
          console.log('🔐 [AUTH DEBUG] Will prompt for password:', !!globalPasswordPrompt);

          // Try using the provided passwordPrompt
          if (passwordPrompt) {
            try {
              password = await passwordPrompt();
            } catch (error) {
              return {
                success: false,
                data: null,
                error: 'Authentication cancelled by user',
                timestamp: new Date().toISOString(),
              };
            }
          }
          // Try using the global password prompt (from PasswordModalProvider)
          else if (globalPasswordPrompt) {
            try {
              password = await globalPasswordPrompt();
            } catch (error) {
              return {
                success: false,
                data: null,
                error: 'Authentication cancelled by user',
                timestamp: new Date().toISOString(),
              };
            }
          }
          // Fallback to browser prompt if no modal available
          else {
            password = prompt('Enter wallet password to sign request:');
          }

          if (!password) {
            return {
              success: false,
              data: null,
              error: 'Authentication required: Password not provided',
              timestamp: new Date().toISOString(),
            };
          }

          try {
            const wallet = await loadWallet(password);
            // For "never expire", also store the mnemonic for future auto-restore
            if (sessionTimeout === 'never') {
              const mnemonic = await recoverMnemonic(password);
              walletSession.setSession(wallet.privateKey, wallet.address, mnemonic);
            } else {
              walletSession.setSession(wallet.privateKey, wallet.address);
            }
            session = { privateKey: wallet.privateKey, address: wallet.address };
          } catch (error) {
            return {
              success: false,
              data: null,
              error: `Authentication failed: ${error instanceof Error ? error.message : 'Invalid password'}`,
              timestamp: new Date().toISOString(),
            };
          }
        }
      }

      // Generate authentication header
      // CRITICAL FIX: Sign the FULL path including baseURL prefix
      // The backend verifies parts.uri.path() which includes /api prefix from proxy
      // BUT EXCLUDES query parameters (parts.uri.path() strips ?limit=... etc.)
      // Example: endpoint="/v1/dex/swap?foo=bar" -> sign="/api/v1/dex/swap"
      let fullPath = `${this.baseURL}${endpoint}`.replace(window.location.origin, '');

      // Strip query parameters - backend only signs the path portion
      const queryIndex = fullPath.indexOf('?');
      if (queryIndex !== -1) {
        fullPath = fullPath.substring(0, queryIndex);
      }

      // Check if AEGIS-QL keys are available for AegisQLHybrid authentication
      const hasAegisKeys = !!(
        localStorage.getItem('walletEncryptedAegisKey') &&
        localStorage.getItem('walletAegisPublicKey')
      );

      let authHeader: string;
      if (hasAegisKeys && password) {
        // Load AEGIS-QL keys and use AegisQLHybrid authentication
        // ONLY if we already have the password from the initial session unlock
        try {
          const wallet = await loadWallet(password);
          if (wallet.aegisPublicKey && wallet.aegisPrivateKey) {
            authHeader = await generateAuthHeader(
              session.privateKey,
              session.address,
              fullPath,
              'AegisQLHybrid',
              {
                publicKey: wallet.aegisPublicKey,
                secretKey: wallet.aegisPrivateKey
              }
            );
            console.log('✅ Using AegisQLHybrid authentication (Ed25519 + AEGIS-QL)');
          } else {
            // Fall back to Ed25519 only
            authHeader = await generateAuthHeader(
              session.privateKey,
              session.address,
              fullPath
            );
            console.log('⚠️ AEGIS-QL keys not loaded, falling back to Ed25519');
          }
        } catch (error) {
          console.warn('❌ Failed to load AEGIS-QL keys, using Ed25519 only:', error);
          authHeader = await generateAuthHeader(
            session.privateKey,
            session.address,
            fullPath
          );
        }
      } else {
        // Use Ed25519 only (no AEGIS-QL keys or no password available from session unlock)
        authHeader = await generateAuthHeader(
          session.privateKey,
          session.address,
          fullPath
        );
        console.log('ℹ️ Using Ed25519 authentication (AEGIS-QL not available or session active)');
      }

      // Make authenticated request
      return await this.request<T>(endpoint, {
        ...options,
        headers: {
          'X-Wallet-Auth': authHeader,
          ...options?.headers,
        },
      });
    } catch (error) {
      return {
        success: false,
        data: null,
        error: `Authentication error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Generate quantum-enhanced BIP39 mnemonic
  async generateMnemonic(): Promise<ApiResponse<MnemonicResponse>> {
    return this.request<MnemonicResponse>('/v1/mnemonic');
  }

  // Get node status
  async getNodeStatus(): Promise<ApiResponse<NodeStatus>> {
    return this.request<NodeStatus>('/v1/node/status');
  }

  // Get network supply statistics (max supply, mined coins, hashrate)
  async getNetworkSupply(): Promise<ApiResponse<NetworkSupply>> {
    return this.request<NetworkSupply>('/v1/network/supply');
  }

  // v2.3.8-beta: Get QUGUSD stablecoin vault statistics (real CDP data)
  async getVaultStats(): Promise<ApiResponse<VaultStats>> {
    return this.request<VaultStats>('/v1/stablecoin/vault/stats');
  }

  // Get hashpower-weighted security metrics (v1.3.0-beta)
  async getHashpowerSecurity(): Promise<ApiResponse<HashpowerSecurityData>> {
    return this.request<HashpowerSecurityData>('/v1/security/hashpower');
  }

  // v1.4.15-beta: Get startup progress for DAG integrity check display
  async getStartupProgress(): Promise<ApiResponse<{
    phase: string;
    message: string;
    phase_progress: number;
    total_blocks: number;
    blocks_checked: number;
    is_ready: boolean;
    elapsed_seconds: number;
    current_height: number;
    network_height: number;
  }>> {
    return this.request<{
      phase: string;
      message: string;
      phase_progress: number;
      total_blocks: number;
      blocks_checked: number;
      is_ready: boolean;
      elapsed_seconds: number;
      current_height: number;
      network_height: number;
    }>('/v1/startup-progress');
  }

  // v1.4.12-beta: Get P2P network health status (for peer dropdown)
  async getP2PHealth(): Promise<ApiResponse<{
    connected_peers: number;
    network_height: number;
    current_height: number;
    sync_progress_percent: number;
    network_status: string;
    turbo_sync_available: boolean;
    bootstrap_peer_configured: boolean;
    gossipsub_topics: string[];
  }>> {
    return this.request<{
      connected_peers: number;
      network_height: number;
      current_height: number;
      sync_progress_percent: number;
      network_status: string;
      turbo_sync_available: boolean;
      bootstrap_peer_configured: boolean;
      gossipsub_topics: string[];
    }>('/v1/p2p/health');
  }

  // Create a new wallet (or import with mnemonic)
  async createWallet(mnemonic?: string, password?: string): Promise<ApiResponse<WalletData>> {
    const endpoint = mnemonic ? '/v1/wallets/import' : '/v1/wallets/create';
    return this.request<WalletData>(endpoint, {
      method: 'POST',
      body: JSON.stringify({
        mnemonic: mnemonic || undefined,
        password: password || undefined
      }),
    });
  }

  // List all wallets (PUBLIC - no auth required)
  async listWallets(): Promise<ApiResponse<WalletData[]>> {
    return this.request<WalletData[]>('/v1/wallets');
  }

  // Get specific wallet by ID (PUBLIC - no auth required)
  async getWallet(id: string): Promise<ApiResponse<WalletData>> {
    return this.request<WalletData>(`/v1/wallets/${id}`);
  }

  // Request test tokens from faucet
  async requestFaucet(walletAddress?: string): Promise<ApiResponse<any>> {
    const body = walletAddress ? { wallet_address: walletAddress } : {};
    console.log('🚰 Faucet request body:', body);
    return this.request<any>('/v1/faucet', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  // Get wallet balance by address (AUTHENTICATED - requires signature)
  async getWalletBalance(walletAddress?: string): Promise<ApiResponse<any>> {
    // Use stored wallet address if none provided
    const address = walletAddress || localStorage.getItem('walletAddress') || '';
    console.log('🔍 Fetching balance for wallet address:', address);
    return this.authenticatedRequest<any>(`/v1/wallets/${address}/balance`);
  }

  // Get multi-token balances (QUG + QUGUSD) (AUTHENTICATED - requires signature)
  async getMultiTokenBalance(): Promise<ApiResponse<any>> {
    console.log('🔍 [AUTHENTICATED] Fetching multi-token balance for wallet (address not in URL)');
    // Address is extracted from X-Wallet-Auth header on backend for privacy
    return this.authenticatedRequest<any>('/v1/wallet/tokens');
  }

  // Send a transaction
  async sendTransaction(from: string, to: string, amount: number, memo?: string, tokenType?: string): Promise<ApiResponse<any>> {
    // Use stored wallet address if none provided for 'from'
    const fromAddress = from || localStorage.getItem('walletAddress') || '';

    // Check if we have an active session first
    const session = walletSession.getSession();
    let mnemonic = '';

    if (session && session.mnemonic) {
      // Session has stored mnemonic (from "Never expire" setting)
      mnemonic = session.mnemonic;
      console.log('✅ Using mnemonic from active session (no password required)');
    } else if (session) {
      // Session is active but no stored mnemonic - need to decrypt
      const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');

      if (encryptedMnemonic) {
        try {
          // Use the SessionTimeoutContext to request password with modal
          const { getGlobalPasswordRequester } = await import('../contexts/SessionTimeoutContext');
          const passwordRequester = getGlobalPasswordRequester();

          if (!passwordRequester) {
            // Fallback to window.prompt if context not available
            const password = window.prompt('🔒 Enter your password to continue:');
            if (!password) {
              return {
                success: false,
                data: null,
                error: 'Password required to decrypt wallet. Transaction cancelled.',
                timestamp: new Date().toISOString(),
              };
            }

            // Manual recovery with window.prompt
            const { recoverMnemonic } = await import('./walletAuth');
            mnemonic = await recoverMnemonic(password);
            console.log('✅ Mnemonic recovered from encrypted storage (fallback)');
          } else {
            // Use modal to request password
            mnemonic = await passwordRequester();
            console.log('✅ Mnemonic recovered via modal');
          }
        } catch (error) {
          return {
            success: false,
            data: null,
            error: error instanceof Error ? error.message : 'Failed to decrypt wallet. Incorrect password.',
            timestamp: new Date().toISOString(),
          };
        }
      } else {
        // No encrypted mnemonic found
        return {
          success: false,
          data: null,
          error: 'Wallet seed not found. Please log in again with your mnemonic phrase.',
          timestamp: new Date().toISOString(),
        };
      }
    } else {
      // No active session - need to decrypt mnemonic with password
      const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');

      if (encryptedMnemonic) {
        try {
          // Use the SessionTimeoutContext to request password with modal
          const { getGlobalPasswordRequester } = await import('../contexts/SessionTimeoutContext');
          const passwordRequester = getGlobalPasswordRequester();

          if (!passwordRequester) {
            // Fallback to window.prompt if context not available
            const password = window.prompt('🔒 Session expired. Enter your password to continue:');
            if (!password) {
              return {
                success: false,
                data: null,
                error: 'Password required to decrypt wallet. Transaction cancelled.',
                timestamp: new Date().toISOString(),
              };
            }

            // Manual recovery with window.prompt
            const { recoverMnemonic, walletSession, keypairFromMnemonic } = await import('./walletAuth');
            mnemonic = await recoverMnemonic(password);
            const keyPair = await keypairFromMnemonic(mnemonic);
            walletSession.setSession(keyPair.privateKey, keyPair.address, mnemonic);
            console.log('✅ Mnemonic recovered and session created (fallback)');
          } else {
            // Use modal to request password
            // The SessionTimeoutContext handles decryption and session restoration internally
            mnemonic = await passwordRequester();
            console.log('✅ Mnemonic recovered via modal');
          }
        } catch (error) {
          return {
            success: false,
            data: null,
            error: error instanceof Error ? error.message : 'Failed to decrypt wallet. Incorrect password.',
            timestamp: new Date().toISOString(),
          };
        }
      } else {
        // No encrypted mnemonic found - user must log in again
        return {
          success: false,
          data: null,
          error: 'Wallet seed not found. Please log in again with your mnemonic phrase.',
          timestamp: new Date().toISOString(),
        };
      }
    }

    // Fix: Ensure amount is sent as QNK value, not converted to smallest units
    // If amount looks like it's been unit-converted (> 1,000,000), convert it back
    // QUG uses 9 decimals (1 QUG = 1,000,000,000 base units)
    let fixedAmount = amount;
    if (amount > 1000000) {
      console.warn(`⚠️ Detected unit conversion: ${amount} -> ${amount / 1000000000} QNK`);
      fixedAmount = amount / 1000000000;
    }

    console.log('📤 Sending transaction:', { from: fromAddress, to, amount: fixedAmount, memo });

    // Generate authentication header using Ed25519
    try {
      const { keypairFromMnemonic, generateAuthHeader, walletSession } = await import('./walletAuth');

      // Get or create session
      let activeSession = walletSession.getSession();
      if (!activeSession) {
        // If no session, create one from the decrypted mnemonic
        if (!mnemonic) {
          throw new Error('No active session and no mnemonic available');
        }
        console.log('🔐 Mnemonic found for Ed25519 signing:', mnemonic.split(' ').length, 'words');
        const keyPair = await keypairFromMnemonic(mnemonic);
        walletSession.setSession(keyPair.privateKey, keyPair.address);
        activeSession = { privateKey: keyPair.privateKey, address: keyPair.address };
      }

      // Use Ed25519 authentication for transaction
      // (AEGIS-QL support omitted to avoid asking for password again)
      const authHeader = await generateAuthHeader(
        activeSession.privateKey,
        activeSession.address,
        '/api/v1/transactions/send'
      );
      console.log('ℹ️ Using Ed25519 authentication for transaction');

      console.log('✅ Generated X-Wallet-Auth header for transaction');
      console.log('🔍 X-Wallet-Auth header length:', authHeader.length);
      console.log('🔍 X-Wallet-Auth header preview:', authHeader.substring(0, 100) + '...');

      // Send transaction with authentication header
      // Only include mnemonic if we just decrypted it (no session was active)
      const requestBody: any = {
        from: fromAddress,
        to: to,
        amount: fixedAmount,
        memo: memo,
        token_type: tokenType || 'QUG', // Default to QUG if not specified
      };

      // Only include mnemonic if we had to decrypt it
      if (mnemonic) {
        requestBody.mnemonic = mnemonic;
      }

      console.log('📤 Sending transaction with token_type:', requestBody.token_type);

      return this.request<any>('/v1/transactions/send', {
        method: 'POST',
        headers: {
          'X-Wallet-Auth': authHeader,
        },
        body: JSON.stringify(requestBody),
      });
    } catch (authError) {
      console.error('❌ Failed to generate authentication header:', authError);
      return {
        success: false,
        data: null,
        error: `Authentication error: ${authError instanceof Error ? authError.message : 'Unknown error'}`,
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<string>> {
    return this.request<string>('/v1/health');
  }

  // DAG-Knight consensus status
  async getDagKnightStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/v1/consensus/dag-knight');
  }

  // Narwhal consensus status  
  async getNarwhalStatus(): Promise<ApiResponse<any>> {
    return this.request<any>('/v1/consensus/narwhal');
  }

  // Get mempool DAG analysis
  async getMempoolDagAnalysis(): Promise<ApiResponse<any>> {
    return this.request<any>('/v1/mempool/dag-analysis');
  }

  // Get specific block by height
  async getBlock(height: number): Promise<ApiResponse<any>> {
    return this.request<any>(`/v1/blocks/${height}`);
  }

  // Get blocks in range (simulate by calling multiple block endpoints)
  async getBlockRange(startHeight: number, endHeight: number): Promise<ApiResponse<any[]>> {
    const blockPromises = [];
    for (let height = startHeight; height <= endHeight; height++) {
      blockPromises.push(this.getBlock(height));
    }
    
    try {
      const responses = await Promise.all(blockPromises);
      const blocks = responses
        .filter(response => response.success && response.data)
        .map((response, index) => ({
          height: startHeight + index,
          transactions: response.data,
          timestamp: new Date().toISOString(),
        }));
      
      return {
        success: true,
        data: blocks,
        error: null,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof Error ? error.message : 'Failed to fetch block range',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // ============================================
  // v3.4.0-beta: FEE ESTIMATION API
  // ============================================

  /**
   * Estimate transaction fees based on type and network conditions
   *
   * Fee reduction at block 350,000:
   * - Before: 0.00021 QUG (21,000 satoshis) for simple transfer
   * - After: 0.000021 QUG (2,100 satoshis) - 10x cheaper!
   *
   * @param txType - Transaction type: 'transfer', 'swap', 'token_transfer', 'contract_call', etc.
   * @param priority - Priority level: 'low', 'medium', 'high', 'urgent' (default: 'medium')
   */
  async estimateFee(txType: string = 'transfer', priority: string = 'medium'): Promise<ApiResponse<FeeEstimate>> {
    console.log('💰 Estimating fee for:', txType, 'priority:', priority);
    return this.request<FeeEstimate>('/v1/transactions/estimate-fee', {
      method: 'POST',
      body: JSON.stringify({ tx_type: txType, priority }),
    });
  }

  /**
   * Get current network height (for checking fee reduction activation)
   */
  async getNetworkHeight(): Promise<number> {
    try {
      const response = await this.request<any>('/v1/status');
      if (response.success && response.data) {
        return response.data.current_height || 0;
      }
      return 0;
    } catch (error) {
      console.warn('Failed to get network height:', error);
      return 0;
    }
  }

  /**
   * Check if reduced fees are active (after block 350,000)
   */
  async isReducedFeesActive(): Promise<boolean> {
    const height = await this.getNetworkHeight();
    return height >= FEE_REDUCTION_ACTIVATION_HEIGHT;
  }

  /**
   * Get current minimum fee in QUG based on network height
   */
  async getCurrentMinFee(): Promise<number> {
    const isReduced = await this.isReducedFeesActive();
    return isReduced ? NEW_MIN_FEE_QUG : CURRENT_MIN_FEE_QUG;
  }

  // Quantum Privacy Mixer API methods
  async sendPrivateTransaction(request: {
    to: string;
    amount: number;
    privacy_level: string;
    enable_quantum_mixing?: boolean;
    decoy_multiplier?: number;
    memo?: string;
    password?: string;
  }): Promise<ApiResponse<any>> {
    try {
      // Get wallet address from localStorage (most reliable source)
      let walletAddress = localStorage.getItem('walletAddress') || '';

      // Strip "qnk" prefix if present - backend expects 64-char hex
      if (walletAddress.startsWith('qnk')) {
        walletAddress = walletAddress.substring(3);
      }

      // Validate we have a wallet address
      if (!walletAddress || walletAddress.length !== 64) {
        console.error('❌ [MIXER] Invalid or missing wallet address:', {
          raw: localStorage.getItem('walletAddress'),
          processed: walletAddress,
          length: walletAddress.length
        });
        return {
          success: false,
          data: null,
          error: `Invalid wallet address format. Expected 64-char hex, got: ${walletAddress.length} chars. Please refresh the page.`,
          timestamp: new Date().toISOString(),
        };
      }

      console.log('✅ [MIXER] Sending private transaction with wallet address:', {
        from: walletAddress.substring(0, 8) + '...',
        to: request.to.substring(0, 8) + '...',
        amount: request.amount
      });

      // CRITICAL: Add 'from' field to request - backend uses this for balance check
      const requestWithFrom = {
        ...request,
        from: walletAddress  // Always include - never undefined
      };

      const response = await fetch(`${this.baseURL}/v1/mixer/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestWithFrom),
      });

      // Handle 404 or other non-JSON responses gracefully
      if (response.status === 404) {
        return {
          success: false,
          data: null,
          error: 'Mixer endpoint not available. Falling back to standard transaction.',
          timestamp: new Date().toISOString(),
        };
      }

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return {
          success: false,
          data: null,
          error: `Server returned non-JSON response (${response.status}): ${response.statusText}`,
          timestamp: new Date().toISOString(),
        };
      }

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof Error ? error.message : 'Failed to send private transaction',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getMixingStatus(mixingId: string): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/v1/mixer/status/${mixingId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      // Handle 404 or other non-JSON responses gracefully
      if (response.status === 404) {
        return {
          success: false,
          data: null,
          error: 'Mixer status endpoint not available.',
          timestamp: new Date().toISOString(),
        };
      }

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return {
          success: false,
          data: null,
          error: `Server returned non-JSON response (${response.status}): ${response.statusText}`,
          timestamp: new Date().toISOString(),
        };
      }

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof Error ? error.message : 'Failed to get mixing status',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getMixingPoolsStatus(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/v1/mixer/pools`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      // Handle 404 or other non-JSON responses gracefully
      if (response.status === 404) {
        return {
          success: false,
          data: null,
          error: 'Mixer pools endpoint not available.',
          timestamp: new Date().toISOString(),
        };
      }

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return {
          success: false,
          data: null,
          error: `Server returned non-JSON response (${response.status}): ${response.statusText}`,
          timestamp: new Date().toISOString(),
        };
      }

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof Error ? error.message : 'Failed to get mixing pools status',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async joinMixingPool(request: {
    amount: number;
    output_addresses: string[];
    privacy_level: string;
    decoy_count?: number;
    mixer_fee?: number;
  }): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/v1/mixer/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      // Handle 404 or other non-JSON responses gracefully
      if (response.status === 404) {
        return {
          success: false,
          data: null,
          error: 'Mixer join endpoint not available.',
          timestamp: new Date().toISOString(),
        };
      }

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        return {
          success: false,
          data: null,
          error: `Server returned non-JSON response (${response.status}): ${response.statusText}`,
          timestamp: new Date().toISOString(),
        };
      }

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof Error ? error.message : 'Failed to join mixing pool',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Get supported tokens for DEX
  async getSupportedTokens(): Promise<ApiResponse<any[]>> {
    return this.request<any[]>('/v1/dex/tokens');
  }

  // Get oracle price for a specific feed (e.g., QUG/USD, QUGUSD/USD, or token address)
  async getOraclePrice(feedId: string): Promise<ApiResponse<any>> {
    console.log('💰 Fetching oracle price for feed:', feedId);
    return this.request<any>(`/v1/defi/oracle/price/${encodeURIComponent(feedId)}`);
  }

  // Get all available oracle price feeds
  async getOracleFeeds(): Promise<ApiResponse<any[]>> {
    console.log('📊 Fetching all oracle price feeds');
    return this.request<any[]>('/v1/defi/oracle/feeds');
  }

  // v2.3.6-beta: Get real-time token price from AMM oracle (not hardcoded!)
  async getAMMPrice(token: string): Promise<ApiResponse<{
    token: string;
    price_usd: number;
    source: string;
    last_updated: number;
    pool_reserves?: {
      token0: string;
      token1: string;
      reserve0: number;
      reserve1: number;
      pool_id: string;
    };
  }>> {
    console.log('📊 Fetching AMM oracle price for:', token);
    return this.request(`/v1/oracle/price/${encodeURIComponent(token)}`);
  }

  // v2.3.6-beta: Get all token prices from AMM oracle
  async getAllAMMPrices(): Promise<ApiResponse<Array<{
    token: string;
    price_usd: number;
    source: string;
    last_updated: number;
    pool_reserves?: {
      token0: string;
      token1: string;
      reserve0: number;
      reserve1: number;
      pool_id: string;
    };
  }>>> {
    console.log('📊 Fetching all AMM oracle prices');
    return this.request('/v1/oracle/prices');
  }

  // Get recent transactions (filtered by wallet address for privacy)
  async getRecentTransactions(limit = 100): Promise<ApiResponse<any[]>> {
    // Get wallet address from localStorage for privacy-filtered results
    const walletAddress = localStorage.getItem('walletAddress') || '';
    console.log('🔍 Fetching transactions for wallet address:', walletAddress);

    // Use authenticated request with cryptographic signature
    console.log('📋 Fetching transactions WITH authentication (Ed25519 signature)');
    return await this.authenticatedRequest<any[]>(`/v1/transactions/recent?limit=${limit}&wallet_address=${walletAddress}`);
  }

  // Get contract/token information by address
  async getContractInfo(contractAddress: string): Promise<ApiResponse<any>> {
    console.log('🔍 Fetching contract info for address:', contractAddress);
    return this.request<any>(`/v1/contracts/${contractAddress}`);
  }

  // Get token balance for a specific address and token contract
  async getTokenBalance(walletAddress: string, tokenAddress: string): Promise<ApiResponse<any>> {
    console.log('🔍 Fetching token balance:', { walletAddress, tokenAddress });
    return this.request<any>(`/v1/contracts/${tokenAddress}/balance/${walletAddress}`);
  }

  // Add liquidity to a pool
  // v3.2.14-beta: Support string amounts for u128 precision (BigInt values converted to strings)
  // v3.2.15-beta: Use authenticatedRequest for proper wallet authentication
  async addLiquidity(request: {
    token0: string;
    token1: string;
    amount0: number | string | bigint;
    amount1: number | string | bigint;
    provider: string;
  }): Promise<ApiResponse<any>> {
    // Convert BigInt to string for JSON serialization
    const serializedRequest = {
      ...request,
      amount0: typeof request.amount0 === 'bigint' ? request.amount0.toString() : request.amount0,
      amount1: typeof request.amount1 === 'bigint' ? request.amount1.toString() : request.amount1,
    };
    console.log('💧 Adding liquidity (authenticated):', serializedRequest);
    return this.authenticatedRequest<any>('/v1/liquidity/add', {
      method: 'POST',
      body: JSON.stringify(serializedRequest),
    });
  }

  // Get all liquidity pools
  async getLiquidityPools(): Promise<ApiResponse<any[]>> {
    console.log('🔍 Fetching all liquidity pools');
    return this.request<any[]>('/v1/liquidity/pools');
  }

  // Get specific pool info
  async getPoolInfo(poolId: string): Promise<ApiResponse<any>> {
    console.log('🔍 Fetching pool info for:', poolId);
    return this.request<any>(`/v1/liquidity/pools/${poolId}`);
  }

  // Remove liquidity from a pool
  // v3.2.15-beta: Use authenticatedRequest for proper wallet authentication
  async removeLiquidity(request: {
    pool_id: string;
    percentage: number;
    provider: string;
  }): Promise<ApiResponse<any>> {
    console.log('💧 Removing liquidity (authenticated):', request);
    return this.authenticatedRequest<any>('/v1/liquidity/remove', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Get all Nitro boosts (aggregated by token)
  async getNitroBoosts(): Promise<ApiResponse<Record<string, number>>> {
    console.log('🚀 Fetching Nitro boosts');
    return this.request<Record<string, number>>('/v1/nitro/boosts');
  }

  // Add Nitro boost to a token
  async addNitroBoost(tokenId: string, points: number, walletAddress: string): Promise<ApiResponse<any>> {
    console.log('🚀 Adding Nitro boost:', { tokenId, points, walletAddress });
    return this.request<any>('/v1/nitro/boost', {
      method: 'POST',
      body: JSON.stringify({
        token_id: tokenId,
        points,
        wallet_address: walletAddress
      }),
    });
  }

  // Execute token swap through liquidity pools (AUTHENTICATED)
  async executeSwap(request: {
    from_token: string;
    to_token: string;
    amount_in: number;
    min_amount_out: number;
    wallet_address: string;
  }): Promise<ApiResponse<any>> {
    console.log('💱 Executing swap (authenticated):', request);
    return this.authenticatedRequest<any>('/v1/dex/swap', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Get token price history for charts
  async getTokenPriceHistory(tokenId: string, timeframe: string): Promise<ApiResponse<any[]>> {
    console.log('📈 Fetching price history for:', tokenId, timeframe);
    return this.request<any[]>(`/v1/oracle/price-history/${encodeURIComponent(tokenId)}?timeframe=${timeframe}`);
  }

  // Get token transactions
  async getTokenTransactions(tokenId: string): Promise<ApiResponse<any[]>> {
    console.log('📜 Fetching transactions for token:', tokenId);
    return this.request<any[]>(`/v1/transactions/token/${encodeURIComponent(tokenId)}`);
  }

  // ============================================
  // EXPLORER API ENDPOINTS
  // ============================================

  // Get comprehensive network statistics
  async getNetworkStatistics(): Promise<ApiResponse<any>> {
    console.log('📊 Fetching network statistics');
    return this.request<any>('/v1/statistics/network');
  }

  // Get recent blocks with metadata
  async getRecentBlocks(limit = 10): Promise<ApiResponse<any[]>> {
    console.log('🧱 Fetching recent blocks, limit:', limit);
    return this.request<any[]>(`/v1/blocks/recent?limit=${limit}`);
  }

  // Get recent smart contract deployments
  async getRecentContracts(limit = 10): Promise<ApiResponse<any[]>> {
    console.log('📜 Fetching recent contracts, limit:', limit);
    return this.request<any[]>(`/v1/contracts/recent?limit=${limit}`);
  }

  // Get recent DAG vertices
  async getRecentVertices(limit = 10): Promise<ApiResponse<any[]>> {
    console.log('⚛️ Fetching recent vertices, limit:', limit);
    return this.request<any[]>(`/v1/dag/vertices/recent?limit=${limit}`);
  }

  // Get recent transactions for Explorer (no auth required - all network transactions)
  async getExplorerTransactions(limit = 10): Promise<ApiResponse<any[]>> {
    console.log('📊 Fetching recent Explorer transactions, limit:', limit);
    return this.request<any[]>(`/v1/transactions/explorer?limit=${limit}`);
  }

  // Universal search (blocks, transactions, wallets, contracts)
  async universalSearch(query: string): Promise<ApiResponse<any[]>> {
    console.log('🔍 Universal search for:', query);
    return this.request<any[]>(`/v1/search?query=${encodeURIComponent(query)}`);
  }

  // Get user's deployed contracts
  async getUserContracts(walletAddress: string): Promise<ApiResponse<any[]>> {
    console.log('🔍 Fetching deployed contracts for wallet:', walletAddress);
    return this.request<any[]>(`/v1/contracts/user/${walletAddress}/contracts`);
  }

  // Mint tokens (for contracts that support minting)
  async mintTokens(contractAddress: string, amount: string): Promise<ApiResponse<any>> {
    console.log('🪙 Minting tokens:', { contractAddress, amount });
    return this.request<any>('/v1/contracts/mint', {
      method: 'POST',
      body: JSON.stringify({
        contract_address: contractAddress,
        amount: amount
      }),
    });
  }

  // Mint QUGUSD (QNKUSD) stablecoin with collateral
  async mintQUGUSD(request: {
    amount: number;
    collateral_type: string;
    collateral_amount: number;
    reason?: string;
  }): Promise<ApiResponse<any>> {
    console.log('💵 Minting QUGUSD with collateral:', request);

    // Backend expects amount as u64 in base units (smallest denomination)
    // Convert from human-readable decimal to base units by multiplying by 100,000,000
    const amountBaseUnits = Math.floor(request.amount * 100000000);

    console.log(`💵 Converting amount: ${request.amount} QUGUSD → ${amountBaseUnits} base units`);

    // ✅ CRITICAL FIX: Get wallet address from localStorage and send it to backend
    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) {
      return {
        success: false,
        data: null,
        error: 'No wallet address found. Please create or import a wallet first.',
        timestamp: new Date().toISOString(),
      };
    }

    console.log('👤 Minting QUGUSD for wallet:', walletAddress);

    return this.request<any>('/v1/quillon-bank/stablecoin/mint', {
      method: 'POST',
      body: JSON.stringify({
        ...request,
        amount: amountBaseUnits,  // Send as integer in base units
        wallet_address: walletAddress,  // ✅ Send wallet address to backend
      }),
    });
  }

  // Burn QUGUSD to release collateral
  async burnQUGUSD(request: {
    amount: number;
    recipient: string;
    collateral_type: string;
  }): Promise<ApiResponse<any>> {
    console.log('🔥 Burning QUGUSD to release collateral:', request);
    return this.request<any>('/v1/quillon-bank/stablecoin/burn', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Get stablecoin status
  async getStablecoinStatus(): Promise<ApiResponse<any>> {
    console.log('📊 Fetching stablecoin status');
    return this.request<any>('/v1/quillon-bank/stablecoin/status');
  }

  // Get collateral status
  async getCollateralStatus(): Promise<ApiResponse<any>> {
    console.log('📊 Fetching collateral status');
    return this.request<any>('/v1/quillon-bank/stablecoin/collateral');
  }

  /**
   * Subscribe to real-time mining rewards via SSE
   * @param walletAddress - Miner's wallet address to filter events
   * @param onReward - Callback for mining reward events
   * @param onBalanceUpdate - Callback for balance updates
   * @param onMiningStats - Callback for mining statistics updates
   * @returns EventSource instance (call .close() to unsubscribe)
   */
  subscribeToMiningRewards(
    walletAddress: string,
    onReward: (event: MiningRewardEvent) => void,
    onBalanceUpdate: (event: BalanceUpdateEvent) => void,
    onMiningStats?: (event: MiningStatsEvent) => void
  ): EventSource {
    // Connect to SSE endpoint with wallet_address parameter for filtered events
    const url = `${this.baseURL}/v1/events?wallet_address=${encodeURIComponent(walletAddress)}`;
    console.log('🔌 SSE: Connecting to', url);
    console.log('🔌 SSE: Filtering for wallet:', walletAddress);

    const eventSource = new EventSource(url);

    eventSource.onopen = () => {
      console.log('✅ SSE: Connection opened successfully');
    };

    // v3.3.10-beta: Helper to normalize addresses for comparison (handle qnk prefix and length differences)
    const normalizeAddress = (addr: string): string => {
      // Remove qnk prefix if present, then take first 40 chars for comparison
      const withoutPrefix = addr.startsWith('qnk') ? addr.slice(3) : addr;
      return withoutPrefix.slice(0, 40).toLowerCase();
    };
    const normalizedWallet = normalizeAddress(walletAddress);

    eventSource.addEventListener('mining_reward', (e: MessageEvent) => {
      console.log('📨 SSE: Received mining_reward event');
      try {
        const data = JSON.parse(e.data);
        console.log('📨 SSE: mining_reward data:', data);
        const normalizedReceived = normalizeAddress(data.miner_address || '');
        const addressMatch = normalizedReceived === normalizedWallet;
        console.log('📨 SSE: Comparing addresses:', { received: data.miner_address, expected: walletAddress, normalizedReceived, normalizedWallet, match: addressMatch });
        if (addressMatch) {
          console.log('✅ SSE: Address matches! Calling onReward callback');
          onReward(data);
        } else {
          console.log('❌ SSE: Address mismatch, ignoring event');
        }
      } catch (error) {
        console.error('❌ SSE: Failed to parse mining_reward event:', error);
      }
    });

    // v1.1.9-beta: FIX - Listen for 'balance-updated' (hyphen) not 'balance_updated' (underscore)
    // Backend sends: "balance-updated" (streaming.rs:765)
    eventSource.addEventListener('balance-updated', (e: MessageEvent) => {
      console.log('📨 SSE: Received balance_updated event - RAW DATA:', e.data);
      try {
        const parsed = JSON.parse(e.data);
        console.log('📨 SSE: balance_updated parsed:', parsed);

        // v2.7.4-beta FIX: Handle wrapped format {"type":"BalanceUpdated","data":{...}}
        // Backend sends events with serde tag format - extract actual data
        const data = parsed.data || parsed;

        // v2.7.8-beta: EXTENSIVE DEBUGGING for P2P balance propagation
        console.log('🔍 [DEBUG] balance_updated full data:', JSON.stringify(data, null, 2));
        console.log('🔍 [DEBUG] Address comparison:', {
          received: data.wallet_address,
          receivedLength: data.wallet_address?.length,
          expected: walletAddress,
          expectedLength: walletAddress?.length,
          exactMatch: data.wallet_address === walletAddress,
          receivedFirst16: data.wallet_address?.substring(0, 16),
          expectedFirst16: walletAddress?.substring(0, 16),
        });
        console.log('🔍 [DEBUG] Change reason:', data.change_reason);
        console.log('🔍 [DEBUG] Balance values:', { old: data.old_balance, new: data.new_balance, diff: data.new_balance - data.old_balance });

        // Backend now sends addresses WITH "qnk" prefix - compare directly
        // Accept mining_reward, mining_reward_instant, mining_reward_batch_X, p2p_mining_reward, pending_mining_reward, and development_fee reasons
        const isMiningReward = data.change_reason === 'mining_reward' ||
                               data.change_reason === 'mining_reward_instant' ||
                               data.change_reason === 'p2p_mining_reward' ||  // v1.1.9-beta: P2P mining rewards from other nodes
                               data.change_reason === 'pending_mining_reward' ||  // v2.7.6-beta: Pending rewards via P2P gossipsub
                               (data.change_reason && data.change_reason.startsWith('mining_reward_batch_'));
        const isDevFee = data.change_reason === 'development_fee';

        console.log('🔍 [DEBUG] Filter results:', { isMiningReward, isDevFee, addressMatch: data.wallet_address === walletAddress });

        if (data.wallet_address === walletAddress && (isMiningReward || isDevFee)) {
          console.log('✅ SSE: Address matches and reason is mining-related! Calling onBalanceUpdate callback');
          console.log('✅ [DEBUG] CALLING onBalanceUpdate with:', data);
          onBalanceUpdate(data);
        } else {
          console.log('❌ SSE: Address mismatch or wrong reason, ignoring event');
          console.log('❌ [DEBUG] REJECTED because:', {
            addressMatch: data.wallet_address === walletAddress,
            isMiningReward,
            isDevFee,
            reason: data.change_reason
          });
        }
      } catch (error) {
        console.error('❌ SSE: Failed to parse balance_updated event:', error);
      }
    });

    eventSource.addEventListener('mining_stats', (e: MessageEvent) => {
      console.log('📨 SSE: Received mining_stats event');
      try {
        const parsed = JSON.parse(e.data);
        console.log('📨 SSE: mining_stats parsed:', parsed);

        // Extract the actual data (handle both wrapped and unwrapped formats)
        const statsData = parsed.data || parsed;
        console.log('📨 SSE: mining_stats data:', statsData);

        // v3.3.10-beta: Use shared normalizeAddress helper
        const normalizedReceived = normalizeAddress(statsData.miner_address || '');
        const addressMatch = normalizedReceived === normalizedWallet;

        console.log('📨 SSE: Comparing addresses:', {
          received: statsData.miner_address,
          expected: walletAddress,
          normalizedReceived,
          normalizedWallet,
          match: addressMatch
        });

        if (addressMatch && onMiningStats) {
          console.log('✅ SSE: Address matches! Calling onMiningStats callback');
          onMiningStats(statsData);
        } else {
          console.log('❌ SSE: Address mismatch or no callback, ignoring event');
        }
      } catch (error) {
        console.error('❌ SSE: Failed to parse mining_stats event:', error);
      }
    });

    // v1.3.9-beta: Listen for pending_mining_reward events from P2P gossipsub
    // This enables instant balance updates when mining to localhost but frontend connected to bootstrap
    // The bootstrap receives miner stats via P2P and emits PendingMiningReward SSE events
    eventSource.addEventListener('pending_mining_reward', (e: MessageEvent) => {
      console.log('📨 SSE: Received pending_mining_reward event (P2P propagated)');
      try {
        const parsed = JSON.parse(e.data);
        console.log('📨 SSE: pending_mining_reward parsed:', parsed);

        // v2.7.4-beta FIX: Handle wrapped format {"type":"PendingMiningReward","data":{...}}
        // Backend sends events with serde tag format - extract actual data
        const data = parsed.data || parsed;
        console.log('📨 SSE: pending_mining_reward data:', data);
        // v3.3.10-beta: Use shared normalizeAddress helper for comparison
        const normalizedReceived = normalizeAddress(data.miner_address || '');
        const addressMatch = normalizedReceived === normalizedWallet;
        console.log('📨 SSE: Comparing addresses:', { received: data.miner_address, expected: walletAddress, normalizedReceived, normalizedWallet, match: addressMatch });

        if (addressMatch) {
          console.log('✅ SSE: Address matches! Processing pending mining reward');
          // Convert to balance update event format for the callback
          // v2.7.7-beta FIX: Backend sends values in QNK, NOT base units!
          // DO NOT multiply - the frontend expects QNK values directly
          const balanceUpdateEvent: BalanceUpdateEvent = {
            wallet_address: data.miner_address,
            old_balance: 0, // Pending reward is incremental, old_balance=0 so new_balance-old_balance gives the reward amount
            new_balance: data.pending_reward_qnk, // Already in QNK - DO NOT multiply!
            change_reason: 'pending_mining_reward',
            timestamp: data.timestamp || new Date().toISOString()
          };
          console.log('💎 SSE: Emitting pending reward as balance update:', balanceUpdateEvent);
          onBalanceUpdate(balanceUpdateEvent);
        } else {
          console.log('❌ SSE: Address mismatch, ignoring pending_mining_reward event');
        }
      } catch (error) {
        console.error('❌ SSE: Failed to parse pending_mining_reward event:', error);
      }
    });

    eventSource.onerror = (error) => {
      console.error('❌ SSE: Connection error:', error);
      console.error('❌ SSE: ReadyState:', eventSource.readyState);
    };

    return eventSource;
  }

  // ============================================
  // ADDRESS BOOK API - ZK-STARK/SNARK VERIFIED
  // ============================================

  /**
   * Get all saved addresses from address book
   */
  async getAddressBook(): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>('/v1/addressbook');
  }

  /**
   * Save a new address to address book with optional ZK proof
   */
  async saveAddress(addressData: any): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>('/v1/addressbook', {
      method: 'POST',
      body: JSON.stringify(addressData)
    });
  }

  /**
   * Update an existing address in address book
   */
  async updateAddress(addressId: string, addressData: any): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>(`/v1/addressbook/${addressId}`, {
      method: 'PUT',
      body: JSON.stringify(addressData)
    });
  }

  /**
   * Delete an address from address book
   */
  async deleteAddress(addressId: string): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>(`/v1/addressbook/${addressId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Generate ZK-STARK or ZK-SNARK proof for address verification
   * @param address - Wallet address to verify
   * @param proofType - 'stark' or 'snark'
   */
  async generateAddressProof(address: string, proofType: 'stark' | 'snark' = 'stark'): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>('/v1/addressbook/proof', {
      method: 'POST',
      body: JSON.stringify({ address, proof_type: proofType })
    });
  }

  /**
   * Verify a ZK proof for an address
   */
  async verifyAddressProof(address: string, proof: any): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>('/v1/addressbook/verify', {
      method: 'POST',
      body: JSON.stringify({ address, proof })
    });
  }

  /**
   * Get address book sync status via gossipsub
   */
  async getAddressBookSyncStatus(): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>('/v1/addressbook/sync/status');
  }

  /**
   * Force sync address book via gossipsub P2P network
   */
  async syncAddressBookGossipsub(): Promise<ApiResponse<any>> {
    return this.authenticatedRequest<any>('/v1/addressbook/sync', {
      method: 'POST'
    });
  }

  // ============================================
  // QNO STAKING ENDPOINTS
  // ============================================

  /**
   * Stake QUG for QNO prediction rewards
   */
  async stakePrediction(params: {
    domain: string;
    amount: number;
    confidence: number;  // Frontend sends 10-100
    lockDays: number;
    walletAddress: string;
    predictionValue: number;  // v1.4.3: User's predicted value for resolution
  }): Promise<ApiResponse<StakingResponse>> {
    return this.authenticatedRequest<StakingResponse>('/v1/qno/stake', {
      method: 'POST',
      body: JSON.stringify({
        domain: params.domain,
        amount: params.amount.toString(),  // Backend expects string
        confidence: params.confidence / 100,  // Convert 10-100 to 0.1-1.0
        lock_days: params.lockDays,
        prediction_value: params.predictionValue  // v1.4.3: Captured for oracle resolution
      })
    });
  }

  /**
   * Get active staking positions for authenticated wallet
   */
  async getStakingPositions(_walletAddress?: string): Promise<ApiResponse<StakingPosition[]>> {
    // Backend uses authentication, not wallet address in URL
    return this.authenticatedRequest<StakingPosition[]>('/v1/qno/stakes');
  }

  /**
   * Claim staking rewards for a completed prediction
   */
  async claimStakingReward(stakeId: string): Promise<ApiResponse<ClaimResponse>> {
    return this.authenticatedRequest<ClaimResponse>(`/v1/qno/stakes/${stakeId}/claim`, {
      method: 'POST'
    });
  }

  /**
   * Get QNO prediction domains and their current stats
   */
  async getPredictionDomains(): Promise<ApiResponse<PredictionDomain[]>> {
    return this.request<PredictionDomain[]>('/v1/qno/domains');
  }

  /**
   * Get QNO staking statistics
   */
  async getStakingStats(): Promise<ApiResponse<StakingStats>> {
    return this.request<StakingStats>('/v1/qno/stats');
  }

  /**
   * Get resolution history for a domain (v1.4.3)
   */
  async getResolutionHistory(domain: string): Promise<ApiResponse<ResolutionResult[]>> {
    return this.request<ResolutionResult[]>(`/v1/qno/domains/${domain}/resolutions`);
  }

  /**
   * Get oracle data for a domain (v1.4.3)
   */
  async getOracleData(domain: string): Promise<ApiResponse<OracleData>> {
    return this.request<OracleData>(`/v1/qno/domains/${domain}/oracle`);
  }

  /**
   * Get resolution configuration (v1.4.3)
   */
  async getResolutionConfig(): Promise<ApiResponse<ResolutionConfig>> {
    return this.request<ResolutionConfig>('/v1/qno/resolution-config');
  }
}

// QNO Staking interfaces
export interface StakingResponse {
  stake_id: string;
  domain: string;
  amount: number;
  confidence: number;
  lock_end: number;
  predicted_value?: number;
  status: 'active' | 'pending' | 'claimable';
  transaction_hash: string;
}

export interface StakingPosition {
  id: string;
  domain: string;
  amount: number;
  confidence: number;
  lock_end: number;
  predicted_value: number;
  status: 'active' | 'pending' | 'claimable';
  reward: number;
  created_at: number;
}

export interface ClaimResponse {
  success: boolean;
  reward_amount: number;
  transaction_hash: string;
}

export interface PredictionDomain {
  id: string;
  name: string;
  description: string;
  apy: number;
  risk_level: 'low' | 'medium' | 'high';
  total_staked: number;
  accuracy: number;
  active_predictions: number;
}

export interface StakingStats {
  total_staked: number;
  total_rewards_distributed: number;
  active_stakers: number;
  average_apy: number;
  total_predictions: number;
  successful_predictions: number;
}

// v1.4.3: Resolution and Oracle interfaces
export interface ResolutionResult {
  stake_id: string;
  domain: string;
  predicted_value: number;
  actual_value: number;
  accuracy_score: number;  // 0.0-1.0
  reward_adjustment: number;  // + or - QUG
  slashing_applied: number;
  resolved_at: number;
  is_accurate: boolean;
}

export interface OracleData {
  domain: string;
  value: number;
  confidence: number;
  timestamp: number;
  sources: OracleSource[];
}

export interface OracleSource {
  provider: string;
  value: number;
  confidence: number;
  timestamp: number;
}

export interface ResolutionConfig {
  accuracy_threshold: number;  // As percentage (0-100)
  slash_after_failures: number;
  base_slash_percentage: number;
  max_slash_percentage: number;
  accuracy_bonus_multiplier: number;
  inaccuracy_penalty_multiplier: number;
}

// Mining reward event interfaces
export interface MiningRewardEvent {
  miner_address: string;
  reward_qnk: number;
  nonce: number;
  block_height: number;
  difficulty: string;
  hash_rate: number;
  miner_id?: string; // v3.3.3-beta: Unique miner instance ID for identification
  worker_name?: string; // v0.6.2-beta: Human-readable miner name (e.g., "Server Alpha", "Mining Rig 1")
  origin_node_id?: string; // v2.3.5-beta: Which node mined this reward (peer ID)
  origin_node_name?: string; // v2.3.5-beta: Human-friendly node name (e.g., "Bootstrap", "Alpha")
  timestamp: string;
}

export interface BalanceUpdateEvent {
  wallet_address: string;
  old_balance: number;
  new_balance: number;
  change_reason: string;
  timestamp: string;
}

export interface MiningStatsEvent {
  miner_address: string;
  total_rewards: number;
  total_blocks_found: number;
  current_balance: number;
  avg_hash_rate: number;
  // v3.2.25-beta: Added for multi-miner tracking
  miner_id?: string;
  worker_id?: string;
  timestamp: string;
}

// Export singleton instance
export const qnkAPI = new QNarwhalKnightAPI();

// Export for custom configurations
export { QNarwhalKnightAPI };

// Export throttling utilities for use in components
export { throttle, debounce };

// v1.0.53: Re-export node discovery functions for direct use
export {
  discoverNode,
  clearDiscoveryCache,
  getDiscoveredNodeUrl,
  getDiscoveredPort,
  initializeNodeConnection,
  testConnection,
  onNodeDiscovered,
} from './nodeDiscovery';
export type { NodeDiscoveryConfig, NodeDiscoveryResult } from './nodeDiscovery';

// THROTTLED API METHODS FOR POLLING PROTECTION
// ============================================

/**
 * Throttled wrapper for frequently called API methods
 * Prevents excessive polling by limiting calls to once per interval
 */
export class ThrottledAPI {
  // Throttle node status to max once per 500ms (still allows fast updates)
  static getNodeStatus = throttle(() => qnkAPI.getNodeStatus(), 500);

  // Throttle transaction fetching to max once per 800ms
  static getRecentTransactions = throttle((limit?: number) => qnkAPI.getRecentTransactions(limit), 800);

  // Throttle balance fetching to max once per 500ms
  static getWalletBalance = throttle((address?: string) => qnkAPI.getWalletBalance(address), 500);

  // Throttle multi-token balance to max once per 500ms
  static getMultiTokenBalance = throttle(() => qnkAPI.getMultiTokenBalance(), 500);
}

/**
 * Debounced API wrapper for user-triggered actions
 * Delays execution until user stops triggering the action
 */
export class DebouncedAPI {
  // Debounce search with 500ms delay
  static universalSearch = debounce((query: string) => qnkAPI.universalSearch(query), 500);

  // Debounce token price history with 1s delay
  static getTokenPriceHistory = debounce(
    (tokenId: string, timeframe: string) => qnkAPI.getTokenPriceHistory(tokenId, timeframe),
    1000
  );
}