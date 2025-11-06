// Q-NarwhalKnight API Service
// Handles all communication with the quantum consensus node

import { generateAuthHeader, walletSession, loadWallet } from './walletAuth';

// Get API base URL from localStorage (set by network selector) or use default
const getApiBaseUrl = () => {
  const storedBaseURL = localStorage.getItem('apiBaseURL');
  if (storedBaseURL) {
    return storedBaseURL + '/api';
  }
  return import.meta.env.VITE_API_URL || '/api';
};

const API_BASE_URL = getApiBaseUrl();

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
  timestamp: string;
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
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
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
                  errorMessage = errorText;
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

      // If no active session, try to decrypt wallet with password
      if (!session) {
        const encryptedKey = localStorage.getItem('walletEncryptedKey');

        console.log('🔐 [AUTH DEBUG] No session, encrypted key exists:', !!encryptedKey);
        console.log('🔐 [AUTH DEBUG] Will prompt for password:', !!encryptedKey && !!globalPasswordPrompt);

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

        // Wallet is encrypted - need password

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
          walletSession.setSession(wallet.privateKey, wallet.address);
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
    let fixedAmount = amount;
    if (amount > 1000000) {
      console.warn(`⚠️ Detected unit conversion: ${amount} -> ${amount / 100000000} QNK`);
      fixedAmount = amount / 100000000;
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
  async addLiquidity(request: {
    token0: string;
    token1: string;
    amount0: number;
    amount1: number;
    provider: string;
  }): Promise<ApiResponse<any>> {
    console.log('💧 Adding liquidity:', request);
    return this.request<any>('/v1/liquidity/add', {
      method: 'POST',
      body: JSON.stringify(request),
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
  async removeLiquidity(request: {
    pool_id: string;
    percentage: number;
    provider: string;
  }): Promise<ApiResponse<any>> {
    console.log('💧 Removing liquidity:', request);
    return this.request<any>('/v1/liquidity/remove', {
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
   * @returns EventSource instance (call .close() to unsubscribe)
   */
  subscribeToMiningRewards(
    walletAddress: string,
    onReward: (event: MiningRewardEvent) => void,
    onBalanceUpdate: (event: BalanceUpdateEvent) => void
  ): EventSource {
    // Connect to SSE endpoint with wallet_address parameter for filtered events
    const url = `${this.baseURL}/v1/events?wallet_address=${encodeURIComponent(walletAddress)}`;
    console.log('🔌 SSE: Connecting to', url);
    console.log('🔌 SSE: Filtering for wallet:', walletAddress);

    const eventSource = new EventSource(url);

    eventSource.onopen = () => {
      console.log('✅ SSE: Connection opened successfully');
    };

    eventSource.addEventListener('mining_reward', (e: MessageEvent) => {
      console.log('📨 SSE: Received mining_reward event');
      try {
        const data = JSON.parse(e.data);
        console.log('📨 SSE: mining_reward data:', data);
        console.log('📨 SSE: Comparing addresses:', { received: data.miner_address, expected: walletAddress, match: data.miner_address === walletAddress });
        if (data.miner_address === walletAddress) {
          console.log('✅ SSE: Address matches! Calling onReward callback');
          onReward(data);
        } else {
          console.log('❌ SSE: Address mismatch, ignoring event');
        }
      } catch (error) {
        console.error('❌ SSE: Failed to parse mining_reward event:', error);
      }
    });

    eventSource.addEventListener('balance_updated', (e: MessageEvent) => {
      console.log('📨 SSE: Received balance_updated event');
      try {
        const data = JSON.parse(e.data);
        console.log('📨 SSE: balance_updated data:', data);
        console.log('📨 SSE: Comparing addresses:', { received: data.wallet_address, expected: walletAddress, match: data.wallet_address === walletAddress });
        console.log('📨 SSE: Change reason:', data.change_reason);
        // Backend now sends addresses WITH "qnk" prefix - compare directly
        if (data.wallet_address === walletAddress && data.change_reason === 'mining_reward') {
          console.log('✅ SSE: Address matches and reason is mining_reward! Calling onBalanceUpdate callback');
          onBalanceUpdate(data);
        } else {
          console.log('❌ SSE: Address mismatch or wrong reason, ignoring event');
        }
      } catch (error) {
        console.error('❌ SSE: Failed to parse balance_updated event:', error);
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
}

// Mining reward event interfaces
export interface MiningRewardEvent {
  miner_address: string;
  reward_qnk: number;
  nonce: number;
  block_height: number;
  difficulty: string;
  hash_rate: number;
  timestamp: string;
}

export interface BalanceUpdateEvent {
  wallet_address: string;
  old_balance: number;
  new_balance: number;
  change_reason: string;
  timestamp: string;
}

// Export singleton instance
export const qnkAPI = new QNarwhalKnightAPI();

// Export for custom configurations
export { QNarwhalKnightAPI };

// Export throttling utilities for use in components
export { throttle, debounce };

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