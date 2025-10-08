// Q-NarwhalKnight API Service
// Handles all communication with the quantum consensus node

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

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

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const response = await fetch(url, {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
          },
        });

        // Handle rate limiting with exponential backoff
        if (response.status === 429) {
          if (attempt < retries) {
            const retryAfter = response.headers.get('Retry-After');
            const delay = retryAfter ? parseInt(retryAfter) * 1000 : Math.pow(2, attempt) * 1000;
            console.warn(`Rate limited. Retrying after ${delay}ms (attempt ${attempt + 1}/${retries})`);
            await new Promise(resolve => setTimeout(resolve, delay));
            continue;
          }
          throw new Error('Rate limit exceeded. Please try again later.');
        }

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
      } catch (error) {
        if (attempt === retries) {
          console.error('API request failed after retries:', error);
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
  }

  // Generate quantum-enhanced BIP39 mnemonic
  async generateMnemonic(): Promise<ApiResponse<MnemonicResponse>> {
    return this.request<MnemonicResponse>('/v1/mnemonic');
  }

  // Get node status
  async getNodeStatus(): Promise<ApiResponse<NodeStatus>> {
    return this.request<NodeStatus>('/v1/node/status');
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

  // List all wallets
  async listWallets(): Promise<ApiResponse<WalletData[]>> {
    return this.request<WalletData[]>('/v1/wallets');
  }

  // Get specific wallet by ID
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

  // Get wallet balance by address
  async getWalletBalance(walletAddress?: string): Promise<ApiResponse<any>> {
    // Use stored wallet address if none provided
    const address = walletAddress || localStorage.getItem('walletAddress') || '';
    console.log('🔍 Fetching balance for wallet address:', address);
    return this.request<any>(`/v1/wallets/${address}/balance`);
  }

  // Send a transaction
  async sendTransaction(from: string, to: string, amount: number, memo?: string): Promise<ApiResponse<any>> {
    // Use stored wallet address if none provided for 'from'
    const fromAddress = from || localStorage.getItem('walletAddress') || '';
    
    // Fix: Ensure amount is sent as QNK value, not converted to smallest units
    // If amount looks like it's been unit-converted (> 1,000,000), convert it back
    let fixedAmount = amount;
    if (amount > 1000000) {
      console.warn(`⚠️ Detected unit conversion: ${amount} -> ${amount / 100000000} QNK`);
      fixedAmount = amount / 100000000;
    }
    
    console.log('📤 Sending transaction:', { from: fromAddress, to, amount: fixedAmount, memo });
    
    return this.request<any>('/v1/transactions/send', {
      method: 'POST',
      body: JSON.stringify({
        from: fromAddress,
        to: to,
        amount: fixedAmount,
        memo: memo
      }),
    });
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
      const response = await fetch(`${this.baseURL}/v1/mixer/send`, {
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

  // Get recent transactions (filtered by wallet address for privacy)
  async getRecentTransactions(limit = 100): Promise<ApiResponse<any[]>> {
    // Get wallet address from localStorage for privacy-filtered results
    const walletAddress = localStorage.getItem('walletAddress') || '';
    console.log('🔍 Fetching transactions for wallet address:', walletAddress);

    // Include wallet_address query parameter for backend filtering
    return this.request<any[]>(`/v1/transactions/recent?limit=${limit}&wallet_address=${walletAddress}`);
  }
}

// Export singleton instance
export const qnkAPI = new QNarwhalKnightAPI();

// Export for custom configurations
export { QNarwhalKnightAPI };