/**
 * Quillon Wallet OAuth2 SDK - TypeScript Definitions
 * @version 1.0.0
 */

export interface QullionOAuth2Config {
  /** Your OAuth2 client ID from Quillon */
  clientId: string;

  /** Your OAuth2 client secret (server-side only, never expose in browser) */
  clientSecret?: string;

  /** The redirect URI registered with Quillon */
  redirectUri: string;

  /** Requested permission scopes */
  scopes?: string[];

  /** Quillon Wallet URL (default: https://wallet.quillon.xyz) */
  walletUrl?: string;

  /** Quillon API URL (default: https://api.quillon.xyz) */
  apiUrl?: string;
}

export interface TokenResponse {
  /** Access token for API requests */
  access_token: string;

  /** Token type (always "Bearer") */
  token_type: string;

  /** Token expiration time in seconds */
  expires_in: number;

  /** Refresh token for getting new access tokens */
  refresh_token?: string;

  /** Granted permission scopes */
  scope: string;
}

export interface UserInfo {
  /** User's wallet address */
  wallet_address: string;

  /** Granted permission scopes */
  scopes: string[];

  /** OAuth2 client ID */
  client_id: string;
}

export interface TransactionParams {
  /** Recipient wallet address */
  to: string;

  /** Amount to send (in token's base unit) */
  amount: number;

  /** Token symbol (default: 'QUG') */
  token?: string;
}

export interface Transaction {
  /** Transaction ID */
  id: string;

  /** Sender wallet address */
  from: string;

  /** Recipient wallet address */
  to: string;

  /** Transaction amount */
  amount: number;

  /** Token symbol */
  token: string;

  /** Transaction timestamp */
  timestamp: number;

  /** Transaction status */
  status: 'pending' | 'confirmed' | 'failed';

  /** Block number (if confirmed) */
  block_number?: number;
}

export default class QullionOAuth2Client {
  /** Your OAuth2 client ID */
  clientId: string;

  /** Your OAuth2 client secret */
  clientSecret?: string;

  /** Registered redirect URI */
  redirectUri: string;

  /** Requested scopes */
  scopes: string[];

  /** Quillon Wallet URL */
  walletUrl: string;

  /** Quillon API URL */
  apiUrl: string;

  /** Current access token */
  accessToken: string | null;

  /** Current refresh token */
  refreshToken: string | null;

  /** Token expiry timestamp */
  tokenExpiry: number | null;

  /**
   * Initialize the OAuth2 client
   */
  constructor(config: QullionOAuth2Config);

  /**
   * Start the OAuth2 authorization flow
   * Redirects the user to Quillon Wallet for consent
   */
  authorize(): Promise<void>;

  /**
   * Handle the OAuth2 callback after user consent
   * Call this in your redirect URI page
   *
   * @returns Token response with access_token and user info
   * @throws Error if authorization failed or state mismatch
   */
  handleCallback(): Promise<TokenResponse>;

  /**
   * Exchange authorization code for access token
   *
   * @param code - Authorization code from callback
   * @returns Token response
   */
  exchangeCodeForToken(code: string): Promise<TokenResponse>;

  /**
   * Check if access token is expired
   */
  isTokenExpired(): boolean;

  /**
   * Refresh the access token using refresh token
   * @returns New token response
   */
  refreshAccessToken(): Promise<TokenResponse>;

  /**
   * Get valid access token (refreshes if expired)
   * @returns Valid access token
   */
  getAccessToken(): Promise<string>;

  /**
   * Get user information (wallet address, etc.)
   * @returns User info
   */
  getUserInfo(): Promise<UserInfo>;

  /**
   * Revoke access token and log out
   */
  revoke(): Promise<void>;

  /**
   * Make an authenticated API request
   *
   * @param endpoint - API endpoint path
   * @param options - Fetch options
   * @returns Response data
   */
  authenticatedRequest(endpoint: string, options?: RequestInit): Promise<any>;

  /**
   * Get user's token balance
   *
   * @param token - Token symbol (e.g., 'QUG', 'QUGUSD')
   * @returns Token balance
   */
  getBalance(token?: string): Promise<number>;

  /**
   * Send a transaction
   *
   * @param params - Transaction parameters
   * @returns Transaction result
   */
  sendTransaction(params: TransactionParams): Promise<Transaction>;

  /**
   * Get transaction history
   *
   * @param limit - Maximum number of transactions to return
   * @returns Transaction history
   */
  getTransactionHistory(limit?: number): Promise<Transaction[]>;

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean;
}

/**
 * Available OAuth2 scopes
 */
export type OAuth2Scope =
  | 'read:balance'
  | 'send:transaction'
  | 'read:transactions'
  | 'manage:tokens';

/**
 * OAuth2 error codes
 */
export type OAuth2Error =
  | 'invalid_request'
  | 'unauthorized_client'
  | 'access_denied'
  | 'unsupported_response_type'
  | 'invalid_scope'
  | 'server_error'
  | 'temporarily_unavailable';
