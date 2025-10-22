/**
 * Quillon Wallet OAuth2 SDK
 *
 * Allows third-party websites to authenticate users via Quillon Wallet
 * and access blockchain functionality with user consent.
 *
 * @version 1.0.0
 * @license MIT
 */

class QullionOAuth2Client {
  /**
   * Initialize the OAuth2 client
   *
   * @param {Object} config - Configuration options
   * @param {string} config.clientId - Your OAuth2 client ID
   * @param {string} config.clientSecret - Your OAuth2 client secret (server-side only)
   * @param {string} config.redirectUri - The redirect URI registered with Quillon
   * @param {string[]} config.scopes - Requested permission scopes
   * @param {string} config.walletUrl - Quillon Wallet URL (default: https://wallet.quillon.xyz)
   * @param {string} config.apiUrl - Quillon API URL (default: https://api.quillon.xyz)
   */
  constructor(config) {
    this.clientId = config.clientId;
    this.clientSecret = config.clientSecret;
    this.redirectUri = config.redirectUri;
    this.scopes = config.scopes || ['read:balance'];
    this.walletUrl = config.walletUrl || 'https://wallet.quillon.xyz';
    this.apiUrl = config.apiUrl || 'https://api.quillon.xyz';

    // PKCE state
    this.codeVerifier = null;
    this.codeChallenge = null;
    this.state = null;

    // Token storage
    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiry = null;
  }

  /**
   * Generate random string for PKCE
   * @private
   */
  _generateRandomString(length = 128) {
    const array = new Uint8Array(length);
    crypto.getRandomValues(array);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('').substring(0, length);
  }

  /**
   * Generate SHA-256 hash
   * @private
   */
  async _sha256(plain) {
    const encoder = new TextEncoder();
    const data = encoder.encode(plain);
    const hash = await crypto.subtle.digest('SHA-256', data);
    return hash;
  }

  /**
   * Base64 URL encode
   * @private
   */
  _base64UrlEncode(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary)
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');
  }

  /**
   * Generate PKCE code verifier and challenge
   * @private
   */
  async _generatePKCE() {
    this.codeVerifier = this._generateRandomString(128);
    const hashed = await this._sha256(this.codeVerifier);
    this.codeChallenge = this._base64UrlEncode(hashed);
  }

  /**
   * Store PKCE state in sessionStorage
   * @private
   */
  _storePKCEState() {
    sessionStorage.setItem('quillon_oauth2_verifier', this.codeVerifier);
    sessionStorage.setItem('quillon_oauth2_state', this.state);
  }

  /**
   * Retrieve PKCE state from sessionStorage
   * @private
   */
  _retrievePKCEState() {
    this.codeVerifier = sessionStorage.getItem('quillon_oauth2_verifier');
    const storedState = sessionStorage.getItem('quillon_oauth2_state');
    return storedState;
  }

  /**
   * Clear PKCE state from sessionStorage
   * @private
   */
  _clearPKCEState() {
    sessionStorage.removeItem('quillon_oauth2_verifier');
    sessionStorage.removeItem('quillon_oauth2_state');
  }

  /**
   * Start the OAuth2 authorization flow
   * Redirects the user to Quillon Wallet for consent
   *
   * @returns {Promise<void>}
   */
  async authorize() {
    // Generate PKCE parameters
    await this._generatePKCE();
    this.state = this._generateRandomString(32);

    // Store PKCE state for callback
    this._storePKCEState();

    // Build authorization URL
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: this.redirectUri,
      response_type: 'code',
      scope: this.scopes.join(' '),
      state: this.state,
      code_challenge: this.codeChallenge,
      code_challenge_method: 'S256'
    });

    const authUrl = `${this.walletUrl}/oauth2/authorize?${params.toString()}`;

    // Redirect to Quillon Wallet
    window.location.href = authUrl;
  }

  /**
   * Handle the OAuth2 callback after user consent
   * Call this in your redirect URI page
   *
   * @returns {Promise<Object>} Token response with access_token and user info
   * @throws {Error} If authorization failed or state mismatch
   */
  async handleCallback() {
    const urlParams = new URLSearchParams(window.location.search);

    // Check for error response
    const error = urlParams.get('error');
    if (error) {
      const errorDescription = urlParams.get('error_description') || 'Unknown error';
      throw new Error(`OAuth2 Error: ${error} - ${errorDescription}`);
    }

    // Get authorization code and state
    const code = urlParams.get('code');
    const returnedState = urlParams.get('state');

    if (!code) {
      throw new Error('No authorization code received');
    }

    // Verify state to prevent CSRF
    const storedState = this._retrievePKCEState();
    if (returnedState !== storedState) {
      throw new Error('State mismatch - possible CSRF attack');
    }

    // Exchange code for access token
    const tokenResponse = await this.exchangeCodeForToken(code);

    // Clear PKCE state
    this._clearPKCEState();

    return tokenResponse;
  }

  /**
   * Exchange authorization code for access token
   *
   * @param {string} code - Authorization code from callback
   * @returns {Promise<Object>} Token response
   */
  async exchangeCodeForToken(code) {
    const response = await fetch(`${this.apiUrl}/api/v1/oauth2/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'authorization_code',
        code: code,
        redirect_uri: this.redirectUri,
        client_id: this.clientId,
        client_secret: this.clientSecret,
        code_verifier: this.codeVerifier
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Token exchange failed: ${error.error || 'Unknown error'}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(`Token exchange failed: ${data.error || 'Unknown error'}`);
    }

    // Store tokens
    this.accessToken = data.data.access_token;
    this.refreshToken = data.data.refresh_token;
    this.tokenExpiry = Date.now() + (data.data.expires_in * 1000);

    // Store in localStorage for persistence
    this._storeTokens();

    return data.data;
  }

  /**
   * Store tokens in localStorage
   * @private
   */
  _storeTokens() {
    localStorage.setItem('quillon_access_token', this.accessToken);
    localStorage.setItem('quillon_refresh_token', this.refreshToken);
    localStorage.setItem('quillon_token_expiry', this.tokenExpiry.toString());
  }

  /**
   * Load tokens from localStorage
   * @private
   */
  _loadTokens() {
    this.accessToken = localStorage.getItem('quillon_access_token');
    this.refreshToken = localStorage.getItem('quillon_refresh_token');
    const expiry = localStorage.getItem('quillon_token_expiry');
    this.tokenExpiry = expiry ? parseInt(expiry) : null;
  }

  /**
   * Clear tokens from storage
   * @private
   */
  _clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiry = null;
    localStorage.removeItem('quillon_access_token');
    localStorage.removeItem('quillon_refresh_token');
    localStorage.removeItem('quillon_token_expiry');
  }

  /**
   * Check if access token is expired
   * @returns {boolean}
   */
  isTokenExpired() {
    if (!this.tokenExpiry) return true;
    return Date.now() >= this.tokenExpiry;
  }

  /**
   * Refresh the access token using refresh token
   * @returns {Promise<Object>} New token response
   */
  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.apiUrl}/api/v1/oauth2/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        refresh_token: this.refreshToken,
        client_id: this.clientId,
        client_secret: this.clientSecret
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Token refresh failed: ${error.error || 'Unknown error'}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(`Token refresh failed: ${data.error || 'Unknown error'}`);
    }

    // Update tokens
    this.accessToken = data.data.access_token;
    if (data.data.refresh_token) {
      this.refreshToken = data.data.refresh_token;
    }
    this.tokenExpiry = Date.now() + (data.data.expires_in * 1000);

    // Store updated tokens
    this._storeTokens();

    return data.data;
  }

  /**
   * Get valid access token (refreshes if expired)
   * @returns {Promise<string>} Valid access token
   */
  async getAccessToken() {
    // Load from storage if not in memory
    if (!this.accessToken) {
      this._loadTokens();
    }

    // Refresh if expired
    if (this.isTokenExpired()) {
      await this.refreshAccessToken();
    }

    return this.accessToken;
  }

  /**
   * Get user information (wallet address, etc.)
   * @returns {Promise<Object>} User info
   */
  async getUserInfo() {
    const token = await this.getAccessToken();

    const response = await fetch(`${this.apiUrl}/api/v1/oauth2/userinfo`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user info');
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || 'Failed to fetch user info');
    }

    return data.data;
  }

  /**
   * Revoke access token and log out
   * @returns {Promise<void>}
   */
  async revoke() {
    if (!this.accessToken) return;

    try {
      await fetch(`${this.apiUrl}/api/v1/oauth2/revoke`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          token: this.accessToken,
          client_id: this.clientId,
          client_secret: this.clientSecret
        })
      });
    } catch (error) {
      console.error('Token revocation failed:', error);
    }

    // Clear local tokens regardless of server response
    this._clearTokens();
  }

  /**
   * Make an authenticated API request
   * @param {string} endpoint - API endpoint path
   * @param {Object} options - Fetch options
   * @returns {Promise<Object>} Response data
   */
  async authenticatedRequest(endpoint, options = {}) {
    const token = await this.getAccessToken();

    const headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...options.headers
    };

    const response = await fetch(`${this.apiUrl}${endpoint}`, {
      ...options,
      headers
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get user's token balance
   * @param {string} token - Token symbol (e.g., 'QUG', 'QUGUSD')
   * @returns {Promise<number>} Token balance
   */
  async getBalance(token = 'QUG') {
    const userInfo = await this.getUserInfo();
    const response = await this.authenticatedRequest(`/api/v1/wallet/${userInfo.wallet_address}/balance/${token}`);
    return response.data.balance;
  }

  /**
   * Send a transaction
   * @param {Object} params - Transaction parameters
   * @param {string} params.to - Recipient wallet address
   * @param {number} params.amount - Amount to send
   * @param {string} params.token - Token to send (default: 'QUG')
   * @returns {Promise<Object>} Transaction result
   */
  async sendTransaction(params) {
    const userInfo = await this.getUserInfo();

    const response = await this.authenticatedRequest('/api/v1/transaction/send', {
      method: 'POST',
      body: JSON.stringify({
        from: userInfo.wallet_address,
        to: params.to,
        amount: params.amount,
        token: params.token || 'QUG'
      })
    });

    return response.data;
  }

  /**
   * Get transaction history
   * @param {number} limit - Maximum number of transactions to return
   * @returns {Promise<Array>} Transaction history
   */
  async getTransactionHistory(limit = 50) {
    const userInfo = await this.getUserInfo();
    const response = await this.authenticatedRequest(`/api/v1/wallet/${userInfo.wallet_address}/transactions?limit=${limit}`);
    return response.data;
  }

  /**
   * Check if user is authenticated
   * @returns {boolean}
   */
  isAuthenticated() {
    this._loadTokens();
    return !!this.accessToken && !this.isTokenExpired();
  }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = QullionOAuth2Client;
}

if (typeof window !== 'undefined') {
  window.QullionOAuth2Client = QullionOAuth2Client;
}
