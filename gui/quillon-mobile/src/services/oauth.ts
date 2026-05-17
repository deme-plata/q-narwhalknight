/**
 * OAuth2 Authorization Code + PKCE flow for Quillon wallet authentication.
 *
 * Flow:
 * 1. Generate PKCE code_verifier + code_challenge
 * 2. Open auth session browser to authorization endpoint (consent page)
 * 3. User connects wallet and approves consent
 * 4. Browser redirects back → openAuthSessionAsync captures the URL
 * 5. Parse code from URL, exchange code + verifier for access token
 */

import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import * as Linking from 'expo-linking';
import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';

// Ensure any previous auth sessions are cleaned up
WebBrowser.maybeCompleteAuthSession();

// ============================================================================
// Configuration
// ============================================================================

const QUILLON_AUTH_BASE = 'https://quillon.xyz';
const OAUTH_CONFIG = {
  authorizationEndpoint: `${QUILLON_AUTH_BASE}/api/v1/oauth2/authorize`,
  tokenEndpoint: `${QUILLON_AUTH_BASE}/api/v1/oauth2/token`,
  userinfoEndpoint: `${QUILLON_AUTH_BASE}/api/v1/oauth2/userinfo`,
  clientId: 'quillon-mobile-wallet',
  scopes: ['read:balance', 'read:history', 'read:tokens', 'send:transaction'],
};

// Deep link redirect URI — uses a path that has NO matching Expo Router route
// so that openAuthSessionAsync captures it instead of Expo Router intercepting it.
const REDIRECT_URI = AuthSession.makeRedirectUri({
  scheme: 'qnk',
  path: 'oauth-callback',
});

// ============================================================================
// Types
// ============================================================================

export interface OAuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
  scope: string;
}

export interface OAuthWalletInfo {
  walletAddress: string;
  scopes: string[];
}

export interface OAuthError {
  code: 'cancelled' | 'network' | 'auth_failed' | 'unknown';
  message: string;
}

export type OAuthResult =
  | { success: true; tokens: OAuthTokens; walletInfo: OAuthWalletInfo | null }
  | { success: false; error: OAuthError };

// ============================================================================
// PKCE Helpers
// ============================================================================

function generateCodeVerifier(): string {
  const bytes = new Uint8Array(32);
  globalThis.crypto.getRandomValues(bytes);
  return bytesToHex(bytes);
}

function generateCodeChallenge(verifier: string): string {
  const encoded = new TextEncoder().encode(verifier);
  const hash = sha256(encoded);
  return btoa(String.fromCharCode(...hash))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

function generateState(): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return bytesToHex(bytes);
}

// ============================================================================
// Main OAuth2 Flow — opens browser, captures redirect, exchanges code
// ============================================================================

/**
 * Runs the full OAuth2 PKCE flow:
 * 1. Opens auth session browser to consent page
 * 2. Waits for redirect back — uses openAuthSessionAsync + Linking fallback
 * 3. Parses auth code from redirect URL
 * 4. Exchanges code for tokens
 * 5. Fetches wallet info
 *
 * The Linking fallback handles Android/Expo Go where Chrome Custom Tabs
 * sometimes show a brief error page before the deep link is captured.
 */
export async function runOAuthFlow(): Promise<OAuthResult> {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = generateCodeChallenge(codeVerifier);
  const state = generateState();

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: OAUTH_CONFIG.clientId,
    redirect_uri: REDIRECT_URI,
    scope: OAUTH_CONFIG.scopes.join(' '),
    state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
  });

  const authUrl = `${OAUTH_CONFIG.authorizationEndpoint}?${params.toString()}`;
  console.log('[OAUTH] Opening auth session, redirect:', REDIRECT_URI);

  // Set up a Linking listener as fallback — on Android/Expo Go, the deep link
  // may arrive via Linking before openAuthSessionAsync resolves.
  let capturedUrl: string | null = null;
  let linkingResolve: ((url: string) => void) | null = null;
  const linkingPromise = new Promise<string>((resolve) => {
    linkingResolve = resolve;
  });

  const linkingSub = Linking.addEventListener('url', (event) => {
    if (event.url && event.url.includes('oauth-callback')) {
      console.log('[OAUTH] Linking fallback captured URL');
      capturedUrl = event.url;
      linkingResolve?.(event.url);
    }
  });

  try {
    // Open browser and wait for redirect capture
    const result = await WebBrowser.openAuthSessionAsync(authUrl, REDIRECT_URI);
    console.log('[OAUTH] Browser result:', result.type);

    let redirectUrlStr: string;

    if (result.type === 'success') {
      redirectUrlStr = result.url;
    } else if (capturedUrl) {
      // Linking fallback already captured the redirect URL
      console.log('[OAUTH] Using Linking fallback URL');
      redirectUrlStr = capturedUrl;
    } else {
      // Give Linking listener a brief window to capture the URL
      const timeoutPromise = new Promise<string>((_, reject) =>
        setTimeout(() => reject(new Error('timeout')), 2000)
      );
      try {
        redirectUrlStr = await Promise.race([linkingPromise, timeoutPromise]);
        console.log('[OAUTH] Linking fallback captured URL (delayed)');
      } catch {
        return {
          success: false,
          error: {
            code: 'cancelled',
            message: result.type === 'cancel' ? 'Authentication cancelled' : 'Browser dismissed',
          },
        };
      }
    }

    return await processRedirectUrl(redirectUrlStr, state, codeVerifier);
  } finally {
    linkingSub.remove();
  }
}

/**
 * Process the OAuth redirect URL — parse code, exchange for tokens, fetch wallet info.
 */
async function processRedirectUrl(
  urlStr: string,
  expectedState: string,
  codeVerifier: string,
): Promise<OAuthResult> {
  const redirectUrl = new URL(urlStr);
  const error = redirectUrl.searchParams.get('error');
  const errorDesc = redirectUrl.searchParams.get('error_description');
  const code = redirectUrl.searchParams.get('code');
  const returnedState = redirectUrl.searchParams.get('state');

  if (error) {
    return {
      success: false,
      error: { code: 'auth_failed', message: errorDesc || error },
    };
  }

  if (!code) {
    return {
      success: false,
      error: { code: 'auth_failed', message: 'No authorization code received' },
    };
  }

  if (returnedState !== expectedState) {
    return {
      success: false,
      error: { code: 'auth_failed', message: 'State mismatch — possible CSRF attack' },
    };
  }

  // Exchange code for tokens
  try {
    const tokens = await exchangeCodeForTokens(code, codeVerifier);
    const walletInfo = await fetchWalletInfo(tokens.accessToken);
    console.log('[OAUTH] Token exchange + userinfo complete, wallet:', walletInfo?.walletAddress ?? 'unknown');
    return { success: true, tokens, walletInfo };
  } catch (err) {
    console.error('[OAUTH] Token exchange failed:', err);
    return {
      success: false,
      error: {
        code: 'network',
        message: err instanceof Error ? err.message : 'Token exchange failed',
      },
    };
  }
}

// ============================================================================
// Token refresh
// ============================================================================

export async function refreshAccessToken(refreshToken: string): Promise<OAuthTokens | null> {
  try {
    const response = await fetch(OAUTH_CONFIG.tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: OAUTH_CONFIG.clientId,
      }),
    });

    if (!response.ok) return null;

    const data = await response.json();
    const tokenData = data.data ?? data;
    return parseTokenResponse(tokenData);
  } catch {
    return null;
  }
}

// ============================================================================
// Internal Helpers
// ============================================================================

async function exchangeCodeForTokens(
  code: string,
  codeVerifier: string
): Promise<OAuthTokens> {
  const response = await fetch(OAUTH_CONFIG.tokenEndpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'authorization_code',
      code,
      redirect_uri: REDIRECT_URI,
      client_id: OAUTH_CONFIG.clientId,
      code_verifier: codeVerifier,
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => 'Unknown error');
    throw new Error(`Token exchange failed: ${response.status} ${text}`);
  }

  const data = await response.json();
  const tokenData = data.data ?? data;
  return parseTokenResponse(tokenData);
}

function parseTokenResponse(data: Record<string, unknown>): OAuthTokens {
  const expiresIn = typeof data.expires_in === 'number' ? data.expires_in : 3600;
  return {
    accessToken: String(data.access_token || ''),
    refreshToken: String(data.refresh_token || ''),
    expiresAt: Date.now() + expiresIn * 1000,
    scope: String(data.scope || ''),
  };
}

async function fetchWalletInfo(accessToken: string): Promise<OAuthWalletInfo | null> {
  try {
    const response = await fetch(OAUTH_CONFIG.userinfoEndpoint, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        Accept: 'application/json',
      },
    });

    if (!response.ok) return null;

    const data = await response.json();
    const info = data.data ?? data;
    return {
      walletAddress: String(info.wallet_address || ''),
      scopes: Array.isArray(info.scopes) ? info.scopes : [],
    };
  } catch {
    return null;
  }
}
