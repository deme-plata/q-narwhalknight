import axios from 'axios'

const API_BASE_URL = import.meta.env?.VITE_API_URL || 'https://quillon.xyz/bounty-api'
const MAIN_DOMAIN = 'https://quillon.xyz'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
})

// Main domain API for OAuth2 userinfo
const mainApi = axios.create({
  baseURL: MAIN_DOMAIN,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
})

// Add response error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.message || error.response?.data || error.message || 'Network error'
    return Promise.reject(new Error(typeof message === 'string' ? message : JSON.stringify(message)))
  }
)

export interface RegisterRequest {
  testnet_address: string
  mainnet_address?: string
}

export interface RegisterResponse {
  user_id: string
  token?: string
  message: string
}

export interface UserScore {
  user_id: string
  total_score: number
  rank: number | null
  tier: string
  category_scores: CategoryScores
  early_multiplier: number
  consistency_bonus: number
}

export interface CategoryScores {
  node_ops: number
  transactions: number
  bug_reports: number
  community: number
  social: number
}

export interface LeaderboardEntry {
  rank: number
  testnet_address: string
  total_score: number
  tier: string
  category_scores: CategoryScores
}

export interface BugReportRequest {
  issue_url: string
  severity: 'Critical' | 'High' | 'Medium' | 'Low'
  description: string
}

export interface BugReportResponse {
  report_id: string
  points_awarded: number
  message: string
}

export interface SocialActivityRequest {
  platform: 'twitter' | 'code_quillon' | 'discord' | 'medium' | 'youtube'
  activity_url: string
  activity_type: 'Tweet' | 'Thread' | 'Article' | 'Video' | 'DiscordMessage' | 'MergeRequest' | 'CodeIssue'
}

export interface SocialActivityResponse {
  message: string
  base_points: number
  status: string
}

// API functions
export const bountyApi = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health')
    return response.data
  },

  // Register a new user
  async register(data: RegisterRequest): Promise<RegisterResponse> {
    const response = await api.post('/v1/testnet/register', data)
    return response.data
  },

  // Get user score
  async getUserScore(userId: string): Promise<UserScore> {
    const response = await api.get(`/v1/testnet/score/${userId}`)
    return response.data
  },

  // Get leaderboard
  async getLeaderboard(limit: number = 100): Promise<LeaderboardEntry[]> {
    const response = await api.get('/v1/testnet/leaderboard', {
      params: { limit },
    })
    return response.data
  },

  // Submit bug report (no auth required for MVP)
  async submitBugReport(data: BugReportRequest & { user_id: string }): Promise<BugReportResponse> {
    const response = await api.post('/v1/testnet/bug-report', data, {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('bounty_token') || ''}` },
    })
    return response.data
  },

  // Submit social activity (no auth required for MVP)
  async submitSocialActivity(data: SocialActivityRequest & { user_id: string }): Promise<SocialActivityResponse> {
    const response = await api.post('/v1/testnet/social-activity', data, {
      headers: { 'Authorization': `Bearer ${localStorage.getItem('bounty_token') || ''}` },
    })
    return response.data
  },
}

// ============================================================================
// OAuth2 Authorization Code + PKCE Wallet Connect
// ============================================================================

const OAUTH2_CLIENT_ID = 'qnk_bounty_campaign'
const OAUTH2_CLIENT_SECRET = 'qnk_bounty_secret_2026'
const OAUTH2_SCOPES = 'read:balance read:profile'

export interface WalletSession {
  address: string
  connected: boolean
  oauth_token?: string
}

// -- PKCE helpers --

function generateRandomString(length: number): string {
  const array = new Uint8Array(length)
  crypto.getRandomValues(array)
  return Array.from(array, (b) => b.toString(16).padStart(2, '0')).join('')
}

async function sha256(plain: string): Promise<ArrayBuffer> {
  const encoder = new TextEncoder()
  return crypto.subtle.digest('SHA-256', encoder.encode(plain))
}

function base64urlEncode(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (const b of bytes) binary += String.fromCharCode(b)
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

async function generateCodeChallenge(verifier: string): Promise<string> {
  const hash = await sha256(verifier)
  return base64urlEncode(hash)
}

/**
 * Start OAuth2 Authorization Code + PKCE flow.
 * Redirects the browser to quillon.xyz authorize endpoint.
 */
export async function startOAuth2Connect(): Promise<void> {
  const codeVerifier = generateRandomString(64)
  const codeChallenge = await generateCodeChallenge(codeVerifier)
  const state = generateRandomString(32)

  // Persist PKCE verifier + state for callback validation
  sessionStorage.setItem('oauth2_code_verifier', codeVerifier)
  sessionStorage.setItem('oauth2_state', state)

  const redirectUri = `${window.location.origin}/callback`

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: OAUTH2_CLIENT_ID,
    redirect_uri: redirectUri,
    scope: OAUTH2_SCOPES,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
  })

  window.location.href = `${MAIN_DOMAIN}/api/v1/oauth2/authorize?${params.toString()}`
}

/**
 * Handle OAuth2 callback — exchange auth code for access token, then fetch userinfo.
 * Called from the /callback route after redirect back from quillon.xyz.
 */
export async function handleOAuth2Callback(code: string, state: string): Promise<WalletSession> {
  // Validate state
  const savedState = sessionStorage.getItem('oauth2_state')
  if (!savedState || savedState !== state) {
    throw new Error('Invalid OAuth2 state — possible CSRF attack')
  }

  const codeVerifier = sessionStorage.getItem('oauth2_code_verifier')
  if (!codeVerifier) {
    throw new Error('Missing PKCE code verifier — please try connecting again')
  }

  // Clean up PKCE storage
  sessionStorage.removeItem('oauth2_code_verifier')
  sessionStorage.removeItem('oauth2_state')

  const redirectUri = `${window.location.origin}/callback`

  // Exchange auth code for access token
  let tokenResponse
  try {
    tokenResponse = await mainApi.post('/api/v1/oauth2/token', {
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri,
      client_id: OAUTH2_CLIENT_ID,
      client_secret: OAUTH2_CLIENT_SECRET,
      code_verifier: codeVerifier,
    })
  } catch (err: any) {
    const status = err?.response?.status
    const detail = err?.response?.data?.error || err?.response?.data?.message || ''
    throw new Error(`Token exchange failed (${status || 'network error'}): ${detail || err.message}`)
  }

  // v7.4.0: Token endpoint returns ApiResponse wrapper: { success, data: { access_token, ... } }
  const tokenData = tokenResponse.data?.data || tokenResponse.data
  const accessToken: string = tokenData?.access_token || tokenData?.token
  if (!accessToken) {
    const serverError = tokenResponse.data?.error || 'empty response'
    throw new Error(`No access token received: ${serverError}`)
  }

  // Extract wallet address from the JWT payload (self-contained — no extra API call needed)
  // JWT format: header.payload.signature (each part is base64url-encoded)
  let address = ''
  try {
    const payloadPart = accessToken.split('.')[1]
    if (payloadPart) {
      // base64url → base64 → decode
      const b64 = payloadPart.replace(/-/g, '+').replace(/_/g, '/')
      const json = atob(b64)
      const payload = JSON.parse(json)
      address = payload.sub || payload.wallet_address || ''
    }
  } catch {
    // JWT decode failed — fall back to userinfo endpoint
  }

  // Fallback: call userinfo endpoint if JWT decode didn't work
  if (!address) {
    try {
      const userinfoResponse = await mainApi.get('/api/v1/oauth2/userinfo', {
        headers: { Authorization: `Bearer ${accessToken}` },
      })
      const userinfo = userinfoResponse.data?.data || userinfoResponse.data
      address = userinfo?.wallet_address || userinfo?.address || userinfo?.sub || ''
    } catch (err: any) {
      const status = err?.response?.status
      throw new Error(`Failed to fetch wallet info (${status || 'network'}): ${err.message}`)
    }
  }

  if (!address) {
    throw new Error('No wallet address found in token or userinfo')
  }

  const session: WalletSession = {
    address,
    connected: true,
    oauth_token: accessToken,
  }

  localStorage.setItem('bounty_wallet_session', JSON.stringify(session))
  return session
}

/**
 * Get stored wallet session from localStorage
 */
export function getStoredWalletSession(): WalletSession | null {
  const stored = localStorage.getItem('bounty_wallet_session')
  if (!stored) return null
  try {
    return JSON.parse(stored)
  } catch {
    return null
  }
}

/**
 * Clear wallet session (disconnect)
 */
export function disconnectWallet(): void {
  localStorage.removeItem('bounty_wallet_session')
}

/**
 * Fetch wallet info from main domain using OAuth2 token
 */
export async function fetchWalletInfo(token: string): Promise<{ address: string; balance?: string }> {
  const response = await mainApi.get('/api/v1/oauth2/userinfo', {
    headers: { Authorization: `Bearer ${token}` },
  })
  return response.data?.data || response.data
}

export default bountyApi
