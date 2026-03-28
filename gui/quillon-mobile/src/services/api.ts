/**
 * REST API client for the Quillon blockchain.
 *
 * Base URL: https://quillon.xyz/api/v1
 * Authentication: X-Wallet-Auth header with Ed25519 signed challenge
 */

import * as ed from '@noble/ed25519';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex } from '@noble/hashes/utils.js';
import { getMnemonic } from './secureStorage';

const API_ENDPOINTS = [
  'https://quillon.xyz/api/v1',
  'http://89.149.241.126:8080/api/v1', // Epsilon direct fallback (HTTP — IP has no TLS cert)
] as const;

const REQUEST_TIMEOUT_MS = 15_000;
const MAX_RETRIES = 2;

let currentEndpointIndex = 0;
let authToken: string | null = null;

// ---------- Types ----------

export interface BalanceResponse {
  address: string;
  /** Raw u128 balance in base units (10^24 per QUG) */
  balance: string;
  /** Human-readable QUG balance */
  balance_qnk: number;
  token_balances: Record<string, string>;
  nonce: number;
}

export interface Transaction {
  /** Transaction ID/hash */
  id: string;
  /** "transfer" | "swap" | "token_transfer" | "mining_reward" */
  tx_type: string;
  from: string;
  to: string;
  amount: string;
  timestamp: number;
  block_height: number;
  status: string;
  /** "sent" | "received" | "swap" */
  direction: string;
  token_symbol?: string;
  token_address?: string;
  amount_out?: string;
  token_in?: string;
  token_out?: string;
  memo?: string;
}

/** Server wraps responses in { success, data } */
interface ApiDataResponse<T> {
  success: boolean;
  data: T;
}

export interface HistoryResponse {
  transactions: Transaction[];
  total: number;
  page: number;
  page_size: number;
}

export interface TransferRequest {
  from: string;
  to: string;
  amount: string;
  token: string;
  mnemonic?: string;
  memo?: string;
}

export interface TransferResponse {
  tx_hash: string;
  status: string;
}

export interface DexToken {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  logo_url?: string;
  price_usd?: number;
}

export interface DexQuote {
  amount_in: string;
  amount_out: string;
  minimum_amount_out: string;
  price_impact: number;
  gas_estimate: number;
  route: string[];
  execution_price: number;
  valid_until: number;
}

export interface SwapRequest {
  token_in: string;
  token_out: string;
  amount_in: string;
  minimum_amount_out: string;
  recipient: string;
  deadline: number;
  signature: string;
}

export interface SwapResponse {
  transaction_hash: string;
  status: string;
  amount_in: string;
  amount_out: string;
  gas_used: number;
}

export interface WorkerStats {
  worker_id: string;
  worker_name: string | null;
  hash_rate: number;
  blocks_found: number;
  rewards_earned: string;
  rewards_earned_raw: string;
  solutions_submitted: number;
  last_activity_secs: number;
  is_active: boolean;
}

export interface MiningStats {
  wallet: string;
  blocks_found: number;
  hash_rate: number;
  rewards_earned: string;
  rewards_earned_raw: string;
  total_workers: number;
  last_activity_secs: number;
  is_active: boolean;
  workers: WorkerStats[];
}

export interface HealthResponse {
  status: string;
  version: string;
  height: number;
  peers: number;
  uptime_seconds: number;
  network_id: string;
  tps: number;
}

// ---------- Internal Helpers ----------

function getBaseUrl(): string {
  return API_ENDPOINTS[currentEndpointIndex];
}

function failover(): void {
  currentEndpointIndex = (currentEndpointIndex + 1) % API_ENDPOINTS.length;
  console.warn(`[API] Failing over to ${getBaseUrl()}`);
}

function buildHeaders(extraHeaders?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  if (extraHeaders) {
    Object.assign(headers, extraHeaders);
  }
  return headers;
}

/**
 * Generate X-Wallet-Auth header with Ed25519 signed challenge.
 * Challenge = SHA3-256(address_bytes || timestamp_le8 || path_utf8)
 */
async function generateWalletAuthHeader(
  address: string,
  requestPath: string
): Promise<string | null> {
  try {
    const mnemonic = await getMnemonic();
    if (!mnemonic) return null;

    const mnemonicBytes = new TextEncoder().encode(mnemonic);
    const privateKey = sha3_256(mnemonicBytes);

    const timestamp = Math.floor(Date.now() / 1000);

    // Build challenge: address_bytes || timestamp_le8 || path_utf8
    const addressHex = address.startsWith('qnk') ? address.substring(3) : address;
    const addressBytes = hexToBytes(addressHex);

    const timestampBytes = new Uint8Array(8);
    new DataView(timestampBytes.buffer).setBigInt64(0, BigInt(timestamp), true);

    const pathBytes = new TextEncoder().encode(requestPath);

    const combined = new Uint8Array(addressBytes.length + timestampBytes.length + pathBytes.length);
    combined.set(addressBytes, 0);
    combined.set(timestampBytes, addressBytes.length);
    combined.set(pathBytes, addressBytes.length + timestampBytes.length);

    const challenge = sha3_256(combined);
    const signature = await ed.signAsync(challenge, privateKey);

    // Zero the private key
    privateKey.fill(0);

    return JSON.stringify({
      address,
      timestamp,
      scheme: 'Ed25519',
      signature: bytesToHex(signature),
    });
  } catch (err) {
    console.warn('[API] Failed to generate auth header:', err);
    return null;
  }
}

function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes;
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number = REQUEST_TIMEOUT_MS,
  externalSignal?: AbortSignal
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  // If an external signal is provided, abort our controller when it fires
  const onExternalAbort = () => controller.abort();
  externalSignal?.addEventListener('abort', onExternalAbort);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(id);
    externalSignal?.removeEventListener('abort', onExternalAbort);
  }
}

async function apiRequest<T>(
  method: string,
  path: string,
  body?: unknown,
  retries: number = MAX_RETRIES,
  extraHeaders?: Record<string, string>,
  signal?: AbortSignal
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      // If the caller's signal is already aborted, bail immediately
      if (signal?.aborted) {
        const err = new Error('Request aborted');
        err.name = 'AbortError';
        throw err;
      }

      const url = `${getBaseUrl()}${path}`;
      const options: RequestInit = {
        method,
        headers: buildHeaders(extraHeaders),
      };

      if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
        options.body = JSON.stringify(body);
      }

      const response = await fetchWithTimeout(url, options, REQUEST_TIMEOUT_MS, signal);

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(`API ${response.status}: ${errorText}`);
      }

      const data = (await response.json()) as T;
      return data;
    } catch (error) {
      // Don't retry AbortErrors — they're intentional cancellations
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }

      lastError = error instanceof Error ? error : new Error(String(error));
      console.warn(`[API] Attempt ${attempt + 1} failed: ${lastError.message}`);

      if (attempt < retries) {
        failover();
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      }
    }
  }

  throw lastError ?? new Error('API request failed');
}

// ---------- Public API ----------

export function setAuthToken(token: string | null): void {
  authToken = token;
}

export function getActiveEndpoint(): string {
  return getBaseUrl();
}

export async function getHealth(): Promise<HealthResponse> {
  // Server endpoint: /api/v1/node/status
  // Returns { success, data: { current_height, connected_peers, network_id, uptime_seconds, tps_current, ... } }
  const resp = await apiRequest<ApiDataResponse<Record<string, unknown>>>('GET', '/node/status');
  const d = resp.data ?? {};
  return {
    status: (d.network_health as string) ?? 'ok',
    version: (d.version as string) ?? '',
    height: (d.current_height as number) ?? 0,
    peers: (d.connected_peers as number) ?? 0,
    uptime_seconds: (d.uptime_seconds as number) ?? 0,
    network_id: (d.network_id as string) ?? '',
    tps: (d.tps_current as number) ?? 0,
  };
}

export async function getBalance(address: string): Promise<BalanceResponse> {
  const path = `/wallets/${address}/balance`;
  const walletAuth = await generateWalletAuthHeader(address, `/api/v1${path}`);
  const headers: Record<string, string> = {};
  if (walletAuth) {
    headers['X-Wallet-Auth'] = walletAuth;
  }

  const resp = await apiRequest<ApiDataResponse<{
    wallet_address: string;
    balance: string;
    balance_qnk: number;
  }>>('GET', path, undefined, MAX_RETRIES, headers);

  const data = resp.data ?? { wallet_address: address, balance: '0', balance_qnk: 0 };
  return {
    address: data.wallet_address ?? address,
    balance: data.balance ?? '0',
    balance_qnk: data.balance_qnk ?? 0,
    token_balances: {},
    nonce: 0,
  };
}

export async function getHistory(
  address: string,
  page: number = 1,
  pageSize: number = 20,
  signal?: AbortSignal
): Promise<HistoryResponse> {
  // Server endpoint: /api/v1/wallet/:address/history
  // Returns { success: true, data: UnifiedTransactionEntry[] }
  const resp = await apiRequest<ApiDataResponse<Transaction[]>>(
    'GET',
    `/wallet/${address}/history`,
    undefined,
    MAX_RETRIES,
    undefined,
    signal
  );

  const txs = resp.data ?? [];
  return {
    transactions: txs,
    total: txs.length,
    page,
    page_size: pageSize,
  };
}

export async function transfer(request: TransferRequest): Promise<TransferResponse> {
  // Server's /transactions/send requires X-Wallet-Auth header
  const walletAuth = await generateWalletAuthHeader(request.from, '/api/v1/transactions/send');
  const headers: Record<string, string> = {};
  if (walletAuth) {
    headers['X-Wallet-Auth'] = walletAuth;
  }

  const resp = await apiRequest<ApiDataResponse<{ tx_hash: string; status: string }>>(
    'POST',
    '/transactions/send',
    {
      from: request.from,
      to: request.to,
      amount: parseFloat(request.amount),  // Server expects f64
      token_type: request.token || 'QUG',
      mnemonic: request.mnemonic,          // Server signs server-side
      memo: request.memo,
    },
    MAX_RETRIES,
    headers
  );

  const data = resp.data ?? { tx_hash: '', status: 'unknown' };
  return { tx_hash: data.tx_hash, status: data.status };
}

export interface MultiTokenBalance {
  address: string;
  tokens: Record<string, {
    balance: string;
    balance_base_units: string;
    usd_value: number;
    name?: string;
    contract_address?: string;
    decimals?: number;
  }>;
  total_usd_value: number;
}

export async function getMultiTokenBalance(
  address: string,
  signal?: AbortSignal
): Promise<MultiTokenBalance> {
  const path = '/wallet/tokens';
  const walletAuth = await generateWalletAuthHeader(address, `/api/v1${path}`);
  const headers: Record<string, string> = {};
  if (walletAuth) {
    headers['X-Wallet-Auth'] = walletAuth;
  }

  const resp = await apiRequest<ApiDataResponse<MultiTokenBalance>>(
    'GET', path, undefined, MAX_RETRIES, headers, signal
  );
  return resp.data ?? { address, tokens: {}, total_usd_value: 0 };
}

export async function getDexTokens(): Promise<DexToken[]> {
  const resp = await apiRequest<ApiDataResponse<DexToken[]>>('GET', '/dex/tokens');
  return resp.data ?? [];
}

/**
 * Get a DEX swap quote.
 * The server expects amount_in in 24-decimal base units (u128 string).
 * We convert the display amount (e.g. "1.5") to base units here.
 */
export async function getDexQuote(
  tokenIn: string,
  tokenOut: string,
  amountIn: string,
  signal?: AbortSignal
): Promise<DexQuote> {
  // Convert display amount to 24-decimal base units
  const amountInBaseUnits = displayToBaseUnits(amountIn, 24);

  const resp = await apiRequest<{ success: boolean; data?: DexQuote; error?: string }>(
    'POST',
    '/dex/swap/quote',
    {
      token_in: tokenIn,
      token_out: tokenOut,
      amount_in: amountInBaseUnits,
    },
    MAX_RETRIES,
    undefined,
    signal
  );

  if (!resp.success || !resp.data) {
    throw new Error(resp.error || 'Quote unavailable');
  }

  return resp.data;
}

/** Convert a human-readable amount (e.g. "1.5") to base-unit string with `decimals` precision. */
function displayToBaseUnits(display: string, decimals: number): string {
  const parts = display.split('.');
  const whole = parts[0] || '0';
  let frac = (parts[1] || '').slice(0, decimals); // truncate excess
  frac = frac.padEnd(decimals, '0');
  // Remove leading zeros but keep at least "0"
  const raw = (whole + frac).replace(/^0+/, '') || '0';
  return raw;
}

/** Convert a base-unit string back to display amount with `decimals` precision. */
export function baseUnitsToDisplay(baseUnits: string, decimals: number): string {
  const padded = baseUnits.padStart(decimals + 1, '0');
  const whole = padded.slice(0, padded.length - decimals) || '0';
  const frac = padded.slice(padded.length - decimals);
  // Trim trailing zeros from fractional part
  const trimmedFrac = frac.replace(/0+$/, '');
  return trimmedFrac ? `${whole}.${trimmedFrac}` : whole;
}

export async function executeSwap(request: SwapRequest): Promise<SwapResponse> {
  const resp = await apiRequest<{ success: boolean; data?: SwapResponse; error?: string }>(
    'POST',
    '/dex/swap/execute',
    request
  );
  if (!resp.success || !resp.data) {
    throw new Error(resp.error || 'Swap failed');
  }
  return resp.data;
}

export async function getMiningStats(address: string): Promise<MiningStats> {
  const resp = await apiRequest<ApiDataResponse<MiningStats>>('GET', `/mining/stats/${address}`);
  if (!resp.data) {
    throw new Error('No mining stats available');
  }
  return resp.data;
}

export async function getNetworkStats(): Promise<{
  height: number;
  peers: number;
  tps: number;
  difficulty: number;
}> {
  // Reuse /node/status since /network/stats doesn't exist
  const health = await getHealth();
  return {
    height: health.height,
    peers: health.peers,
    tps: health.tps,
    difficulty: 0,
  };
}

export async function getNonce(address: string): Promise<{ nonce: number }> {
  return apiRequest('GET', `/nonce/${address}`);
}

/**
 * Login to the server vault (OAuth-style). Returns a session token.
 */
export async function loginVault(credentials: {
  wallet_address: string;
  signature: string;
  public_key: string;
  timestamp: number;
}): Promise<{ token: string; expires_at: number }> {
  return apiRequest('POST', '/auth/login', credentials);
}
