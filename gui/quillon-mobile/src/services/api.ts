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
  'https://89.149.241.126:8080/api/v1', // Epsilon direct fallback
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
  token_in: string;
  token_out: string;
  amount_in: string;
  amount_out: string;
  price_impact: number;
  fee: string;
  route: string[];
  expires_at: number;
}

export interface SwapRequest {
  token_in: string;
  token_out: string;
  amount_in: string;
  min_amount_out: string;
  slippage_bps: number;
  sender: string;
  signature: string;
  public_key: string;
}

export interface SwapResponse {
  tx_hash: string;
  amount_out: string;
  status: string;
}

export interface MiningStats {
  hashrate: number;
  blocks_found: number;
  reward_total: string;
  difficulty: number;
  network_hashrate: number;
  last_block_time: number;
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
  timeout: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(id);
  }
}

async function apiRequest<T>(
  method: string,
  path: string,
  body?: unknown,
  retries: number = MAX_RETRIES,
  extraHeaders?: Record<string, string>
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const url = `${getBaseUrl()}${path}`;
      const options: RequestInit = {
        method,
        headers: buildHeaders(extraHeaders),
      };

      if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
        options.body = JSON.stringify(body);
      }

      const response = await fetchWithTimeout(url, options);

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(`API ${response.status}: ${errorText}`);
      }

      const data = (await response.json()) as T;
      return data;
    } catch (error) {
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
  pageSize: number = 20
): Promise<HistoryResponse> {
  // Server endpoint: /api/v1/wallet/:address/history
  // Returns { success: true, data: UnifiedTransactionEntry[] }
  const resp = await apiRequest<ApiDataResponse<Transaction[]>>(
    'GET',
    `/wallet/${address}/history`
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

export async function getMultiTokenBalance(address: string): Promise<MultiTokenBalance> {
  const path = '/wallet/tokens';
  const walletAuth = await generateWalletAuthHeader(address, `/api/v1${path}`);
  const headers: Record<string, string> = {};
  if (walletAuth) {
    headers['X-Wallet-Auth'] = walletAuth;
  }

  const resp = await apiRequest<ApiDataResponse<MultiTokenBalance>>(
    'GET', path, undefined, MAX_RETRIES, headers
  );
  return resp.data ?? { address, tokens: {}, total_usd_value: 0 };
}

export async function getDexTokens(): Promise<DexToken[]> {
  const resp = await apiRequest<ApiDataResponse<DexToken[]>>('GET', '/dex/tokens');
  return resp.data ?? [];
}

export async function getDexQuote(
  tokenIn: string,
  tokenOut: string,
  amountIn: string
): Promise<DexQuote> {
  return apiRequest<DexQuote>(
    'GET',
    `/dex/quote?token_in=${tokenIn}&token_out=${tokenOut}&amount_in=${amountIn}`
  );
}

export async function executeSwap(request: SwapRequest): Promise<SwapResponse> {
  return apiRequest<SwapResponse>('POST', '/dex/swap', request);
}

export async function getMiningStats(address: string): Promise<MiningStats> {
  return apiRequest<MiningStats>('GET', `/mining/stats/${address}`);
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
