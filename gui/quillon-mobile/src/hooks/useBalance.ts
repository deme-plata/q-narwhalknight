import { useEffect, useRef, useCallback } from 'react';
import { useWalletStore } from '../stores/walletStore';
import { useSSEEvent } from './useSSE';

const POLL_INTERVAL_MS = 60_000; // Fallback polling every 60s (only when SSE is down)

/**
 * Hook that subscribes to real-time balance updates via SSE.
 *
 * The server pushes `balance-updated` and `token-balance-updated` events
 * whenever a wallet's balance changes (mining reward, transfer, swap, etc.).
 * On each event we apply the new balance instantly from the SSE payload,
 * then kick off a full REST refresh for accurate multi-token data.
 *
 * Polling is only a fallback for when SSE is disconnected.
 */
export function useBalance() {
  const address = useWalletStore((s) => s.address);
  const qugBalance = useWalletStore((s) => s.qugBalance);
  const tokenBalances = useWalletStore((s) => s.tokenBalances);
  const refreshBalance = useWalletStore((s) => s.refreshBalance);
  const setBalance = useWalletStore((s) => s.setBalance);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);
  // Debounce REST refreshes triggered by rapid SSE events
  const refreshDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Debounced REST refresh — coalesces rapid SSE bursts into one API call */
  const debouncedRefresh = useCallback(() => {
    if (refreshDebounceRef.current) clearTimeout(refreshDebounceRef.current);
    refreshDebounceRef.current = setTimeout(() => {
      if (mountedRef.current) refreshBalance();
    }, 300); // 300ms debounce
  }, [refreshBalance]);

  // SSE balance-updated handler (QUG balance changes)
  useSSEEvent('balance-updated', useCallback((data: unknown) => {
    if (!mountedRef.current) return;

    const update = data as {
      wallet_address?: string;
      new_balance?: number;
      old_balance?: number;
      change_reason?: string;
    };

    // Instant visual update from SSE payload
    if (update.new_balance !== undefined) {
      setBalance(update.new_balance, tokenBalances);
    }

    // Also trigger a full REST refresh for accurate multi-token data
    debouncedRefresh();
  }, [setBalance, tokenBalances, debouncedRefresh]));

  // SSE token-balance-updated handler (token balance changes — QUGUSD, custom tokens)
  useSSEEvent('token-balance-updated', useCallback((_data: unknown) => {
    if (!mountedRef.current) return;
    // Token balance changed — refresh full token list via REST
    debouncedRefresh();
  }, [debouncedRefresh]));

  // Initial fetch + slow polling fallback (only covers SSE outages)
  useEffect(() => {
    if (!address) return;

    mountedRef.current = true;

    // Initial fetch
    refreshBalance();

    // Slow polling fallback (SSE is the primary update channel)
    pollTimerRef.current = setInterval(() => {
      if (mountedRef.current) {
        refreshBalance();
      }
    }, POLL_INTERVAL_MS);

    return () => {
      mountedRef.current = false;
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      if (refreshDebounceRef.current) {
        clearTimeout(refreshDebounceRef.current);
        refreshDebounceRef.current = null;
      }
    };
  }, [address, refreshBalance]);

  return {
    qugBalance,
    tokenBalances,
    refresh: refreshBalance,
  };
}
