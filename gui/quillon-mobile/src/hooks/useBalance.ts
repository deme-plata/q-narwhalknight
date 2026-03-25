import { useEffect, useRef, useCallback } from 'react';
import { useWalletStore } from '../stores/walletStore';
import { useSSEEvent } from './useSSE';

const POLL_INTERVAL_MS = 30_000; // Fallback polling every 30s

/**
 * Hook that subscribes to balance updates via SSE and falls back to polling.
 * Automatically refreshes balance when the wallet address changes.
 */
export function useBalance() {
  const address = useWalletStore((s) => s.address);
  const qugBalance = useWalletStore((s) => s.qugBalance);
  const tokenBalances = useWalletStore((s) => s.tokenBalances);
  const refreshBalance = useWalletStore((s) => s.refreshBalance);
  const setBalance = useWalletStore((s) => s.setBalance);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // SSE balance update handler
  useSSEEvent('balance', useCallback((data: unknown) => {
    const update = data as {
      balance?: number;
      token_balances?: Record<string, number>;
    };

    if (update.balance !== undefined) {
      const tokens = update.token_balances
        ? Object.entries(update.token_balances).map(([symbol, balance]) => ({
            symbol,
            name: symbol,
            balance,
            rawBalance: balance.toString(),
            decimals: 24,
            priceUsd: 0,
            valueUsd: 0,
          }))
        : tokenBalances;

      setBalance(update.balance, tokens);
    }
  }, [setBalance, tokenBalances]));

  // Initial fetch + polling fallback
  useEffect(() => {
    if (!address) return;

    // Initial fetch
    refreshBalance();

    // Polling fallback
    pollTimerRef.current = setInterval(() => {
      refreshBalance();
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [address, refreshBalance]);

  return {
    qugBalance,
    tokenBalances,
    refresh: refreshBalance,
  };
}
