import { useEffect, useRef } from 'react';
import { useDexStore } from '../stores/dexStore';

const DEBOUNCE_MS = 500;

/**
 * Hook that debounces DEX quote fetching when the input amount changes.
 */
export function useDexQuote() {
  const amountIn = useDexStore((s) => s.amountIn);
  const tokenIn = useDexStore((s) => s.tokenIn);
  const tokenOut = useDexStore((s) => s.tokenOut);
  const amountOut = useDexStore((s) => s.amountOut);
  const quote = useDexStore((s) => s.quote);
  const quoteLoading = useDexStore((s) => s.quoteLoading);
  const quoteError = useDexStore((s) => s.quoteError);
  const fetchQuote = useDexStore((s) => s.fetchQuote);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    if (!tokenIn || !tokenOut || !amountIn || parseFloat(amountIn) <= 0) {
      return;
    }

    timerRef.current = setTimeout(() => {
      fetchQuote();
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [amountIn, tokenIn, tokenOut, fetchQuote]);

  const priceImpact = quote ? quote.price_impact * 100 : 0; // Server sends as fraction (0.003), display as percent
  const fee = quote ? '0.3%' : '0'; // AMM uses constant 0.3% swap fee
  const rate =
    tokenIn && tokenOut && amountOut && parseFloat(amountOut) > 0
      ? `1 ${tokenIn.symbol} = ${(parseFloat(amountOut) / parseFloat(amountIn)).toFixed(6)} ${tokenOut.symbol}`
      : null;

  return {
    amountOut,
    quote,
    quoteLoading,
    quoteError,
    priceImpact,
    fee,
    rate,
  };
}
