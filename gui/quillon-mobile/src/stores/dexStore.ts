import { create } from 'zustand';
import * as api from '../services/api';

interface DexState {
  // Token selection
  tokenIn: api.DexToken | null;
  tokenOut: api.DexToken | null;
  availableTokens: api.DexToken[];

  // Amounts
  amountIn: string;
  amountOut: string;

  // Quote
  quote: api.DexQuote | null;
  quoteLoading: boolean;
  quoteError: string | null;

  // Settings
  slippageBps: number; // basis points (50 = 0.5%)

  // Swap state
  swapping: boolean;
  swapResult: api.SwapResponse | null;
  swapError: string | null;

  // Actions
  loadTokens: () => Promise<void>;
  setTokenIn: (token: api.DexToken) => void;
  setTokenOut: (token: api.DexToken) => void;
  setAmountIn: (amount: string) => void;
  setSlippage: (bps: number) => void;
  swapTokens: () => void;
  fetchQuote: () => Promise<void>;
  executeSwap: (sender: string, signature: string, publicKey: string) => Promise<void>;
  reset: () => void;
}

export const useDexStore = create<DexState>((set, get) => ({
  tokenIn: null,
  tokenOut: null,
  availableTokens: [],
  amountIn: '',
  amountOut: '',
  quote: null,
  quoteLoading: false,
  quoteError: null,
  slippageBps: 50,
  swapping: false,
  swapResult: null,
  swapError: null,

  loadTokens: async () => {
    try {
      const tokens = await api.getDexTokens();
      set({ availableTokens: tokens });

      // Default: QUG as tokenIn if available
      const qug = tokens.find((t) => t.symbol === 'QUG');
      if (qug && !get().tokenIn) {
        set({ tokenIn: qug });
      }
    } catch (error) {
      console.error('[DexStore] Load tokens error:', error);
    }
  },

  setTokenIn: (token) => {
    const { tokenOut } = get();
    // Prevent same token on both sides
    if (tokenOut && tokenOut.address === token.address) {
      set({ tokenIn: token, tokenOut: null, quote: null, amountOut: '' });
    } else {
      set({ tokenIn: token, quote: null, amountOut: '' });
    }
  },

  setTokenOut: (token) => {
    const { tokenIn } = get();
    if (tokenIn && tokenIn.address === token.address) {
      set({ tokenOut: token, tokenIn: null, quote: null, amountOut: '' });
    } else {
      set({ tokenOut: token, quote: null, amountOut: '' });
    }
  },

  setAmountIn: (amount) => {
    // Only allow valid numeric input
    if (amount === '' || /^\d*\.?\d*$/.test(amount)) {
      set({ amountIn: amount, quote: null, amountOut: '' });
    }
  },

  setSlippage: (bps) => {
    if (bps >= 1 && bps <= 5000) {
      set({ slippageBps: bps });
    }
  },

  swapTokens: () => {
    const { tokenIn, tokenOut } = get();
    set({
      tokenIn: tokenOut,
      tokenOut: tokenIn,
      amountIn: '',
      amountOut: '',
      quote: null,
    });
  },

  fetchQuote: async () => {
    const { tokenIn, tokenOut, amountIn } = get();
    if (!tokenIn || !tokenOut || !amountIn || parseFloat(amountIn) <= 0) {
      set({ quote: null, amountOut: '' });
      return;
    }

    try {
      set({ quoteLoading: true, quoteError: null });
      const quote = await api.getDexQuote(tokenIn.address, tokenOut.address, amountIn);
      set({
        quote,
        amountOut: quote.amount_out,
        quoteLoading: false,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to get quote';
      set({ quoteError: message, quoteLoading: false, amountOut: '' });
    }
  },

  executeSwap: async (sender, signature, publicKey) => {
    const { tokenIn, tokenOut, amountIn, amountOut, slippageBps, quote } = get();
    if (!tokenIn || !tokenOut || !quote) {
      throw new Error('Incomplete swap parameters');
    }

    // Calculate minimum output with slippage
    const minOut = (parseFloat(amountOut) * (10000 - slippageBps)) / 10000;

    try {
      set({ swapping: true, swapError: null });
      const result = await api.executeSwap({
        token_in: tokenIn.address,
        token_out: tokenOut.address,
        amount_in: amountIn,
        min_amount_out: minOut.toString(),
        slippage_bps: slippageBps,
        sender,
        signature,
        public_key: publicKey,
      });
      set({ swapResult: result, swapping: false });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Swap failed';
      set({ swapError: message, swapping: false });
      throw error;
    }
  },

  reset: () => {
    set({
      amountIn: '',
      amountOut: '',
      quote: null,
      quoteError: null,
      swapResult: null,
      swapError: null,
    });
  },
}));
