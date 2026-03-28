import React, { useEffect, useState, useCallback } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { Text, Button, TextInput, IconButton, Chip, Snackbar, ActivityIndicator } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { COLORS, tokenColor } from '../../src/theme';
import { useDexStore } from '../../src/stores/dexStore';
import { useWalletStore } from '../../src/stores/walletStore';
import { useDexQuote } from '../../src/hooks/useDexQuote';
import { getMnemonic } from '../../src/services/secureStorage';
import { signTransaction } from '../../src/services/wallet';
import type { DexToken } from '../../src/services/api';

function DexTokenIcon({ symbol }: { symbol: string }) {
  const { bg, fg } = tokenColor(symbol);
  return (
    <View style={[styles.tokenIcon, { backgroundColor: bg }]}>
      <Text style={[styles.tokenIconText, { color: fg }]}>
        {symbol.slice(0, 2).toUpperCase()}
      </Text>
    </View>
  );
}

function TokenSelector({
  label,
  token,
  onPress,
}: {
  label: string;
  token: DexToken | null;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.tokenSelector} onPress={onPress} activeOpacity={0.7}>
      <Text variant="labelSmall" style={styles.tokenSelectorLabel}>
        {label}
      </Text>
      <View style={styles.tokenSelectorContent}>
        {token ? (
          <>
            <DexTokenIcon symbol={token.symbol} />
            <Text variant="titleMedium" style={styles.tokenSymbol}>
              {token.symbol}
            </Text>
          </>
        ) : (
          <Text variant="bodyMedium" style={styles.selectTokenText}>
            Select token
          </Text>
        )}
        <MaterialCommunityIcons name="chevron-down" size={20} color="rgba(255,255,255,0.5)" />
      </View>
    </TouchableOpacity>
  );
}

export default function DexScreen() {
  const address = useWalletStore((s) => s.address);
  const loadTokens = useDexStore((s) => s.loadTokens);
  const availableTokens = useDexStore((s) => s.availableTokens);
  const tokenIn = useDexStore((s) => s.tokenIn);
  const tokenOut = useDexStore((s) => s.tokenOut);
  const amountIn = useDexStore((s) => s.amountIn);
  const setTokenIn = useDexStore((s) => s.setTokenIn);
  const setTokenOut = useDexStore((s) => s.setTokenOut);
  const setAmountIn = useDexStore((s) => s.setAmountIn);
  const swapTokens = useDexStore((s) => s.swapTokens);
  const slippageBps = useDexStore((s) => s.slippageBps);
  const setSlippage = useDexStore((s) => s.setSlippage);
  const swapping = useDexStore((s) => s.swapping);
  const executeSwap = useDexStore((s) => s.executeSwap);
  const reset = useDexStore((s) => s.reset);

  const { amountOut, quoteLoading, quoteError, priceImpact, fee, rate } = useDexQuote();

  const [selectingFor, setSelectingFor] = useState<'in' | 'out' | null>(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    loadTokens();
  }, [loadTokens]);

  const handleSelectToken = useCallback(
    (token: DexToken) => {
      if (selectingFor === 'in') setTokenIn(token);
      else if (selectingFor === 'out') setTokenOut(token);
      setSelectingFor(null);
    },
    [selectingFor, setTokenIn, setTokenOut]
  );

  const handleSwap = async () => {
    if (!address) return;

    try {
      const mnemonic = await getMnemonic();
      if (!mnemonic) throw new Error('Wallet not found');

      const payload = {
        action: 'dex_swap',
        sender: address,
        token_in: tokenIn?.address,
        token_out: tokenOut?.address,
        amount_in: amountIn,
        timestamp: Math.floor(Date.now() / 1000),
      };

      const { signature, publicKey } = await signTransaction(mnemonic, payload);
      await executeSwap(address, signature, publicKey);
      setSuccess('Swap executed successfully!');
      reset();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Swap failed');
    }
  };

  // Token selector modal
  if (selectingFor) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.selectorHeader}>
          <Text variant="titleMedium" style={styles.selectorTitle}>
            Select Token
          </Text>
          <IconButton
            icon="close"
            iconColor="rgba(255,255,255,0.6)"
            onPress={() => setSelectingFor(null)}
          />
        </View>
        <ScrollView contentContainerStyle={styles.tokenList}>
          {availableTokens.map((token) => (
            <TouchableOpacity
              key={token.address}
              style={styles.tokenListItem}
              onPress={() => handleSelectToken(token)}
              activeOpacity={0.6}
            >
              <DexTokenIcon symbol={token.symbol} />
              <View style={styles.tokenListInfo}>
                <Text variant="bodyLarge" style={styles.tokenListName}>
                  {token.symbol}
                </Text>
                <Text variant="bodySmall" style={styles.tokenListFullName}>
                  {token.name}
                </Text>
              </View>
            </TouchableOpacity>
          ))}
          {availableTokens.length === 0 && (
            <Text style={styles.noTokens}>No tokens available</Text>
          )}
        </ScrollView>
      </SafeAreaView>
    );
  }

  const canSwap = tokenIn && tokenOut && parseFloat(amountIn) > 0 && amountOut && !quoteLoading;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text variant="headlineSmall" style={styles.title}>
          Swap
        </Text>

        {/* From Token */}
        <View style={styles.swapCardOuter}>
          <LinearGradient
            colors={['rgba(0, 188, 212, 0.06)', 'rgba(26, 26, 38, 0.95)']}
            start={{ x: 0, y: 0 }}
            end={{ x: 0, y: 1 }}
            style={styles.swapCard}
          >
            <TokenSelector label="You Pay" token={tokenIn} onPress={() => setSelectingFor('in')} />
            <TextInput
              value={amountIn}
              onChangeText={setAmountIn}
              mode="flat"
              placeholder="0.0"
              placeholderTextColor="rgba(255, 255, 255, 0.3)"
              style={styles.swapInput}
              textColor="#FFFFFF"
              keyboardType="decimal-pad"
              underlineColor="transparent"
              activeUnderlineColor="transparent"
            />
          </LinearGradient>
        </View>

        {/* Swap Direction Button */}
        <View style={styles.swapButtonContainer}>
          <TouchableOpacity onPress={swapTokens} activeOpacity={0.7}>
            <LinearGradient
              colors={['rgba(0, 188, 212, 0.2)', 'rgba(124, 77, 255, 0.15)']}
              style={styles.swapDirectionButton}
            >
              <MaterialCommunityIcons name="swap-vertical" size={24} color={COLORS.cyan} />
            </LinearGradient>
          </TouchableOpacity>
        </View>

        {/* To Token */}
        <View style={styles.swapCardOuter}>
          <LinearGradient
            colors={['rgba(124, 77, 255, 0.06)', 'rgba(26, 26, 38, 0.95)']}
            start={{ x: 0, y: 0 }}
            end={{ x: 0, y: 1 }}
            style={styles.swapCard}
          >
            <TokenSelector label="You Receive" token={tokenOut} onPress={() => setSelectingFor('out')} />
            <View style={styles.outputRow}>
              {quoteLoading ? (
                <ActivityIndicator size="small" color={COLORS.cyan} />
              ) : (
                <Text variant="headlineSmall" style={styles.outputAmount}>
                  {amountOut || '0.0'}
                </Text>
              )}
            </View>
          </LinearGradient>
        </View>

        {/* Quote Details */}
        {rate && (
          <View style={styles.detailsCardOuter}>
            <LinearGradient
              colors={['rgba(0, 188, 212, 0.04)', 'rgba(20, 20, 32, 0.9)']}
              style={styles.detailsCard}
            >
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Rate</Text>
                <Text style={styles.detailValue}>{rate}</Text>
              </View>
              <View style={styles.detailDivider} />
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Price Impact</Text>
                <Text
                  style={[
                    styles.detailValue,
                    priceImpact > 3 ? { color: COLORS.red } : priceImpact > 1 ? { color: COLORS.orange } : {},
                  ]}
                >
                  {priceImpact.toFixed(2)}%
                </Text>
              </View>
              <View style={styles.detailDivider} />
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Network Fee</Text>
                <Text style={styles.detailValue}>{fee}</Text>
              </View>
              <View style={styles.detailDivider} />
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Slippage</Text>
                <View style={styles.slippageRow}>
                  {[50, 100, 300].map((bps) => (
                    <Chip
                      key={bps}
                      compact
                      selected={slippageBps === bps}
                      onPress={() => setSlippage(bps)}
                      style={[
                        styles.slippageChip,
                        slippageBps === bps && styles.slippageChipActive,
                      ]}
                      textStyle={[
                        styles.slippageChipText,
                        slippageBps === bps && styles.slippageChipTextActive,
                      ]}
                    >
                      {(bps / 100).toFixed(1)}%
                    </Chip>
                  ))}
                </View>
              </View>
            </LinearGradient>
          </View>
        )}

        {quoteError && (
          <Text variant="bodySmall" style={styles.quoteError}>
            {quoteError}
          </Text>
        )}

        {/* Swap Button */}
        <Button
          mode="contained"
          onPress={handleSwap}
          loading={swapping}
          disabled={!canSwap || swapping}
          style={styles.swapExecuteButton}
          buttonColor={COLORS.cyan}
          textColor="#000000"
          icon="swap-horizontal"
          contentStyle={styles.buttonContent}
          labelStyle={{ fontWeight: '700' }}
        >
          {!tokenIn || !tokenOut
            ? 'Select Tokens'
            : !amountIn || parseFloat(amountIn) <= 0
            ? 'Enter Amount'
            : quoteLoading
            ? 'Getting Quote...'
            : 'Swap'}
        </Button>
      </ScrollView>

      <Snackbar visible={!!error} onDismiss={() => setError('')} duration={4000} style={styles.errorSnackbar}>
        {error}
      </Snackbar>
      <Snackbar visible={!!success} onDismiss={() => setSuccess('')} duration={4000} style={styles.successSnackbar}>
        {success}
      </Snackbar>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.darkBg,
  },
  scrollContent: {
    padding: 24,
  },
  title: {
    color: '#FFFFFF',
    fontWeight: '700',
    marginBottom: 24,
  },
  swapCardOuter: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.08)',
    overflow: 'hidden',
  },
  swapCard: {
    padding: 16,
  },
  tokenSelector: {
    marginBottom: 12,
  },
  tokenSelectorLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  tokenSelectorContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  tokenIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tokenIconText: {
    fontWeight: '700',
    fontSize: 12,
  },
  tokenSymbol: {
    color: '#FFFFFF',
    fontWeight: '600',
    flex: 1,
  },
  selectTokenText: {
    color: 'rgba(255, 255, 255, 0.4)',
    flex: 1,
  },
  swapInput: {
    backgroundColor: 'transparent',
    fontSize: 24,
    fontWeight: '600',
    paddingHorizontal: 0,
  },
  swapButtonContainer: {
    alignItems: 'center',
    marginVertical: -12,
    zIndex: 10,
  },
  swapDirectionButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: COLORS.darkBg,
  },
  outputRow: {
    minHeight: 40,
    justifyContent: 'center',
  },
  outputAmount: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  detailsCardOuter: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0, 188, 212, 0.06)',
    overflow: 'hidden',
    marginTop: 16,
  },
  detailsCard: {
    padding: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  detailDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
  },
  detailLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 13,
  },
  detailValue: {
    color: '#FFFFFF',
    fontSize: 13,
  },
  slippageRow: {
    flexDirection: 'row',
    gap: 6,
  },
  slippageChip: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
    height: 28,
  },
  slippageChipActive: {
    backgroundColor: 'rgba(0, 188, 212, 0.15)',
    borderColor: COLORS.cyan,
  },
  slippageChipText: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 11,
  },
  slippageChipTextActive: {
    color: COLORS.cyan,
  },
  quoteError: {
    color: COLORS.red,
    textAlign: 'center',
    marginTop: 8,
  },
  swapExecuteButton: {
    marginTop: 24,
    borderRadius: 14,
    elevation: 4,
    shadowColor: COLORS.cyan,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  buttonContent: {
    paddingVertical: 6,
  },
  errorSnackbar: {
    backgroundColor: COLORS.red,
  },
  successSnackbar: {
    backgroundColor: COLORS.green,
  },
  // Token selector
  selectorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  selectorTitle: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  tokenList: {
    padding: 16,
    gap: 4,
  },
  tokenListItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 12,
    gap: 12,
  },
  tokenListInfo: {
    flex: 1,
  },
  tokenListName: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  tokenListFullName: {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  noTokens: {
    color: 'rgba(255, 255, 255, 0.4)',
    textAlign: 'center',
    padding: 40,
  },
});
