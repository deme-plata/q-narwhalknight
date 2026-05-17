import { MD3DarkTheme, configureFonts } from 'react-native-paper';
import type { MD3Theme } from 'react-native-paper';

const fontConfig = {
  displayLarge: { fontFamily: 'System', fontWeight: '400' as const },
  displayMedium: { fontFamily: 'System', fontWeight: '400' as const },
  displaySmall: { fontFamily: 'System', fontWeight: '400' as const },
  headlineLarge: { fontFamily: 'System', fontWeight: '400' as const },
  headlineMedium: { fontFamily: 'System', fontWeight: '400' as const },
  headlineSmall: { fontFamily: 'System', fontWeight: '400' as const },
  titleLarge: { fontFamily: 'System', fontWeight: '500' as const },
  titleMedium: { fontFamily: 'System', fontWeight: '500' as const },
  titleSmall: { fontFamily: 'System', fontWeight: '500' as const },
  bodyLarge: { fontFamily: 'System', fontWeight: '400' as const },
  bodyMedium: { fontFamily: 'System', fontWeight: '400' as const },
  bodySmall: { fontFamily: 'System', fontWeight: '400' as const },
  labelLarge: { fontFamily: 'System', fontWeight: '500' as const },
  labelMedium: { fontFamily: 'System', fontWeight: '500' as const },
  labelSmall: { fontFamily: 'System', fontWeight: '500' as const },
};

export const quilTheme: MD3Theme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    primary: '#00BCD4',
    onPrimary: '#003738',
    primaryContainer: '#004F51',
    onPrimaryContainer: '#6FF7FB',
    secondary: '#7C4DFF',
    onSecondary: '#1F0060',
    secondaryContainer: '#3A1D8E',
    onSecondaryContainer: '#E4DFFF',
    tertiary: '#FF6E40',
    onTertiary: '#3F1100',
    tertiaryContainer: '#5C1A00',
    onTertiaryContainer: '#FFDBCE',
    error: '#FFB4AB',
    onError: '#690005',
    errorContainer: '#93000A',
    onErrorContainer: '#FFDAD6',
    background: '#0A0A10',
    onBackground: '#E4E1E6',
    surface: '#0A0A10',
    onSurface: '#E4E1E6',
    surfaceVariant: '#1A1A26',
    onSurfaceVariant: '#C7C5D0',
    outline: '#918F9A',
    outlineVariant: '#46464F',
    inverseSurface: '#E4E1E6',
    inverseOnSurface: '#313036',
    inversePrimary: '#006A6C',
    elevation: {
      level0: 'transparent',
      level1: '#141420',
      level2: '#1A1A28',
      level3: '#20202F',
      level4: '#222232',
      level5: '#262638',
    },
    surfaceDisabled: 'rgba(228, 225, 230, 0.12)',
    onSurfaceDisabled: 'rgba(228, 225, 230, 0.38)',
    backdrop: 'rgba(0, 0, 0, 0.5)',
    scrim: '#000000',
    shadow: '#000000',
  },
  fonts: configureFonts({ config: fontConfig }),
};

export const COLORS = {
  cyan: '#00BCD4',
  cyanDark: '#008BA3',
  cyanLight: '#62EFFF',
  purple: '#7C4DFF',
  purpleDark: '#3F1DCB',
  purpleLight: '#B47CFF',
  orange: '#FF6E40',
  green: '#4CAF50',
  red: '#F44336',
  gold: '#FFD700',
  darkBg: '#0A0A10',
  cardBg: '#141420',
  surfaceBg: '#1A1A26',
  gradientStart: '#00BCD4',
  gradientMid: '#5E35B1',
  gradientEnd: '#7C4DFF',
} as const;

/** Gradient presets for LinearGradient components */
export const GRADIENTS = {
  /** Cyan → purple horizontal */
  primary: ['#00BCD4', '#7C4DFF'] as const,
  /** Subtle dark gradient for card backgrounds */
  card: ['rgba(0, 188, 212, 0.08)', 'rgba(124, 77, 255, 0.06)'] as const,
  /** Deep dark background gradient */
  background: ['#0A0A10', '#0E0E1A', '#0A0A10'] as const,
  /** Accent glow for hero sections */
  heroGlow: ['rgba(0, 188, 212, 0.15)', 'rgba(124, 77, 255, 0.08)', 'transparent'] as const,
  /** Button gradient */
  button: ['#00BCD4', '#00A5B8'] as const,
  /** Success */
  success: ['#4CAF50', '#2E7D32'] as const,
  /** Danger */
  danger: ['#F44336', '#C62828'] as const,
} as const;

/**
 * Generate a deterministic color from a token symbol.
 * Used for token icons to give each token a unique hue.
 */
export function tokenColor(symbol: string): { bg: string; fg: string } {
  const TOKEN_COLORS: Record<string, { bg: string; fg: string }> = {
    QUG: { bg: 'rgba(0, 188, 212, 0.18)', fg: '#00BCD4' },
    ETH: { bg: 'rgba(98, 126, 234, 0.18)', fg: '#627EEA' },
    WETH: { bg: 'rgba(98, 126, 234, 0.18)', fg: '#627EEA' },
    BTC: { bg: 'rgba(247, 147, 26, 0.18)', fg: '#F7931A' },
    WBTC: { bg: 'rgba(247, 147, 26, 0.18)', fg: '#F7931A' },
    USDT: { bg: 'rgba(38, 161, 123, 0.18)', fg: '#26A17B' },
    USDC: { bg: 'rgba(39, 117, 202, 0.18)', fg: '#2775CA' },
    DAI: { bg: 'rgba(245, 172, 55, 0.18)', fg: '#F5AC37' },
  };

  if (TOKEN_COLORS[symbol.toUpperCase()]) {
    return TOKEN_COLORS[symbol.toUpperCase()];
  }

  // Hash-based fallback for unknown tokens
  let hash = 0;
  for (let i = 0; i < symbol.length; i++) {
    hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = ((hash % 360) + 360) % 360;
  return {
    bg: `hsla(${hue}, 70%, 55%, 0.18)`,
    fg: `hsl(${hue}, 70%, 65%)`,
  };
}

/** Common shadow for floating elements */
export const GLOW_SHADOW = {
  shadowColor: '#00BCD4',
  shadowOffset: { width: 0, height: 0 },
  shadowOpacity: 0.25,
  shadowRadius: 12,
  elevation: 8,
};
