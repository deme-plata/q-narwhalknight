import React from 'react';

// Import cryptocurrency SVG icons from the installed package
import btcIcon from 'cryptocurrency-icons/svg/color/btc.svg';
import ethIcon from 'cryptocurrency-icons/svg/color/eth.svg';
import zecIcon from 'cryptocurrency-icons/svg/color/zec.svg';
import usdIcon from 'cryptocurrency-icons/svg/color/usd.svg';
import usdtIcon from 'cryptocurrency-icons/svg/color/usdt.svg';
import usdcIcon from 'cryptocurrency-icons/svg/color/usdc.svg';
import wbtcIcon from 'cryptocurrency-icons/svg/color/wbtc.svg';
import daiIcon from 'cryptocurrency-icons/svg/color/dai.svg';
import linkIcon from 'cryptocurrency-icons/svg/color/link.svg';
import aaveIcon from 'cryptocurrency-icons/svg/color/aave.svg';
import uniIcon from 'cryptocurrency-icons/svg/color/uni.svg';

// Map token symbols to their icons
const ICON_MAP: Record<string, string> = {
  // Bridge tokens
  'wBTC': wbtcIcon,
  'WBTC': wbtcIcon,
  'wETH': ethIcon,
  'WETH': ethIcon,
  'wZEC': zecIcon,
  'WZEC': zecIcon,
  'ETH': ethIcon,
  'BTC': btcIcon,
  'ZEC': zecIcon,
  // Stablecoins
  'USDT': usdtIcon,
  'USDC': usdcIcon,
  'DAI': daiIcon,
  // DeFi
  'LINK': linkIcon,
  'AAVE': aaveIcon,
  'UNI': uniIcon,
  // Fiat
  'USD': usdIcon,
};

// Color palette for fallback letter icons
const SYMBOL_COLORS: Record<string, { bg: string; ring: string }> = {
  'wBTC':   { bg: 'from-orange-500 to-amber-600', ring: 'ring-orange-400/50' },
  'WBTC':   { bg: 'from-orange-500 to-amber-600', ring: 'ring-orange-400/50' },
  'wETH':   { bg: 'from-blue-400 to-indigo-600', ring: 'ring-blue-400/50' },
  'WETH':   { bg: 'from-blue-400 to-indigo-600', ring: 'ring-blue-400/50' },
  'wZEC':   { bg: 'from-yellow-400 to-amber-500', ring: 'ring-yellow-400/50' },
  'WZEC':   { bg: 'from-yellow-400 to-amber-500', ring: 'ring-yellow-400/50' },
  'wIRON':  { bg: 'from-slate-400 to-zinc-600', ring: 'ring-slate-300/50' },
  'WIRON':  { bg: 'from-slate-400 to-zinc-600', ring: 'ring-slate-300/50' },
  'QUGUSD': { bg: 'from-emerald-500 to-green-600', ring: 'ring-emerald-400/50' },
  'USD':    { bg: 'from-emerald-500 to-green-600', ring: 'ring-emerald-400/50' },
};

// Iron Fish custom SVG (no icon in cryptocurrency-icons package)
const IronFishSVG: React.FC<{ size: number }> = ({ size }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="16" r="16" fill="#1D2C3A"/>
    <path d="M8 16c0-4.4 3.6-8 8-8s8 3.6 8 8-3.6 8-8 8-8-3.6-8-8z" stroke="#4AC1E0" strokeWidth="1.5" fill="none"/>
    <path d="M11 16h10M14 12l-3 4 3 4M18 12l3 4-3 4" stroke="#4AC1E0" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

interface TokenIconProps {
  symbol: string;
  icon?: string;
  logoUrl?: string;
  size?: number;       // px, default 32
  className?: string;
}

const TokenIcon: React.FC<TokenIconProps> = ({
  symbol,
  icon,
  logoUrl,
  size = 32,
  className = '',
}) => {
  const sizeClass = `w-${size <= 20 ? 5 : size <= 24 ? 6 : 8} h-${size <= 20 ? 5 : size <= 24 ? 6 : 8}`;

  // 1. QUG native — use the Quillon logo
  if (icon === 'qug-logo' || symbol === 'QUG') {
    return (
      <div
        className={`relative rounded-full overflow-hidden flex-shrink-0 ${className}`}
        style={{ width: size, height: size }}
      >
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
            padding: '1.5px',
          }}
        >
          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-0.5">
            <img
              src="/quillon-logo.png"
              alt="QUG"
              className="w-full h-full object-contain"
              style={{ filter: 'invert(1)' }}
            />
          </div>
        </div>
      </div>
    );
  }

  // 2. QUGUSD stablecoin — green ring + Quillon logo
  if (icon === 'qugusd-logo' || symbol === 'QUGUSD') {
    return (
      <div
        className={`relative rounded-full overflow-hidden flex-shrink-0 ${className}`}
        style={{ width: size, height: size }}
      >
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: 'linear-gradient(135deg, #10B981 0%, #34D399 50%, #10B981 100%)',
            padding: '1.5px',
          }}
        >
          <div className="w-full h-full bg-gradient-to-b from-slate-900 via-emerald-950 to-slate-900 rounded-full flex items-center justify-center p-0.5">
            <img
              src="/quillon-logo.png"
              alt="QUGUSD"
              className="w-full h-full object-contain"
              style={{ filter: 'invert(1) sepia(1) saturate(5) hue-rotate(90deg)' }}
            />
          </div>
        </div>
      </div>
    );
  }

  // 3. Iron Fish (wIRON) — custom SVG
  if (symbol === 'wIRON' || symbol === 'WIRON' || symbol === 'IRON') {
    return (
      <div
        className={`rounded-full overflow-hidden flex-shrink-0 ${className}`}
        style={{ width: size, height: size }}
      >
        <IronFishSVG size={size} />
      </div>
    );
  }

  // 4. Known crypto icon from cryptocurrency-icons package
  const knownIcon = ICON_MAP[symbol];
  if (knownIcon) {
    return (
      <div
        className={`rounded-full overflow-hidden flex-shrink-0 bg-white/10 ${className}`}
        style={{ width: size, height: size }}
      >
        <img
          src={knownIcon}
          alt={symbol}
          className="w-full h-full object-cover"
        />
      </div>
    );
  }

  // 5. Custom logoUrl from API
  if (logoUrl) {
    return (
      <div
        className={`rounded-full overflow-hidden flex-shrink-0 bg-white/10 ${className}`}
        style={{ width: size, height: size }}
      >
        <img
          src={logoUrl}
          alt={symbol}
          className="w-full h-full object-cover"
          onError={(e) => {
            // Fallback to letter icon if URL fails
            const parent = e.currentTarget.parentElement;
            if (parent) {
              e.currentTarget.style.display = 'none';
              parent.classList.add('fallback-letter-icon');
            }
          }}
        />
      </div>
    );
  }

  // 6. Fallback — stylized letter icon with gradient
  const colors = SYMBOL_COLORS[symbol] || {
    bg: 'from-violet-500 to-purple-600',
    ring: 'ring-violet-400/50',
  };
  const letter = symbol.replace(/^w/, '').charAt(0).toUpperCase();

  return (
    <div
      className={`rounded-full flex-shrink-0 bg-gradient-to-br ${colors.bg} ring-1 ${colors.ring} flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
    >
      <span
        className="font-bold text-white"
        style={{ fontSize: size * 0.45 }}
      >
        {letter}
      </span>
    </div>
  );
};

export default TokenIcon;
