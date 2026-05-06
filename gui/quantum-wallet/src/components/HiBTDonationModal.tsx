import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Copy, CheckCircle, Bitcoin, ExternalLink, Zap, Target } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';

const DONATION_ADDRESS = 'bc1qnqdj5kuka522kctk4v99l22jpjut3lums2kepl';
const GOAL_USD = 15000;

interface DonationStatus {
  address: string;
  goal_usd: number;
  goal_btc: number;
  received_btc: number;
  received_usd: number;
  btc_price_usd: number;
  percent_complete: number;
  campaign: string;
  exchange_url: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export default function HiBTDonationModal({ isOpen, onClose }: Props) {
  const [status, setStatus] = useState<DonationStatus | null>(null);
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<'onchain' | 'lightning'>('onchain');

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/donation/hibt-status');
      const data = await res.json();
      if (data.success) setStatus(data.data);
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchStatus();
      const interval = setInterval(fetchStatus, 30_000);
      return () => clearInterval(interval);
    }
  }, [isOpen, fetchStatus]);

  const copy = () => {
    navigator.clipboard.writeText(DONATION_ADDRESS);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const pct = status?.percent_complete ?? 0;
  const receivedUsd = status?.received_usd ?? 0;
  const receivedBtc = status?.received_btc ?? 0;
  const btcPrice = status?.btc_price_usd ?? 81000;
  const goalBtc = status?.goal_btc ?? (GOAL_USD / btcPrice);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}
          onClick={e => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.93, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.93, y: 20 }}
            transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            className="relative w-full max-w-md rounded-3xl overflow-hidden"
            style={{
              background: 'linear-gradient(160deg, #0a0f0a 0%, #0d1a0d 40%, #091209 100%)',
              border: '1.5px solid rgba(132,204,22,0.3)',
              boxShadow: '0 0 60px rgba(132,204,22,0.15), 0 24px 64px rgba(0,0,0,0.7)',
            }}
          >
            {/* Banner image */}
            <div className="relative w-full overflow-hidden" style={{ height: 80 }}>
              <img
                src="/hibt-banner.png"
                alt="HiBT QUG Listing"
                className="w-full h-full object-cover object-center"
              />
              <div className="absolute inset-0" style={{ background: 'linear-gradient(to bottom, transparent 60%, #0a0f0a 100%)' }} />
            </div>

            {/* Close */}
            <button
              onClick={onClose}
              className="absolute top-3 right-3 p-1.5 rounded-full text-gray-400 hover:text-white transition-colors"
              style={{ background: 'rgba(0,0,0,0.5)' }}
            >
              <X size={16} />
            </button>

            <div className="px-5 pb-5 pt-2 space-y-4">
              {/* Title */}
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <Bitcoin size={18} className="text-orange-400" />
                  <h2 className="text-base font-bold text-white">Community Listing Fund</h2>
                </div>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Help get <span className="text-lime-400 font-semibold">$QUG</span> listed on{' '}
                  <a href="https://hibt.com" target="_blank" rel="noopener noreferrer" className="text-lime-400 hover:underline inline-flex items-center gap-0.5">
                    HiBT Exchange <ExternalLink size={10} />
                  </a>
                  . 100% of donations go toward the{' '}
                  <span className="text-white font-medium">${GOAL_USD.toLocaleString()} listing fee</span>.
                </p>
              </div>

              {/* Progress */}
              <div className="rounded-2xl p-3.5 space-y-2" style={{ background: 'rgba(132,204,22,0.06)', border: '1px solid rgba(132,204,22,0.15)' }}>
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5">
                    <Target size={12} className="text-lime-400" />
                    <span className="text-gray-400">Raised</span>
                  </div>
                  <span className="text-white font-semibold">
                    ${receivedUsd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{' '}
                    <span className="text-gray-500 font-normal">/ ${GOAL_USD.toLocaleString()}</span>
                  </span>
                </div>

                {/* Progress bar */}
                <div className="relative h-2.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(pct, pct > 0 ? 2 : 0)}%` }}
                    transition={{ duration: 1, ease: 'easeOut' }}
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{ background: 'linear-gradient(90deg, #84cc16, #a3e635)' }}
                  />
                </div>

                <div className="flex justify-between text-[10px] text-gray-500">
                  <span>{receivedBtc.toFixed(6)} BTC received</span>
                  <span className="text-lime-400/70 font-medium">{pct.toFixed(1)}% of goal</span>
                </div>
              </div>

              {/* Tab switcher */}
              <div className="flex rounded-xl overflow-hidden text-xs font-medium" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                {(['onchain', 'lightning'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className="flex-1 py-2 flex items-center justify-center gap-1.5 transition-all"
                    style={{
                      background: tab === t ? 'rgba(132,204,22,0.15)' : 'transparent',
                      color: tab === t ? '#a3e635' : '#6b7280',
                    }}
                  >
                    {t === 'onchain' ? <Bitcoin size={12} /> : <Zap size={12} />}
                    {t === 'onchain' ? 'On-Chain BTC' : 'Lightning'}
                  </button>
                ))}
              </div>

              {/* On-chain tab */}
              {tab === 'onchain' && (
                <motion.div
                  key="onchain"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-3"
                >
                  {/* QR code */}
                  <div className="flex justify-center">
                    <div className="rounded-2xl p-3" style={{ background: 'white' }}>
                      <QRCodeSVG
                        value={`bitcoin:${DONATION_ADDRESS}`}
                        size={160}
                        level="M"
                        includeMargin={false}
                        fgColor="#000000"
                        bgColor="#ffffff"
                      />
                    </div>
                  </div>

                  {/* Address */}
                  <div className="rounded-xl p-3 flex items-center gap-2" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div className="flex-1 font-mono text-[11px] text-green-300 break-all select-all leading-relaxed">
                      {DONATION_ADDRESS}
                    </div>
                    <button
                      onClick={copy}
                      className="flex-shrink-0 p-2 rounded-lg transition-all"
                      style={{ background: copied ? 'rgba(132,204,22,0.2)' : 'rgba(255,255,255,0.06)', color: copied ? '#a3e635' : '#9ca3af' }}
                    >
                      {copied ? <CheckCircle size={15} /> : <Copy size={15} />}
                    </button>
                  </div>

                  <div className="text-[10px] text-gray-600 text-center leading-relaxed">
                    Send any amount to this address. Transactions are confirmed on the{' '}
                    <span className="text-gray-500">Bitcoin mainnet</span> and tracked live.
                  </div>
                </motion.div>
              )}

              {/* Lightning tab */}
              {tab === 'lightning' && (
                <motion.div
                  key="lightning"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col items-center gap-3 py-4"
                >
                  <div className="w-14 h-14 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.2)' }}>
                    <Zap size={28} className="text-yellow-400" />
                  </div>
                  <div className="text-center space-y-1">
                    <p className="text-sm font-semibold text-white">Lightning Coming Soon</p>
                    <p className="text-xs text-gray-500 leading-relaxed max-w-xs">
                      Our Lightning node is being set up on Delta. Use on-chain BTC for now — we'll add instant Lightning payments shortly.
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Footer */}
              <div className="flex items-center gap-2 pt-1">
                <div className="h-px flex-1" style={{ background: 'rgba(255,255,255,0.06)' }} />
                <span className="text-[10px] text-gray-600">Powered by Bitcoin Knots · Delta node</span>
                <div className="h-px flex-1" style={{ background: 'rgba(255,255,255,0.06)' }} />
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
