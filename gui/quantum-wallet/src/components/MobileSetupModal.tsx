import { useState, useEffect, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { QRCodeCanvas } from 'qrcode.react';
import { X, Smartphone, Wifi, Shield, Check, Copy, ChevronRight, RefreshCw, Zap } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';

interface MobileSetupModalProps {
  onClose: () => void;
}

const STORAGE_KEY = 'mobile_setup_seen_v1';

// Deep-link payload refreshes every 60s to keep the session token valid
function buildDeepLink(walletAddress: string): string {
  const payload = {
    action: 'pair',
    wallet: walletAddress,
    node: window.location.origin,
    ts: Math.floor(Date.now() / 1000),
    network: 'mainnet2026.1',
  };
  return `quillon://pair?data=${encodeURIComponent(JSON.stringify(payload))}`;
}

// Animated ring that orbits the QR code
function OrbitRing({ delay, size, color }: { delay: number; size: number; color: string }) {
  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        width: size,
        height: size,
        border: `1.5px solid ${color}`,
        top: '50%',
        left: '50%',
        marginTop: -size / 2,
        marginLeft: -size / 2,
      }}
      initial={{ opacity: 0, scale: 0.6 }}
      animate={{
        opacity: [0, 0.6, 0],
        scale: [0.6, 1, 1.4],
        rotate: [0, 180],
      }}
      transition={{
        duration: 3,
        repeat: Infinity,
        delay,
        ease: 'easeOut',
      }}
    />
  );
}

// Steps data for the setup flow
const STEPS = [
  {
    id: 'scan',
    title: 'Scan QR Code',
    subtitle: 'Open Quillon Mobile and tap "Scan to Connect"',
    icon: Smartphone,
  },
  {
    id: 'confirm',
    title: 'Confirm Pairing',
    subtitle: 'Approve the connection on your phone',
    icon: Shield,
  },
  {
    id: 'done',
    title: 'All Set!',
    subtitle: 'Your mobile wallet is synced and ready',
    icon: Zap,
  },
] as const;

export default function MobileSetupModal({ onClose }: MobileSetupModalProps) {
  const [step, setStep] = useState(0);
  const [copied, setCopied] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const walletAddress = useMemo(
    () => localStorage.getItem('walletAddress') || '',
    [],
  );

  const deepLink = useMemo(
    () => buildDeepLink(walletAddress),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [walletAddress, refreshKey],
  );

  // Auto-refresh QR every 60s
  useEffect(() => {
    const iv = setInterval(() => setRefreshKey((k) => k + 1), 60_000);
    return () => clearInterval(iv);
  }, []);

  // Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', handler);
      document.body.style.overflow = 'auto';
    };
  }, [onClose]);

  const handleClose = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, 'true');
    onClose();
  }, [onClose]);

  const copyLink = useCallback(() => {
    navigator.clipboard.writeText(deepLink);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [deepLink]);

  const modal = (
    <AnimatePresence>
      {/* Backdrop */}
      <motion.div
        key="mobilesetup-backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.3 }}
        onClick={handleClose}
        className="fixed inset-0 z-[9999]"
        style={{
          background: 'radial-gradient(ellipse at center, rgba(0,229,255,0.06) 0%, rgba(0,0,0,0.92) 70%)',
          backdropFilter: 'blur(20px)',
        }}
      />

      {/* Modal card */}
      <motion.div
        key="mobilesetup-card"
        initial={{ opacity: 0, y: 60, scale: 0.92 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 40, scale: 0.95 }}
        transition={{ type: 'spring', damping: 28, stiffness: 340 }}
        className="fixed inset-0 z-[9999] flex items-center justify-center p-4 pointer-events-none"
      >
        <div
          className="relative w-full max-w-[420px] rounded-[28px] overflow-hidden pointer-events-auto"
          style={{
            background: 'linear-gradient(165deg, #0f0f1e 0%, #141428 40%, #0d1117 100%)',
            boxShadow:
              '0 0 0 1px rgba(0,229,255,0.12), 0 0 80px rgba(0,229,255,0.08), 0 32px 64px rgba(0,0,0,0.6)',
          }}
        >
          {/* Animated top accent line */}
          <motion.div
            className="absolute top-0 left-0 right-0 h-[2px]"
            style={{
              background: 'linear-gradient(90deg, transparent 0%, #00E5FF 30%, #7C4DFF 70%, transparent 100%)',
            }}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />

          {/* Close button */}
          <motion.button
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            onClick={handleClose}
            className="absolute top-4 right-4 z-10 p-2 rounded-full"
            style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            <X className="w-4 h-4 text-white/60" />
          </motion.button>

          <div className="px-7 pt-7 pb-6">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-center mb-6"
            >
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-3"
                style={{ background: 'rgba(0,229,255,0.08)', border: '1px solid rgba(0,229,255,0.15)' }}>
                <Wifi className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-[11px] font-semibold tracking-wider text-cyan-300 uppercase">
                  Mobile Sync
                </span>
              </div>
              <h2
                className="text-[22px] font-bold tracking-tight"
                style={{
                  background: 'linear-gradient(135deg, #fff 0%, #b0c4ff 50%, #00E5FF 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Connect Quillon Mobile
              </h2>
              <p className="text-[13px] text-white/40 mt-1.5 leading-relaxed">
                Pair your Android device in seconds
              </p>
            </motion.div>

            {/* Step indicator */}
            <div className="flex items-center justify-center gap-2 mb-5">
              {STEPS.map((s, i) => {
                const Icon = s.icon;
                const isActive = i === step;
                const isDone = i < step;
                return (
                  <motion.button
                    key={s.id}
                    onClick={() => setStep(i)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-colors"
                    style={{
                      background: isActive
                        ? 'rgba(0,229,255,0.12)'
                        : isDone
                        ? 'rgba(16,185,129,0.1)'
                        : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${
                        isActive ? 'rgba(0,229,255,0.3)' : isDone ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.06)'
                      }`,
                    }}
                    animate={isActive ? { scale: [1, 1.04, 1] } : {}}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    {isDone ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Icon className="w-3.5 h-3.5" style={{ color: isActive ? '#00E5FF' : 'rgba(255,255,255,0.3)' }} />
                    )}
                    <span
                      className="text-[11px] font-medium"
                      style={{ color: isActive ? '#00E5FF' : isDone ? '#10B981' : 'rgba(255,255,255,0.3)' }}
                    >
                      {s.title}
                    </span>
                  </motion.button>
                );
              })}
            </div>

            {/* QR Code area — always visible, fades based on step */}
            <AnimatePresence mode="wait">
              {step === 0 && (
                <motion.div
                  key="qr"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.25 }}
                  className="flex flex-col items-center"
                >
                  {/* QR with orbit rings */}
                  <div className="relative mb-4">
                    <OrbitRing delay={0} size={260} color="rgba(0,229,255,0.15)" />
                    <OrbitRing delay={1} size={280} color="rgba(124,77,255,0.12)" />
                    <OrbitRing delay={2} size={300} color="rgba(0,229,255,0.08)" />

                    <motion.div
                      className="relative rounded-2xl p-1"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0,229,255,0.2), rgba(124,77,255,0.2))',
                      }}
                      animate={{
                        boxShadow: [
                          '0 0 20px rgba(0,229,255,0.1)',
                          '0 0 40px rgba(0,229,255,0.2)',
                          '0 0 20px rgba(0,229,255,0.1)',
                        ],
                      }}
                      transition={{ duration: 3, repeat: Infinity }}
                    >
                      <div className="bg-white rounded-xl p-3">
                        <QRCodeCanvas
                          value={deepLink}
                          size={180}
                          level="H"
                          includeMargin={false}
                          bgColor="#ffffff"
                          fgColor="#0f0f1e"
                        />
                      </div>
                    </motion.div>

                    {/* Refresh indicator */}
                    <motion.button
                      onClick={() => setRefreshKey((k) => k + 1)}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9, rotate: 180 }}
                      className="absolute -bottom-2 -right-2 p-2 rounded-full"
                      style={{
                        background: 'linear-gradient(135deg, #141428, #1a1a2e)',
                        border: '1px solid rgba(0,229,255,0.2)',
                      }}
                      title="Refresh QR code"
                    >
                      <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
                    </motion.button>
                  </div>

                  {/* Wallet address pill */}
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="flex items-center gap-2 px-3 py-2 rounded-xl w-full"
                    style={{
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-[10px] text-white/30 mb-0.5 uppercase tracking-wider font-medium">
                        Wallet
                      </p>
                      <p className="text-[12px] font-mono text-white/60 truncate">
                        {walletAddress || 'No wallet found'}
                      </p>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={copyLink}
                      className="p-1.5 rounded-lg shrink-0"
                      style={{ background: 'rgba(0,229,255,0.08)' }}
                    >
                      {copied ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5 text-cyan-400" />
                      )}
                    </motion.button>
                  </motion.div>

                  {/* Hint text */}
                  <p className="text-[11px] text-white/25 mt-3 text-center">
                    QR refreshes automatically &middot; Scan within 60s
                  </p>
                </motion.div>
              )}

              {step === 1 && (
                <motion.div
                  key="confirm"
                  initial={{ opacity: 0, x: 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -30 }}
                  className="flex flex-col items-center py-8"
                >
                  <motion.div
                    className="w-20 h-20 rounded-full flex items-center justify-center mb-5"
                    style={{
                      background: 'radial-gradient(circle, rgba(0,229,255,0.15) 0%, rgba(0,229,255,0.03) 70%)',
                      border: '1.5px solid rgba(0,229,255,0.2)',
                    }}
                    animate={{ scale: [1, 1.06, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Shield className="w-9 h-9 text-cyan-400" />
                  </motion.div>
                  <h3 className="text-lg font-semibold text-white mb-2">Confirm on Mobile</h3>
                  <p className="text-[13px] text-white/40 text-center max-w-[280px] leading-relaxed">
                    A pairing request will appear on your phone. Tap{' '}
                    <span className="text-cyan-400 font-medium">Approve</span> to link your wallet securely.
                  </p>

                  {/* Animated connection dots */}
                  <div className="flex items-center gap-2 mt-6">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 rounded-full bg-cyan-400"
                        animate={{ opacity: [0.2, 1, 0.2], scale: [0.8, 1.2, 0.8] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.3 }}
                      />
                    ))}
                  </div>
                </motion.div>
              )}

              {step === 2 && (
                <motion.div
                  key="done"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center py-8"
                >
                  <motion.div
                    className="w-20 h-20 rounded-full flex items-center justify-center mb-5"
                    style={{
                      background: 'radial-gradient(circle, rgba(16,185,129,0.2) 0%, rgba(16,185,129,0.03) 70%)',
                      border: '1.5px solid rgba(16,185,129,0.3)',
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', damping: 12, stiffness: 200 }}
                  >
                    <Check className="w-10 h-10 text-emerald-400" />
                  </motion.div>
                  <h3 className="text-lg font-semibold text-white mb-2">You're Connected!</h3>
                  <p className="text-[13px] text-white/40 text-center max-w-[280px] leading-relaxed">
                    Your {TICKER_SYMBOL} wallet is now synced. Transactions, mining rewards, and balances
                    update in real-time on both devices.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Bottom actions */}
            <div className="flex items-center gap-3 mt-5">
              {step > 0 && (
                <motion.button
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => setStep((s) => s - 1)}
                  className="px-4 py-2.5 rounded-xl text-[13px] font-medium text-white/50 transition-colors"
                  style={{
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  Back
                </motion.button>
              )}

              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => {
                  if (step < STEPS.length - 1) {
                    setStep((s) => s + 1);
                  } else {
                    handleClose();
                  }
                }}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-[13px] font-semibold text-white"
                style={{
                  background:
                    step === STEPS.length - 1
                      ? 'linear-gradient(135deg, #10B981 0%, #059669 100%)'
                      : 'linear-gradient(135deg, #00B8D4 0%, #7C4DFF 100%)',
                  boxShadow:
                    step === STEPS.length - 1
                      ? '0 4px 20px rgba(16,185,129,0.3)'
                      : '0 4px 20px rgba(0,229,255,0.2)',
                }}
              >
                {step === STEPS.length - 1 ? (
                  'Done'
                ) : (
                  <>
                    {step === 0 ? "I've Scanned" : 'Next'}
                    <ChevronRight className="w-4 h-4" />
                  </>
                )}
              </motion.button>
            </div>

            {/* Skip link */}
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              onClick={handleClose}
              className="w-full text-center mt-3 text-[11px] text-white/20 hover:text-white/40 transition-colors"
            >
              Skip for now
            </motion.button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );

  return createPortal(modal, document.body);
}

export { STORAGE_KEY as MOBILE_SETUP_STORAGE_KEY };
