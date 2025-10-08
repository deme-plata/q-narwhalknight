import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { QRCodeCanvas } from 'qrcode.react';
import { X, Copy, Check, Download, Share2, Sparkles, Coins } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';

interface QRCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress: string;
  balance?: number;
}

const particleColors = [
  'rgba(212, 175, 55, 0.8)',  // gold
  'rgba(255, 215, 0, 0.8)',   // yellow
  'rgba(255, 165, 0, 0.8)',   // orange
  'rgba(251, 191, 36, 0.8)',  // amber-400
  'rgba(245, 158, 11, 0.8)',  // amber-500
];

export default function QRCodeModal({ isOpen, onClose, walletAddress, balance = 0 }: QRCodeModalProps) {
  const [copied, setCopied] = useState(false);

  const copyAddress = () => {
    navigator.clipboard.writeText(walletAddress);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadQRCode = () => {
    const canvas = document.querySelector('#qr-code-canvas') as HTMLCanvasElement;
    if (canvas) {
      const url = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `${TICKER_SYMBOL}-wallet-${walletAddress.slice(0, 8)}.png`;
      link.href = url;
      link.click();
    }
  };

  const shareAddress = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Quillon Graph Wallet',
          text: `Receive ${TICKER_SYMBOL} tokens at: ${walletAddress}`,
        });
      } catch (err) {
        console.log('Share failed:', err);
      }
    } else {
      copyAddress();
    }
  };

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'auto';
    };
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/80 backdrop-blur-xl z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 50 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed inset-0 flex items-center justify-center z-50 p-4 pointer-events-none"
          >
            <div className="relative backdrop-blur-2xl rounded-3xl p-8 max-w-md w-full shadow-2xl pointer-events-auto overflow-hidden"
              style={{
                background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.98) 0%, rgba(50, 30, 80, 0.98) 100%)',
                border: '2px solid rgba(212, 175, 55, 0.3)',
                boxShadow: '0 0 40px rgba(212, 175, 55, 0.3), inset 0 0 20px rgba(212, 175, 55, 0.1)'
              }}
            >
              {/* Animated background effects */}
              <div className="absolute inset-0 opacity-30">
                {/* Floating particles */}
                {[...Array(20)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-1 h-1 rounded-full"
                    style={{
                      background: particleColors[i % particleColors.length],
                      left: `${Math.random() * 100}%`,
                      top: `${Math.random() * 100}%`,
                    }}
                    animate={{
                      y: [0, -30, 0],
                      x: [0, Math.random() * 20 - 10, 0],
                      opacity: [0.2, 1, 0.2],
                    }}
                    transition={{
                      duration: 3 + Math.random() * 2,
                      repeat: Infinity,
                      delay: Math.random() * 2,
                    }}
                  />
                ))}

                {/* Rotating gradient orbs */}
                <motion.div
                  className="absolute -top-20 -left-20 w-40 h-40 rounded-full blur-3xl"
                  style={{ background: 'rgba(212, 175, 55, 0.3)' }}
                  animate={{
                    rotate: 360,
                    scale: [1, 1.2, 1],
                  }}
                  transition={{
                    duration: 20,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                />
                <motion.div
                  className="absolute -bottom-20 -right-20 w-40 h-40 rounded-full blur-3xl"
                  style={{ background: 'rgba(255, 165, 0, 0.3)' }}
                  animate={{
                    rotate: -360,
                    scale: [1, 1.3, 1],
                  }}
                  transition={{
                    duration: 15,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                />
              </div>

              {/* Close button */}
              <motion.button
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
                onClick={onClose}
                className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
              >
                <X className="w-5 h-5 text-white" />
              </motion.button>

              {/* Header */}
              <div className="relative text-center mb-6">
                <motion.div
                  initial={{ y: -20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.1 }}
                  className="flex items-center justify-center gap-2 mb-2"
                >
                  <Sparkles className="w-6 h-6 text-amber-400 animate-pulse" />
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-amber-300 via-amber-400 to-yellow-500 bg-clip-text text-transparent">
                    Receive {TICKER_SYMBOL}
                  </h2>
                  <Coins className="w-6 h-6 text-amber-500 animate-pulse" />
                </motion.div>
                <motion.p
                  initial={{ y: -10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.15 }}
                  className="text-amber-200/80 text-sm"
                >
                  Scan code to receive Quillon Graph tokens
                </motion.p>
              </div>

              {/* QR Code Container */}
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="relative bg-white p-4 rounded-2xl mx-auto w-fit mb-6 shadow-2xl"
              >
                {/* Animated border gradient */}
                <div className="absolute inset-0 bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 rounded-2xl animate-gradient-xy opacity-70 blur-md" />

                <div className="relative bg-white p-2 rounded-xl">
                  <QRCodeCanvas
                    id="qr-code-canvas"
                    value={walletAddress}
                    size={200}
                    level="H"
                    includeMargin={true}
                    imageSettings={{
                      src: '',
                      x: undefined,
                      y: undefined,
                      height: 24,
                      width: 24,
                      excavate: true,
                    }}
                  />
                </div>
              </motion.div>

              {/* Wallet info */}
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="space-y-3 mb-6"
              >
                <div className="rounded-xl p-3 backdrop-blur-sm border border-amber-500/30"
                  style={{
                    background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%)'
                  }}
                >
                  <p className="text-xs text-amber-300/70 mb-1">Wallet Address</p>
                  <p className="font-mono text-xs text-amber-50 break-all">
                    {walletAddress.slice(0, 20)}...{walletAddress.slice(-20)}
                  </p>
                </div>

                {balance > 0 && (
                  <div className="rounded-xl p-3 backdrop-blur-sm border border-amber-400/30"
                    style={{
                      background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(255, 215, 0, 0.1) 100%)'
                    }}
                  >
                    <p className="text-xs text-amber-300/70 mb-1">Current Balance</p>
                    <p className="text-lg font-bold bg-gradient-to-r from-amber-300 via-yellow-400 to-amber-500 bg-clip-text text-transparent">
                      {balance.toLocaleString()} {TICKER_SYMBOL}
                    </p>
                  </div>
                )}
              </motion.div>

              {/* Action buttons */}
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="flex gap-3"
              >
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={copyAddress}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium text-white transition-all"
                  style={{
                    background: copied
                      ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
                      : 'linear-gradient(135deg, rgba(212, 175, 55, 0.8) 0%, rgba(180, 148, 46, 0.8) 100%)',
                    boxShadow: '0 4px 15px rgba(212, 175, 55, 0.3)'
                  }}
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Copy
                    </>
                  )}
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={downloadQRCode}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium text-white transition-all"
                  style={{
                    background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.8) 0%, rgba(255, 193, 7, 0.8) 100%)',
                    boxShadow: '0 4px 15px rgba(255, 215, 0, 0.3)'
                  }}
                >
                  <Download className="w-4 h-4" />
                  Save
                </motion.button>

                {typeof navigator !== 'undefined' && 'share' in navigator && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={shareAddress}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium text-white transition-all"
                    style={{
                      background: 'linear-gradient(135deg, rgba(255, 165, 0, 0.8) 0%, rgba(255, 140, 0, 0.8) 100%)',
                      boxShadow: '0 4px 15px rgba(255, 165, 0, 0.3)'
                    }}
                  >
                    <Share2 className="w-4 h-4" />
                    Share
                  </motion.button>
                )}
              </motion.div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}