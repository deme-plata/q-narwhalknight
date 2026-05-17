import { motion, AnimatePresence } from 'framer-motion';
import { QRCodeSVG } from 'qrcode.react';
import { X, Copy, Download, CheckCircle } from 'lucide-react';
import { useState } from 'react';

interface QRDisplayProps {
  data: string;
  title: string;
  subtitle?: string;
  onClose: () => void;
  isOpen: boolean;
}

export default function QRDisplay({ data, title, subtitle, onClose, isOpen }: QRDisplayProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(data);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleDownload = () => {
    const svg = document.getElementById('qr-code-svg');
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      const pngFile = canvas.toDataURL('image/png');

      const downloadLink = document.createElement('a');
      downloadLink.download = 'quillon-wallet-qr.png';
      downloadLink.href = pngFile;
      downloadLink.click();
    };

    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ backgroundColor: 'rgba(0, 0, 0, 0.8)' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="relative w-full max-w-md rounded-3xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.95) 0%, rgba(50, 30, 80, 0.95) 100%)',
            border: '2px solid rgba(212, 175, 55, 0.3)',
            boxShadow: '0 0 40px rgba(212, 175, 55, 0.3)'
          }}
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 20 }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-amber-500/20">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white">{title}</h2>
                {subtitle && <p className="text-sm text-gray-400 mt-1">{subtitle}</p>}
              </div>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* QR Code */}
          <div className="p-8 flex justify-center">
            <div className="p-4 bg-white rounded-2xl shadow-2xl">
              <QRCodeSVG
                id="qr-code-svg"
                value={data}
                size={256}
                level="H" // High error correction for better scanning
                includeMargin={true}
                imageSettings={{
                  src: '/quillon-logo.png',
                  x: undefined,
                  y: undefined,
                  height: 40,
                  width: 40,
                  excavate: true,
                }}
              />
            </div>
          </div>

          {/* Address Display */}
          <div className="px-6 pb-6">
            <div className="bg-slate-900/70 rounded-xl p-4">
              <p className="text-xs text-gray-400 mb-2">Wallet Address</p>
              <p className="text-sm text-amber-300 font-mono break-all">
                {data}
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="p-6 border-t border-amber-500/20 flex gap-3">
            <button
              onClick={handleCopy}
              className="flex-1 py-3 px-4 bg-amber-600/20 border border-amber-500/40 rounded-xl text-amber-300 font-medium flex items-center justify-center gap-2 hover:bg-amber-600/30 transition-colors"
            >
              {copied ? (
                <>
                  <CheckCircle className="w-5 h-5" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-5 h-5" />
                  <span>Copy Address</span>
                </>
              )}
            </button>
            <button
              onClick={handleDownload}
              className="flex-1 py-3 px-4 bg-blue-600/20 border border-blue-500/40 rounded-xl text-blue-300 font-medium flex items-center justify-center gap-2 hover:bg-blue-600/30 transition-colors"
            >
              <Download className="w-5 h-5" />
              <span>Save QR</span>
            </button>
          </div>

          {/* Footer Info */}
          <div className="p-6 pt-0">
            <p className="text-xs text-gray-500 text-center">
              🔐 Share this QR code to receive QUG tokens
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
