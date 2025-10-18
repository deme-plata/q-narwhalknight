import { useEffect, useRef, useState } from 'react';
import { Html5Qrcode } from 'html5-qrcode';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Camera, AlertCircle, CheckCircle } from 'lucide-react';

interface QRScannerProps {
  onScan: (data: string) => void;
  onClose: () => void;
  isOpen: boolean;
}

export default function QRScanner({ onScan, onClose, isOpen }: QRScannerProps) {
  const [scanner, setScanner] = useState<Html5Qrcode | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const scannerRef = useRef<HTMLDivElement>(null);
  const qrCodeRegionId = "qr-reader";

  useEffect(() => {
    if (!isOpen) {
      // Clean up scanner when modal closes
      if (scanner) {
        scanner.stop().then(() => {
          scanner.clear();
          setScanner(null);
          setIsScanning(false);
        }).catch(err => {
          console.error("Error stopping scanner:", err);
        });
      }
      return;
    }

    // Initialize scanner when modal opens
    const initScanner = async () => {
      try {
        setError(null);
        const html5QrCode = new Html5Qrcode(qrCodeRegionId);
        setScanner(html5QrCode);

        // Request camera permission and start scanning
        const config = {
          fps: 10, // Frames per second for scanning
          qrbox: { width: 250, height: 250 }, // QR box size
          aspectRatio: 1.0, // Square aspect ratio
        };

        await html5QrCode.start(
          { facingMode: "environment" }, // Use back camera on mobile
          config,
          (decodedText) => {
            // Success callback - QR code found
            console.log(`✅ QR Code detected: ${decodedText}`);
            setSuccess(true);
            onScan(decodedText);

            // Auto-close after successful scan
            setTimeout(() => {
              html5QrCode.stop().then(() => {
                html5QrCode.clear();
                onClose();
              });
            }, 500);
          },
          () => {
            // Error callback - called frequently while scanning
            // Don't show these errors to user as they're normal during scanning
          }
        );

        setIsScanning(true);
      } catch (err) {
        console.error("Camera access error:", err);
        let errorMsg = "Camera access denied";

        if (err instanceof Error) {
          if (err.message.includes("NotAllowedError")) {
            errorMsg = "Camera permission denied. Please allow camera access in your browser settings.";
          } else if (err.message.includes("NotFoundError")) {
            errorMsg = "No camera found on this device.";
          } else if (err.message.includes("NotReadableError")) {
            errorMsg = "Camera is already in use by another application.";
          } else {
            errorMsg = err.message;
          }
        }

        setError(errorMsg);
      }
    };

    // Small delay to ensure DOM is ready
    const timer = setTimeout(initScanner, 100);

    return () => {
      clearTimeout(timer);
      if (scanner) {
        scanner.stop().catch(() => {});
      }
    };
  }, [isOpen]);

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
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-amber-500/20">
                  <Camera className="w-6 h-6 text-amber-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Scan QR Code</h2>
                  <p className="text-sm text-gray-400">Point camera at QR code</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* Scanner Area */}
          <div className="p-6">
            <div
              ref={scannerRef}
              id={qrCodeRegionId}
              className="rounded-xl overflow-hidden"
              style={{
                width: '100%',
                minHeight: '300px'
              }}
            />

            {/* Error Message */}
            {error && (
              <motion.div
                className="mt-4 bg-red-500/20 border border-red-500/30 rounded-xl p-4 flex items-center gap-3"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-red-300 text-sm">{error}</p>
              </motion.div>
            )}

            {/* Success Message */}
            {success && (
              <motion.div
                className="mt-4 bg-green-500/20 border border-green-500/30 rounded-xl p-4 flex items-center gap-3"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                <p className="text-green-300 text-sm">QR Code scanned successfully!</p>
              </motion.div>
            )}

            {/* Instructions */}
            {isScanning && !error && !success && (
              <motion.div
                className="mt-4 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <p className="text-amber-200 text-sm text-center">
                  Position the QR code within the frame
                </p>
              </motion.div>
            )}
          </div>

          {/* Footer */}
          <div className="p-6 border-t border-amber-500/20">
            <p className="text-xs text-gray-500 text-center">
              Supports wallet addresses and payment requests
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
