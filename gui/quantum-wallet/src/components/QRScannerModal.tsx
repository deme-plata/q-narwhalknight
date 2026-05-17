import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Camera, CameraOff, Zap, AlertCircle, CheckCircle2 } from 'lucide-react';
import { BrowserMultiFormatReader, NotFoundException } from '@zxing/library';

interface QRScannerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onScan: (result: string) => void;
}

export default function QRScannerModal({ isOpen, onClose, onScan }: QRScannerModalProps) {
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [scannedValue, setScannedValue] = useState<string>('');
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const readerRef = useRef<BrowserMultiFormatReader | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Initialize scanner
  useEffect(() => {
    if (isOpen) {
      readerRef.current = new BrowserMultiFormatReader();
      startScanning();
    }

    return () => {
      stopScanning();
    };
  }, [isOpen]);

  const requestCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera for better QR scanning
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      setHasPermission(true);
      streamRef.current = stream;
      return stream;
    } catch (err) {
      console.error('Camera permission denied:', err);
      setHasPermission(false);
      setError('Camera access is required to scan QR codes. Please enable camera permissions and try again.');
      return null;
    }
  };

  const startScanning = async () => {
    if (!readerRef.current || !videoRef.current) return;

    setError(null);
    setScanning(true);
    setSuccess(false);

    try {
      const stream = await requestCameraPermission();
      if (!stream) return;

      // Set video stream
      videoRef.current.srcObject = stream;

      // Start scanning
      readerRef.current.decodeFromVideoDevice(null, videoRef.current, (result: any, error: any) => {
        if (result) {
          const text = result.getText();
          console.log('QR Code scanned:', text);

          setScannedValue(text);
          setSuccess(true);
          setScanning(false);

          // Auto-apply after a short delay
          setTimeout(() => {
            onScan(text);
            handleClose();
          }, 1500);
        }

        if (error && !(error instanceof NotFoundException)) {
          console.error('Scanning error:', error);
        }
      });

    } catch (err) {
      console.error('Failed to start scanning:', err);
      setError('Failed to access camera. Please check your device permissions.');
      setScanning(false);
    }
  };

  const stopScanning = () => {
    if (readerRef.current) {
      readerRef.current.reset();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    setScanning(false);
    setSuccess(false);
    setScannedValue('');
  };

  const handleClose = () => {
    stopScanning();
    setError(null);
    setHasPermission(null);
    onClose();
  };

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose();
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'auto';
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            className="fixed inset-0 bg-black/90 backdrop-blur-xl z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 50 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed inset-0 flex items-center justify-center z-50 p-4 pointer-events-none"
          >
            <div className="relative bg-gradient-to-br from-quantum-indigo/95 via-quantum-purple/90 to-quantum-dark/95 backdrop-blur-2xl rounded-3xl p-6 max-w-md w-full shadow-2xl pointer-events-auto border border-quantum-cyan/20 overflow-hidden">

              {/* Animated background effects */}
              <div className="absolute inset-0 opacity-30">
                <motion.div
                  className="absolute -top-20 -left-20 w-40 h-40 bg-quantum-purple/30 rounded-full blur-3xl"
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
                  className="absolute -bottom-20 -right-20 w-40 h-40 bg-quantum-cyan/30 rounded-full blur-3xl"
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
                onClick={handleClose}
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
                  <Camera className="w-6 h-6 text-quantum-cyan animate-pulse" />
                  <h2 className="text-2xl font-bold text-white">Scan QR Code</h2>
                  <Zap className="w-6 h-6 text-quantum-yellow animate-pulse" />
                </motion.div>
                <motion.p
                  initial={{ y: -10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.15 }}
                  className="text-gray-300 text-sm"
                >
                  Point your camera at a QR code
                </motion.p>
              </div>

              {/* Camera View */}
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="relative aspect-square bg-black rounded-2xl overflow-hidden mb-4 shadow-2xl"
              >
                {/* Scanning overlay */}
                <div className="absolute inset-0 z-10">
                  {/* Corner guides */}
                  <div className="absolute top-4 left-4 w-6 h-6 border-l-4 border-t-4 border-quantum-cyan"></div>
                  <div className="absolute top-4 right-4 w-6 h-6 border-r-4 border-t-4 border-quantum-cyan"></div>
                  <div className="absolute bottom-4 left-4 w-6 h-6 border-l-4 border-b-4 border-quantum-cyan"></div>
                  <div className="absolute bottom-4 right-4 w-6 h-6 border-r-4 border-b-4 border-quantum-cyan"></div>

                  {/* Scanning line */}
                  {scanning && (
                    <motion.div
                      className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-quantum-cyan to-transparent"
                      animate={{
                        y: [0, 250, 0]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                    />
                  )}
                </div>

                {/* Video element */}
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  playsInline
                  muted
                />

                {/* Success overlay */}
                {success && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="absolute inset-0 bg-quantum-green/20 flex items-center justify-center backdrop-blur-sm"
                  >
                    <div className="text-center">
                      <CheckCircle2 className="w-16 h-16 text-quantum-green mx-auto mb-2" />
                      <p className="text-white font-semibold">QR Code Detected!</p>
                      <p className="text-sm text-gray-300 mt-1 font-mono break-all px-4">
                        {scannedValue.length > 40 ? `${scannedValue.slice(0, 40)}...` : scannedValue}
                      </p>
                    </div>
                  </motion.div>
                )}
              </motion.div>

              {/* Status and controls */}
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="space-y-4"
              >
                {/* Error message */}
                {error && (
                  <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-3 flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                    <p className="text-red-200 text-sm">{error}</p>
                  </div>
                )}

                {/* Permission prompt */}
                {hasPermission === false && (
                  <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-xl p-3">
                    <p className="text-yellow-200 text-sm mb-3">
                      Camera access is needed to scan QR codes. Please allow camera permissions in your browser.
                    </p>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={startScanning}
                      className="w-full bg-gradient-to-r from-quantum-cyan to-quantum-blue rounded-xl py-2 px-4 text-white font-medium"
                    >
                      Enable Camera
                    </motion.button>
                  </div>
                )}

                {/* Status */}
                {hasPermission && (
                  <div className="bg-quantum-dark/50 rounded-xl p-3 text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      {scanning ? (
                        <>
                          <Camera className="w-5 h-5 text-quantum-cyan animate-pulse" />
                          <span className="text-quantum-cyan font-medium">Scanning...</span>
                        </>
                      ) : success ? (
                        <>
                          <CheckCircle2 className="w-5 h-5 text-quantum-green" />
                          <span className="text-quantum-green font-medium">Success!</span>
                        </>
                      ) : (
                        <>
                          <CameraOff className="w-5 h-5 text-gray-400" />
                          <span className="text-gray-400 font-medium">Camera Ready</span>
                        </>
                      )}
                    </div>
                    <p className="text-gray-300 text-xs">
                      Position QR code within the scanning area
                    </p>
                  </div>
                )}
              </motion.div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}