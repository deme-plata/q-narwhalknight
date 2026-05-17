import { useEffect, useRef, useState, useCallback } from 'react';
import { Html5Qrcode } from 'html5-qrcode';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Camera, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';

// BarcodeDetector is native in Chromium (Chrome, Brave, Edge, Opera).
// It reads directly from video frames — no canvas, immune to Brave's
// fingerprint protection which poisons canvas.getImageData().
declare class BarcodeDetector {
  constructor(options?: { formats: string[] });
  detect(src: HTMLVideoElement | HTMLCanvasElement | ImageBitmap): Promise<Array<{ rawValue: string }>>;
}
const HAS_BARCODE_DETECTOR = typeof window !== 'undefined' && 'BarcodeDetector' in window;

const QR_REGION_ID = 'qr-reader-region';

interface QRScannerProps {
  onScan: (data: string) => void;
  onClose: () => void;
  isOpen: boolean;
}

export default function QRScanner({ onScan, onClose, isOpen }: QRScannerProps) {
  const [status, setStatus] = useState<'idle' | 'loading' | 'scanning' | 'success' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const scannerRef = useRef<Html5Qrcode | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // ── Stop everything and release camera ───────────────────────────────────
  const stopAll = useCallback(async () => {
    // Cancel BarcodeDetector RAF loop
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    // Release camera stream (turns off the camera indicator light)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;

    // Stop html5-qrcode fallback scanner
    const s = scannerRef.current;
    if (s) {
      scannerRef.current = null;
      try { await s.stop(); } catch { /* already stopped */ }
      try { s.clear(); } catch { /* element gone */ }
    }
  }, []);

  // ── BarcodeDetector path (Brave / Chrome / Edge) ─────────────────────────
  const startNativeScanner = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
    streamRef.current = stream;

    const video = videoRef.current!;
    video.srcObject = stream;
    await video.play();

    const detector = new BarcodeDetector({ formats: ['qr_code'] });

    const tick = async () => {
      if (!mountedRef.current) return;
      if (video.readyState >= 2) {
        try {
          const results = await detector.detect(video);
          if (results.length > 0 && results[0].rawValue) {
            if (!mountedRef.current) return;
            setStatus('success');
            onScan(results[0].rawValue);
            setTimeout(async () => {
              await stopAll();
              if (mountedRef.current) onClose();
            }, 600);
            return; // Don't schedule next frame
          }
        } catch { /* per-frame errors are normal */ }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [onScan, onClose, stopAll]);

  // ── html5-qrcode fallback (Firefox / Safari) ─────────────────────────────
  const startHtml5Scanner = useCallback(async () => {
    await new Promise(r => setTimeout(r, 250)); // wait for DOM
    const el = document.getElementById(QR_REGION_ID);
    if (!el || !mountedRef.current) return;

    const qr = new Html5Qrcode(QR_REGION_ID);
    scannerRef.current = qr;

    const tryStart = async (facing: ConstrainDOMString) => {
      await qr.start(
        { facingMode: facing },
        { fps: 10, qrbox: { width: 250, height: 250 } },
        (decoded) => {
          if (!mountedRef.current) return;
          setStatus('success');
          onScan(decoded);
          setTimeout(async () => {
            await stopAll();
            if (mountedRef.current) onClose();
          }, 600);
        },
        () => { /* frame errors are normal */ }
      );
    };

    try {
      await tryStart('environment');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // If back camera is overconstrained, try any camera
      if (msg.includes('Overconstrained') || msg.includes('overconstrained')) {
        await tryStart('user');
      } else {
        throw err;
      }
    }
  }, [onScan, onClose, stopAll]);

  // ── Main effect ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isOpen) {
      stopAll();
      setStatus('idle');
      setErrorMsg('');
      return;
    }

    setStatus('loading');
    setErrorMsg('');
    let cancelled = false;

    const init = async () => {
      try {
        if (HAS_BARCODE_DETECTOR) {
          await startNativeScanner();
        } else {
          await startHtml5Scanner();
        }
        if (!cancelled && mountedRef.current) setStatus('scanning');
      } catch (err) {
        if (!cancelled && mountedRef.current) {
          const msg = err instanceof Error ? err.message : String(err);
          let friendly = 'Camera access failed. Please try again.';
          if (msg.includes('NotAllowedError') || msg.includes('Permission denied')) {
            friendly = HAS_BARCODE_DETECTOR
              ? 'Camera access was blocked. In Brave, click the camera icon in the address bar and allow access.'
              : 'Camera permission denied. Allow camera access in your browser settings.';
          } else if (msg.includes('NotFoundError')) {
            friendly = 'No camera found on this device.';
          } else if (msg.includes('NotReadableError') || msg.includes('Could not start video source')) {
            friendly = 'Camera is in use by another app. Close it and try again.';
          }
          setErrorMsg(friendly);
          setStatus('error');
        }
        await stopAll();
      }
    };

    init();

    return () => {
      cancelled = true;
      stopAll();
    };
  }, [isOpen, startNativeScanner, startHtml5Scanner, stopAll]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ backgroundColor: 'rgba(0, 0, 0, 0.85)' }}
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
            boxShadow: '0 0 40px rgba(212, 175, 55, 0.3)',
          }}
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 20 }}
          onClick={e => e.stopPropagation()}
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
                  <p className="text-sm text-gray-400">Point camera at wallet QR code</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-white transition-colors rounded-lg"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* Scanner body */}
          <div className="p-6 space-y-4">
            {/* Loading */}
            {status === 'loading' && (
              <div className="flex flex-col items-center justify-center py-12 gap-3">
                <Loader2 className="w-8 h-8 text-amber-400 animate-spin" />
                <p className="text-gray-400 text-sm">Starting camera…</p>
                {HAS_BARCODE_DETECTOR && (
                  <p className="text-xs text-gray-600 text-center max-w-xs leading-relaxed">
                    If using Brave, look for a camera icon in the address bar to allow access.
                  </p>
                )}
              </div>
            )}

            {/* BarcodeDetector path: native <video> element */}
            {HAS_BARCODE_DETECTOR && (
              <div
                className="relative rounded-xl overflow-hidden"
                style={{ display: status === 'loading' || status === 'error' ? 'none' : 'block' }}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full block"
                  style={{ borderRadius: 12 }}
                />
                {/* Targeting overlay */}
                {status === 'scanning' && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div
                      className="relative"
                      style={{ width: 220, height: 220 }}
                    >
                      {/* Corner brackets */}
                      {[
                        'top-0 left-0 border-t-2 border-l-2 rounded-tl',
                        'top-0 right-0 border-t-2 border-r-2 rounded-tr',
                        'bottom-0 left-0 border-b-2 border-l-2 rounded-bl',
                        'bottom-0 right-0 border-b-2 border-r-2 rounded-br',
                      ].map((cls, i) => (
                        <div
                          key={i}
                          className={`absolute w-8 h-8 border-amber-400 ${cls}`}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* html5-qrcode fallback path: attaches to this div */}
            {!HAS_BARCODE_DETECTOR && (
              <div
                id={QR_REGION_ID}
                className="rounded-xl overflow-hidden"
                style={{
                  width: '100%',
                  // Keep in DOM so html5-qrcode can attach; hide while loading
                  visibility: status === 'loading' ? 'hidden' : 'visible',
                  minHeight: status === 'loading' ? 0 : 300,
                }}
              />
            )}

            {/* Error */}
            {status === 'error' && (
              <motion.div
                className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 flex items-start gap-3"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <p className="text-red-300 text-sm leading-relaxed">{errorMsg}</p>
              </motion.div>
            )}

            {/* Success */}
            {status === 'success' && (
              <motion.div
                className="bg-green-500/20 border border-green-500/30 rounded-xl p-4 flex items-center gap-3"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <CheckCircle className="w-5 h-5 text-green-400 shrink-0" />
                <p className="text-green-300 text-sm">QR code scanned successfully!</p>
              </motion.div>
            )}

            {/* Scanning hint */}
            {status === 'scanning' && (
              <motion.div
                className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4"
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
