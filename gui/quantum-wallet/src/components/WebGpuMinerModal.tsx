// WebGPU Miner quick-launch modal (PR #94 companion).
// Renders the existing /webgpu-miner/ as an iframe with the user's wallet
// auto-injected via URL param. Phase 2 will fold it inline as React component;
// for now iframe is the lowest-risk path.

import { motion, AnimatePresence } from 'framer-motion';
import { X, Cpu, Pickaxe, ExternalLink } from 'lucide-react';

interface WebGpuMinerModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress?: string;
}

export default function WebGpuMinerModal({ isOpen, onClose, walletAddress }: WebGpuMinerModalProps) {
  const minerUrl = walletAddress
    ? `/webgpu-miner/?wallet=${encodeURIComponent(walletAddress)}`
    : '/webgpu-miner/';

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="bg-slate-900 border border-purple-500/30 rounded-2xl shadow-2xl w-[640px] max-w-[95vw] max-h-[90vh] overflow-hidden flex flex-col"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-purple-500/20 bg-purple-500/5">
              <div className="flex items-center gap-2">
                <Pickaxe className="w-5 h-5 text-purple-400" />
                <h2 className="text-lg font-bold text-purple-200">Browser Mining (WebGPU)</h2>
                <span className="text-xs px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded">Preview</span>
              </div>
              <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              <div className="space-y-4 text-sm text-slate-300">
                <div className="flex items-start gap-3 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <Cpu className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-semibold text-purple-200 mb-1">Mine QUG directly in this browser tab.</div>
                    <p className="text-xs text-slate-400">
                      WebGPU compute shader (Keccak-f[1600] / SHA3-256), with automatic CPU Web Worker fallback for browsers without WebGPU.
                      Typical hashrate: 2-5 MH/s on integrated GPU, 50-150 MH/s on discrete GPU.
                    </p>
                  </div>
                </div>

                <div className="text-xs space-y-2 text-slate-400">
                  <p>
                    <span className="font-semibold text-slate-200">Note:</span> This is a proof-of-concept. Production
                    Quillon Graph mining uses a Genus-2 hyperelliptic-curve VDF, not SHA3 PoW. The browser miner uses SHA3
                    with a configurable target as a stand-in. See <code className="bg-slate-800 px-1 rounded">gui/webgpu-miner/README.md</code> for details.
                  </p>
                  <p>
                    <span className="font-semibold text-slate-200">Status:</span> PR #94 ships the PoC. Genus-2 VDF port to
                    WebGPU is a ~6-week follow-up effort.
                  </p>
                </div>

                {walletAddress && (
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-xs text-slate-500 mb-1">Mining to wallet:</div>
                    <div className="font-mono text-xs text-purple-300 truncate">{walletAddress}</div>
                  </div>
                )}

                <a
                  href={minerUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-center gap-2 w-full py-3 bg-purple-500 hover:bg-purple-600 transition-colors rounded-lg font-medium text-white"
                >
                  <Pickaxe className="w-4 h-4" />
                  Launch Browser Miner
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
