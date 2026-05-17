import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Check, Palette } from 'lucide-react';
import { THEME_LIST, changeBorderTheme, FRAME_VERSION } from './AnimatedBorder';
import type { BorderTheme } from './AnimatedBorder';

interface ThemeChooserModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ThemeChooserModal({ isOpen, onClose }: ThemeChooserModalProps) {
  const [selectedTheme, setSelectedTheme] = useState<BorderTheme>(() => {
    const stored = localStorage.getItem('borderTheme');
    if (stored && THEME_LIST.some(t => t.id === stored)) return stored as BorderTheme;
    return 'frameless';
  });

  const [previewTheme, setPreviewTheme] = useState<BorderTheme | null>(null);

  // Filter out 'red' from chooser since it's only used for transaction flash
  const choosableThemes = THEME_LIST.filter(t => t.id !== 'red');

  const handleSelect = (themeId: BorderTheme) => {
    setSelectedTheme(themeId);
    changeBorderTheme(themeId);
  };

  // Reset preview on close
  useEffect(() => {
    if (!isOpen) setPreviewTheme(null);
  }, [isOpen]);

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[10002]"
            onClick={onClose}
          />

          {/* Modal - Big centered white/light box like the frame reference images */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed inset-0 z-[10003] flex items-center justify-center p-4"
            onClick={onClose}
          >
            <div
              className="w-full max-w-[900px] max-h-[85vh] overflow-hidden rounded-3xl shadow-2xl"
              onClick={(e) => e.stopPropagation()}
              style={{
                background: 'linear-gradient(135deg, #f8f9fc 0%, #ffffff 30%, #f0f2f8 100%)',
                boxShadow: '0 0 80px rgba(147, 51, 234, 0.15), 0 25px 60px rgba(0,0,0,0.3)',
              }}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-8 py-5 border-b border-slate-200">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 shadow-lg">
                    <Palette className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-slate-800">Choose Your Theme</h2>
                    <p className="text-slate-500 text-sm mt-0.5">Select a border frame for your wallet</p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-xl hover:bg-slate-100 transition-colors"
                >
                  <X className="w-6 h-6 text-slate-400" />
                </button>
              </div>

              {/* Theme Grid */}
              <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(85vh - 90px)' }}>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                  {choosableThemes.map((theme) => {
                    const isActive = selectedTheme === theme.id;
                    return (
                      <motion.button
                        key={theme.id}
                        onClick={() => handleSelect(theme.id)}
                        onMouseEnter={() => setPreviewTheme(theme.id)}
                        onMouseLeave={() => setPreviewTheme(null)}
                        className={`relative group rounded-2xl overflow-hidden transition-all duration-200 ${
                          isActive
                            ? 'ring-3 ring-offset-2 ring-offset-white shadow-xl scale-[1.02]'
                            : 'hover:shadow-lg hover:scale-[1.01]'
                        }`}
                        style={{
                          border: isActive ? `3px solid ${theme.accent}` : '2px solid #e2e8f0',
                          boxShadow: isActive ? `0 0 20px ${theme.accent}40` : undefined,
                        }}
                        whileHover={{ y: -2 }}
                        whileTap={{ scale: 0.97 }}
                      >
                        {/* Frame Preview Image */}
                        <div className="relative aspect-[16/10] bg-slate-900 overflow-hidden">
                          <img
                            src={`/borders/${theme.id}/frame-full.png?v=${FRAME_VERSION}`}
                            alt={theme.name}
                            className="w-full h-full object-fill"
                            loading="lazy"
                          />

                          {/* Active checkmark overlay */}
                          {isActive && (
                            <motion.div
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className="absolute top-2 right-2 p-1.5 rounded-full shadow-lg"
                              style={{ backgroundColor: theme.accent }}
                            >
                              <Check className="w-3.5 h-3.5 text-white" strokeWidth={3} />
                            </motion.div>
                          )}

                          {/* Hover glow overlay */}
                          <div
                            className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                            style={{
                              background: `radial-gradient(ellipse at center, ${theme.accent}15 0%, transparent 70%)`,
                            }}
                          />
                        </div>

                        {/* Theme Name + Font Preview */}
                        <div className="px-3 py-2.5 bg-white">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-3 h-3 rounded-full flex-shrink-0 shadow-sm"
                              style={{ backgroundColor: theme.accent }}
                            />
                            <span
                              className={`text-sm font-semibold truncate ${
                                isActive ? 'text-slate-900' : 'text-slate-600'
                              }`}
                              style={{ fontFamily: theme.font }}
                            >
                              {theme.name}
                            </span>
                          </div>
                          {/* Accent color chips */}
                          <div className="flex items-center gap-1 mt-1.5 ml-5">
                            <div className="w-4 h-1.5 rounded-full" style={{ backgroundColor: theme.accent }} />
                            <div className="w-4 h-1.5 rounded-full" style={{ backgroundColor: theme.accentAlt }} />
                            <div className="w-4 h-1.5 rounded-full" style={{ backgroundColor: theme.borderGlow }} />
                          </div>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>

                {/* Current theme indicator at bottom */}
                <div className="mt-6 flex items-center justify-center gap-2 text-sm text-slate-500">
                  <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: THEME_LIST.find(t => t.id === selectedTheme)?.accent }}
                  />
                  <span>
                    Active: <span className="font-semibold text-slate-700">
                      {THEME_LIST.find(t => t.id === selectedTheme)?.name}
                    </span>
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
}
