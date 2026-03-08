import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  Send,
  BarChart3,
  ArrowDownUp,
  Pickaxe,
  Wallet,
  Search,
  X,
  MessageSquare,
  Zap,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════
// AIWheelButton — Floating AI assistant with radial tool wheel
// v8.9.0: Provides quick access to AI-powered wallet actions
// from any screen. Renders as a floating button in the bottom-
// right corner that expands into a radial menu on click.
// ═══════════════════════════════════════════════════════════════

interface WheelAction {
  id: string;
  icon: typeof Sparkles;
  label: string;
  color: string;
  bgColor: string;
  description: string;
}

const WHEEL_ACTIONS: WheelAction[] = [
  {
    id: 'chat',
    icon: MessageSquare,
    label: 'AI Chat',
    color: '#22d3ee',
    bgColor: 'rgba(34, 211, 238, 0.15)',
    description: 'Ask the AI assistant anything',
  },
  {
    id: 'send',
    icon: Send,
    label: 'Quick Send',
    color: '#a78bfa',
    bgColor: 'rgba(167, 139, 250, 0.15)',
    description: 'Send QUG with natural language',
  },
  {
    id: 'swap',
    icon: ArrowDownUp,
    label: 'Swap',
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.15)',
    description: 'Swap tokens on the DEX',
  },
  {
    id: 'balance',
    icon: Wallet,
    label: 'Balance',
    color: '#10b981',
    bgColor: 'rgba(16, 185, 129, 0.15)',
    description: 'Check wallet balances',
  },
  {
    id: 'mining',
    icon: Pickaxe,
    label: 'Mining',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.15)',
    description: 'View mining stats',
  },
  {
    id: 'analytics',
    icon: BarChart3,
    label: 'Analytics',
    color: '#06b6d4',
    bgColor: 'rgba(6, 182, 212, 0.15)',
    description: 'Network analytics overview',
  },
];

export default function AIWheelButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [hoveredAction, setHoveredAction] = useState<string | null>(null);
  const [quickInput, setQuickInput] = useState('');
  const [showInput, setShowInput] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close wheel when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
        setShowInput(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  // Focus input when it appears
  useEffect(() => {
    if (showInput && inputRef.current) {
      inputRef.current.focus();
    }
  }, [showInput]);

  const handleActionClick = (actionId: string) => {
    setIsOpen(false);
    setShowInput(false);

    // Navigate to the appropriate screen or trigger action
    switch (actionId) {
      case 'chat':
        // Navigate to AI Chat screen
        const chatNav = new CustomEvent('navigate-screen', { detail: { screen: 'aichat' } });
        window.dispatchEvent(chatNav);
        // Also try direct approach via navigation
        document.querySelector<HTMLButtonElement>('[data-nav="aichat"]')?.click();
        break;
      case 'send':
        setShowInput(true);
        setIsOpen(true);
        break;
      case 'swap':
        window.dispatchEvent(new CustomEvent('navigate-screen', { detail: { screen: 'dex' } }));
        document.querySelector<HTMLButtonElement>('[data-nav="dex"]')?.click();
        break;
      case 'balance':
        window.dispatchEvent(new CustomEvent('navigate-screen', { detail: { screen: 'dashboard' } }));
        document.querySelector<HTMLButtonElement>('[data-nav="dashboard"]')?.click();
        break;
      case 'mining':
        window.dispatchEvent(new CustomEvent('navigate-screen', { detail: { screen: 'mining' } }));
        document.querySelector<HTMLButtonElement>('[data-nav="mining"]')?.click();
        break;
      case 'analytics':
        window.dispatchEvent(new CustomEvent('navigate-screen', { detail: { screen: 'analytics' } }));
        document.querySelector<HTMLButtonElement>('[data-nav="analytics"]')?.click();
        break;
    }
  };

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!quickInput.trim()) return;

    // Navigate to AI chat with the pre-filled message
    localStorage.setItem('aiChatPrefill', quickInput);
    window.dispatchEvent(new CustomEvent('navigate-screen', { detail: { screen: 'aichat' } }));
    document.querySelector<HTMLButtonElement>('[data-nav="aichat"]')?.click();

    setQuickInput('');
    setShowInput(false);
    setIsOpen(false);
  };

  // Radial positions for 6 items around a circle
  const radius = 90;
  const getPosition = (index: number, total: number) => {
    // Spread items in an arc from -150deg to -30deg (upper-left half)
    const startAngle = -150;
    const endAngle = -30;
    const angle = startAngle + (index / (total - 1)) * (endAngle - startAngle);
    const rad = (angle * Math.PI) / 180;
    return {
      x: Math.cos(rad) * radius,
      y: Math.sin(rad) * radius,
    };
  };

  return (
    <div
      ref={containerRef}
      className="fixed bottom-6 right-6 z-[9990]"
      style={{ pointerEvents: 'auto' }}
    >
      {/* Quick input bar */}
      <AnimatePresence>
        {showInput && (
          <motion.form
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            onSubmit={handleQuickSubmit}
            className="absolute bottom-16 right-0 w-80"
          >
            <div
              className="flex items-center gap-2 rounded-xl px-4 py-3 backdrop-blur-2xl"
              style={{
                background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.95), rgba(20, 15, 40, 0.95))',
                border: '1px solid rgba(34, 211, 238, 0.3)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(34, 211, 238, 0.1)',
              }}
            >
              <Sparkles className="w-4 h-4 text-cyan-400 flex-shrink-0" />
              <input
                ref={inputRef}
                type="text"
                value={quickInput}
                onChange={(e) => setQuickInput(e.target.value)}
                placeholder="Ask AI anything..."
                className="flex-1 bg-transparent text-sm text-white placeholder-gray-500 outline-none"
              />
              <button
                type="submit"
                className="p-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 transition-colors"
              >
                <Send className="w-3.5 h-3.5 text-cyan-400" />
              </button>
            </div>
          </motion.form>
        )}
      </AnimatePresence>

      {/* Radial wheel actions */}
      <AnimatePresence>
        {isOpen && !showInput && (
          <>
            {/* Backdrop glow */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute bottom-0 right-0 w-64 h-64 pointer-events-none"
              style={{
                background: 'radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.08) 0%, transparent 70%)',
              }}
            />

            {WHEEL_ACTIONS.map((action, i) => {
              const pos = getPosition(i, WHEEL_ACTIONS.length);
              const isHovered = hoveredAction === action.id;

              return (
                <motion.button
                  key={action.id}
                  initial={{ opacity: 0, x: 0, y: 0, scale: 0.3 }}
                  animate={{
                    opacity: 1,
                    x: pos.x,
                    y: pos.y,
                    scale: isHovered ? 1.15 : 1,
                  }}
                  exit={{ opacity: 0, x: 0, y: 0, scale: 0.3 }}
                  transition={{
                    delay: i * 0.04,
                    duration: 0.3,
                    type: 'spring',
                    stiffness: 300,
                    damping: 20,
                  }}
                  onMouseEnter={() => setHoveredAction(action.id)}
                  onMouseLeave={() => setHoveredAction(null)}
                  onClick={() => handleActionClick(action.id)}
                  className="absolute bottom-2 right-2 w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-xl transition-shadow"
                  style={{
                    background: action.bgColor,
                    border: `1.5px solid ${action.color}40`,
                    boxShadow: isHovered
                      ? `0 0 20px ${action.color}40, 0 4px 12px rgba(0,0,0,0.3)`
                      : `0 2px 8px rgba(0,0,0,0.3)`,
                  }}
                >
                  <action.icon
                    className="w-4.5 h-4.5"
                    style={{ color: action.color, width: 18, height: 18 }}
                  />

                  {/* Label tooltip */}
                  <AnimatePresence>
                    {isHovered && (
                      <motion.div
                        initial={{ opacity: 0, x: 8 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 8 }}
                        className="absolute right-full mr-3 whitespace-nowrap"
                      >
                        <div
                          className="px-3 py-1.5 rounded-lg text-xs font-semibold"
                          style={{
                            background: 'rgba(15, 10, 35, 0.95)',
                            border: `1px solid ${action.color}30`,
                            color: action.color,
                            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
                          }}
                        >
                          {action.label}
                          <div className="text-[9px] text-gray-500 font-normal mt-0.5">
                            {action.description}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              );
            })}
          </>
        )}
      </AnimatePresence>

      {/* Main floating button */}
      <motion.button
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        onClick={() => {
          setIsOpen(!isOpen);
          setShowInput(false);
          setHoveredAction(null);
        }}
        className="relative w-14 h-14 rounded-full flex items-center justify-center shadow-2xl"
        style={{
          background: isOpen
            ? 'linear-gradient(135deg, #1e1b4b, #0f172a)'
            : 'linear-gradient(135deg, #0e7490, #06b6d4, #22d3ee)',
          border: isOpen
            ? '2px solid rgba(239, 68, 68, 0.4)'
            : '2px solid rgba(34, 211, 238, 0.5)',
          boxShadow: isOpen
            ? '0 0 20px rgba(239, 68, 68, 0.2), 0 8px 32px rgba(0,0,0,0.4)'
            : '0 0 25px rgba(34, 211, 238, 0.3), 0 8px 32px rgba(0,0,0,0.4)',
        }}
      >
        <motion.div
          animate={{ rotate: isOpen ? 135 : 0 }}
          transition={{ duration: 0.3, type: 'spring', stiffness: 200 }}
        >
          {isOpen ? (
            <X className="w-6 h-6 text-red-400" />
          ) : (
            <Sparkles className="w-6 h-6 text-white" />
          )}
        </motion.div>

        {/* Pulse ring when closed */}
        {!isOpen && (
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{ border: '2px solid rgba(34, 211, 238, 0.4)' }}
            animate={{
              scale: [1, 1.4, 1.4],
              opacity: [0.6, 0, 0],
            }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              ease: 'easeOut',
            }}
          />
        )}
      </motion.button>
    </div>
  );
}
