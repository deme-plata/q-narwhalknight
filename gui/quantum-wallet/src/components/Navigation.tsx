import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Home, Send, Settings, Search, ArrowDownUp, Pickaxe, Boxes, Download, MessageSquare, Building, Mail, BarChart3, MapPin, Activity, Video, Magnet, MoreHorizontal, X } from 'lucide-react';

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'rwamarket' | 'gameitems' | 'download' | 'aichat' | 'email' | 'analytics' | 'settings' | 'map' | 'bank' | 'chat' | 'torrent';

const MASTER_WALLET = 'efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';

interface NavigationProps {
  currentScreen: Screen;
  onNavigate: (screen: Screen) => void;
  className?: string;
  walletAddress?: string;
}

export default function Navigation({ currentScreen, onNavigate, className, walletAddress }: NavigationProps) {
  const isMaster = (walletAddress || '').toLowerCase().replace(/^0x/, '') === MASTER_WALLET;
  const [moreOpen, setMoreOpen] = useState(false);

  const navItems = [
    { id: 'dashboard' as Screen, icon: Home, label: 'Dashboard' },
    { id: 'transactions' as Screen, icon: Send, label: 'Transactions' },
    { id: 'dex' as Screen, icon: ArrowDownUp, label: 'DEX' },
    { id: 'explorer' as Screen, icon: Search, label: 'Explorer' },
    { id: 'mining' as Screen, icon: Pickaxe, label: 'Mining' },
    { id: 'vm' as Screen, icon: Boxes, label: 'QVM' },
    { id: 'map' as Screen, icon: MapPin, label: 'Map' },
    { id: 'rwamarket' as Screen, icon: Building, label: 'RWA' },
    { id: 'chat' as Screen, icon: Video, label: 'Chat & Calls' },
    { id: 'aichat' as Screen, icon: MessageSquare, label: 'AI Chat' },
    { id: 'email' as Screen, icon: Mail, label: 'Mail' },
    { id: 'analytics' as Screen, icon: BarChart3, label: 'Analytics' },
    { id: 'download' as Screen, icon: Download, label: 'Downloads' },
    ...(isMaster ? [{ id: 'torrent' as Screen, icon: Magnet, label: 'Torrent' }] : []),
    { id: 'settings' as Screen, icon: Settings, label: 'Settings' },
  ];

  // Mobile/tablet bottom bar: only the 4 most-used screens get a permanent,
  // properly-sized touch target. Everything else lives behind "More" — cramming
  // all 14+ items into one row (the old behavior) gave each button ~27px on a
  // typical phone width, well under the 44-48px minimum touch target guidelines
  // (Apple HIG / Material Design) — that's the actual cause of "hard to press".
  const primaryIds: Screen[] = ['dashboard', 'transactions', 'dex', 'explorer'];
  const primaryItems = navItems.filter((item) => primaryIds.includes(item.id));
  const moreItems = navItems.filter((item) => !primaryIds.includes(item.id));
  const moreItemActive = moreItems.some((item) => item.id === currentScreen);

  return (
    <nav
      className={`${className} backdrop-blur-xl border-r lg:border-r-0 lg:border-t fixed bottom-0 left-0 right-0 lg:static lg:h-full`}
      style={{
        background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
        borderColor: 'rgba(212, 175, 55, 0.2)',
        boxShadow: '0 0 20px rgba(212, 175, 55, 0.1)'
      }}
    >
      {/* Desktop Navigation */}
      <div className="hidden lg:flex flex-col h-full p-6">
        <div className="flex items-center gap-3 mb-12">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center relative"
            style={{
              background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
              boxShadow: '0 0 15px rgba(212, 175, 55, 0.4)'
            }}
          >
            <Activity className="w-6 h-6 text-slate-900" />
          </div>
          <span className="xl:block hidden text-xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent">
            Quillon Graph
          </span>
        </div>

        <div className="space-y-3 flex-1">
          {navItems.map((item) => (
            <motion.button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-4 p-4 rounded-xl transition-all relative overflow-hidden ${
                currentScreen === item.id
                  ? 'text-amber-50'
                  : 'text-amber-200/50 hover:text-amber-100'
              }`}
              style={
                currentScreen === item.id
                  ? {
                      background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.15) 100%)',
                      border: '2px solid rgba(212, 175, 55, 0.4)',
                      boxShadow: '0 0 20px rgba(212, 175, 55, 0.2), inset 0 0 15px rgba(212, 175, 55, 0.1)'
                    }
                  : {
                      border: '2px solid transparent'
                    }
              }
              whileHover={{ scale: 1.03, x: 5 }}
              whileTap={{ scale: 0.97 }}
            >
              {currentScreen === item.id && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-amber-500/10 via-yellow-500/10 to-amber-500/10"
                  initial={{ x: '-100%' }}
                  animate={{ x: '100%' }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
              )}
              <item.icon className={`w-6 h-6 flex-shrink-0 ${currentScreen === item.id ? 'text-amber-400' : ''}`} />
              <span className="xl:block hidden font-semibold">{item.label}</span>
              {currentScreen === item.id && (
                <motion.div
                  className="ml-auto w-2 h-2 rounded-full bg-amber-400"
                  animate={{ scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
            </motion.button>
          ))}
        </div>

        {/* Quantum Status Indicator */}
        <div className="mt-auto">
          <div
            className="p-4 rounded-xl relative overflow-hidden"
            style={{
              background: 'linear-gradient(135deg, rgba(22, 163, 74, 0.15) 0%, rgba(34, 197, 94, 0.1) 100%)',
              border: '2px solid rgba(34, 197, 94, 0.3)',
              boxShadow: '0 0 15px rgba(34, 197, 94, 0.2)'
            }}
          >
            <div className="flex items-center gap-3">
              <motion.div
                className="w-3 h-3 rounded-full"
                style={{ background: 'linear-gradient(135deg, #10B981, #34D399)' }}
                animate={{ scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="xl:block hidden text-sm font-semibold text-green-300">Network Online</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile/Tablet Navigation — primary items get real touch targets (min 56px
          tall including padding), everything else lives in the "More" drawer below. */}
      <div className="lg:hidden flex justify-around items-stretch h-16 px-1 safe-area-pb">
        {primaryItems.map((item) => (
          <motion.button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={`relative flex flex-col items-center justify-center gap-0.5 flex-1 min-w-[56px] rounded-xl ${
              currentScreen === item.id
                ? 'text-amber-400'
                : 'text-amber-300/40'
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.92 }}
          >
            {currentScreen === item.id && (
              <motion.div
                className="absolute -top-1 w-12 h-1 rounded-full"
                style={{
                  background: 'linear-gradient(90deg, #D4AF37, #FFD700, #D4AF37)',
                  boxShadow: '0 0 10px rgba(212, 175, 55, 0.5)'
                }}
                layoutId="mobile-indicator"
              />
            )}
            <item.icon className="w-6 h-6" />
            <span className="text-[11px] font-medium leading-tight">{item.label}</span>
          </motion.button>
        ))}
        <motion.button
          onClick={() => setMoreOpen(true)}
          className={`relative flex flex-col items-center justify-center gap-0.5 flex-1 min-w-[56px] rounded-xl ${
            moreItemActive ? 'text-amber-400' : 'text-amber-300/40'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.92 }}
        >
          {moreItemActive && (
            <motion.div
              className="absolute -top-1 w-12 h-1 rounded-full"
              style={{
                background: 'linear-gradient(90deg, #D4AF37, #FFD700, #D4AF37)',
                boxShadow: '0 0 10px rgba(212, 175, 55, 0.5)'
              }}
              layoutId="mobile-indicator"
            />
          )}
          <MoreHorizontal className="w-6 h-6" />
          <span className="text-[11px] font-medium leading-tight">More</span>
        </motion.button>
      </div>

      {/* "More" drawer — slides up from the bottom, big generously-spaced tap
          targets (min 64px tall each) instead of the old cramped single row. */}
      <AnimatePresence>
        {moreOpen && (
          <>
            <motion.div
              className="lg:hidden fixed inset-0 z-40 bg-black/60"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMoreOpen(false)}
            />
            <motion.div
              className="lg:hidden fixed bottom-0 left-0 right-0 z-50 rounded-t-2xl backdrop-blur-xl border-t max-h-[70vh] overflow-y-auto safe-area-pb"
              style={{
                background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%)',
                borderColor: 'rgba(212, 175, 55, 0.3)',
              }}
              initial={{ y: '100%' }}
              animate={{ y: 0 }}
              exit={{ y: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            >
              <div className="flex items-center justify-between px-5 pt-4 pb-2">
                <span className="text-sm font-semibold text-amber-200/70 uppercase tracking-wide">More</span>
                <button
                  onClick={() => setMoreOpen(false)}
                  className="p-2 -mr-2 rounded-lg text-amber-300/60 hover:text-amber-100"
                  aria-label="Close menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2 px-3 pb-4">
                {moreItems.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => { onNavigate(item.id); setMoreOpen(false); }}
                    className={`flex flex-col items-center justify-center gap-2 py-4 min-h-[64px] rounded-xl transition-colors ${
                      currentScreen === item.id
                        ? 'text-amber-50'
                        : 'text-amber-200/60 active:text-amber-100'
                    }`}
                    style={
                      currentScreen === item.id
                        ? {
                            background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.15) 100%)',
                            border: '2px solid rgba(212, 175, 55, 0.4)',
                          }
                        : { border: '2px solid transparent' }
                    }
                  >
                    <item.icon className={`w-6 h-6 ${currentScreen === item.id ? 'text-amber-400' : ''}`} />
                    <span className="text-xs font-medium text-center leading-tight">{item.label}</span>
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </nav>
  );
}