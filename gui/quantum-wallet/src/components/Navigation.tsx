import { motion } from 'framer-motion';
import { Home, Send, Settings, Search, ArrowDownUp, Pickaxe, Boxes, Download, MessageSquare, Building, Mail, BarChart3, MapPin, Activity, Video, Magnet } from 'lucide-react';

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

      {/* Mobile Navigation */}
      <div className="lg:hidden flex justify-around items-center h-16 px-4 safe-area-pb">
        {navItems.map((item) => (
          <motion.button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={`relative flex flex-col items-center p-3 rounded-xl ${
              currentScreen === item.id
                ? 'text-amber-400'
                : 'text-amber-300/40'
            }`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
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
            <item.icon className="w-6 h-6 mb-1" />
            <span className="text-xs font-medium">{item.label}</span>
          </motion.button>
        ))}
      </div>
    </nav>
  );
}