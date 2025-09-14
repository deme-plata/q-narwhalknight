import { motion } from 'framer-motion';
import { Home, Send, Settings, Activity, Search } from 'lucide-react';

type Screen = 'dashboard' | 'transactions' | 'explorer' | 'settings';

interface NavigationProps {
  currentScreen: Screen;
  onNavigate: (screen: Screen) => void;
  className?: string;
}

export default function Navigation({ currentScreen, onNavigate, className }: NavigationProps) {
  const navItems = [
    { id: 'dashboard' as Screen, icon: Home, label: 'Dashboard' },
    { id: 'transactions' as Screen, icon: Send, label: 'Transactions' },
    { id: 'explorer' as Screen, icon: Search, label: 'Explorer' },
    { id: 'settings' as Screen, icon: Settings, label: 'Settings' },
  ];

  return (
    <nav className={`${className} bg-quantum-indigo/30 backdrop-blur-xl border-r border-quantum-purple/20 lg:border-r-0 lg:border-t border-quantum-purple/20 fixed bottom-0 left-0 right-0 lg:static lg:h-full`}>
      {/* Desktop Navigation */}
      <div className="hidden lg:flex flex-col h-full p-6">
        <div className="flex items-center gap-3 mb-12">
          <div className="w-10 h-10 rainbow-box rounded-lg flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <span className="xl:block hidden text-xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
            Q-Wallet
          </span>
        </div>

        <div className="space-y-4 flex-1">
          {navItems.map((item) => (
            <motion.button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-4 p-4 rounded-xl transition-all relative overflow-hidden ${
                currentScreen === item.id
                  ? 'bg-gradient-to-r from-quantum-purple/30 to-quantum-cyan/30 text-white border border-quantum-cyan/30'
                  : 'text-gray-400 hover:text-white hover:bg-quantum-purple/20'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {currentScreen === item.id && (
                <motion.div
                  className="absolute inset-0 rainbow-box opacity-10"
                  initial={{ x: '-100%' }}
                  animate={{ x: '0%' }}
                  transition={{ type: 'spring', bounce: 0.2 }}
                />
              )}
              <item.icon className="w-6 h-6 flex-shrink-0" />
              <span className="xl:block hidden font-medium">{item.label}</span>
            </motion.button>
          ))}
        </div>

        {/* Quantum Status Indicator */}
        <div className="mt-auto">
          <div className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-green/30">
            <div className="flex items-center gap-3">
              <motion.div
                className="w-3 h-3 bg-quantum-green rounded-full"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="xl:block hidden text-sm text-quantum-green">Network Online</span>
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
                ? 'text-quantum-cyan'
                : 'text-gray-500'
            }`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {currentScreen === item.id && (
              <motion.div
                className="absolute -top-1 w-12 h-1 bg-gradient-to-r from-quantum-purple to-quantum-cyan rounded-full"
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