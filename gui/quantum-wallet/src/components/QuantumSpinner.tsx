import React from 'react';
import { motion } from 'framer-motion';

interface QuantumSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'default' | 'orbital' | 'pulse' | 'helix' | 'quantum';
  message?: string;
  progress?: number; // 0-100 for progress bar
  className?: string;
}

const QuantumSpinner: React.FC<QuantumSpinnerProps> = ({
  size = 'md',
  variant = 'quantum',
  message,
  progress,
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  };

  const renderSpinner = () => {
    switch (variant) {
      case 'orbital':
        return (
          <div className={`relative ${sizeClasses[size]}`}>
            {/* Central core */}
            <motion.div
              className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500"
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            />
            {/* Orbiting particles */}
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-white rounded-full shadow-lg"
                style={{
                  top: '50%',
                  left: '50%',
                  transformOrigin: `${20 + i * 8}px 0`,
                }}
                animate={{ rotate: 360 }}
                transition={{
                  duration: 1.5 + i * 0.5,
                  repeat: Infinity,
                  ease: 'linear',
                  delay: i * 0.2,
                }}
              />
            ))}
          </div>
        );

      case 'pulse':
        return (
          <motion.div
            className={`${sizeClasses[size]} rounded-full bg-gradient-to-r from-cyan-500 to-purple-600`}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        );

      case 'helix':
        return (
          <div className={`relative ${sizeClasses[size]}`}>
            {/* DNA-like double helix */}
            {[0, 1].map((strand) => (
              <motion.div
                key={strand}
                className="absolute inset-0"
                animate={{ rotateY: 360 }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: 'linear',
                  delay: strand * 1.5,
                }}
              >
                {[0, 1, 2, 3].map((i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 bg-gradient-to-r from-cyan-400 to-purple-500 rounded-full"
                    style={{
                      top: `${25 * i}%`,
                      left: strand === 0 ? '20%' : '80%',
                    }}
                    animate={{
                      x: strand === 0 ? [0, 20, 0] : [0, -20, 0],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'easeInOut',
                      delay: i * 0.2,
                    }}
                  />
                ))}
              </motion.div>
            ))}
          </div>
        );

      case 'quantum':
        return (
          <div className={`relative ${sizeClasses[size]}`}>
            {/* Quantum field effect */}
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-cyan-400 border-opacity-30"
              animate={{
                scale: [1, 1.5, 1],
                rotate: 360,
                borderColor: ['rgba(34, 211, 238, 0.3)', 'rgba(168, 85, 247, 0.3)', 'rgba(236, 72, 153, 0.3)', 'rgba(34, 211, 238, 0.3)'],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />

            {/* Inner quantum particles */}
            {[0, 1, 2, 3, 4].map((i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 bg-white rounded-full shadow-lg"
                style={{
                  top: '50%',
                  left: '50%',
                  transformOrigin: `${15}px 0`,
                }}
                animate={{
                  rotate: 360,
                  scale: [0.5, 1, 0.5],
                  opacity: [0.3, 1, 0.3],
                }}
                transition={{
                  duration: 2 + i * 0.3,
                  repeat: Infinity,
                  ease: 'linear',
                  delay: i * 0.4,
                }}
              />
            ))}

            {/* Central core with glow */}
            <motion.div
              className="absolute inset-2 rounded-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 shadow-lg"
              style={{
                boxShadow: '0 0 20px rgba(34, 211, 238, 0.5), 0 0 40px rgba(168, 85, 247, 0.3)',
              }}
              animate={{
                rotate: -360,
                boxShadow: [
                  '0 0 20px rgba(34, 211, 238, 0.5), 0 0 40px rgba(168, 85, 247, 0.3)',
                  '0 0 30px rgba(168, 85, 247, 0.7), 0 0 50px rgba(236, 72, 153, 0.4)',
                  '0 0 20px rgba(34, 211, 238, 0.5), 0 0 40px rgba(168, 85, 247, 0.3)',
                ],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                ease: 'linear',
              }}
            />
          </div>
        );

      default:
        return (
          <motion.div
            className={`${sizeClasses[size]} border-4 border-cyan-200 border-t-cyan-500 rounded-full`}
            animate={{ rotate: 360 }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        );
    }
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-4 ${className}`}>
      {/* Spinner */}
      <div className="relative">
        {renderSpinner()}
      </div>

      {/* Progress bar */}
      {progress !== undefined && (
        <div className="w-full max-w-xs">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Progress</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 rounded-full relative"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            >
              {/* Animated glow effect */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30"
                animate={{
                  x: ['-100%', '100%'],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />
            </motion.div>
          </div>
        </div>
      )}

      {/* Message */}
      {message && (
        <motion.p
          className="text-sm text-gray-300 text-center max-w-xs"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {message}
        </motion.p>
      )}

      {/* Quantum sparkles */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0, 1, 0],
              scale: [0, 1, 0],
            }}
            transition={{
              duration: 2 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default QuantumSpinner;