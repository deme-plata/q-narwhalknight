import { useEffect, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Zap, Eye, Globe, CheckCircle, Clock } from 'lucide-react';

interface MixingStage {
  name: string;
  progress: number;
  color: string;
  icon: React.ReactNode;
  description: string;
}

interface QuantumMixerVisualizationProps {
  sessionId: string;
  privacyLevel: 'standard' | 'high' | 'maximum';
  onComplete?: () => void;
}

// Animated particle representing a transaction in the mixing pool
function MixingParticle({ position, color, scale, speed }: any) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [offset] = useState(() => Math.random() * Math.PI * 2);

  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.getElapsedTime();
      meshRef.current.position.y = Math.sin(time * speed + offset) * 0.5;
      meshRef.current.rotation.x = time * 0.5;
      meshRef.current.rotation.y = time * 0.3;

      // Pulsing effect
      const pulseScale = 1 + Math.sin(time * 2 + offset) * 0.1;
      meshRef.current.scale.set(scale * pulseScale, scale * pulseScale, scale * pulseScale);
    }
  });

  return (
    <Sphere ref={meshRef} args={[0.1, 16, 16]} position={position}>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
    </Sphere>
  );
}

// Ring signature visualization
function RingSignatureRing({ radius, particleCount, color, progress }: any) {
  const points = [];
  const actualCount = Math.floor(particleCount * progress);

  for (let i = 0; i < actualCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    points.push(new THREE.Vector3(x, 0, z));
  }

  // Close the ring if complete
  if (progress >= 0.99 && points.length > 0) {
    points.push(points[0]);
  }

  return points.length > 1 ? (
    <Line points={points} color={color} lineWidth={2} />
  ) : null;
}

// Decoy transaction cloud
function DecoyCloud({ count, spread, color }: any) {
  const decoys = [];

  for (let i = 0; i < count; i++) {
    const position = [
      (Math.random() - 0.5) * spread,
      (Math.random() - 0.5) * spread,
      (Math.random() - 0.5) * spread,
    ];
    decoys.push(
      <MixingParticle
        key={i}
        position={position}
        color={color}
        scale={0.5 + Math.random() * 0.3}
        speed={0.5 + Math.random() * 0.5}
      />
    );
  }

  return <>{decoys}</>;
}

// Main 3D scene
function MixerScene({ stage, progress }: { stage: number; progress: number }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00ffff" />

      {/* Central transaction being mixed */}
      <MixingParticle position={[0, 0, 0]} color="#9333ea" scale={1.5} speed={0.3} />

      {/* Stage 1: Ring Signatures (0-20%) */}
      {stage >= 0 && (
        <>
          <RingSignatureRing
            radius={2}
            particleCount={11}
            color="#22c55e"
            progress={Math.min(progress / 0.2, 1)}
          />
          {progress > 0.1 && (
            <Text position={[0, 2.5, 0]} fontSize={0.3} color="#22c55e">
              Ring Signatures
            </Text>
          )}
        </>
      )}

      {/* Stage 2: Decoy Generation (20-40%) */}
      {stage >= 1 && progress > 0.2 && (
        <>
          <DecoyCloud
            count={Math.floor((progress - 0.2) / 0.2 * 15)}
            spread={4}
            color="#f59e0b"
          />
          {progress > 0.3 && (
            <Text position={[0, -2.5, 0]} fontSize={0.3} color="#f59e0b">
              Decoy Amplification
            </Text>
          )}
        </>
      )}

      {/* Stage 3: Stealth Addresses (40-60%) */}
      {stage >= 2 && progress > 0.4 && (
        <>
          <Sphere args={[3, 32, 32]} position={[0, 0, 0]}>
            <meshBasicMaterial
              color="#06b6d4"
              wireframe
              transparent
              opacity={0.3 * ((progress - 0.4) / 0.2)}
            />
          </Sphere>
          {progress > 0.5 && (
            <Text position={[3.5, 0, 0]} fontSize={0.3} color="#06b6d4">
              Stealth Layer
            </Text>
          )}
        </>
      )}

      {/* Stage 4: Dandelion++ Gossip (60-80%) */}
      {stage >= 3 && progress > 0.6 && (
        <>
          {[...Array(5)].map((_, i) => {
            const angle = (i / 5) * Math.PI * 2;
            const distance = 4 + (progress - 0.6) / 0.2 * 2;
            return (
              <MixingParticle
                key={`gossip-${i}`}
                position={[
                  Math.cos(angle) * distance,
                  Math.sin(angle) * 0.5,
                  Math.sin(angle) * distance,
                ]}
                color="#ec4899"
                scale={0.8}
                speed={1}
              />
            );
          })}
          {progress > 0.7 && (
            <Text position={[-3.5, 0, 0]} fontSize={0.3} color="#ec4899">
              Network Gossip
            </Text>
          )}
        </>
      )}

      {/* Stage 5: Quantum Finalization (80-100%) */}
      {stage >= 4 && progress > 0.8 && (
        <>
          <Sphere args={[1.8, 32, 32]} position={[0, 0, 0]}>
            <meshStandardMaterial
              color="#8b5cf6"
              emissive="#8b5cf6"
              emissiveIntensity={0.5 * ((progress - 0.8) / 0.2)}
              metalness={0.8}
              roughness={0.2}
            />
          </Sphere>
          {progress > 0.9 && (
            <Text position={[0, 3.5, 0]} fontSize={0.4} color="#8b5cf6">
              Quantum Sealed
            </Text>
          )}
        </>
      )}
    </group>
  );
}

export default function QuantumMixerVisualization({
  sessionId,
  privacyLevel,
  onComplete
}: QuantumMixerVisualizationProps) {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);

  const stages: MixingStage[] = [
    {
      name: 'Ring Signatures',
      progress: 0.2,
      color: '#22c55e',
      icon: <Shield className="w-5 h-5" />,
      description: 'Forming anonymity set of 11+ participants',
    },
    {
      name: 'Decoy Generation',
      progress: 0.4,
      color: '#f59e0b',
      icon: <Zap className="w-5 h-5" />,
      description: 'Creating 15x decoy transactions',
    },
    {
      name: 'Stealth Addresses',
      progress: 0.6,
      color: '#06b6d4',
      icon: <Eye className="w-5 h-5" />,
      description: 'Generating unlinkable addresses',
    },
    {
      name: 'Dandelion++ Gossip',
      progress: 0.8,
      color: '#ec4899',
      icon: <Globe className="w-5 h-5" />,
      description: 'Anonymous network propagation',
    },
    {
      name: 'Quantum Finalization',
      progress: 1.0,
      color: '#8b5cf6',
      icon: <CheckCircle className="w-5 h-5" />,
      description: 'Sealing with quantum entropy',
    },
  ];

  useEffect(() => {
    // Determine duration based on privacy level (standard=15s, high=30s, maximum=60s)
    const durationSeconds = privacyLevel === 'standard' ? 15 : privacyLevel === 'high' ? 30 : 60;
    const duration = durationSeconds * 1000; // Convert to milliseconds
    const interval = 100; // Update every 100ms
    const increment = interval / duration;

    console.log(`🎨 [MIXER VIZ] Starting ${privacyLevel} privacy level (${durationSeconds}s duration)`);

    const timer = setInterval(() => {
      setProgress((prev) => {
        const newProgress = Math.min(prev + increment, 1);

        // Update stage based on progress
        const newStage = stages.findIndex(s => newProgress < s.progress);
        setStage(newStage === -1 ? stages.length - 1 : newStage);

        // Check if complete
        if (newProgress >= 1) {
          setIsComplete(true);
          clearInterval(timer);
          console.log(`✅ [MIXER VIZ] Mixing visualization complete after ${durationSeconds}s`);
          if (onComplete) {
            setTimeout(onComplete, 1000);
          }
        }

        return newProgress;
      });

      setElapsedTime((prev) => prev + interval);
    }, interval);

    // Store session ID in localStorage for persistence
    localStorage.setItem('activeMixingSession', sessionId);
    localStorage.setItem('mixingStartTime', Date.now().toString());

    return () => {
      clearInterval(timer);
      if (isComplete) {
        localStorage.removeItem('activeMixingSession');
        localStorage.removeItem('mixingStartTime');
      }
    };
  }, [sessionId, onComplete, privacyLevel]); // Added privacyLevel to dependencies

  // Calculate remaining time based on privacy level
  const totalDuration = privacyLevel === 'standard' ? 15 : privacyLevel === 'high' ? 30 : 60;
  const remainingTime = Math.max(0, totalDuration - Math.floor(elapsedTime / 1000));

  return (
    <div className="relative w-full h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* 3D Canvas */}
      <Canvas camera={{ position: [0, 5, 10], fov: 50 }}>
        <OrbitControls
          enableZoom={true}
          enablePan={true}
          autoRotate={!isComplete}
          autoRotateSpeed={0.5}
        />
        <MixerScene stage={stage} progress={progress} />
      </Canvas>

      {/* HUD Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Top Info Bar */}
        <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-auto">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-slate-800/80 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30"
          >
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-purple-500 rounded-full blur-md animate-pulse"></div>
                <Shield className="w-8 h-8 text-purple-400 relative z-10" />
              </div>
              <div>
                <div className="text-sm text-slate-400">Quantum Privacy Mixing</div>
                <div className="text-xl font-bold text-white">
                  {privacyLevel.charAt(0).toUpperCase() + privacyLevel.slice(1)} Level
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-slate-800/80 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30"
          >
            <div className="flex items-center gap-3">
              <Clock className="w-6 h-6 text-purple-400" />
              <div>
                <div className="text-sm text-slate-400">Time Remaining</div>
                <div className="text-2xl font-mono font-bold text-white">
                  {remainingTime}s
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Progress Bar */}
        <div className="absolute bottom-8 left-8 right-8 pointer-events-auto">
          <div className="bg-slate-800/80 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
            {/* Overall Progress */}
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-400">Overall Progress</span>
                <span className="text-sm font-mono text-purple-400">
                  {Math.floor(progress * 100)}%
                </span>
              </div>
              <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500"
                  style={{ width: `${progress * 100}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
            </div>

            {/* Stages */}
            <div className="grid grid-cols-5 gap-4">
              <AnimatePresence>
                {stages.map((stageInfo, index) => {
                  const isActive = index === stage;
                  const isPast = index < stage || isComplete;
                  const opacity = isPast ? 1 : isActive ? 1 : 0.5;

                  return (
                    <motion.div
                      key={stageInfo.name}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{
                        opacity,
                        scale: isActive ? 1.05 : 1,
                      }}
                      transition={{ delay: index * 0.1 }}
                      className={`
                        relative p-3 rounded-lg border-2 transition-all duration-300
                        ${isPast
                          ? 'bg-gradient-to-br from-green-900/50 to-emerald-900/50 border-green-500/50'
                          : isActive
                            ? 'bg-gradient-to-br from-purple-900/50 to-pink-900/50 border-purple-500 shadow-lg shadow-purple-500/50'
                            : 'bg-slate-800/50 border-slate-700'
                        }
                      `}
                    >
                      {isPast && (
                        <div className="absolute -top-2 -right-2 bg-green-500 rounded-full p-1">
                          <CheckCircle className="w-4 h-4 text-white" />
                        </div>
                      )}

                      <div className="flex flex-col items-center gap-2">
                        <div style={{ color: stageInfo.color }}>
                          {stageInfo.icon}
                        </div>
                        <div className="text-xs font-semibold text-white text-center">
                          {stageInfo.name}
                        </div>
                        <div className="text-[10px] text-slate-400 text-center leading-tight">
                          {stageInfo.description}
                        </div>
                      </div>

                      {isActive && (
                        <div className="absolute inset-0 rounded-lg animate-pulse">
                          <div className="absolute inset-0 rounded-lg border-2 border-purple-400/50"></div>
                        </div>
                      )}
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>

            {/* Completion Message */}
            <AnimatePresence>
              {isComplete && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="mt-6 p-4 bg-gradient-to-r from-green-900/50 to-emerald-900/50 rounded-lg border-2 border-green-500/50"
                >
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-8 h-8 text-green-400" />
                    <div>
                      <div className="text-lg font-bold text-white">
                        Quantum Mixing Complete!
                      </div>
                      <div className="text-sm text-slate-300">
                        Your transaction is now fully anonymous and ready for confirmation
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Session ID (for debugging) */}
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 pointer-events-auto">
          <div className="bg-slate-800/60 backdrop-blur-sm rounded-lg px-4 py-2 border border-slate-700">
            <div className="text-xs text-slate-400 font-mono">
              Session: {sessionId.substring(0, 16)}...
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
