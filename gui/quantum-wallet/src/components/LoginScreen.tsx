import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Key, AlertCircle, Search, HelpCircle, X, Shield, Zap, Lock, Globe } from 'lucide-react';
import { qnkAPI } from '../services/api';
import { storeWallet, walletSession, verifyPasswordHash, hasPasswordHash } from '../services/walletAuth';
import ExplorerSearchBar from './ExplorerSearchBar';

interface LoginScreenProps {
  onAuthenticate: () => void;
}

// --- Twinkling Universe Background ---
function UniverseBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  const initStars = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return { stars: [], shootingStars: [], nebulae: [] };

    const w = canvas.width = window.innerWidth;
    const h = canvas.height = window.innerHeight;

    // Generate star layers
    const stars: Array<{
      x: number; y: number; radius: number;
      baseAlpha: number; twinkleSpeed: number; twinkleOffset: number;
      color: string;
    }> = [];

    // Deep background stars (tiny, many)
    for (let i = 0; i < 400; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        radius: Math.random() * 0.8 + 0.2,
        baseAlpha: Math.random() * 0.5 + 0.3,
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        twinkleOffset: Math.random() * Math.PI * 2,
        color: ['#ffffff', '#ffe4b5', '#b0c4de', '#add8e6', '#ffd700'][Math.floor(Math.random() * 5)],
      });
    }

    // Medium stars
    for (let i = 0; i < 120; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        radius: Math.random() * 1.5 + 0.8,
        baseAlpha: Math.random() * 0.6 + 0.4,
        twinkleSpeed: Math.random() * 0.03 + 0.01,
        twinkleOffset: Math.random() * Math.PI * 2,
        color: ['#ffffff', '#ffecd2', '#c9d6ff', '#ffd700', '#f0e68c'][Math.floor(Math.random() * 5)],
      });
    }

    // Bright prominent stars (few, large)
    for (let i = 0; i < 25; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        radius: Math.random() * 2.0 + 1.5,
        baseAlpha: Math.random() * 0.3 + 0.7,
        twinkleSpeed: Math.random() * 0.04 + 0.015,
        twinkleOffset: Math.random() * Math.PI * 2,
        color: ['#ffffff', '#ffd700', '#87ceeb', '#f5f5dc', '#fffacd'][Math.floor(Math.random() * 5)],
      });
    }

    // Nebula clouds
    const nebulae: Array<{
      x: number; y: number; radiusX: number; radiusY: number;
      color: string; alpha: number; rotation: number;
    }> = [];
    const nebulaColors = [
      'rgba(212, 175, 55, 0.03)',   // gold
      'rgba(139, 92, 246, 0.025)',  // purple
      'rgba(59, 130, 246, 0.02)',   // blue
      'rgba(245, 158, 11, 0.025)', // amber
      'rgba(168, 85, 247, 0.02)',  // violet
    ];
    for (let i = 0; i < 6; i++) {
      nebulae.push({
        x: Math.random() * w,
        y: Math.random() * h,
        radiusX: Math.random() * 300 + 150,
        radiusY: Math.random() * 200 + 100,
        color: nebulaColors[i % nebulaColors.length],
        alpha: Math.random() * 0.5 + 0.5,
        rotation: Math.random() * Math.PI,
      });
    }

    // Shooting stars
    const shootingStars: Array<{
      x: number; y: number; length: number; speed: number;
      angle: number; alpha: number; active: boolean; timer: number;
      delay: number;
    }> = [];
    for (let i = 0; i < 3; i++) {
      shootingStars.push({
        x: 0, y: 0, length: Math.random() * 80 + 40,
        speed: Math.random() * 8 + 4,
        angle: Math.random() * 0.5 + 0.3,
        alpha: 0, active: false,
        timer: 0,
        delay: Math.random() * 500 + 200,
      });
    }

    return { stars, shootingStars, nebulae };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let { stars, shootingStars, nebulae } = initStars();
    let frame = 0;

    const handleResize = () => {
      const result = initStars();
      stars = result.stars;
      shootingStars = result.shootingStars;
      nebulae = result.nebulae;
    };
    window.addEventListener('resize', handleResize);

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      frame++;

      // Clear with deep space gradient
      const gradient = ctx.createLinearGradient(0, 0, 0, h);
      gradient.addColorStop(0, '#020617');    // slate-950
      gradient.addColorStop(0.3, '#0a0f1e');  // deep navy
      gradient.addColorStop(0.6, '#0c0a1a');  // deep purple-black
      gradient.addColorStop(1, '#050210');     // near black with hint of blue
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);

      // Draw nebulae (soft gradient clouds)
      for (const neb of nebulae) {
        ctx.save();
        ctx.translate(neb.x, neb.y);
        ctx.rotate(neb.rotation);
        const nebGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, neb.radiusX);
        nebGrad.addColorStop(0, neb.color);
        nebGrad.addColorStop(0.5, neb.color.replace(/[\d.]+\)$/, '0.01)'));
        nebGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = nebGrad;
        ctx.scale(1, neb.radiusY / neb.radiusX);
        ctx.beginPath();
        ctx.arc(0, 0, neb.radiusX, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Draw twinkling stars
      for (const star of stars) {
        const twinkle = Math.sin(frame * star.twinkleSpeed + star.twinkleOffset);
        const alpha = star.baseAlpha + twinkle * 0.3;
        const clampedAlpha = Math.max(0.05, Math.min(1, alpha));

        // Star glow
        if (star.radius > 1.2) {
          const glowGrad = ctx.createRadialGradient(
            star.x, star.y, 0,
            star.x, star.y, star.radius * 4
          );
          glowGrad.addColorStop(0, star.color.replace(')', `, ${clampedAlpha * 0.3})`).replace('rgb', 'rgba'));
          glowGrad.addColorStop(1, 'transparent');
          ctx.fillStyle = glowGrad;
          ctx.beginPath();
          ctx.arc(star.x, star.y, star.radius * 4, 0, Math.PI * 2);
          ctx.fill();
        }

        // Star core
        ctx.globalAlpha = clampedAlpha;
        ctx.fillStyle = star.color;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fill();

        // Cross-shaped twinkle for bright stars
        if (star.radius > 1.8 && clampedAlpha > 0.7) {
          const spikeLen = star.radius * 3 * clampedAlpha;
          ctx.strokeStyle = star.color;
          ctx.lineWidth = 0.5;
          ctx.globalAlpha = clampedAlpha * 0.5;
          ctx.beginPath();
          ctx.moveTo(star.x - spikeLen, star.y);
          ctx.lineTo(star.x + spikeLen, star.y);
          ctx.moveTo(star.x, star.y - spikeLen);
          ctx.lineTo(star.x, star.y + spikeLen);
          ctx.stroke();
        }

        ctx.globalAlpha = 1;
      }

      // Draw shooting stars
      for (const ss of shootingStars) {
        ss.timer++;
        if (!ss.active) {
          if (ss.timer > ss.delay) {
            ss.active = true;
            ss.timer = 0;
            ss.x = Math.random() * w * 0.7;
            ss.y = Math.random() * h * 0.4;
            ss.alpha = 1;
            ss.delay = Math.random() * 600 + 300;
          }
          continue;
        }

        // Move shooting star
        ss.x += Math.cos(ss.angle) * ss.speed;
        ss.y += Math.sin(ss.angle) * ss.speed;
        ss.alpha -= 0.015;

        if (ss.alpha <= 0 || ss.x > w || ss.y > h) {
          ss.active = false;
          ss.timer = 0;
          continue;
        }

        // Draw trail
        const tailX = ss.x - Math.cos(ss.angle) * ss.length;
        const tailY = ss.y - Math.sin(ss.angle) * ss.length;
        const trailGrad = ctx.createLinearGradient(tailX, tailY, ss.x, ss.y);
        trailGrad.addColorStop(0, `rgba(255, 255, 255, 0)`);
        trailGrad.addColorStop(0.7, `rgba(255, 215, 0, ${ss.alpha * 0.4})`);
        trailGrad.addColorStop(1, `rgba(255, 255, 255, ${ss.alpha})`);

        ctx.strokeStyle = trailGrad;
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(ss.x, ss.y);
        ctx.stroke();

        // Bright head
        const headGlow = ctx.createRadialGradient(ss.x, ss.y, 0, ss.x, ss.y, 4);
        headGlow.addColorStop(0, `rgba(255, 255, 255, ${ss.alpha})`);
        headGlow.addColorStop(1, `rgba(255, 215, 0, 0)`);
        ctx.fillStyle = headGlow;
        ctx.beginPath();
        ctx.arc(ss.x, ss.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [initStars]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full"
      style={{ zIndex: 0 }}
    />
  );
}

// --- Floating particles around the logo ---
function FloatingParticles() {
  const particles = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    delay: Math.random() * 5,
    duration: Math.random() * 4 + 3,
    x: Math.random() * 160 - 80,
    y: Math.random() * 160 - 80,
    size: Math.random() * 3 + 1,
  }));

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full"
          style={{
            width: p.size,
            height: p.size,
            left: '50%',
            top: '50%',
            background: `radial-gradient(circle, rgba(255, 215, 0, 0.8), rgba(212, 175, 55, 0))`,
          }}
          animate={{
            x: [0, p.x, -p.x * 0.5, p.x * 0.3, 0],
            y: [0, p.y, -p.y * 0.7, p.y * 0.5, 0],
            opacity: [0, 0.8, 0.4, 0.7, 0],
            scale: [0, 1.5, 0.8, 1.2, 0],
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}


export default function LoginScreen({ onAuthenticate }: LoginScreenProps) {
  const [seedPhrase, setSeedPhrase] = useState('');
  const [password, setPassword] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [showQuantumGenerator, setShowQuantumGenerator] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);
  // Persistent Tor onion address for the frontend (v3 hidden service)
  const TOR_ONION_URL = "http://3c6ixbraumi7ljfqzji4ovnsd75kduicu2tuwmg5qrr57zyt76lghsyd.onion";

  // Open Tor version of the site
  const openTorSite = () => {
    window.open(TOR_ONION_URL, '_blank', 'noopener,noreferrer');
  };

  // Validate seed phrase is a proper BIP39 mnemonic (12 or 24 words)
  const validateSeedPhrase = (phrase: string): { valid: boolean; error?: string } => {
    const words = phrase.trim().split(/\s+/).filter(w => w.length > 0);

    if (words.length === 0) {
      return { valid: false, error: 'Seed phrase is required' };
    }

    if (words.length !== 12 && words.length !== 24) {
      return { valid: false, error: `Seed phrase must be exactly 12 or 24 words (got ${words.length} words)` };
    }

    // Check each word is at least 3 characters (BIP39 words are 3-8 chars)
    const invalidWords = words.filter(w => w.length < 3 || w.length > 8);
    if (invalidWords.length > 0) {
      return { valid: false, error: `Invalid word length detected. BIP39 words are 3-8 characters.` };
    }

    return { valid: true };
  };

  const handleAuthenticate = async () => {
    setIsAuthenticating(true);
    setGenerationError(null);

    try {
      // Validate seed phrase FIRST
      const seedValidation = validateSeedPhrase(seedPhrase);
      if (!seedValidation.valid) {
        throw new Error(seedValidation.error);
      }

      // Password is REQUIRED
      if (!password) {
        throw new Error('Password is required for wallet encryption');
      }

      // CRITICAL SECURITY FIX v1.0.68-beta: MANDATORY password verification for existing wallets
      const encryptedMnemonic = localStorage.getItem('walletEncryptedMnemonic');
      const encryptedKey = localStorage.getItem('walletEncryptedKey');
      const storedAddress = localStorage.getItem('walletAddress');

      const { keypairFromMnemonic, recoverMnemonic, loadWallet } = await import('../services/walletAuth');
      const providedKeyPair = await keypairFromMnemonic(seedPhrase);
      const providedWalletAddress = providedKeyPair.address;

      const hasExistingEncryptedWallet = storedAddress && (encryptedMnemonic || encryptedKey);

      if (hasExistingEncryptedWallet && providedWalletAddress === storedAddress) {
        console.log('🔐 Existing wallet found - MANDATORY password verification...');
        console.log('🔐 Same wallet detected (addresses match) - password verification is MANDATORY');

        try {
          if (encryptedMnemonic) {
            const storedMnemonic = await recoverMnemonic(password);
            if (storedMnemonic.trim() !== seedPhrase.trim()) {
              console.error('❌ CRITICAL: Address matched but mnemonic different!');
              throw new Error('Wallet data corruption detected. Please contact support.');
            }
            console.log('✅ Password verified via mnemonic decryption!');
          } else if (encryptedKey) {
            await loadWallet(password);
            console.log('✅ Password verified via key decryption (legacy wallet)!');
          }
        } catch (decryptError) {
          console.error('❌ WRONG PASSWORD - Authentication BLOCKED');
          console.error('   Error:', decryptError);
          setIsAuthenticating(false);
          setGenerationError('Incorrect password. Please enter the correct password for your existing wallet.');
          return;
        }
      } else if (storedAddress && providedWalletAddress !== storedAddress) {
        console.warn('⚠️ Different wallet detected (address mismatch)');
        localStorage.removeItem('walletEncryptedMnemonic');
        localStorage.removeItem('walletEncryptedKey');
        localStorage.removeItem('walletAddress');
        localStorage.removeItem('walletPublicKey');
        localStorage.removeItem('walletEncryptedAegisKey');
        localStorage.removeItem('walletAegisPublicKey');
        localStorage.removeItem('walletPasswordHash');
      } else if (!hasExistingEncryptedWallet && hasPasswordHash()) {
        console.log('🔐 No encrypted data but password hash exists - verifying password...');
        const isPasswordValid = await verifyPasswordHash(password);
        if (!isPasswordValid) {
          console.error('❌ WRONG PASSWORD - Password hash verification failed');
          setIsAuthenticating(false);
          setGenerationError('Incorrect password. Please enter the correct password for your wallet.');
          return;
        }
        console.log('✅ Password verified via hash!');
      }

      const response = await qnkAPI.createWallet(seedPhrase, password);

      if (response.success && response.data) {
        localStorage.setItem('walletAddress', response.data.address_formatted || '');
        localStorage.setItem('walletId', response.data.id);
        localStorage.removeItem('cachedBalance');
        console.log('🗑️ Cleared cached balance from previous wallet');

        try {
          const wallet = await storeWallet(seedPhrase, password, true, true, true);
          walletSession.setSession(
            wallet.privateKey,
            wallet.address,
            seedPhrase,
            wallet.dilithium5SecretKey,
            wallet.dilithium5PublicKey
          );
          console.log('✅ Wallet encrypted with password-protected AES-256-GCM');
          console.log('✅ AEGIS-QL post-quantum keys generated and encrypted');
          if (wallet.dilithium5SecretKey) {
            console.log('✅ Dilithium5 post-quantum keys stored in session (NIST Level 5)');
          }
        } catch (error) {
          console.error('Failed to encrypt wallet:', error);
          throw new Error(`Wallet encryption failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
        onAuthenticate();
      } else {
        throw new Error(response.error || 'Failed to import wallet');
      }
    } catch (error) {
      console.error('Wallet authentication failed:', error);
      setGenerationError(error instanceof Error ? error.message : 'Authentication failed');
      setIsAuthenticating(false);
    }
  };

  const generateQuantumSeed = async () => {
    setIsGenerating(true);
    setGenerationError(null);
    setShowQuantumGenerator(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 800));

      const response = await qnkAPI.generateMnemonic();

      if (response.success && response.data) {
        setSeedPhrase(response.data.mnemonic);
        await new Promise(resolve => setTimeout(resolve, 700));
      } else {
        throw new Error(response.error || 'Failed to generate mnemonic');
      }
    } catch (error) {
      console.error('Quantum seed generation failed:', error);
      setGenerationError(error instanceof Error ? error.message : 'Unknown error');
      setSeedPhrase('abandon ability able about above absent absorb abstract absurd abuse access accident');
    } finally {
      setShowQuantumGenerator(false);
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen px-4 relative overflow-hidden">
      {/* Twinkling Universe Background */}
      <UniverseBackground />

      {/* Subtle radial vignette overlay */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          zIndex: 1,
          background: 'radial-gradient(ellipse at 50% 40%, transparent 0%, rgba(0,0,0,0.4) 70%, rgba(0,0,0,0.7) 100%)',
        }}
      />

      {/* All content sits above the background */}
      <div className="relative" style={{ zIndex: 2 }}>

        {/* Help & Tor Icons - Top Right */}
        <div className="absolute top-4 right-4 flex items-center gap-2 z-50">
          {/* Tor Onion Icon */}
          <motion.button
            className="p-2 bg-purple-600/30 hover:bg-purple-600/50 border border-purple-400/50 rounded-full transition-all cursor-pointer group backdrop-blur-sm"
            whileHover={{ scale: 1.15, rotate: 10 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, scale: 0, rotate: -20 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
            title="Open Tor version of Quillon Graph"
            onClick={openTorSite}
          >
            <svg viewBox="0 0 100 100" className="w-8 h-8" fill="none">
              <ellipse cx="50" cy="55" rx="35" ry="40" fill="#7B4397" opacity="0.9"/>
              <ellipse cx="50" cy="53" rx="28" ry="32" fill="#9B59B6"/>
              <ellipse cx="50" cy="51" rx="21" ry="24" fill="#A569BD"/>
              <ellipse cx="50" cy="49" rx="14" ry="16" fill="#BB8FCE"/>
              <ellipse cx="50" cy="47" rx="7" ry="8" fill="#D7BDE2"/>
              <path d="M50 15 Q52 20 50 25 Q48 30 50 35" stroke="#5D4E37" strokeWidth="4" fill="none" strokeLinecap="round"/>
              <path d="M50 20 Q60 15 58 25 Q55 30 50 25" fill="#27AE60"/>
            </svg>
            <div className="absolute inset-0 rounded-full bg-purple-500/0 group-hover:bg-purple-500/20 transition-all blur-md" />
          </motion.button>

          {/* Help Icon */}
          <motion.button
            className="p-3 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 rounded-full transition-all backdrop-blur-sm"
            onClick={() => setShowInfoModal(true)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
          >
            <HelpCircle className="w-5 h-5 text-amber-400" />
          </motion.button>
        </div>

        {/* Explorer Search Bar at top */}
        <motion.div
          className="pt-6 pb-4"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center justify-center gap-3 mb-2">
            <Search className="w-4 h-4 text-amber-400" />
            <span className="text-amber-200/70 text-sm">Search transactions, blocks, addresses</span>
          </div>
          <ExplorerSearchBar />
        </motion.div>

        <div className="flex-1 flex items-center justify-center">
          <motion.div
            className="w-full max-w-md"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {/* Logo and Title */}
            <div className="text-center mb-12">
              <motion.div
                className="inline-block mb-6"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="w-40 h-40 mx-auto relative">
                  {/* Floating particles around logo */}
                  <FloatingParticles />

                  {/* Cosmic glow effect - enhanced with pulsing rings */}
                  <motion.div
                    className="absolute inset-[-20px] rounded-full"
                    style={{
                      background: 'radial-gradient(circle, rgba(212,175,55,0.15) 0%, rgba(255,165,0,0.08) 40%, transparent 70%)',
                    }}
                    animate={{
                      scale: [1, 1.15, 1],
                      opacity: [0.6, 1, 0.6],
                    }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                  />

                  {/* Outer orbit ring */}
                  <motion.div
                    className="absolute inset-[-10px] rounded-full border border-amber-500/10"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                  >
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-amber-400/60" />
                  </motion.div>

                  {/* Gold border ring */}
                  <div className="absolute inset-0 rounded-full" style={{
                    background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 25%, #FFA500 50%, #FFD700 75%, #D4AF37 100%)',
                    padding: '3px'
                  }}>
                    <div className="w-full h-full bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-full flex items-center justify-center p-4">
                      <img
                        src="/quillon-logo.png"
                        alt="Quillon Graph Logo"
                        className="w-full h-full object-contain"
                        style={{ filter: 'invert(1)' }}
                      />
                    </div>
                  </div>
                </div>
              </motion.div>

              <motion.h1
                className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent"
                style={{ textShadow: '0 0 40px rgba(251,191,36,0.3)' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                Quillon Graph
              </motion.h1>
              <motion.p
                className="text-amber-200/70 mt-2 font-medium"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.6 }}
              >
                Quantum Consensus Wallet
              </motion.p>
            </div>

            {/* Login Form - frosted glass card */}
            <motion.div
              className="relative rounded-3xl p-8 backdrop-blur-xl"
              style={{
                background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.85) 100%)',
                border: '2px solid',
                borderImage: 'linear-gradient(135deg, #D4AF37, #FFD700, #FFA500, #FFD700, #D4AF37) 1',
                boxShadow: '0 0 40px rgba(212, 175, 55, 0.15), inset 0 0 30px rgba(212, 175, 55, 0.05), 0 25px 50px rgba(0,0,0,0.5)'
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.5 }}
            >
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-amber-200 mb-2">
                    BIP39 Quantum Seed Phrase
                  </label>
                  <textarea
                    value={seedPhrase}
                    onChange={(e) => setSeedPhrase(e.target.value)}
                    className="w-full h-24 px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all resize-none backdrop-blur-sm"
                    placeholder="Enter your 12-word seed phrase..."
                  />
                  {/* Word count hint */}
                  {seedPhrase && (
                    <div className={`text-xs mt-1 ${
                      seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length === 12 ||
                      seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length === 24
                        ? 'text-green-400'
                        : 'text-amber-400/60'
                    }`}>
                      {seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length} / 12 words
                      {seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length !== 12 &&
                       seedPhrase.trim().split(/\s+/).filter(w => w.length > 0).length !== 24 &&
                        ' (need 12 or 24)'}
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-amber-200 mb-2">
                    Password (Required for wallet encryption)
                  </label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all backdrop-blur-sm"
                    placeholder="Enter password for wallet encryption..."
                    required
                  />
                </div>

                {/* Quantum Generator Button */}
                <motion.button
                  onClick={generateQuantumSeed}
                  disabled={isGenerating}
                  className="w-full py-4 px-6 bg-gradient-to-r from-amber-900/40 to-yellow-900/40 border-2 border-amber-500/40 rounded-xl text-amber-100 font-medium flex items-center justify-center gap-3 hover:border-amber-400 hover:shadow-[0_0_20px_rgba(251,191,36,0.3)] transition-all disabled:opacity-50 disabled:cursor-not-allowed backdrop-blur-sm"
                  whileHover={{ scale: isGenerating ? 1 : 1.02 }}
                  whileTap={{ scale: isGenerating ? 1 : 0.98 }}
                >
                  {isGenerating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Sparkles className="w-5 h-5 text-amber-400" />
                      </motion.div>
                      <span>Generating BIP39 Mnemonic...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 text-amber-400" />
                      <span>Generate Quantum Entropy</span>
                    </>
                  )}
                </motion.button>

                {/* Error Display */}
                {generationError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-2 backdrop-blur-sm"
                  >
                    <AlertCircle className="w-4 h-4" />
                    <span>Generation failed: {generationError}</span>
                  </motion.div>
                )}

                {/* Quantum Generator Animation */}
                {showQuantumGenerator && (
                  <motion.div
                    className="h-32 rounded-xl overflow-hidden relative border-2 border-amber-500/30"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-amber-500/20 via-yellow-500/20 to-orange-500/20 animate-pulse" />
                    <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center backdrop-blur-sm">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Sparkles className="w-12 h-12 text-amber-400" />
                      </motion.div>
                    </div>
                  </motion.div>
                )}

                {/* Authenticate Button */}
                <motion.button
                  onClick={handleAuthenticate}
                  disabled={!validateSeedPhrase(seedPhrase).valid || !password || isAuthenticating}
                  className="w-full py-5 px-6 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold text-lg flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_30px_rgba(251,191,36,0.5)] hover:from-amber-500 hover:to-yellow-500 transition-all"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {isAuthenticating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Key className="w-6 h-6" />
                      </motion.div>
                      <span>Quantum Authenticating...</span>
                    </>
                  ) : (
                    <>
                      <Key className="w-6 h-6" />
                      <span>Authenticate</span>
                    </>
                  )}
                </motion.button>
              </div>

              {/* Photon Waterfall Effect */}
              {isAuthenticating && (
                <motion.div
                  className="mt-6 h-2 rounded-full overflow-hidden border border-amber-500/30"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-amber-600 via-yellow-500 to-amber-600"
                    initial={{ x: '-100%' }}
                    animate={{ x: '100%' }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </motion.div>
              )}
            </motion.div>

            {/* Security Note */}
            <motion.p
              className="text-center text-amber-300/60 text-sm mt-6 font-medium"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              🔐 Post-Quantum Cryptography &bull; 🌊 DAG-BFT Consensus &bull; 🧅 Tor Integration
            </motion.p>
          </motion.div>
        </div>
      </div>

      {/* Info Modal - What is Quillon Graph? */}
      <AnimatePresence>
        {showInfoModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-start justify-center z-[100] overflow-y-auto py-8"
            onClick={() => setShowInfoModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl p-6 max-w-2xl w-full mx-4 shadow-2xl my-auto"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 60px rgba(212, 175, 55, 0.3)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-500/30 to-yellow-500/30 border border-amber-500/50 flex items-center justify-center">
                    <img src="/quillon-logo.png" alt="Quillon" className="w-8 h-8" style={{ filter: 'invert(1)' }} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-amber-400 to-yellow-500 bg-clip-text text-transparent">
                      What is Quillon Graph?
                    </h2>
                    <p className="text-amber-300/60 text-sm">Next-Generation Quantum-Resistant Blockchain</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowInfoModal(false)}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6 text-amber-400" />
                </button>
              </div>

              {/* Content */}
              <div className="space-y-6 text-amber-100/90">
                <p className="text-lg leading-relaxed">
                  <span className="font-bold text-amber-400">Quillon Graph</span> is a revolutionary Layer 1 blockchain
                  built from the ground up to be <span className="text-amber-300">quantum-resistant</span>,
                  <span className="text-amber-300"> privacy-preserving</span>, and
                  <span className="text-amber-300"> blazingly fast</span>.
                </p>

                {/* Features Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Shield className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">Post-Quantum Security</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Protected by Dilithium5 & Kyber1024 cryptography, designed to withstand attacks from future quantum computers.
                    </p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Lock className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">ZK-STARK Privacy</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Zero-knowledge proofs ensure transaction details remain private while still being verifiable on-chain.
                    </p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Zap className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">DAG-BFT Consensus</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Narwhal-Bullshark DAG consensus achieves 100,000+ TPS with sub-second finality.
                    </p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-xl border border-amber-500/20">
                    <div className="flex items-center gap-3 mb-2">
                      <Globe className="w-5 h-5 text-amber-400" />
                      <h3 className="font-bold text-amber-200">Tor Integration</h3>
                    </div>
                    <p className="text-sm text-amber-100/70">
                      Built-in onion routing provides network-level anonymity and censorship resistance.
                    </p>
                  </div>
                </div>

                {/* Token Info */}
                <div className="p-4 bg-gradient-to-r from-amber-500/10 to-yellow-500/10 rounded-xl border border-amber-500/30">
                  <h3 className="font-bold text-amber-300 mb-2">Native Token: QUG</h3>
                  <p className="text-sm text-amber-100/70">
                    QUG powers the Quillon Graph network - used for transaction fees, staking, governance,
                    and accessing the decentralized exchange (DEX) with privacy-preserving swaps.
                  </p>
                </div>

                {/* Getting Started */}
                <div className="pt-4 border-t border-amber-500/20">
                  <h3 className="font-bold text-amber-200 mb-2">Getting Started</h3>
                  <ol className="text-sm text-amber-100/70 space-y-2 list-decimal list-inside">
                    <li>Click <span className="text-amber-300">"Generate Quantum Entropy"</span> to create a new wallet</li>
                    <li>Securely save your 12-word seed phrase - it's the only way to recover your wallet</li>
                    <li>Set a strong password to encrypt your wallet locally</li>
                    <li>Start mining, trading, or exploring the quantum-resistant blockchain!</li>
                  </ol>
                </div>
              </div>

              {/* Close Button */}
              <div className="mt-6 flex justify-center">
                <motion.button
                  onClick={() => setShowInfoModal(false)}
                  className="px-8 py-3 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold hover:shadow-[0_0_30px_rgba(251,191,36,0.5)] transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Got it!
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
