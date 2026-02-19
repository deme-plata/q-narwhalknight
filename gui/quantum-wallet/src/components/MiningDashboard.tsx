import { useEffect, useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, TrendingUp, Zap, Clock, Award, Sparkles, DollarSign } from 'lucide-react';
import { qnkAPI, type MiningRewardEvent, type BalanceUpdateEvent, type MiningStatsEvent, type WalletMiningStats } from '../services/api';

interface MiningStats {
  totalRewards: number;
  blocksFound: number;
  currentBalance: number;
  avgHashRate: number;
  networkHashRate: number; // v1.1.9-beta: Network-wide hashrate
}

// v3.3.4-beta: Individual miner tracking for hash rate breakdown
interface MinerInfo {
  minerId: string;
  workerName: string | null;
  hashRate: number;
  lastSeen: Date;
  blocksFound: number;
  totalRewards: number;
}

interface RewardWithAnimation extends MiningRewardEvent {
  id: string;
  isNew: boolean;
}

export default function MiningDashboard() {
  const [rewards, setRewards] = useState<RewardWithAnimation[]>([]);
  const [stats, setStats] = useState<MiningStats>({
    totalRewards: 0,
    blocksFound: 0,
    currentBalance: 0,
    avgHashRate: 0,
    networkHashRate: 0,
  });
  const [showRewardPopup, setShowRewardPopup] = useState(false);
  const [latestReward, setLatestReward] = useState<MiningRewardEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // v3.3.4-beta: Track individual miners for hash rate breakdown tooltip
  const [miners, setMiners] = useState<Map<string, MinerInfo>>(new Map());
  const [showMinerTooltip, setShowMinerTooltip] = useState(false);
  const [showNetworkAnimation, setShowNetworkAnimation] = useState(false);
  const [connectedMiners, setConnectedMiners] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const displayHashRateRef = useRef<number>(0);

  // Use the same wallet as Dashboard - from localStorage
  const [walletAddress, setWalletAddress] = useState('');

  // v3.4.21-beta: Track session rewards separately from total balance
  // This prevents jumps caused by mixing local accumulation with backend absolute values
  const [sessionRewardsTotal, setSessionRewardsTotal] = useState(0);
  const initialBalanceRef = useRef<number | null>(null);

  // Daily earnings calculation
  const [qugPriceUsd, setQugPriceUsd] = useState(0);
  const [blockReward, setBlockReward] = useState(0);

  // ═══════════════════════════════════════════════════════════════
  // v7.4.3: Epic Network Power Canvas Animation
  // Renders all connected miners as orbiting particles around a
  // central pulsing core, with energy beams and hash sparks
  // ═══════════════════════════════════════════════════════════════
  const startNetworkCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;
    const cx = W / 2;
    const cy = H / 2;

    const minerCount = Math.max(connectedMiners, 1);
    const totalHashKhs = stats.networkHashRate / 1000;

    // Generate miner particles in concentric orbital rings
    interface MinerParticle {
      angle: number;
      radius: number;
      speed: number;
      size: number;
      hue: number;
      brightness: number;
      ring: number;
      pulsePhase: number;
    }

    const particles: MinerParticle[] = [];
    const rings = Math.min(Math.ceil(minerCount / 30), 8); // Up to 8 orbital rings
    let placed = 0;

    for (let ring = 0; ring < rings && placed < minerCount; ring++) {
      const ringRadius = 30 + ring * (Math.min(W, H) * 0.38 / rings);
      const capacity = Math.min(Math.floor(2 * Math.PI * ringRadius / 6), minerCount - placed);
      const speed = (0.3 + Math.random() * 0.2) / (ring + 1); // outer = slower

      for (let j = 0; j < capacity && placed < minerCount; j++) {
        const angle = (j / capacity) * Math.PI * 2 + ring * 0.5;
        particles.push({
          angle,
          radius: ringRadius + (Math.random() - 0.5) * 8,
          speed: speed * (0.8 + Math.random() * 0.4) * (Math.random() > 0.5 ? 1 : -1),
          size: 1.2 + Math.random() * 1.8,
          hue: 180 + ring * 25 + Math.random() * 20, // cyan → blue → purple gradient
          brightness: 0.5 + Math.random() * 0.5,
          ring,
          pulsePhase: Math.random() * Math.PI * 2,
        });
        placed++;
      }
    }

    // Spark particles (hash operations flying inward)
    interface Spark {
      x: number; y: number;
      vx: number; vy: number;
      life: number; maxLife: number;
      hue: number; size: number;
    }
    const sparks: Spark[] = [];
    let sparkTimer = 0;

    let frame = 0;
    const animate = () => {
      frame++;
      ctx.clearRect(0, 0, W, H);

      // === Background: radial gradient glow ===
      const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(W, H) * 0.5);
      bgGrad.addColorStop(0, 'rgba(0, 240, 255, 0.08)');
      bgGrad.addColorStop(0.4, 'rgba(100, 50, 255, 0.04)');
      bgGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // === Orbital ring guides ===
      for (let ring = 0; ring < rings; ring++) {
        const r = 30 + ring * (Math.min(W, H) * 0.38 / rings);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 200, 255, ${0.06 - ring * 0.005})`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }

      // === Central core: pulsing energy ===
      const pulse = Math.sin(frame * 0.04) * 0.3 + 0.7;
      const coreSize = 18 + pulse * 8;

      // Outer glow
      const coreGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreSize * 2.5);
      coreGlow.addColorStop(0, `rgba(0, 240, 255, ${0.3 * pulse})`);
      coreGlow.addColorStop(0.5, `rgba(100, 50, 255, ${0.15 * pulse})`);
      coreGlow.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = coreGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, coreSize * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Core body
      const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreSize);
      coreGrad.addColorStop(0, '#ffffff');
      coreGrad.addColorStop(0.3, '#00f0ff');
      coreGrad.addColorStop(0.7, '#6432ff');
      coreGrad.addColorStop(1, 'rgba(100, 50, 255, 0)');
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, coreSize, 0, Math.PI * 2);
      ctx.fill();

      // === Energy beams from random miners to core ===
      if (frame % 3 === 0 && particles.length > 0) {
        const beamCount = Math.min(3, Math.floor(minerCount / 30) + 1);
        for (let b = 0; b < beamCount; b++) {
          const p = particles[Math.floor(Math.random() * particles.length)];
          const px = cx + Math.cos(p.angle) * p.radius;
          const py = cy + Math.sin(p.angle) * p.radius;
          const beamGrad = ctx.createLinearGradient(px, py, cx, cy);
          beamGrad.addColorStop(0, `hsla(${p.hue}, 100%, 70%, 0.4)`);
          beamGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(cx, cy);
          ctx.strokeStyle = beamGrad;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      // === Miner particles ===
      for (const p of particles) {
        p.angle += p.speed * 0.01;
        const px = cx + Math.cos(p.angle) * p.radius;
        const py = cy + Math.sin(p.angle) * p.radius;
        const pPulse = Math.sin(frame * 0.06 + p.pulsePhase) * 0.3 + 0.7;
        const sz = p.size * pPulse;

        // Particle glow
        const glow = ctx.createRadialGradient(px, py, 0, px, py, sz * 3);
        glow.addColorStop(0, `hsla(${p.hue}, 100%, 80%, ${0.6 * p.brightness})`);
        glow.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(px, py, sz * 3, 0, Math.PI * 2);
        ctx.fill();

        // Particle core
        ctx.fillStyle = `hsla(${p.hue}, 100%, 90%, ${0.9 * p.brightness})`;
        ctx.beginPath();
        ctx.arc(px, py, sz, 0, Math.PI * 2);
        ctx.fill();
      }

      // === Hash sparks flying inward ===
      sparkTimer++;
      if (sparkTimer % 2 === 0 && sparks.length < 40) {
        const angle = Math.random() * Math.PI * 2;
        const dist = Math.min(W, H) * 0.45;
        const sx = cx + Math.cos(angle) * dist;
        const sy = cy + Math.sin(angle) * dist;
        const speed = 1.5 + Math.random() * 2;
        const dx = cx - sx;
        const dy = cy - sy;
        const len = Math.sqrt(dx * dx + dy * dy);
        sparks.push({
          x: sx, y: sy,
          vx: (dx / len) * speed,
          vy: (dy / len) * speed,
          life: 1, maxLife: 40 + Math.random() * 20,
          hue: 40 + Math.random() * 30, // gold/amber sparks
          size: 1 + Math.random() * 1.5,
        });
      }

      for (let i = sparks.length - 1; i >= 0; i--) {
        const s = sparks[i];
        s.x += s.vx;
        s.y += s.vy;
        s.life++;
        const alpha = 1 - (s.life / s.maxLife);
        if (alpha <= 0) { sparks.splice(i, 1); continue; }

        ctx.fillStyle = `hsla(${s.hue}, 100%, 70%, ${alpha * 0.8})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size * alpha, 0, Math.PI * 2);
        ctx.fill();

        // Trail
        ctx.fillStyle = `hsla(${s.hue}, 100%, 60%, ${alpha * 0.3})`;
        ctx.beginPath();
        ctx.arc(s.x - s.vx, s.y - s.vy, s.size * alpha * 0.6, 0, Math.PI * 2);
        ctx.fill();
      }

      // === Text overlay: miner count + hashrate ===
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Miner count (large)
      ctx.font = 'bold 28px system-ui, -apple-system, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 12;
      ctx.fillText(`${minerCount}`, cx, cy - 16);
      ctx.shadowBlur = 0;

      // "Miners" label
      ctx.font = '11px system-ui, -apple-system, sans-serif';
      ctx.fillStyle = 'rgba(0, 240, 255, 0.9)';
      ctx.fillText('MINERS', cx, cy + 4);

      // Hashrate
      ctx.font = 'bold 13px system-ui, -apple-system, sans-serif';
      ctx.fillStyle = '#ffd700';
      ctx.shadowColor = '#ffd700';
      ctx.shadowBlur = 6;
      const hrText = totalHashKhs >= 1000
        ? `${(totalHashKhs / 1000).toFixed(1)} MH/s`
        : `${totalHashKhs.toFixed(0)} KH/s`;
      ctx.fillText(hrText, cx, cy + 22);
      ctx.shadowBlur = 0;

      // Network share (bottom)
      const yourSharePct = stats.networkHashRate > 0
        ? ((displayHashRateRef.current / stats.networkHashRate) * 100).toFixed(1)
        : '0.0';
      ctx.font = '10px system-ui, -apple-system, sans-serif';
      ctx.fillStyle = 'rgba(200, 160, 255, 0.8)';
      ctx.fillText(`Your share: ${yourSharePct}%`, cx, cy + 40);

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animate();
  }, [connectedMiners, stats.networkHashRate]);

  // v3.4.21-beta: Fetch authoritative balance from API
  const fetchBalance = async () => {
    if (!walletAddress) return;
    try {
      const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
      if (balanceResponse.success && balanceResponse.data) {
        const balance = balanceResponse.data.balance_qnk || 0;
        console.log('💰 [MiningDashboard] Fetched authoritative balance from API:', balance);

        // Store initial balance on first fetch
        if (initialBalanceRef.current === null) {
          initialBalanceRef.current = balance;
          console.log('💰 [MiningDashboard] Set initial balance reference:', balance);
        }

        setStats(prev => ({
          ...prev,
          currentBalance: balance,
          totalRewards: balance, // Total rewards = current balance (authoritative from API)
        }));
      }
    } catch (error) {
      console.error('Failed to fetch balance:', error);
    }
  };

  // v3.5.0-beta: Fetch mining stats from backend (blocks found, hash rate)
  // This allows stats to survive page refresh instead of resetting to 0
  const fetchMiningStats = async () => {
    if (!walletAddress) return;
    try {
      const miningStatsResponse = await qnkAPI.getMiningStats(walletAddress);
      if (miningStatsResponse.success && miningStatsResponse.data) {
        const serverStats = miningStatsResponse.data;
        console.log('⛏️ [MiningDashboard] Fetched mining stats from server:', serverStats);

        setStats(prev => ({
          ...prev,
          blocksFound: serverStats.blocks_found,
          avgHashRate: serverStats.hash_rate, // KH/s from server
        }));

        // v7.4.2: Populate miners map from REST response (preserves names across refresh)
        if (serverStats.workers && serverStats.workers.length > 0) {
          setMiners(prev => {
            const newMiners = new Map(prev);
            for (const worker of serverStats.workers!) {
              const minerId = worker.worker_id || 'unknown';
              const existing = newMiners.get(minerId);
              newMiners.set(minerId, {
                minerId,
                workerName: worker.worker_name || existing?.workerName || null,
                hashRate: worker.hash_rate || existing?.hashRate || 0,
                lastSeen: existing?.lastSeen || new Date(),
                blocksFound: worker.blocks_found || existing?.blocksFound || 0,
                totalRewards: existing?.totalRewards || 0,
              });
            }
            return newMiners;
          });
        }

        // Only log if there are actual mining stats
        if (serverStats.blocks_found > 0 || serverStats.hash_rate > 0) {
          console.log(`⛏️ [MiningDashboard] Restored: ${serverStats.blocks_found} blocks, ${serverStats.hash_rate.toFixed(2)} KH/s, ${serverStats.workers?.length || 0} workers`);
        }
      }
    } catch (error) {
      console.error('Failed to fetch mining stats:', error);
    }
  };

  // Get wallet address from localStorage (same as Dashboard)
  useEffect(() => {
    const storedWallet = localStorage.getItem('walletAddress');
    if (storedWallet) {
      setWalletAddress(storedWallet);
      console.log('✅ Mining Dashboard tracking wallet:', storedWallet);
    } else {
      console.warn('⚠️ No wallet found in localStorage - user needs to login/create wallet first');
    }
  }, []);

  useEffect(() => {
    if (!walletAddress) {
      console.warn('⚠️ No wallet address found for mining dashboard');
      return;
    }

    console.log('🔌 Connecting to SSE for wallet:', walletAddress);

    // v3.4.21-beta: Fetch initial balance from API (single source of truth)
    fetchBalance();

    // v3.5.0-beta: Fetch mining stats (blocks found, hash rate) from backend
    // This restores stats on page refresh instead of starting from 0
    fetchMiningStats();

    // v3.4.21-beta: Refresh balance periodically to stay in sync with backend
    const balanceRefreshInterval = setInterval(fetchBalance, 10000); // Every 10s

    // v3.5.0-beta: Refresh mining stats periodically (every 30s)
    const miningStatsInterval = setInterval(fetchMiningStats, 30000);

    // v1.1.9-beta: Fetch network hashrate on load and periodically
    fetchNetworkHashrate();
    const networkHashrateInterval = setInterval(fetchNetworkHashrate, 30000); // Every 30s

    // Fetch QUG price and block reward for daily earnings calculation
    const fetchEarningsData = async () => {
      try {
        const [priceRes, challengeRes] = await Promise.all([
          fetch('/api/v1/oracle/price/QUG').catch(() => null),
          fetch('/api/v1/mining/challenge').catch(() => null),
        ]);
        if (priceRes?.ok) {
          const json = await priceRes.json();
          const price = json.data?.price_usd || json.data?.price || 0;
          if (price > 0 && price < 1_000_000) setQugPriceUsd(price);
        }
        if (challengeRes?.ok) {
          const json = await challengeRes.json();
          const reward = json.data?.block_reward || 0;
          if (reward > 0) setBlockReward(reward);
        }
      } catch { /* endpoints may not be available */ }
    };
    fetchEarningsData();
    const priceInterval = setInterval(fetchEarningsData, 60000);

    // Subscribe to mining rewards via SSE
    const eventSource = qnkAPI.subscribeToMiningRewards(
      walletAddress,
      handleMiningReward,
      handleBalanceUpdate,
      handleMiningStats
    );

    console.log('✅ SSE EventSource created:', eventSource.url);
    eventSourceRef.current = eventSource;

    // Request notification permissions
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Cleanup on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      clearInterval(networkHashrateInterval);
      clearInterval(balanceRefreshInterval);
      clearInterval(miningStatsInterval);
      clearInterval(priceInterval);
    };
  }, [walletAddress]);

  const handleMiningReward = (reward: MiningRewardEvent) => {
    console.log('🎉 Mining reward received:', reward);
    console.log('🎉 Reward details:', {
      miner: reward.miner_address,
      amount: reward.reward_qnk,
      nonce: reward.nonce,
      hash_rate: reward.hash_rate
    });

    // Add to rewards list with animation flag
    const rewardWithId: RewardWithAnimation = {
      ...reward,
      id: `${reward.block_height}-${reward.nonce}`,
      isNew: true,
    };

    setRewards(prev => {
      const updated = [rewardWithId, ...prev].slice(0, 10); // Keep last 10
      return updated;
    });

    // v3.4.21-beta: Track session rewards (small incremental values only)
    // This is separate from total balance which comes from API
    if (reward.reward_qnk > 0 && reward.reward_qnk < 10) { // Sanity check: individual rewards should be < 10 QUG
      setSessionRewardsTotal(prev => prev + reward.reward_qnk);
    }

    // v3.4.21-beta: Update hash rate and blocks found, but NOT balance
    // Balance is fetched from API periodically to stay authoritative
    setStats(prev => ({
      ...prev,
      blocksFound: prev.blocksFound + 1,
      avgHashRate: reward.hash_rate > 0 ? reward.hash_rate : prev.avgHashRate,
    }));

    // v3.4.21-beta: Trigger a balance refresh from API after receiving a reward
    // This ensures we show the authoritative balance, not a locally accumulated one
    setTimeout(() => fetchBalance(), 500);

    // v3.3.4-beta: Track individual miners for hash rate breakdown
    const minerId = reward.miner_id || reward.miner_address.substring(0, 16);
    setMiners(prev => {
      const newMiners = new Map(prev);
      const existing = newMiners.get(minerId);

      newMiners.set(minerId, {
        minerId,
        workerName: reward.worker_name || null,
        hashRate: reward.hash_rate > 0 ? reward.hash_rate : (existing?.hashRate || 0),
        lastSeen: new Date(),
        blocksFound: (existing?.blocksFound || 0) + 1,
        totalRewards: (existing?.totalRewards || 0) + reward.reward_qnk,
      });

      console.log('⛏️ [MiningDashboard] Updated miner tracking:', {
        minerId,
        workerName: reward.worker_name,
        hashRate: reward.hash_rate,
        totalMiners: newMiners.size
      });

      return newMiners;
    });

    // Show reward popup
    setLatestReward(reward);
    setShowRewardPopup(true);
    setTimeout(() => setShowRewardPopup(false), 5000);

    // Show browser notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Mining Reward Received!', {
        body: `You earned ${reward.reward_qnk.toFixed(8)} QUG from block #${reward.block_height}`,
        icon: '/quillon-logo.png',
        tag: `mining-reward-${reward.block_height}`,
      });
    }

    // Remove animation flag after animation completes
    setTimeout(() => {
      setRewards(prev =>
        prev.map(r => (r.id === rewardWithId.id ? { ...r, isNew: false } : r))
      );
    }, 1000);
  };

  const handleBalanceUpdate = (update: BalanceUpdateEvent) => {
    // v3.4.21-beta: SIMPLIFIED - Just log and trigger API refresh
    // We no longer try to accumulate locally - API is the single source of truth
    console.log('🔔 [MiningDashboard] handleBalanceUpdate - triggering API refresh:', {
      wallet: update.wallet_address,
      old: update.old_balance,
      new: update.new_balance,
      reason: update.change_reason
    });

    // Check if this is a P2P mining reward (needs to be added to rewards list since no MiningReward event)
    const isP2PMiningReward = update.change_reason === 'p2p_mining_reward' ||
                              update.change_reason === 'pending_mining_reward';

    if (isP2PMiningReward) {
      // P2P mining rewards - add to rewards list for display only
      const rewardAmount = update.new_balance - update.old_balance;

      // Sanity check: individual rewards should be small (< 10 QUG)
      if (rewardAmount > 0 && rewardAmount < 10) {
        console.log('⛏️ [MiningDashboard] P2P Mining reward - adding to list:', rewardAmount);

        const rewardWithId: RewardWithAnimation = {
          id: `p2p-${update.timestamp}-${Math.random()}`,
          miner_address: update.wallet_address,
          reward_qnk: rewardAmount,
          nonce: 0,
          block_height: (update as { block_height?: number }).block_height || 0,
          difficulty: '0',
          hash_rate: 0,
          timestamp: update.timestamp,
          isNew: true,
        };

        setRewards(prev => [rewardWithId, ...prev].slice(0, 10));
        setSessionRewardsTotal(prev => prev + rewardAmount);
        setStats(prev => ({ ...prev, blocksFound: prev.blocksFound + 1 }));

        setTimeout(() => {
          setRewards(prev => prev.map(r => r.id === rewardWithId.id ? { ...r, isNew: false } : r));
        }, 1000);
      }
    }

    // v3.4.21-beta: Trigger API refresh to get authoritative balance
    // This is the ONLY place we update the displayed balance
    fetchBalance();
  };

  const handleMiningStats = (statsUpdate: MiningStatsEvent) => {
    // v3.4.21-beta: SIMPLIFIED - Only update hashrate, let API handle balance
    console.log('📊 [MiningDashboard] Mining stats received:', {
      miner: statsUpdate.miner_address,
      hash_rate: statsUpdate.avg_hash_rate,
      miner_id: statsUpdate.miner_id,
      worker_id: statsUpdate.worker_id
    });

    // Only update hashrate - balance comes from API
    setStats(prev => ({
      ...prev,
      avgHashRate: statsUpdate.avg_hash_rate,
    }));

    // Track individual miners for hash rate breakdown
    if (statsUpdate.worker_id || statsUpdate.miner_id) {
      const minerId = statsUpdate.miner_id || statsUpdate.worker_id || 'unknown';
      setMiners(prev => {
        const newMiners = new Map(prev);
        const existing = newMiners.get(minerId);

        newMiners.set(minerId, {
          minerId,
          workerName: statsUpdate.worker_name || existing?.workerName || null,
          hashRate: statsUpdate.avg_hash_rate,
          lastSeen: new Date(),
          blocksFound: existing?.blocksFound || 0,
          totalRewards: existing?.totalRewards || 0,
        });

        console.log('⛏️ [MiningDashboard] Updated miner hashrate:', {
          minerId,
          hashRate: statsUpdate.avg_hash_rate,
          totalMiners: newMiners.size
        });

        return newMiners;
      });
    }
  };

  // v1.1.9-beta: Fetch network-wide hashrate from /api/v1/network/supply
  const fetchNetworkHashrate = async () => {
    try {
      const response = await fetch('/api/v1/network/supply');
      if (response.ok) {
        const json = await response.json();
        // API returns { success: true, data: { network_hashrate: 186539, ... } }
        const networkHashrate = json.data?.network_hashrate || json.network_hashrate || 0;
        const minersCount = json.data?.connected_miners || 0;
        console.log('🌐 Network hashrate fetched:', networkHashrate, 'H/s, miners:', minersCount);
        setStats(prev => ({
          ...prev,
          networkHashRate: networkHashrate,
        }));
        if (minersCount > 0) setConnectedMiners(minersCount);
      }
    } catch (error) {
      console.error('Failed to fetch network hashrate:', error);
    }
  };

  const formatHashRate = (hashRate: number) => {
    if (hashRate >= 1e9) return `${(hashRate / 1e9).toFixed(2)} GH/s`;
    if (hashRate >= 1e6) return `${(hashRate / 1e6).toFixed(2)} MH/s`;
    if (hashRate >= 1e3) return `${(hashRate / 1e3).toFixed(2)} KH/s`;
    return `${hashRate.toFixed(2)} H/s`;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  // v3.3.4-beta: Calculate total hash rate from all tracked miners
  const totalMinerHashRate = Array.from(miners.values()).reduce(
    (sum, miner) => sum + miner.hashRate,
    0
  );
  // v3.5.4-beta: Use maximum of API-reported hashrate and SSE-tracked miners
  // SSE events may have hash_rate=0, but API endpoint calculates from solution timestamps
  const displayHashRate = Math.max(totalMinerHashRate, stats.avgHashRate);
  displayHashRateRef.current = displayHashRate;

  if (!walletAddress) {
    return (
      <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6">
        <p className="text-quantum-yellow">
          No mining wallet configured. Please set VITE_DEFAULT_MINING_WALLET in .env or connect your wallet.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Reward Popup */}
      <AnimatePresence>
        {showRewardPopup && latestReward && (
          <motion.div
            initial={{ opacity: 0, y: -50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -50, scale: 0.9 }}
            className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50 max-w-md w-full"
          >
            <div className="bg-gradient-to-r from-quantum-green/20 to-quantum-cyan/20 backdrop-blur-xl border-2 border-quantum-green rounded-2xl p-6 shadow-2xl">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Sparkles className="w-12 h-12 text-quantum-green animate-pulse" />
                  <div className="absolute inset-0 bg-quantum-green/20 rounded-full animate-ping" />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-quantum-green mb-1">
                    Mining Reward!
                  </h3>
                  <p className="text-white text-lg">
                    +{latestReward.reward_qnk.toFixed(8)} QUG
                  </p>
                  <p className="text-gray-300 text-sm">
                    Block #{latestReward.block_height}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-cyan/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="w-6 h-6 text-quantum-cyan" />
            <span className="text-sm text-gray-400">Balance</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.currentBalance.toFixed(4)}
          </div>
          <div className="text-sm text-quantum-cyan">QUG</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Trophy className="w-6 h-6 text-quantum-green" />
            <span className="text-sm text-gray-400">Total Rewards</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.totalRewards.toFixed(4)}
          </div>
          <div className="text-sm text-quantum-green">QUG</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Award className="w-6 h-6 text-quantum-purple" />
            <span className="text-sm text-gray-400">Blocks Found</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.blocksFound}
          </div>
          <div className="text-sm text-quantum-purple">Blocks</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6 relative cursor-pointer overflow-visible"
          style={{ zIndex: showMinerTooltip ? 100 : 1 }}
          onMouseEnter={() => setShowMinerTooltip(true)}
          onMouseLeave={() => setShowMinerTooltip(false)}
        >
          <div className="flex items-center justify-between mb-3">
            <Zap className="w-6 h-6 text-quantum-yellow" />
            <span className="text-sm text-gray-400">Your Hash Rate</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {formatHashRate(displayHashRate)}
          </div>
          <div className="text-sm text-quantum-yellow flex items-center gap-2">
            Personal
            {miners.size > 0 && (
              <span className="text-xs bg-quantum-yellow/20 px-2 py-0.5 rounded-full">
                {miners.size} miner{miners.size !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {/* v3.3.4-beta: Miner List Tooltip */}
          <AnimatePresence>
            {showMinerTooltip && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="absolute left-0 right-0 top-full mt-2 bg-quantum-dark/95 backdrop-blur-xl border border-quantum-yellow/40 rounded-xl p-4 shadow-2xl"
                style={{ zIndex: 9999 }}
              >
                <div className="text-sm font-semibold text-quantum-yellow mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Active Miners ({miners.size})
                </div>
                {miners.size > 0 ? (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {Array.from(miners.values())
                      .sort((a, b) => b.hashRate - a.hashRate)
                      .map((miner) => (
                        <div
                          key={miner.minerId}
                          className="flex items-center justify-between bg-quantum-indigo/20 rounded-lg p-3 border border-quantum-purple/20"
                        >
                          <div className="flex flex-col">
                            <span className="text-white font-medium">
                              {miner.workerName || `Miner ${miner.minerId.substring(0, 8)}...`}
                            </span>
                            <span className="text-xs text-gray-400">
                              ID: {miner.minerId.substring(0, 12)}...
                            </span>
                          </div>
                          <div className="flex flex-col items-end">
                            <span className="text-quantum-cyan font-bold">
                              {formatHashRate(miner.hashRate)}
                            </span>
                            <span className="text-xs text-gray-400">
                              {miner.blocksFound} block{miner.blocksFound !== 1 ? 's' : ''} | {miner.totalRewards.toFixed(4)} QUG
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm text-center py-4">
                    <div className="mb-2">No miners detected yet</div>
                    <div className="text-xs">Mining rewards will appear here as they are received via SSE</div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Network Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Daily Earnings Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-emerald-500/20 backdrop-blur-xl border border-emerald-500/40 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <DollarSign className="w-6 h-6 text-emerald-400" />
            <span className="text-sm text-gray-400">Est. Daily Earnings</span>
          </div>
          {(() => {
            // v4.3.0: Use daily emission target directly instead of blocksPerDay * blockReward.
            // blockReward is per-solution (0.001288 QUG), NOT per-block. A miner submits
            // many solutions per block, so blocksPerDay * blockReward massively underestimates.
            // Correct formula: yourShare * dailyTarget * (1 - devFee)
            const ERA_0_DAILY_QUG = 224.7465; // Austrian economics: Era 0 daily emission target
            const DEV_FEE = 0.01; // 1% dev fee
            const yourShare = stats.networkHashRate > 0 ? displayHashRate / stats.networkHashRate : 0;
            const dailyQug = yourShare * ERA_0_DAILY_QUG * (1 - DEV_FEE);
            const dailyUsd = dailyQug * qugPriceUsd;
            return (
              <>
                <div className="text-3xl font-bold text-white mb-1">
                  {dailyUsd > 0 ? `$${dailyUsd.toFixed(2)}` : '$0.00'}
                </div>
                <div className="text-sm text-emerald-400">{dailyQug.toFixed(4)} QUG/day</div>
                {qugPriceUsd > 0 && (
                  <div className="text-xs text-gray-500 mt-1">@ ${qugPriceUsd.toFixed(2)}/QUG</div>
                )}
                {yourShare > 0 && (
                  <div className="text-xs text-gray-500">Network share: {(yourShare * 100).toFixed(2)}%</div>
                )}
              </>
            );
          })()}
        </motion.div>

        {/* v7.4.3: Epic Network Power Visualization — Canvas-based miner galaxy */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-quantum-purple/20 backdrop-blur-xl border border-quantum-cyan/40 rounded-xl relative cursor-pointer overflow-hidden"
          style={{ zIndex: showNetworkAnimation ? 100 : 1, minHeight: showNetworkAnimation ? 340 : 'auto' }}
          onMouseEnter={() => {
            setShowNetworkAnimation(true);
            // Start canvas animation on next tick
            setTimeout(() => startNetworkCanvas(), 50);
          }}
          onMouseLeave={() => {
            setShowNetworkAnimation(false);
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
          }}
        >
          {/* Default compact view */}
          {!showNetworkAnimation && (
            <div className="p-6">
              <div className="flex items-center justify-between mb-3">
                <TrendingUp className="w-6 h-6 text-quantum-cyan" />
                <span className="text-sm text-gray-400">Network Hash Rate</span>
              </div>
              <div className="text-3xl font-bold text-white mb-1">
                {formatHashRate(stats.networkHashRate)}
              </div>
              <div className="text-sm text-quantum-cyan flex items-center gap-2">
                Total Network Power
                {connectedMiners > 0 && (
                  <span className="text-xs bg-quantum-cyan/20 px-2 py-0.5 rounded-full">
                    {connectedMiners} miners
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Expanded canvas visualization */}
          <AnimatePresence>
            {showNetworkAnimation && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="w-full"
                style={{ height: 340 }}
              >
                <canvas
                  ref={canvasRef}
                  className="w-full h-full rounded-xl"
                  style={{ background: 'rgba(5, 5, 20, 0.9)' }}
                />
                {/* Bottom label */}
                <div className="absolute bottom-2 left-0 right-0 text-center">
                  <span className="text-[10px] text-gray-500">
                    Each particle = 1 miner contributing hash power
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-quantum-green/20 backdrop-blur-xl border border-quantum-green/40 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Sparkles className="w-6 h-6 text-quantum-green" />
            <span className="text-sm text-gray-400">Your Share</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.networkHashRate > 0
              ? ((stats.avgHashRate / stats.networkHashRate) * 100).toFixed(2)
              : '0.00'}%
          </div>
          <div className="text-sm text-quantum-green">of Network Power</div>
        </motion.div>
      </div>

      {/* Recent Rewards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
      >
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-quantum-cyan" />
          Recent Mining Rewards
        </h3>

        {rewards.length === 0 ? (
          <div className="text-center py-8">
            <Zap className="w-12 h-12 text-gray-500 mx-auto mb-3 opacity-50" />
            <p className="text-gray-400">
              Waiting for mining rewards...
            </p>
            <p className="text-gray-500 text-sm mt-1">
              Rewards will appear here in real-time via SSE
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {rewards.map((reward) => (
                <motion.div
                  key={reward.id}
                  initial={reward.isNew ? { opacity: 0, x: -20, scale: 0.95 } : false}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: 20, scale: 0.95 }}
                  className={`bg-quantum-dark/50 rounded-lg p-4 border transition-all ${
                    reward.isNew
                      ? 'border-quantum-green shadow-lg shadow-quantum-green/20'
                      : 'border-quantum-purple/20'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Trophy className="w-4 h-4 text-quantum-green" />
                        <span className="text-quantum-green font-bold">
                          +{reward.reward_qnk.toFixed(8)} QUG
                        </span>
                        {reward.isNew && (
                          <span className="text-xs bg-quantum-green/20 text-quantum-green px-2 py-0.5 rounded-full animate-pulse">
                            NEW
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-300">
                        Block #{reward.block_height} • Nonce: {reward.nonce}
                      </div>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {reward.origin_node_name && (
                          <span className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-0.5 rounded">
                            {reward.origin_node_name}
                          </span>
                        )}
                        {reward.worker_name && (
                          <span className="text-xs bg-quantum-cyan/20 text-quantum-cyan px-2 py-0.5 rounded">
                            Miner: {reward.worker_name}
                          </span>
                        )}
                        {reward.miner_id && !reward.worker_name && (
                          <span className="text-xs bg-quantum-yellow/20 text-quantum-yellow px-2 py-0.5 rounded">
                            ID: {reward.miner_id.substring(0, 8)}...
                          </span>
                        )}
                        {reward.miner_id && reward.worker_name && (
                          <span className="text-xs bg-quantum-yellow/20 text-quantum-yellow px-2 py-0.5 rounded opacity-70">
                            [{reward.miner_id.substring(0, 8)}]
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Difficulty: {reward.difficulty}
                        {reward.origin_node_id && (
                          <span className="ml-2 text-gray-600">
                            • Node: {reward.origin_node_id.substring(0, 12)}...
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-400">
                        {formatTime(reward.timestamp)}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Smart Accumulation Tip */}
        {rewards.length >= 2 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-lg p-4"
          >
            <div className="flex items-center gap-2 text-quantum-cyan text-sm">
              <TrendingUp className="w-4 h-4" />
              <span className="font-semibold">Accumulation Rate</span>
            </div>
            <div className="text-gray-300 text-sm mt-2">
              {(() => {
                // Calculate rewards per hour based on recent activity
                const recentRewards = rewards.slice(0, Math.min(5, rewards.length));
                const totalAmount = recentRewards.reduce((sum, r) => sum + r.reward_qnk, 0);
                const firstTime = new Date(recentRewards[recentRewards.length - 1]?.timestamp || Date.now()).getTime();
                const lastTime = new Date(recentRewards[0]?.timestamp || Date.now()).getTime();
                const timeDiffHours = Math.max((lastTime - firstTime) / (1000 * 60 * 60), 0.01);
                const ratePerHour = totalAmount / timeDiffHours;
                const ratePerDay = ratePerHour * 24;

                if (ratePerHour > 0.001) {
                  return (
                    <>
                      At your current hashrate, you're earning approximately{' '}
                      <span className="text-quantum-green font-bold">{ratePerHour.toFixed(4)} QUG/hour</span>
                      {' '}({ratePerDay.toFixed(2)} QUG/day)
                    </>
                  );
                }
                return 'Keep mining to calculate your accumulation rate!';
              })()}
            </div>
            <div className="text-gray-500 text-xs mt-2">
              All mining rewards are credited instantly across the network via P2P propagation
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Download Miner */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-green/10 border border-quantum-green/30 rounded-xl p-6"
      >
        <h4 className="text-lg font-bold text-quantum-green mb-3 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Download Optimized Miner v3.3.3
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          v3.3.3: Miner identification + Lock-free multi-threading + P2P propagation
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <a
            href="/downloads/q-miner-v3.3.3-beta"
            download
            className="flex items-center justify-center gap-2 bg-quantum-green/20 hover:bg-quantum-green/30 border border-quantum-green/50 text-quantum-green font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            Linux x64 (Latest)
          </a>
          <a
            href="/downloads/q-miner-windows-x64.exe"
            download
            className="flex items-center justify-center gap-2 bg-quantum-cyan/20 hover:bg-quantum-cyan/30 border border-quantum-cyan/50 text-quantum-cyan font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            Windows x64
          </a>
          <a
            href="/downloads/q-miner-macos-arm64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/50 text-quantum-purple font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            macOS ARM64
          </a>
          <a
            href="/downloads/q-miner-macos-x64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-yellow/20 hover:bg-quantum-yellow/30 border border-quantum-yellow/50 text-quantum-yellow font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            macOS Intel x64
          </a>
        </div>
        <div className="mt-4 p-4 bg-black/30 rounded-lg space-y-3">
          <div>
            <p className="text-xs text-gray-400 mb-1">Connect to Network (with miner name):</p>
            <code className="text-xs text-quantum-cyan block">
              ./q-miner --wallet {walletAddress.slice(0, 20)}... --server http://quillon.xyz:8080 --miner-name "My Rig"
            </code>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Solo Mining (Local Node):</p>
            <code className="text-xs text-quantum-yellow block">
              ./q-miner --wallet {walletAddress.slice(0, 20)}... --server http://localhost:8080 --miner-name "Local"
            </code>
            <p className="text-xs text-gray-500 mt-1">Start node with: Q_ALLOW_SOLO_MINING=true ./q-api-server</p>
          </div>
        </div>
      </motion.div>

      {/* Mining Tips */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl p-6"
      >
        <h4 className="text-lg font-bold text-quantum-purple mb-3">Mining Tips</h4>
        <ul className="space-y-2 text-gray-300 text-sm">
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Real-time rewards appear instantly via Server-Sent Events (SSE)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Dashboard updates automatically when your miner finds a block</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Browser notifications alert you to new rewards (enable in settings)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Hash rate is calculated from your actual mining performance</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>New optimized miner: 2.5x faster with AVX2 SIMD + CPU core pinning</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-yellow mt-0.5">⚡</span>
            <span>Solo mining: Balance updates in &lt;100ms with AEGIS-256 authenticated rewards</span>
          </li>
        </ul>
      </motion.div>
    </div>
  );
}
