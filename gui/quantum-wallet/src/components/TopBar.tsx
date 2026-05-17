import { useState, useEffect, useRef, memo, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Copy, Check, ExternalLink, Hash, User, Blocks, Shield, X, Clock, ArrowRight, CheckCircle, XCircle, Key, FileCode, Wifi, Zap, Globe, MessageCircle, Bell, Send, UserCircle, CreditCard, LogOut, BookOpen, Palette, Pickaxe, Settings, FileText, Code, Twitter, Facebook, Download, ChevronDown, Trophy } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';
import { qnkAPI } from '../services/api';
import type { MiningStatsEvent } from '../services/api';
import SmartContractModal from './SmartContractModal';
import NetworkMapModal from './NetworkMapModal';
import ThemeChooserModal from './ThemeChooserModal';
import MinerLinkModal from './MinerLinkModal';
import PapersLibraryModal from './PapersLibraryModal';
import { useMinerLink } from '../hooks/useMinerLink';
import { sseManager } from '../services/sseManager';

// v3.6.1-beta: SANITY CHECK - Max possible balance is 21 million QUG (total supply)
// Any balance exceeding this is corrupted data and must be rejected
const MAX_SANE_BALANCE = 21_000_000; // 21 million QUG

/**
 * v3.6.1-beta: Validate balance value to prevent corrupted data from displaying
 * Returns true if the balance is sane, false if it's corrupted
 */
function isValidBalance(balance: number): boolean {
  if (typeof balance !== 'number') return false;
  if (isNaN(balance) || !isFinite(balance)) return false;
  if (balance < 0) return false;
  if (balance > MAX_SANE_BALANCE) {
    console.warn(`🚨 [TopBar] Rejected corrupted balance: ${balance.toExponential()} > max supply ${MAX_SANE_BALANCE}`);
    return false;
  }
  return true;
}

/**
 * v3.6.1-beta: Clear corrupted localStorage balance values
 */
function clearCorruptedBalanceCache(): void {
  const cached = localStorage.getItem('cachedBalance');
  if (cached) {
    const value = parseFloat(cached);
    if (!isValidBalance(value)) {
      console.warn(`🚨 [TopBar] Clearing corrupted localStorage cachedBalance: ${value}`);
      localStorage.removeItem('cachedBalance');
    }
  }
  const locked = localStorage.getItem('dexLockedBalance');
  if (locked) {
    const value = parseFloat(locked);
    if (!isValidBalance(value)) {
      console.warn(`🚨 [TopBar] Clearing corrupted localStorage dexLockedBalance: ${value}`);
      localStorage.removeItem('dexLockedBalance');
    }
  }
}

interface SearchResult {
  type: 'transaction' | 'block' | 'address' | 'node' | 'contract' | 'error';
  id: string;
  title: string;
  subtitle?: string;
  hash?: string;
  data?: any; // Full data for detail view
}

// v3.9.1-beta: Bank Messaging System Types
interface BankMessage {
  id: string;
  from: 'user' | 'bank';
  content: string;
  timestamp: number;
  read: boolean;
  subject?: string;
  loanId?: string;
}

interface LoanDetails {
  id: string;
  amount: number;
  collateral: number;
  interestRate: number;
  status: 'pending' | 'approved' | 'rejected' | 'active' | 'paid' | 'liquidated';
  createdAt: number;
  dueDate?: number;
  remainingBalance?: number;
}

interface TopBarProps {
  currentBalance: number;
  nodeId: string;
  blockHeight: number;
  peers: number;
  isOnline: boolean;
  qci: number; // Quantum Coherence Index
  onNavigate?: (screen: 'dashboard' | 'transactions' | 'explorer' | 'dex' | 'mining' | 'vm' | 'download' | 'aichat' | 'settings') => void;
}

/**
 * Network Health Gauge — compact canvas animation that mirrors the KOrbitalViz
 * from DeployControlPanel. Orbiting factor particles (G Q T I R), phase-boundary
 * dashed arcs at K=5 and K=10, background radial glow, center K value.
 */
function NHGClock({ kValue, kPhase }: { kValue: number; kPhase: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const kRef = useRef(kValue);
  const phaseRef = useRef(kPhase);
  kRef.current = kValue;
  phaseRef.current = kPhase;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maybeCtx = canvas.getContext('2d');
    if (!maybeCtx) return;
    const ctx = maybeCtx;

    const SIZE = 72; // larger so particles + letters are legible
    const dpr = window.devicePixelRatio || 1;
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    canvas.style.width = `${SIZE}px`;
    canvas.style.height = `${SIZE}px`;
    ctx.scale(dpr, dpr);

    const cx = SIZE / 2;
    const cy = SIZE / 2;
    const R = 28; // orbit radius (matches KOrbitalViz feel)
    const stroke = 2;

    // Factor particles — same labels as KOrbitalViz
    const factors = [
      { letter: 'G', color: '#10b981', speed: 0.8,  offset: 0 },
      { letter: 'Q', color: '#06b6d4', speed: 1.1,  offset: Math.PI * 0.4 },
      { letter: 'T', color: '#f97316', speed: 0.6,  offset: Math.PI * 0.8 },
      { letter: 'I', color: '#3b82f6', speed: 0.9,  offset: Math.PI * 1.2 },
      { letter: 'R', color: '#a855f7', speed: 0.7,  offset: Math.PI * 1.6 },
    ];

    const phaseColor = (p: string) =>
      p === 'critical' ? '#ef4444' : p === 'approaching' ? '#f59e0b' : '#10b981';
    const phaseGlowRgba = (p: string) =>
      p === 'critical' ? 'rgba(239,68,68,' : p === 'approaching' ? 'rgba(245,158,11,' : 'rgba(16,185,129,';

    // Trail sparks
    const trails: Array<{ x: number; y: number; vx: number; vy: number; life: number; color: string }> = [];

    let t = 0;
    function draw() {
      t += 0.016;
      const k = kRef.current;
      const p = phaseRef.current || (k < 5 ? 'stable' : k < 10 ? 'approaching' : 'critical');
      const pColor = phaseColor(p);
      const glowRgba = phaseGlowRgba(p);
      const kNorm = Math.min(k / 15, 1);

      ctx.clearRect(0, 0, SIZE, SIZE);

      // Background radial glow (phase-colored)
      const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R + 6);
      bg.addColorStop(0, `${glowRgba}0.15)`);
      bg.addColorStop(1, `${glowRgba}0)`);
      ctx.fillStyle = bg;
      ctx.beginPath();
      ctx.arc(cx, cy, R + 8, 0, Math.PI * 2);
      ctx.fill();

      // Outer track (faint ring)
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      ctx.lineWidth = stroke;
      ctx.stroke();

      // Phase boundary dashed arcs at K=5 (33%) and K=10 (67%)
      [0.333, 0.667].forEach(frac => {
        const angle = frac * 2 * Math.PI - Math.PI / 2;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(R - 4, 0);
        ctx.lineTo(R + 4, 0);
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      });

      // Health arc: low k = full arc (healthy). starts at top (-π/2), sweeps CW.
      const arcSweep = (1 - kNorm) * 2 * Math.PI;
      if (arcSweep > 0.02) {
        ctx.beginPath();
        ctx.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + arcSweep);
        ctx.strokeStyle = pColor;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.shadowColor = pColor;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.lineCap = 'butt';
      }

      // Update + draw trail sparks
      for (let i = trails.length - 1; i >= 0; i--) {
        const s = trails[i];
        s.x += s.vx; s.y += s.vy; s.life -= 0.03;
        if (s.life <= 0) { trails.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(s.x, s.y, 1.5 * s.life, 0, Math.PI * 2);
        ctx.fillStyle = s.color.replace(')', `,${s.life})`).replace('rgb', 'rgba');
        ctx.fill();
      }

      // Orbiting factor particles (letters)
      factors.forEach(f => {
        const angle = t * f.speed + f.offset;
        const px = cx + Math.cos(angle) * R;
        const py = cy + Math.sin(angle) * R;

        // Emit trail sparks occasionally
        if (Math.random() < 0.12) {
          trails.push({
            x: px, y: py,
            vx: (Math.random() - 0.5) * 1.2,
            vy: (Math.random() - 0.5) * 1.2,
            life: 0.8 + Math.random() * 0.4,
            color: f.color,
          });
        }

        // Glow behind particle
        ctx.shadowColor = f.color;
        ctx.shadowBlur = 6;

        // Draw particle dot
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = f.color;
        ctx.fill();

        // Letter label
        ctx.shadowBlur = 0;
        ctx.font = 'bold 6px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(f.letter, px, py);
      });

      // Center orb glow
      const centerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 10);
      centerGrad.addColorStop(0, `${glowRgba}0.4)`);
      centerGrad.addColorStop(1, `${glowRgba}0)`);
      ctx.fillStyle = centerGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fill();

      // Center K value + label (show "---" when no data yet)
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = k === 0 ? 'rgba(255,255,255,0.3)' : pColor;
      ctx.font = k === 0 ? '8px monospace' : 'bold 10px monospace';
      ctx.shadowColor = k === 0 ? 'transparent' : pColor;
      ctx.shadowBlur = k === 0 ? 0 : 4;
      ctx.fillText(k === 0 ? '---' : k.toFixed(2), cx, cy - 3);
      ctx.shadowBlur = 0;
      ctx.font = '6px monospace';
      ctx.fillStyle = k === 0 ? 'rgba(255,255,255,0.2)' : `${glowRgba}0.8)`;
      ctx.fillText('K', cx, cy + 8);

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  const phaseLabel = kPhase === 'critical' ? 'Critical' : kPhase === 'approaching' ? 'Warning' : 'Healthy';
  return (
    <canvas
      ref={canvasRef}
      title={`Network Health Gauge — K=${kValue.toFixed(4)} (${phaseLabel})`}
      style={{ cursor: 'default' }}
    />
  );
}

// v2.4.0: Memoized for performance
const TopBar = memo(function TopBar({ currentBalance, nodeId, blockHeight, peers, isOnline, qci, onNavigate }: TopBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<SearchResult | null>(null); // v3.4.2: Detail modal
  const [selectedContract, setSelectedContract] = useState<any | null>(null); // v3.4.20: Enhanced contract modal
  const [showNetworkMap, setShowNetworkMap] = useState(false); // v3.4.21: Network map modal
  const [showThemeChooser, setShowThemeChooser] = useState(false); // v5.7.0: Theme chooser modal

  // v3.9.1-beta: Profile & Banking Messaging System
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [hasActiveLoan, setHasActiveLoan] = useState(false);
  const [unreadMessages, setUnreadMessages] = useState(0);
  const [bankMessages, setBankMessages] = useState<BankMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [loanDetails, setLoanDetails] = useState<LoanDetails | null>(null);

  // v3.9.2-beta: Universal profile dropdown
  const [copiedWallet, setCopiedWallet] = useState(false);
  const [recentInboxItems, setRecentInboxItems] = useState<any[]>([]);
  const walletAddr = useMemo(() => localStorage.getItem('walletAddress') || '', []);
  const [showMinerLinkModal, setShowMinerLinkModal] = useState(false);
  const [showPapersLibrary, setShowPapersLibrary] = useState(false);
  const [showTaxModal, setShowTaxModal] = useState(false);
  const minerLink = useMinerLink(walletAddr || null);

  // MetaMask linked account data
  const [metamaskAddress, setMetamaskAddress] = useState<string | null>(null);
  const [metamaskBalance, setMetamaskBalance] = useState<string | null>(null);
  const [metamaskChainId, setMetamaskChainId] = useState<string | null>(null);
  const [metamaskChainName, setMetamaskChainName] = useState<string>('');

  // v7.3.0: Node admin check via API (--admin-wallet)
  const [isNodeAdmin, setIsNodeAdmin] = useState(false);

  // v8.5.10: Bounty score from bounty API
  const [bountyScore, setBountyScore] = useState<number>(0);

  // v3.4.16-beta: SSE-updated live metrics
  const [liveBlockHeight, setLiveBlockHeight] = useState(blockHeight);
  const [livePeers, setLivePeers] = useState(peers);
  const [personalHashrate, setPersonalHashrate] = useState<number>(0);

  // Network power + miners (polled from /api/v1/network/supply)
  const [networkHashrate, setNetworkHashrate] = useState<number>(0);
  const [networkMiners, setNetworkMiners] = useState<number>(0);

  // Network Health Gauge — k-parameter from /api/v1/k-parameter
  const [kValue, setKValue] = useState<number>(0);
  const [kPhase, setKPhase] = useState<string>('stable');
  const [kData, setKData] = useState<Record<string, any> | null>(null);
  const [showKTooltip, setShowKTooltip] = useState(false);
  const nhgRef = useRef<HTMLDivElement>(null);
  const nhgHideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const showNHGTooltip = () => {
    if (nhgHideTimerRef.current) clearTimeout(nhgHideTimerRef.current);
    setShowKTooltip(true);
  };
  const hideNHGTooltip = () => {
    nhgHideTimerRef.current = setTimeout(() => setShowKTooltip(false), 150);
  };
  const [isTorConnected, setIsTorConnected] = useState(false);
  const [torOnionUrl, setTorOnionUrl] = useState("http://ca3jpub2haxboxjw4ws6run36ekdh3pv7pneqg2tbac5rxzvxhd2i5id.onion");
  const [minerLinkCount, setMinerLinkCount] = useState(0);
  // v8.6.2: sseRef removed — TopBar now uses shared sseManager instead of its own EventSource

  // v3.6.1-beta: Clear corrupted balance caches on mount
  useEffect(() => {
    clearCorruptedBalanceCache();
  }, []);

  // v7.3.0: Check if current wallet is the node's admin wallet
  useEffect(() => {
    if (!walletAddr) return;
    fetch('/api/v1/admin/is-admin', {
      headers: { 'X-Wallet-Auth': walletAddr, 'Authorization': `Bearer ${walletAddr}` },
    })
      .then(r => r.json())
      .then(data => setIsNodeAdmin(data.is_admin === true))
      .catch(() => setIsNodeAdmin(false));
  }, [walletAddr]);

  // v8.5.10: Fetch user's bounty score from bounty API
  useEffect(() => {
    if (!walletAddr) return;
    fetch(`/bounty-api/score/${walletAddr}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.total_score !== undefined) setBountyScore(Math.round(data.total_score));
        else if (data?.score !== undefined) setBountyScore(Math.round(data.score));
      })
      .catch(() => {}); // silently fail — bounty server may be down
  }, [walletAddr]);

  // MetaMask: Fetch linked account data on mount
  useEffect(() => {
    const linked = localStorage.getItem('metamaskLinked');
    if (!linked) return;
    setMetamaskAddress(linked);

    const fetchMetamaskData = async () => {
      try {
        const ethereum = (window as any).ethereum;
        // Find MetaMask provider (handles multiple wallet extensions)
        const provider = ethereum?.providers?.length
          ? ethereum.providers.find((p: any) => p.isMetaMask)
          : ethereum?.isMetaMask ? ethereum : null;
        if (!provider) return;

        // Get chain info
        const chainId: string = await provider.request({ method: 'eth_chainId' });
        setMetamaskChainId(chainId);
        const chainNames: Record<string, string> = {
          '0x1': 'Ethereum', '0x89': 'Polygon', '0xa86a': 'Avalanche',
          '0xa4b1': 'Arbitrum', '0xa': 'Optimism', '0x38': 'BSC',
          '0x2105': 'Base', '0xaa36a7': 'Sepolia',
        };
        setMetamaskChainName(chainNames[chainId] || `Chain ${parseInt(chainId, 16)}`);

        // Get balance
        const balHex: string = await provider.request({
          method: 'eth_getBalance',
          params: [linked, 'latest'],
        });
        const balWei = BigInt(balHex);
        const ethBal = Number(balWei) / 1e18;
        setMetamaskBalance(ethBal.toFixed(4));
      } catch (err) {
        console.warn('MetaMask data fetch failed:', err);
      }
    };

    fetchMetamaskData();
    // Refresh every 30s
    const interval = setInterval(fetchMetamaskData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Link MetaMask from profile dropdown (for users who logged in with seed phrase)
  const handleLinkMetaMask = async () => {
    try {
      const ethereum = (window as any).ethereum;
      const provider = ethereum?.providers?.length
        ? ethereum.providers.find((p: any) => p.isMetaMask)
        : ethereum?.isMetaMask ? ethereum : null;
      if (!provider) {
        alert('MetaMask not detected. Please install the MetaMask browser extension.');
        return;
      }
      const accounts: string[] = await provider.request({ method: 'eth_requestAccounts' });
      if (!accounts || accounts.length === 0) return;
      const ethAddress = accounts[0].toLowerCase();
      localStorage.setItem('metamaskLinked', ethAddress);
      setMetamaskAddress(ethAddress);

      // Fetch chain + balance immediately
      try {
        const chainId: string = await provider.request({ method: 'eth_chainId' });
        setMetamaskChainId(chainId);
        const chainNames: Record<string, string> = {
          '0x1': 'Ethereum', '0x89': 'Polygon', '0xa86a': 'Avalanche',
          '0xa4b1': 'Arbitrum', '0xa': 'Optimism', '0x38': 'BSC',
          '0x2105': 'Base', '0xaa36a7': 'Sepolia',
        };
        setMetamaskChainName(chainNames[chainId] || `Chain ${parseInt(chainId, 16)}`);
        const balHex: string = await provider.request({
          method: 'eth_getBalance', params: [ethAddress, 'latest'],
        });
        setMetamaskBalance((Number(BigInt(balHex)) / 1e18).toFixed(4));
      } catch { /* chain/balance fetch is best-effort */ }
    } catch (err: any) {
      if (err?.code === 4001) return; // user rejected
      console.warn('MetaMask link failed:', err);
    }
  };

  // v2.9.0-beta: STABLE balance display - prevent bouncing between multiple sources
  // v3.6.1-beta: Add sanity check to reject corrupted values
  const [stableBalance, setStableBalance] = useState<number>(() => {
    // Initialize from localStorage cache to prevent flash
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : 0;
    // v3.6.1-beta: Validate cached value before using
    if (isValidBalance(cachedValue)) {
      return cachedValue;
    }
    // If cached is corrupted, try currentBalance
    if (isValidBalance(currentBalance)) {
      return currentBalance;
    }
    // All sources corrupted - start at 0
    return 0;
  });
  const lastBalanceUpdateRef = useRef<number>(Date.now());
  const balanceStabilityWindowMs = 2000; // Don't change balance more than once per 2 seconds

  // v2.9.0-beta: Stabilized balance getter - prevents rapid flickering
  // v3.6.1-beta: Added sanity checks to reject corrupted values
  const getDisplayBalance = (): number => {
    // Check if we have a locked balance from DEX (SOURCE OF TRUTH during cooldown)
    const lockedBalance = localStorage.getItem('dexLockedBalance');
    const cooldownUntil = parseInt(localStorage.getItem('dexCooldownUntil') || '0');

    if (lockedBalance && Date.now() < cooldownUntil) {
      const locked = parseFloat(lockedBalance);
      // v3.6.1-beta: Validate before returning
      if (isValidBalance(locked)) {
        return locked;
      } else {
        console.warn(`🚨 [TopBar] getDisplayBalance: Rejected corrupted dexLockedBalance: ${locked}`);
        // Clear corrupted value
        localStorage.removeItem('dexLockedBalance');
      }
    }

    // Return the stable balance if valid, otherwise 0
    if (isValidBalance(stableBalance)) {
      return stableBalance;
    }

    console.warn(`🚨 [TopBar] getDisplayBalance: stableBalance is corrupted: ${stableBalance}`);
    return 0;
  };

  // v2.9.0-beta: Update stable balance with debouncing to prevent flickering
  // v3.6.1-beta: Added sanity checks to reject corrupted values
  // v6.0.3: Removed stableBalance from deps to prevent potential infinite re-render loop (React Error #185)
  const stableBalanceRef = useRef(stableBalance);
  stableBalanceRef.current = stableBalance;

  useEffect(() => {
    const cached = localStorage.getItem('cachedBalance');
    const cachedValue = cached ? parseFloat(cached) : currentBalance;

    // v3.6.1-beta: Only use values that pass sanity check
    const validCached = isValidBalance(cachedValue) ? cachedValue : 0;
    const validCurrent = isValidBalance(currentBalance) ? currentBalance : 0;
    const currentStable = stableBalanceRef.current;
    const validStable = isValidBalance(currentStable) ? currentStable : 0;

    const newBalance = validCached || validCurrent;

    // Only update if enough time has passed (prevents rapid flickering)
    const timeSinceLastUpdate = Date.now() - lastBalanceUpdateRef.current;
    const balanceDifference = Math.abs(newBalance - validStable);

    // Update if: significant change (>1 QUG) OR stability window passed
    if (balanceDifference > 1 || timeSinceLastUpdate > balanceStabilityWindowMs) {
      // v3.6.1-beta: Use Math.max ONLY on validated values to prevent corrupted values from persisting
      // Filter out any corrupted values before comparing
      const candidates = [newBalance, validStable, validCurrent].filter(v => isValidBalance(v));
      const bestBalance = candidates.length > 0 ? Math.max(...candidates) : 0;

      if (Math.abs(bestBalance - currentStable) > 0.0001) {
        console.log('💰 TopBar: Stable balance update:', currentStable.toFixed(4), '→', bestBalance.toFixed(4));
        setStableBalance(bestBalance);
        lastBalanceUpdateRef.current = Date.now();
      }
    }
  }, [currentBalance]);

  // v2.9.0-beta: Listen for balance change events and update stable balance
  // v3.6.1-beta: Added sanity checks to reject corrupted values
  useEffect(() => {
    const handleBalanceChanged = (event: Event) => {
      const customEvent = event as CustomEvent;
      const newBalance = customEvent.detail?.balance;

      // v3.6.1-beta: CRITICAL - Validate balance before using
      if (!isValidBalance(newBalance)) {
        console.warn(`🚨 [TopBar] Rejecting invalid qug-balance-changed: ${newBalance}`);
        return;
      }

      console.log('🔥 TopBar: qug-balance-changed received:', newBalance);
      // v2.9.3-beta: DEX swaps are AUTHORITATIVE - allow both increases AND decreases
      // The qug-balance-changed event is ONLY dispatched by DexScreen after successful swaps
      // so we MUST trust the value even if it's lower (e.g., QUG -> QUGUSD swap)
      setStableBalance(newBalance);
      lastBalanceUpdateRef.current = Date.now();
    };

    const handleDexCooldownExpired = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { qugBalance } = customEvent.detail || {};
      console.log('🔄 TopBar: DEX cooldown expired, balance:', qugBalance);

      // v3.6.1-beta: CRITICAL - Validate balance before using
      if (!isValidBalance(qugBalance)) {
        console.warn(`🚨 [TopBar] Rejecting invalid dex-cooldown-expired balance: ${qugBalance}`);
        return;
      }

      setStableBalance(qugBalance);
      lastBalanceUpdateRef.current = Date.now();
    };

    const handleWalletBalanceUpdated = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { symbol, balance, reason, infoOnly } = customEvent.detail || {};
      if (symbol !== 'QUG') return;

      // infoOnly events (pending_mining_reward) are display hints only — not confirmed balance
      if (infoOnly) return;

      // v3.6.1-beta: CRITICAL - Validate balance before using
      if (!isValidBalance(balance)) {
        console.warn(`🚨 [TopBar] Rejecting invalid wallet-balance-updated: ${balance} (reason: ${reason})`);
        return;
      }

      console.log('💰 TopBar: wallet-balance-updated:', balance, 'reason:', reason);

      // v2.9.1-beta: Mining updates are authoritative - use directly
      // DEX updates use Math.max() to prevent race condition drops
      const isMiningUpdate = reason && (
        reason === 'mining_reward' ||
        reason === 'p2p_mining_reward' ||
        reason === 'mining_stats_update' ||
        reason.startsWith('mining_reward')
      );

      // v2.9.3-beta: Check if this is a DEX swap deduction
      const isDexSwapDeduct = reason && (
        reason === 'dex-swap-deduct' ||
        reason === 'DexScreen.swap.deduct'
      );

      // v6.0.1: Check if this is a transaction send (balance should decrease)
      const isTransactionSent = reason && (
        reason === 'transaction_sent' ||
        reason === 'transaction_received'
      );

      if (isMiningUpdate || isDexSwapDeduct || isTransactionSent) {
        // Mining updates, DEX swaps, and transaction sends: use value directly
        // These are authoritative - balance MUST update (including decreases)
        console.log(isDexSwapDeduct ? '💸 TopBar: DEX deduct - setting balance to:' : isTransactionSent ? '📤 TopBar: Transaction sent - setting balance to:' : '⛏️ TopBar: Mining update - setting balance to:', balance);
        setStableBalance(balance);
      } else {
        // v3.6.1-beta: Use Math.max only if BOTH values are valid
        setStableBalance(prev => {
          if (!isValidBalance(prev)) return balance;
          return Math.max(prev, balance);
        });
      }
      lastBalanceUpdateRef.current = Date.now();
    };

    window.addEventListener('qug-balance-changed', handleBalanceChanged);
    window.addEventListener('dex-cooldown-expired', handleDexCooldownExpired);
    window.addEventListener('wallet-balance-updated', handleWalletBalanceUpdated);

    return () => {
      window.removeEventListener('qug-balance-changed', handleBalanceChanged);
      window.removeEventListener('dex-cooldown-expired', handleDexCooldownExpired);
      window.removeEventListener('wallet-balance-updated', handleWalletBalanceUpdated);
    };
  }, []);

  // Fetch total personal hashrate from API (sums all workers for this wallet)
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    const fetchMiningHashrate = async () => {
      try {
        const res = await qnkAPI.getMiningStats(walletAddress);
        if (res.success && res.data) {
          // Only show hashrate if miner is actually active
          if (res.data.is_active && res.data.hash_rate > 0) {
            setPersonalHashrate(res.data.hash_rate);
          } else {
            setPersonalHashrate(0);
          }
        }
      } catch {
        // Mining stats endpoint may not be available
      }
    };

    fetchMiningHashrate();
    const interval = setInterval(fetchMiningHashrate, 15000);
    return () => clearInterval(interval);
  }, []);

  // v3.4.16-beta: Detect Tor connection (.onion domain)
  useEffect(() => {
    if (typeof window !== 'undefined' && window.location) {
      const isTor = window.location.hostname.endsWith('.onion');
      setIsTorConnected(isTor);
      if (isTor) {
        console.log('🧅 [TopBar] Tor hidden service detected');
      }
    }
  }, []);

  // v9.1.4: Fetch Tor onion address from backend
  useEffect(() => {
    const fetchTorStatus = async () => {
      try {
        const res = await fetch('/api/v1/tor/status');
        if (res.ok) {
          const data = await res.json();
          const addr = data?.data?.onion_address;
          if (addr && typeof addr === 'string' && addr.length > 10) {
            const cleanAddr = addr.endsWith('.onion') ? addr : `${addr}.onion`;
            setTorOnionUrl(`http://${cleanAddr}`);
          }
        }
      } catch {
        // Keep hardcoded default onion URL
      }
    };
    fetchTorStatus();
  }, []);

  // v8.6.2: SSE subscription via shared sseManager (no duplicate EventSource)
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    const unsubs: (() => void)[] = [];

    // Listen for node status updates (block height, peers)
    // SSE format: {type: "NodeStatusUpdate", data: {status: {current_height, connected_peers, ...}, timestamp}}
    unsubs.push(sseManager.on('node-status', (data: any) => {
      const s = data.data?.status ?? data.data ?? data;
      if (s.current_height) {
        setLiveBlockHeight(s.current_height);
      }
      if (s.connected_peers !== undefined) {
        setLivePeers(s.connected_peers);
      }
    }));

    // Listen for mining stats updates (personal hashrate)
    // Server sends event type "mining_stats" with format {type: "MiningStats", data: {miner_address, avg_hash_rate, ...}}
    unsubs.push(sseManager.on('mining_stats', (data: any) => {
      const d = data.data ?? data;
      const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
      const normalizedMiner = (d.miner_address || '').replace(/^qnk/, '').toLowerCase();
      if (normalizedMiner === normalizedWallet && d.avg_hash_rate) {
        setPersonalHashrate(d.avg_hash_rate);
      }
      if (typeof d?.network_hashrate_hs === 'number' && d.network_hashrate_hs > 0) {
        setNetworkHashrate(d.network_hashrate_hs);
      }
      if (typeof d?.total_miners === 'number' && d.total_miners > 0) {
        setNetworkMiners(d.total_miners);
      }
    }));

    // v10.4.11: Subscribe to new-block SSE for instant block height updates
    // SSE format: {type: "NewBlock", data: {height, hash, dag_round, ...}}
    unsubs.push(sseManager.on('new-block', (data: any) => {
      const h = data.data?.height ?? data.height;
      if (typeof h === 'number' && h > liveBlockHeightRef.current) {
        setLiveBlockHeight(h);
      }
    }));

    // Also listen for block height from mining rewards
    // SSE format: {type: "MiningReward", data: {block_height, hash_rate, miner_address, ...}}
    unsubs.push(sseManager.on('mining_reward', (data: any) => {
      const d = data.data ?? data;
      if (d.block_height) {
        setLiveBlockHeight(d.block_height);
      }
      if (d.hash_rate) {
        const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
        const normalizedMiner = (d.miner_address || '').replace(/^qnk/, '').toLowerCase();
        if (normalizedMiner === normalizedWallet) {
          setPersonalHashrate(d.hash_rate);
        }
      }
    }));

    // v6.0.2: Listen for balance-updated SSE events
    unsubs.push(sseManager.on('balance-updated', (data: any) => {
      const eventAddr = (data.wallet_address || '').toLowerCase();
      const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
      if (eventAddr === normalizedWallet && data.new_balance !== undefined) {
        console.log('[TopBar] SSE balance-updated:', data.new_balance, 'reason:', data.change_reason);
        window.dispatchEvent(new CustomEvent('wallet-balance-updated', {
          detail: {
            symbol: 'QUG',
            balance: data.new_balance,
            reason: data.change_reason || 'sse_balance_update'
          }
        }));
      }
    }));

    return () => {
      unsubs.forEach(unsub => unsub());
    };
  }, []);

  // v3.4.16-beta: Update from props when SSE hasn't provided updates yet
  useEffect(() => {
    if (blockHeight > liveBlockHeight) {
      setLiveBlockHeight(blockHeight);
    }
    if (peers !== livePeers && livePeers === 0) {
      setLivePeers(peers);
    }
  }, [blockHeight, peers]);

// v10.4.11: Poll /api/v1/status every 1s for fresh block height (fast fallback when SSE is slow)
  const liveBlockHeightRef = useRef(liveBlockHeight);
  liveBlockHeightRef.current = liveBlockHeight;
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/v1/status');
        if (!res.ok) return;
        const json = await res.json();
        const h = json?.data?.current_height ?? json?.current_height;
        if (typeof h === 'number' && h > liveBlockHeightRef.current) {
          setLiveBlockHeight(h);
        }
        const p = json?.data?.connected_peers ?? json?.connected_peers;
        if (typeof p === 'number') {
          setLivePeers(p);
        }
      } catch {}
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  // Network Health Gauge: fetch k-parameter every 60s
  useEffect(() => {
    const fetchK = async () => {
      try {
        const res = await fetch('/api/v1/k-parameter');
        if (!res.ok) return;
        const json = await res.json();
        const d = json.data ?? json;
        if (typeof d.k_value === 'number') setKValue(d.k_value);
        if (typeof d.phase === 'string') setKPhase(d.phase);
        setKData(d);
      } catch { /* ignore */ }
    };
    fetchK();
    const interval = setInterval(fetchK, 60000);
    return () => clearInterval(interval);
  }, []);

  // v7.2.0: Poll miner-link status (lightweight REST check every 10s)
  useEffect(() => {
    const walletAddr = localStorage.getItem('walletAddress') || '';
    if (!walletAddr) return;

    const fetchMinerLinkStatus = async () => {
      try {
        const res = await fetch(`/api/v1/miner-link/status/${walletAddr}`);
        if (res.ok) {
          const data = await res.json();
          setMinerLinkCount(data?.data?.connected_miners ?? 0);
        }
      } catch {
        // Silently ignore — miner link is optional
      }
    };

    fetchMinerLinkStatus();
    const interval = setInterval(fetchMinerLinkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // v3.9.1-beta: Fetch loan status and bank messages
  useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    const fetchBankingData = async () => {
      try {
        const baseUrl = localStorage.getItem('nodeUrl') || '';

        // Fetch loan applications for this wallet
        const loanRes = await fetch(`${baseUrl}/api/v1/quillon-bank/lending/applications`);
        if (loanRes.ok) {
          const loans = await loanRes.json();
          // Find active loan for this wallet
          const normalizedWallet = walletAddress.replace(/^qnk/, '').toLowerCase();
          const activeLoan = loans.find((loan: any) => {
            const loanWallet = (loan.borrower_address || loan.borrower || '').replace(/^qnk/, '').toLowerCase();
            return loanWallet === normalizedWallet &&
              (loan.status === 'approved' || loan.status === 'active');
          });

          if (activeLoan) {
            setHasActiveLoan(true);
            setLoanDetails({
              id: activeLoan.id || activeLoan.loan_id,
              amount: activeLoan.amount || activeLoan.loan_amount,
              collateral: activeLoan.collateral || activeLoan.collateral_amount,
              interestRate: activeLoan.interest_rate || 5,
              status: activeLoan.status,
              createdAt: activeLoan.created_at || activeLoan.timestamp || Date.now(),
              dueDate: activeLoan.due_date,
              remainingBalance: activeLoan.remaining_balance || activeLoan.amount,
            });
          }
        }

        // Fetch messages for this wallet
        const msgRes = await fetch(`${baseUrl}/api/v1/quillon-bank/messages/${encodeURIComponent(walletAddress)}`);
        if (msgRes.ok) {
          const messages = await msgRes.json();
          setBankMessages(messages);
          const unread = messages.filter((m: BankMessage) => !m.read && m.from === 'bank').length;
          setUnreadMessages(unread);
        }
      } catch (err) {
        console.log('📬 [TopBar] Banking data fetch (endpoints may not exist yet):', err);
      }
    };

    // Initial fetch
    fetchBankingData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchBankingData, 30000);
    return () => clearInterval(interval);
  }, []);

  // v3.9.1-beta: Send message to bank
  const sendMessageToBank = async () => {
    if (!newMessage.trim() || isSendingMessage) return;

    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) return;

    setIsSendingMessage(true);
    try {
      const baseUrl = localStorage.getItem('nodeUrl') || '';
      const res = await fetch(`${baseUrl}/api/v1/quillon-bank/messages/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddress,
          content: newMessage.trim(),
          subject: loanDetails?.id ? `Re: Loan #${loanDetails.id}` : 'General Inquiry',
        }),
      });

      if (res.ok) {
        const newMsg: BankMessage = {
          id: `msg_${Date.now()}`,
          from: 'user',
          content: newMessage.trim(),
          timestamp: Date.now(),
          read: true,
          loanId: loanDetails?.id,
        };
        setBankMessages(prev => [...prev, newMsg]);
        setNewMessage('');
      }
    } catch (err) {
      console.error('❌ [TopBar] Failed to send message:', err);
    } finally {
      setIsSendingMessage(false);
    }
  };

  // v3.9.2-beta: Fetch recent transactions for inbox
  useEffect(() => {
    if (!walletAddr) return;
    const fetchInbox = async () => {
      try {
        const res = await qnkAPI.getWalletHistory(walletAddr, 10);
        if (res.success && res.data) {
          setRecentInboxItems(res.data.slice(0, 5));
        }
      } catch (err) {
        console.log('📬 [TopBar] Inbox fetch:', err);
      }
    };
    fetchInbox();
    const interval = setInterval(fetchInbox, 60000);
    return () => clearInterval(interval);
  }, [walletAddr]);

  // Network power + miners — same source as MiningDashboard (/api/v1/network/supply)
  useEffect(() => {
    const fetchNetworkPower = async () => {
      try {
        const res = await fetch('/api/v1/network/supply');
        if (!res.ok) return;
        const d = await res.json();
        const data = d?.data ?? d;
        // Only update if non-zero (don't clobber a good value with a stale 0)
        if (typeof data?.network_hashrate === 'number' && data.network_hashrate > 0)
          setNetworkHashrate(data.network_hashrate);
        if (typeof data?.connected_miners === 'number' && data.connected_miners > 0)
          setNetworkMiners(data.connected_miners);
      } catch {}
    };
    fetchNetworkPower();
    const interval = setInterval(fetchNetworkPower, 15000);
    return () => clearInterval(interval);
  }, []);

  // v3.9.2-beta: Copy wallet address to clipboard
  const copyWalletAddress = () => {
    if (walletAddr) {
      navigator.clipboard.writeText(walletAddr);
      setCopiedWallet(true);
      setTimeout(() => setCopiedWallet(false), 2000);
    }
  };

  // Get the display balance (always from localStorage)
  const displayBalance = getDisplayBalance();

  // v3.4.16-beta: Format hashrate for display
  const formatHashrate = (hashrate: number): string => {
    if (hashrate === 0) return '0 H/s';
    if (hashrate >= 1e12) return `${(hashrate / 1e12).toFixed(2)} TH/s`;
    if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)} GH/s`;
    if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)} MH/s`;
    if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)} KH/s`;
    return `${hashrate.toFixed(0)} H/s`;
  };

  // v3.4.2: Real API search function (same as ExplorerScreen)
  const performSearch = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    const results: SearchResult[] = [];

    try {
      // Determine search type based on query format
      let searchType = '';
      if (query.match(/^tx_[a-f0-9]+/i) || query.match(/^[a-f0-9]{32}$/i) || query.match(/^[a-f0-9]{64}$/i)) searchType = 'transaction';
      else if (query.match(/^vtx_[a-f0-9]+/i)) searchType = 'vertex';
      else if (query.match(/^0x[a-f0-9]{40}$/i)) searchType = 'contract';
      else if (query.match(/^qnk[a-z0-9]{39}$/i)) searchType = 'address';
      else if (query.match(/^\d+$/)) searchType = 'block';

      console.log(`🔍 TopBar searching for: ${query} (type: ${searchType})`);

      if (searchType === 'block') {
        const blockNum = parseInt(query);
        const blockResponse = await qnkAPI.getBlock(blockNum);
        if (blockResponse.success && blockResponse.data) {
          results.push({
            type: 'block',
            id: blockNum.toString(),
            title: `Block #${blockNum}`,
            subtitle: `${Array.isArray(blockResponse.data) ? blockResponse.data.length : 0} transactions`,
            data: {
              height: blockNum,
              tx_count: Array.isArray(blockResponse.data) ? blockResponse.data.length : 0,
              hash: blockResponse.data[0]?.hash || 'N/A',
              transactions: blockResponse.data
            }
          });
        } else {
          results.push({
            type: 'error',
            id: 'not-found',
            title: 'Block Not Found',
            subtitle: `Block #${blockNum} does not exist yet`
          });
        }
      } else if (searchType === 'transaction') {
        const txResponse = await qnkAPI.getTransactionByHash(query);
        if (txResponse.success && txResponse.data) {
          const txData = txResponse.data;
          // v3.4.3: Store raw amount/fee - modal will convert using DECIMALS (1e6)
          results.push({
            type: 'transaction',
            id: query,
            title: 'Transaction Found',
            subtitle: txData.status || 'confirmed',
            hash: txData.hash || query,
            data: {
              hash: txData.hash || query,
              amount: txData.amount || 0,  // Raw value - modal converts
              status: txData.status || 'confirmed',
              timestamp: txData.timestamp ? new Date(txData.timestamp * 1000).toLocaleString() : 'N/A',
              from: txData.from || 'N/A',
              to: txData.to || 'N/A',
              block_height: txData.block_height,
              confirmations: txData.confirmations,
              fee: txData.fee || 0,  // Raw value - modal converts
              token_type: txData.token_type
            }
          });
        } else {
          results.push({
            type: 'error',
            id: 'not-found',
            title: 'Transaction Not Found',
            subtitle: `${query.substring(0, 16)}...${query.substring(48)}`
          });
        }
      } else if (searchType === 'address') {
        // v3.4.3: Try contract lookup first, then wallet lookup
        // Both contracts and wallets use qnk... format
        const contractResponse = await qnkAPI.getContractInfo(query);
        if (contractResponse.success && contractResponse.data && contractResponse.data.name) {
          // Found a contract at this address
          results.push({
            type: 'contract',
            id: query,
            title: contractResponse.data.name || 'Smart Contract',
            subtitle: contractResponse.data.symbol ? `${contractResponse.data.symbol} Token` : 'Contract Address',
            hash: query,
            data: {
              address: query,
              name: contractResponse.data.name,
              symbol: contractResponse.data.symbol,
              contract_type: contractResponse.data.contract_type || 'Unknown',
              total_supply: contractResponse.data.total_supply,
              decimals: contractResponse.data.decimals || 18,
              deployer: contractResponse.data.deployer,
              deployment_height: contractResponse.data.deployment_height,
              verified: contractResponse.data.verified || false
            }
          });
        } else {
          // Not a contract, try as wallet address
          const balanceResponse = await qnkAPI.getWalletBalance(query);
          if (balanceResponse.success && balanceResponse.data) {
            results.push({
              type: 'address',
              id: query,
              title: 'Wallet Address',
              subtitle: `Balance: ${(balanceResponse.data.balance_qnk || 0).toFixed(4)} ${TICKER_SYMBOL}`,
              hash: query,
              data: {
                address: query,
                balance: balanceResponse.data.balance_qnk || 0,
                nonce: balanceResponse.data.nonce || 0
              }
            });
          } else {
            results.push({
              type: 'address',
              id: query,
              title: 'New Wallet Address',
              subtitle: 'Balance: 0 ' + TICKER_SYMBOL,
              hash: query,
              data: { address: query, balance: 0, nonce: 0 }
            });
          }
        }
      } else if (searchType === 'contract') {
        const contractResponse = await qnkAPI.getContractInfo(query);
        if (contractResponse.success && contractResponse.data) {
          results.push({
            type: 'contract',
            id: query,
            title: contractResponse.data.name || 'Smart Contract',
            subtitle: contractResponse.data.symbol ? `${contractResponse.data.symbol} Token` : 'Contract Address',
            hash: query,
            data: {
              address: query,
              name: contractResponse.data.name,
              symbol: contractResponse.data.symbol,
              contract_type: contractResponse.data.contract_type || 'Unknown',
              total_supply: contractResponse.data.total_supply,
              decimals: contractResponse.data.decimals || 18,
              deployer: contractResponse.data.deployer,
              deployment_height: contractResponse.data.deployment_height,
              verified: contractResponse.data.verified || false
            }
          });
        } else {
          // Contract address format but not found - might be newly deployed
          results.push({
            type: 'contract',
            id: query,
            title: 'Unknown Contract',
            subtitle: 'Contract not found or not indexed',
            hash: query,
            data: { address: query }
          });
        }
      } else if (query.length >= 3) {
        // Generic search - show helpful hints
        results.push({
          type: 'block',
          id: 'hint-block',
          title: 'Search by block number',
          subtitle: `Enter a number like "${blockHeight}" to find a block`
        });
        results.push({
          type: 'transaction',
          id: 'hint-tx',
          title: 'Search by transaction hash',
          subtitle: 'Enter a 64-character hex hash'
        });
        results.push({
          type: 'address',
          id: 'hint-address',
          title: 'Search by address',
          subtitle: 'Enter wallet or contract address starting with "qnk"'
        });
      }
    } catch (error) {
      console.error('Search failed:', error);
      results.push({
        type: 'error',
        id: 'error',
        title: 'Search Failed',
        subtitle: 'Unable to connect to the network'
      });
    }

    setSearchResults(results);
    setIsSearching(false);
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim()) {
        performSearch(searchQuery);
      } else {
        setSearchResults([]);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  // Cmd/Ctrl+K opens search; Escape closes it
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(true);
        setTimeout(() => searchInputRef.current?.focus(), 50);
      }
      if (e.key === 'Escape') {
        setSearchOpen(false);
        setSearchQuery('');
        setSearchResults([]);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const openSearch = () => {
    setSearchOpen(true);
    setTimeout(() => searchInputRef.current?.focus(), 50);
  };

  const closeSearch = () => {
    setSearchOpen(false);
    setSearchQuery('');
    setSearchResults([]);
    setShowResults(false);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getQCIStatus = (qci: number) => {
    if (qci >= 0.9) return 'Sublime';
    if (qci >= 0.8) return 'Coherent';
    return 'Stabilizing';
  };

  const getResultIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4" />;
      case 'block': return <Blocks className="w-4 h-4" />;
      case 'address': return <User className="w-4 h-4" />;
      case 'node': return <User className="w-4 h-4" />;
      case 'contract': return <FileCode className="w-4 h-4" />;
    }
  };

  return (
    <>
    <div
      className="topbar-main backdrop-blur-xl border-b px-6 pb-4 relative z-50"
      style={{
        background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95) 0%, rgba(25, 25, 40, 0.95) 100%)',
        borderColor: 'rgba(212, 175, 55, 0.2)',
        boxShadow: '0 4px 20px rgba(212, 175, 55, 0.15)',
        paddingTop: '20px'
      }}
    >
      <div className="flex items-center justify-between relative">
        {/* Left: Search trigger button */}
        <div className="ml-4">
          <motion.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.93 }}
            onClick={openSearch}
            className="flex items-center gap-2 px-3 py-2 rounded-xl text-amber-400/70 hover:text-amber-300 transition-all group"
            style={{
              background: 'rgba(212,175,55,0.07)',
              border: '1.5px solid rgba(212,175,55,0.2)',
            }}
            title="Search (⌘K)"
          >
            <Search className="w-4 h-4" />
            <span className="text-xs text-amber-400/50 font-mono hidden sm:block">⌘K</span>
          </motion.button>
        </div>

        {/* Full-width search overlay — slides in and hides other topbar elements */}
        <AnimatePresence>
          {searchOpen && (
            <motion.div
              initial={{ opacity: 0, scaleX: 0.7, y: -8 }}
              animate={{ opacity: 1, scaleX: 1, y: 0 }}
              exit={{ opacity: 0, scaleX: 0.7, y: -8 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className="absolute inset-x-0 top-1/2 -translate-y-1/2 z-[60] px-4"
              style={{ transformOrigin: 'left center' }}
            >
              <div
                className="relative w-full rounded-2xl overflow-visible"
                style={{
                  background: 'linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,20,35,0.98) 100%)',
                  border: '2px solid rgba(212,175,55,0.5)',
                  boxShadow: '0 0 60px rgba(212,175,55,0.25), 0 20px 60px rgba(0,0,0,0.6)',
                }}
              >
                {/* Search icon */}
                <div className="absolute left-5 top-1/2 -translate-y-1/2 pointer-events-none">
                  {isSearching ? (
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}>
                      <Search className="w-5 h-5 text-amber-400" />
                    </motion.div>
                  ) : (
                    <Search className="w-5 h-5 text-amber-400/70" />
                  )}
                </div>

                {/* The big input */}
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => { setSearchQuery(e.target.value); setShowResults(true); }}
                  onBlur={() => setTimeout(() => { if (!searchQuery) closeSearch(); }, 200)}
                  placeholder="Search blocks, transactions, addresses, contracts…"
                  className="w-full pl-14 pr-20 py-4 bg-transparent text-amber-50 text-lg placeholder-amber-300/30 focus:outline-none font-light tracking-wide"
                />

                {/* Hint + close */}
                <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-3">
                  <span className="text-xs text-amber-400/30 font-mono hidden md:block">ESC to close</span>
                  <motion.button
                    whileHover={{ scale: 1.15, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={closeSearch}
                    className="p-1.5 rounded-lg text-amber-400/50 hover:text-amber-300 hover:bg-amber-500/10 transition-all"
                  >
                    <X className="w-4 h-4" />
                  </motion.button>
                </div>

                {/* Results dropdown */}
                <AnimatePresence>
                  {showResults && searchResults.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      className="absolute top-full left-0 right-0 mt-2 rounded-2xl overflow-hidden max-h-[60vh] overflow-y-auto"
                      style={{
                        background: 'linear-gradient(135deg, rgba(12,12,22,0.99) 0%, rgba(22,22,38,0.99) 100%)',
                        border: '2px solid rgba(212,175,55,0.3)',
                        boxShadow: '0 20px 60px rgba(0,0,0,0.7), 0 0 40px rgba(212,175,55,0.15)',
                      }}
                    >
                {searchResults.map((result, index) => (
                  <motion.div
                    key={`${result.type}-${result.id}-${index}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-4 border-b last:border-b-0 cursor-pointer group"
                    style={{
                      borderColor: 'rgba(212, 175, 55, 0.1)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                    onClick={() => {
                      console.log('View detail:', result);
                      if (result.data) {
                        // v3.4.20: Use enhanced SmartContractModal for contracts
                        if (result.type === 'contract') {
                          setSelectedContract(result.data);
                        } else {
                          setSelectedDetail(result);
                        }
                      }
                      closeSearch();
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="p-2 rounded-lg"
                        style={{
                          background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                          border: '1px solid rgba(212, 175, 55, 0.3)'
                        }}
                      >
                        <div className="text-amber-400">{getResultIcon(result.type)}</div>
                      </div>
                      <div>
                        <div className="text-amber-100 font-semibold">{result.title}</div>
                        {result.subtitle && (
                          <div className="text-amber-300/60 text-sm">{result.subtitle}</div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      {result.hash && (
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(result.hash!, `${result.type}-${result.id}`);
                          }}
                          className="p-1 text-amber-400/60 hover:text-amber-400 transition-colors"
                        >
                          {copiedId === `${result.type}-${result.id}` ? (
                            <Check className="w-4 h-4 text-green-400" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </motion.button>
                      )}
                      <ExternalLink className="w-4 h-4 text-amber-400/60" />
                    </div>
                  </motion.div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Center: Network Status */}
        <div className="flex items-center gap-4 shrink-0 px-6">
          <div className="text-center relative group">
            <motion.div
              className="font-bold text-2xl bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent cursor-help font-mono whitespace-nowrap"
              key="stable-balance-display"
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              title={`Full 24-decimal precision balance`}
            >
              {/* v3.6.10-beta: Always show full 24 decimal precision */}
              {displayBalance.toFixed(24)} {TICKER_SYMBOL}
            </motion.div>
            <div className="text-amber-300/60 text-sm font-medium flex items-center gap-2 justify-center">
              <span>Total Balance</span>
              <motion.div
                className="w-1.5 h-1.5 rounded-full bg-green-400"
                animate={{ scale: [1, 1.3, 1], opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 2, repeat: Infinity }}
                title="Live updates enabled"
              />
            </div>
          </div>

          <div className="h-8 w-px bg-gradient-to-b from-transparent via-amber-500/30 to-transparent" />

          <div className="flex items-center gap-3">
            {/* v3.4.16-beta: Tor privacy indicator */}
            {isTorConnected && (
              <motion.div
                className="flex items-center gap-1.5 px-2 py-1 bg-purple-500/20 border border-purple-500/40 rounded-lg"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                title="Connected via Tor Hidden Service - Enhanced Privacy"
              >
                <span className="text-lg">🧅</span>
                <span className="text-purple-300 text-xs font-medium">Tor</span>
              </motion.div>
            )}

            {/* v7.2.0: Miner Link indicator */}
            {minerLinkCount > 0 && (
              <motion.div
                className="flex items-center gap-1.5 px-2 py-1 bg-green-500/15 border border-green-500/30 rounded-lg cursor-pointer"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title={`${minerLinkCount} personal miner${minerLinkCount > 1 ? 's' : ''} connected — Click to manage`}
                onClick={() => setShowMinerLinkModal(true)}
              >
                <div className="relative">
                  <Pickaxe className="w-3.5 h-3.5 text-green-400" />
                  <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                </div>
                <span className="text-green-300 text-xs font-medium">{minerLinkCount}</span>
              </motion.div>
            )}

            {/* ── Network Stats Strip ─────────────────────────────────── */}
            {/* Online dot */}
            <motion.div
              className={`w-2 h-2 rounded-full flex-shrink-0 ${isOnline ? 'bg-green-400' : 'bg-red-500'}`}
              animate={isOnline ? { scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] } : {}}
              transition={{ duration: 2, repeat: Infinity }}
            />

            {/* Block height — flashes amber on each new block */}
            <div className="flex flex-col items-center px-3 py-1 rounded-xl border min-w-[72px] bg-amber-500/8 border-amber-500/20">
              <span className="text-sm font-bold font-mono leading-tight text-amber-100">#{liveBlockHeight.toLocaleString()}</span>
              <span className="text-amber-400/50 text-[9px] font-semibold uppercase tracking-wider">Block</span>
            </div>

            {/* Peers — clickable */}
            <motion.button
              onClick={() => setShowNetworkMap(true)}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              className="flex flex-col items-center px-3 py-1 rounded-xl border transition-all min-w-[60px]"
              style={{ background: 'rgba(251,191,36,0.06)', borderColor: 'rgba(251,191,36,0.2)' }}
              title="View P2P network map"
            >
              <span className="flex items-center gap-1 text-amber-200 text-sm font-bold leading-tight">
                <Globe className="w-3 h-3 text-amber-400" />{livePeers}
              </span>
              <span className="text-amber-400/50 text-[9px] font-semibold uppercase tracking-wider">Peers</span>
            </motion.button>

            {/* Network power — always visible */}
            <motion.div
              className="flex flex-col items-center px-3 py-1 rounded-xl min-w-[72px]"
              style={{ background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.28)' }}
              animate={{ borderColor: networkHashrate > 0 ? 'rgba(139,92,246,0.45)' : 'rgba(139,92,246,0.2)' }}
              title="Total Network Mining Power"
            >
              <span className="flex items-center gap-1 text-violet-200 text-sm font-bold leading-tight">
                <motion.span animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 1.8, repeat: Infinity }}>
                  <Zap className="w-3 h-3 text-violet-400" />
                </motion.span>
                {networkHashrate > 0 ? formatHashrate(networkHashrate) : '—'}
              </span>
              <span className="text-violet-400/50 text-[9px] font-semibold uppercase tracking-wider">Net Power</span>
            </motion.div>

            {/* Miners count — always visible */}
            <motion.div
              className="flex flex-col items-center px-3 py-1 rounded-xl min-w-[60px]"
              style={{ background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.25)' }}
              animate={{ borderColor: networkMiners > 0 ? 'rgba(249,115,22,0.4)' : 'rgba(249,115,22,0.18)' }}
              title="Active Miners on Network"
            >
              <span className="flex items-center gap-1 text-orange-200 text-sm font-bold leading-tight">
                <Pickaxe className="w-3 h-3 text-orange-400" />
                {networkMiners > 0 ? networkMiners.toLocaleString() : '—'}
              </span>
              <span className="text-orange-400/50 text-[9px] font-semibold uppercase tracking-wider">Miners</span>
            </motion.div>

            {/* Personal hashrate — only when active, click to manage */}
            {personalHashrate > 0 && (
              <motion.div
                className="flex flex-col items-center px-3 py-1 rounded-xl cursor-pointer min-w-[72px]"
                style={{ background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.35)' }}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                whileHover={{ scale: 1.06 }}
                whileTap={{ scale: 0.94 }}
                title="Your Personal Mining Hashrate — Click to manage miners"
                onClick={() => setShowMinerLinkModal(true)}
              >
                <span className="flex items-center gap-1 text-cyan-200 text-sm font-bold leading-tight">
                  <Zap className="w-3 h-3 text-cyan-400" />{formatHashrate(personalHashrate)}
                </span>
                <span className="text-cyan-400/50 text-[9px] font-semibold uppercase tracking-wider">My Power</span>
              </motion.div>
            )}
          </div>
        </div>

        {/* Right: Network Health Gauge — canvas clock animation matching admin panel */}
        <div className="flex items-center gap-2">
          {/* NHG with hover tooltip — tooltip is DOM child so onMouseLeave doesn't fire when cursor moves into it */}
          <div
            ref={nhgRef}
            className="relative"
            onMouseEnter={() => setShowKTooltip(true)}
            onMouseLeave={() => setShowKTooltip(false)}
            style={{ zIndex: 200 }}
          >
            <NHGClock kValue={kValue} kPhase={kPhase} />

            {/* Tooltip: absolute inside nhgRef, appears below-right, no portal needed */}
            <AnimatePresence>
              {showKTooltip && (
                <motion.div
                  initial={{ opacity: 0, y: -6, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -6, scale: 0.96 }}
                  transition={{ duration: 0.15, ease: 'easeOut' }}
                  className="absolute top-full right-0 mt-2 w-[300px] rounded-2xl overflow-hidden pointer-events-none"
                  style={{
                    zIndex: 9999,
                    background: 'linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(10,15,30,0.99) 100%)',
                    backdropFilter: 'blur(24px)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    boxShadow: kPhase === 'critical'
                      ? '0 0 40px rgba(239,68,68,0.2), 0 20px 60px rgba(0,0,0,0.85)'
                      : kPhase === 'approaching'
                        ? '0 0 40px rgba(245,158,11,0.2), 0 20px 60px rgba(0,0,0,0.85)'
                        : '0 0 40px rgba(16,185,129,0.15), 0 20px 60px rgba(0,0,0,0.85)',
                  }}
                >
                  {/* Header */}
                  <div className={`px-4 py-3 border-b border-white/5 flex items-center justify-between ${
                    kPhase === 'critical' ? 'bg-red-500/10' : kPhase === 'approaching' ? 'bg-amber-500/10' : 'bg-emerald-500/10'
                  }`}>
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.15em] text-white/40 mb-0.5">Network Health Gauge</div>
                      <div className="text-white/60 text-[10px] font-mono">K = 2π √(ΔH · Δs · ℏ) / τ</div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <div className={`text-xl font-black font-mono ${
                        kPhase === 'critical' ? 'text-red-400' : kPhase === 'approaching' ? 'text-amber-400' : 'text-emerald-400'
                      }`}>
                        {kValue === 0 ? '—' : kValue.toFixed(4)}
                      </div>
                      <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full ${
                        kPhase === 'critical'
                          ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                          : kPhase === 'approaching'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      }`}>
                        {kPhase === 'critical' ? '⚠ Critical' : kPhase === 'approaching' ? '◎ Warning' : '✓ Stable'}
                      </span>
                    </div>
                  </div>

                  {/* Phase thresholds */}
                  <div className="px-4 py-2 border-b border-white/5 flex gap-3 text-[9px] font-mono">
                    <span className="flex items-center gap-1 text-emerald-400/70"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />K&lt;5 Stable</span>
                    <span className="flex items-center gap-1 text-amber-400/70"><span className="w-1.5 h-1.5 rounded-full bg-amber-500 inline-block" />5≤K&lt;10 Warn</span>
                    <span className="flex items-center gap-1 text-red-400/70"><span className="w-1.5 h-1.5 rounded-full bg-red-500 inline-block" />K≥10 Critical</span>
                  </div>

                  {/* Factor grid */}
                  <div className="px-4 py-3 grid grid-cols-2 gap-2 border-b border-white/5">
                    {[
                      { letter: 'G', label: 'ΔH (Entropy)', color: '#10b981', val: kData?.delta_h },
                      { letter: 'Q', label: 'Δs (State Div.)', color: '#06b6d4', val: kData?.delta_s },
                      { letter: 'T', label: 'Rejection Ratio', color: '#f97316', val: kData?.rejection_ratio },
                      { letter: 'I', label: 'Observer Cover.', color: '#3b82f6', val: kData?.observer_coverage },
                      { letter: 'R', label: 'λ Commitment', color: '#a855f7', val: kData?.lambda_commit },
                      { letter: 'F', label: 'f_irrev', color: '#ec4899', val: kData?.f_irrev },
                    ].map(({ letter, label, color, val }) => (
                      <div key={letter} className="flex items-center gap-2 min-w-0">
                        <div
                          className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-black"
                          style={{ background: `${color}22`, border: `1px solid ${color}66`, color }}
                        >
                          {letter}
                        </div>
                        <div className="min-w-0">
                          <div className="text-[9px] text-white/40 truncate">{label}</div>
                          <div className="text-[10px] font-mono font-semibold text-white/80">
                            {val !== undefined && val !== null ? (typeof val === 'number' ? val.toFixed(4) : String(val)) : '—'}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Stats row */}
                  <div className="px-4 py-3 grid grid-cols-2 gap-x-4 gap-y-1">
                    {[
                      { label: 'Rounds', val: kData?.rounds_computed },
                      { label: 'K-Enhanced', val: typeof kData?.k_enhanced === 'number' ? kData.k_enhanced.toFixed(4) : null },
                      { label: 'ZK Commit', val: kData?.lambda_commit !== undefined ? `${(kData.lambda_commit * 100).toFixed(1)}%` : null },
                      { label: 'Phase', val: kData?.phase ?? kPhase },
                    ].map(({ label, val }) => (
                      <div key={label} className="flex items-baseline justify-between gap-1">
                        <span className="text-[9px] text-white/35 uppercase tracking-wider">{label}</span>
                        <span className="text-[10px] font-mono text-white/70">{val ?? '—'}</span>
                      </div>
                    ))}
                  </div>

                  {/* Health bar */}
                  <div className="px-4 pb-3">
                    <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${Math.max(2, 100 - Math.min(kValue / 15 * 100, 100))}%`,
                          background: kPhase === 'critical'
                            ? 'linear-gradient(90deg, #ef4444, #dc2626)'
                            : kPhase === 'approaching'
                              ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                              : 'linear-gradient(90deg, #10b981, #059669)',
                        }}
                      />
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className="text-[8px] text-white/25">Critical</span>
                      <span className="text-[8px] text-white/25">Healthy</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* v5.1.1: Deploy Panel - Visible for all logged-in users (read-only status) */}
          {(() => {
            const MASTER_WALLET = 'efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';
            const isMaster = walletAddr.replace('qnk', '').replace('qug', '') === MASTER_WALLET;
            if (!walletAddr) return null;
            return (
              <>
                <div className="w-px h-8 bg-amber-500/30 mx-2" />
                <motion.button
                  onClick={() => {
                    const event = new CustomEvent('open-deploy-panel');
                    window.dispatchEvent(event);
                  }}
                  className="p-2 rounded-lg bg-gradient-to-br from-emerald-500/20 to-green-500/20 border border-emerald-500/40 hover:border-emerald-400/60 transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  title="Node Deploy Panel"
                >
                  <Shield className="w-5 h-5 text-emerald-400" />
                </motion.button>
              </>
            );
          })()}
          {/* v9.0.3: Node Settings - Visible for admin wallet OR master wallet */}
          {(() => {
            const MASTER_WALLET = 'efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';
            const isMasterOrAdmin = isNodeAdmin || walletAddr.replace('qnk', '').replace('qug', '') === MASTER_WALLET;
            if (!isMasterOrAdmin || !walletAddr) return null;
            return (
              <motion.button
                onClick={() => {
                  window.dispatchEvent(new CustomEvent('open-node-settings'));
                }}
                className="relative p-2 rounded-lg bg-gradient-to-br from-blue-500/20 to-indigo-500/20 border border-blue-500/40 hover:border-blue-400/60 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Node Admin — Connected"
              >
                <Settings className="w-5 h-5 text-blue-400" />
                {/* v9.0.4: Green active dot — admin wallet online indicator */}
                <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500 border border-green-300/50" />
                </span>
              </motion.button>
            );
          })()}

          {/* v8.5.10: Bounty Campaign Button — genie target for BountyModal animation */}
          <div className="w-px h-8 bg-emerald-500/30 mx-2" />
          <motion.button
            id="bounty-genie-target"
            onClick={() => {
              window.dispatchEvent(new CustomEvent('open-bounty-modal'));
            }}
            className="relative flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/40 hover:border-emerald-400/60 transition-all"
            whileHover={{ scale: 1.08, boxShadow: '0 0 20px rgba(16,185,129,0.3)' }}
            whileTap={{ scale: 0.92 }}
            title="Bounty Campaign — Earn rewards"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', damping: 15, stiffness: 300 }}
          >
            <motion.div
              animate={{ rotate: [0, -10, 10, -5, 0] }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
            >
              <Trophy className="w-4 h-4 text-emerald-400" />
            </motion.div>
            <span className="text-emerald-300 text-xs font-bold">Bounty</span>
            {/* Points badge — shows user's actual score, pulses when > 0 */}
            <motion.span
              className="absolute -top-1.5 -right-1.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full text-[10px] font-black px-1"
              style={{
                background: bountyScore > 0
                  ? 'linear-gradient(135deg, #10B981, #06B6D4)'
                  : 'linear-gradient(135deg, #374151, #4B5563)',
                color: '#fff',
                boxShadow: bountyScore > 0 ? '0 0 8px rgba(16,185,129,0.5)' : 'none',
              }}
              animate={bountyScore > 0 ? { scale: [1, 1.15, 1] } : {}}
              transition={{ duration: 2, repeat: Infinity }}
            >
              {bountyScore >= 1000 ? `${(bountyScore / 1000).toFixed(1)}K` : bountyScore}
            </motion.span>
          </motion.button>

          {/* v3.9.2-beta: Profile Icon - Always visible for all users */}
          <div className="w-px h-8 bg-amber-500/30 mx-2" />
          <div className="relative">
            <motion.button
              onClick={() => setShowProfileModal(!showProfileModal)}
              className="relative p-2 rounded-lg bg-gradient-to-br from-amber-500/20 to-yellow-500/20 border border-amber-500/40 hover:border-amber-400/60 transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Profile"
            >
              <UserCircle className="w-5 h-5 text-amber-400" />
              {metamaskAddress && (
                <div className="absolute -bottom-1 -left-1 w-3.5 h-3.5 rounded-full bg-[#E2761B] border border-slate-800 flex items-center justify-center" title="MetaMask linked">
                  <svg width="8" height="8" viewBox="0 0 318.6 318.6" fill="none">
                    <path d="M274.1 35.5l-99.5 73.9L193 65.8z" fill="#fff"/>
                    <path d="M44.4 35.5l98.7 74.6-17.5-44.3z" fill="#fff"/>
                  </svg>
                </div>
              )}
              {unreadMessages > 0 && (
                <motion.div
                  className="absolute -top-1 -right-1 bg-red-500 text-white text-xs font-bold rounded-full min-w-[18px] h-[18px] flex items-center justify-center px-1"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 500 }}
                >
                  {unreadMessages > 9 ? '9+' : unreadMessages}
                </motion.div>
              )}
            </motion.button>
          </div>
        </div>
      </div>
    </div>

    {/* v3.9.2-beta: Universal Profile Dropdown Panel - Portal to escape AnimatedBorder z-index */}
    {createPortal(
    <AnimatePresence>
      {showProfileModal && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-[10000]"
            onClick={() => setShowProfileModal(false)}
          />
          {/* Dropdown Panel */}
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="fixed right-4 w-80 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl shadow-2xl z-[10001] overflow-hidden flex flex-col"
            style={{ top: 'calc(var(--topbar-height, 4rem) + 4px)', maxHeight: 'calc(100vh - var(--topbar-height, 4rem) - 12px)', boxShadow: '0 0 40px rgba(212, 175, 55, 0.15)' }}
          >
            {/* Wallet Section */}
            <div className="p-4 border-b border-amber-500/20">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-amber-500/20 to-yellow-500/20">
                  <UserCircle className="w-6 h-6 text-amber-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base font-bold text-amber-100">My Wallet</h3>
                  {/* Full-row address copy button */}
                  {walletAddr ? (
                    <motion.button
                      onClick={copyWalletAddress}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.97 }}
                      className="mt-1 w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-800/70 hover:bg-amber-500/10 border border-slate-700/50 hover:border-amber-500/40 transition-all group"
                      title="Click to copy full address"
                    >
                      <span className="text-amber-300/70 text-xs font-mono truncate flex-1 text-left">
                        {walletAddr.slice(0, 14)}…{walletAddr.slice(-8)}
                      </span>
                      <span className="flex-shrink-0 text-amber-400/50 group-hover:text-amber-400 transition-colors">
                        {copiedWallet ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                      </span>
                    </motion.button>
                  ) : (
                    <span className="text-slate-500 text-xs">Not connected</span>
                  )}
                </div>
              </div>
              <div className="flex items-center justify-between bg-slate-800/60 rounded-lg p-3">
                <span className="text-slate-400 text-sm">Balance</span>
                <span className="text-amber-100 font-bold font-mono">
                  {displayBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })} {TICKER_SYMBOL}
                </span>
              </div>
            </div>

            {/* MetaMask Linked Account */}
            {metamaskAddress && (
              <div className="px-4 py-3 border-b border-amber-500/20">
                <div className="flex items-center gap-2 mb-2">
                  <svg width="16" height="16" viewBox="0 0 318.6 318.6" fill="none" className="flex-shrink-0">
                    <path d="M274.1 35.5l-99.5 73.9L193 65.8z" fill="#E2761B" stroke="#E2761B" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M44.4 35.5l98.7 74.6-17.5-44.3z" fill="#E4761B" stroke="#E4761B" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span className="text-[#E2761B] font-medium text-sm">MetaMask</span>
                  <span className="ml-auto px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-[#E2761B]/20 text-[#E2761B]">LINKED</span>
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between bg-slate-800/60 rounded-lg px-3 py-2">
                    <span className="text-slate-400 text-xs">Address</span>
                    <span className="text-orange-200 text-xs font-mono">{metamaskAddress.slice(0, 6)}...{metamaskAddress.slice(-4)}</span>
                  </div>
                  {metamaskBalance !== null && (
                    <div className="flex items-center justify-between bg-slate-800/60 rounded-lg px-3 py-2">
                      <span className="text-slate-400 text-xs">Balance</span>
                      <span className="text-orange-200 text-xs font-mono">{metamaskBalance} ETH</span>
                    </div>
                  )}
                  {metamaskChainName && (
                    <div className="flex items-center justify-between bg-slate-800/60 rounded-lg px-3 py-2">
                      <span className="text-slate-400 text-xs">Network</span>
                      <span className="text-orange-200 text-xs font-mono">{metamaskChainName}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Network Status */}
            <div className="px-4 py-3 border-b border-amber-500/20">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex items-center gap-2">
                  <Blocks className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-slate-400 text-xs">Block:</span>
                  <span className="text-amber-100 text-xs font-mono">{liveBlockHeight.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Globe className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-slate-400 text-xs">Peers:</span>
                  <span className="text-emerald-100 text-xs font-mono">{livePeers}</span>
                </div>
                {personalHashrate > 0 && (
                  <div className="flex items-center gap-2 col-span-2">
                    <Zap className="w-3.5 h-3.5 text-yellow-400" />
                    <span className="text-slate-400 text-xs">Mining:</span>
                    <span className="text-yellow-100 text-xs font-mono">{formatHashrate(personalHashrate)}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Inbox / Recent Transactions */}
            <div className="px-4 py-3 border-b border-amber-500/20">
              <div className="flex items-center gap-2 mb-2">
                <Bell className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-amber-300 font-medium text-sm">Recent Activity</span>
              </div>
              {recentInboxItems.length === 0 ? (
                <div className="text-center text-slate-500 py-3 text-xs">
                  No recent transactions
                </div>
              ) : (
                <div className="space-y-1.5 max-h-[140px] overflow-y-auto">
                  {recentInboxItems.map((item: any, i: number) => (
                    <div key={item.id || i} className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/30 text-xs">
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        item.memo && item.direction === 'received' ? 'bg-cyan-400' :
                        item.direction === 'received' ? 'bg-emerald-400' :
                        item.tx_type === 'swap' ? 'bg-blue-400' :
                        item.tx_type === 'mining_reward' ? 'bg-yellow-400' :
                        'bg-amber-400'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <div className="text-slate-200 truncate">
                          {item.direction === 'received' ? 'Received' :
                           item.tx_type === 'swap' ? 'Swap' :
                           item.tx_type === 'mining_reward' ? 'Mining Reward' :
                           'Sent'}{' '}
                          <span className="font-mono text-amber-300">
                            {parseFloat(item.amount || '0').toLocaleString(undefined, { maximumFractionDigits: 4 })}
                          </span>{' '}
                          {item.token_symbol || TICKER_SYMBOL}
                        </div>
                        {item.memo && (
                          <div className="text-cyan-300/80 truncate flex items-center gap-1 mt-0.5">
                            <MessageCircle className="w-2.5 h-2.5 flex-shrink-0" />
                            <span className="truncate">{item.memo}</span>
                          </div>
                        )}
                        <div className="text-slate-500">Block #{item.block_height?.toLocaleString()}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Loan Section - Only shows if user has active loan */}
            {hasActiveLoan && loanDetails && (
              <div className="px-4 py-3 border-b border-amber-500/20">
                <div className="flex items-center gap-2 mb-2">
                  <CreditCard className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-300 font-medium text-sm">Active Loan</span>
                  <span className={`ml-auto px-2 py-0.5 rounded-full text-[10px] font-bold ${
                    loanDetails.status === 'active' ? 'bg-emerald-500/20 text-emerald-400' :
                    loanDetails.status === 'approved' ? 'bg-blue-500/20 text-blue-400' :
                    'bg-amber-500/20 text-amber-400'
                  }`}>
                    {loanDetails.status.toUpperCase()}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400">Amount:</span>
                    <span className="text-white ml-1 font-mono">{loanDetails.amount.toLocaleString()} {TICKER_SYMBOL}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Remaining:</span>
                    <span className="text-amber-400 ml-1 font-mono">{(loanDetails.remainingBalance || loanDetails.amount).toLocaleString()}</span>
                  </div>
                </div>
                {bankMessages.length > 0 && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-emerald-400 cursor-pointer hover:text-emerald-300">
                    <MessageCircle className="w-3 h-3" />
                    <span>{bankMessages.length} bank message{bankMessages.length !== 1 ? 's' : ''}</span>
                    {unreadMessages > 0 && <span className="text-red-400">({unreadMessages} new)</span>}
                  </div>
                )}
              </div>
            )}

            {/* Quick Links — login-screen icon+label grid style */}
            <div className="p-3 overflow-y-auto flex-1">
              {/* Nav grid */}
              <div className="grid grid-cols-3 gap-2 mb-3">
                {[
                  { label: 'History', icon: <Clock className="w-5 h-5" />, color: 'amber', onClick: () => { onNavigate?.('transactions'); setShowProfileModal(false); } },
                  { label: 'DEX', icon: <ArrowRight className="w-5 h-5" />, color: 'cyan', onClick: () => { onNavigate?.('dex'); setShowProfileModal(false); } },
                  { label: 'Mining', icon: <Zap className="w-5 h-5" />, color: 'yellow', onClick: () => { onNavigate?.('mining'); setShowProfileModal(false); } },
                  { label: 'Theme', icon: <Palette className="w-5 h-5" />, color: 'purple', onClick: () => { setShowThemeChooser(true); setShowProfileModal(false); } },
                  { label: 'Settings', icon: <Settings className="w-5 h-5" />, color: 'blue', onClick: () => { onNavigate?.('settings'); setShowProfileModal(false); } },
                  { label: 'Tax', icon: <FileText className="w-5 h-5" />, color: 'emerald', onClick: () => { setShowTaxModal(true); setShowProfileModal(false); } },
                ].map(item => {
                  const colorMap: {[k:string]:string} = {
                    amber: 'bg-amber-600/20 hover:bg-amber-600/40 border-amber-400/30 hover:border-amber-400/60 text-amber-400',
                    cyan: 'bg-cyan-600/20 hover:bg-cyan-600/40 border-cyan-400/30 hover:border-cyan-400/60 text-cyan-400',
                    yellow: 'bg-yellow-600/20 hover:bg-yellow-600/40 border-yellow-400/30 hover:border-yellow-400/60 text-yellow-400',
                    purple: 'bg-purple-600/20 hover:bg-purple-600/40 border-purple-400/30 hover:border-purple-400/60 text-purple-400',
                    blue: 'bg-blue-600/20 hover:bg-blue-600/40 border-blue-400/30 hover:border-blue-400/60 text-blue-400',
                    emerald: 'bg-emerald-600/20 hover:bg-emerald-600/40 border-emerald-400/30 hover:border-emerald-400/60 text-emerald-400',
                  };
                  const c = colorMap[item.color] ?? '';
                  return (
                    <motion.button
                      key={item.label}
                      onClick={item.onClick}
                      whileHover={{ scale: 1.05, y: -1 }}
                      whileTap={{ scale: 0.95 }}
                      className={`relative flex flex-col items-center gap-0.5 px-2 py-2.5 border rounded-xl transition-all cursor-pointer backdrop-blur-md ${c}`}
                    >
                      {item.icon}
                      <span className="text-[9px] font-bold tracking-wider uppercase opacity-90">{item.label}</span>
                    </motion.button>
                  );
                })}
              </div>

              {/* Subdomain links grid */}
              <div className="border-t border-slate-700/50 pt-2 mb-2">
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider px-1 mb-2">Resources</p>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: 'API Docs', icon: <Globe className="w-5 h-5" />, color: 'cyan', href: 'https://api.quillon.xyz' },
                    { label: 'Source', icon: <Code className="w-5 h-5" />, color: 'purple', href: 'https://code.quillon.xyz' },
                    { label: 'Papers', icon: <BookOpen className="w-5 h-5" />, color: 'amber', onClick: () => { setShowPapersLibrary(true); setShowProfileModal(false); } },
                    { label: 'Deep Dive', icon: <BookOpen className="w-5 h-5" />, color: 'emerald', href: 'https://technical-deepdive.quillon.xyz/' },
                    { label: 'Tor', icon: <span className="text-xl leading-none">🧅</span>, color: 'purple', href: torOnionUrl },
                    { label: 'Download', icon: <Download className="w-5 h-5" />, color: 'blue', onClick: () => { onNavigate?.('download'); setShowProfileModal(false); } },
                  ].map(item => {
                    const colorMap2: {[k:string]:string} = {
                      amber: 'bg-amber-600/15 hover:bg-amber-600/30 border-amber-400/25 hover:border-amber-400/50 text-amber-400',
                      cyan: 'bg-cyan-600/15 hover:bg-cyan-600/30 border-cyan-400/25 hover:border-cyan-400/50 text-cyan-400',
                      purple: 'bg-purple-600/15 hover:bg-purple-600/30 border-purple-400/25 hover:border-purple-400/50 text-purple-400',
                      emerald: 'bg-emerald-600/15 hover:bg-emerald-600/30 border-emerald-400/25 hover:border-emerald-400/50 text-emerald-400',
                      blue: 'bg-blue-600/15 hover:bg-blue-600/30 border-blue-400/25 hover:border-blue-400/50 text-blue-400',
                    };
                    const c = colorMap2[item.color] ?? '';
                    const Wrap = item.href ? 'a' : motion.button as any;
                    const props = item.href
                      ? { href: item.href, target: '_blank', rel: 'noopener noreferrer' }
                      : { onClick: item.onClick, whileHover: { scale: 1.05, y: -1 }, whileTap: { scale: 0.95 } };
                    return (
                      <Wrap
                        key={item.label}
                        {...props}
                        className={`flex flex-col items-center gap-0.5 px-2 py-2.5 border rounded-xl transition-all cursor-pointer backdrop-blur-md ${c}`}
                      >
                        {item.icon}
                        <span className="text-[9px] font-bold tracking-wider uppercase opacity-90">{item.label}</span>
                      </Wrap>
                    );
                  })}
                </div>
              </div>

              {/* Social Links */}
              <div className="border-t border-slate-700/50 mt-1 pt-1 flex items-center gap-1 px-3 py-2">
                <a
                  href="https://x.com/quillongraph"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-white transition-colors"
                  title="X (Twitter)"
                >
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                  </svg>
                </a>
                <a
                  href="https://facebook.com/QuilloNetwork"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-[#1877F2] transition-colors"
                  title="Facebook"
                >
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
                    <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                  </svg>
                </a>
                <a
                  href="https://discord.gg/jEhaYtAhfx"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-[#5865F2] transition-colors"
                  title="Discord"
                >
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                </a>
              </div>

              <div className="border-t border-slate-700/50 mt-1 pt-1">
                {metamaskAddress ? (
                  <button
                    onClick={() => {
                      localStorage.removeItem('metamaskLinked');
                      setMetamaskAddress(null);
                      setMetamaskBalance(null);
                      setMetamaskChainId(null);
                      setMetamaskChainName('');
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-orange-500/10 text-slate-400 hover:text-orange-400 transition-colors text-sm"
                  >
                    <svg width="16" height="16" viewBox="0 0 318.6 318.6" fill="none" className="opacity-60">
                      <path d="M274.1 35.5l-99.5 73.9L193 65.8z" fill="currentColor"/>
                      <path d="M44.4 35.5l98.7 74.6-17.5-44.3z" fill="currentColor"/>
                    </svg>
                    <span>Unlink MetaMask</span>
                  </button>
                ) : (
                  <button
                    onClick={handleLinkMetaMask}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[#E2761B]/10 text-slate-400 hover:text-[#E2761B] transition-colors text-sm"
                  >
                    <svg width="16" height="16" viewBox="0 0 318.6 318.6" fill="none" className="opacity-60">
                      <path d="M274.1 35.5l-99.5 73.9L193 65.8z" fill="currentColor"/>
                      <path d="M44.4 35.5l98.7 74.6-17.5-44.3z" fill="currentColor"/>
                    </svg>
                    <span>Link MetaMask</span>
                  </button>
                )}
                <button
                  onClick={() => {
                    localStorage.removeItem('walletAddress');
                    localStorage.removeItem('cachedBalance');
                    localStorage.removeItem('authToken');
                    localStorage.removeItem('metamaskLinked');
                    window.location.reload();
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-red-500/10 text-slate-400 hover:text-red-400 transition-colors text-sm"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Disconnect Wallet</span>
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
    )}

    {/* v3.4.2: Detail Modal for search results - OUTSIDE TopBar div for proper z-index */}
    <AnimatePresence>
      {selectedDetail && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999] p-4 overflow-y-auto"
          onClick={() => setSelectedDetail(null)}
        >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl p-6 max-w-lg w-full shadow-2xl max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
              style={{ boxShadow: '0 0 40px rgba(212, 175, 55, 0.2)' }}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${selectedDetail.type === 'contract' ? 'bg-purple-500/20' : 'bg-amber-500/20'}`}>
                    {selectedDetail.type === 'transaction' && <Hash className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'block' && <Blocks className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'address' && <User className="w-6 h-6 text-amber-400" />}
                    {selectedDetail.type === 'contract' && <FileCode className="w-6 h-6 text-purple-400" />}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-amber-100">{selectedDetail.title}</h3>
                    <p className="text-amber-300/60 text-sm">{selectedDetail.type.toUpperCase()}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedDetail(null)}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-amber-400" />
                </button>
              </div>

              {/* Transaction Details - Show based on what data is actually available */}
              {selectedDetail.type === 'transaction' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 font-medium capitalize">{selectedDetail.data.status}</span>
                    {selectedDetail.data.confirmations && (
                      <span className="text-green-300/60 text-sm">({selectedDetail.data.confirmations} confirmations)</span>
                    )}
                  </div>

                  <div className="grid gap-3">
                    {/* Check if we have full access (from/to are populated) */}
                    {(() => {
                      // v3.4.3: Fix hasFullAccess check - 'N/A' is truthy but means no access
                      const hasFullAccess = selectedDetail.data.from && selectedDetail.data.to &&
                                           selectedDetail.data.from !== 'N/A' && selectedDetail.data.to !== 'N/A';
                      // v3.4.3: Convert raw amounts to human-readable (1 QUG = 10^24 raw units - 24 decimal precision)
                      const DECIMALS = 1e24;
                      const humanAmount = selectedDetail.data.amount ? selectedDetail.data.amount / DECIMALS : 0;
                      const humanFee = selectedDetail.data.fee ? selectedDetail.data.fee / DECIMALS : 0;
                      return (
                        <>
                          {/* Amount - only show if we have full access AND amount exists */}
                          {hasFullAccess && humanAmount > 0 && (
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <div className="text-amber-300/60 text-xs mb-1">Amount</div>
                              <div className="text-xl font-bold text-amber-100">{humanAmount.toFixed(6)} {TICKER_SYMBOL}</div>
                            </div>
                          )}

                          <div className="p-3 bg-slate-800/50 rounded-lg">
                            <div className="text-amber-300/60 text-xs mb-1">Transaction Hash</div>
                            <div className="flex items-center gap-2">
                              <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.hash}</code>
                              <button
                                onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-hash')}
                                className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                              >
                                {copiedId === 'modal-hash' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                              </button>
                            </div>
                          </div>

                          {/* From/To - only show if we have full access */}
                          {hasFullAccess ? (
                            <div className="grid grid-cols-2 gap-3">
                              <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-amber-300/60 text-xs mb-1">From</div>
                                <code className="text-amber-100 text-xs font-mono break-all">
                                  {selectedDetail.data.from || 'Coinbase (Mining)'}
                                </code>
                              </div>
                              <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-amber-300/60 text-xs mb-1">To</div>
                                <code className="text-amber-100 text-xs font-mono break-all">
                                  {selectedDetail.data.to}
                                </code>
                              </div>
                            </div>
                          ) : null}

                          <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <div className="text-amber-300/60 text-xs mb-1">Block</div>
                              <div className="text-amber-100 font-medium">#{selectedDetail.data.block_height || 'Pending'}</div>
                            </div>
                            {selectedDetail.data.timestamp && (
                              <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-amber-300/60 text-xs mb-1">Time</div>
                                <div className="text-amber-100 font-medium text-xs">{selectedDetail.data.timestamp}</div>
                              </div>
                            )}
                          </div>

                          {/* Fee - only show if we have full access AND fee exists */}
                          {hasFullAccess && humanFee > 0 && (
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <div className="text-amber-300/60 text-xs mb-1">Fee</div>
                              <div className="text-amber-100 font-medium">{humanFee.toFixed(6)} {TICKER_SYMBOL}</div>
                            </div>
                          )}

                          {/* Show appropriate notice based on access level */}
                          {hasFullAccess ? (
                            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                              <div className="flex items-center gap-2 text-amber-300 text-sm">
                                <Key className="w-4 h-4" />
                                <span>Authenticated - full transaction details visible</span>
                              </div>
                            </div>
                          ) : (
                            <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                              <div className="flex items-center gap-2 text-purple-300 text-sm">
                                <Shield className="w-4 h-4" />
                                <span>ZK-STARK Privacy: Only sender/receiver can view full details</span>
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                </div>
              )}

              {/* Block Details */}
              {selectedDetail.type === 'block' && selectedDetail.data && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Block Height</div>
                      <div className="text-xl font-bold text-amber-100">#{selectedDetail.data.height}</div>
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Transactions</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.tx_count}</div>
                    </div>
                  </div>

                  {selectedDetail.data.hash && selectedDetail.data.hash !== 'N/A' && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Block Hash</div>
                      <div className="flex items-center gap-2">
                        <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.hash}</code>
                        <button
                          onClick={() => copyToClipboard(selectedDetail.data.hash, 'modal-block-hash')}
                          className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                        >
                          {copiedId === 'modal-block-hash' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Address Details */}
              {selectedDetail.type === 'address' && selectedDetail.data && (
                <div className="space-y-4">
                  {/* Wallet Badge */}
                  <div className="flex items-center gap-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <User className="w-5 h-5 text-amber-400" />
                    <span className="text-amber-400 font-medium">Wallet Address</span>
                    {selectedDetail.data.balance > 0 && (
                      <span className="ml-auto px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">Active</span>
                    )}
                  </div>

                  {/* Address */}
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-amber-300/60 text-xs mb-1">Address</div>
                    <div className="flex items-center gap-2">
                      <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.address}</code>
                      <button
                        onClick={() => copyToClipboard(selectedDetail.data.address, 'modal-address')}
                        className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                      >
                        {copiedId === 'modal-address' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                      </button>
                    </div>
                  </div>

                  {/* Balance & Nonce */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Balance</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.balance?.toFixed(6)} {TICKER_SYMBOL}</div>
                    </div>
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Nonce</div>
                      <div className="text-xl font-bold text-amber-100">{selectedDetail.data.nonce}</div>
                    </div>
                  </div>

                  {/* Wallet Info Notice */}
                  <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <div className="flex items-center gap-2 text-amber-300 text-sm">
                      <User className="w-4 h-4" />
                      <span>External wallet on Q-NarwhalKnight Network</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Smart Contract Details */}
              {selectedDetail.type === 'contract' && selectedDetail.data && (
                <div className="space-y-4">
                  {/* Contract Badge */}
                  <div className="flex items-center gap-2 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                    <FileCode className="w-5 h-5 text-purple-400" />
                    <span className="text-purple-400 font-medium">{selectedDetail.data.contract_type || 'Smart Contract'}</span>
                    {selectedDetail.data.verified && (
                      <span className="ml-auto px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">Verified</span>
                    )}
                  </div>

                  {/* Contract Address */}
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-amber-300/60 text-xs mb-1">Contract Address</div>
                    <div className="flex items-center gap-2">
                      <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.address}</code>
                      <button
                        onClick={() => copyToClipboard(selectedDetail.data.address, 'modal-contract')}
                        className="p-1 hover:bg-amber-500/20 rounded transition-colors"
                      >
                        {copiedId === 'modal-contract' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-amber-400" />}
                      </button>
                    </div>
                  </div>

                  {/* Token Info (if token contract) */}
                  {selectedDetail.data.symbol && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Token Name</div>
                        <div className="text-lg font-bold text-amber-100">{selectedDetail.data.name}</div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Symbol</div>
                        <div className="text-lg font-bold text-amber-100">{selectedDetail.data.symbol}</div>
                      </div>
                    </div>
                  )}

                  {/* Supply & Decimals */}
                  {selectedDetail.data.total_supply !== undefined && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Total Supply</div>
                        <div className="text-amber-100 font-medium">
                          {Number(selectedDetail.data.total_supply).toLocaleString()}
                        </div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg">
                        <div className="text-amber-300/60 text-xs mb-1">Decimals</div>
                        <div className="text-amber-100 font-medium">{selectedDetail.data.decimals}</div>
                      </div>
                    </div>
                  )}

                  {/* Deployer & Block */}
                  {selectedDetail.data.deployer && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Deployed By</div>
                      <code className="text-amber-100 text-xs font-mono break-all">{selectedDetail.data.deployer}</code>
                    </div>
                  )}

                  {selectedDetail.data.deployment_height && (
                    <div className="p-3 bg-slate-800/50 rounded-lg">
                      <div className="text-amber-300/60 text-xs mb-1">Deployment Block</div>
                      <div className="text-amber-100 font-medium">#{selectedDetail.data.deployment_height.toLocaleString()}</div>
                    </div>
                  )}

                  {/* Contract Info Notice */}
                  <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                    <div className="flex items-center gap-2 text-purple-300 text-sm">
                      <FileCode className="w-4 h-4" />
                      <span>Smart Contract on Q-NarwhalKnight Network</span>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* v3.4.20: Enhanced Smart Contract Modal (Polygonscan-inspired) */}
      <SmartContractModal
        isOpen={!!selectedContract}
        onClose={() => setSelectedContract(null)}
        contractData={selectedContract || {}}
      />

      {/* v3.4.21: P2P Network Map Modal with Tor visualization */}
      <NetworkMapModal
        isOpen={showNetworkMap}
        onClose={() => setShowNetworkMap(false)}
        peers={livePeers}
        blockHeight={liveBlockHeight}
      />

      {/* v5.7.0: Theme Chooser Modal */}
      <ThemeChooserModal
        isOpen={showThemeChooser}
        onClose={() => setShowThemeChooser(false)}
      />

      {/* Miner Link Modal — opened from hashrate/miner badges */}
      <MinerLinkModal
        isOpen={showMinerLinkModal}
        onClose={() => setShowMinerLinkModal(false)}
        minerLink={minerLink}
      />

      {/* Research Library Modal — 78 whitepapers organized by category */}
      <PapersLibraryModal
        isOpen={showPapersLibrary}
        onClose={() => setShowPapersLibrary(false)}
      />

      {/* Tax Report Modal */}
      {showTaxModal && createPortal(
        <TaxReportModal
          isOpen={showTaxModal}
          onClose={() => setShowTaxModal(false)}
          walletAddress={walletAddr}
          currentBalance={currentBalance}
        />,
        document.body
      )}
    </>
  );
});

// =====================================================
// TAX REPORT MODAL — Multi-jurisdiction PDF generator
// =====================================================

interface TaxJurisdiction {
  id: string;
  name: string;
  flag: string;
  shortGainRate: number;  // % for < 1 year
  longGainRate: number;   // % for >= 1 year
  freeAllowance: number;  // tax-free threshold in USD
  currency: string;
  currencySymbol: string;
}

const JURISDICTIONS: TaxJurisdiction[] = [
  { id: 'us',  name: 'United States',  flag: '\u{1F1FA}\u{1F1F8}', shortGainRate: 37, longGainRate: 20, freeAllowance: 0,     currency: 'USD', currencySymbol: '$' },
  { id: 'uk',  name: 'United Kingdom', flag: '\u{1F1EC}\u{1F1E7}', shortGainRate: 20, longGainRate: 20, freeAllowance: 6000,  currency: 'GBP', currencySymbol: '\u00A3' },
  { id: 'de',  name: 'Germany',        flag: '\u{1F1E9}\u{1F1EA}', shortGainRate: 26.375, longGainRate: 0, freeAllowance: 600, currency: 'EUR', currencySymbol: '\u20AC' },
  { id: 'nl',  name: 'Netherlands',    flag: '\u{1F1F3}\u{1F1F1}', shortGainRate: 36, longGainRate: 36, freeAllowance: 57000, currency: 'EUR', currencySymbol: '\u20AC' },
  { id: 'au',  name: 'Australia',      flag: '\u{1F1E6}\u{1F1FA}', shortGainRate: 45, longGainRate: 22.5, freeAllowance: 0,   currency: 'AUD', currencySymbol: 'A$' },
  { id: 'ca',  name: 'Canada',         flag: '\u{1F1E8}\u{1F1E6}', shortGainRate: 26.76, longGainRate: 26.76, freeAllowance: 0, currency: 'CAD', currencySymbol: 'C$' },
  { id: 'jp',  name: 'Japan',          flag: '\u{1F1EF}\u{1F1F5}', shortGainRate: 55, longGainRate: 55, freeAllowance: 200000, currency: 'JPY', currencySymbol: '\u00A5' },
  { id: 'sg',  name: 'Singapore',      flag: '\u{1F1F8}\u{1F1EC}', shortGainRate: 0, longGainRate: 0, freeAllowance: 0,       currency: 'SGD', currencySymbol: 'S$' },
  { id: 'ae',  name: 'UAE',            flag: '\u{1F1E6}\u{1F1EA}', shortGainRate: 0, longGainRate: 0, freeAllowance: 0,       currency: 'AED', currencySymbol: 'AED' },
  { id: 'pt',  name: 'Portugal',       flag: '\u{1F1F5}\u{1F1F9}', shortGainRate: 28, longGainRate: 28, freeAllowance: 0,     currency: 'EUR', currencySymbol: '\u20AC' },
];

const TRANSLATIONS: Record<string, Record<string, string>> = {
  en: { title: 'Crypto Tax Report', jurisdiction: 'Jurisdiction', taxYear: 'Tax Year', holdings: 'Current Holdings', costBasis: 'Cost Basis (USD)', currentValue: 'Current Value', unrealizedGain: 'Unrealized Gain/Loss', shortTermRate: 'Short-Term Rate', longTermRate: 'Long-Term Rate', freeAllowance: 'Tax-Free Allowance', estimatedTax: 'Estimated Tax', generate: 'Generate PDF Report', disclaimer: 'This is an estimate only. Consult a tax professional for official filing.', noTax: 'No Capital Gains Tax', totalTransactions: 'Total Transactions', miningIncome: 'Mining Income', wallet: 'Wallet', summary: 'Summary', period: 'Period' },
  de: { title: 'Krypto-Steuerbericht', jurisdiction: 'Steuergebiet', taxYear: 'Steuerjahr', holdings: 'Aktueller Bestand', costBasis: 'Anschaffungskosten (EUR)', currentValue: 'Aktueller Wert', unrealizedGain: 'Unrealisierter Gewinn/Verlust', shortTermRate: 'Kurzfristiger Steuersatz', longTermRate: 'Langfristiger Steuersatz', freeAllowance: 'Steuerfreier Freibetrag', estimatedTax: 'Geschätzte Steuer', generate: 'PDF-Bericht erstellen', disclaimer: 'Dies ist nur eine Schätzung. Konsultieren Sie einen Steuerberater.', noTax: 'Keine Kapitalertragssteuer', totalTransactions: 'Gesamttransaktionen', miningIncome: 'Mining-Einkommen', wallet: 'Wallet', summary: 'Zusammenfassung', period: 'Zeitraum' },
  nl: { title: 'Crypto Belastingrapport', jurisdiction: 'Jurisdictie', taxYear: 'Belastingjaar', holdings: 'Huidige Bezittingen', costBasis: 'Kostprijs (EUR)', currentValue: 'Huidige Waarde', unrealizedGain: 'Ongerealiseerde Winst/Verlies', shortTermRate: 'Korte Termijn Tarief', longTermRate: 'Lange Termijn Tarief', freeAllowance: 'Belastingvrije Drempel', estimatedTax: 'Geschatte Belasting', generate: 'PDF Rapport Genereren', disclaimer: 'Dit is slechts een schatting. Raadpleeg een belastingadviseur.', noTax: 'Geen Vermogenswinstbelasting', totalTransactions: 'Totaal Transacties', miningIncome: 'Mining Inkomsten', wallet: 'Portemonnee', summary: 'Samenvatting', period: 'Periode' },
  ja: { title: '\u6697\u53F7\u8CC7\u7523\u7A0E\u52D9\u30EC\u30DD\u30FC\u30C8', jurisdiction: '\u7BA1\u8F44\u5730\u57DF', taxYear: '\u8AB2\u7A0E\u5E74\u5EA6', holdings: '\u4FDD\u6709\u8CC7\u7523', costBasis: '\u53D6\u5F97\u4FA1\u683C', currentValue: '\u73FE\u5728\u4FA1\u5024', unrealizedGain: '\u542B\u307F\u640D\u76CA', shortTermRate: '\u77ED\u671F\u7A0E\u7387', longTermRate: '\u9577\u671F\u7A0E\u7387', freeAllowance: '\u975E\u8AB2\u7A0E\u67A0', estimatedTax: '\u63A8\u5B9A\u7A0E\u984D', generate: 'PDF\u30EC\u30DD\u30FC\u30C8\u751F\u6210', disclaimer: '\u3053\u308C\u306F\u63A8\u5B9A\u5024\u3067\u3059\u3002\u7A0E\u7406\u58EB\u306B\u3054\u76F8\u8AC7\u304F\u3060\u3055\u3044\u3002', noTax: '\u30AD\u30E3\u30D4\u30BF\u30EB\u30B2\u30A4\u30F3\u7A0E\u306A\u3057', totalTransactions: '\u7DCF\u53D6\u5F15\u6570', miningIncome: '\u30DE\u30A4\u30CB\u30F3\u30B0\u53CE\u5165', wallet: '\u30A6\u30A9\u30EC\u30C3\u30C8', summary: '\u6982\u8981', period: '\u671F\u9593' },
  fr: { title: 'Rapport Fiscal Crypto', jurisdiction: 'Juridiction', taxYear: 'Ann\u00E9e Fiscale', holdings: 'Avoirs Actuels', costBasis: 'Co\u00FBt de Base (EUR)', currentValue: 'Valeur Actuelle', unrealizedGain: 'Gain/Perte Non R\u00E9alis\u00E9', shortTermRate: 'Taux Court Terme', longTermRate: 'Taux Long Terme', freeAllowance: 'Abattement Fiscal', estimatedTax: 'Imp\u00F4t Estim\u00E9', generate: 'G\u00E9n\u00E9rer le Rapport PDF', disclaimer: 'Ceci est une estimation. Consultez un fiscaliste.', noTax: 'Pas d\'imp\u00F4t sur les plus-values', totalTransactions: 'Total des Transactions', miningIncome: 'Revenus du Minage', wallet: 'Portefeuille', summary: 'R\u00E9sum\u00E9', period: 'P\u00E9riode' },
};

function TaxReportModal({ isOpen, onClose, walletAddress, currentBalance }: { isOpen: boolean; onClose: () => void; walletAddress: string; currentBalance: number }) {
  const [jurisdiction, setJurisdiction] = useState<TaxJurisdiction>(JURISDICTIONS[0]);
  const [taxYear, setTaxYear] = useState(new Date().getFullYear());
  const [costBasis, setCostBasis] = useState(0);
  const [qugPrice, setQugPrice] = useState(0.0001);
  const [language, setLanguage] = useState('en');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showJurisdictions, setShowJurisdictions] = useState(false);
  // v8.5.5: Enhanced tax report state
  const [costBasisMethod, setCostBasisMethod] = useState<'fifo' | 'lifo'>('fifo');
  const [miningIncome, setMiningIncome] = useState(0);
  const [yieldIncome, setYieldIncome] = useState(0);
  const [holdingPeriod, setHoldingPeriod] = useState<'short' | 'long'>('short');
  const [activeTab, setActiveTab] = useState<'capital' | 'income'>('capital');

  const t = TRANSLATIONS[language] || TRANSLATIONS.en;
  const currentValue = currentBalance * qugPrice;
  const unrealizedGain = currentValue - costBasis;
  const gainRate = holdingPeriod === 'long' ? jurisdiction.longGainRate : jurisdiction.shortGainRate;
  const taxableGain = Math.max(0, unrealizedGain - jurisdiction.freeAllowance);
  const capitalGainsTax = taxableGain * (gainRate / 100);
  // Income tax on mining + yield (treated as ordinary income at short-term rate)
  const totalIncome = (miningIncome + yieldIncome) * qugPrice;
  const incomeTax = totalIncome * (jurisdiction.shortGainRate / 100);
  const estimatedTax = capitalGainsTax + incomeTax;

  const generatePDF = async () => {
    setIsGenerating(true);
    await new Promise(r => setTimeout(r, 800));

    const doc = document.createElement('div');
    const cs = jurisdiction.currencySymbol;
    const fmtNum = (n: number) => n.toLocaleString(language === 'de' || language === 'nl' ? 'de-DE' : language === 'ja' ? 'ja-JP' : language === 'fr' ? 'fr-FR' : 'en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    doc.innerHTML = `
      <div style="font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 48px; color: #0f172a; background: white;">
        <div style="text-align: center; margin-bottom: 40px; border-bottom: 3px solid #10b981; padding-bottom: 24px;">
          <h1 style="font-size: 28px; font-weight: 800; color: #064e3b; margin: 0 0 8px 0; letter-spacing: -0.5px;">${jurisdiction.flag} ${t.title}</h1>
          <p style="color: #6b7280; font-size: 14px; margin: 0;">Quillon Network (${TICKER_SYMBOL}) | ${t.period}: ${taxYear}</p>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 32px;">
          <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 20px; border-radius: 12px; border: 1px solid #a7f3d0;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #059669; font-weight: 600;">${t.jurisdiction}</div>
            <div style="font-size: 20px; font-weight: 700; color: #064e3b; margin-top: 4px;">${jurisdiction.flag} ${jurisdiction.name}</div>
          </div>
          <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 20px; border-radius: 12px; border: 1px solid #93c5fd;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #2563eb; font-weight: 600;">${t.wallet}</div>
            <div style="font-size: 13px; font-weight: 600; color: #1e3a5f; margin-top: 4px; word-break: break-all; font-family: monospace;">${walletAddress.slice(0, 12)}...${walletAddress.slice(-8)}</div>
          </div>
        </div>

        <h2 style="font-size: 18px; font-weight: 700; color: #064e3b; margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid #d1fae5;">${t.summary}</h2>

        <table style="width: 100%; border-collapse: collapse; margin-bottom: 32px;">
          <tbody>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.holdings}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${fmtNum(currentBalance)} ${TICKER_SYMBOL}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.costBasis}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${cs}${fmtNum(costBasis)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.currentValue}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${cs}${fmtNum(currentValue)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.unrealizedGain}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: ${unrealizedGain >= 0 ? '#059669' : '#dc2626'};">${unrealizedGain >= 0 ? '+' : ''}${cs}${fmtNum(unrealizedGain)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.shortTermRate}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${jurisdiction.shortGainRate}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.longTermRate}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${jurisdiction.longGainRate === 0 ? t.noTax : jurisdiction.longGainRate + '%'}</td>
            </tr>
            ${jurisdiction.freeAllowance > 0 ? `
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">${t.freeAllowance}</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #059669;">${cs}${fmtNum(jurisdiction.freeAllowance)}</td>
            </tr>` : ''}
          </tbody>
        </table>

        ${(miningIncome > 0 || yieldIncome > 0) ? `
        <h2 style="font-size: 18px; font-weight: 700; color: #064e3b; margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid #d1fae5;">Income Summary</h2>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 32px;">
          <tbody>
            ${miningIncome > 0 ? `<tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">Mining Rewards</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${fmtNum(miningIncome)} ${TICKER_SYMBOL} (${cs}${fmtNum(miningIncome * qugPrice)})</td>
            </tr>` : ''}
            ${yieldIncome > 0 ? `<tr style="border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">QCREDIT Yield</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #0f172a;">${fmtNum(yieldIncome)} ${TICKER_SYMBOL} (${cs}${fmtNum(yieldIncome * qugPrice)})</td>
            </tr>` : ''}
            <tr style="border-bottom: 1px solid #e5e7eb;">
              <td style="padding: 14px 16px; font-weight: 500; color: #374151;">Income Tax (${jurisdiction.shortGainRate}%)</td>
              <td style="padding: 14px 16px; text-align: right; font-weight: 700; font-family: monospace; color: #dc2626;">${cs}${fmtNum(incomeTax)}</td>
            </tr>
          </tbody>
        </table>` : ''}

        <div style="background: linear-gradient(135deg, ${estimatedTax === 0 ? '#ecfdf5, #d1fae5' : '#fef2f2, #fecaca'}); padding: 24px; border-radius: 16px; text-align: center; margin-bottom: 32px; border: 2px solid ${estimatedTax === 0 ? '#10b981' : '#f87171'};">
          <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: ${estimatedTax === 0 ? '#059669' : '#dc2626'}; font-weight: 600; margin-bottom: 8px;">${t.estimatedTax} (Capital Gains + Income)</div>
          <div style="font-size: 36px; font-weight: 800; color: ${estimatedTax === 0 ? '#064e3b' : '#991b1b'};">${estimatedTax === 0 ? t.noTax : cs + fmtNum(estimatedTax)}</div>
          <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Cost Basis: ${costBasisMethod.toUpperCase()} | Holding: ${holdingPeriod === 'long' ? '>1 year' : '<1 year'} (${gainRate}%)</div>
        </div>

        <div style="background: #fffbeb; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 24px;">
          <p style="margin: 0; font-size: 12px; color: #92400e; line-height: 1.5;">${t.disclaimer}</p>
        </div>

        <div style="text-align: center; color: #9ca3af; font-size: 11px; margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb;">
          Generated by Quillon Network Tax Calculator | ${new Date().toLocaleDateString()} | quillon.xyz
        </div>
      </div>
    `;

    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(`<!DOCTYPE html><html><head><title>${t.title} - ${taxYear}</title><style>@media print { body { margin: 0; } @page { size: A4; margin: 20mm; } }</style></head><body>${doc.innerHTML}</body></html>`);
      printWindow.document.close();
      setTimeout(() => { printWindow.print(); }, 500);
    }

    setIsGenerating(false);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[10000] overflow-y-auto py-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 25 }}
          className="relative bg-gradient-to-br from-slate-900 via-emerald-950/30 to-slate-900 border-2 border-emerald-500/40 rounded-2xl p-6 max-w-xl w-full mx-4 shadow-2xl overflow-hidden"
          onClick={(e) => e.stopPropagation()}
          style={{ boxShadow: '0 0 60px rgba(16, 185, 129, 0.2)' }}
        >
          {/* Background orb */}
          <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-500/[0.06] rounded-full blur-3xl -translate-y-1/3 translate-x-1/3 pointer-events-none" />

          {/* Header */}
          <div className="relative flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500/30 to-teal-500/30 border border-emerald-400/40 flex items-center justify-center">
                <FileText className="w-6 h-6 text-emerald-300" />
              </div>
              <div>
                <h2 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">{t.title}</h2>
                <p className="text-emerald-300/50 text-xs">{TICKER_SYMBOL} | {t.period} {taxYear}</p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-emerald-500/20 rounded-lg transition-colors">
              <X className="w-5 h-5 text-emerald-400" />
            </button>
          </div>

          {/* Language Selector */}
          <div className="relative flex gap-1 mb-4 p-1 bg-slate-800/60 rounded-lg border border-slate-700/40">
            {[
              { code: 'en', label: 'EN' },
              { code: 'de', label: 'DE' },
              { code: 'nl', label: 'NL' },
              { code: 'fr', label: 'FR' },
              { code: 'ja', label: 'JA' },
            ].map(lang => (
              <button
                key={lang.code}
                onClick={() => setLanguage(lang.code)}
                className={`flex-1 py-1.5 rounded text-xs font-bold transition-all ${language === lang.code ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/30' : 'text-slate-500 hover:text-slate-300'}`}
              >
                {lang.label}
              </button>
            ))}
          </div>

          {/* Jurisdiction Selector */}
          <div className="relative mb-4">
            <button
              onClick={() => setShowJurisdictions(!showJurisdictions)}
              className="w-full flex items-center justify-between p-3 bg-slate-800/60 border border-emerald-500/20 rounded-xl hover:border-emerald-500/40 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="text-xl">{jurisdiction.flag}</span>
                <div className="text-left">
                  <div className="text-sm font-bold text-emerald-100">{jurisdiction.name}</div>
                  <div className="text-[10px] text-emerald-400/50">
                    {jurisdiction.shortGainRate === 0 && jurisdiction.longGainRate === 0 ? t.noTax : `${jurisdiction.shortGainRate}% / ${jurisdiction.longGainRate}%`}
                  </div>
                </div>
              </div>
              <ChevronDown className={`w-4 h-4 text-emerald-400 transition-transform ${showJurisdictions ? 'rotate-180' : ''}`} />
            </button>
            <AnimatePresence>
              {showJurisdictions && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scaleY: 0.9 }}
                  animate={{ opacity: 1, y: 0, scaleY: 1 }}
                  exit={{ opacity: 0, y: -10, scaleY: 0.9 }}
                  className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-emerald-500/20 rounded-xl overflow-hidden z-10 max-h-52 overflow-y-auto shadow-xl"
                  style={{ transformOrigin: 'top' }}
                >
                  {JURISDICTIONS.map(j => (
                    <button
                      key={j.id}
                      onClick={() => { setJurisdiction(j); setShowJurisdictions(false); }}
                      className={`w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-emerald-500/10 transition-colors ${j.id === jurisdiction.id ? 'bg-emerald-500/20 text-emerald-200' : 'text-slate-300'}`}
                    >
                      <span className="text-lg">{j.flag}</span>
                      <span className="text-sm flex-1">{j.name}</span>
                      <span className="text-xs text-emerald-400/50">
                        {j.shortGainRate === 0 && j.longGainRate === 0 ? t.noTax : `${j.shortGainRate}%`}
                      </span>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Tax Year & Cost Basis Inputs */}
          <div className="relative grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">{t.taxYear}</label>
              <select
                value={taxYear}
                onChange={(e) => setTaxYear(Number(e.target.value))}
                className="w-full p-2.5 bg-slate-800/60 border border-emerald-500/20 rounded-lg text-emerald-100 text-sm focus:border-emerald-400/50 focus:outline-none"
              >
                {[2026, 2025, 2024, 2023].map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">{t.costBasis}</label>
              <input
                type="number"
                value={costBasis || ''}
                onChange={(e) => setCostBasis(Number(e.target.value))}
                placeholder="0.00"
                className="w-full p-2.5 bg-slate-800/60 border border-emerald-500/20 rounded-lg text-emerald-100 text-sm focus:border-emerald-400/50 focus:outline-none font-mono placeholder:text-slate-600"
              />
            </div>
          </div>

          {/* Price Input */}
          <div className="relative mb-4">
            <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">{TICKER_SYMBOL} Price (USD)</label>
            <input
              type="number"
              value={qugPrice}
              onChange={(e) => setQugPrice(Number(e.target.value))}
              step="0.0001"
              className="w-full p-2.5 bg-slate-800/60 border border-emerald-500/20 rounded-lg text-emerald-100 text-sm focus:border-emerald-400/50 focus:outline-none font-mono"
            />
          </div>

          {/* v8.5.5: Tab selector — Capital Gains vs Income */}
          <div className="relative flex gap-1 mb-4 p-1 bg-slate-800/60 rounded-lg border border-slate-700/40">
            {[
              { id: 'capital' as const, label: 'Capital Gains' },
              { id: 'income' as const, label: 'Income (Mining/Yield)' },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 py-1.5 rounded text-xs font-bold transition-all ${activeTab === tab.id ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/30' : 'text-slate-500 hover:text-slate-300'}`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === 'capital' && (
            <>
              {/* Cost Basis Method + Holding Period */}
              <div className="relative grid grid-cols-2 gap-3 mb-4">
                <div>
                  <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">Cost Basis Method</label>
                  <div className="flex gap-1">
                    {(['fifo', 'lifo'] as const).map(method => (
                      <button
                        key={method}
                        onClick={() => setCostBasisMethod(method)}
                        className={`flex-1 py-2 rounded text-xs font-bold transition-all ${costBasisMethod === method ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/30' : 'bg-slate-800/60 text-slate-500 border border-slate-700/30'}`}
                      >
                        {method.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">Holding Period</label>
                  <div className="flex gap-1">
                    {([{ id: 'short' as const, label: '<1yr' }, { id: 'long' as const, label: '>1yr' }]).map(hp => (
                      <button
                        key={hp.id}
                        onClick={() => setHoldingPeriod(hp.id)}
                        className={`flex-1 py-2 rounded text-xs font-bold transition-all ${holdingPeriod === hp.id ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/30' : 'bg-slate-800/60 text-slate-500 border border-slate-700/30'}`}
                      >
                        {hp.label} ({hp.id === 'short' ? jurisdiction.shortGainRate : jurisdiction.longGainRate}%)
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === 'income' && (
            <>
              {/* Mining + Yield Income */}
              <div className="relative grid grid-cols-2 gap-3 mb-4">
                <div>
                  <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">Mining Rewards ({TICKER_SYMBOL})</label>
                  <input
                    type="number"
                    value={miningIncome || ''}
                    onChange={(e) => setMiningIncome(Number(e.target.value))}
                    placeholder="0.00"
                    className="w-full p-2.5 bg-slate-800/60 border border-emerald-500/20 rounded-lg text-emerald-100 text-sm focus:border-emerald-400/50 focus:outline-none font-mono placeholder:text-slate-600"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-emerald-400/60 uppercase tracking-wider font-semibold mb-1 block">QCREDIT Yield ({TICKER_SYMBOL})</label>
                  <input
                    type="number"
                    value={yieldIncome || ''}
                    onChange={(e) => setYieldIncome(Number(e.target.value))}
                    placeholder="0.00"
                    className="w-full p-2.5 bg-slate-800/60 border border-emerald-500/20 rounded-lg text-emerald-100 text-sm focus:border-emerald-400/50 focus:outline-none font-mono placeholder:text-slate-600"
                  />
                </div>
              </div>
              {totalIncome > 0 && (
                <div className="relative p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-amber-300/70">Total Income (USD)</span>
                    <span className="text-amber-200 font-bold font-mono">{jurisdiction.currencySymbol}{totalIncome.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                  </div>
                  <div className="flex justify-between text-sm mt-1">
                    <span className="text-amber-300/70">Income Tax ({jurisdiction.shortGainRate}%)</span>
                    <span className="text-amber-200 font-bold font-mono">{jurisdiction.currencySymbol}{incomeTax.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Results Summary */}
          <div className="relative grid grid-cols-2 gap-3 mb-4">
            <div className="p-3 bg-slate-800/40 border border-slate-700/30 rounded-xl">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">{t.holdings}</div>
              <div className="text-lg font-bold text-white font-mono">{currentBalance.toLocaleString()} <span className="text-xs text-amber-400">{TICKER_SYMBOL}</span></div>
            </div>
            <div className="p-3 bg-slate-800/40 border border-slate-700/30 rounded-xl">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">{t.currentValue}</div>
              <div className="text-lg font-bold text-white font-mono">{jurisdiction.currencySymbol}{currentValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
            </div>
          </div>

          {/* Gain/Loss & Tax */}
          <div className="relative p-4 rounded-xl mb-4 border-2" style={{ background: unrealizedGain >= 0 ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)', borderColor: unrealizedGain >= 0 ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">{t.unrealizedGain}</span>
              <span className={`text-xl font-bold font-mono ${unrealizedGain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {unrealizedGain >= 0 ? '+' : ''}{jurisdiction.currencySymbol}{unrealizedGain.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">{t.estimatedTax}</span>
              <span className={`text-xl font-bold font-mono ${estimatedTax === 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {estimatedTax === 0 ? t.noTax : `${jurisdiction.currencySymbol}${estimatedTax.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
              </span>
            </div>
          </div>

          {/* Disclaimer */}
          <p className="relative text-[10px] text-amber-400/50 mb-4 leading-relaxed">{t.disclaimer}</p>

          {/* Generate PDF Button */}
          <motion.button
            onClick={generatePDF}
            disabled={isGenerating}
            className="relative w-full py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 rounded-xl text-white font-bold text-sm flex items-center justify-center gap-2 disabled:opacity-50 transition-all"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isGenerating ? (
              <motion.div
                className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
              />
            ) : (
              <Download className="w-4 h-4" />
            )}
            {isGenerating ? 'Generating...' : t.generate}
          </motion.button>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

export default TopBar;