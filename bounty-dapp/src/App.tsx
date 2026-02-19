import { BrowserRouter as Router, Routes, Route, Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Trophy, Award, Bug, Share2, BarChart3, Shield, Zap, Globe, ChevronRight, ExternalLink, ArrowRight, Users, Target, Clock, Gem, Wallet, LogOut } from 'lucide-react'
import { useEffect, useState, useRef, createContext, useContext, useCallback } from 'react'
import Dashboard from './pages/Dashboard'
import Register from './pages/Register'
import Leaderboard from './pages/Leaderboard'
import BugReports from './pages/BugReports'
import SocialActivity from './pages/SocialActivity'
import { startOAuth2Connect, handleOAuth2Callback, getStoredWalletSession, disconnectWallet, type WalletSession } from './services/api'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 2,
    },
  },
})

/* ── Wallet Auth Context ── */
interface WalletContextType {
  wallet: WalletSession | null
  connect: () => Promise<void>
  disconnect: () => void
  connecting: boolean
  completeOAuth2: (code: string, state: string) => Promise<void>
}

const WalletContext = createContext<WalletContextType>({
  wallet: null,
  connect: async () => {},
  disconnect: () => {},
  connecting: false,
  completeOAuth2: async () => {},
})

export function useWallet() {
  return useContext(WalletContext)
}

function WalletProvider({ children }: { children: React.ReactNode }) {
  const [wallet, setWallet] = useState<WalletSession | null>(() => getStoredWalletSession())
  const [connecting, setConnecting] = useState(false)

  // Called by the /callback route after OAuth2 redirect
  const completeOAuth2 = useCallback(async (code: string, state: string) => {
    setConnecting(true)
    try {
      const session = await handleOAuth2Callback(code, state)
      setWallet(session)
    } finally {
      setConnecting(false)
    }
  }, [])

  const connect = useCallback(async () => {
    setConnecting(true)
    try {
      await startOAuth2Connect() // Redirects — does not return
    } catch {
      setConnecting(false)
    }
  }, [])

  const disconnect = useCallback(() => {
    disconnectWallet()
    setWallet(null)
  }, [])

  return (
    <WalletContext.Provider value={{ wallet, connect, disconnect, connecting, completeOAuth2 }}>
      {children}
    </WalletContext.Provider>
  )
}

/* ── Animated counter hook ── */
function useCounter(end: number, duration = 2000) {
  const [count, setCount] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const started = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true
          const start = 0
          const startTime = performance.now()
          const tick = (now: number) => {
            const elapsed = now - startTime
            const progress = Math.min(elapsed / duration, 1)
            const eased = 1 - Math.pow(1 - progress, 3) // easeOutCubic
            setCount(Math.floor(start + (end - start) * eased))
            if (progress < 1) requestAnimationFrame(tick)
          }
          requestAnimationFrame(tick)
        }
      },
      { threshold: 0.3 }
    )
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [end, duration])

  return { count, ref }
}

/* ── Landing page ── */
function LandingPage() {
  const stats = [
    { label: 'Daily Reward Pool', value: 306, suffix: '', prefix: '$', icon: <Gem className="w-5 h-5" /> },
    { label: 'Activity Categories', value: 5, suffix: '', prefix: '', icon: <Target className="w-5 h-5" /> },
    { label: 'Campaign Duration', value: 90, suffix: ' Days', prefix: '', icon: <Clock className="w-5 h-5" /> },
    { label: 'Max Multiplier', value: 2, suffix: 'x', prefix: '', icon: <Zap className="w-5 h-5" /> },
  ]

  return (
    <div className="space-y-24">
      {/* Hero */}
      <section className="relative pt-12 pb-20 text-center">
        {/* Glowing orb behind title */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-gradient-to-br from-purple-600/20 via-blue-500/10 to-cyan-400/15 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 text-sm font-medium mb-8">
            <div className="pulse-dot" />
            Mainnet Bounty &mdash; Earning Rewards Now
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            <span className="gradient-text">Mainnet Bounty</span>
            <br />
            <span className="text-white/90">Program</span>
          </h1>

          <p className="max-w-2xl mx-auto text-lg md:text-xl text-slate-400 leading-relaxed mb-10">
            Contribute to Q-NarwhalKnight's quantum-resistant blockchain and earn up to $27,500+ in QUG rewards over 90 days.
            Run nodes, report bugs, create content, and climb the leaderboard.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/register"
              className="btn-shine group inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:from-purple-500 hover:to-blue-500 transition-all shadow-lg shadow-purple-500/25"
            >
              Join the Bounty
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/leaderboard"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white/5 border border-white/10 text-white font-semibold rounded-xl hover:bg-white/10 transition-all"
            >
              View Leaderboard
              <ExternalLink className="w-4 h-4 opacity-50" />
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
        {stats.map((stat) => {
          const counter = useCounter(stat.value)
          return (
            <div
              key={stat.label}
              ref={counter.ref}
              className="glass-card p-6 text-center"
            >
              <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-purple-500/10 text-purple-400 mb-4">
                {stat.icon}
              </div>
              <div className="text-3xl md:text-4xl font-bold text-white stat-number mb-1">
                {stat.prefix}{counter.count.toLocaleString()}{stat.suffix}
              </div>
              <div className="text-sm text-slate-400">{stat.label}</div>
            </div>
          )
        })}
      </section>

      {/* How It Works */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">How It Works</h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Three simple steps to start earning rewards for your contributions
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              step: '01',
              title: 'Register',
              desc: 'Connect your testnet wallet address to create your bounty profile. Link your mainnet address for seamless reward distribution.',
              icon: <Users className="w-6 h-6" />,
              color: 'from-purple-500 to-purple-700',
            },
            {
              step: '02',
              title: 'Contribute',
              desc: 'Run a node, report bugs, create educational content, engage on social media, or use the DEX. Every action earns points.',
              icon: <Zap className="w-6 h-6" />,
              color: 'from-blue-500 to-blue-700',
            },
            {
              step: '03',
              title: 'Earn Rewards',
              desc: 'Climb the leaderboard and unlock higher tiers. Pioneers (top 1%) earn 20% of the $306/day reward pool with early participation bonuses.',
              icon: <Trophy className="w-6 h-6" />,
              color: 'from-cyan-500 to-cyan-700',
            },
          ].map((item) => (
            <div key={item.step} className="glass-card p-8 group hover:translate-y-[-4px] transition-all duration-300">
              <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br ${item.color} text-white mb-6 group-hover:scale-110 transition-transform`}>
                {item.icon}
              </div>
              <div className="text-xs font-bold text-purple-400 uppercase tracking-widest mb-2">Step {item.step}</div>
              <h3 className="text-xl font-bold text-white mb-3">{item.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Scoring Categories */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Scoring Categories</h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Your total score is calculated across five weighted categories
          </p>
        </div>

        <div className="grid md:grid-cols-5 gap-4">
          {[
            { name: 'Node Operations', weight: '30%', max: 300, desc: 'Uptime, blocks produced, governance votes', color: 'border-purple-500/40 hover:border-purple-400' },
            { name: 'Transactions', weight: '25%', max: 250, desc: 'Volume, diversity, DEX swaps, contract calls', color: 'border-blue-500/40 hover:border-blue-400' },
            { name: 'Bug Reports', weight: '20%', max: 200, desc: 'Critical (100pts), High (50pts), Medium (20pts), Low (10pts)', color: 'border-cyan-500/40 hover:border-cyan-400' },
            { name: 'Community', weight: '15%', max: 150, desc: 'Documentation, tutorials, translations, tools', color: 'border-teal-500/40 hover:border-teal-400' },
            { name: 'Social', weight: '10%', max: 100, desc: 'Tweets, threads, videos, articles about QNK', color: 'border-pink-500/40 hover:border-pink-400' },
          ].map((cat) => (
            <div key={cat.name} className={`glass-card p-5 border ${cat.color} transition-all duration-300`}>
              <div className="text-2xl font-extrabold gradient-text mb-1">{cat.weight}</div>
              <h4 className="text-white font-semibold text-sm mb-2">{cat.name}</h4>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">{cat.desc}</p>
              <div className="text-xs text-slate-500">Max: {cat.max} pts</div>
            </div>
          ))}
        </div>
      </section>

      {/* Tier System */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Reward Tiers</h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Your tier determines your share of the $306/day reward pool (~$27,500 over 90 days)
          </p>
        </div>

        <div className="grid md:grid-cols-4 gap-6">
          {[
            { tier: 'Pioneer', rank: 'Top 1%', pool: '20%', usd: '~$5,500', vesting: '6 months', gradient: 'from-yellow-400 to-amber-600', glow: 'tier-pioneer', badge: 'bg-yellow-500/15 border-yellow-500/30 text-yellow-300' },
            { tier: 'Contributor', rank: 'Top 10%', pool: '30%', usd: '~$8,250', vesting: '3 months', gradient: 'from-blue-400 to-indigo-600', glow: 'tier-contributor', badge: 'bg-blue-500/15 border-blue-500/30 text-blue-300' },
            { tier: 'Participant', rank: 'Top 50%', pool: '40%', usd: '~$11,000', vesting: '1 month', gradient: 'from-cyan-400 to-teal-600', glow: 'tier-participant', badge: 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300' },
            { tier: 'Supporter', rank: 'All valid', pool: '10%', usd: '~$2,750', vesting: 'None', gradient: 'from-purple-400 to-violet-600', glow: 'tier-supporter', badge: 'bg-purple-500/15 border-purple-500/30 text-purple-300' },
          ].map((t) => (
            <div key={t.tier} className={`glass-card p-6 ${t.glow} hover:translate-y-[-4px] transition-all duration-300`}>
              <div className={`inline-flex px-3 py-1 rounded-full text-xs font-bold border mb-4 ${t.badge}`}>
                {t.tier}
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Rank</span>
                  <span className="text-white font-semibold">{t.rank}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Pool Share</span>
                  <span className="text-white font-semibold">{t.pool} ({t.usd})</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Vesting</span>
                  <span className="text-white font-semibold">{t.vesting}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Why Participate?</h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              icon: <Shield className="w-7 h-7" />,
              title: 'Post-Quantum Security',
              desc: 'Built with Dilithium5 and Kyber1024 cryptography. Help test the first quantum-resistant DAG-BFT consensus system.',
            },
            {
              icon: <Zap className="w-7 h-7" />,
              title: 'Early Adopter Bonus',
              desc: 'The earlier you join, the higher your multiplier (up to 2x). Early participants shape the network and earn more.',
            },
            {
              icon: <Globe className="w-7 h-7" />,
              title: 'Merkle-Verified Claims',
              desc: 'All scores are anchored in a Merkle tree. Your mainnet rewards are cryptographically provable and tamper-proof.',
            },
          ].map((f) => (
            <div key={f.title} className="glass-card p-8">
              <div className="text-purple-400 mb-4">{f.icon}</div>
              <h3 className="text-lg font-bold text-white mb-2">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="glass-card p-12 text-center quantum-glow">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
          Ready to Start Earning?
        </h2>
        <p className="text-slate-400 max-w-lg mx-auto mb-8">
          Register now, contribute to the network, and secure your share of the $306/day reward pool (~$27,500 over 90 days).
        </p>
        <Link
          to="/register"
          className="btn-shine inline-flex items-center gap-2 px-10 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold rounded-xl text-lg hover:from-purple-500 hover:to-blue-500 transition-all shadow-lg shadow-purple-500/25"
        >
          Register Now
          <ChevronRight className="w-5 h-5" />
        </Link>
      </section>
    </div>
  )
}

/* ── Navigation ── */
function NavLink({ to, icon, children }: { to: string; icon: React.ReactNode; children: React.ReactNode }) {
  const location = useLocation()
  const active = location.pathname === to

  return (
    <Link
      to={to}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
        active
          ? 'bg-purple-500/20 text-white'
          : 'text-slate-400 hover:text-white hover:bg-white/5'
      }`}
    >
      {icon}
      <span className="hidden md:inline">{children}</span>
    </Link>
  )
}

function WalletButton() {
  const { wallet, connect, disconnect, connecting } = useWallet()

  if (wallet?.connected) {
    return (
      <div className="flex items-center gap-2">
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-green-300 text-xs font-mono">
            {wallet.address.slice(0, 6)}...{wallet.address.slice(-4)}
          </span>
        </div>
        <button
          onClick={disconnect}
          className="p-2 text-slate-400 hover:text-red-400 transition-colors"
          title="Disconnect wallet"
        >
          <LogOut size={16} />
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={connect}
      disabled={connecting}
      className="flex items-center gap-1.5 px-3 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white text-sm font-semibold rounded-lg hover:from-purple-500 hover:to-blue-500 transition-all disabled:opacity-50"
    >
      {connecting ? (
        <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white" />
      ) : (
        <Wallet size={14} />
      )}
      <span className="hidden md:inline">{connecting ? 'Connecting...' : 'Connect Wallet'}</span>
    </button>
  )
}

/* ── OAuth2 Callback Handler ── */
function OAuthCallback() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { completeOAuth2 } = useWallet()
  const [error, setError] = useState<string | null>(null)
  const handled = useRef(false)

  useEffect(() => {
    if (handled.current) return
    handled.current = true

    const code = searchParams.get('code')
    const state = searchParams.get('state')
    const oauthError = searchParams.get('error')

    if (oauthError) {
      setError(searchParams.get('error_description') || oauthError)
      return
    }

    if (!code || !state) {
      setError('Missing authorization code or state parameter')
      return
    }

    completeOAuth2(code, state)
      .then(() => {
        // If user already registered, go to dashboard; otherwise go to register
        const hasUserId = !!localStorage.getItem('bounty_user_id')
        navigate(hasUserId ? '/dashboard' : '/register', { replace: true })
      })
      .catch((err) => setError(err.message || 'OAuth2 token exchange failed'))
  }, [searchParams, completeOAuth2, navigate])

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glass-card p-8 max-w-md text-center">
          <div className="text-red-400 text-lg font-semibold mb-2">Connection Failed</div>
          <p className="text-slate-400 text-sm mb-6">{error}</p>
          <button
            onClick={() => navigate('/', { replace: true })}
            className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors"
          >
            Back to Home
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="glass-card p-8 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-400 mx-auto mb-4" />
        <p className="text-slate-300">Connecting wallet...</p>
      </div>
    </div>
  )
}

function AppContent() {
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [location.pathname])

  return (
    <div className="min-h-screen grid-pattern">
      <div className="mesh-gradient" />

      {/* Header */}
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-slate-900/80 backdrop-blur-xl border-b border-white/5 shadow-lg shadow-black/20'
          : 'bg-transparent'
      }`}>
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center text-white font-black text-sm">
                Q
              </div>
              <span className="font-bold text-white hidden sm:inline">QNK Bounty</span>
            </Link>

            <div className="flex items-center gap-1">
              <NavLink to="/" icon={<BarChart3 size={16} />}>Home</NavLink>
              <NavLink to="/dashboard" icon={<Award size={16} />}>Dashboard</NavLink>
              <NavLink to="/leaderboard" icon={<Trophy size={16} />}>Leaderboard</NavLink>
              <NavLink to="/bugs" icon={<Bug size={16} />}>Bugs</NavLink>
              <NavLink to="/social" icon={<Share2 size={16} />}>Social</NavLink>
              <WalletButton />
            </div>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/callback" element={<OAuthCallback />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/register" element={<Register />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/bugs" element={<BugReports />} />
          <Route path="/social" element={<SocialActivity />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid md:grid-cols-3 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center text-white font-black text-xs">Q</div>
                <span className="font-bold text-white text-sm">Q-NarwhalKnight</span>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                The world's first quantum-resistant DAG-BFT consensus system with post-quantum cryptography.
              </p>
            </div>
            <div>
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">Bounty Program</h4>
              <ul className="space-y-2 text-xs text-slate-500">
                <li><Link to="/register" className="hover:text-purple-400 transition-colors">Register</Link></li>
                <li><Link to="/dashboard" className="hover:text-purple-400 transition-colors">Dashboard</Link></li>
                <li><Link to="/leaderboard" className="hover:text-purple-400 transition-colors">Leaderboard</Link></li>
                <li><Link to="/bugs" className="hover:text-purple-400 transition-colors">Bug Reports</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">Resources</h4>
              <ul className="space-y-2 text-xs text-slate-500">
                <li><a href="https://quillon.xyz" className="hover:text-purple-400 transition-colors" target="_blank" rel="noopener noreferrer">Main Website</a></li>
                <li><a href="https://quillon.xyz/downloads/" className="hover:text-purple-400 transition-colors" target="_blank" rel="noopener noreferrer">Download Node</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/5 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-slate-600">
              Q-NarwhalKnight Mainnet Bounty Campaign
            </p>
            <div className="flex items-center gap-2 text-xs text-slate-600">
              <div className="pulse-dot" />
              Campaign Active
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WalletProvider>
        <Router>
          <AppContent />
        </Router>
      </WalletProvider>
    </QueryClientProvider>
  )
}

export default App
