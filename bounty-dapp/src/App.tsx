import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Trophy, Award, Bug, Share2, BarChart3 } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Register from './pages/Register'
import Leaderboard from './pages/Leaderboard'
import BugReports from './pages/BugReports'
import SocialActivity from './pages/SocialActivity'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
          {/* Header */}
          <header className="bg-slate-900/50 backdrop-blur-lg border-b border-purple-500/30">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-16">
                <div className="flex items-center">
                  <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400">
                    Q-NarwhalKnight Testnet Bounty
                  </h1>
                </div>
                <div className="flex space-x-6">
                  <NavLink to="/" icon={<BarChart3 size={18} />}>Dashboard</NavLink>
                  <NavLink to="/leaderboard" icon={<Trophy size={18} />}>Leaderboard</NavLink>
                  <NavLink to="/bugs" icon={<Bug size={18} />}>Bug Reports</NavLink>
                  <NavLink to="/social" icon={<Share2 size={18} />}>Social</NavLink>
                  <NavLink to="/register" icon={<Award size={18} />}>Register</NavLink>
                </div>
              </div>
            </nav>
          </header>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/register" element={<Register />} />
              <Route path="/leaderboard" element={<Leaderboard />} />
              <Route path="/bugs" element={<BugReports />} />
              <Route path="/social" element={<SocialActivity />} />
            </Routes>
          </main>

          {/* Footer */}
          <footer className="bg-slate-900/50 backdrop-blur-lg border-t border-purple-500/30 mt-16">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              <p className="text-center text-slate-400">
                Q-NarwhalKnight Testnet Bounty Campaign | Quantum-Enhanced Consensus
              </p>
            </div>
          </footer>
        </div>
      </Router>
    </QueryClientProvider>
  )
}

interface NavLinkProps {
  to: string
  icon: React.ReactNode
  children: React.ReactNode
}

function NavLink({ to, icon, children }: NavLinkProps) {
  return (
    <Link
      to={to}
      className="flex items-center space-x-2 px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-purple-500/20 transition-all duration-200"
    >
      {icon}
      <span>{children}</span>
    </Link>
  )
}

export default App
