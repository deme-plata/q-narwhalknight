import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface AuthContextType {
  isAuthenticated: boolean
  userId: string | null
  walletAddress: string | null
  token: string | null
  login: (userId: string, walletAddress: string, token: string) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [userId, setUserId] = useState<string | null>(null)
  const [walletAddress, setWalletAddress] = useState<string | null>(null)
  const [token, setToken] = useState<string | null>(null)

  // Load auth state from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('bounty_token')
    const storedUserId = localStorage.getItem('bounty_user_id')
    const storedAddress = localStorage.getItem('bounty_wallet_address')

    if (storedToken && storedUserId && storedAddress) {
      setToken(storedToken)
      setUserId(storedUserId)
      setWalletAddress(storedAddress)
      setIsAuthenticated(true)
    }
  }, [])

  const login = (newUserId: string, newAddress: string, newToken: string) => {
    setUserId(newUserId)
    setWalletAddress(newAddress)
    setToken(newToken)
    setIsAuthenticated(true)

    // Persist to localStorage
    localStorage.setItem('bounty_token', newToken)
    localStorage.setItem('bounty_user_id', newUserId)
    localStorage.setItem('bounty_wallet_address', newAddress)
  }

  const logout = () => {
    setUserId(null)
    setWalletAddress(null)
    setToken(null)
    setIsAuthenticated(false)

    // Clear localStorage
    localStorage.removeItem('bounty_token')
    localStorage.removeItem('bounty_user_id')
    localStorage.removeItem('bounty_wallet_address')
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, userId, walletAddress, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
