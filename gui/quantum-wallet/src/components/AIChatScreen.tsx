import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/atom-one-dark.css';
import {
  Send,
  Sparkles,
  Zap,
  Shield,
  Clock,
  Bot,
  User,
  Settings,
  Trash2,
  Plus,
  DollarSign,
  Activity,
  Cpu,
  Database,
  TrendingUp,
  Network,
  Users,
  Layers,
  Brain,
  ChevronDown,
  Copy,
  Check,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Square,
  Pencil,
  Wrench,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ArrowUpDown,
  Wallet,
  BarChart3,
  Mail,
  Coins,
  PieChart,
  Trophy
} from 'lucide-react';
import { qnkAPI } from '../services/api';
import TransactionPreviewModal from './TransactionPreviewModal';
import VerificationMonitor from './VerificationMonitor';

// Function call interface for Ministral-3B native function calling
interface FunctionCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
  result?: {
    success: boolean;
    data?: any;
    error?: string;
  };
  status: 'pending' | 'executing' | 'completed' | 'failed';
}

// ✅ v7.3.3 - Financial crypto action types for AI-driven commands
type CryptoActionType = 'send' | 'swap' | 'balance' | 'price' | 'history' | 'stake' | 'deploy' | 'pool_info' | 'mail' | 'mint' | 'portfolio' | 'top_tokens';
interface CryptoAction {
  id: string;
  type: CryptoActionType;
  params: Record<string, string>;
  displayText: string;
  status: 'pending' | 'confirming' | 'executing' | 'completed' | 'failed' | 'cancelled';
  result?: { success: boolean; data?: any; error?: string; txHash?: string };
}

// Address book contact for name resolution
interface AddressBookEntry {
  id: string;
  address: string;
  label: string;
  tags?: string[];
  notes?: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  reasoning?: string; // Kimi K2 thinking process (v1.0.5)
  functionCalls?: FunctionCall[]; // Ministral-3B native function calls
  cryptoActions?: CryptoAction[]; // v7.3.3 - AI-detected financial actions
  stats?: {
    tokens: number;
    latency_ms: number;
    tokens_per_second: number;
  };
}

interface Chat {
  chat_id: string;
  title: string;
  created_at: number;
  message_count: number;
}

export default function AIChatScreen() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState(''); // Kimi K2 reasoning (v1.0.5)
  const [maxTokens, setMaxTokens] = useState(2048); // Default 2048 tokens for full responses
  const [showSettings, setShowSettings] = useState(false);
  const [showCostsUsage, setShowCostsUsage] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);

  // Wallet & Usage Data
  const [walletData, setWalletData] = useState<any>(null);
  const [usageData, setUsageData] = useState<any>(null);
  const [pricingData, setPricingData] = useState<any>(null);
  const [isLoadingUsageData, setIsLoadingUsageData] = useState(false);

  // AI Metrics Data
  const [metricsData, setMetricsData] = useState<any>(null);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [isInitialMetricsLoad, setIsInitialMetricsLoad] = useState(true); // Track first load only
  const [workersData, setWorkersData] = useState<any>(null); // v1.0: Active workers for data parallelism

  // AI Settings
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  // Default to Gemma4 (Ollama) — has live network context (hashrate, height, supply)
  const [selectedModel, setSelectedModel] = useState('Gemma4-Ollama');
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [modelSwitchStatus, setModelSwitchStatus] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const backgroundGenerationRef = useRef<boolean>(false);
  const userHasScrolled = useRef(false);
  const lastScrollTop = useRef(0);
  const currentChatIdRef = useRef<string | null>(null); // Track current chat ID for race condition prevention

  // ✅ v0.9.36-beta - AI Transaction Assistant State
  const [transactionPreview, setTransactionPreview] = useState<any>(null);
  const [showTransactionPreview, setShowTransactionPreview] = useState(false);
  const [pendingTransactionMessage, setPendingTransactionMessage] = useState<string>('');

  // ✅ v7.3.3 - Financial Crypto Actions State
  const [addressBook, setAddressBook] = useState<AddressBookEntry[]>([]);
  const [pendingActions, setPendingActions] = useState<Record<string, CryptoAction>>({});

  // ✅ v1.4.2 - Enhanced Chat UX Features
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [copiedCodeIndex, setCopiedCodeIndex] = useState<string | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState<string>('');
  const [messageFeedback, setMessageFeedback] = useState<Record<string, 'up' | 'down' | null>>({});
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load wallet and usage data
  const loadWalletData = async () => {
    const walletAddress = localStorage.getItem('walletAddress') || 'default';
    setIsLoadingUsageData(true);

    try {
      // Fetch wallet balance
      const walletResponse = await fetch(`/api/wallet/balance?wallet_address=${walletAddress}`);
      const walletJson = await walletResponse.json();
      if (walletJson.success) {
        setWalletData(walletJson.data);
      }

      // Fetch usage stats
      const usageResponse = await fetch(`/api/wallet/usage?wallet_address=${walletAddress}`);
      const usageJson = await usageResponse.json();
      if (usageJson.success) {
        setUsageData(usageJson.data);
      }

      // Fetch pricing info
      const pricingResponse = await fetch('/api/pricing');
      const pricingJson = await pricingResponse.json();
      if (pricingJson.success) {
        setPricingData(pricingJson.data);
      }
    } catch (error) {
      console.error('Failed to load wallet data:', error);
    } finally {
      setIsLoadingUsageData(false);
    }
  };

  // Debug: Log whenever messages state changes
  useEffect(() => {
    console.log(`🔍 [STATE] messages changed: ${messages.length} messages, currentChatId: ${currentChatId}`);
    if (messages.length > 0) {
      console.log(`   First message: ${messages[0].role} - ${messages[0].content.substring(0, 50)}...`);
      console.log(`   Last message: ${messages[messages.length - 1].role} - ${messages[messages.length - 1].content.substring(0, 50)}...`);
    }
    console.trace('Stack trace for messages change:');
  }, [messages]);

  // Debug: Log whenever currentChatId changes AND update ref
  // ✅ v2.3.19: Also persist to localStorage for session restoration
  useEffect(() => {
    console.log(`🔍 [STATE] currentChatId changed to: ${currentChatId}`);
    currentChatIdRef.current = currentChatId; // Keep ref in sync for async operations

    // Persist current chat ID for restoration on refresh
    if (currentChatId) {
      localStorage.setItem('ai_current_chat_id', currentChatId);
    }
  }, [currentChatId]);

  // Load wallet data when costs modal is opened
  useEffect(() => {
    if (showCostsUsage) {
      loadWalletData();
    }
  }, [showCostsUsage]);

  // Load AI metrics when metrics modal is opened
  const loadMetrics = async () => {
    setIsLoadingMetrics(true);
    try {
      const response = await fetch('/api/chat/metrics');
      const json = await response.json();
      if (json.success) {
        setMetricsData(json.data);
        setIsInitialMetricsLoad(false); // Mark initial load complete
      }
    } catch (error) {
      console.error('Failed to load AI metrics:', error);
    } finally {
      setIsLoadingMetrics(false);
    }
  };

  // v1.0: Load active workers for data parallelism verification
  const loadWorkers = async () => {
    try {
      const response = await fetch('/api/chat/workers');
      const json = await response.json();
      if (json.success) {
        setWorkersData(json.data);
      }
    } catch (error) {
      console.error('Failed to load workers:', error);
    }
  };

  // Unified metrics/workers loading effect - prevents flickering and duplicate fetches
  useEffect(() => {
    // Initial load
    loadMetrics();
    loadWorkers();

    // Set up interval based on modal state
    const refreshInterval = showMetrics ? 3000 : 5000; // 3s when modal open, 5s when closed

    const interval = setInterval(() => {
      // Only fetch if not currently loading to prevent overlap
      if (!isLoadingMetrics) {
        loadMetrics();
        loadWorkers();
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [showMetrics]); // Only depend on showMetrics, not isLoadingMetrics

  // Auto-scroll to bottom when new messages arrive (only if user hasn't manually scrolled up)
  useEffect(() => {
    if (!messagesContainerRef.current || !messagesEndRef.current) return;

    // Check if user has manually scrolled up
    if (!userHasScrolled.current) {
      // Use instant scroll to prevent fighting with user scroll
      messagesEndRef.current.scrollIntoView({ behavior: 'auto', block: 'end' });
    }
  }, [messages, streamingMessage]);

  // Detect user scroll to disable auto-scroll
  const handleScroll = () => {
    if (!messagesContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;

    // If user scrolled up (scrollTop decreased), mark as manually scrolled
    if (scrollTop < lastScrollTop.current - 10) {
      userHasScrolled.current = true;
    }

    // If user scrolled back to bottom, re-enable auto-scroll
    if (isAtBottom) {
      userHasScrolled.current = false;
    }

    lastScrollTop.current = scrollTop;
  };

  // Load user's chats on mount and check for background generation
  useEffect(() => {
    // Check if there's an ongoing generation in localStorage FIRST
    const activeGeneration = localStorage.getItem('activeAIGeneration');
    if (activeGeneration) {
      try {
        const genData = JSON.parse(activeGeneration);
        // If generation is less than 5 minutes old, restore that chat
        const age = Date.now() - genData.startTime;
        if (age < 5 * 60 * 1000) { // 5 minutes
          console.log('🔄 Detected background generation, restoring chat:', genData.chatId);

          // CRITICAL: Set the chat ID FIRST so messages will display
          setCurrentChatId(genData.chatId);
          currentChatIdRef.current = genData.chatId; // Also update ref immediately

          // Load the chat's messages immediately
          loadMessages(genData.chatId);

          // Mark as generating
          backgroundGenerationRef.current = true;
          setIsGenerating(true);

          // Capture the chat ID for this poll session
          const pollingChatId = genData.chatId;

          // Poll for new messages every 2 seconds
          const pollInterval = setInterval(async () => {
            try {
              // CRITICAL: Check if user switched to a different chat
              if (currentChatIdRef.current !== pollingChatId) {
                console.log(`🛑 [mount] Chat switched from ${pollingChatId} to ${currentChatIdRef.current}, stopping poll`);
                clearInterval(pollInterval);
                return;
              }

              const response = await fetch(`/api/chat/${pollingChatId}/messages`);
              if (response.ok) {
                const backendMessages = await response.json();

                // Ensure we have valid array data before setting state
                if (backendMessages.success && Array.isArray(backendMessages.data)) {
                  // Double-check we're still on the same chat
                  if (currentChatIdRef.current === pollingChatId) {
                    setMessages(backendMessages.data);
                  } else {
                    console.log(`🛑 [mount] Chat changed during fetch, discarding`);
                    clearInterval(pollInterval);
                    return;
                  }

                  // If we got a new assistant message, generation is complete
                  const lastMsg = backendMessages.data[backendMessages.data.length - 1];
                  if (lastMsg && lastMsg.role === 'assistant' && lastMsg.timestamp > genData.startTime / 1000) {
                    console.log('✅ Background generation completed!');
                    setIsGenerating(false);
                    backgroundGenerationRef.current = false;
                    localStorage.removeItem('activeAIGeneration');
                    clearInterval(pollInterval);
                  }
                }
              }
            } catch (err) {
              console.error('Polling error on mount:', err);
            }
          }, 2000);

          // Stop polling after 5 minutes
          setTimeout(() => {
            clearInterval(pollInterval);
            setIsGenerating(false);
            backgroundGenerationRef.current = false;
            localStorage.removeItem('activeAIGeneration');
          }, 5 * 60 * 1000);
        } else {
          // Too old, clear it
          localStorage.removeItem('activeAIGeneration');
        }
      } catch (e) {
        console.error('Failed to parse active generation:', e);
        localStorage.removeItem('activeAIGeneration');
      }
    } else {
      // ✅ v2.3.19: Restore last active chat from localStorage (persistence across refresh)
      const savedChatId = localStorage.getItem('ai_current_chat_id');
      if (savedChatId) {
        console.log('📂 Restoring last active chat:', savedChatId);
        setCurrentChatId(savedChatId);
        currentChatIdRef.current = savedChatId;
        loadMessages(savedChatId);
      }
    }

    // Load chats list (this will NOT override currentChatId if already set)
    loadChats(false);
  }, []);

  // Monitor chat switches and check for ongoing generation
  useEffect(() => {
    if (!currentChatId || isGenerating) return;

    // Check if there's an ongoing generation for this specific chat
    const activeGeneration = localStorage.getItem('activeAIGeneration');
    let pollInterval: NodeJS.Timeout | null = null;
    let timeoutId: NodeJS.Timeout | null = null;

    if (activeGeneration) {
      try {
        const genData = JSON.parse(activeGeneration);

        // Only reconnect if this is the chat that's generating
        if (genData.chatId === currentChatId) {
          const age = Date.now() - genData.startTime;

          // If generation is less than 5 minutes old, start polling for completion
          if (age < 5 * 60 * 1000) {
            console.log('🔄 Reconnecting to ongoing generation for current chat:', currentChatId);
            setIsGenerating(true);
            backgroundGenerationRef.current = true;

            // Capture chatId at the time polling starts to avoid stale closure
            const pollingChatId = currentChatId;

            // Poll for new messages every 2 seconds
            pollInterval = setInterval(async () => {
              try {
                // CRITICAL: Check if user switched to a different chat - if so, stop polling
                if (currentChatIdRef.current !== pollingChatId) {
                  console.log(`🛑 Chat switched from ${pollingChatId} to ${currentChatIdRef.current}, stopping poll`);
                  if (pollInterval) clearInterval(pollInterval);
                  return;
                }

                const response = await fetch(`/api/chat/${pollingChatId}/messages`);
                if (response.ok) {
                  const backendMessages = await response.json();
                  if (backendMessages.success && Array.isArray(backendMessages.data) && backendMessages.data.length > 0) {
                    // CRITICAL: Double-check we're still on the same chat before updating UI
                    if (currentChatIdRef.current === pollingChatId) {
                      setMessages(backendMessages.data);
                    } else {
                      console.log(`🛑 Chat changed during fetch, discarding messages for ${pollingChatId}`);
                      if (pollInterval) clearInterval(pollInterval);
                      return;
                    }

                    // Check if generation completed (new assistant message after startTime)
                    const lastMsg = backendMessages.data[backendMessages.data.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant' && lastMsg.timestamp > genData.startTime / 1000) {
                      console.log('✅ Background generation completed on return!');
                      setIsGenerating(false);
                      backgroundGenerationRef.current = false;
                      localStorage.removeItem('activeAIGeneration');
                      if (pollInterval) clearInterval(pollInterval);
                    }
                  } else if (!backendMessages.success) {
                    console.warn('Failed to fetch messages, stopping reconnection');
                    setIsGenerating(false);
                    backgroundGenerationRef.current = false;
                    localStorage.removeItem('activeAIGeneration');
                    if (pollInterval) clearInterval(pollInterval);
                  }
                }
              } catch (err) {
                console.error('Polling error:', err);
                // On error, stop trying to reconnect
                setIsGenerating(false);
                backgroundGenerationRef.current = false;
                localStorage.removeItem('activeAIGeneration');
                if (pollInterval) clearInterval(pollInterval);
              }
            }, 2000);

            // Stop polling after 5 minutes
            timeoutId = setTimeout(() => {
              if (pollInterval) clearInterval(pollInterval);
              setIsGenerating(false);
              backgroundGenerationRef.current = false;
              localStorage.removeItem('activeAIGeneration');
            }, 5 * 60 * 1000);
          } else {
            // Too old, clear it
            localStorage.removeItem('activeAIGeneration');
          }
        }
      } catch (e) {
        console.error('Failed to reconnect to generation:', e);
      }
    }

    // CRITICAL: Cleanup when currentChatId changes or component unmounts
    // This prevents messages from one chat overwriting another chat's messages
    return () => {
      if (pollInterval) {
        console.log('🧹 Cleaning up poll interval for chat switch');
        clearInterval(pollInterval);
      }
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [currentChatId]);

  // Monitor page visibility - check for ongoing generation when user returns
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden && currentChatId && !isGenerating) {
        // Page became visible, check for ongoing generation
        const activeGeneration = localStorage.getItem('activeAIGeneration');
        if (activeGeneration) {
          try {
            const genData = JSON.parse(activeGeneration);

            // Only reconnect if this is the chat that's generating
            if (genData.chatId === currentChatId) {
              const age = Date.now() - genData.startTime;

              if (age < 5 * 60 * 1000) {
                console.log('👁️ Page visible again, checking for ongoing generation...');

                // Capture the chat ID for this poll session
                const pollingChatId = currentChatId;

                // Load latest messages to show any progress
                loadMessages(pollingChatId);

                // Start polling to check if still generating
                setIsGenerating(true);
                backgroundGenerationRef.current = true;

                const pollInterval = setInterval(async () => {
                  try {
                    // CRITICAL: Check if user switched to a different chat
                    if (currentChatIdRef.current !== pollingChatId) {
                      console.log(`🛑 [visibility] Chat switched from ${pollingChatId} to ${currentChatIdRef.current}, stopping poll`);
                      clearInterval(pollInterval);
                      return;
                    }

                    const response = await fetch(`/api/chat/${pollingChatId}/messages`);
                    if (response.ok) {
                      const backendMessages = await response.json();
                      if (backendMessages.success && Array.isArray(backendMessages.data) && backendMessages.data.length > 0) {
                        // Double-check we're still on the same chat
                        if (currentChatIdRef.current === pollingChatId) {
                          setMessages(backendMessages.data);
                        } else {
                          console.log(`🛑 [visibility] Chat changed during fetch, discarding`);
                          clearInterval(pollInterval);
                          return;
                        }

                        // Check if generation completed
                        const lastMsg = backendMessages.data[backendMessages.data.length - 1];
                        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.timestamp > genData.startTime / 1000) {
                          console.log('✅ Generation completed while away!');
                          setIsGenerating(false);
                          backgroundGenerationRef.current = false;
                          localStorage.removeItem('activeAIGeneration');
                          clearInterval(pollInterval);
                        }
                      }
                    }
                  } catch (err) {
                    console.error('Visibility polling error:', err);
                    setIsGenerating(false);
                    backgroundGenerationRef.current = false;
                    localStorage.removeItem('activeAIGeneration');
                    clearInterval(pollInterval);
                  }
                }, 2000);

                // Stop polling after 5 minutes
                setTimeout(() => {
                  clearInterval(pollInterval);
                  setIsGenerating(false);
                  backgroundGenerationRef.current = false;
                  localStorage.removeItem('activeAIGeneration');
                }, 5 * 60 * 1000);
              } else {
                localStorage.removeItem('activeAIGeneration');
              }
            }
          } catch (e) {
            console.error('Failed to handle visibility change:', e);
          }
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [currentChatId, isGenerating]);

  const loadChats = async (autoSelect: boolean = true) => {
    try {
      const userId = localStorage.getItem('walletAddress') || 'default';
      const response = await fetch(`/api/chat/list?user_id=${userId}`);
      const data = await response.json();

      if (data.success && data.data) {
        setChats(data.data);

        // If no current chat, select the most recent one (only if autoSelect is true)
        if (autoSelect && !currentChatId && data.data.length > 0) {
          // CRITICAL: Update ref FIRST before async operations
          currentChatIdRef.current = data.data[0].chat_id;
          setCurrentChatId(data.data[0].chat_id);
          loadMessages(data.data[0].chat_id);
        }
      }
    } catch (error) {
      console.error('Failed to load chats:', error);
    }
  };

  const loadMessages = async (chatId: string) => {
    try {
      const caller = new Error().stack?.split('\n')[2]?.trim() || 'unknown';
      console.log(`📥 Loading messages for chat: ${chatId}`);
      console.log(`   Called from: ${caller}`);
      const response = await fetch(`/api/chat/${chatId}/messages`);
      const data = await response.json();

      console.log(`📊 Received ${data.success ? 'success' : 'failure'}, data:`, data);

      if (data.success && data.data && Array.isArray(data.data)) {
        // CRITICAL: Verify we're still on the same chat before setting messages
        // This prevents race conditions when rapidly switching chats
        if (currentChatIdRef.current !== chatId) {
          console.log(`🛑 [loadMessages] Chat changed from ${chatId} to ${currentChatIdRef.current}, discarding fetched messages`);
          return;
        }

        // ✅ v2.3.19: Merge with localStorage cached messages (for partial responses)
        // This preserves messages that were saved when stopping generation
        const cacheKey = `chat_messages_${chatId}`;
        const cachedData = localStorage.getItem(cacheKey);
        let finalMessages = data.data;

        if (cachedData) {
          try {
            const cachedMessages = JSON.parse(cachedData);
            // Find cached messages that don't exist in backend (by ID prefix)
            const backendIds = new Set(data.data.map((m: Message) => m.id));
            const uniqueCachedMessages = cachedMessages.filter(
              (m: Message) => !backendIds.has(m.id) && m.id.startsWith('partial-')
            );

            if (uniqueCachedMessages.length > 0) {
              console.log(`📦 Restoring ${uniqueCachedMessages.length} cached partial responses`);
              finalMessages = [...data.data, ...uniqueCachedMessages].sort((a, b) => a.timestamp - b.timestamp);
            }
          } catch (e) {
            console.warn('Failed to parse cached messages:', e);
          }
        }

        console.log(`✅ Setting ${finalMessages.length} messages for chat ${chatId}`);
        setMessages(finalMessages);

        // Check if there's an active generation that just completed
        const activeGeneration = localStorage.getItem('activeAIGeneration');
        if (activeGeneration) {
          try {
            const genData = JSON.parse(activeGeneration);
            if (genData.chatId === chatId) {
              const lastMsg = data.data[data.data.length - 1];
              if (lastMsg && lastMsg.role === 'assistant' && lastMsg.timestamp > genData.startTime / 1000) {
                console.log('✅ Generation completed! Clearing marker.');
                localStorage.removeItem('activeAIGeneration');
                setIsGenerating(false);
                backgroundGenerationRef.current = false;
              }
            }
          } catch (e) {
            console.error('Error checking generation status:', e);
          }
        }
      } else {
        // CRITICAL: Only clear messages if we're still on the same chat
        if (currentChatIdRef.current === chatId) {
          console.warn('⚠️ No valid messages data, setting empty array');
          setMessages([]);
        }
      }
    } catch (error) {
      console.error('❌ Failed to load messages:', error);
      // CRITICAL: Only clear messages if we're still on the same chat
      if (currentChatIdRef.current === chatId) {
        setMessages([]);
      }
    }
  };

  const createNewChat = async () => {
    try {
      const userId = localStorage.getItem('walletAddress') || 'default';
      const response = await fetch('/api/chat/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          title: 'New Chat',
          encryption_enabled: true,
          distributed_enabled: true,
          enable_kv_cache: true
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to create chat - HTTP', response.status, errorText);
        alert(`Failed to create new chat: ${response.status} - ${errorText}`);
        return;
      }

      const data = await response.json();
      if (data.success && data.data) {
        // CRITICAL: Update ref FIRST before state
        currentChatIdRef.current = data.data.chat_id;
        setCurrentChatId(data.data.chat_id);
        setMessages([]);
        loadChats(false); // Don't auto-select, we already set the current chat
      } else {
        console.error('Failed to create chat - API returned error:', data);
        alert(`Failed to create new chat: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to create chat:', error);
      alert(`Failed to create new chat: ${error}`);
    }
  };

  const deleteChat = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const userId = localStorage.getItem('walletAddress') || 'default';
      await fetch(`/api/chat/${chatId}?user_id=${userId}`, {
        method: 'DELETE'
      });

      if (chatId === currentChatId) {
        // CRITICAL: Update ref FIRST
        currentChatIdRef.current = null;
        setCurrentChatId(null);
        setMessages([]);
      }
      loadChats(true); // Auto-select another chat after deletion
    } catch (error) {
      console.error('Failed to delete chat:', error);
    }
  };

  const generateChatTitle = async (chatId: string, firstMessage: string) => {
    try {
      // Generate a concise title from the first message (client-side extraction)
      // DO NOT use the /stream endpoint — it saves messages to the chat DB,
      // which causes the title prompt + response to appear as duplicate messages.
      const words = firstMessage.trim().split(/\s+/);
      let generatedTitle = words.slice(0, 5).join(' ');
      if (words.length > 5) generatedTitle += '...';
      // Clean up: remove quotes, newlines, and excessive punctuation
      generatedTitle = generatedTitle.replace(/[\n\r]/g, ' ').replace(/['"]/g, '').trim();
      if (!generatedTitle) generatedTitle = 'New Chat';

      // Update the chat title via the rename endpoint (no messages saved)
      await fetch(`/api/chat/${chatId}/rename`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: generatedTitle })
      });
      loadChats(false); // Refresh chat list without auto-selecting
    } catch (error) {
      console.error('Failed to generate title:', error);
    }
  };

  const switchModel = async (modelName: string) => {
    if (isSwitchingModel) return;

    // Update UI immediately (optimistic update)
    setSelectedModel(modelName);
    setIsSwitchingModel(true);
    setModelSwitchStatus(`Switching to ${modelName}...`);

    // BitNet uses external API - no backend model switch needed
    if (modelName === 'BitNet-b1.58-2B-4T') {
      setModelSwitchStatus(`✅ Switched to BitNet b1.58-2B (1-bit quantized, via llama-server)`);
      setTimeout(() => setModelSwitchStatus(null), 3000);
      setIsSwitchingModel(false);
      return;
    }

    // If no chat exists yet, just update the state for future use
    if (!currentChatId) {
      setModelSwitchStatus(`✅ Model set to ${modelName}`);
      setTimeout(() => setModelSwitchStatus(null), 2000);
      setIsSwitchingModel(false);
      return;
    }

    try {
      const response = await fetch(`/api/chat/${currentChatId}/switch-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName })
      });

      const result = await response.json();

      if (result.success && result.data.success) {
        setModelSwitchStatus(`✅ Switched to ${modelName} (${result.data.model_size_gb.toFixed(1)} GB)`);
        setTimeout(() => setModelSwitchStatus(null), 3000);
      } else {
        const errorMsg = result.data?.message || result.error || 'Unknown error';
        setModelSwitchStatus(`❌ Failed: ${errorMsg}`);
        console.error('Model switch failed:', errorMsg);
        setTimeout(() => setModelSwitchStatus(null), 5000);
      }
    } catch (error) {
      console.error('Failed to switch model:', error);
      setModelSwitchStatus(`❌ Failed to switch model: ${error}`);
      setTimeout(() => setModelSwitchStatus(null), 5000);
    } finally {
      setIsSwitchingModel(false);
    }
  };

  // ✅ v0.9.36-beta - AI Transaction Detection and Preparation
  const detectAndPrepareTransaction = async (message: string): Promise<boolean> => {
    // Check if message contains transaction keywords
    const transactionKeywords = /\b(send|pay|transfer)\b.*\b(\d+(\.\d+)?)\s*(qug|quillon)\b/i;

    if (!transactionKeywords.test(message)) {
      return false; // Not a transaction request
    }

    console.log('💰 Transaction detected in message:', message);

    try {
      const walletAddress = localStorage.getItem('walletAddress') || 'default';

      // Call AI Transaction Preparation API
      const response = await fetch('/api/v1/ai/transaction/prepare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Address': walletAddress,
        },
        body: JSON.stringify({
          natural_language_query: message,
        }),
      });

      const data = await response.json();

      if (data.success && data.data) {
        console.log('✅ Transaction preview generated:', data.data);
        setTransactionPreview(data.data);
        setShowTransactionPreview(true);
        setPendingTransactionMessage(message);
        return true; // Transaction detected and preview shown
      } else {
        console.error('❌ Failed to prepare transaction:', data.error);
        return false;
      }
    } catch (error) {
      console.error('❌ Transaction preparation error:', error);
      return false;
    }
  };

  const handleTransactionConfirm = async () => {
    // Close modal
    setShowTransactionPreview(false);

    // TODO: Actually send the transaction to the blockchain
    // For now, just send the message to AI chat as normal
    setPendingTransactionMessage('');

    // TODO: Implement actual transaction signing and submission
    console.log('🚀 Transaction confirmed, would send:', transactionPreview);
    console.log('📝 Original message:', pendingTransactionMessage);

    // For now, proceed with sending the message to the AI
    // In the future, this should create a signed transaction and submit it
  };

  const handleTransactionCancel = () => {
    setShowTransactionPreview(false);
    setTransactionPreview(null);
    setPendingTransactionMessage('');
  };

  // ✅ v1.4.2 - Copy message content to clipboard
  const copyMessageContent = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // ✅ v1.4.2 - Copy code block to clipboard
  const copyCodeBlock = async (codeIndex: string, code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCodeIndex(codeIndex);
      setTimeout(() => setCopiedCodeIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  // ✅ v1.4.2 - Regenerate last AI response
  const regenerateResponse = async () => {
    if (isGenerating || messages.length === 0) return;

    // Find the last user message
    const lastUserMessageIndex = [...messages].reverse().findIndex(m => m.role === 'user');
    if (lastUserMessageIndex === -1) return;

    const actualIndex = messages.length - 1 - lastUserMessageIndex;
    const lastUserMessage = messages[actualIndex];

    // Remove the last assistant message if it exists
    const newMessages = messages.filter((_, i) => {
      // Keep everything up to and including the last user message
      return i <= actualIndex;
    });
    setMessages(newMessages);

    // Re-send the last user message
    setInput(lastUserMessage.content);
    // Small delay to ensure state updates, then trigger send
    setTimeout(() => {
      const sendBtn = document.querySelector('[data-send-button]') as HTMLButtonElement;
      if (sendBtn) sendBtn.click();
    }, 100);
  };

  // ✅ v1.4.2 - Edit user message
  const startEditingMessage = (messageId: string, content: string) => {
    setEditingMessageId(messageId);
    setEditingContent(content);
  };

  const cancelEditingMessage = () => {
    setEditingMessageId(null);
    setEditingContent('');
  };

  const saveEditedMessage = async () => {
    if (!editingMessageId || !editingContent.trim()) return;

    // Find the message index
    const messageIndex = messages.findIndex(m => m.id === editingMessageId);
    if (messageIndex === -1) return;

    // Remove all messages after this one (including AI responses)
    const newMessages = messages.slice(0, messageIndex);
    setMessages(newMessages);

    // Set the edited content as input and send
    setInput(editingContent);
    setEditingMessageId(null);
    setEditingContent('');

    // Trigger send after state updates
    setTimeout(() => {
      const sendBtn = document.querySelector('[data-send-button]') as HTMLButtonElement;
      if (sendBtn) sendBtn.click();
    }, 100);
  };

  // ✅ v1.4.2 - Message feedback
  const submitFeedback = async (messageId: string, feedback: 'up' | 'down') => {
    const currentFeedback = messageFeedback[messageId];
    const newFeedback = currentFeedback === feedback ? null : feedback;

    setMessageFeedback(prev => ({
      ...prev,
      [messageId]: newFeedback
    }));

    // Optionally send feedback to backend
    try {
      await fetch('/api/chat/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: messageId,
          feedback: newFeedback,
          chat_id: currentChatId
        })
      });
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    }
  };

  // ✅ v2.3.19 - Stop generation - PRESERVES partial response
  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    // ✅ CRITICAL FIX: Save the partial response BEFORE clearing
    // This preserves the AI's answer even when stopping mid-generation
    if (streamingMessage.trim()) {
      const partialMessage: Message = {
        id: `partial-${Date.now()}`,
        role: 'assistant',
        content: streamingMessage + '\n\n*[Generation stopped]*',
        timestamp: Date.now() / 1000,
        reasoning: streamingReasoning || undefined,
      };
      setMessages(prev => [...prev, partialMessage]);
      console.log('📝 Saved partial AI response:', streamingMessage.length, 'chars');

      // Also save to localStorage for persistence across refresh
      if (currentChatId) {
        const cacheKey = `chat_messages_${currentChatId}`;
        const existingCache = localStorage.getItem(cacheKey);
        const existingMessages = existingCache ? JSON.parse(existingCache) : [];
        existingMessages.push(partialMessage);
        localStorage.setItem(cacheKey, JSON.stringify(existingMessages));
      }
    }

    setIsGenerating(false);
    setStreamingMessage('');
    setStreamingReasoning('');
    localStorage.removeItem('activeAIGeneration');
    backgroundGenerationRef.current = false;
  };

  // ✅ v7.3.3 - Load address book for name resolution
  useEffect(() => {
    const loadAddressBook = async () => {
      try {
        const walletAddress = localStorage.getItem('walletAddress') || '';
        if (!walletAddress) return;
        const resp = await fetch('/api/v1/addressbook', {
          headers: {
            'X-Wallet-Address': walletAddress,
            'X-Auth-Signature': localStorage.getItem('authSignature') || '',
            'X-Auth-Timestamp': localStorage.getItem('authTimestamp') || '',
            'X-Auth-Public-Key': localStorage.getItem('authPublicKey') || '',
          }
        });
        const data = await resp.json();
        if (data.success && Array.isArray(data.data)) {
          setAddressBook(data.data);
        }
      } catch (err) {
        console.log('📒 Address book not loaded (non-critical):', err);
      }
    };
    loadAddressBook();
  }, []);

  // ✅ v7.3.3 - Resolve name to address from address book
  const resolveNameToAddress = (name: string): { address: string; label: string } | null => {
    const lower = name.toLowerCase().trim();
    // Exact match first
    const exact = addressBook.find(e => e.label.toLowerCase() === lower);
    if (exact) return { address: exact.address, label: exact.label };
    // Partial match
    const partial = addressBook.find(e => e.label.toLowerCase().includes(lower));
    if (partial) return { address: partial.address, label: partial.label };
    // Tag match
    const tagged = addressBook.find(e => e.tags?.some(t => t.toLowerCase() === lower));
    if (tagged) return { address: tagged.address, label: tagged.label };
    return null;
  };

  // ✅ v7.4.1 - Build a CryptoAction from type + params
  const buildAction = (type: CryptoActionType, params: Record<string, string>, index: number): CryptoAction => {
    let displayText = '';
    switch (type) {
      case 'send':
        displayText = `Send ${params.amount || '?'} ${(params.token || 'QUG').toUpperCase()} to ${params.to_name || params.to || '?'}`;
        break;
      case 'swap':
        displayText = `Swap ${params.amount || '?'} ${(params.from || 'QUG').toUpperCase()} for ${(params.to || '?').toUpperCase()}`;
        break;
      case 'balance':
        displayText = `Check ${(params.token || 'all').toUpperCase()} balance`;
        break;
      case 'price':
        displayText = `Get ${(params.token || 'QUG').toUpperCase()} price`;
        break;
      case 'history':
        displayText = `Show transaction history${params.limit ? ` (last ${params.limit})` : ''}`;
        break;
      case 'pool_info':
        displayText = `Show ${(params.pair || 'all').toUpperCase()} pool info`;
        break;
      case 'mail':
        displayText = `Send mail to ${params.to_name || params.to || '?'}: "${(params.subject || 'No subject').slice(0, 40)}"`;
        break;
      case 'mint':
        displayText = `Mint ${params.amount || '?'} QUGUSD (collateral: ${params.collateral || '?'} QUG)`;
        break;
      case 'portfolio':
        displayText = `Show full portfolio summary`;
        break;
      case 'top_tokens':
        displayText = `Show top ${params.limit || '10'} tokens by ${params.sort || 'volume'}`;
        break;
      default:
        displayText = `${type}: ${JSON.stringify(params)}`;
    }
    return {
      id: `action-${Date.now()}-${index}`,
      type,
      params,
      displayText,
      status: 'pending',
    };
  };

  // ✅ v7.3.3 - Parse crypto actions from AI response text
  const parseCryptoActions = (text: string): CryptoAction[] => {
    const actions: CryptoAction[] = [];

    // Pattern 1: [ACTION:type key=val ...] (standard format, closing ] optional)
    const actionRegex = /\[ACTION:(\w+)(?:\s+([^\]\n]+))?\]?/g;
    let match;
    while ((match = actionRegex.exec(text)) !== null) {
      const type = match[1].toLowerCase() as CryptoActionType;
      const paramsStr = match[2] || '';
      const params: Record<string, string> = {};
      const paramRegex = /(\w+)=(?:"([^"]+)"|(\S+))/g;
      let pm;
      while ((pm = paramRegex.exec(paramsStr)) !== null) {
        params[pm[1]] = pm[2] || pm[3];
      }
      actions.push(buildAction(type, params, actions.length));
    }

    // Pattern 2: [type] without ACTION: prefix (BitNet 2B sometimes does this)
    if (actions.length === 0) {
      const simpleTypes = ['portfolio', 'balance', 'price', 'history', 'top_tokens', 'pool_info', 'send', 'swap', 'mail', 'mint'];
      for (const t of simpleTypes) {
        const simpleRegex = new RegExp(`\\[${t}(?:[:\\s]([^\\]\\n]*))?\\]?`, 'gi');
        let sm;
        while ((sm = simpleRegex.exec(text)) !== null) {
          const params: Record<string, string> = {};
          if (sm[1]) {
            const paramRegex = /(\w+)=(?:"([^"]+)"|(\S+))/g;
            let pm;
            while ((pm = paramRegex.exec(sm[1])) !== null) {
              params[pm[1]] = pm[2] || pm[3];
            }
          }
          actions.push(buildAction(t as CryptoActionType, params, actions.length));
        }
      }
    }

    return actions;
  };

  // ✅ v7.4.1 - Pre-process user input to detect financial commands directly
  // This bypasses the AI when the intent is clear, giving instant results
  const parseUserIntent = (userText: string): CryptoAction | null => {
    const text = userText.toLowerCase().trim();

    // Portfolio / balance checks
    if (/^(show\s+)?my\s+portfolio$/i.test(text) || /^portfolio$/i.test(text)) {
      return buildAction('portfolio', {}, 0);
    }
    if (/^(what'?s?\s+)?my\s+balance\??$/i.test(text) || /^(check\s+)?balance$/i.test(text)) {
      return buildAction('balance', { token: 'all' }, 0);
    }

    // Price check: "price of QUG", "how much is BORK", "QUG price"
    const priceMatch = text.match(/(?:price\s+(?:of\s+)?|how\s+much\s+is\s+)(\w+)/i) || text.match(/^(\w+)\s+price\??$/i);
    if (priceMatch) {
      return buildAction('price', { token: priceMatch[1].toUpperCase() }, 0);
    }

    // Top tokens
    if (/^(show\s+)?top\s+tokens/i.test(text) || /^trending\s+tokens/i.test(text)) {
      const limitMatch = text.match(/top\s+(\d+)/i);
      return buildAction('top_tokens', { limit: limitMatch?.[1] || '10', sort: 'volume' }, 0);
    }

    // Transaction history
    if (/^(show\s+)?(my\s+)?(last\s+\d+\s+)?transactions?\s*(history)?/i.test(text) || /^(tx\s+)?history/i.test(text)) {
      const limitMatch = text.match(/last\s+(\d+)/i);
      return buildAction('history', { limit: limitMatch?.[1] || '10' }, 0);
    }

    // Send: "send 50 QUG to alice", "send 10 to bob"
    const sendMatch = text.match(/^send\s+([\d.]+)\s*(\w+)?\s+to\s+(\w+)/i);
    if (sendMatch) {
      return buildAction('send', {
        amount: sendMatch[1],
        token: (sendMatch[2] || 'QUG').toUpperCase(),
        to_name: sendMatch[3],
      }, 0);
    }

    // Swap/buy: "buy 100 BORK with QUG", "swap 50 QUG for BORK"
    const buyMatch = text.match(/^buy\s+([\d.]+)\s+(\w+)\s+(?:with|using)\s+(\w+)/i);
    if (buyMatch) {
      return buildAction('swap', { amount: buyMatch[1], from: buyMatch[3].toUpperCase(), to: buyMatch[2].toUpperCase() }, 0);
    }
    const swapMatch = text.match(/^swap\s+([\d.]+)\s+(\w+)\s+(?:for|to)\s+(\w+)/i);
    if (swapMatch) {
      return buildAction('swap', { amount: swapMatch[1], from: swapMatch[2].toUpperCase(), to: swapMatch[3].toUpperCase() }, 0);
    }

    // Pool info: "show QUG/QUGUSD pool"
    const poolMatch = text.match(/(?:show\s+)?(\w+)\s*\/\s*(\w+)\s+pool/i);
    if (poolMatch) {
      return buildAction('pool_info', { pair: `${poolMatch[1]}/${poolMatch[2]}` }, 0);
    }

    // Mint: "mint 100 QUGUSD", "mint stablecoin with 100 QUG"
    const mintMatch = text.match(/^mint\s+([\d.]+)\s*(?:qugusd|stablecoin)/i) || text.match(/^mint\s+(?:qugusd|stablecoin)\s+(?:with\s+)?([\d.]+)/i);
    if (mintMatch) {
      return buildAction('mint', { collateral: mintMatch[1] }, 0);
    }

    return null; // No direct intent detected, let AI handle it
  };

  // ✅ v7.3.3 - Strip action tags from display text
  const stripActionTags = (text: string): string => {
    // v7.4.1: Match both complete and incomplete ACTION tags (missing closing ])
    return text.replace(/\[ACTION:\w+(?:\s+[^\]\n]+)?\]?/g, '').trim();
  };

  // ✅ v7.3.3 - Execute a crypto action
  const executeCryptoAction = async (action: CryptoAction) => {
    const walletAddress = localStorage.getItem('walletAddress') || '';
    const authHeaders = {
      'Content-Type': 'application/json',
      'X-Wallet-Address': walletAddress,
      'X-Auth-Signature': localStorage.getItem('authSignature') || '',
      'X-Auth-Timestamp': localStorage.getItem('authTimestamp') || '',
      'X-Auth-Public-Key': localStorage.getItem('authPublicKey') || '',
    };

    // Update action status to executing
    setPendingActions(prev => ({
      ...prev,
      [action.id]: { ...action, status: 'executing' }
    }));
    setMessages(prev => prev.map(m => ({
      ...m,
      cryptoActions: m.cryptoActions?.map(a =>
        a.id === action.id ? { ...a, status: 'executing' as const } : a
      )
    })));

    try {
      let result: CryptoAction['result'];

      switch (action.type) {
        case 'send': {
          // Resolve recipient name to address
          let toAddress = action.params.to || '';
          if (action.params.to_name) {
            const resolved = resolveNameToAddress(action.params.to_name);
            if (resolved) {
              toAddress = resolved.address;
            } else {
              throw new Error(`Contact "${action.params.to_name}" not found in address book`);
            }
          }
          const amount = parseFloat(action.params.amount || '0');
          if (!toAddress || amount <= 0) throw new Error('Invalid send parameters');

          const sendResp = await qnkAPI.sendTransaction(
            walletAddress,
            toAddress,
            amount,
            action.params.memo || 'Sent via AI Chat',
            (action.params.token || 'QUG').toUpperCase()
          );
          result = sendResp.success
            ? { success: true, data: sendResp.data, txHash: sendResp.data?.transaction_hash || sendResp.data?.tx_hash }
            : { success: false, error: sendResp.error || 'Transaction failed' };
          break;
        }
        case 'swap': {
          const fromToken = (action.params.from || 'QUG').toUpperCase();
          const toToken = (action.params.to || '').toUpperCase();
          const amountInStr = action.params.amount || '0';
          if (!toToken) throw new Error('Missing target token for swap');

          // Convert to 24-decimal BigInt string for the API
          const amountFloat = parseFloat(amountInStr);
          const amountIn24 = (BigInt(Math.floor(amountFloat * 1e6)) * BigInt(1e18)).toString();
          const minOut24 = (BigInt(Math.floor(amountFloat * 0.95 * 1e6)) * BigInt(1e18)).toString();

          const swapResp = await qnkAPI.executeSwap({
            from_token: fromToken,
            to_token: toToken,
            amount_in: amountIn24,
            min_amount_out: minOut24,
            wallet_address: walletAddress,
          });
          result = swapResp.success
            ? { success: true, data: swapResp.data }
            : { success: false, error: swapResp.error || 'Swap failed' };
          break;
        }
        case 'balance': {
          const token = (action.params.token || '').toUpperCase();
          if (!token || token === 'ALL' || token === 'QUG') {
            // Use authenticated API service for proper wallet auth
            const balResp = await qnkAPI.getWalletBalance(walletAddress);
            const tokResp = await qnkAPI.getMultiTokenBalance();
            const qugBal = balResp.data?.balance ?? balResp.data?.confirmed ?? 0;
            // v7.4.1: tokResp.data = { tokens: { QUG: {...}, QUGUSD: {...} }, total_usd_value }
            // tokens is a HashMap/Object, NOT an array - must use Object.entries
            const tokensMap = tokResp.data?.tokens || {};
            const tokensList = Object.entries(tokensMap).map(([symbol, t]: [string, any]) => ({
              symbol,
              balance: t.balance,
              value_usd: t.usd_value?.toFixed?.(2) || String(t.usd_value || '0'),
            }));
            result = {
              success: true,
              data: {
                qug_balance: typeof qugBal === 'number' ? qugBal.toFixed(4) : qugBal,
                tokens: tokensList,
                total_usd_value: tokResp.data?.total_usd_value?.toFixed?.(2) || '0',
              }
            };
          } else {
            const tokResp = await qnkAPI.getMultiTokenBalance();
            // tokens is a HashMap keyed by symbol
            const tokensMap = tokResp.data?.tokens || {};
            const tokenBal = tokensMap[token] || tokensMap[token.toLowerCase()];
            result = { success: true, data: tokenBal ? { symbol: token, ...tokenBal } : { token, balance: 'Token not found' } };
          }
          break;
        }
        case 'price': {
          const token = (action.params.token || 'QUG').toUpperCase();
          const priceResp = await qnkAPI.getOraclePrice(`${token}/USD`);
          result = { success: true, data: priceResp.data || priceResp };
          break;
        }
        case 'history': {
          const limit = parseInt(action.params.limit || '10');
          try {
            const txResp = await qnkAPI.getRecentTransactions(limit);
            result = { success: true, data: txResp.data || [] };
          } catch {
            result = { success: true, data: { message: 'No recent transactions found' } };
          }
          break;
        }
        case 'pool_info': {
          const poolsResp = await qnkAPI.getLiquidityPools();
          const pools = poolsResp.data || [];
          if (action.params.pair) {
            const pair = action.params.pair.toUpperCase().replace(/\s+/g, '');
            const pool = pools.find?.((p: any) => {
              const t0 = (p.token0 || p.token_a || '').toUpperCase();
              const t1 = (p.token1 || p.token_b || '').toUpperCase();
              return pair.includes(t0) && pair.includes(t1) ||
                `${t0}/${t1}`.includes(pair) || `${t1}/${t0}`.includes(pair);
            });
            result = { success: true, data: pool || { pair, info: 'Pool not found', available_pools: pools.length } };
          } else {
            result = { success: true, data: { pools_count: pools.length, pools: pools.slice(0, 10) } };
          }
          break;
        }
        case 'mail': {
          // Send in-node wallet mail
          let toAddress = action.params.to || '';
          if (action.params.to_name) {
            const resolved = resolveNameToAddress(action.params.to_name);
            if (resolved) {
              toAddress = resolved.address;
            } else {
              throw new Error(`Contact "${action.params.to_name}" not found in address book`);
            }
          }
          if (!toAddress) throw new Error('No recipient specified');
          const mailResp = await fetch('/api/v1/mail/send', {
            method: 'POST',
            headers: authHeaders,
            body: JSON.stringify({
              to: toAddress,
              subject: action.params.subject || 'Message from AI Chat',
              body: action.params.body || action.params.message || '',
            }),
          });
          const mailData = await mailResp.json();
          result = mailData.success
            ? { success: true, data: { message: 'Mail sent successfully', ...mailData.data } }
            : { success: false, error: mailData.error || 'Failed to send mail' };
          break;
        }
        case 'mint': {
          // Mint QUGUSD stablecoin with QUG collateral
          const collateralAmount = parseFloat(action.params.collateral || action.params.amount || '0');
          if (collateralAmount <= 0) throw new Error('Invalid collateral amount');
          const mintResp = await fetch('/api/v1/stablecoin/mint', {
            method: 'POST',
            headers: authHeaders,
            body: JSON.stringify({
              collateral_amount: collateralAmount,
              wallet_address: walletAddress,
            }),
          });
          const mintData = await mintResp.json();
          result = mintData.success
            ? { success: true, data: mintData.data }
            : { success: false, error: mintData.error || 'Minting failed' };
          break;
        }
        case 'portfolio': {
          // v7.4.1: Use authenticated qnkAPI service (not raw fetch)
          const [balResp2, tokResp2, priceResp2] = await Promise.all([
            qnkAPI.getWalletBalance(walletAddress),
            qnkAPI.getMultiTokenBalance(),
            qnkAPI.getOraclePrice('QUG/USD'),
          ]);
          const qugBalance = parseFloat(balResp2.data?.balance || balResp2.data?.confirmed || '0');
          const qugPrice = parseFloat(priceResp2.data?.price || (priceResp2 as any)?.price || '3000.00');
          // v7.4.1: tokens is a HashMap/Object, NOT an array
          const tokensMap2 = tokResp2.data?.tokens || {};
          let totalValueUsd = qugBalance * qugPrice;
          const tokenSummary = Object.entries(tokensMap2).map(([symbol, t]: [string, any]) => {
            const val = parseFloat(t.usd_value || '0');
            totalValueUsd += val;
            return { symbol, balance: t.balance, value_usd: val.toFixed(2) };
          });
          result = {
            success: true,
            data: {
              qug_balance: qugBalance.toFixed(4),
              qug_price_usd: qugPrice.toFixed(2),
              qug_value_usd: (qugBalance * qugPrice).toFixed(2),
              tokens: tokenSummary,
              total_portfolio_usd: totalValueUsd.toFixed(2),
            }
          };
          break;
        }
        case 'top_tokens': {
          const resp = await fetch('/api/v1/dex/supported-tokens', { headers: authHeaders });
          const data = await resp.json();
          const tokens = data.data || [];
          const sortBy = (action.params.sort || 'volume').toLowerCase();
          const limit = parseInt(action.params.limit || '10');
          const sorted = [...tokens].sort((a: any, b: any) => {
            if (sortBy === 'price') return parseFloat(b.price_usd || '0') - parseFloat(a.price_usd || '0');
            if (sortBy === 'mcap' || sortBy === 'market_cap') return parseFloat(b.market_cap || '0') - parseFloat(a.market_cap || '0');
            return parseFloat(b.volume_24h || '0') - parseFloat(a.volume_24h || '0');
          }).slice(0, limit);
          result = { success: true, data: { tokens: sorted, sort_by: sortBy, count: sorted.length } };
          break;
        }
        default:
          result = { success: false, error: `Unknown action type: ${action.type}` };
      }

      // Update action with result
      const completedAction = { ...action, status: 'completed' as const, result };
      setPendingActions(prev => ({ ...prev, [action.id]: completedAction }));
      setMessages(prev => prev.map(m => ({
        ...m,
        cryptoActions: m.cryptoActions?.map(a =>
          a.id === action.id ? completedAction : a
        )
      })));
    } catch (error: any) {
      const failedAction = { ...action, status: 'failed' as const, result: { success: false, error: error.message || String(error) } };
      setPendingActions(prev => ({ ...prev, [action.id]: failedAction }));
      setMessages(prev => prev.map(m => ({
        ...m,
        cryptoActions: m.cryptoActions?.map(a =>
          a.id === action.id ? failedAction : a
        )
      })));
    }
  };

  // ✅ v7.3.3 - Cancel a crypto action
  const cancelCryptoAction = (actionId: string) => {
    setPendingActions(prev => ({ ...prev, [actionId]: { ...prev[actionId], status: 'cancelled' } }));
    setMessages(prev => prev.map(m => ({
      ...m,
      cryptoActions: m.cryptoActions?.map(a =>
        a.id === actionId ? { ...a, status: 'cancelled' as const } : a
      )
    })));
  };

  // ✅ v7.3.3 - Build BitNet system prompt with financial commands
  const buildCryptoSystemPrompt = (): string => {
    const contacts = addressBook.slice(0, 20).map(c => c.label).join(', ');
    return `You are QNK Assistant, an AI financial assistant for the Q-NarwhalKnight blockchain. You are powered by BitNet b1.58 (1-bit quantized 2B parameter model).

CAPABILITIES:
You can help users with blockchain transactions, token swaps, balance checks, and price queries. When the user requests a financial action, output an ACTION tag that the system will parse and execute.

ACTION FORMAT (output these EXACTLY when the user requests an action):
[ACTION:send amount=<number> token=<symbol> to=<address> to_name=<name> memo=<optional>]
[ACTION:swap amount=<number> from=<symbol> to=<symbol>]
[ACTION:balance token=<symbol_or_all>]
[ACTION:price token=<symbol>]
[ACTION:history limit=<number>]
[ACTION:pool_info pair=<TOKEN_A/TOKEN_B>]
[ACTION:mail to_name=<name> subject=<subject> body=<message>]
[ACTION:mint collateral=<qug_amount>]
[ACTION:portfolio]
[ACTION:top_tokens limit=<number> sort=<volume|price|mcap>]

RULES:
- Native coin is QUG. Stablecoin is QUGUSD (USD-pegged, minted with QUG collateral).
- For "send to <name>", use to_name=<name> (system resolves from address book).${contacts ? `\n- Known contacts: ${contacts}` : ''}
- For swaps, "buy X with Y" means from=Y to=X. "sell X for Y" means from=X to=Y.
- "mint stablecoin" / "mint QUGUSD" uses QUG as collateral. Ask how much QUG to lock.
- "mail" or "message" sends in-node encrypted P2P mail to a contact or address.
- Always confirm the action details in your text response BEFORE the ACTION tag.
- If the user's intent is ambiguous, ask for clarification instead of guessing.
- For non-financial questions, respond normally without ACTION tags.
- Keep responses concise and helpful. Use markdown formatting.

MINING KNOWLEDGE:
- Quillon uses dual-lane mining (v10.3.5+): GPU BLAKE3 lane + CPU VDF lane, 50/50 reward split.
- CPU VDF lane: sequential BLAKE3 × 4,300 iterations — CPUs compete fairly, no GPU needed for 50% of rewards.
- GPU lane: massively parallel BLAKE3 hashing via OpenCL (AMD + NVIDIA).
- Difficulty: LWMA (Linear Weighted Moving Average) per-block adjustment.
- Download miner: quillon.xyz/downloads. Run: ./q-miner --server https://quillon.xyz --wallet <address>
- Max supply: 21M QUG, ~2.6M/year emission (Era 0), 4-year halving like Bitcoin.

EXAMPLES:
User: "send 50 QUG to alice"
Response: "I'll send **50 QUG** to **Alice** from your wallet.\n\n[ACTION:send amount=50 token=QUG to_name=alice]"

User: "buy 100 BORK with QUG"
Response: "Swapping **100 QUG** for **BORK** tokens on the DEX.\n\n[ACTION:swap amount=100 from=QUG to=BORK]"

User: "what's my balance?"
Response: "Let me check your wallet balance.\n\n[ACTION:balance token=all]"

User: "how much is QUG worth?"
Response: "Checking the current QUG price.\n\n[ACTION:price token=QUG]"

User: "show my portfolio"
Response: "Here's your full portfolio breakdown.\n\n[ACTION:portfolio]"

User: "what are the top tokens?"
Response: "Let me fetch the top tokens by trading volume.\n\n[ACTION:top_tokens limit=10 sort=volume]"

User: "mint 100 QUGUSD"
Response: "Minting **QUGUSD** stablecoins with **100 QUG** as collateral.\n\n[ACTION:mint collateral=100]"

User: "send a message to bob saying meeting at 3pm"
Response: "I'll send a mail to **Bob** through the node's P2P messaging.\n\n[ACTION:mail to_name=bob subject=Meeting body=Meeting at 3pm]"`;
  };

  // Gemma4 via Ollama — streaming from /api/v1/ai/chat with live network context
  const sendGemma4Message = async (userMessage: string) => {
    setIsGenerating(true);
    setStreamingMessage('');

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userMessage,
      timestamp: Date.now() / 1000,
    };
    setMessages(prev => [...prev, userMsg]);

    // Build conversation history for multi-turn context (last 20 messages)
    const history = [...messages.slice(-20), userMsg].map(m => ({
      role: m.role,
      content: m.content,
    }));

    const walletAddress = localStorage.getItem('walletAddress') || undefined;
    const startTime = Date.now();

    try {
      const response = await fetch('/api/v1/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history, wallet: walletAddress }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const reader = response.body?.getReader();
      if (!reader) throw new Error('ReadableStream not supported');

      const decoder = new TextDecoder();
      let buffer = '';
      let accumulated = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let eventType = '';
        for (const line of lines) {
          if (line.startsWith('event:')) { eventType = line.slice(6).trim(); continue; }
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (!data) continue;
          try {
            const parsed = JSON.parse(data);
            if (eventType === 'token' && parsed.content) {
              accumulated += parsed.content;
              setStreamingMessage(accumulated);
            } else if (eventType === 'error') {
              throw new Error(parsed.message || 'AI error');
            }
          } catch { /* skip malformed */ }
        }
      }

      const elapsed = Date.now() - startTime;
      const tokenCount = Math.round(accumulated.length / 4);
      const assistantMsg: Message = {
        id: `gemma4-${Date.now()}`,
        role: 'assistant',
        content: accumulated || '*(no response)*',
        timestamp: Date.now() / 1000,
        stats: { tokens: tokenCount, latency_ms: elapsed, tokens_per_second: tokenCount / (elapsed / 1000) },
      };
      setMessages(prev => [...prev, assistantMsg]);
      setStreamingMessage('');
    } catch (error: any) {
      const errMsg: Message = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `**Error:** ${error?.message || 'Failed to reach AI'}`,
        timestamp: Date.now() / 1000,
      };
      setMessages(prev => [...prev, errMsg]);
      setStreamingMessage('');
    } finally {
      setIsGenerating(false);
    }
  };

  // BitNet b1.58-2B-4T streaming via OpenAI-compatible API
  const sendBitNetMessage = async (userMessage: string) => {
    setIsGenerating(true);
    setStreamingMessage('');
    setStreamingReasoning('');

    // Build conversation history from current messages for context
    const conversationMessages: { role: string; content: string }[] = [
      {
        role: 'system',
        content: buildCryptoSystemPrompt()
      },
      ...messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      { role: 'user', content: userMessage }
    ];

    // Add user message to UI immediately
    const tempUserMessage: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: userMessage,
      timestamp: Date.now() / 1000,
    };
    setMessages(prev => [...prev, tempUserMessage]);

    try {
      const response = await fetch('/bitnet-api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'bitnet-b1.58-2B-4T',
          messages: conversationMessages,
          stream: true,
          max_tokens: maxTokens,
          temperature: temperature,
          top_p: topP,
          stop: ['<|end|>', '<|user|>', '<|assistant|>', '<|endoftext|>']
        })
      });

      if (!response.ok) {
        throw new Error(`BitNet API error: HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('ReadableStream not supported');

      const decoder = new TextDecoder();
      let buffer = '';
      let cumulativeText = '';
      const startTime = Date.now();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;

          const data = trimmed.substring(5).trim();
          if (data === '[DONE]') {
            // Stream complete
            const elapsed = Date.now() - startTime;
            const tokensEstimate = cumulativeText.split(/\s+/).length;
            console.log(`✅ BitNet complete: ~${tokensEstimate} tokens in ${elapsed}ms`);

            // ✅ v7.3.3 - Parse crypto actions from response
            const detectedActions = parseCryptoActions(cumulativeText);
            const cleanContent = stripActionTags(cumulativeText);

            // Auto-execute read-only actions (balance, price, history, pool_info)
            const readOnlyTypes: CryptoActionType[] = ['balance', 'price', 'history', 'pool_info', 'portfolio', 'top_tokens'];
            for (const action of detectedActions) {
              if (readOnlyTypes.includes(action.type)) {
                action.status = 'executing';
                // Execute immediately without confirmation
                executeCryptoAction(action);
              } else {
                action.status = 'confirming';
              }
            }

            const assistantMessage: Message = {
              id: `bitnet-${Date.now()}`,
              role: 'assistant',
              content: cleanContent,
              timestamp: Date.now() / 1000,
              cryptoActions: detectedActions.length > 0 ? detectedActions : undefined,
              stats: {
                tokens: tokensEstimate,
                latency_ms: elapsed,
                tokens_per_second: (tokensEstimate / elapsed) * 1000
              }
            };
            setMessages(prev => [...prev, assistantMessage]);
            setStreamingMessage('');
            setIsGenerating(false);
            return;
          }

          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta;
            if (delta?.content) {
              // Filter BitNet special tokens
              let content = delta.content;
              content = content.replace(/<\|end\|>/g, '');
              content = content.replace(/<\|user\|>/g, '');
              content = content.replace(/<\|assistant\|>/g, '');
              content = content.replace(/<\|endoftext\|>/g, '');
              if (content) {
                cumulativeText += content;
                // Strip [ACTION:...] tags from live streaming display
                setStreamingMessage(stripActionTags(cumulativeText));
              }
            }
          } catch {
            // Skip unparseable lines
          }
        }
      }

      // Stream ended without [DONE] - save what we have (with action parsing)
      if (cumulativeText) {
        const detectedActions = parseCryptoActions(cumulativeText);
        const cleanContent = stripActionTags(cumulativeText);
        const readOnlyTypes: CryptoActionType[] = ['balance', 'price', 'history', 'pool_info', 'portfolio', 'top_tokens'];
        for (const action of detectedActions) {
          if (readOnlyTypes.includes(action.type)) {
            executeCryptoAction(action);
          } else {
            action.status = 'confirming';
          }
        }
        const assistantMessage: Message = {
          id: `bitnet-${Date.now()}`,
          role: 'assistant',
          content: cleanContent,
          timestamp: Date.now() / 1000,
          cryptoActions: detectedActions.length > 0 ? detectedActions : undefined,
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('BitNet streaming error:', error);
      setStreamingMessage(`BitNet Error: ${error}`);
      setTimeout(() => setStreamingMessage(''), 5000);
    } finally {
      setIsGenerating(false);
      setStreamingMessage('');
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return;

    // Auto-create chat if none exists (for ALL models including BitNet)
    if (!currentChatId) {
      console.log('📝 No current chat, auto-creating before send...');
      try {
        const userId = localStorage.getItem('walletAddress') || 'default';
        const response = await fetch('/api/chat/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            title: 'New Chat',
            encryption_enabled: true,
            distributed_enabled: true,
            enable_kv_cache: true
          })
        });
        const data = await response.json();
        if (data.success && data.data) {
          const newChatId = data.data.chat_id;
          currentChatIdRef.current = newChatId;
          setCurrentChatId(newChatId);
          setMessages([]);
          loadChats(false);
          console.log(`✅ Auto-created chat: ${newChatId}`);
        }
      } catch (error) {
        console.error('❌ Failed to auto-create chat:', error);
      }
    }

    // ✅ v7.4.1: Pre-process clear financial intents BEFORE calling any AI model
    // This bypasses the AI entirely for unambiguous commands like "balance", "portfolio", "price of QUG"
    const directIntent = parseUserIntent(input);
    if (directIntent) {
      const userMessage = input;
      setInput('');
      // Add user message to chat
      const userMsg: Message = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: userMessage,
        timestamp: Date.now() / 1000,
      };
      setMessages(prev => [...prev, userMsg]);

      // Determine if action should auto-execute or need confirmation
      const readOnlyTypes: CryptoActionType[] = ['balance', 'price', 'history', 'pool_info', 'portfolio', 'top_tokens'];
      if (readOnlyTypes.includes(directIntent.type)) {
        directIntent.status = 'executing';
        executeCryptoAction(directIntent);
      } else {
        directIntent.status = 'confirming';
      }

      // Add assistant message with the action card
      const assistantMsg: Message = {
        id: `intent-${Date.now()}`,
        role: 'assistant',
        content: readOnlyTypes.includes(directIntent.type)
          ? `Executing ${directIntent.type.replace('_', ' ')}...`
          : `Ready to ${directIntent.type}. Please confirm:`,
        timestamp: Date.now() / 1000,
        cryptoActions: [directIntent],
      };
      setMessages(prev => [...prev, assistantMsg]);
      return; // Skip AI model entirely
    }

    // Gemma4 (Ollama) — live network context, handles hashrate + general questions
    if (selectedModel === 'Gemma4-Ollama') {
      const userMessage = input;
      setInput('');
      await sendGemma4Message(userMessage);
      return;
    }

    // BitNet uses its own streaming path (OpenAI-compatible API)
    if (selectedModel === 'BitNet-b1.58-2B-4T') {
      const userMessage = input;
      setInput('');
      await sendBitNetMessage(userMessage);
      return;
    }

    // ✅ v0.9.36-beta - Check if this is a transaction request
    const isTransaction = await detectAndPrepareTransaction(input);
    if (isTransaction) {
      return; // Transaction preview is shown, wait for user confirmation
    }

    const userMessage = input;
    setInput('');
    setIsGenerating(true);
    setStreamingMessage('');

    // Auto-create chat if none exists (UX improvement: no manual "New Chat" click needed)
    let chatId = currentChatId;
    if (!chatId) {
      console.log('📝 No current chat, auto-creating...');
      try {
        const userId = localStorage.getItem('walletAddress') || 'default';
        const response = await fetch('/api/chat/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            title: 'New Chat',
            encryption_enabled: true,
            distributed_enabled: true,
            enable_kv_cache: true
          })
        });

        const data = await response.json();
        if (data.success && data.data) {
          chatId = data.data.chat_id;
          // CRITICAL: Update ref FIRST before state
          currentChatIdRef.current = chatId;
          setCurrentChatId(chatId);
          setMessages([]);
          loadChats(false); // Refresh chat list without auto-select
          console.log(`✅ Auto-created chat: ${chatId}`);
        } else {
          throw new Error(data.error || 'Failed to create chat');
        }
      } catch (error) {
        console.error('❌ Failed to auto-create chat:', error);
        setIsGenerating(false);
        setStreamingMessage('Failed to create chat. Please try clicking "New Chat" manually.');
        setTimeout(() => setStreamingMessage(''), 5000);
        return;
      }
    }

    // Track active generation in localStorage for background support
    localStorage.setItem('activeAIGeneration', JSON.stringify({
      chatId: chatId,
      startTime: Date.now(),
      prompt: userMessage
    }));

    // Close any existing event source
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // CRITICAL: Load existing messages from database FIRST to ensure we have the latest state
    // This prevents wiping out previous messages when loadMessages is called later
    const loadedMessages = await new Promise<Message[]>((resolve) => {
      fetch(`/api/chat/${chatId}/messages`)
        .then(res => res.json())
        .then(data => {
          if (data.success && data.data && Array.isArray(data.data)) {
            resolve(data.data);
          } else {
            resolve([]);
          }
        })
        .catch(() => resolve([]));
    });

    // Optimistically add user message to UI immediately
    const tempUserMessage: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: userMessage,
      timestamp: Date.now() / 1000,
    };
    setMessages([...loadedMessages, tempUserMessage]);

    try {
      // ✅ v1.0.2: Use regular streaming endpoint (works with or without distributed workers)
      // Falls back to local inference automatically if no workers available
      const response = await fetch(`/api/chat/${chatId}/stream?content=${encodeURIComponent(userMessage)}&max_tokens=${maxTokens}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('ReadableStream not supported');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let cumulativeText = '';
      let workerNodeId = '';

      // Read SSE stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEventType = '';
        for (const line of lines) {
          if (line.startsWith('event:')) {
            // Track event type for the next data line
            currentEventType = line.substring(6).trim();
            continue;
          }

          if (line.startsWith('data:')) {
            const data = line.substring(5).trim();
            if (!data) continue;

            try {
              const parsed = JSON.parse(data);

              // Handle different event types
              if (parsed.mode === 'data_parallel') {
                // Started event
                workerNodeId = parsed.worker_node;
                console.log(`🌊 Data parallel stream started on worker: ${workerNodeId}`);
              } else if (parsed.reasoning !== undefined) {
                // Reasoning event (Kimi K2 thinking process)
                setStreamingReasoning((prev) => prev + parsed.reasoning);
                console.log(`🧠 Reasoning: ${parsed.reasoning}`);
              } else if (parsed.token !== undefined) {
                // Token event
                cumulativeText += parsed.token;
                setStreamingMessage(cumulativeText);
              } else if (parsed.finish_reason || currentEventType === 'complete') {
                // Complete event — detected by finish_reason field OR event: complete SSE type
                console.log(`✅ Complete: ${parsed.total_tokens || parsed.tokens_generated} tokens in ${parsed.total_time_ms}ms`);
                console.log(`   Throughput: ${parsed.tokens_per_second} tok/s`);
                console.log(`   Engine: ${parsed.engine || 'unknown'}`);

                // DON'T clear streaming message yet - keep it visible while loading from DB
                setIsGenerating(false);

                // Clear background generation tracking
                localStorage.removeItem('activeAIGeneration');

                // Reload messages from backend to get the complete conversation
                await loadMessages(chatId!);

                // NOW clear streaming message and reasoning after database messages are loaded
                setStreamingMessage('');
                setStreamingReasoning('');

                // If this is the first message, generate a title
                const currentChat = chats.find(c => c.chat_id === chatId);
                if (currentChat && currentChat.message_count === 0 && userMessage) {
                  generateChatTitle(chatId!, userMessage);
                }
                break;
              } else if (parsed.code) {
                // Error event
                console.error('❌ Stream error:', parsed.message);
                setIsGenerating(false);
                localStorage.removeItem('activeAIGeneration');
                setStreamingMessage(`Error: ${parsed.message}`);
                setTimeout(() => setStreamingMessage(''), 5000);
                break;
              }
            } catch (error) {
              console.error('Failed to parse SSE data:', error);
            }
          }
        }
      }

      // ✅ v1.4.2 FIX: Safety cleanup when stream ends without finish_reason
      // This handles cases where connection closes unexpectedly
      if (isGenerating) {
        console.log('⚠️ Stream ended without finish_reason, cleaning up...');
        setIsGenerating(false);
        localStorage.removeItem('activeAIGeneration');
        backgroundGenerationRef.current = false;

        // If we have streaming content, try to save it
        if (cumulativeText) {
          console.log('📝 Preserving streamed content...');
          // Reload messages to get any saved content
          await loadMessages(chatId!);
        }
        setStreamingMessage('');
        setStreamingReasoning('');
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      setIsGenerating(false);
      localStorage.removeItem('activeAIGeneration');
      backgroundGenerationRef.current = false;
      setStreamingMessage(`Failed to send message: ${error}`);
      setTimeout(() => setStreamingMessage(''), 5000);
    }
  };

  return (
    <div className="h-full flex">
      {/* Sidebar - Chat History */}
      <motion.div
        className="w-80 border-r flex flex-col"
        style={{
          background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%)',
          borderColor: 'rgba(212, 175, 55, 0.2)'
        }}
        initial={{ x: -320 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="p-6 border-b border-amber-500/20">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center p-1">
              <img
                src="/quantum-ai-logo.svg"
                alt="Quantum AI"
                className="w-full h-full object-contain"
                style={{
                  filter: 'drop-shadow(0 0 10px rgba(168, 85, 247, 0.4))'
                }}
              />
            </div>
            <div>
              <h2 className="text-xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent">
                AI Chat
              </h2>
              <p className="text-xs text-amber-200/60">Quantum-Enhanced AI</p>
            </div>
          </div>

          <button
            onClick={createNewChat}
            className="w-full flex items-center justify-center gap-2 p-3 rounded-xl transition-all"
            style={{
              background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.15) 100%)',
              border: '2px solid rgba(212, 175, 55, 0.4)',
              boxShadow: '0 0 20px rgba(212, 175, 55, 0.2)'
            }}
          >
            <Plus className="w-5 h-5" />
            <span className="font-medium">New Chat</span>
          </button>
        </div>

        {/* Ask Me Anything Input - Always visible at top of sidebar */}
        <div
          className="px-4 py-3 border-b"
          style={{
            background: 'linear-gradient(180deg, rgba(20, 30, 50, 0.95) 0%, rgba(25, 35, 55, 0.95) 100%)',
            borderColor: 'rgba(212, 175, 55, 0.15)'
          }}
        >
          {/* Max Tokens Slider - Compact */}
          <div className="mb-2 flex items-center gap-2">
            <label className="text-amber-300 text-xs font-medium whitespace-nowrap">
              Tokens: {maxTokens}
            </label>
            <input
              type="range"
              min="50"
              max="2048"
              step="50"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="flex-1 h-1.5 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #D4AF37 0%, #D4AF37 ${((maxTokens - 50) / (2048 - 50)) * 100}%, rgba(30, 41, 59, 0.5) ${((maxTokens - 50) / (2048 - 50)) * 100}%, rgba(30, 41, 59, 0.5) 100%)`
              }}
            />
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="Ask me anything..."
              disabled={isGenerating}
              className="flex-1 px-3 py-2.5 rounded-lg text-amber-50 text-sm placeholder-amber-200/40 focus:outline-none transition-all disabled:opacity-50"
              style={{
                background: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid rgba(212, 175, 55, 0.2)',
                boxShadow: '0 0 10px rgba(212, 175, 55, 0.05)'
              }}
            />
            {isGenerating ? (
              <button
                onClick={stopGeneration}
                className="px-3 py-2.5 rounded-lg font-medium transition-all hover:scale-105"
                style={{
                  background: 'linear-gradient(135deg, #DC2626 0%, #EF4444 50%, #DC2626 100%)',
                  color: '#FFF',
                  boxShadow: '0 0 15px rgba(220, 38, 38, 0.3)'
                }}
                title="Stop generation"
              >
                <Square className="w-5 h-5" />
              </button>
            ) : (
              <button
                data-send-button
                onClick={sendMessage}
                disabled={!input.trim()}
                className="px-3 py-2.5 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105"
                style={{
                  background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                  color: '#0F172A',
                  boxShadow: '0 0 15px rgba(212, 175, 55, 0.3)'
                }}
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {chats.map((chat) => (
            <motion.button
              key={chat.chat_id}
              onClick={() => {
                // Clean up streaming UI state when switching chats
                // BUT preserve the localStorage marker so we can reconnect when coming back
                if (eventSourceRef.current) {
                  eventSourceRef.current.close();
                  eventSourceRef.current = null;
                }
                setStreamingMessage('');
                setStreamingReasoning('');

                // Always clear isGenerating when switching chats
                // The localStorage marker (activeAIGeneration) is preserved
                // so the reconnection logic will kick in when user returns
                setIsGenerating(false);
                backgroundGenerationRef.current = false;

                // CRITICAL: Update ref FIRST before calling setCurrentChatId or loadMessages
                // This ensures async operations know which chat is current
                currentChatIdRef.current = chat.chat_id;

                // Now switch to the new chat
                setCurrentChatId(chat.chat_id);
                loadMessages(chat.chat_id);
              }}
              className={`w-full text-left p-3 rounded-xl transition-all group relative ${
                currentChatId === chat.chat_id ? 'text-amber-50' : 'text-amber-200/70 hover:text-amber-100'
              }`}
              style={
                currentChatId === chat.chat_id
                  ? {
                      background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(255, 215, 0, 0.1) 100%)',
                      border: '1px solid rgba(212, 175, 55, 0.3)'
                    }
                  : {
                      border: '1px solid transparent'
                    }
              }
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{chat.title}</p>
                  <p className="text-xs text-amber-200/50 mt-1">
                    {chat.message_count} messages
                  </p>
                </div>
                <button
                  onClick={(e) => deleteChat(chat.chat_id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                >
                  <Trash2 className="w-4 h-4 text-red-400" />
                </button>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div
          className="p-6 border-b flex items-center justify-between"
          style={{
            background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
            borderColor: 'rgba(212, 175, 55, 0.2)'
          }}
        >
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-400" />
              <span className="font-medium text-amber-200">
                {selectedModel.includes('Small') || selectedModel.includes('24B')
                  ? 'Mistral Small 24B'
                  : selectedModel.includes('Ministral-3B')
                  ? 'Ministral 3B'
                  : selectedModel.includes('Qwen3')
                  ? 'Qwen3 VL 8B'
                  : 'Mistral 7B'}
              </span>
              <span className="text-xs text-amber-400/60">
                ({selectedModel.includes('Small') || selectedModel.includes('24B') ? '14 GB' : selectedModel.includes('Ministral-3B') ? '2.1 GB' : selectedModel.includes('Qwen3') ? '5.1 GB' : '4.3 GB'})
              </span>
            </div>
            <div className="flex items-center gap-4 text-sm text-amber-200/60">
              <div className="flex items-center gap-1">
                <Zap className="w-4 h-4" />
                <span>Fast</span>
              </div>
              <div className="flex items-center gap-1">
                <Shield className="w-4 h-4" />
                <span>Private</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowMetrics(true)}
              className={`p-2 rounded-lg hover:bg-purple-500/10 transition-all relative ${
                (workersData?.total_workers > 1 || metricsData?.distributed?.nodes_participated > 1) ? 'animate-pulse' : ''
              }`}
              title={`AI Performance Metrics${
                workersData?.total_workers > 1
                  ? ` - Tensor Parallelism: ${workersData.total_workers} Nodes = ~${(Math.min(workersData.total_workers, 8) * 0.75 + 0.25).toFixed(1)}x Faster!`
                  : metricsData?.distributed?.nodes_participated > 1
                  ? ` - ${metricsData.distributed.nodes_participated} Nodes Active!`
                  : ''
              }`}
            >
              <Activity
                className={`w-5 h-5 ${
                  (workersData?.total_workers > 1 || metricsData?.distributed?.nodes_participated > 1)
                    ? 'text-purple-400 drop-shadow-[0_0_8px_rgba(168,85,247,0.8)]'
                    : 'text-purple-400'
                }`}
              />
              {(workersData?.total_workers > 1 || metricsData?.distributed?.nodes_participated > 1) && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-500 rounded-full animate-ping" />
              )}
              {(workersData?.total_workers > 1 || metricsData?.distributed?.nodes_participated > 1) && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-500 rounded-full" />
              )}
            </button>
            <button
              onClick={() => setShowCostsUsage(true)}
              className="p-2 rounded-lg hover:bg-amber-500/10 transition-all"
              title="Costs & Usage"
            >
              <DollarSign className="w-5 h-5 text-amber-400" />
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-lg hover:bg-amber-500/10 transition-all"
              title="Settings"
            >
              <Settings className="w-5 h-5 text-amber-400" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={messagesContainerRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto p-6 space-y-4"
        >
          {!currentChatId ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-xl">
                <div className="relative inline-block mb-6">
                  <div className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto" style={{
                    background: 'linear-gradient(135deg, rgba(212,175,55,0.2), rgba(0,229,255,0.15))',
                    border: '2px solid rgba(212,175,55,0.3)',
                    boxShadow: '0 0 40px rgba(212,175,55,0.15)',
                  }}>
                    <Bot className="w-10 h-10 text-amber-400" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center" style={{
                    background: 'linear-gradient(135deg, #22C55E, #16A34A)',
                    boxShadow: '0 0 8px rgba(34,197,94,0.5)',
                  }}>
                    <Zap className="w-3 h-3 text-white" />
                  </div>
                </div>
                <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 bg-clip-text text-transparent">
                  QNK Financial Assistant
                </h3>
                <p className="text-amber-200/50 text-sm mb-8">
                  {selectedModel === 'Gemma4-Ollama' ? 'Powered by Gemma 4 — knows live hashrate, block height, supply and more' : 'Powered by BitNet b1.58 — ask questions, execute transactions, swap tokens, check balances'}
                </p>

                {/* Command Suggestion Chips */}
                <div className="grid grid-cols-2 gap-3 mb-6">
                  {[
                    { icon: <Send className="w-4 h-4" />, label: 'Send QUG', example: 'Send 50 QUG to alice', color: '#22C55E' },
                    { icon: <ArrowUpDown className="w-4 h-4" />, label: 'Swap Tokens', example: 'Buy 100 BORK with QUG', color: '#3B82F6' },
                    { icon: <PieChart className="w-4 h-4" />, label: 'My Portfolio', example: 'Show my portfolio', color: '#FFD700' },
                    { icon: <TrendingUp className="w-4 h-4" />, label: 'Token Price', example: 'How much is QUG worth?', color: '#00E5FF' },
                    { icon: <Coins className="w-4 h-4" />, label: 'Mint QUGUSD', example: 'Mint stablecoin with 100 QUG', color: '#10B981' },
                    { icon: <Trophy className="w-4 h-4" />, label: 'Top Tokens', example: 'Show top tokens by volume', color: '#F59E0B' },
                    { icon: <Mail className="w-4 h-4" />, label: 'Send Mail', example: 'Send message to alice about meeting', color: '#EC4899' },
                    { icon: <Wallet className="w-4 h-4" />, label: 'Check Balance', example: "What's my balance?", color: '#8B5CF6' },
                    { icon: <Clock className="w-4 h-4" />, label: 'TX History', example: 'Show my last 5 transactions', color: '#A855F7' },
                    { icon: <Layers className="w-4 h-4" />, label: 'Pool Info', example: 'Show QUG/QUGUSD pool', color: '#E040FB' },
                  ].map((cmd) => (
                    <motion.button
                      key={cmd.label}
                      whileHover={{ scale: 1.03, y: -2 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => {
                        // Auto-send: set input and trigger sendMessage
                        setInput(cmd.example);
                        // Use a small delay to ensure state updates, then auto-send
                        setTimeout(() => {
                          const sendBtn = document.querySelector('[data-send-button]') as HTMLButtonElement;
                          if (sendBtn) sendBtn.click();
                        }, 100);
                      }}
                      className="flex items-center gap-3 p-3.5 rounded-xl text-left transition-all group"
                      style={{
                        background: `linear-gradient(135deg, ${cmd.color}10 0%, ${cmd.color}08 100%)`,
                        border: `1px solid ${cmd.color}30`,
                      }}
                    >
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-all group-hover:scale-110" style={{
                        background: `${cmd.color}20`,
                        color: cmd.color,
                      }}>
                        {cmd.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-amber-100">{cmd.label}</p>
                        <p className="text-xs text-amber-200/40 truncate">{cmd.example}</p>
                      </div>
                    </motion.button>
                  ))}
                </div>

                <p className="text-xs text-amber-200/30">
                  Type naturally — the AI understands your intent and executes blockchain actions
                </p>
              </div>
            </div>
          ) : (
            <>
              <AnimatePresence>
                {Array.isArray(messages) && messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.role === 'assistant' && (
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{
                          background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)'
                        }}
                      >
                        <Bot className="w-5 h-5 text-slate-900" />
                      </div>
                    )}

                    <div className={`max-w-2xl ${message.role === 'user' ? 'order-first' : ''}`}>
                      <div
                        className="p-4 rounded-2xl"
                        style={
                          message.role === 'user'
                            ? {
                                background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.15) 100%)',
                                border: '1px solid rgba(212, 175, 55, 0.3)'
                              }
                            : {
                                background: 'rgba(30, 41, 59, 0.5)',
                                border: '1px solid rgba(212, 175, 55, 0.1)'
                              }
                        }
                      >
                        {message.role === 'assistant' ? (
                          <div className="text-amber-50 leading-relaxed prose prose-invert prose-amber max-w-none">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              rehypePlugins={[rehypeHighlight, rehypeRaw]}
                              components={{
                                code: ({ node, inline, className, children, ...props }: any) => {
                                  const match = /language-(\w+)/.exec(className || '');
                                  const codeString = String(children).replace(/\n$/, '');
                                  const codeIndex = `${message.id}-${match?.[1] || 'code'}-${codeString.slice(0, 20)}`;
                                  return !inline && match ? (
                                    <div className="relative my-4 group">
                                      <div className="absolute top-0 right-0 flex items-center gap-1 px-2 py-1 text-xs bg-slate-800/80 rounded-bl-lg rounded-tr-lg border-l border-b border-amber-500/20">
                                        <span className="text-amber-400">{match[1]}</span>
                                        <button
                                          onClick={() => copyCodeBlock(codeIndex, codeString)}
                                          className="ml-2 p-1 rounded hover:bg-amber-500/20 transition-all"
                                          title="Copy code"
                                        >
                                          {copiedCodeIndex === codeIndex ? (
                                            <Check className="w-3.5 h-3.5 text-green-400" />
                                          ) : (
                                            <Copy className="w-3.5 h-3.5 text-amber-400/70 hover:text-amber-400" />
                                          )}
                                        </button>
                                      </div>
                                      <code
                                        className={`${className} block p-4 pt-8 rounded-lg overflow-x-auto`}
                                        style={{
                                          background: 'rgba(15, 23, 42, 0.8)',
                                          border: '1px solid rgba(212, 175, 55, 0.2)',
                                        }}
                                        {...props}
                                      >
                                        {children}
                                      </code>
                                    </div>
                                  ) : (
                                    <code
                                      className="px-1.5 py-0.5 rounded text-sm"
                                      style={{
                                        background: 'rgba(212, 175, 55, 0.15)',
                                        border: '1px solid rgba(212, 175, 55, 0.3)',
                                        color: '#FCD34D',
                                      }}
                                      {...props}
                                    >
                                      {children}
                                    </code>
                                  );
                                },
                                pre: ({ children }: any) => <div className="not-prose">{children}</div>,
                                p: ({ children }: any) => <p className="mb-3 last:mb-0">{children}</p>,
                                ul: ({ children }: any) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                                ol: ({ children }: any) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                                li: ({ children }: any) => <li className="text-amber-50/90">{children}</li>,
                                h1: ({ children }: any) => <h1 className="text-2xl font-bold text-amber-400 mb-3 mt-4">{children}</h1>,
                                h2: ({ children }: any) => <h2 className="text-xl font-bold text-amber-400 mb-2 mt-3">{children}</h2>,
                                h3: ({ children }: any) => <h3 className="text-lg font-bold text-amber-400 mb-2 mt-3">{children}</h3>,
                                blockquote: ({ children }: any) => (
                                  <blockquote className="border-l-4 border-amber-500/50 pl-4 italic text-amber-200/80 my-3">
                                    {children}
                                  </blockquote>
                                ),
                                a: ({ children, href }: any) => (
                                  <a
                                    href={href}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-amber-400 hover:text-amber-300 underline"
                                  >
                                    {children}
                                  </a>
                                ),
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          /* User Message - with inline edit support */
                          editingMessageId === message.id ? (
                            <div className="space-y-2">
                              <textarea
                                value={editingContent}
                                onChange={(e) => setEditingContent(e.target.value)}
                                className="w-full p-3 rounded-lg text-amber-50 bg-slate-800/50 border border-amber-500/30 focus:outline-none focus:border-amber-400 resize-none"
                                rows={3}
                                autoFocus
                              />
                              <div className="flex gap-2 justify-end">
                                <button
                                  onClick={cancelEditingMessage}
                                  className="px-3 py-1.5 rounded-lg text-sm text-amber-200/70 hover:text-amber-200 hover:bg-amber-500/10 transition-all"
                                >
                                  Cancel
                                </button>
                                <button
                                  onClick={saveEditedMessage}
                                  disabled={!editingContent.trim()}
                                  className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all disabled:opacity-50"
                                  style={{
                                    background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                                    color: '#0F172A'
                                  }}
                                >
                                  Save & Resend
                                </button>
                              </div>
                            </div>
                          ) : (
                            <p className="text-amber-50 whitespace-pre-wrap leading-relaxed">
                              {message.content}
                            </p>
                          )
                        )}

                        {/* Function Call Cards (Ministral-3B Agentic) */}
                        {message.functionCalls && message.functionCalls.length > 0 && (
                          <div className="mt-4 space-y-2">
                            <div className="flex items-center gap-2 text-xs text-violet-400 mb-2">
                              <Wrench className="w-3 h-3" />
                              <span>Function Calls</span>
                            </div>
                            {message.functionCalls.map((fc) => (
                              <motion.div
                                key={fc.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="p-3 rounded-xl bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center gap-2">
                                    {fc.name === 'transfer' && <Wallet className="w-4 h-4 text-green-400" />}
                                    {fc.name === 'swap' && <ArrowUpDown className="w-4 h-4 text-blue-400" />}
                                    {fc.name === 'check_balance' && <DollarSign className="w-4 h-4 text-yellow-400" />}
                                    {fc.name === 'get_price' && <TrendingUp className="w-4 h-4 text-cyan-400" />}
                                    {fc.name === 'analyze_market' && <BarChart3 className="w-4 h-4 text-purple-400" />}
                                    {!['transfer', 'swap', 'check_balance', 'get_price', 'analyze_market'].includes(fc.name) && (
                                      <Wrench className="w-4 h-4 text-violet-400" />
                                    )}
                                    <span className="font-mono text-sm text-white">{fc.name}</span>
                                  </div>
                                  <div className="flex items-center gap-1.5">
                                    {fc.status === 'pending' && (
                                      <span className="flex items-center gap-1 text-xs text-amber-400 px-2 py-0.5 rounded-full bg-amber-500/20">
                                        <Clock className="w-3 h-3" />
                                        Pending
                                      </span>
                                    )}
                                    {fc.status === 'executing' && (
                                      <span className="flex items-center gap-1 text-xs text-blue-400 px-2 py-0.5 rounded-full bg-blue-500/20">
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                        Executing
                                      </span>
                                    )}
                                    {fc.status === 'completed' && (
                                      <span className="flex items-center gap-1 text-xs text-green-400 px-2 py-0.5 rounded-full bg-green-500/20">
                                        <CheckCircle2 className="w-3 h-3" />
                                        Success
                                      </span>
                                    )}
                                    {fc.status === 'failed' && (
                                      <span className="flex items-center gap-1 text-xs text-red-400 px-2 py-0.5 rounded-full bg-red-500/20">
                                        <AlertCircle className="w-3 h-3" />
                                        Failed
                                      </span>
                                    )}
                                  </div>
                                </div>

                                {/* Function Arguments */}
                                <div className="text-xs text-gray-400 font-mono bg-black/30 rounded-lg p-2 mb-2">
                                  {Object.entries(fc.arguments).map(([key, value]) => (
                                    <div key={key} className="flex gap-2">
                                      <span className="text-violet-400">{key}:</span>
                                      <span className="text-gray-300">
                                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                      </span>
                                    </div>
                                  ))}
                                </div>

                                {/* Function Result */}
                                {fc.result && (
                                  <div className={`text-xs rounded-lg p-2 ${
                                    fc.result.success
                                      ? 'bg-green-500/10 border border-green-500/30'
                                      : 'bg-red-500/10 border border-red-500/30'
                                  }`}>
                                    <div className="flex items-center gap-1 mb-1">
                                      <ArrowRight className="w-3 h-3" />
                                      <span className="font-medium">Result:</span>
                                    </div>
                                    {fc.result.success ? (
                                      <pre className="text-green-300 font-mono whitespace-pre-wrap">
                                        {typeof fc.result.data === 'object'
                                          ? JSON.stringify(fc.result.data, null, 2)
                                          : String(fc.result.data)}
                                      </pre>
                                    ) : (
                                      <span className="text-red-300">{fc.result.error}</span>
                                    )}
                                  </div>
                                )}
                              </motion.div>
                            ))}
                          </div>
                        )}

                        {/* ✅ v7.3.3 - Crypto Action Cards */}
                        {message.cryptoActions && message.cryptoActions.length > 0 && (
                          <div className="mt-4 space-y-2">
                            {message.cryptoActions.map((action) => (
                              <motion.div
                                key={action.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="rounded-xl overflow-hidden"
                                style={{
                                  background: action.type === 'send' || action.type === 'swap'
                                    ? 'linear-gradient(135deg, rgba(212,175,55,0.12) 0%, rgba(255,215,0,0.08) 100%)'
                                    : 'linear-gradient(135deg, rgba(0,229,255,0.08) 0%, rgba(124,77,255,0.08) 100%)',
                                  border: `1px solid ${
                                    action.status === 'completed' ? 'rgba(34,197,94,0.4)' :
                                    action.status === 'failed' ? 'rgba(239,68,68,0.4)' :
                                    action.status === 'cancelled' ? 'rgba(100,116,139,0.3)' :
                                    action.status === 'executing' ? 'rgba(59,130,246,0.4)' :
                                    'rgba(212,175,55,0.3)'
                                  }`,
                                }}
                              >
                                <div className="px-4 py-3 flex items-center justify-between gap-3">
                                  <div className="flex items-center gap-3 flex-1 min-w-0">
                                    <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0" style={{
                                      background: action.type === 'send' ? 'rgba(34,197,94,0.2)' :
                                        action.type === 'swap' ? 'rgba(59,130,246,0.2)' :
                                        action.type === 'balance' ? 'rgba(255,215,0,0.2)' :
                                        action.type === 'price' ? 'rgba(0,229,255,0.2)' :
                                        'rgba(168,85,247,0.2)',
                                    }}>
                                      {action.type === 'send' && <Send className="w-4 h-4 text-green-400" />}
                                      {action.type === 'swap' && <ArrowUpDown className="w-4 h-4 text-blue-400" />}
                                      {action.type === 'balance' && <Wallet className="w-4 h-4 text-amber-400" />}
                                      {action.type === 'price' && <TrendingUp className="w-4 h-4 text-cyan-400" />}
                                      {action.type === 'history' && <Clock className="w-4 h-4 text-purple-400" />}
                                      {action.type === 'pool_info' && <Layers className="w-4 h-4 text-fuchsia-400" />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-medium text-amber-50 truncate">{action.displayText}</p>
                                      <p className="text-xs text-amber-200/50 mt-0.5">
                                        {action.status === 'confirming' && 'Awaiting your confirmation'}
                                        {action.status === 'executing' && 'Executing...'}
                                        {action.status === 'completed' && 'Completed'}
                                        {action.status === 'failed' && (action.result?.error || 'Failed')}
                                        {action.status === 'cancelled' && 'Cancelled'}
                                      </p>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-2 flex-shrink-0">
                                    {action.status === 'confirming' && (
                                      <>
                                        <button
                                          onClick={() => executeCryptoAction(action)}
                                          className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all hover:scale-105"
                                          style={{
                                            background: 'linear-gradient(135deg, #22C55E, #16A34A)',
                                            color: '#fff',
                                            boxShadow: '0 0 12px rgba(34,197,94,0.3)',
                                          }}
                                        >
                                          Confirm
                                        </button>
                                        <button
                                          onClick={() => cancelCryptoAction(action.id)}
                                          className="px-3 py-1.5 rounded-lg text-xs font-medium text-red-400 hover:bg-red-500/20 transition-all"
                                          style={{ border: '1px solid rgba(239,68,68,0.3)' }}
                                        >
                                          Cancel
                                        </button>
                                      </>
                                    )}
                                    {action.status === 'executing' && (
                                      <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                                    )}
                                    {action.status === 'completed' && (
                                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                                    )}
                                    {action.status === 'failed' && (
                                      <AlertCircle className="w-5 h-5 text-red-400" />
                                    )}
                                  </div>
                                </div>

                                {/* Result display for completed actions */}
                                {action.status === 'completed' && action.result?.data && (
                                  <div className="px-4 pb-3 border-t" style={{ borderColor: 'rgba(34,197,94,0.15)' }}>
                                    <pre className="text-xs text-green-200/80 font-mono mt-2 whitespace-pre-wrap overflow-x-auto max-h-32 overflow-y-auto" style={{
                                      background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '8px',
                                    }}>
                                      {typeof action.result.data === 'object'
                                        ? JSON.stringify(action.result.data, null, 2)
                                        : String(action.result.data)}
                                    </pre>
                                    {action.result.txHash && (
                                      <p className="text-xs text-green-400 mt-1 font-mono">
                                        TX: {action.result.txHash.substring(0, 24)}...
                                      </p>
                                    )}
                                  </div>
                                )}
                                {action.status === 'failed' && action.result?.error && (
                                  <div className="px-4 pb-3 border-t" style={{ borderColor: 'rgba(239,68,68,0.15)' }}>
                                    <p className="text-xs text-red-300 mt-2">{action.result.error}</p>
                                  </div>
                                )}
                              </motion.div>
                            ))}
                          </div>
                        )}

                        {/* Kimi K2 Reasoning Display (v1.0.5) */}
                        {message.reasoning && (
                          <details className="mt-3 border-l-2 border-purple-400 pl-3">
                            <summary className="cursor-pointer text-sm text-purple-400 hover:text-purple-300 flex items-center gap-2">
                              <Brain className="w-4 h-4" />
                              <span>View Reasoning Process</span>
                              <ChevronDown className="w-4 h-4" />
                            </summary>
                            <div className="mt-2 text-sm text-gray-400 whitespace-pre-wrap font-mono bg-purple-500/5 p-3 rounded">
                              {message.reasoning}
                            </div>
                          </details>
                        )}

                        {message.stats && (
                          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-amber-500/20 text-xs text-amber-200/60">
                            <div className="flex items-center gap-1">
                              <Zap className="w-3 h-3" />
                              <span>{message.stats.tokens_per_second.toFixed(1)} tok/s</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              <span>{(message.stats.latency_ms / 1000).toFixed(1)}s</span>
                            </div>
                          </div>
                        )}

                        {/* ✅ v1.4.2 - Message Action Buttons */}
                        <div className="flex items-center gap-1 mt-3 pt-2 border-t border-amber-500/10">
                          {/* Copy Button */}
                          <button
                            onClick={() => copyMessageContent(message.id, message.content)}
                            className="p-1.5 rounded-lg hover:bg-amber-500/10 transition-all group"
                            title="Copy message"
                          >
                            {copiedMessageId === message.id ? (
                              <Check className="w-4 h-4 text-green-400" />
                            ) : (
                              <Copy className="w-4 h-4 text-amber-200/50 group-hover:text-amber-400" />
                            )}
                          </button>

                          {message.role === 'assistant' && (
                            <>
                              {/* Thumbs Up */}
                              <button
                                onClick={() => submitFeedback(message.id, 'up')}
                                className={`p-1.5 rounded-lg transition-all group ${
                                  messageFeedback[message.id] === 'up'
                                    ? 'bg-green-500/20'
                                    : 'hover:bg-amber-500/10'
                                }`}
                                title="Good response"
                              >
                                <ThumbsUp className={`w-4 h-4 ${
                                  messageFeedback[message.id] === 'up'
                                    ? 'text-green-400'
                                    : 'text-amber-200/50 group-hover:text-amber-400'
                                }`} />
                              </button>

                              {/* Thumbs Down */}
                              <button
                                onClick={() => submitFeedback(message.id, 'down')}
                                className={`p-1.5 rounded-lg transition-all group ${
                                  messageFeedback[message.id] === 'down'
                                    ? 'bg-red-500/20'
                                    : 'hover:bg-amber-500/10'
                                }`}
                                title="Bad response"
                              >
                                <ThumbsDown className={`w-4 h-4 ${
                                  messageFeedback[message.id] === 'down'
                                    ? 'text-red-400'
                                    : 'text-amber-200/50 group-hover:text-amber-400'
                                }`} />
                              </button>

                              {/* Regenerate - only show on last assistant message */}
                              {messages[messages.length - 1]?.id === message.id && (
                                <button
                                  onClick={regenerateResponse}
                                  disabled={isGenerating}
                                  className="p-1.5 rounded-lg hover:bg-amber-500/10 transition-all group disabled:opacity-50"
                                  title="Regenerate response"
                                >
                                  <RefreshCw className="w-4 h-4 text-amber-200/50 group-hover:text-amber-400" />
                                </button>
                              )}
                            </>
                          )}

                          {message.role === 'user' && (
                            <>
                              {/* Edit Button */}
                              <button
                                onClick={() => startEditingMessage(message.id, message.content)}
                                disabled={isGenerating}
                                className="p-1.5 rounded-lg hover:bg-amber-500/10 transition-all group disabled:opacity-50"
                                title="Edit message"
                              >
                                <Pencil className="w-4 h-4 text-amber-200/50 group-hover:text-amber-400" />
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    {message.role === 'user' && (
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{
                          background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.3) 0%, rgba(255, 215, 0, 0.2) 100%)',
                          border: '1px solid rgba(212, 175, 55, 0.3)'
                        }}
                      >
                        <User className="w-5 h-5 text-amber-300" />
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>

              {/* Streaming Message */}
              {streamingMessage && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{
                      background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)'
                    }}
                  >
                    <Bot className="w-5 h-5 text-slate-900" />
                  </div>

                  <div className="max-w-2xl">
                    <div
                      className="p-4 rounded-2xl"
                      style={{
                        background: 'rgba(30, 41, 59, 0.5)',
                        border: '1px solid rgba(212, 175, 55, 0.1)'
                      }}
                    >
                      {/* Streaming Reasoning (Kimi K2) */}
                      {streamingReasoning && (
                        <div className="mb-3 border-l-2 border-purple-400 pl-3">
                          <div className="flex items-center gap-2 text-sm text-purple-400 mb-2">
                            <Brain className="w-4 h-4 animate-pulse" />
                            <span>Thinking...</span>
                          </div>
                          <div className="text-sm text-gray-400 whitespace-pre-wrap font-mono bg-purple-500/5 p-3 rounded">
                            {streamingReasoning}
                            <span className="inline-block w-2 h-4 ml-1 bg-purple-400 animate-pulse" />
                          </div>
                        </div>
                      )}

                      <div className="text-amber-50 leading-relaxed prose prose-invert prose-amber max-w-none">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeHighlight, rehypeRaw]}
                          components={{
                            code: ({ node, inline, className, children, ...props }: any) => {
                              const match = /language-(\w+)/.exec(className || '');
                              return !inline && match ? (
                                <div className="relative my-4">
                                  <div className="absolute top-0 right-0 px-3 py-1 text-xs text-amber-400 bg-slate-800/50 rounded-bl-lg rounded-tr-lg border-l border-b border-amber-500/20">
                                    {match[1]}
                                  </div>
                                  <code
                                    className={`${className} block p-4 rounded-lg overflow-x-auto`}
                                    style={{
                                      background: 'rgba(15, 23, 42, 0.8)',
                                      border: '1px solid rgba(212, 175, 55, 0.2)',
                                    }}
                                    {...props}
                                  >
                                    {children}
                                  </code>
                                </div>
                              ) : (
                                <code
                                  className="px-1.5 py-0.5 rounded text-sm"
                                  style={{
                                    background: 'rgba(212, 175, 55, 0.15)',
                                    border: '1px solid rgba(212, 175, 55, 0.3)',
                                    color: '#FCD34D',
                                  }}
                                  {...props}
                                >
                                  {children}
                                </code>
                              );
                            },
                            pre: ({ children }: any) => <div className="not-prose">{children}</div>,
                            p: ({ children }: any) => <p className="mb-3 last:mb-0">{children}</p>,
                            ul: ({ children }: any) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                            ol: ({ children }: any) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                            li: ({ children }: any) => <li className="text-amber-50/90">{children}</li>,
                            h1: ({ children }: any) => <h1 className="text-2xl font-bold text-amber-400 mb-3 mt-4">{children}</h1>,
                            h2: ({ children }: any) => <h2 className="text-xl font-bold text-amber-400 mb-2 mt-3">{children}</h2>,
                            h3: ({ children }: any) => <h3 className="text-lg font-bold text-amber-400 mb-2 mt-3">{children}</h3>,
                            blockquote: ({ children }: any) => (
                              <blockquote className="border-l-4 border-amber-500/50 pl-4 italic text-amber-200/80 my-3">
                                {children}
                              </blockquote>
                            ),
                            a: ({ children, href }: any) => (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-amber-400 hover:text-amber-300 underline"
                              >
                                {children}
                              </a>
                            ),
                          }}
                        >
                          {streamingMessage}
                        </ReactMarkdown>
                        <span className="inline-block w-2 h-5 ml-1 bg-amber-400 animate-pulse" />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

      </div>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowSettings(false)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            />

            {/* Modal */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
              onClick={() => setShowSettings(false)}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-2xl rounded-2xl p-8 shadow-2xl"
                style={{
                  background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%)',
                  border: '2px solid rgba(212, 175, 55, 0.3)',
                  boxShadow: '0 0 60px rgba(212, 175, 55, 0.2), 0 20px 50px rgba(0, 0, 0, 0.5)',
                }}
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center gap-3">
                    <div
                      className="p-3 rounded-xl"
                      style={{
                        background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.2) 100%)',
                        boxShadow: '0 0 20px rgba(212, 175, 55, 0.3)',
                      }}
                    >
                      <Settings className="w-6 h-6 text-amber-400" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-amber-50">AI Settings</h2>
                      <p className="text-sm text-amber-200/60">Customize your AI chat experience</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="p-2 rounded-lg hover:bg-amber-500/10 transition-all"
                  >
                    <svg className="w-6 h-6 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Settings Content */}
                <div className="space-y-6">
                  {/* Model Selector */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-amber-50 font-medium flex items-center gap-2">
                        <Bot className="w-4 h-4 text-amber-400" />
                        AI Model
                      </label>
                      <span className="text-amber-400 font-mono text-xs px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {selectedModel === 'Gemma4-Ollama' ? 'live context' : selectedModel.includes('BitNet') ? '2B 1-bit' : selectedModel.includes('Small') ? '24B params' : selectedModel.includes('Ministral-3B') ? '3B params' : selectedModel.includes('Qwen3') ? '8B params' : '7B params'}
                      </span>
                    </div>
                    <select
                      value={selectedModel}
                      onChange={(e) => switchModel(e.target.value)}
                      disabled={isSwitchingModel || isGenerating}
                      className="w-full p-3 rounded-xl bg-slate-800/50 border-2 border-amber-500/30 text-amber-50 font-medium focus:outline-none focus:border-amber-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        boxShadow: '0 0 20px rgba(212, 175, 55, 0.1)',
                      }}
                    >
                      <option value="Gemma4-Ollama">✨ Gemma 4 (Ollama) - Live Network Data</option>
                      <option value="BitNet-b1.58-2B-4T">⚡ BitNet b1.58 2B (0.4 GB) - 1-Bit Quantized</option>
                      <option value="Ministral-3B-Instruct">🔧 Ministral 3B (2.1 GB) - Agentic + Functions</option>
                      <option value="Mistral-7B-Instruct-v0.3">Mistral 7B Instruct (4.3 GB) - Fast</option>
                      <option value="Qwen3-VL-8B-Instruct">🖼️ Qwen3 VL 8B (5.1 GB) - Vision & Language</option>
                      <option value="Mistral-Small-3.2-24B-Instruct">Mistral Small 24B (14 GB) - Higher Quality</option>
                      <option value="Kimi-K2-Thinking">🧠 Kimi K2 Thinking (245 GB) - Advanced Reasoning</option>
                    </select>
                    {modelSwitchStatus && (
                      <div className="text-sm text-amber-300 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {modelSwitchStatus}
                      </div>
                    )}
                    <div className="flex justify-between text-xs text-amber-200/60">
                      <span>Faster responses</span>
                      <span>Better quality</span>
                    </div>
                  </div>

                  {/* Temperature Slider */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-amber-50 font-medium flex items-center gap-2">
                        <Zap className="w-4 h-4 text-amber-400" />
                        Temperature
                      </label>
                      <span className="text-amber-400 font-mono text-sm px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {temperature.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.01"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full h-2 rounded-lg appearance-none cursor-pointer slider-gradient"
                      style={{
                        background: `linear-gradient(to right,
                          rgba(59, 130, 246, 0.5) 0%,
                          rgba(212, 175, 55, 0.5) ${(temperature / 2) * 100}%,
                          rgba(239, 68, 68, 0.5) 100%)`,
                      }}
                    />
                    <div className="flex justify-between text-xs text-amber-200/60">
                      <span>More Focused</span>
                      <span>More Creative</span>
                    </div>
                  </div>

                  {/* Max Tokens Slider */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-amber-50 font-medium flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-amber-400" />
                        Max Tokens
                      </label>
                      <span className="text-amber-400 font-mono text-sm px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {maxTokens}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="64"
                      max="2048"
                      step="64"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right,
                          rgba(212, 175, 55, 0.5) 0%,
                          rgba(212, 175, 55, 0.2) ${(maxTokens / 2048) * 100}%,
                          rgba(30, 41, 59, 0.5) 100%)`,
                      }}
                    />
                    <div className="flex justify-between text-xs text-amber-200/60">
                      <span>64</span>
                      <span>2048</span>
                    </div>
                  </div>

                  {/* Top P Slider */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-amber-50 font-medium flex items-center gap-2">
                        <Shield className="w-4 h-4 text-amber-400" />
                        Top P (Nucleus Sampling)
                      </label>
                      <span className="text-amber-400 font-mono text-sm px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {topP.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={topP}
                      onChange={(e) => setTopP(parseFloat(e.target.value))}
                      className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right,
                          rgba(212, 175, 55, 0.5) 0%,
                          rgba(212, 175, 55, 0.2) ${topP * 100}%,
                          rgba(30, 41, 59, 0.5) 100%)`,
                      }}
                    />
                    <div className="flex justify-between text-xs text-amber-200/60">
                      <span>Focused</span>
                      <span>Diverse</span>
                    </div>
                  </div>

                  {/* Frequency Penalty Slider */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-amber-50 font-medium flex items-center gap-2">
                        <Clock className="w-4 h-4 text-amber-400" />
                        Frequency Penalty
                      </label>
                      <span className="text-amber-400 font-mono text-sm px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {frequencyPenalty.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.01"
                      value={frequencyPenalty}
                      onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                      className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right,
                          rgba(212, 175, 55, 0.5) 0%,
                          rgba(212, 175, 55, 0.2) ${(frequencyPenalty / 2) * 100}%,
                          rgba(30, 41, 59, 0.5) 100%)`,
                      }}
                    />
                    <div className="flex justify-between text-xs text-amber-200/60">
                      <span>Repetitive</span>
                      <span>Varied</span>
                    </div>
                  </div>

                  {/* Presence Penalty Slider */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-amber-50 font-medium flex items-center gap-2">
                        <Bot className="w-4 h-4 text-amber-400" />
                        Presence Penalty
                      </label>
                      <span className="text-amber-400 font-mono text-sm px-3 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20">
                        {presencePenalty.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.01"
                      value={presencePenalty}
                      onChange={(e) => setPresencePenalty(parseFloat(e.target.value))}
                      className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right,
                          rgba(212, 175, 55, 0.5) 0%,
                          rgba(212, 175, 55, 0.2) ${(presencePenalty / 2) * 100}%,
                          rgba(30, 41, 59, 0.5) 100%)`,
                      }}
                    />
                    <div className="flex justify-between text-xs text-amber-200/60">
                      <span>Allow Repeats</span>
                      <span>New Topics</span>
                    </div>
                  </div>
                </div>

                {/* Presets */}
                <div className="mt-8 pt-6 border-t border-amber-500/20">
                  <h3 className="text-sm font-medium text-amber-200/80 mb-4">Quick Presets</h3>
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      onClick={() => {
                        setTemperature(0.3);
                        setTopP(0.8);
                        setFrequencyPenalty(0.0);
                        setPresencePenalty(0.0);
                      }}
                      className="px-4 py-3 rounded-lg text-sm font-medium transition-all hover:scale-105"
                      style={{
                        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        color: '#93C5FD',
                      }}
                    >
                      Precise
                    </button>
                    <button
                      onClick={() => {
                        setTemperature(0.7);
                        setTopP(0.9);
                        setFrequencyPenalty(0.0);
                        setPresencePenalty(0.0);
                      }}
                      className="px-4 py-3 rounded-lg text-sm font-medium transition-all hover:scale-105"
                      style={{
                        background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.1) 100%)',
                        border: '1px solid rgba(212, 175, 55, 0.3)',
                        color: '#FCD34D',
                      }}
                    >
                      Balanced
                    </button>
                    <button
                      onClick={() => {
                        setTemperature(1.2);
                        setTopP(0.95);
                        setFrequencyPenalty(0.5);
                        setPresencePenalty(0.5);
                      }}
                      className="px-4 py-3 rounded-lg text-sm font-medium transition-all hover:scale-105"
                      style={{
                        background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%)',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        color: '#FCA5A5',
                      }}
                    >
                      Creative
                    </button>
                  </div>
                </div>

                {/* Save Button */}
                <div className="mt-8 flex gap-3">
                  <button
                    onClick={() => setShowSettings(false)}
                    className="flex-1 px-6 py-3 rounded-xl font-medium transition-all hover:scale-105"
                    style={{
                      background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                      color: '#0F172A',
                      boxShadow: '0 0 20px rgba(212, 175, 55, 0.4)',
                    }}
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Costs & Usage Modal */}
      <AnimatePresence>
        {showCostsUsage && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowCostsUsage(false)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            />

            {/* Modal */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
              onClick={() => setShowCostsUsage(false)}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-3xl rounded-2xl p-8 shadow-2xl max-h-[90vh] overflow-y-auto"
                style={{
                  background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%)',
                  border: '2px solid rgba(212, 175, 55, 0.3)',
                  boxShadow: '0 0 60px rgba(212, 175, 55, 0.2), 0 20px 50px rgba(0, 0, 0, 0.5)',
                }}
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center gap-3">
                    <div
                      className="p-3 rounded-xl"
                      style={{
                        background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.2) 100%)',
                        boxShadow: '0 0 20px rgba(212, 175, 55, 0.3)',
                      }}
                    >
                      <DollarSign className="w-6 h-6 text-amber-400" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-amber-50">Costs & Usage</h2>
                      <p className="text-sm text-amber-200/60">Track your AI inference spending</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowCostsUsage(false)}
                    className="p-2 rounded-lg hover:bg-amber-500/10 transition-all"
                  >
                    <svg className="w-6 h-6 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Balance Overview */}
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div
                    className="p-6 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(255, 215, 0, 0.1) 100%)',
                      border: '1px solid rgba(212, 175, 55, 0.3)',
                    }}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-5 h-5 text-amber-400" />
                      <h3 className="text-sm font-medium text-amber-200/70">QUG Balance</h3>
                    </div>
                    {isLoadingUsageData ? (
                      <p className="text-2xl font-bold text-amber-50">Loading...</p>
                    ) : walletData ? (
                      <>
                        <p className="text-3xl font-bold text-amber-50">
                          {walletData.balance_qnk?.toLocaleString() || '0'}
                        </p>
                        <p className="text-xs text-amber-200/50 mt-1">
                          ≈ ${walletData.balance_qnk_usd?.toFixed(2) || '0.00'} USD
                        </p>
                      </>
                    ) : (
                      <p className="text-2xl font-bold text-amber-50">No wallet data</p>
                    )}
                  </div>

                  <div
                    className="p-6 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                      border: '1px solid rgba(168, 85, 247, 0.3)',
                    }}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-5 h-5 text-purple-400" />
                      <h3 className="text-sm font-medium text-purple-200/70">Tokens Generated</h3>
                    </div>
                    {isLoadingUsageData ? (
                      <p className="text-2xl font-bold text-purple-50">Loading...</p>
                    ) : walletData ? (
                      <>
                        <p className="text-3xl font-bold text-purple-50">
                          {walletData.total_tokens_generated?.toLocaleString() || '0'}
                        </p>
                        <p className="text-xs text-purple-200/50 mt-1">Across all chats</p>
                      </>
                    ) : (
                      <p className="text-2xl font-bold text-purple-50">0</p>
                    )}
                  </div>
                </div>

                {/* Pricing Information */}
                <div
                  className="p-6 rounded-xl mb-6"
                  style={{
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid rgba(212, 175, 55, 0.2)',
                  }}
                >
                  <h3 className="text-lg font-bold text-amber-50 mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-400" />
                    Current Pricing
                  </h3>
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <p className="text-sm text-amber-200/60 mb-1">Cost per Token</p>
                      {pricingData ? (
                        <>
                          <p className="text-xl font-bold text-amber-400">
                            {pricingData.cost_per_token_qnk} QUG
                          </p>
                          <p className="text-xs text-amber-200/40 mt-1">
                            ≈ ${pricingData.cost_per_token_usd?.toFixed(6)} USD
                          </p>
                        </>
                      ) : (
                        <p className="text-xl font-bold text-amber-400">Loading...</p>
                      )}
                    </div>
                    <div>
                      <p className="text-sm text-amber-200/60 mb-1">Estimated Cost (512 tokens)</p>
                      {pricingData ? (
                        <>
                          <p className="text-xl font-bold text-amber-400">
                            {pricingData.estimated_cost_512_tokens_qnk?.toLocaleString()} QUG
                          </p>
                          <p className="text-xs text-amber-200/40 mt-1">
                            ≈ ${pricingData.estimated_cost_512_tokens_usd?.toFixed(2)} USD
                          </p>
                        </>
                      ) : (
                        <p className="text-xl font-bold text-amber-400">Loading...</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Usage Stats */}
                <div
                  className="p-6 rounded-xl mb-6"
                  style={{
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid rgba(212, 175, 55, 0.2)',
                  }}
                >
                  <h3 className="text-lg font-bold text-amber-50 mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-amber-400" />
                    Usage Statistics
                  </h3>
                  {isLoadingUsageData ? (
                    <p className="text-amber-200/70">Loading...</p>
                  ) : usageData ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-amber-200/70">Total Spent</span>
                        <div className="text-right">
                          <span className="text-amber-50 font-bold">
                            {usageData.total_spent_qnk?.toLocaleString() || '0'} QUG
                          </span>
                          <p className="text-xs text-amber-200/50">
                            ≈ ${usageData.total_spent_usd?.toFixed(2) || '0.00'} USD
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-amber-200/70">Total Requests</span>
                        <span className="text-amber-50 font-bold">
                          {usageData.total_requests?.toLocaleString() || '0'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-amber-200/70">Average Cost per Request</span>
                        <div className="text-right">
                          <span className="text-amber-50 font-bold">
                            {usageData.average_cost_per_request_qnk?.toLocaleString() || '0'} QUG
                          </span>
                          <p className="text-xs text-amber-200/50">
                            ≈ ${((usageData.average_cost_per_request_qnk || 0) * 0.000005).toFixed(3)} USD
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-amber-200/70">No usage data available</p>
                  )}
                </div>

                {/* Recent Transactions */}
                <div
                  className="p-6 rounded-xl"
                  style={{
                    background: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid rgba(212, 175, 55, 0.2)',
                  }}
                >
                  <h3 className="text-lg font-bold text-amber-50 mb-4 flex items-center gap-2">
                    <Clock className="w-5 h-5 text-amber-400" />
                    Recent Transactions
                  </h3>
                  <div className="text-center py-8">
                    <p className="text-amber-200/60">
                      Transaction history coming soon
                    </p>
                    <p className="text-xs text-amber-200/40 mt-2">
                      Full payment consensus integration in progress
                    </p>
                  </div>
                </div>

                {/* Treasury Info */}
                <div
                  className="mt-6 p-4 rounded-xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%)',
                    border: '1px solid rgba(59, 130, 246, 0.2)',
                  }}
                >
                  <div className="flex items-start gap-3">
                    <Shield className="w-5 h-5 text-blue-400 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-sm text-blue-200/90 font-medium mb-1">Revenue Model</p>
                      <p className="text-xs text-blue-200/60 leading-relaxed">
                        100% of AI inference costs currently flow to the master treasury wallet.
                        Future updates will enable revenue sharing with node operators who host AI models.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Close Button */}
                <div className="mt-8">
                  <button
                    onClick={() => setShowCostsUsage(false)}
                    className="w-full px-6 py-3 rounded-xl font-medium transition-all hover:scale-105"
                    style={{
                      background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                      color: '#0F172A',
                      boxShadow: '0 0 20px rgba(212, 175, 55, 0.4)',
                    }}
                  >
                    Close
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* AI Metrics Modal */}
      <AnimatePresence>
        {showMetrics && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowMetrics(false)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            />

            {/* Modal */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
              onClick={() => setShowMetrics(false)}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-4xl rounded-2xl p-8 shadow-2xl max-h-[90vh] overflow-y-auto"
                style={{
                  background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%)',
                  border: '2px solid rgba(168, 85, 247, 0.3)',
                  boxShadow: '0 0 60px rgba(168, 85, 247, 0.2), 0 20px 50px rgba(0, 0, 0, 0.5)',
                }}
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center gap-3">
                    <div
                      className="p-3 rounded-xl"
                      style={{
                        background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(147, 51, 234, 0.2) 100%)',
                        boxShadow: '0 0 20px rgba(168, 85, 247, 0.3)',
                      }}
                    >
                      <Activity className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-purple-50">AI Performance Metrics</h2>
                      <p className="text-sm text-purple-200/60">Real-time inference performance and statistics</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowMetrics(false)}
                    className="p-2 rounded-lg hover:bg-purple-500/10 transition-all"
                  >
                    <svg className="w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {isInitialMetricsLoad && isLoadingMetrics ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400"></div>
                  </div>
                ) : metricsData ? (
                  <>
                    {/* Single-Node Metrics */}
                    <div className="mb-8">
                      <h3 className="text-xl font-bold text-purple-50 mb-4 flex items-center gap-2">
                        <Cpu className="w-5 h-5 text-purple-400" />
                        Single-Node Performance
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                            border: '1px solid rgba(168, 85, 247, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Zap className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Tokens Generated</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.single_node?.tokens_generated?.toLocaleString() || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                            border: '1px solid rgba(168, 85, 247, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Activity className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Tokens/Second</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.single_node?.tokens_per_second?.toFixed(1) || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                            border: '1px solid rgba(168, 85, 247, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Database className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">KV Cache Hits</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.single_node?.kv_cache_hits?.toLocaleString() || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                            border: '1px solid rgba(168, 85, 247, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <TrendingUp className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Cache Hit Rate</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.single_node?.cache_hit_rate ?
                              `${(metricsData.single_node.cache_hit_rate * 100).toFixed(1)}%` :
                              '0%'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                            border: '1px solid rgba(168, 85, 247, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Cpu className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Speedup Factor</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.single_node?.speedup_factor?.toFixed(2) || '1.00'}x
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)',
                            border: '1px solid rgba(168, 85, 247, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Clock className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Avg Latency</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.single_node?.average_latency_ms?.toFixed(0) || '0'}ms
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Distributed AI Metrics */}
                    <div>
                      <h3 className="text-xl font-bold text-purple-50 mb-4 flex items-center gap-2">
                        <Network className="w-5 h-5 text-purple-400" />
                        Distributed AI Network
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%)',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Send className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Total Requests</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.distributed?.total_requests?.toLocaleString() || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%)',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Users className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Nodes Participated</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.distributed?.nodes_participated?.toLocaleString() || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%)',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <TrendingUp className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Avg Nodes/Request</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.distributed?.average_nodes_per_request?.toFixed(1) || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%)',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Layers className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Layers Processed</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.distributed?.layers_processed?.toLocaleString() || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%)',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Users className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Available Nodes</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.distributed?.available_nodes?.toLocaleString() || '0'}
                          </p>
                        </div>

                        <div
                          className="p-4 rounded-xl"
                          style={{
                            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%)',
                            border: '1px solid rgba(147, 51, 234, 0.3)',
                          }}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <Clock className="w-4 h-4 text-purple-400" />
                            <h4 className="text-xs font-medium text-purple-200/70">Avg Network Latency</h4>
                          </div>
                          <p className="text-2xl font-bold text-purple-50">
                            {metricsData.distributed?.average_network_latency_ms?.toFixed(0) || '0'}ms
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* v2.5.0: Active Workers Section (Tensor Parallelism - Golden Standard) */}
                    {workersData && workersData.workers && workersData.workers.length > 0 && (() => {
                      // Parse CPU cores from capability strings
                      const parseCores = (cap: string) => {
                        const match = cap?.match(/cores:\s*(\d+)/i);
                        return match ? parseInt(match[1], 10) : 4;
                      };
                      const parseRam = (cap: string) => {
                        const match = cap?.match(/ram_gb:\s*(\d+)/i);
                        return match ? parseInt(match[1], 10) : 8;
                      };
                      const totalCores = workersData.workers.reduce((sum: number, w: any) => sum + parseCores(w.capability), 0);
                      const totalRam = workersData.workers.reduce((sum: number, w: any) => sum + parseRam(w.capability), 0);

                      return (
                      <div
                        className="mt-6 p-6 rounded-xl"
                        style={{
                          background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(34, 197, 94, 0.1) 100%)',
                          border: '1px solid rgba(168, 85, 247, 0.3)',
                        }}
                      >
                        <h3 className="flex items-center gap-2 text-lg font-semibold text-purple-200 mb-4">
                          <Users className="w-5 h-5 text-purple-400" />
                          Tensor Parallel Cluster
                          <span className="ml-2 px-2 py-0.5 text-xs font-bold bg-purple-500/30 rounded-full text-purple-200">
                            {workersData.total_workers} {workersData.total_workers === 1 ? 'Node' : 'Nodes'} • {totalCores} CPU Cores
                          </span>
                        </h3>

                        {/* Tensor Parallelism Explanation Banner */}
                        <div className="mb-4 p-3 rounded-lg" style={{ background: 'rgba(168, 85, 247, 0.1)', border: '1px solid rgba(168, 85, 247, 0.2)' }}>
                          <p className="text-xs text-purple-200/90 font-medium mb-1">⚡ Tensor Parallelism (Golden Standard)</p>
                          <p className="text-xs text-purple-200/70">
                            {workersData.total_workers === 1 ? (
                              <>Single-node inference active. <span className="text-yellow-300">Connect more nodes</span> to enable tensor sharding for faster inference!</>
                            ) : (
                              <>Model weights are <span className="text-purple-300 font-bold">sharded across {workersData.total_workers} nodes</span> with {totalCores} combined CPU cores working in parallel = ~{(Math.min(workersData.total_workers, 8) * 0.75 + 0.25).toFixed(1)}x faster inference!</>
                            )}
                          </p>
                        </div>

                        {/* Cluster Overview Stats */}
                        <div className="mb-4 grid grid-cols-4 gap-3">
                          <div className="p-3 rounded-lg text-center" style={{ background: 'rgba(168, 85, 247, 0.1)' }}>
                            <p className="text-2xl font-bold text-purple-300">{workersData.total_workers}</p>
                            <p className="text-xs text-purple-200/60">Nodes</p>
                          </div>
                          <div className="p-3 rounded-lg text-center" style={{ background: 'rgba(251, 191, 36, 0.1)' }}>
                            <p className="text-2xl font-bold text-yellow-300">{totalCores}</p>
                            <p className="text-xs text-yellow-200/60">CPU Cores</p>
                          </div>
                          <div className="p-3 rounded-lg text-center" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
                            <p className="text-2xl font-bold text-green-300">~{(Math.min(workersData.total_workers, 8) * 0.75 + 0.25).toFixed(1)}x</p>
                            <p className="text-xs text-green-200/60">Speedup</p>
                          </div>
                          <div className="p-3 rounded-lg text-center" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
                            <p className="text-2xl font-bold text-blue-300">{totalRam}GB</p>
                            <p className="text-xs text-blue-200/60">Total RAM</p>
                          </div>
                        </div>

                        {/* Worker Details */}
                        <p className="text-xs text-purple-200/60 mb-2 font-medium">Active Tensor Parallel Workers:</p>
                        <div className="space-y-3">
                          {workersData.workers.map((worker: any, index: number) => {
                            const cores = parseCores(worker.capability);
                            const ram = parseRam(worker.capability);
                            const isGpu = worker.capability?.toLowerCase().includes('gpu');
                            return (
                            <div
                              key={worker.node_id || index}
                              className="p-4 rounded-lg"
                              style={{
                                background: 'rgba(168, 85, 247, 0.1)',
                                border: '1px solid rgba(168, 85, 247, 0.2)',
                              }}
                            >
                              <div className="flex items-center justify-between flex-wrap gap-2">
                                <div className="flex items-center gap-3">
                                  <div className={`w-3 h-3 ${isGpu ? 'bg-green-400' : 'bg-purple-400'} rounded-full animate-pulse`} />
                                  <div>
                                    <p className="text-sm font-medium text-purple-50 flex items-center gap-2">
                                      Worker #{index + 1}
                                      {isGpu && <span className="px-1.5 py-0.5 text-[10px] bg-green-500/30 rounded text-green-300">GPU</span>}
                                    </p>
                                    <p className="text-xs text-purple-200/60 font-mono">
                                      {worker.peer_id?.substring(0, 16)}...
                                    </p>
                                  </div>
                                </div>
                                <div className="flex items-center gap-3 flex-wrap">
                                  <div className="text-center px-3 py-1 rounded bg-purple-500/20">
                                    <p className="text-xs text-purple-200/60">Rank</p>
                                    <p className="text-sm font-bold text-purple-200">{index}/{workersData.total_workers}</p>
                                  </div>
                                  <div className="text-center px-3 py-1 rounded bg-yellow-500/20">
                                    <p className="text-xs text-yellow-200/60">Cores</p>
                                    <p className="text-sm font-bold text-yellow-200">{cores}</p>
                                  </div>
                                  <div className="text-center px-3 py-1 rounded bg-blue-500/20">
                                    <p className="text-xs text-blue-200/60">RAM</p>
                                    <p className="text-sm font-bold text-blue-200">{ram}GB</p>
                                  </div>
                                  <div className="text-center px-3 py-1 rounded bg-green-500/20">
                                    <p className="text-xs text-green-200/60">Heads</p>
                                    <p className="text-sm font-bold text-green-200">{Math.ceil(32 / workersData.total_workers)}</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )})}
                        </div>

                        {/* How Tensor Parallelism Works */}
                        <div className="mt-4 p-3 rounded-lg" style={{ background: 'rgba(168, 85, 247, 0.05)', border: '1px dashed rgba(168, 85, 247, 0.2)' }}>
                          <p className="text-xs text-purple-200/80 font-medium mb-1">🧠 How It Works:</p>
                          <p className="text-xs text-purple-200/60">
                            Each worker processes <span className="text-purple-300 font-bold">{Math.ceil(32 / workersData.total_workers)} attention heads</span> of the model's 32 total heads.
                            Results are combined via <span className="text-green-300 font-bold">Ring All-Reduce</span> (O(2(N-1)) bandwidth-optimal).
                            This is the same architecture used by OpenAI, Anthropic, Google & Meta for large-scale inference!
                          </p>
                        </div>
                      </div>
                    );})()}

                    {/* ✨ NEW: Proof-of-Inference Verification Monitor */}
                    <div className="mt-6" key="verification-monitor-container">
                      <VerificationMonitor />
                    </div>

                    {/* Info Note */}
                    <div
                      className="mt-6 p-4 rounded-xl"
                      style={{
                        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%)',
                        border: '1px solid rgba(59, 130, 246, 0.2)',
                      }}
                    >
                      <div className="flex items-start gap-3">
                        <Shield className="w-5 h-5 text-blue-400 mt-0.5" />
                        <div className="flex-1">
                          <p className="text-sm text-blue-200/90 font-medium mb-1">Performance Optimization & Verification</p>
                          <p className="text-xs text-blue-200/60 leading-relaxed">
                            Metrics are updated in real-time. KV cache sharing and distributed inference
                            enable significantly faster response times. All worker computations are verified
                            using cryptographic proofs to ensure trustless distributed AI.
                          </p>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-purple-200/70">No metrics data available</p>
                  </div>
                )}

                {/* Close Button */}
                <div className="mt-8">
                  <button
                    onClick={() => setShowMetrics(false)}
                    className="w-full px-6 py-3 rounded-xl font-medium transition-all hover:scale-105"
                    style={{
                      background: 'linear-gradient(135deg, #A855F7 0%, #9333EA 50%, #A855F7 100%)',
                      color: '#FFFFFF',
                      boxShadow: '0 0 20px rgba(168, 85, 247, 0.4)',
                    }}
                  >
                    Close
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* ✅ v0.9.36-beta - AI Transaction Preview Modal */}
      {showTransactionPreview && (
        <TransactionPreviewModal
          preview={transactionPreview}
          onClose={handleTransactionCancel}
          onConfirm={handleTransactionConfirm}
          onCancel={handleTransactionCancel}
        />
      )}
    </div>
  );
}
