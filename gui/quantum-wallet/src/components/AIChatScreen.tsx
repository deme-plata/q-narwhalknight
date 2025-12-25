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
  Pencil
} from 'lucide-react';
import TransactionPreviewModal from './TransactionPreviewModal';
import VerificationMonitor from './VerificationMonitor';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  reasoning?: string; // Kimi K2 thinking process (v1.0.5)
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
  const [maxTokens, setMaxTokens] = useState(512); // Default 512 tokens
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
  const [selectedModel, setSelectedModel] = useState('Mistral-7B-Instruct-v0.3');
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
  useEffect(() => {
    console.log(`🔍 [STATE] currentChatId changed to: ${currentChatId}`);
    currentChatIdRef.current = currentChatId; // Keep ref in sync for async operations
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

        console.log(`✅ Setting ${data.data.length} messages for chat ${chatId}`);
        setMessages(data.data);

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
      // Use AI to generate a concise title from the first message
      const titlePrompt = `Generate a short 3-5 word title for a chat that starts with: "${firstMessage.substring(0, 100)}". Only respond with the title, nothing else.`;
      const response = await fetch(`/api/chat/${chatId}/stream?content=${encodeURIComponent(titlePrompt)}&max_tokens=20`);

      if (!response.ok) return;

      const reader = response.body?.getReader();
      if (!reader) return;

      let generatedTitle = '';
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.cumulative) {
                generatedTitle = data.cumulative.trim();
              }
            } catch {}
          }
        }
      }

      // Update the chat title
      if (generatedTitle && generatedTitle.length > 0) {
        await fetch(`/api/chat/${chatId}/rename`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: generatedTitle.replace(/['"]/g, '') })
        });
        loadChats(false); // Refresh chat list without auto-selecting
      }
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

  // ✅ v1.4.2 - Stop generation
  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsGenerating(false);
    setStreamingMessage('');
    setStreamingReasoning('');
    localStorage.removeItem('activeAIGeneration');
    backgroundGenerationRef.current = false;
  };

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return;

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

        for (const line of lines) {
          if (line.startsWith('event:')) {
            // Event type line (started, token, complete, error)
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
              } else if (parsed.finish_reason) {
                // Complete event
                console.log(`✅ Complete: ${parsed.tokens_generated} tokens in ${parsed.total_time_ms}ms`);
                console.log(`   Throughput: ${parsed.tokens_per_second} tok/s`);
                console.log(`   Worker: ${parsed.worker_node}, Mode: ${parsed.mode}`);

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
                src="/quantum-ai-logo.png"
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
                  ? ` - ${workersData.total_workers} Workers Online!`
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
              <div className="text-center">
                <Bot className="w-16 h-16 mx-auto mb-4 text-amber-400/50" />
                <h3 className="text-xl font-bold text-amber-200 mb-2">
                  Start a New Chat
                </h3>
                <p className="text-amber-200/60">
                  Create a new chat to begin conversing with the AI
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
                        {selectedModel.includes('Small') ? '24B params' : selectedModel.includes('Ministral-3B') ? '3B params' : selectedModel.includes('Qwen3') ? '8B params' : '7B params'}
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
                      <option value="Ministral-3B-Instruct">⚡ Ministral 3B (2.1 GB) - Ultra Fast</option>
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

                    {/* v1.0: Active Workers Section (Data Parallelism) */}
                    {workersData && workersData.workers && workersData.workers.length > 0 && (
                      <div
                        className="mt-6 p-6 rounded-xl"
                        style={{
                          background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.1) 100%)',
                          border: '1px solid rgba(34, 197, 94, 0.3)',
                        }}
                      >
                        <h3 className="flex items-center gap-2 text-lg font-semibold text-green-200 mb-4">
                          <Users className="w-5 h-5 text-green-400" />
                          Active Workers ({workersData.total_workers})
                        </h3>
                        <div className="space-y-3">
                          {workersData.workers.map((worker: any, index: number) => (
                            <div
                              key={worker.node_id || index}
                              className="p-4 rounded-lg"
                              style={{
                                background: 'rgba(34, 197, 94, 0.1)',
                                border: '1px solid rgba(34, 197, 94, 0.2)',
                              }}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                  <div>
                                    <p className="text-sm font-medium text-green-50">{worker.node_id}</p>
                                    <p className="text-xs text-green-200/60 font-mono">
                                      {worker.peer_id.substring(0, 20)}...
                                    </p>
                                  </div>
                                </div>
                                <div className="flex items-center gap-4">
                                  <div className="text-right">
                                    <p className="text-xs text-green-200/60">Active Requests</p>
                                    <p className="text-sm font-bold text-green-200">
                                      {worker.active_requests}
                                    </p>
                                  </div>
                                  <div className="text-right">
                                    <p className="text-xs text-green-200/60">Capability</p>
                                    <p className="text-sm font-bold text-green-200">{worker.capability}</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-4 p-3 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.05)' }}>
                          <p className="text-xs text-green-200/70">
                            💡 Data Parallelism: {workersData.total_workers} worker{workersData.total_workers > 1 ? 's' : ''} can process requests simultaneously for perfect linear scaling!
                          </p>
                        </div>
                      </div>
                    )}

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
