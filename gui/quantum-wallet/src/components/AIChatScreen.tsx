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
  Layers
} from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
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

  // Debug: Log whenever currentChatId changes
  useEffect(() => {
    console.log(`🔍 [STATE] currentChatId changed to: ${currentChatId}`);
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
      }
    } catch (error) {
      console.error('Failed to load AI metrics:', error);
    } finally {
      setIsLoadingMetrics(false);
    }
  };

  useEffect(() => {
    if (showMetrics) {
      loadMetrics();
      // Auto-refresh metrics every 3 seconds while modal is open
      const interval = setInterval(() => {
        loadMetrics();
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [showMetrics]);

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

          // Load the chat's messages immediately
          loadMessages(genData.chatId);

          // Mark as generating
          backgroundGenerationRef.current = true;
          setIsGenerating(true);

          // Poll for new messages every 2 seconds
          const pollInterval = setInterval(async () => {
            try {
              const response = await fetch(`/api/chat/${genData.chatId}/messages`);
              if (response.ok) {
                const backendMessages = await response.json();

                // Ensure we have valid array data before setting state
                if (backendMessages.success && Array.isArray(backendMessages.data)) {
                  setMessages(backendMessages.data);

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
    if (activeGeneration) {
      try {
        const genData = JSON.parse(activeGeneration);

        // Only reconnect if this is the chat that's generating
        if (genData.chatId === currentChatId) {
          const age = Date.now() - genData.startTime;

          // If generation is less than 5 minutes old, start polling for completion
          if (age < 5 * 60 * 1000) {
            console.log('🔄 Reconnecting to ongoing generation for current chat');
            setIsGenerating(true);
            backgroundGenerationRef.current = true;

            // Poll for new messages every 2 seconds
            const pollInterval = setInterval(async () => {
              try {
                const response = await fetch(`/api/chat/${currentChatId}/messages`);
                if (response.ok) {
                  const backendMessages = await response.json();
                  if (backendMessages.success && Array.isArray(backendMessages.data) && backendMessages.data.length > 0) {
                    setMessages(backendMessages.data);

                    // Check if generation completed (new assistant message after startTime)
                    const lastMsg = backendMessages.data[backendMessages.data.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant' && lastMsg.timestamp > genData.startTime / 1000) {
                      console.log('✅ Background generation completed on return!');
                      setIsGenerating(false);
                      backgroundGenerationRef.current = false;
                      localStorage.removeItem('activeAIGeneration');
                      clearInterval(pollInterval);
                    }
                  } else if (!backendMessages.success) {
                    console.warn('Failed to fetch messages, stopping reconnection');
                    setIsGenerating(false);
                    backgroundGenerationRef.current = false;
                    localStorage.removeItem('activeAIGeneration');
                    clearInterval(pollInterval);
                  }
                }
              } catch (err) {
                console.error('Polling error:', err);
                // On error, stop trying to reconnect
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
            // Too old, clear it
            localStorage.removeItem('activeAIGeneration');
          }
        }
      } catch (e) {
        console.error('Failed to reconnect to generation:', e);
      }
    }
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

                // Load latest messages to show any progress
                loadMessages(currentChatId);

                // Start polling to check if still generating
                setIsGenerating(true);
                backgroundGenerationRef.current = true;

                const pollInterval = setInterval(async () => {
                  try {
                    const response = await fetch(`/api/chat/${currentChatId}/messages`);
                    if (response.ok) {
                      const backendMessages = await response.json();
                      if (backendMessages.success && Array.isArray(backendMessages.data) && backendMessages.data.length > 0) {
                        setMessages(backendMessages.data);

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
        console.log(`✅ Setting ${data.data.length} messages`);
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
        // Ensure messages is always an array
        console.warn('⚠️ No valid messages data, setting empty array');
        setMessages([]);
      }
    } catch (error) {
      console.error('❌ Failed to load messages:', error);
      // Ensure messages is always an array even on error
      setMessages([]);
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

      const data = await response.json();
      if (data.success && data.data) {
        setCurrentChatId(data.data.chat_id);
        setMessages([]);
        loadChats(false); // Don't auto-select, we already set the current chat
      }
    } catch (error) {
      console.error('Failed to create chat:', error);
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

  const sendMessage = async () => {
    if (!input.trim() || isGenerating) return;

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
      // Create EventSource for SSE streaming (backend now saves messages automatically)
      const encodedContent = encodeURIComponent(userMessage);
      const url = `/api/chat/${chatId}/stream?content=${encodedContent}&max_tokens=${maxTokens}`;
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      let cumulativeText = '';

      eventSource.addEventListener('start', () => {
        console.log('🌊 Stream started');
      });

      eventSource.addEventListener('progress', (event) => {
        console.log('📊 Progress:', event.data);
      });

      eventSource.addEventListener('token', (event) => {
        try {
          const data = JSON.parse(event.data);
          cumulativeText = data.cumulative || '';
          setStreamingMessage(cumulativeText);
        } catch (error) {
          console.error('Failed to parse token:', error);
        }
      });

      eventSource.addEventListener('complete', async (event) => {
        try {
          const stats = JSON.parse(event.data);
          console.log('✅ Complete:', stats);

          // DON'T clear streaming message yet - keep it visible while loading from DB
          // This prevents the jarring "disappear then reappear" effect
          setIsGenerating(false);
          eventSource.close();

          // Clear background generation tracking
          localStorage.removeItem('activeAIGeneration');

          // Reload messages from backend to get the complete conversation
          // Backend has already saved both user and assistant messages
          // This ensures we display the authoritative database state
          await loadMessages(chatId!);

          // NOW clear streaming message after database messages are loaded
          // This creates a smooth transition from streaming → persisted message
          setStreamingMessage('');

          // If this is the first message (chat was just created), generate a title
          const currentChat = chats.find(c => c.chat_id === chatId);
          if (currentChat && currentChat.message_count === 0 && userMessage) {
            generateChatTitle(chatId!, userMessage);
          }
        } catch (error) {
          console.error('Failed to parse complete:', error);
          // Keep the streaming message visible on error
          setIsGenerating(false);
          localStorage.removeItem('activeAIGeneration');
          // Don't clear streaming message on error - it's valuable to user
        }
      });

      eventSource.addEventListener('error', (event: any) => {
        console.error('❌ Stream error:', event);
        // DON'T clear streaming message - it might just be a connection hiccup
        // The message is still valuable to the user
        setIsGenerating(false);
        localStorage.removeItem('activeAIGeneration');
        // Try to load messages in case generation finished on backend
        loadMessages(chatId!).catch(err =>
          console.error('Failed to load messages after error:', err)
        );
        eventSource.close();
      });

      eventSource.onerror = () => {
        console.error('❌ EventSource connection error');
        setIsGenerating(false);
        localStorage.removeItem('activeAIGeneration');
        // Try to recover by loading messages from backend
        loadMessages(chatId!).catch(err =>
          console.error('Failed to load messages after connection error:', err)
        );
        eventSource.close();
      };

    } catch (error) {
      console.error('Failed to send message:', error);
      setIsGenerating(false);
      localStorage.removeItem('activeAIGeneration');
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

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {chats.map((chat) => (
            <motion.button
              key={chat.chat_id}
              onClick={() => {
                // Clean up any ongoing streaming before switching chats
                if (eventSourceRef.current) {
                  eventSourceRef.current.close();
                  eventSourceRef.current = null;
                }
                setStreamingMessage('');
                setIsGenerating(false);
                backgroundGenerationRef.current = false;

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
                  : 'Mistral 7B'}
              </span>
              <span className="text-xs text-amber-400/60">
                ({selectedModel.includes('Small') || selectedModel.includes('24B') ? '14 GB' : '4.3 GB'})
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
              className="p-2 rounded-lg hover:bg-purple-500/10 transition-all"
              title="AI Performance Metrics"
            >
              <Activity className="w-5 h-5 text-purple-400" />
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
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <p className="text-amber-50 whitespace-pre-wrap leading-relaxed">
                            {message.content}
                          </p>
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

        {/* Input Area */}
        {currentChatId && (
          <div
            className="p-6 border-t"
            style={{
              background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
              borderColor: 'rgba(212, 175, 55, 0.2)'
            }}
          >
            {/* Max Tokens Slider */}
            <div className="mb-4 flex items-center gap-3">
              <label className="text-amber-300 text-sm font-medium whitespace-nowrap">
                Max Tokens: {maxTokens}
              </label>
              <input
                type="range"
                min="50"
                max="2048"
                step="50"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #D4AF37 0%, #D4AF37 ${((maxTokens - 50) / (2048 - 50)) * 100}%, rgba(30, 41, 59, 0.5) ${((maxTokens - 50) / (2048 - 50)) * 100}%, rgba(30, 41, 59, 0.5) 100%)`
                }}
              />
              <span className="text-amber-200/60 text-xs whitespace-nowrap">
                {maxTokens < 256 ? 'Short' : maxTokens < 512 ? 'Medium' : maxTokens < 1024 ? 'Long' : 'Very Long'}
              </span>
            </div>

            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                placeholder="Ask me anything..."
                disabled={isGenerating}
                className="flex-1 px-6 py-4 rounded-xl text-amber-50 placeholder-amber-200/40 focus:outline-none transition-all disabled:opacity-50"
                style={{
                  background: 'rgba(30, 41, 59, 0.5)',
                  border: '2px solid rgba(212, 175, 55, 0.2)',
                  boxShadow: '0 0 20px rgba(212, 175, 55, 0.1)'
                }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isGenerating}
                className="px-6 py-4 rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                  color: '#0F172A',
                  boxShadow: '0 0 20px rgba(212, 175, 55, 0.4)'
                }}
              >
                {isGenerating ? (
                  <div className="w-6 h-6 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="w-6 h-6" />
                )}
              </button>
            </div>
          </div>
        )}
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
                        {selectedModel.includes('Small') ? '24B params' : '7B params'}
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
                      <option value="Mistral-7B-Instruct-v0.3">Mistral 7B Instruct (4.3 GB) - Fast</option>
                      <option value="Mistral-Small-3.2-24B-Instruct">Mistral Small 24B (14 GB) - Higher Quality</option>
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

                {isLoadingMetrics ? (
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
                          <p className="text-sm text-blue-200/90 font-medium mb-1">Performance Optimization</p>
                          <p className="text-xs text-blue-200/60 leading-relaxed">
                            Metrics are updated in real-time. KV cache sharing and distributed inference
                            enable significantly faster response times across the network.
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
    </div>
  );
}
