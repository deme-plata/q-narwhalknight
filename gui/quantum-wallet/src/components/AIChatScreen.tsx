import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare,
  Send,
  Sparkles,
  Zap,
  Shield,
  Clock,
  Bot,
  User,
  Settings,
  Trash2,
  Plus
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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const userHasScrolled = useRef(false);
  const lastScrollTop = useRef(0);

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

  // Load user's chats on mount
  useEffect(() => {
    loadChats();
  }, []);

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
      const response = await fetch(`/api/chat/${chatId}/messages`);
      const data = await response.json();

      if (data.success && data.data) {
        setMessages(data.data);
      }
    } catch (error) {
      console.error('Failed to load messages:', error);
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

  const sendMessage = async () => {
    if (!input.trim() || !currentChatId || isGenerating) return;

    const userMessage = input;
    setInput('');
    setIsGenerating(true);
    setStreamingMessage('');

    // Close any existing event source
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Optimistically add user message to UI immediately
    const tempUserMessage: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: userMessage,
      timestamp: Date.now() / 1000,
    };
    setMessages(prev => [...prev, tempUserMessage]);

    try {
      // Create EventSource for SSE streaming (backend now saves messages automatically)
      const encodedContent = encodeURIComponent(userMessage);
      const url = `/api/chat/${currentChatId}/stream?content=${encodedContent}&max_tokens=150`;
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

          // Backend has saved both messages - reload from storage
          await loadMessages(currentChatId);

          // Small delay to ensure messages are rendered before clearing streaming
          await new Promise(resolve => setTimeout(resolve, 100));

          // Now clear the streaming message
          setStreamingMessage('');
          setIsGenerating(false);
          eventSource.close();

          // If this is the first message (chat was just created), generate a title
          const currentChat = chats.find(c => c.chat_id === currentChatId);
          if (currentChat && currentChat.message_count === 0 && userMessage) {
            generateChatTitle(currentChatId, userMessage);
          }
        } catch (error) {
          console.error('Failed to parse complete:', error);
          setStreamingMessage('');
          setIsGenerating(false);
        }
      });

      eventSource.addEventListener('error', (event: any) => {
        console.error('❌ Stream error:', event);
        setStreamingMessage('');
        setIsGenerating(false);
        eventSource.close();
      });

      eventSource.onerror = () => {
        console.error('❌ EventSource connection error');
        setIsGenerating(false);
        eventSource.close();
      };

    } catch (error) {
      console.error('Failed to send message:', error);
      setIsGenerating(false);
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
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                boxShadow: '0 0 20px rgba(212, 175, 55, 0.4)'
              }}
            >
              <MessageSquare className="w-6 h-6 text-slate-900" />
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
              <span className="font-medium text-amber-200">Mistral-7B-Instruct</span>
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
          <button
            className="p-2 rounded-lg hover:bg-amber-500/10 transition-all"
            title="Settings"
          >
            <Settings className="w-5 h-5 text-amber-400" />
          </button>
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
                {messages.map((message) => (
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
                        <p className="text-amber-50 whitespace-pre-wrap leading-relaxed">
                          {message.content}
                        </p>

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
                      <p className="text-amber-50 whitespace-pre-wrap leading-relaxed">
                        {streamingMessage}
                        <span className="inline-block w-2 h-5 ml-1 bg-amber-400 animate-pulse" />
                      </p>
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
    </div>
  );
}
