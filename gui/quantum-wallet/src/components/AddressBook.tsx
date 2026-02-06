import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users,
  Plus,
  Edit2,
  Trash2,
  Search,
  Shield,
  Zap,
  Copy,
  Check,
  Star,
  Lock,
  Sparkles
} from 'lucide-react';
import { qnkAPI } from '../services/api';

// ZK-STARK/SNARK proof types for address verification
interface ZKProof {
  proof_type: 'stark' | 'snark';
  proof_data: string;
  verified: boolean;
  verification_timestamp: number;
}

// Saved address with cryptographic proofs
interface SavedAddress {
  id: string;
  address: string;
  label: string;
  favorite: boolean;
  tags: string[];
  notes: string;
  zk_proof: ZKProof | null;
  created_at: number;
  last_used: number;
  usage_count: number;
  // Gossipsub sync metadata
  sync_status: 'synced' | 'pending' | 'local';
  sync_timestamp: number | null;
}

interface AddressBookProps {
  onSelectAddress?: (address: string) => void;
  compactMode?: boolean;
}

export default function AddressBook({ onSelectAddress, compactMode = false }: AddressBookProps) {
  const [addresses, setAddresses] = useState<SavedAddress[]>([]);
  const [filteredAddresses, setFilteredAddresses] = useState<SavedAddress[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // New address form
  const [newAddress, setNewAddress] = useState({
    address: '',
    label: '',
    tags: '',
    notes: '',
    generateProof: true
  });

  // Load addresses from backend on mount
  useEffect(() => {
    loadAddresses();

    // Set up SSE for real-time gossipsub sync updates
    const eventSource = new EventSource('/api/v1/addressbook/sync/stream');

    eventSource.onmessage = (event) => {
      try {
        const syncUpdate = JSON.parse(event.data);
        if (syncUpdate.type === 'address_synced') {
          updateAddressSyncStatus(syncUpdate.address_id, 'synced', Date.now());
        }
      } catch (error) {
        console.error('❌ Address sync update parse error:', error);
      }
    };

    eventSource.onerror = () => {
      // SSE connection failed - addresses will stay as saved locally
      // This is fine - they still work, just without real-time sync status
      console.warn('⚠️ Address book sync stream unavailable');
    };

    return () => {
      eventSource.close();
    };
  }, []);

  // Filter addresses based on search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredAddresses(addresses);
      return;
    }

    const query = searchQuery.toLowerCase();
    const filtered = addresses.filter(addr =>
      addr.label.toLowerCase().includes(query) ||
      addr.address.toLowerCase().includes(query) ||
      addr.tags.some(tag => tag.toLowerCase().includes(query)) ||
      addr.notes.toLowerCase().includes(query)
    );
    setFilteredAddresses(filtered);
  }, [searchQuery, addresses]);

  const loadAddresses = async () => {
    try {
      const response = await qnkAPI.getAddressBook();
      if (response.success && response.data) {
        setAddresses(response.data.addresses || []);
      }
    } catch (error) {
      console.error('❌ Failed to load address book:', error);
    }
  };

  const updateAddressSyncStatus = (addressId: string, status: 'synced' | 'pending' | 'local', timestamp: number) => {
    setAddresses(prev => prev.map(addr =>
      addr.id === addressId
        ? { ...addr, sync_status: status, sync_timestamp: timestamp }
        : addr
    ));
  };

  const generateZKProof = async (address: string): Promise<ZKProof | null> => {
    try {
      // Call backend to generate ZK-STARK proof for address ownership verification
      const response = await qnkAPI.generateAddressProof(address, 'stark');

      if (response.success && response.data) {
        return {
          proof_type: 'stark',
          proof_data: response.data.proof,
          verified: response.data.verified,
          verification_timestamp: Date.now()
        };
      }
      return null;
    } catch (error) {
      console.error('❌ ZK proof generation failed:', error);
      return null;
    }
  };

  const saveAddress = async () => {
    if (!newAddress.address.trim() || !newAddress.label.trim()) {
      setSaveError('Address and label are required');
      return;
    }

    // Clear previous messages
    setSaveError(null);
    setSaveSuccess(false);

    try {
      console.log('💾 [ADDRESS BOOK] Attempting to save address:', {
        address: newAddress.address.trim(),
        label: newAddress.label.trim(),
        generateProof: newAddress.generateProof
      });

      // Generate ZK proof if requested
      let zkProof: ZKProof | null = null;
      if (newAddress.generateProof) {
        console.log('🔐 [ADDRESS BOOK] Generating ZK proof...');
        zkProof = await generateZKProof(newAddress.address);
        console.log('🔐 [ADDRESS BOOK] ZK proof generated:', !!zkProof);
      }

      const addressData: SavedAddress = {
        id: crypto.randomUUID(),
        address: newAddress.address.trim(),
        label: newAddress.label.trim(),
        favorite: false,
        tags: newAddress.tags.split(',').map(t => t.trim()).filter(Boolean),
        notes: newAddress.notes.trim(),
        zk_proof: zkProof,
        created_at: Date.now(),
        last_used: Date.now(),
        usage_count: 0,
        sync_status: 'pending', // Will be updated to 'synced' when gossipsub confirms
        sync_timestamp: null
      };

      console.log('📡 [ADDRESS BOOK] Sending save request to backend...');
      // Save to backend (which will sync via gossipsub)
      const response = await qnkAPI.saveAddress(addressData);

      console.log('📡 [ADDRESS BOOK] Backend response:', {
        success: response.success,
        error: response.error,
        data: response.data
      });

      if (response.success) {
        console.log('✅ [ADDRESS BOOK] Address saved successfully!');
        // Use the entry from response which has sync_status: 'synced' from backend
        const savedEntry = response.data?.entry || { ...addressData, sync_status: 'synced', sync_timestamp: Date.now() };
        setAddresses(prev => [savedEntry, ...prev]);
        resetForm();
        setIsAddingNew(false);
        setSaveSuccess(true);

        // Clear success message after 3 seconds
        setTimeout(() => setSaveSuccess(false), 3000);
      } else {
        // Handle API error response
        const errorMsg = response.error || 'Failed to save address. Please try again.';
        console.error('❌ [ADDRESS BOOK] Save failed:', errorMsg);
        setSaveError(errorMsg);
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Network error. Please check your connection.';
      console.error('❌ [ADDRESS BOOK] Exception during save:', error);
      setSaveError(errorMsg);
    }
  };

  const deleteAddress = async (id: string) => {
    try {
      await qnkAPI.deleteAddress(id);
      setAddresses(prev => prev.filter(addr => addr.id !== id));
    } catch (error) {
      console.error('❌ Failed to delete address:', error);
    }
  };

  const toggleFavorite = async (id: string) => {
    const address = addresses.find(a => a.id === id);
    if (!address) return;

    try {
      await qnkAPI.updateAddress(id, { ...address, favorite: !address.favorite });
      setAddresses(prev => prev.map(addr =>
        addr.id === id ? { ...addr, favorite: !addr.favorite } : addr
      ));
    } catch (error) {
      console.error('❌ Failed to toggle favorite:', error);
    }
  };

  const selectAddress = (address: SavedAddress) => {
    if (onSelectAddress) {
      onSelectAddress(address.address);

      // Update usage statistics
      qnkAPI.updateAddress(address.id, {
        ...address,
        last_used: Date.now(),
        usage_count: address.usage_count + 1
      });

      setAddresses(prev => prev.map(addr =>
        addr.id === address.id
          ? { ...addr, last_used: Date.now(), usage_count: addr.usage_count + 1 }
          : addr
      ));
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const resetForm = () => {
    setNewAddress({
      address: '',
      label: '',
      tags: '',
      notes: '',
      generateProof: true
    });
    setSaveError(null);
    setSaveSuccess(false);
  };

  const getSyncStatusIcon = (status: 'synced' | 'pending' | 'local') => {
    switch (status) {
      case 'synced':
        return <Check className="w-3 h-3 text-green-400" />;
      case 'pending':
        return <Zap className="w-3 h-3 text-yellow-400 animate-pulse" />;
      case 'local':
        return <Lock className="w-3 h-3 text-gray-400" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`quantum-card ${compactMode ? 'p-4' : 'p-6'}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Users className="w-6 h-6 text-cyan-400" />
            <motion.div
              className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-500 rounded-full"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
          <div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 text-transparent bg-clip-text">
              Address Book
            </h2>
            <p className="text-xs text-gray-400 mt-1">
              ZK-STARK verified • Gossipsub synced
            </p>
          </div>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsAddingNew(!isAddingNew)}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 hover:from-cyan-500/30 hover:to-purple-500/30 border border-cyan-500/30 rounded-lg transition-all"
        >
          <Plus className="w-4 h-4" />
          <span className="font-medium">New Address</span>
        </motion.button>
      </div>

      {/* Search Bar */}
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search addresses, labels, tags..."
          className="w-full pl-10 pr-4 py-2 bg-black/30 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 transition-all"
        />
      </div>

      {/* Add New Address Form */}
      <AnimatePresence>
        {isAddingNew && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6 p-4 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 border border-cyan-500/30 rounded-lg"
          >
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Wallet Address *
                </label>
                <input
                  type="text"
                  value={newAddress.address}
                  onChange={(e) => setNewAddress({...newAddress, address: e.target.value})}
                  placeholder="0x..."
                  className="w-full px-3 py-2 bg-black/30 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Label *
                </label>
                <input
                  type="text"
                  value={newAddress.label}
                  onChange={(e) => setNewAddress({...newAddress, label: e.target.value})}
                  placeholder="My Friend, Exchange, etc."
                  className="w-full px-3 py-2 bg-black/30 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Tags (comma-separated)
                </label>
                <input
                  type="text"
                  value={newAddress.tags}
                  onChange={(e) => setNewAddress({...newAddress, tags: e.target.value})}
                  placeholder="personal, trading, defi"
                  className="w-full px-3 py-2 bg-black/30 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Notes
                </label>
                <textarea
                  value={newAddress.notes}
                  onChange={(e) => setNewAddress({...newAddress, notes: e.target.value})}
                  placeholder="Additional information..."
                  rows={2}
                  className="w-full px-3 py-2 bg-black/30 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 resize-none"
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="generateProof"
                  checked={newAddress.generateProof}
                  onChange={(e) => setNewAddress({...newAddress, generateProof: e.target.checked})}
                  className="w-4 h-4 accent-cyan-500"
                />
                <label htmlFor="generateProof" className="text-sm text-gray-300 flex items-center gap-2">
                  <Shield className="w-4 h-4 text-cyan-400" />
                  Generate ZK-STARK proof (recommended)
                </label>
              </div>

              {/* Error/Success Messages */}
              <AnimatePresence>
                {saveError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm"
                  >
                    ❌ {saveError}
                  </motion.div>
                )}
                {saveSuccess && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="px-4 py-3 bg-green-500/10 border border-green-500/30 rounded-lg text-green-400 text-sm"
                  >
                    ✅ Address saved successfully!
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex gap-2 pt-2">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={saveAddress}
                  className="flex-1 px-4 py-2 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  Save Address
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => {
                    setIsAddingNew(false);
                    resetForm();
                  }}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-all"
                >
                  Cancel
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Address List */}
      <div className="space-y-2 max-h-[500px] overflow-y-auto custom-scrollbar">
        <AnimatePresence>
          {filteredAddresses.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center py-12 text-gray-400"
            >
              <Users className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-lg font-medium">No addresses saved yet</p>
              <p className="text-sm mt-1">Add your first address to get started</p>
            </motion.div>
          )}

          {filteredAddresses.map((addr, index) => (
            <motion.div
              key={addr.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.05 }}
              className="group p-4 bg-gradient-to-br from-gray-800/50 to-gray-900/50 hover:from-cyan-500/10 hover:to-purple-500/10 border border-gray-700/50 hover:border-cyan-500/30 rounded-lg transition-all cursor-pointer"
              onClick={() => selectAddress(addr)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  {/* Label and Favorite */}
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-bold text-white truncate">{addr.label}</h3>
                    {addr.favorite && (
                      <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                    )}
                    {addr.zk_proof && (
                      <div className="flex items-center gap-1 px-2 py-0.5 bg-cyan-500/20 border border-cyan-500/30 rounded-full">
                        <Shield className="w-3 h-3 text-cyan-400" />
                        <span className="text-xs text-cyan-400 font-medium">
                          {addr.zk_proof.proof_type.toUpperCase()}
                        </span>
                      </div>
                    )}
                    <div className="flex items-center gap-1 px-2 py-0.5 bg-black/30 rounded-full">
                      {getSyncStatusIcon(addr.sync_status)}
                      <span className="text-xs text-gray-400">{addr.sync_status}</span>
                    </div>
                  </div>

                  {/* Address */}
                  <div className="flex items-center gap-2 mb-2">
                    <code className="text-sm text-gray-300 font-mono truncate flex-1">
                      {addr.address}
                    </code>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        copyToClipboard(addr.address, addr.id);
                      }}
                      className="p-1 hover:bg-cyan-500/20 rounded"
                    >
                      {copiedId === addr.id ? (
                        <Check className="w-3 h-3 text-green-400" />
                      ) : (
                        <Copy className="w-3 h-3 text-gray-400" />
                      )}
                    </motion.button>
                  </div>

                  {/* Tags */}
                  {addr.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {addr.tags.map((tag, i) => (
                        <span
                          key={i}
                          className="px-2 py-0.5 bg-purple-500/20 border border-purple-500/30 rounded-full text-xs text-purple-300"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Stats */}
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span>Used {addr.usage_count} times</span>
                    <span>•</span>
                    <span>Added {new Date(addr.created_at).toLocaleDateString()}</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 ml-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleFavorite(addr.id);
                    }}
                    className="p-2 hover:bg-yellow-500/20 rounded-lg transition-colors"
                  >
                    <Star className={`w-4 h-4 ${addr.favorite ? 'text-yellow-400 fill-yellow-400' : 'text-gray-400'}`} />
                  </motion.button>
                  {/* Edit functionality - to be implemented */}
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      // TODO: Implement edit functionality
                      console.log('Edit address:', addr.id);
                    }}
                    className="p-2 hover:bg-cyan-500/20 rounded-lg transition-colors"
                  >
                    <Edit2 className="w-4 h-4 text-cyan-400" />
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteAddress(addr.id);
                    }}
                    className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4 text-red-400" />
                  </motion.button>
                </div>
              </div>

              {/* Notes (if exists) */}
              {addr.notes && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-3 pt-3 border-t border-gray-700/50"
                >
                  <p className="text-sm text-gray-400 italic">{addr.notes}</p>
                </motion.div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Footer Stats */}
      <div className="mt-6 pt-4 border-t border-gray-700/50 flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <Shield className="w-3 h-3 text-cyan-400" />
            {addresses.filter(a => a.zk_proof).length} verified
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-purple-400" />
            {addresses.filter(a => a.sync_status === 'synced').length} synced
          </span>
          <span className="flex items-center gap-1">
            <Star className="w-3 h-3 text-yellow-400" />
            {addresses.filter(a => a.favorite).length} favorites
          </span>
        </div>
        <span>{addresses.length} total addresses</span>
      </div>
    </motion.div>
  );
}
