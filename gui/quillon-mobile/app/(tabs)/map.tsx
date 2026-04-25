import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, TouchableOpacity, TextInput, Linking, Alert } from 'react-native';
import { Text, ActivityIndicator, Chip } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';

const API = 'https://quillon.xyz/api/v1';
const CATEGORIES = ['All', 'Food', 'Coffee', 'Retail', 'Tech', 'Health', 'Services', 'Transport', 'Property'];

const CAT_ICONS: Record<string, string> = {
  food: 'food', coffee: 'coffee', retail: 'shopping', tech: 'chip', health: 'heart-pulse',
  services: 'briefcase', transport: 'car', property: 'home', other: 'store',
};
const CAT_COLORS: Record<string, string[]> = {
  food: ['#F97316', '#EA580C'], coffee: ['#92400E', '#78350F'], retail: ['#8B5CF6', '#7C3AED'],
  tech: ['#06B6D4', '#0891B2'], health: ['#10B981', '#059669'], services: ['#3B82F6', '#2563EB'],
  transport: ['#F59E0B', '#D97706'], property: ['#EC4899', '#DB2777'], other: ['#6B7280', '#4B5563'],
};

interface Merchant {
  id: string; name: string; category: string; description?: string;
  city: string; country: string; website?: string; phone?: string;
  verified: boolean; accepts_online: boolean; rating?: number;
}

const SAMPLES: Merchant[] = [
  { id: '1', name: 'Quantum Café', category: 'coffee', description: 'Specialty coffee, crypto-friendly', city: 'Berlin', country: 'DE', verified: true, accepts_online: true, rating: 4.8 },
  { id: '2', name: 'Node Runner Electronics', category: 'tech', description: 'GPUs, ASICs, mining gear', city: 'Amsterdam', country: 'NL', verified: true, accepts_online: true, rating: 4.5 },
  { id: '3', name: 'Decentralized Diner', category: 'food', description: 'Farm-to-table, pay in QUG', city: 'Lisbon', country: 'PT', verified: false, accepts_online: false, rating: 4.2 },
];

export default function MapTab() {
  const { address } = useWalletStore();
  const [merchants, setMerchants] = useState<Merchant[]>(SAMPLES);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('All');
  const [showRegister, setShowRegister] = useState(false);
  const [form, setForm] = useState({ name: '', category: 'other', city: '', country: '', description: '', website: '' });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetch(`${API}/merchants`).then(r => r.json()).then(d => {
      if (Array.isArray(d?.data)) setMerchants(d.data);
      else if (Array.isArray(d)) setMerchants(d);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const filtered = merchants.filter(m => {
    const matchCat = category === 'All' || m.category.toLowerCase() === category.toLowerCase();
    const q = search.toLowerCase();
    return matchCat && (!q || m.name.toLowerCase().includes(q) || m.city.toLowerCase().includes(q));
  });

  const submitMerchant = async () => {
    if (!form.name || !form.city) { Alert.alert('Required', 'Please enter business name and city.'); return; }
    setSubmitting(true);
    try {
      await fetch(`${API}/merchants`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...form, wallet_address: address }) });
    } catch {}
    setMerchants(prev => [{ ...form, id: `local-${Date.now()}`, verified: false, accepts_online: false, added_at: Date.now() } as any, ...prev]);
    setShowRegister(false);
    Alert.alert('Listed!', 'Your business has been submitted for verification.');
    setSubmitting(false);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <LinearGradient colors={['#060912', '#080d1a']} style={styles.bg}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>QUG Merchant Map</Text>
            <Text style={styles.subtitle}>Find businesses accepting Quillon</Text>
          </View>
          <TouchableOpacity onPress={() => setShowRegister(!showRegister)} style={styles.addBtn}>
            <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.addGradient}>
              <MaterialCommunityIcons name="plus" size={18} color="#fff" />
              <Text style={styles.addText}>List</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>

        {/* Search */}
        <View style={styles.searchRow}>
          <MaterialCommunityIcons name="magnify" size={18} color="#6B7280" style={styles.searchIcon} />
          <TextInput value={search} onChangeText={setSearch} placeholder="Search merchants..." placeholderTextColor="#4B5563" style={styles.searchInput} />
        </View>

        {/* Category filter */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.catScroll} contentContainerStyle={styles.catContent}>
          {CATEGORIES.map(c => (
            <TouchableOpacity key={c} onPress={() => setCategory(c)} style={[styles.catChip, category === c && styles.catChipActive]}>
              <Text style={[styles.catText, category === c && styles.catTextActive]}>{c}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Register form */}
        {showRegister && (
          <View style={styles.registerCard}>
            <Text style={styles.registerTitle}>List Your Business</Text>
            {[
              { key: 'name', placeholder: 'Business name *' },
              { key: 'city', placeholder: 'City *' },
              { key: 'country', placeholder: 'Country code (e.g. DE)' },
              { key: 'description', placeholder: 'Short description' },
              { key: 'website', placeholder: 'Website URL' },
            ].map(f => (
              <TextInput key={f.key} value={(form as any)[f.key]} onChangeText={v => setForm(prev => ({ ...prev, [f.key]: v }))} placeholder={f.placeholder} placeholderTextColor="#4B5563" style={styles.formInput} />
            ))}
            <TouchableOpacity onPress={submitMerchant} disabled={submitting} style={styles.submitBtn}>
              <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.submitGradient}>
                {submitting ? <ActivityIndicator size={16} color="#fff" /> : <Text style={styles.submitText}>Submit Business</Text>}
              </LinearGradient>
            </TouchableOpacity>
          </View>
        )}

        {/* Merchant list */}
        <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
          {loading ? (
            <ActivityIndicator color={COLORS.cyan} style={{ marginTop: 32 }} />
          ) : filtered.length === 0 ? (
            <View style={styles.empty}>
              <MaterialCommunityIcons name="map-marker-off" size={40} color="#374151" />
              <Text style={styles.emptyText}>No merchants found</Text>
            </View>
          ) : filtered.map(m => {
            const colors = CAT_COLORS[m.category] || CAT_COLORS.other;
            const icon = CAT_ICONS[m.category] || 'store';
            return (
              <View key={m.id} style={styles.card}>
                <LinearGradient colors={[colors[0] + '18', colors[1] + '08']} style={styles.cardGradient}>
                  <View style={styles.cardRow}>
                    <LinearGradient colors={colors} style={styles.iconBg}>
                      <MaterialCommunityIcons name={icon as any} size={18} color="#fff" />
                    </LinearGradient>
                    <View style={styles.cardInfo}>
                      <View style={styles.nameRow}>
                        <Text style={styles.merchantName}>{m.name}</Text>
                        {m.verified && <MaterialCommunityIcons name="check-circle" size={14} color="#10B981" />}
                      </View>
                      <Text style={styles.merchantCity}>{m.city}, {m.country} · {m.category}</Text>
                      {m.description && <Text style={styles.merchantDesc} numberOfLines={1}>{m.description}</Text>}
                    </View>
                    <View style={styles.cardRight}>
                      {m.rating && <Text style={styles.rating}>⭐ {m.rating}</Text>}
                      {m.accepts_online && <Text style={styles.onlineBadge}>Online</Text>}
                      {m.website && (
                        <TouchableOpacity onPress={() => Linking.openURL(m.website!)}>
                          <MaterialCommunityIcons name="open-in-new" size={16} color="#6B7280" />
                        </TouchableOpacity>
                      )}
                    </View>
                  </View>
                </LinearGradient>
              </View>
            );
          })}
          <View style={{ height: 80 }} />
        </ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#060912' },
  bg: { flex: 1 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 12 },
  title: { color: '#fff', fontSize: 20, fontWeight: '800' },
  subtitle: { color: '#6B7280', fontSize: 12, marginTop: 2 },
  addBtn: { borderRadius: 12, overflow: 'hidden' },
  addGradient: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 12, paddingVertical: 8 },
  addText: { color: '#fff', fontSize: 13, fontWeight: '700' },
  searchRow: { flexDirection: 'row', alignItems: 'center', marginHorizontal: 16, marginBottom: 10, backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 12, paddingHorizontal: 12, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  searchIcon: { marginRight: 8 },
  searchInput: { flex: 1, color: '#fff', fontSize: 14, paddingVertical: 10, backgroundColor: 'transparent' },
  catScroll: { marginBottom: 10 },
  catContent: { paddingHorizontal: 16, gap: 8, flexDirection: 'row' },
  catChip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.05)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  catChipActive: { backgroundColor: 'rgba(16,185,129,0.15)', borderColor: '#10B981' },
  catText: { color: '#9CA3AF', fontSize: 12, fontWeight: '600' },
  catTextActive: { color: '#10B981' },
  registerCard: { marginHorizontal: 16, marginBottom: 12, padding: 16, backgroundColor: 'rgba(16,185,129,0.05)', borderRadius: 16, borderWidth: 1, borderColor: 'rgba(16,185,129,0.15)' },
  registerTitle: { color: '#fff', fontSize: 15, fontWeight: '700', marginBottom: 10 },
  formInput: { backgroundColor: 'rgba(255,255,255,0.05)', color: '#fff', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8, marginBottom: 8, fontSize: 13, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  submitBtn: { borderRadius: 12, overflow: 'hidden', marginTop: 4 },
  submitGradient: { padding: 12, alignItems: 'center' },
  submitText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  list: { flex: 1, paddingHorizontal: 16 },
  card: { marginBottom: 8, borderRadius: 14, overflow: 'hidden', borderWidth: 1, borderColor: 'rgba(255,255,255,0.06)' },
  cardGradient: { padding: 12 },
  cardRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  iconBg: { width: 38, height: 38, borderRadius: 10, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  cardInfo: { flex: 1 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  merchantName: { color: '#fff', fontSize: 14, fontWeight: '700', flexShrink: 1 },
  merchantCity: { color: '#9CA3AF', fontSize: 11, marginTop: 2 },
  merchantDesc: { color: '#6B7280', fontSize: 12, marginTop: 2 },
  cardRight: { alignItems: 'flex-end', gap: 3, flexShrink: 0 },
  rating: { color: '#F59E0B', fontSize: 11, fontWeight: '600' },
  onlineBadge: { color: '#10B981', fontSize: 10, backgroundColor: 'rgba(16,185,129,0.12)', paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6 },
  empty: { alignItems: 'center', marginTop: 48 },
  emptyText: { color: '#4B5563', marginTop: 8, fontSize: 14 },
});
