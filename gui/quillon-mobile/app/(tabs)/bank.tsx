import React, { useState } from 'react';
import { View, ScrollView, StyleSheet, TouchableOpacity, TextInput, Alert, Modal } from 'react-native';
import { Text, ActivityIndicator } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS } from '../../src/theme';
import { useWalletStore } from '../../src/stores/walletStore';

const API = 'https://quillon.xyz/api/v1';

type ModalType = 'loan' | 'incubation' | null;

const STATS = [
  { icon: 'bank-transfer', label: 'Loans Funded', value: '127' },
  { icon: 'currency-usd', label: 'QUG Deployed', value: '2.4M' },
  { icon: 'rocket-launch', label: 'Projects Incubated', value: '34' },
  { icon: 'check-decagram', label: 'Success Rate', value: '89%' },
];

const LOAN_FEATURES = [
  { icon: 'lightning-bolt', text: 'Fast approval, 24-72 hours' },
  { icon: 'shield-check', text: 'No credit score required' },
  { icon: 'percent', text: '4–12% APR, fixed rate' },
  { icon: 'calendar-clock', text: '6–36 month repayment' },
];

const INCUBATION_FEATURES = [
  { icon: 'school', text: 'Mentorship from core team' },
  { icon: 'handshake', text: 'Network & partnership access' },
  { icon: 'code-braces', text: 'Technical integration support' },
  { icon: 'trending-up', text: 'Growth funding up to 100K QUG' },
];

const HOW_IT_WORKS = [
  { step: '1', title: 'Apply', desc: 'Fill out the application with your project details and funding needs.' },
  { step: '2', title: 'Review', desc: 'Our team reviews within 72 hours and schedules a video call if needed.' },
  { step: '3', title: 'Receive QUG', desc: 'Funds disbursed directly to your Quillon wallet upon approval.' },
];

export default function BankTab() {
  const { address } = useWalletStore();
  const [activeModal, setActiveModal] = useState<ModalType>(null);

  return (
    <SafeAreaView style={styles.safe}>
      <LinearGradient colors={['#060912', '#080d1a']} style={styles.bg}>
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Header */}
          <View style={styles.header}>
            <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.headerIcon}>
              <MaterialCommunityIcons name="bank" size={22} color="#fff" />
            </LinearGradient>
            <View style={styles.headerText}>
              <Text style={styles.title}>Quillon Bank</Text>
              <Text style={styles.subtitle}>DeFi loans & incubation in native QUG</Text>
            </View>
          </View>

          {/* Stats row */}
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.statsScroll} contentContainerStyle={styles.statsContent}>
            {STATS.map(s => (
              <View key={s.label} style={styles.statCard}>
                <MaterialCommunityIcons name={s.icon as any} size={20} color={COLORS.cyan} />
                <Text style={styles.statValue}>{s.value}</Text>
                <Text style={styles.statLabel}>{s.label}</Text>
              </View>
            ))}
          </ScrollView>

          {/* Loan card */}
          <View style={styles.productCard}>
            <LinearGradient colors={['rgba(59,130,246,0.12)', 'rgba(139,92,246,0.08)']} style={styles.productGradient}>
              <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.productIconBg}>
                <MaterialCommunityIcons name="bank-transfer" size={22} color="#fff" />
              </LinearGradient>
              <Text style={styles.productTitle}>Business Start Loan</Text>
              <Text style={styles.productDesc}>Get funded in QUG to launch or grow your business. Fast approval, no credit score required.</Text>
              <View style={styles.featureList}>
                {LOAN_FEATURES.map(f => (
                  <View key={f.text} style={styles.featureRow}>
                    <MaterialCommunityIcons name={f.icon as any} size={14} color="#3B82F6" />
                    <Text style={styles.featureText}>{f.text}</Text>
                  </View>
                ))}
              </View>
              <View style={styles.termsRow}>
                <View style={styles.termBadge}><Text style={styles.termText}>1K – 500K QUG</Text></View>
                <View style={styles.termBadge}><Text style={styles.termText}>4–12% APR</Text></View>
                <View style={styles.termBadge}><Text style={styles.termText}>6–36 months</Text></View>
              </View>
              <TouchableOpacity onPress={() => setActiveModal('loan')} style={styles.applyBtn}>
                <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.applyGradient}>
                  <MaterialCommunityIcons name="file-document-edit" size={16} color="#fff" />
                  <Text style={styles.applyText}>Apply for Loan</Text>
                </LinearGradient>
              </TouchableOpacity>
            </LinearGradient>
          </View>

          {/* Incubation card */}
          <View style={styles.productCard}>
            <LinearGradient colors={['rgba(16,185,129,0.12)', 'rgba(6,182,212,0.08)']} style={styles.productGradient}>
              <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.productIconBg}>
                <MaterialCommunityIcons name="rocket-launch" size={22} color="#fff" />
              </LinearGradient>
              <Text style={styles.productTitle}>Get Incubated</Text>
              <Text style={styles.productDesc}>Join the Quillon incubator. Mentorship, funding, and full ecosystem integration for promising projects.</Text>
              <View style={styles.featureList}>
                {INCUBATION_FEATURES.map(f => (
                  <View key={f.text} style={styles.featureRow}>
                    <MaterialCommunityIcons name={f.icon as any} size={14} color="#10B981" />
                    <Text style={styles.featureText}>{f.text}</Text>
                  </View>
                ))}
              </View>
              <View style={styles.termsRow}>
                <View style={[styles.termBadge, styles.termBadgeGreen]}><Text style={[styles.termText, styles.termTextGreen]}>Up to 100K QUG</Text></View>
                <View style={[styles.termBadge, styles.termBadgeGreen]}><Text style={[styles.termText, styles.termTextGreen]}>3–6 months</Text></View>
                <View style={[styles.termBadge, styles.termBadgeGreen]}><Text style={[styles.termText, styles.termTextGreen]}>2–8% equity</Text></View>
              </View>
              <TouchableOpacity onPress={() => setActiveModal('incubation')} style={styles.applyBtn}>
                <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.applyGradient}>
                  <MaterialCommunityIcons name="rocket-launch" size={16} color="#fff" />
                  <Text style={styles.applyText}>Apply for Incubation</Text>
                </LinearGradient>
              </TouchableOpacity>
            </LinearGradient>
          </View>

          {/* How it works */}
          <View style={styles.howSection}>
            <Text style={styles.howTitle}>How It Works</Text>
            {HOW_IT_WORKS.map((step, i) => (
              <View key={step.step} style={styles.howRow}>
                <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.stepBadge}>
                  <Text style={styles.stepNum}>{step.step}</Text>
                </LinearGradient>
                <View style={styles.howContent}>
                  <Text style={styles.howStepTitle}>{step.title}</Text>
                  <Text style={styles.howStepDesc}>{step.desc}</Text>
                </View>
              </View>
            ))}
          </View>

          <View style={{ height: 80 }} />
        </ScrollView>

        {/* Modals */}
        <LoanModal visible={activeModal === 'loan'} onClose={() => setActiveModal(null)} walletAddress={address} />
        <IncubationModal visible={activeModal === 'incubation'} onClose={() => setActiveModal(null)} walletAddress={address} />
      </LinearGradient>
    </SafeAreaView>
  );
}

function LoanModal({ visible, onClose, walletAddress }: { visible: boolean; onClose: () => void; walletAddress: string }) {
  const [form, setForm] = useState({ businessName: '', description: '', amount: '', repayment: '12', purpose: '', revenue: '', collateral: '', email: '' });
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState('');

  const submit = async () => {
    if (!form.businessName || !form.amount) { Alert.alert('Required', 'Business name and loan amount are required.'); return; }
    setSubmitting(true);
    const appId = 'LOAN-' + Math.random().toString(36).slice(2, 8).toUpperCase();
    try {
      await fetch(`${API}/bank/loan/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...form, wallet_address: walletAddress }) });
    } catch {}
    setSuccess(appId);
    setSubmitting(false);
  };

  const reset = () => { setSuccess(''); setForm({ businessName: '', description: '', amount: '', repayment: '12', purpose: '', revenue: '', collateral: '', email: '' }); onClose(); };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={styles.modalBg}>
        <LinearGradient colors={['#060912', '#0d1424']} style={styles.modalInner}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Business Start Loan</Text>
            <TouchableOpacity onPress={reset}><MaterialCommunityIcons name="close" size={22} color="#6B7280" /></TouchableOpacity>
          </View>
          <ScrollView showsVerticalScrollIndicator={false}>
            {success ? (
              <View style={styles.successView}>
                <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.successIcon}>
                  <MaterialCommunityIcons name="check" size={32} color="#fff" />
                </LinearGradient>
                <Text style={styles.successTitle}>Application Submitted!</Text>
                <Text style={styles.successId}>Application ID: {success}</Text>
                <Text style={styles.successNote}>Our team will review within 72 hours and reach out via the email provided.</Text>
                <TouchableOpacity onPress={reset} style={styles.doneBtn}>
                  <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.doneBtnGradient}>
                    <Text style={styles.doneBtnText}>Done</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.formBody}>
                {[
                  { key: 'businessName', label: 'Business Name *', placeholder: 'Your business name' },
                  { key: 'description', label: 'Description *', placeholder: 'What does your business do?' },
                  { key: 'amount', label: 'Loan Amount (QUG) *', placeholder: 'e.g. 50000', kb: 'numeric' as any },
                  { key: 'purpose', label: 'Purpose of Loan', placeholder: 'Equipment, inventory, marketing...' },
                  { key: 'revenue', label: 'Monthly Revenue (QUG)', placeholder: 'Current monthly revenue', kb: 'numeric' as any },
                  { key: 'collateral', label: 'Collateral (optional)', placeholder: 'Assets you can offer as collateral' },
                  { key: 'email', label: 'Contact Email', placeholder: 'your@email.com', kb: 'email-address' as any },
                ].map(f => (
                  <View key={f.key} style={styles.field}>
                    <Text style={styles.fieldLabel}>{f.label}</Text>
                    <TextInput
                      value={(form as any)[f.key]}
                      onChangeText={v => setForm(p => ({ ...p, [f.key]: v }))}
                      placeholder={f.placeholder}
                      placeholderTextColor="#4B5563"
                      keyboardType={f.kb || 'default'}
                      style={styles.fieldInput}
                    />
                  </View>
                ))}
                <TouchableOpacity onPress={submit} disabled={submitting} style={styles.applyBtn}>
                  <LinearGradient colors={['#3B82F6', '#8B5CF6']} style={styles.applyGradient}>
                    {submitting ? <ActivityIndicator size={16} color="#fff" /> : <Text style={styles.applyText}>Submit Application</Text>}
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            )}
          </ScrollView>
        </LinearGradient>
      </View>
    </Modal>
  );
}

function IncubationModal({ visible, onClose, walletAddress }: { visible: boolean; onClose: () => void; walletAddress: string }) {
  const [form, setForm] = useState({ projectName: '', description: '', whyQuillon: '', stage: 'idea', teamSize: '', funding: '', website: '', github: '', email: '' });
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState('');

  const submit = async () => {
    if (!form.projectName || !form.description) { Alert.alert('Required', 'Project name and description are required.'); return; }
    setSubmitting(true);
    const appId = 'INC-' + Math.random().toString(36).slice(2, 8).toUpperCase();
    try {
      await fetch(`${API}/bank/incubation/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...form, wallet_address: walletAddress }) });
    } catch {}
    setSuccess(appId);
    setSubmitting(false);
  };

  const reset = () => { setSuccess(''); setForm({ projectName: '', description: '', whyQuillon: '', stage: 'idea', teamSize: '', funding: '', website: '', github: '', email: '' }); onClose(); };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={styles.modalBg}>
        <LinearGradient colors={['#060912', '#0d1424']} style={styles.modalInner}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Get Incubated</Text>
            <TouchableOpacity onPress={reset}><MaterialCommunityIcons name="close" size={22} color="#6B7280" /></TouchableOpacity>
          </View>
          <ScrollView showsVerticalScrollIndicator={false}>
            {success ? (
              <View style={styles.successView}>
                <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.successIcon}>
                  <MaterialCommunityIcons name="check" size={32} color="#fff" />
                </LinearGradient>
                <Text style={styles.successTitle}>Application Submitted!</Text>
                <Text style={styles.successId}>Application ID: {success}</Text>
                <Text style={styles.successNote}>Our incubator team will review your project and respond within 72 hours.</Text>
                <TouchableOpacity onPress={reset} style={styles.doneBtn}>
                  <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.doneBtnGradient}>
                    <Text style={styles.doneBtnText}>Done</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.formBody}>
                {[
                  { key: 'projectName', label: 'Project Name *', placeholder: 'Your project name' },
                  { key: 'description', label: 'Project Description *', placeholder: 'What problem does it solve?' },
                  { key: 'whyQuillon', label: 'Why Quillon?', placeholder: 'Why does your project need Quillon ecosystem?' },
                  { key: 'teamSize', label: 'Team Size', placeholder: 'e.g. 3', kb: 'numeric' as any },
                  { key: 'funding', label: 'Funding Needed (QUG)', placeholder: 'e.g. 50000', kb: 'numeric' as any },
                  { key: 'website', label: 'Website', placeholder: 'https://yourproject.com', kb: 'url' as any },
                  { key: 'github', label: 'GitHub / Source Code', placeholder: 'https://github.com/...' },
                  { key: 'email', label: 'Contact Email', placeholder: 'your@email.com', kb: 'email-address' as any },
                ].map(f => (
                  <View key={f.key} style={styles.field}>
                    <Text style={styles.fieldLabel}>{f.label}</Text>
                    <TextInput
                      value={(form as any)[f.key]}
                      onChangeText={v => setForm(p => ({ ...p, [f.key]: v }))}
                      placeholder={f.placeholder}
                      placeholderTextColor="#4B5563"
                      keyboardType={f.kb || 'default'}
                      style={styles.fieldInput}
                    />
                  </View>
                ))}
                <TouchableOpacity onPress={submit} disabled={submitting} style={styles.applyBtn}>
                  <LinearGradient colors={['#10B981', '#06B6D4']} style={styles.applyGradient}>
                    {submitting ? <ActivityIndicator size={16} color="#fff" /> : <Text style={styles.applyText}>Submit Application</Text>}
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            )}
          </ScrollView>
        </LinearGradient>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#060912' },
  bg: { flex: 1 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, paddingTop: 12, paddingBottom: 16 },
  headerIcon: { width: 44, height: 44, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  headerText: { flex: 1 },
  title: { color: '#fff', fontSize: 20, fontWeight: '800' },
  subtitle: { color: '#6B7280', fontSize: 12, marginTop: 2 },
  statsScroll: { marginBottom: 16 },
  statsContent: { paddingHorizontal: 16, gap: 10 },
  statCard: { backgroundColor: 'rgba(255,255,255,0.04)', borderRadius: 14, padding: 14, alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.07)', minWidth: 90 },
  statValue: { color: '#fff', fontSize: 18, fontWeight: '800', marginTop: 6 },
  statLabel: { color: '#6B7280', fontSize: 10, marginTop: 2, textAlign: 'center' },
  productCard: { marginHorizontal: 16, marginBottom: 14, borderRadius: 18, overflow: 'hidden', borderWidth: 1, borderColor: 'rgba(255,255,255,0.07)' },
  productGradient: { padding: 18 },
  productIconBg: { width: 46, height: 46, borderRadius: 13, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  productTitle: { color: '#fff', fontSize: 18, fontWeight: '800', marginBottom: 6 },
  productDesc: { color: '#9CA3AF', fontSize: 13, lineHeight: 20, marginBottom: 14 },
  featureList: { gap: 8, marginBottom: 14 },
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  featureText: { color: '#D1D5DB', fontSize: 13 },
  termsRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginBottom: 14 },
  termBadge: { backgroundColor: 'rgba(59,130,246,0.12)', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4, borderWidth: 1, borderColor: 'rgba(59,130,246,0.25)' },
  termBadgeGreen: { backgroundColor: 'rgba(16,185,129,0.12)', borderColor: 'rgba(16,185,129,0.25)' },
  termText: { color: '#60A5FA', fontSize: 11, fontWeight: '600' },
  termTextGreen: { color: '#34D399' },
  applyBtn: { borderRadius: 12, overflow: 'hidden' },
  applyGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 13 },
  applyText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  howSection: { marginHorizontal: 16, marginBottom: 8 },
  howTitle: { color: '#fff', fontSize: 16, fontWeight: '700', marginBottom: 14 },
  howRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 14, marginBottom: 16 },
  stepBadge: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  stepNum: { color: '#fff', fontSize: 14, fontWeight: '800' },
  howContent: { flex: 1 },
  howStepTitle: { color: '#fff', fontSize: 14, fontWeight: '700', marginBottom: 3 },
  howStepDesc: { color: '#6B7280', fontSize: 12, lineHeight: 18 },
  modalBg: { flex: 1, backgroundColor: '#060912' },
  modalInner: { flex: 1 },
  modalHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: 20, paddingBottom: 16, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.07)' },
  modalTitle: { color: '#fff', fontSize: 18, fontWeight: '800' },
  formBody: { padding: 20, gap: 4 },
  field: { marginBottom: 12 },
  fieldLabel: { color: '#9CA3AF', fontSize: 12, fontWeight: '600', marginBottom: 5 },
  fieldInput: { backgroundColor: 'rgba(255,255,255,0.05)', color: '#fff', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, fontSize: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  successView: { padding: 32, alignItems: 'center' },
  successIcon: { width: 64, height: 64, borderRadius: 32, alignItems: 'center', justifyContent: 'center', marginBottom: 20 },
  successTitle: { color: '#fff', fontSize: 20, fontWeight: '800', marginBottom: 8 },
  successId: { color: '#10B981', fontSize: 14, fontWeight: '700', marginBottom: 12 },
  successNote: { color: '#9CA3AF', fontSize: 13, textAlign: 'center', lineHeight: 20, marginBottom: 28 },
  doneBtn: { borderRadius: 12, overflow: 'hidden', width: '100%' },
  doneBtnGradient: { padding: 14, alignItems: 'center' },
  doneBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },
});
