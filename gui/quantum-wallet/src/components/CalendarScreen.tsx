import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { qnkAPI as api } from '../services/api';

// ============================================================================
// Types
// ============================================================================

interface CalendarEvent {
  id: string;
  title: string;
  description?: string;
  event_type: string;
  start_time: number;
  end_time?: number;
  all_day?: boolean;
  recurring?: { frequency: string; interval: number; until?: number; count?: number };
  color?: string;
  reminder_minutes?: number[];
  scheduled_tx?: {
    to_wallet: string;
    token: string;
    amount: string;
    executed: boolean;
    tx_hash?: string;
    error?: string;
  };
  shared?: boolean;
  created_at: number;
  updated_at?: number;
  cancelled?: boolean;
  source_peer?: string;
  // network events
  icon?: string;
}

type ViewMode = 'month' | 'week' | 'day' | 'agenda';

// ============================================================================
// Constants
// ============================================================================

const EVENT_COLORS: Record<string, string> = {
  personal: '#06b6d4',        // cyan
  scheduled_tx: '#f59e0b',    // amber
  vesting_unlock: '#22c55e',  // green
  governance_vote: '#8b5cf6', // purple
  network_milestone: '#f43f5e', // rose
  community_event: '#3b82f6', // blue
  price_alert: '#ec4899',     // pink
};

const EVENT_TYPE_LABELS: Record<string, string> = {
  personal: 'Personal',
  scheduled_tx: 'Scheduled TX',
  vesting_unlock: 'Vesting Unlock',
  governance_vote: 'Governance',
  network_milestone: 'Network',
  community_event: 'Community',
  price_alert: 'Price Alert',
};

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'];

// ============================================================================
// Helpers
// ============================================================================

const getDaysInMonth = (year: number, month: number) => new Date(year, month + 1, 0).getDate();
const getFirstDayOfMonth = (year: number, month: number) => new Date(year, month, 1).getDay();

const formatTime = (ts: number) => {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const formatDate = (ts: number) => {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
};

const formatDateKey = (year: number, month: number, day: number) =>
  `${year}${String(month + 1).padStart(2, '0')}${String(day).padStart(2, '0')}`;

const isToday = (year: number, month: number, day: number) => {
  const now = new Date();
  return now.getFullYear() === year && now.getMonth() === month && now.getDate() === day;
};

const getEventColor = (event: CalendarEvent) =>
  event.color || EVENT_COLORS[event.event_type] || EVENT_COLORS.personal;

const countdownText = (targetTs: number) => {
  const now = Math.floor(Date.now() / 1000);
  const diff = targetTs - now;
  if (diff <= 0) return 'Now';
  const days = Math.floor(diff / 86400);
  const hours = Math.floor((diff % 86400) / 3600);
  const mins = Math.floor((diff % 3600) / 60);
  if (days > 365) return `${Math.floor(days / 365)}y ${Math.floor((days % 365) / 30)}m`;
  if (days > 30) return `${Math.floor(days / 30)}m ${days % 30}d`;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
};

// ============================================================================
// Sub-Components
// ============================================================================

const EventDot: React.FC<{ event: CalendarEvent }> = ({ event }) => (
  <div
    style={{
      width: 6, height: 6, borderRadius: '50%',
      background: getEventColor(event),
      boxShadow: `0 0 4px ${getEventColor(event)}80`,
      flexShrink: 0,
    }}
  />
);

/** Glassmorphic day cell for month view */
const DayCell: React.FC<{
  day: number; month: number; year: number; events: CalendarEvent[];
  isCurrentMonth: boolean; onDayClick: (y: number, m: number, d: number) => void;
  onEventClick: (e: CalendarEvent) => void;
}> = ({ day, month, year, events, isCurrentMonth, onDayClick, onEventClick }) => {
  const today = isToday(year, month, day);
  const dayEvents = events.filter(e => {
    const d = new Date(e.start_time * 1000);
    return d.getFullYear() === year && d.getMonth() === month && d.getDate() === day;
  });

  return (
    <motion.div
      whileHover={{ scale: 1.03, zIndex: 10 }}
      onClick={() => onDayClick(year, month, day)}
      style={{
        background: today
          ? 'rgba(6, 182, 212, 0.08)'
          : isCurrentMonth ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.005)',
        border: today ? '1px solid rgba(6, 182, 212, 0.5)' : '1px solid rgba(255,255,255,0.06)',
        borderRadius: 8,
        padding: '4px 6px',
        minHeight: 80,
        cursor: 'pointer',
        position: 'relative',
        overflow: 'hidden',
        animation: today ? 'calendarTodayPulse 3s ease-in-out infinite' : undefined,
      }}
    >
      <div style={{
        fontSize: 12,
        fontWeight: today ? 700 : 400,
        color: today ? '#06b6d4' : isCurrentMonth ? 'rgba(255,255,255,0.8)' : 'rgba(255,255,255,0.25)',
        marginBottom: 4,
      }}>
        {day}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {dayEvents.slice(0, 3).map(ev => (
          <div
            key={ev.id}
            onClick={(e) => { e.stopPropagation(); onEventClick(ev); }}
            style={{
              fontSize: 10,
              padding: '1px 4px',
              borderRadius: 3,
              background: `${getEventColor(ev)}20`,
              borderLeft: `2px solid ${getEventColor(ev)}`,
              color: 'rgba(255,255,255,0.8)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              cursor: 'pointer',
            }}
          >
            {ev.title}
          </div>
        ))}
        {dayEvents.length > 3 && (
          <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>
            +{dayEvents.length - 3} more
          </div>
        )}
      </div>
      {dayEvents.length > 0 && (
        <div style={{ display: 'flex', gap: 2, position: 'absolute', bottom: 3, right: 4 }}>
          {dayEvents.slice(0, 5).map(ev => <EventDot key={ev.id} event={ev} />)}
        </div>
      )}
    </motion.div>
  );
};

/** Week view with hourly time slots */
const WeekView: React.FC<{
  currentDate: Date; events: CalendarEvent[]; onEventClick: (e: CalendarEvent) => void;
}> = ({ currentDate, events, onEventClick }) => {
  const startOfWeek = new Date(currentDate);
  startOfWeek.setDate(currentDate.getDate() - currentDate.getDay());

  const weekDays = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(startOfWeek);
    d.setDate(startOfWeek.getDate() + i);
    return d;
  });

  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 280px)' }}>
      {/* Header */}
      <div style={{
        display: 'grid', gridTemplateColumns: '60px repeat(7, 1fr)',
        borderBottom: '1px solid rgba(255,255,255,0.06)', position: 'sticky', top: 0,
        background: 'rgba(0,0,0,0.8)', zIndex: 5, backdropFilter: 'blur(10px)',
      }}>
        <div style={{ padding: 8 }} />
        {weekDays.map((d, i) => {
          const td = isToday(d.getFullYear(), d.getMonth(), d.getDate());
          return (
            <div key={i} style={{
              padding: '8px 4px', textAlign: 'center',
              borderLeft: '1px solid rgba(255,255,255,0.04)',
            }}>
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)' }}>{DAYS[i]}</div>
              <div style={{
                fontSize: 18, fontWeight: td ? 700 : 400,
                color: td ? '#06b6d4' : 'rgba(255,255,255,0.8)',
                ...(td ? {
                  background: 'rgba(6,182,212,0.15)', borderRadius: '50%',
                  width: 32, height: 32, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                } : {}),
              }}>
                {d.getDate()}
              </div>
            </div>
          );
        })}
      </div>

      {/* Time grid */}
      {hours.map(hour => (
        <div key={hour} style={{
          display: 'grid', gridTemplateColumns: '60px repeat(7, 1fr)',
          minHeight: 48, borderBottom: '1px solid rgba(255,255,255,0.03)',
        }}>
          <div style={{
            fontSize: 10, color: 'rgba(255,255,255,0.3)', padding: '2px 8px',
            textAlign: 'right', borderRight: '1px solid rgba(255,255,255,0.06)',
          }}>
            {hour === 0 ? '12 AM' : hour < 12 ? `${hour} AM` : hour === 12 ? '12 PM' : `${hour - 12} PM`}
          </div>
          {weekDays.map((wd, di) => {
            const cellEvents = events.filter(e => {
              const ed = new Date(e.start_time * 1000);
              return ed.getFullYear() === wd.getFullYear() && ed.getMonth() === wd.getMonth()
                && ed.getDate() === wd.getDate() && ed.getHours() === hour;
            });
            return (
              <div key={di} style={{
                borderLeft: '1px solid rgba(255,255,255,0.04)', padding: 1, position: 'relative',
              }}>
                {cellEvents.map(ev => (
                  <div key={ev.id} onClick={() => onEventClick(ev)} style={{
                    fontSize: 10, padding: '2px 4px', borderRadius: 3, cursor: 'pointer',
                    background: `${getEventColor(ev)}25`, borderLeft: `2px solid ${getEventColor(ev)}`,
                    color: 'rgba(255,255,255,0.8)', marginBottom: 1,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {formatTime(ev.start_time)} {ev.title}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

/** Day view — single day hourly slots */
const DayView: React.FC<{
  currentDate: Date; events: CalendarEvent[]; onEventClick: (e: CalendarEvent) => void;
}> = ({ currentDate, events, onEventClick }) => {
  const dayEvents = events.filter(e => {
    const d = new Date(e.start_time * 1000);
    return d.getFullYear() === currentDate.getFullYear() && d.getMonth() === currentDate.getMonth()
      && d.getDate() === currentDate.getDate();
  });

  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 280px)' }}>
      <div style={{
        textAlign: 'center', padding: '12px 0', fontSize: 16, fontWeight: 600,
        color: isToday(currentDate.getFullYear(), currentDate.getMonth(), currentDate.getDate())
          ? '#06b6d4' : 'rgba(255,255,255,0.8)',
      }}>
        {DAYS[currentDate.getDay()]}, {MONTHS[currentDate.getMonth()]} {currentDate.getDate()}
      </div>
      {hours.map(hour => {
        const hourEvents = dayEvents.filter(e => new Date(e.start_time * 1000).getHours() === hour);
        return (
          <div key={hour} style={{
            display: 'grid', gridTemplateColumns: '60px 1fr',
            minHeight: 48, borderBottom: '1px solid rgba(255,255,255,0.03)',
          }}>
            <div style={{
              fontSize: 11, color: 'rgba(255,255,255,0.3)', padding: '4px 8px', textAlign: 'right',
            }}>
              {hour === 0 ? '12 AM' : hour < 12 ? `${hour} AM` : hour === 12 ? '12 PM' : `${hour - 12} PM`}
            </div>
            <div style={{ padding: '2px 8px', borderLeft: '1px solid rgba(255,255,255,0.06)' }}>
              {hourEvents.map(ev => (
                <motion.div key={ev.id} whileHover={{ scale: 1.02 }} onClick={() => onEventClick(ev)}
                  style={{
                    padding: '6px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 4,
                    background: `${getEventColor(ev)}15`, borderLeft: `3px solid ${getEventColor(ev)}`,
                    backdropFilter: 'blur(8px)',
                  }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.9)' }}>{ev.title}</div>
                  <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.5)' }}>{formatTime(ev.start_time)}</div>
                </motion.div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/** Agenda view — chronological list */
const AgendaView: React.FC<{
  events: CalendarEvent[]; onEventClick: (e: CalendarEvent) => void;
}> = ({ events, onEventClick }) => {
  const now = Math.floor(Date.now() / 1000);
  const upcoming = events
    .filter(e => e.start_time >= now - 86400)
    .sort((a, b) => a.start_time - b.start_time);

  let lastDate = '';

  return (
    <div style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 280px)', padding: '0 8px' }}>
      {upcoming.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 40, color: 'rgba(255,255,255,0.3)' }}>
          No upcoming events
        </div>
      ) : (
        upcoming.map(ev => {
          const dateStr = formatDate(ev.start_time);
          const showHeader = dateStr !== lastDate;
          lastDate = dateStr;
          return (
            <React.Fragment key={ev.id}>
              {showHeader && (
                <div style={{
                  fontSize: 12, fontWeight: 600, color: '#06b6d4', padding: '16px 0 6px',
                  borderBottom: '1px solid rgba(6,182,212,0.2)',
                }}>
                  {dateStr}
                </div>
              )}
              <motion.div whileHover={{ scale: 1.01, x: 4 }} onClick={() => onEventClick(ev)}
                style={{
                  display: 'flex', gap: 12, padding: '10px 12px', cursor: 'pointer',
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                }}>
                <div style={{
                  width: 4, borderRadius: 2, background: getEventColor(ev),
                  boxShadow: `0 0 6px ${getEventColor(ev)}60`,
                  flexShrink: 0,
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'rgba(255,255,255,0.9)' }}>{ev.title}</div>
                  <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', display: 'flex', gap: 8 }}>
                    <span>{formatTime(ev.start_time)}</span>
                    <span style={{
                      background: `${getEventColor(ev)}20`, padding: '0 4px', borderRadius: 3,
                      color: getEventColor(ev), fontSize: 10,
                    }}>
                      {EVENT_TYPE_LABELS[ev.event_type] || ev.event_type}
                    </span>
                  </div>
                  {ev.scheduled_tx && (
                    <div style={{ fontSize: 10, color: '#f59e0b', marginTop: 2 }}>
                      {ev.scheduled_tx.executed ? '✓ Sent' : `→ ${ev.scheduled_tx.amount} ${ev.scheduled_tx.token}`}
                    </div>
                  )}
                </div>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)', whiteSpace: 'nowrap' }}>
                  {countdownText(ev.start_time)}
                </div>
              </motion.div>
            </React.Fragment>
          );
        })
      )}
    </div>
  );
};

// ============================================================================
// Event Create Modal
// ============================================================================

const EventCreateModal: React.FC<{
  onClose: () => void;
  onCreated: () => void;
  initialDate?: Date;
  isScheduledTx?: boolean;
}> = ({ onClose, onCreated, initialDate, isScheduledTx }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [eventType, setEventType] = useState(isScheduledTx ? 'scheduled_tx' : 'personal');
  const [startDate, setStartDate] = useState(() => {
    const base = initialDate ?? new Date();
    base.setMinutes(base.getMinutes() - base.getTimezoneOffset());
    return base.toISOString().slice(0, 16);
  });
  const [endDate, setEndDate] = useState('');
  const [allDay, setAllDay] = useState(false);
  const [color, setColor] = useState('');
  const [reminderMinutes, setReminderMinutes] = useState('15');

  // Scheduled TX fields
  const [toWallet, setToWallet] = useState('');
  const [txToken, setTxToken] = useState('QUG');
  const [txAmount, setTxAmount] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleSubmit = async () => {
    if (!title.trim()) return;
    setErrorMessage('');
    setSubmitting(true);
    try {
      const startTime = Math.floor(new Date(startDate).getTime() / 1000);

      let response;
      if (eventType === 'scheduled_tx') {
        response = await api.createScheduledTransaction({
          title,
          description: description || undefined,
          start_time: startTime,
          to_wallet: toWallet,
          token: txToken,
          amount: txAmount,
          reminder_minutes: reminderMinutes ? [parseInt(reminderMinutes)] : undefined,
        });
      } else {
        response = await api.createCalendarEvent({
          title,
          description: description || undefined,
          event_type: eventType,
          start_time: startTime,
          end_time: endDate ? Math.floor(new Date(endDate).getTime() / 1000) : undefined,
          all_day: allDay,
          color: color || undefined,
          reminder_minutes: reminderMinutes ? [parseInt(reminderMinutes)] : undefined,
        });
      }
      if (!response?.success || response?.error) {
        setErrorMessage(response?.error || 'Failed to create event. Please try again.');
        return;
      }
      onCreated();
      onClose();
    } catch (e) {
      console.error('Failed to create event:', e);
      setErrorMessage('Network error. Please check your connection.');
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 6, padding: '8px 12px', color: 'white', fontSize: 13, width: '100%',
    outline: 'none',
  };

  const labelStyle: React.CSSProperties = {
    fontSize: 11, color: 'rgba(255,255,255,0.5)', marginBottom: 4, display: 'block',
  };

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      }}>
      <motion.div
        initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
        onClick={e => e.stopPropagation()}
        style={{
          background: 'rgba(15,15,25,0.95)', border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 16, padding: 24, width: 420, maxHeight: '80vh', overflowY: 'auto',
          backdropFilter: 'blur(20px)', boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        }}>
        <div style={{
          fontSize: 18, fontWeight: 700, color: '#06b6d4', marginBottom: 20,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          {isScheduledTx ? '⏰ Schedule Transaction' : '📅 New Event'}
          <button onClick={onClose} style={{
            background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)',
            cursor: 'pointer', fontSize: 20,
          }}>×</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div>
            <label style={labelStyle}>Title</label>
            <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Event title..."
              style={inputStyle} autoFocus />
          </div>

          <div>
            <label style={labelStyle}>Description</label>
            <textarea value={description} onChange={e => setDescription(e.target.value)}
              placeholder="Optional description..." rows={2}
              style={{ ...inputStyle, resize: 'vertical' }} />
          </div>

          {!isScheduledTx && (
            <div>
              <label style={labelStyle}>Event Type</label>
              <select value={eventType} onChange={e => setEventType(e.target.value)} style={inputStyle}>
                <option value="personal">Personal</option>
                <option value="governance_vote">Governance Vote</option>
                <option value="vesting_unlock">Vesting Unlock</option>
                <option value="price_alert">Price Alert</option>
              </select>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div>
              <label style={labelStyle}>Start</label>
              <input type="datetime-local" value={startDate} onChange={e => setStartDate(e.target.value)}
                style={inputStyle} />
            </div>
            {!isScheduledTx && (
              <div>
                <label style={labelStyle}>End (optional)</label>
                <input type="datetime-local" value={endDate} onChange={e => setEndDate(e.target.value)}
                  style={inputStyle} />
              </div>
            )}
          </div>

          {!isScheduledTx && (
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input type="checkbox" checked={allDay} onChange={e => setAllDay(e.target.checked)} />
              <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.6)' }}>All day event</span>
            </label>
          )}

          {eventType === 'scheduled_tx' && (
            <>
              <div>
                <label style={labelStyle}>Recipient Wallet</label>
                <input value={toWallet} onChange={e => setToWallet(e.target.value)}
                  placeholder="64-character hex wallet address" style={inputStyle} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <div>
                  <label style={labelStyle}>Amount</label>
                  <input value={txAmount} onChange={e => setTxAmount(e.target.value)}
                    placeholder="0.00" style={inputStyle} />
                </div>
                <div>
                  <label style={labelStyle}>Token</label>
                  <select value={txToken} onChange={e => setTxToken(e.target.value)} style={inputStyle}>
                    <option value="QUG">QUG</option>
                    <option value="QUGUSD">QUGUSD</option>
                  </select>
                </div>
              </div>
            </>
          )}

          <div>
            <label style={labelStyle}>Reminder (minutes before)</label>
            <select value={reminderMinutes} onChange={e => setReminderMinutes(e.target.value)} style={inputStyle}>
              <option value="">No reminder</option>
              <option value="5">5 minutes</option>
              <option value="15">15 minutes</option>
              <option value="30">30 minutes</option>
              <option value="60">1 hour</option>
              <option value="1440">1 day</option>
            </select>
          </div>

          {!isScheduledTx && (
            <div>
              <label style={labelStyle}>Color</label>
              <div style={{ display: 'flex', gap: 6 }}>
                {Object.entries(EVENT_COLORS).map(([type, c]) => (
                  <div key={type} onClick={() => setColor(c)} style={{
                    width: 24, height: 24, borderRadius: '50%', background: c, cursor: 'pointer',
                    border: color === c ? '2px solid white' : '2px solid transparent',
                    boxShadow: color === c ? `0 0 8px ${c}` : 'none',
                  }} title={EVENT_TYPE_LABELS[type]} />
                ))}
              </div>
            </div>
          )}

          {errorMessage && (
            <div style={{
              color: '#f87171',
              background: 'rgba(248,113,113,0.1)',
              border: '1px solid rgba(248,113,113,0.3)',
              borderRadius: 6,
              padding: '8px 12px',
              fontSize: 12,
            }}>
              ⚠️ {errorMessage}
            </div>
          )}

          <motion.button
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
            onClick={handleSubmit} disabled={submitting || !title.trim()}
            style={{
              background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
              border: 'none', borderRadius: 8, padding: '10px 20px',
              color: 'white', fontWeight: 600, fontSize: 14, cursor: 'pointer',
              opacity: submitting || !title.trim() ? 0.5 : 1,
              marginTop: 4,
            }}>
            {submitting ? 'Creating...' : isScheduledTx ? 'Schedule Transaction' : 'Create Event'}
          </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// ============================================================================
// Event Detail Modal
// ============================================================================

const EventDetailModal: React.FC<{
  event: CalendarEvent; onClose: () => void; onDelete: (id: string) => void;
  onShare: (id: string) => void;
}> = ({ event, onClose, onDelete, onShare }) => {
  const [countdown, setCountdown] = useState(countdownText(event.start_time));

  useEffect(() => {
    const interval = setInterval(() => setCountdown(countdownText(event.start_time)), 1000);
    return () => clearInterval(interval);
  }, [event.start_time]);

  const eventColor = getEventColor(event);

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      }}>
      <motion.div
        initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
        onClick={e => e.stopPropagation()}
        style={{
          background: 'rgba(15,15,25,0.95)', border: `1px solid ${eventColor}40`,
          borderRadius: 16, padding: 24, width: 400, maxHeight: '80vh', overflowY: 'auto',
          backdropFilter: 'blur(20px)', boxShadow: `0 20px 60px rgba(0,0,0,0.5), 0 0 40px ${eventColor}10`,
        }}>
        {/* Color bar */}
        <div style={{
          height: 4, borderRadius: 2, background: `linear-gradient(90deg, ${eventColor}, ${eventColor}40)`,
          marginBottom: 16, boxShadow: `0 0 10px ${eventColor}40`,
        }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'white' }}>{event.title}</div>
            <span style={{
              fontSize: 10, padding: '2px 8px', borderRadius: 4,
              background: `${eventColor}20`, color: eventColor, marginTop: 4, display: 'inline-block',
            }}>
              {EVENT_TYPE_LABELS[event.event_type] || event.event_type}
            </span>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)',
            cursor: 'pointer', fontSize: 20,
          }}>×</button>
        </div>

        {/* Countdown timer */}
        <div style={{
          background: 'rgba(255,255,255,0.03)', borderRadius: 8, padding: '12px 16px',
          marginBottom: 16, textAlign: 'center', border: '1px solid rgba(255,255,255,0.06)',
        }}>
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', marginBottom: 4 }}>
            {event.start_time * 1000 > Date.now() ? 'STARTS IN' : 'STARTED'}
          </div>
          <div style={{
            fontSize: 28, fontWeight: 700, color: eventColor,
            fontFamily: '"JetBrains Mono", monospace', letterSpacing: 2,
            textShadow: `0 0 20px ${eventColor}40`,
          }}>
            {countdown}
          </div>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', marginTop: 4 }}>
            {formatDate(event.start_time)} at {formatTime(event.start_time)}
          </div>
        </div>

        {event.description && (
          <div style={{
            fontSize: 13, color: 'rgba(255,255,255,0.6)', marginBottom: 16,
            lineHeight: 1.5, padding: '8px 12px', background: 'rgba(255,255,255,0.02)',
            borderRadius: 6,
          }}>
            {event.description}
          </div>
        )}

        {/* Scheduled TX details */}
        {event.scheduled_tx && (
          <div style={{
            background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
            borderRadius: 8, padding: 12, marginBottom: 16,
          }}>
            <div style={{ fontSize: 11, color: '#f59e0b', fontWeight: 600, marginBottom: 8 }}>
              {event.scheduled_tx.executed ? '✓ TRANSACTION EXECUTED' : '⏰ SCHEDULED TRANSACTION'}
            </div>
            <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.8)' }}>
              {event.scheduled_tx.amount} {event.scheduled_tx.token}
            </div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', marginTop: 4 }}>
              → {event.scheduled_tx.to_wallet.slice(0, 8)}...{event.scheduled_tx.to_wallet.slice(-8)}
            </div>
            {event.scheduled_tx.tx_hash && (
              <div style={{ fontSize: 10, color: '#22c55e', marginTop: 4 }}>
                TX: {event.scheduled_tx.tx_hash}
              </div>
            )}
            {event.scheduled_tx.error && (
              <div style={{ fontSize: 10, color: '#ef4444', marginTop: 4 }}>
                Error: {event.scheduled_tx.error}
              </div>
            )}
          </div>
        )}

        {event.source_peer && (
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginBottom: 12 }}>
            Source: {event.source_peer}
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          {!event.shared && event.event_type !== 'network_milestone' && (
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              onClick={() => onShare(event.id)}
              style={{
                flex: 1, background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)',
                borderRadius: 6, padding: '8px 12px', color: '#3b82f6', cursor: 'pointer',
                fontSize: 12, fontWeight: 600,
              }}>
              Share to Network
            </motion.button>
          )}
          {event.event_type !== 'network_milestone' && (
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              onClick={() => { onDelete(event.id); onClose(); }}
              style={{
                flex: 1, background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: 6, padding: '8px 12px', color: '#ef4444', cursor: 'pointer',
                fontSize: 12, fontWeight: 600,
              }}>
              Delete
            </motion.button>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
};

// ============================================================================
// Main Calendar Component
// ============================================================================

const CalendarScreen: React.FC = () => {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [viewMode, setViewMode] = useState<ViewMode>('month');
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [networkEvents, setNetworkEvents] = useState<CalendarEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showScheduleTxModal, setShowScheduleTxModal] = useState(false);
  const [createDate, setCreateDate] = useState<Date | undefined>();
  const [loading, setLoading] = useState(true);

  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth();

  // Fetch events
  const fetchEvents = useCallback(async () => {
    try {
      // Calculate date range based on view
      const start = new Date(currentYear, currentMonth - 1, 1);
      const end = new Date(currentYear, currentMonth + 2, 0);
      const startDate = formatDateKey(start.getFullYear(), start.getMonth(), 1);
      const endDate = formatDateKey(end.getFullYear(), end.getMonth(), end.getDate());

      const [eventsRes, networkRes] = await Promise.all([
        api.getCalendarEvents(startDate, endDate),
        api.getNetworkEvents(),
      ]);

      if (eventsRes?.data) setEvents(eventsRes.data);
      if (networkRes?.data) {
        setNetworkEvents(networkRes.data.map((ne: any) => ({
          ...ne,
          event_type: ne.event_type || 'network_milestone',
          created_at: Date.now() / 1000,
        })));
      }
    } catch (e) {
      console.error('Failed to fetch calendar events:', e);
    } finally {
      setLoading(false);
    }
  }, [currentYear, currentMonth]);

  useEffect(() => { fetchEvents(); }, [fetchEvents]);

  // SSE event listeners
  useEffect(() => {
    const handleCreated = () => fetchEvents();
    const handleReminder = (e: any) => {
      const detail = (e as CustomEvent).detail;
      if (detail && Notification.permission === 'granted') {
        new Notification(`📅 ${detail.title}`, {
          body: `Starting in ${detail.minutes_until} minutes`,
        });
      }
    };
    const handleTxExecuted = () => fetchEvents();

    window.addEventListener('calendar-event-created', handleCreated);
    window.addEventListener('calendar-reminder', handleReminder);
    window.addEventListener('scheduled-tx-executed', handleTxExecuted);
    return () => {
      window.removeEventListener('calendar-event-created', handleCreated);
      window.removeEventListener('calendar-reminder', handleReminder);
      window.removeEventListener('scheduled-tx-executed', handleTxExecuted);
    };
  }, [fetchEvents]);

  // Combine all events
  const allEvents = useMemo(() => [...events, ...networkEvents], [events, networkEvents]);

  // Navigation
  const navigate = (dir: -1 | 1) => {
    const d = new Date(currentDate);
    if (viewMode === 'month') d.setMonth(d.getMonth() + dir);
    else if (viewMode === 'week') d.setDate(d.getDate() + dir * 7);
    else d.setDate(d.getDate() + dir);
    setCurrentDate(d);
  };

  const goToToday = () => setCurrentDate(new Date());

  const handleDayClick = (y: number, m: number, d: number) => {
    setCurrentDate(new Date(y, m, d));
    if (viewMode === 'month') setViewMode('day');
  };

  const handleDeleteEvent = async (id: string) => {
    try {
      await api.deleteCalendarEvent(id);
      fetchEvents();
    } catch (e) {
      console.error('Failed to delete event:', e);
    }
  };

  const handleShareEvent = async (id: string) => {
    try {
      await api.shareCalendarEvent(id);
      fetchEvents();
    } catch (e) {
      console.error('Failed to share event:', e);
    }
  };

  // Month view grid
  const renderMonthView = () => {
    const daysInMonth = getDaysInMonth(currentYear, currentMonth);
    const firstDay = getFirstDayOfMonth(currentYear, currentMonth);
    const prevMonth = currentMonth === 0 ? 11 : currentMonth - 1;
    const prevYear = currentMonth === 0 ? currentYear - 1 : currentYear;
    const daysInPrevMonth = getDaysInMonth(prevYear, prevMonth);

    const cells: React.ReactNode[] = [];

    // Previous month days
    for (let i = firstDay - 1; i >= 0; i--) {
      const day = daysInPrevMonth - i;
      cells.push(
        <DayCell key={`prev-${day}`} day={day} month={prevMonth} year={prevYear}
          events={allEvents} isCurrentMonth={false} onDayClick={handleDayClick}
          onEventClick={setSelectedEvent} />
      );
    }

    // Current month days
    for (let day = 1; day <= daysInMonth; day++) {
      cells.push(
        <DayCell key={`curr-${day}`} day={day} month={currentMonth} year={currentYear}
          events={allEvents} isCurrentMonth={true} onDayClick={handleDayClick}
          onEventClick={setSelectedEvent} />
      );
    }

    // Next month days
    const remaining = 42 - cells.length;
    const nextMonth = currentMonth === 11 ? 0 : currentMonth + 1;
    const nextYear = currentMonth === 11 ? currentYear + 1 : currentYear;
    for (let day = 1; day <= remaining; day++) {
      cells.push(
        <DayCell key={`next-${day}`} day={day} month={nextMonth} year={nextYear}
          events={allEvents} isCurrentMonth={false} onDayClick={handleDayClick}
          onEventClick={setSelectedEvent} />
      );
    }

    return (
      <div>
        {/* Day headers */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 4, marginBottom: 4 }}>
          {DAYS.map(d => (
            <div key={d} style={{
              textAlign: 'center', fontSize: 11, color: 'rgba(255,255,255,0.4)',
              padding: '6px 0', fontWeight: 600,
            }}>
              {d}
            </div>
          ))}
        </div>
        {/* Calendar grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 4 }}>
          {cells}
        </div>
      </div>
    );
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Pulse animation CSS */}
      <style>{`
        @keyframes calendarTodayPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.2); }
          50% { box-shadow: 0 0 12px 2px rgba(6, 182, 212, 0.15); }
        }
        @keyframes fabGlow {
          0%, 100% { box-shadow: 0 4px 20px rgba(6, 182, 212, 0.3); }
          50% { box-shadow: 0 4px 30px rgba(139, 92, 246, 0.4); }
        }
      `}</style>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
            onClick={() => navigate(-1)}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 6, width: 32, height: 32, color: 'white', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14,
            }}>
            ‹
          </motion.button>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'white', minWidth: 180, textAlign: 'center' }}>
            {viewMode === 'day'
              ? `${MONTHS[currentMonth]} ${currentDate.getDate()}, ${currentYear}`
              : `${MONTHS[currentMonth]} ${currentYear}`
            }
          </div>
          <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
            onClick={() => navigate(1)}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 6, width: 32, height: 32, color: 'white', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14,
            }}>
            ›
          </motion.button>
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
            onClick={goToToday}
            style={{
              background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.3)',
              borderRadius: 6, padding: '4px 12px', color: '#06b6d4', cursor: 'pointer',
              fontSize: 12, fontWeight: 600,
            }}>
            Today
          </motion.button>
        </div>

        {/* View mode selector */}
        <div style={{
          display: 'flex', background: 'rgba(255,255,255,0.03)',
          borderRadius: 8, border: '1px solid rgba(255,255,255,0.06)', overflow: 'hidden',
        }}>
          {(['month', 'week', 'day', 'agenda'] as ViewMode[]).map(mode => (
            <button key={mode} onClick={() => setViewMode(mode)} style={{
              background: viewMode === mode ? 'rgba(6,182,212,0.15)' : 'transparent',
              border: 'none', padding: '6px 14px', color: viewMode === mode ? '#06b6d4' : 'rgba(255,255,255,0.5)',
              cursor: 'pointer', fontSize: 12, fontWeight: viewMode === mode ? 600 : 400,
              borderRight: '1px solid rgba(255,255,255,0.06)',
              transition: 'all 0.2s',
            }}>
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Event type legend */}
      <div style={{
        display: 'flex', gap: 12, padding: '8px 16px', flexWrap: 'wrap',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}>
        {Object.entries(EVENT_TYPE_LABELS).map(([type, label]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: EVENT_COLORS[type], boxShadow: `0 0 4px ${EVENT_COLORS[type]}60`,
            }} />
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>{label}</span>
          </div>
        ))}
      </div>

      {/* Calendar body */}
      <div style={{ flex: 1, padding: '8px 12px', overflow: 'hidden' }}>
        {loading ? (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            height: '100%', color: 'rgba(255,255,255,0.3)',
          }}>
            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
              ⟳
            </motion.div>
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={`${viewMode}-${currentDate.toISOString()}`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              {viewMode === 'month' && renderMonthView()}
              {viewMode === 'week' && (
                <WeekView currentDate={currentDate} events={allEvents} onEventClick={setSelectedEvent} />
              )}
              {viewMode === 'day' && (
                <DayView currentDate={currentDate} events={allEvents} onEventClick={setSelectedEvent} />
              )}
              {viewMode === 'agenda' && (
                <AgendaView events={allEvents} onEventClick={setSelectedEvent} />
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </div>

      {/* Floating Action Button */}
      <div style={{ position: 'absolute', bottom: 24, right: 24, display: 'flex', flexDirection: 'column', gap: 10, zIndex: 50 }}>
        <motion.button
          whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
          onClick={() => { setCreateDate(currentDate); setShowScheduleTxModal(true); }}
          style={{
            width: 44, height: 44, borderRadius: '50%',
            background: 'rgba(245,158,11,0.15)', border: '1px solid rgba(245,158,11,0.4)',
            color: '#f59e0b', cursor: 'pointer', fontSize: 18,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 4px 20px rgba(245,158,11,0.2)',
          }}
          title="Schedule Transaction"
        >
          ⏰
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
          onClick={() => { setCreateDate(currentDate); setShowCreateModal(true); }}
          style={{
            width: 56, height: 56, borderRadius: '50%',
            background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
            border: 'none', color: 'white', cursor: 'pointer',
            fontSize: 24, fontWeight: 300,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            animation: 'fabGlow 3s ease-in-out infinite',
          }}
          title="New Event"
        >
          +
        </motion.button>
      </div>

      {/* Modals */}
      <AnimatePresence>
        {showCreateModal && (
          <EventCreateModal
            onClose={() => setShowCreateModal(false)}
            onCreated={fetchEvents}
            initialDate={createDate}
          />
        )}
        {showScheduleTxModal && (
          <EventCreateModal
            onClose={() => setShowScheduleTxModal(false)}
            onCreated={fetchEvents}
            initialDate={createDate}
            isScheduledTx
          />
        )}
        {selectedEvent && (
          <EventDetailModal
            event={selectedEvent}
            onClose={() => setSelectedEvent(null)}
            onDelete={handleDeleteEvent}
            onShare={handleShareEvent}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default CalendarScreen;
