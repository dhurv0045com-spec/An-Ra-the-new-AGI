import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, BrainCircuit, Cpu, Database, Terminal, Zap } from 'lucide-react';

const fmt = (value, fallback = '-') => {
  if (value === null || value === undefined || value === '') return fallback;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return fallback;
    if (Math.abs(value) >= 1_000_000) return value.toLocaleString();
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
};

const StatTile = ({ icon, label, value, accent = 'var(--accent-cyan)' }) => (
  <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--panel-border)', borderRadius: '8px', padding: '14px', minHeight: '92px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase', marginBottom: '10px' }}>
      {icon}
      {label}
    </div>
    <div className="mono" style={{ color: accent, fontSize: '1rem', lineHeight: 1.35, wordBreak: 'break-word' }}>{fmt(value)}</div>
  </div>
);

const JsonPanel = ({ title, data }) => (
  <div className="glass-panel" style={{ minHeight: 0 }}>
    <div className="panel-header" style={{ padding: '14px 16px', borderBottom: '1px solid var(--panel-border)' }}>
      <h3 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px' }}>
        <Terminal size={15} /> {title}
      </h3>
    </div>
    <pre className="mono" style={{ margin: 0, padding: '16px', overflow: 'auto', color: 'var(--text-secondary)', fontSize: '11px', lineHeight: 1.45, whiteSpace: 'pre-wrap' }}>
      {JSON.stringify(data ?? {}, null, 2)}
    </pre>
  </div>
);

const DeveloperMatrix = () => {
  const [status, setStatus] = useState(null);
  const [hal, setHal] = useState(null);
  const [sessions, setSessions] = useState(null);
  const [phase, setPhase] = useState(null);
  const [error, setError] = useState('');
  const [lastRefresh, setLastRefresh] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [statusRes, halRes, sessionsRes, phaseRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/hal/state'),
        fetch('/api/sessions'),
        fetch('/api/phase-health'),
      ]);
      setStatus(statusRes.ok ? await statusRes.json() : { error: await statusRes.text() });
      setHal(halRes.ok ? await halRes.json() : { error: await halRes.text() });
      setSessions(sessionsRes.ok ? await sessionsRes.json() : { error: await sessionsRes.text() });
      setPhase(phaseRes.ok ? await phaseRes.json() : { error: await phaseRes.text() });
      setError('');
      setLastRefresh(new Date().toLocaleTimeString());
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 2500);
    return () => clearInterval(interval);
  }, [refresh]);

  const checkpointState = status?.checkpoint_state || {};
  const hormoneRows = useMemo(() => {
    const hormones = hal?.hormones || {};
    return Object.entries(hormones).sort(([a], [b]) => a.localeCompare(b));
  }, [hal]);

  return (
    <div className="matrix-view animate-in" style={{ display: 'grid', gridTemplateColumns: '1.25fr 0.9fr', gap: '20px', height: '100%', minHeight: 0 }}>
      <div className="glass-panel" style={{ padding: '20px', gap: '18px', minHeight: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px' }}>
          <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
            <BrainCircuit size={18} /> DEVELOPER MATRIX
          </h2>
          <button onClick={refresh} style={{ border: '1px solid var(--panel-border)', background: 'rgba(255,255,255,0.04)', color: 'var(--text-secondary)', borderRadius: '8px', padding: '8px 10px', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <Activity size={14} /> {lastRefresh || 'refresh'}
          </button>
        </div>

        {error && <div style={{ padding: '10px', border: '1px solid var(--accent-emerald)', color: 'var(--accent-emerald)', borderRadius: '8px' }}>{error}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '12px' }}>
          <StatTile icon={<Activity size={14} />} label="Runtime" value={status?.status} />
          <StatTile icon={<Cpu size={14} />} label="Device" value={status?.device} accent="var(--accent-purple)" />
          <StatTile icon={<Zap size={14} />} label="Profile" value={status?.profile} />
          <StatTile icon={<Database size={14} />} label="Sessions" value={status?.sessions_active ?? sessions?.count} accent="var(--accent-emerald)" />
          <StatTile icon={<BrainCircuit size={14} />} label="Parameters" value={status?.param_count} />
          <StatTile icon={<Terminal size={14} />} label="Context" value={status?.block_size ? `${status.block_size} tokens` : '-'} />
          <StatTile icon={<Activity size={14} />} label="Train Step" value={checkpointState.global_step} accent="var(--accent-purple)" />
          <StatTile icon={<Zap size={14} />} label="Best Loss" value={checkpointState.best_loss} accent="var(--accent-emerald)" />
        </div>

        <div style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid var(--panel-border)', borderRadius: '8px', padding: '14px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase', marginBottom: '8px' }}>Active Checkpoint</div>
          <div className="mono" style={{ color: 'var(--text-primary)', fontSize: '12px', wordBreak: 'break-all' }}>{fmt(status?.checkpoint)}</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', minHeight: 0 }}>
          <JsonPanel title="Runtime Payload" data={status} />
          <JsonPanel title="Phase Health" data={phase} />
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minHeight: 0 }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h3 className="heading-sm" style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} /> HAL HORMONE VECTOR
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {hormoneRows.length === 0 ? (
              <div style={{ color: 'var(--text-muted)' }}>No HAL state published yet.</div>
            ) : hormoneRows.map(([name, raw]) => {
              const value = Number(raw || 0);
              return (
                <div key={name}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '12px', textTransform: 'uppercase' }}>{name}</span>
                    <span className="mono" style={{ color: 'var(--accent-cyan)', fontSize: '12px' }}>{value.toFixed(3)}</span>
                  </div>
                  <div style={{ height: '5px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.max(0, Math.min(100, value * 100))}%`, background: name === 'cortisol' ? 'var(--accent-emerald)' : 'var(--accent-cyan)' }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <JsonPanel title="HAL Raw State" data={hal} />
        <JsonPanel title="Session Index" data={sessions} />
      </div>
    </div>
  );
};

export default DeveloperMatrix;
