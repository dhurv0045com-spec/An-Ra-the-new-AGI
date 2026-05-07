import React, { useEffect, useRef, useState } from 'react';
import { BrainCircuit, AlertTriangle, Sparkles } from 'lucide-react';

const HORMONES = [
  ['dopamine', '#f59e0b'],
  ['serotonin', '#60a5fa'],
  ['cortisol', '#ef4444'],
  ['adrenaline', '#fb923c'],
  ['oxytocin', '#ec4899'],
  ['norepinephrine', '#a78bfa'],
  ['endorphin', '#10b981'],
];

const HormonalStatePanel = () => {
  // # AN: Surface live HAL telemetry where sovereignty decisions are monitored.
  const [state, setState] = useState({ hormones: {}, counters: {} });
  const previous = useRef({ adrenaline: 0, cortisol: 0 });

  useEffect(() => {
    const fetchState = async () => {
      try {
        const response = await fetch('/api/hal/state');
        if (response.ok) {
          const next = await response.json();
          setState(current => {
            previous.current = {
              adrenaline: Number(current.hormones?.adrenaline || 0),
              cortisol: Number(current.hormones?.cortisol || 0),
            };
            return next;
          });
        }
      } catch (error) {
        console.error('HAL state error:', error);
      }
    };
    fetchState();
    const interval = setInterval(fetchState, 3000);
    return () => clearInterval(interval);
  }, []);

  const hormones = state.hormones || {};
  const cortisol = Number(hormones.cortisol || 0);
  const adrenalineWasHigh = Number(previous.current.adrenaline || 0) > 0.5;
  const stressCascade = cortisol > 0.6 && adrenalineWasHigh;
  const flowState = Number(hormones.endorphin || 0) > 0.6;

  return (
    <div className="glass-panel" style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div className="heading-sm" style={{ fontSize: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <BrainCircuit size={14} /> HAL STATE
      </div>

      {stressCascade && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px', borderRadius: '8px', border: '1px solid rgba(239,68,68,0.4)', color: '#fecaca', background: 'rgba(239,68,68,0.12)', fontSize: '0.78rem' }}>
          <AlertTriangle size={14} /> ⚠️ STRESS CASCADE ACTIVE
        </div>
      )}

      {flowState && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px', borderRadius: '8px', border: '1px solid rgba(16,185,129,0.4)', color: '#bbf7d0', background: 'rgba(16,185,129,0.12)', fontSize: '0.78rem' }}>
          <Sparkles size={14} /> ✨ FLOW STATE
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {HORMONES.map(([name, color]) => {
          const value = Math.max(0, Math.min(1, Number(hormones[name] || 0)));
          return (
            <div key={name}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>{name}</span>
                <span className="mono" style={{ fontSize: '0.72rem', color }}>{value.toFixed(2)}</span>
              </div>
              <div style={{ height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${value * 100}%`, background: color, transition: 'width 0.5s ease' }} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
        failures={state.counters?.consecutive_failures || 0}
      </div>
    </div>
  );
};

export default HormonalStatePanel;
