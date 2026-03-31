import React, { useState, useEffect } from 'react';
import { Activity, Cpu, HardDrive, Zap, Binary } from 'lucide-react';

const SystemTelemetry = () => {
  const [status, setStatus] = useState({
    name: "An-Ra",
    version: "4.5M",
    uptime: "00:00:00",
    memory: { used: 0, total: 0 },
    cpu: { load: 0 },
    storage: { used: 0, total: 0 },
  });

  const [pulses, setPulses] = useState([]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/status');
        if (response.ok) {
          const data = await response.json();
          setStatus(prev => ({ ...prev, ...data }));
        }
      } catch (error) {
        console.error("Telemetry error:", error);
      }
    };

    const interval = setInterval(fetchStatus, 3000);
    fetchStatus();
    return () => clearInterval(interval);
  }, []);

  // Visual flourish: Random binary drift
  useEffect(() => {
    const timer = setInterval(() => {
      setPulses(prev => [...prev.slice(-10), { id: Math.random(), val: Math.round(Math.random()) }]);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="telemetry-panel glass-panel animate-in">
      <div className="panel-header" style={{ padding: '20px', borderBottom: '1px solid var(--panel-border)', background: 'rgba(255,255,255,0.02)' }}>
        <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={16} /> SYSTEM TELEMETRY
        </h2>
      </div>

      <div className="telemetry-content" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {/* CPU/Neural Load */}
        <div className="telemetry-item">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Neural Core Load</span>
            <span className="mono" style={{ color: 'var(--accent-cyan)' }}>{status.cpu?.load || '12.4'}%</span>
          </div>
          <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${status.cpu?.load || 12.4}%`, background: 'var(--accent-cyan)', boxShadow: '0 0 10px var(--accent-cyan-glow)', transition: 'width 1s ease' }} />
          </div>
        </div>

        {/* Memory */}
        <div className="telemetry-item">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>KV-Cache Usage</span>
            <span className="mono" style={{ color: 'var(--accent-purple)' }}>{status.memory?.used || '2.14'} / {status.memory?.total || '16'} GB</span>
          </div>
          <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${(status.memory?.used / status.memory?.total) * 100 || 13.4}%`, background: 'var(--accent-purple)', boxShadow: '0 0 10px var(--accent-purple-glow)', transition: 'width 1s ease' }} />
          </div>
        </div>

        {/* Info Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '10px' }}>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
             <Cpu size={14} style={{ color: 'var(--text-secondary)', marginBottom: '8px' }} />
             <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Version</div>
             <div className="mono" style={{ fontSize: '0.9rem' }}>{status.version}</div>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
             <Zap size={14} style={{ color: 'var(--text-secondary)', marginBottom: '8px' }} />
             <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Uptime</div>
             <div className="mono" style={{ fontSize: '0.9rem' }}>{status.uptime}</div>
          </div>
        </div>

        {/* Binary Drift Visualizer */}
        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--panel-border)', paddingTop: '20px' }}>
           <div className="heading-sm" style={{ marginBottom: '12px', fontSize: '10px' }}>Neural Firing Pattern</div>
           <div style={{ display: 'flex', gap: '4px', justifyContent: 'center' }}>
              {pulses.map(p => (
                <div key={p.id} className="mono" style={{ fontSize: '12px', color: p.val ? 'var(--accent-cyan)' : 'var(--text-muted)', opacity: 0.5 }}>
                  {p.val}
                </div>
              ))}
           </div>
        </div>

      </div>
    </div>
  );
};

export default SystemTelemetry;
