import React, { useState, useEffect } from 'react';
import { Brain, Cpu, TrendingDown, Play, Info, Activity, AlertTriangle } from 'lucide-react';

const TrainingPanel = () => {
  const [data, setData] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/train/status');
      if (response.ok) {
        setData(await response.json());
      }
    } catch (e) {
      console.error("Training status fetch error:", e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const triggerTraining = async () => {
    setIsTraining(true);
    try {
      const response = await fetch('/api/train/trigger', { method: 'POST' });
      if (response.ok) {
        alert("Self-training triggered successfully.");
        fetchStatus();
      }
    } catch (e) {
      console.error("Trigger error:", e);
    }
    setIsTraining(false);
  };

  if (!data) return <div style={{ padding: '40px', textAlign: 'center' }}>Initializing Neural Trainer...</div>;

  const { stats, hardware, latest_run } = data;

  return (
    <div className="training-panel animate-in" style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '20px', height: '100%' }}>
      
      {/* Left: Evolution Metrics */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Brain size={18} color="var(--accent-purple)" /> CONTINUOUS EVOLUTION
          </h2>
          <div style={{ display: 'flex', gap: '12px' }}>
             <span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Status: {latest_run?.status || 'idle'}</span>
             {latest_run?.status === 'running' && <Activity size={14} className="animate-pulse" color="var(--accent-cyan)" />}
          </div>
        </div>

        {/* Mini Chart Mockup (Loss Curve) */}
        <div style={{ flex: 1, background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid var(--panel-border)', padding: '20px', position: 'relative', overflow: 'hidden' }}>
           <div className="heading-sm" style={{ fontSize: '10px', marginBottom: '20px' }}>Loss Convergence (Self-Correction)</div>
           <div style={{ height: '150px', display: 'flex', alignItems: 'flex-end', gap: '4px' }}>
              {(latest_run?.loss_history || [0.8, 0.7, 0.75, 0.6, 0.55, 0.4, 0.35]).map((val, i) => (
                <div key={i} style={{ 
                  flex: 1, 
                  background: 'linear-gradient(to top, var(--accent-purple), transparent)', 
                  height: `${(val.loss || val) * 100}%`,
                  borderRadius: '2px 2px 0 0',
                  opacity: 0.8
                }} />
              ))}
           </div>
           <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0.1 }}>
              <TrendingDown size={120} />
           </div>
        </div>

        {/* Stats Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
           <div className="telemetry-item" style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>TOTAL EXAMPLES</div>
              <div className="mono" style={{ fontSize: '1.2rem', color: 'var(--accent-cyan)' }}>{stats.total_examples}</div>
           </div>
           <div className="telemetry-item" style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>HIGH QUALITY</div>
              <div className="mono" style={{ fontSize: '1.2rem', color: 'var(--accent-emerald)' }}>{stats.high_quality}</div>
           </div>
           <div className="telemetry-item" style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>AVG QUALITY</div>
              <div className="mono" style={{ fontSize: '1.2rem', color: 'var(--accent-purple)' }}>{(stats.avg_quality * 100).toFixed(1)}%</div>
           </div>
        </div>
      </div>

      {/* Right: Controls & Hardware */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Hardware Profile */}
        <div className="glass-panel" style={{ padding: '20px' }}>
           <h3 className="heading-sm" style={{ marginBottom: '16px', fontSize: '10px' }}>Neural Hardware Capacity</h3>
           <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                 <span style={{ fontSize: '0.85rem' }}>{hardware.gpu_count > 0 ? `NVIDIA GPU (x${hardware.gpu_count})` : 'CPU Engine'}</span>
                 <Cpu size={14} color={hardware.gpu_count > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                 <span style={{ fontSize: '0.85rem' }}>RAM Capacity</span>
                 <span className="mono" style={{ fontSize: '0.85rem' }}>{hardware.ram_gb ? hardware.ram_gb.toFixed(1) : '??'} GB</span>
              </div>
              <div style={{ padding: '8px', background: 'var(--accent-cyan-glow)', borderRadius: '6px', border: '1px solid var(--accent-cyan)', fontSize: '0.75rem', textAlign: 'center', color: 'var(--accent-cyan)' }}>
                 RECOMMENDED SCALE: {hardware.recommended_config.toUpperCase()}
              </div>
           </div>
        </div>

        {/* Trigger */}
        <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
           <div style={{ marginBottom: '20px' }}>
              <AlertTriangle size={32} color="var(--accent-purple)" style={{ marginBottom: '12px' }} />
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                Found <span style={{ color: 'var(--accent-cyan)' }}>{stats.unused}</span> new high-quality interaction datasets ready for assimilation.
              </p>
           </div>
           <button 
             onClick={triggerTraining}
             disabled={isTraining || stats.unused === 0}
             style={{ 
               background: stats.unused > 0 ? 'linear-gradient(135deg, var(--accent-purple), var(--accent-cyan))' : 'var(--panel-border)',
               color: 'white',
               border: 'none',
               padding: '12px',
               borderRadius: '10px',
               fontWeight: '700',
               cursor: stats.unused > 0 ? 'pointer' : 'not-allowed',
               display: 'flex',
               alignItems: 'center',
               justifyContent: 'center',
               gap: '10px',
               transition: 'var(--transition-smooth)'
             }}
           >
             {isTraining ? <Activity className="animate-spin" size={18} /> : <Play size={18} fill="white" />}
             START SELF-TRAINING
           </button>
           <p style={{ marginTop: '12px', fontSize: '10px', color: 'var(--text-muted)' }}>
             Assimilation may utilize significant system resources.
           </p>
        </div>

      </div>
    </div>
  );
};

export default TrainingPanel;
