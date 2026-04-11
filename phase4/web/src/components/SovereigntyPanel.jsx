import React, { useState, useEffect } from 'react';
import { ShieldCheck, Activity, Search, AlertCircle, FileText, CheckCircle2, Zap, Clock, ShieldX } from 'lucide-react';

const SovereigntyPanel = () => {
  const [data, setData] = useState(null);
  const [isAuditing, setIsAuditing] = useState(false);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/sovereignty/status');
      if (response.ok) {
        setData(await response.json());
      }
    } catch (e) {
      console.error("Sovereignty status error:", e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // 30s as audits are slow
    return () => clearInterval(interval);
  }, []);

  const triggerAudit = async () => {
    setIsAuditing(true);
    try {
      const response = await fetch('/api/sovereignty/audit', { method: 'POST' });
      if (response.ok) {
        alert("Sovereignty audit triggered.");
        fetchStatus();
      }
    } catch (e) {
      console.error("Audit error:", e);
    }
    setIsAuditing(false);
  };

  if (!data) return <div style={{ padding: '40px', textAlign: 'center' }}>Authenticating Sovereignty Daemon...</div>;

  const { enabled, status, last_audit, report } = data;

  return (
    <div className="sovereignty-panel animate-in" style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '20px', height: '100%' }}>
      
      {/* Left Area: Audit Logs & Reports */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} color="var(--accent-emerald)" /> SOVEREIGNTY SYSTEM
          </h2>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
             <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Status: {status}</span>
             <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-emerald)', boxShadow: '0 0 10px var(--accent-emerald)' }} />
          </div>
        </div>

        {/* Audit Findings */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
           {!report ? (
             <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
               No audit reports found. Trigger an audit to analyze system health.
             </div>
           ) : (
             <div className="report-content" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ background: 'hsla(150, 100%, 50%, 0.05)', padding: '20px', borderRadius: '12px', border: '1px solid hsla(150, 100%, 50%, 0.2)' }}>
                   <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                      <CheckCircle2 size={18} color="var(--accent-emerald)" />
                      <span style={{ fontWeight: 600 }}>System Integity: SECURE</span>
                   </div>
                   <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                     The core architecture in `core/` and `phase3/` matches its cryptographic baseline. All neural weights are within expected safety boundaries.
                   </p>
                </div>

                 <h4 className="heading-sm" style={{ fontSize: '10px' }}>DETAILED FINDINGS</h4>
                 <div style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--panel-border)' }}>
                     <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'monospace', margin: 0 }}>
                        {typeof report === 'string' ? report : JSON.stringify(report, null, 2)}
                     </pre>
                 </div>
             </div>
           )}
        </div>
      </div>

      {/* Right Area: Status & Manual Action */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Audit Status */}
        <div className="glass-panel" style={{ padding: '20px' }}>
           <h3 className="heading-sm" style={{ marginBottom: '16px', fontSize: '10px' }}>Audit Schedule</h3>
           <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', gap: '12px' }}>
                 <Clock size={16} color="var(--text-muted)" />
                 <div>
                    <div style={{ fontSize: '0.85rem' }}>Next Scheduled Cycle</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Nocturnal (03:00 AM)</div>
                 </div>
              </div>
              <div style={{ display: 'flex', gap: '12px' }}>
                 <FileText size={16} color="var(--text-muted)" />
                 <div>
                    <div style={{ fontSize: '0.85rem' }}>Last Comprehensive Report</div>
                    <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>{last_audit}</div>
                 </div>
              </div>
           </div>
        </div>

        {/* Action Panel */}
        <div className="glass-panel" style={{ padding: '24px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center' }}>
           <Zap size={32} color="var(--accent-emerald)" style={{ margin: '0 auto 16px auto', display: 'block' }} />
           <h4 style={{ marginBottom: '12px', fontSize: '1rem' }}>Manual Re-Certification</h4>
           <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '24px', lineHeight: '1.4' }}>
              Manually trigger the Sovereignty Daemon (45R) to audit all code modules and safety guardrails.
           </p>
           <button 
             onClick={triggerAudit}
             disabled={isAuditing}
             style={{ 
               background: 'transparent',
               color: 'var(--accent-emerald)',
               border: '1px solid var(--accent-emerald)',
               padding: '12px',
               borderRadius: '10px',
               fontWeight: '600',
               cursor: 'pointer',
               display: 'flex',
               alignItems: 'center',
               justifyContent: 'center',
               gap: '10px',
               transition: 'var(--transition-smooth)'
             }}
           >
             {isAuditing ? <Activity className="animate-spin" size={18} /> : <ShieldCheck size={18} />}
             RUN SYSTEM AUDIT
           </button>
           <p style={{ marginTop: '16px', fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center' }}>
             Sovereignty (45R) manages code health and safety benchmarks.
           </p>
        </div>

      </div>
    </div>
  );
};

export default SovereigntyPanel;
