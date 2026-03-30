import React, { useEffect, useState } from 'react';
import { getSystemStatus, getBriefing } from '../api';
import { Cpu, Server, Activity, Database, AlertCircle, RefreshCw } from 'lucide-react';

export const Dashboard = () => {
  const [status, setStatus] = useState(null);
  const [briefing, setBriefing] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchStatus = async () => {
    try {
      const statData = await getSystemStatus();
      setStatus(statData);
      const briefData = await getBriefing();
      setBriefing(briefData.text || 'No briefing available.');
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Poll every 10s
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <RefreshCw size={40} className="animate-spin" style={{ color: 'var(--text-accent)' }} />
      </div>
    );
  }

  if (!status) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem' }}>
        <AlertCircle size={48} style={{ color: 'var(--text-danger)', margin: '0 auto 1rem auto' }} />
        <h2>Connection Lost</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Unable to reach An-Ra master system backend (Phase 45M).</p>
      </div>
    );
  }

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>System Status</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(16, 185, 129, 0.1)', padding: '0.5rem 1rem', borderRadius: '20px', color: 'var(--text-success)' }}>
          <div className="status-indicator"></div>
          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>ONLINE</span>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="title">Engine Mode</span>
            <Server size={18} style={{ color: 'var(--text-accent)' }} />
          </div>
          <span className="value active">Autonomous</span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Tier {status?.engine_state?.tier || 2}</span>
        </div>
        
        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="title">System Memory</span>
            <Database size={18} style={{ color: 'var(--text-accent)' }} />
          </div>
          <span className="value">{status?.knowledge?.total_entries || 0}</span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Semantic Facts Registered</span>
        </div>

        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="title">Active Goals</span>
            <Activity size={18} style={{ color: 'var(--text-accent)' }} />
          </div>
          <span className="value">{status?.goals?.total_active || 0}</span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Agent tasks in progress</span>
        </div>
      </div>

      <div className="glass-panel" style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>
          <Cpu size={24} style={{ color: 'var(--text-accent)' }} />
          <h3>Morning Briefing & Sovereignty Report</h3>
        </div>
        <div className="briefing" style={{ whiteSpace: 'pre-wrap' }}>
          {briefing}
        </div>
      </div>
    </>
  );
};
