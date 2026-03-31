import React, { useState } from 'react';
import SystemTelemetry from './components/SystemTelemetry';
import AgentGoalTracker from './components/AgentGoalTracker';
import ChatInterface from './components/ChatInterface';
import { Layers, Settings, HelpCircle, Activity, LayoutDashboard, Database, ShieldCheck } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="dashboard-container">
      {/* Main Header */}
      <header className="main-header glass-panel">
        <div className="brand">
          <div className="brand-logo" />
          <h1 className="brand-name">AN-RA AGI</h1>
        </div>

        <nav style={{ display: 'flex', gap: '32px' }}>
           <button onClick={() => setActiveTab('dashboard')} style={{ background: 'transparent', border: 'none', color: activeTab === 'dashboard' ? 'var(--accent-cyan)' : 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', transition: 'var(--transition-smooth)', fontSize: '0.9rem', fontWeight: 600 }}>
             <LayoutDashboard size={18} /> DASHBOARD
           </button>
           <button onClick={() => setActiveTab('memory')} style={{ background: 'transparent', border: 'none', color: activeTab === 'memory' ? 'var(--accent-cyan)' : 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', transition: 'var(--transition-smooth)', fontSize: '0.9rem', fontWeight: 600 }}>
             <Database size={18} /> MEMORY BANK
           </button>
           <button onClick={() => setActiveTab('sovereignty')} style={{ background: 'transparent', border: 'none', color: activeTab === 'sovereignty' ? 'var(--accent-cyan)' : 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', transition: 'var(--transition-smooth)', fontSize: '0.9rem', fontWeight: 600 }}>
             <ShieldCheck size={18} /> SOVEREIGNTY
           </button>
        </nav>

        <div style={{ display: 'flex', gap: '16px' }}>
           <button className="icon-btn" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', borderRadius: '8px', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', cursor: 'pointer' }}>
             <Settings size={18} />
           </button>
           <button className="icon-btn" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', borderRadius: '8px', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', cursor: 'pointer' }}>
             <HelpCircle size={18} />
           </button>
        </div>
      </header>

      {/* Side: Telemetry */}
      <SystemTelemetry />

      {/* Center: Chat Interface */}
      <ChatInterface />

      {/* Side: Goals/Actions */}
      <AgentGoalTracker />
      
      {/* Aesthetic Background Accents */}
      <div style={{ position: 'fixed', bottom: -100, right: -100, width: 400, height: 400, borderRadius: '50%', background: 'var(--accent-purple-glow)', filter: 'blur(100px)', zIndex: -1 }} />
      <div style={{ position: 'fixed', top: -100, left: -100, width: 400, height: 400, borderRadius: '50%', background: 'var(--accent-cyan-glow)', filter: 'blur(100px)', zIndex: -1 }} />
    </div>
  );
}

export default App;
