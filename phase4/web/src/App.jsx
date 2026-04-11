import React, { useState } from 'react';
import SystemTelemetry from './components/SystemTelemetry';
import AgentGoalTracker from './components/AgentGoalTracker';
import ChatInterface from './components/ChatInterface';
import TrainingPanel from './components/TrainingPanel';
import MemoryExplorer from './components/MemoryExplorer';
import SovereigntyPanel from './components/SovereigntyPanel';
import { Layers, Settings, HelpCircle, Activity, LayoutDashboard, Database, ShieldCheck, Brain } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <>
            <SystemTelemetry />
            <ChatInterface />
            <AgentGoalTracker />
          </>
        );
      case 'training':
        return <TrainingPanel />;
      case 'memory':
        return <MemoryExplorer />;
      case 'sovereignty':
        return <SovereigntyPanel />;
      default:
        return <div style={{ padding: '40px', textAlign: 'center' }}>Module under construction.</div>;
    }
  };

  return (
    <div className="dashboard-container" style={{ gridTemplateColumns: activeTab === 'dashboard' ? '350px 1fr 350px' : '1fr' }}>
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
           <button onClick={() => setActiveTab('training')} style={{ background: 'transparent', border: 'none', color: activeTab === 'training' ? 'var(--accent-cyan)' : 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', transition: 'var(--transition-smooth)', fontSize: '0.9rem', fontWeight: 600 }}>
             <Brain size={18} /> NEURAL TRAINING
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

      {/* Main Content Area */}
      {renderContent()}
      
      {/* Aesthetic Background Accents */}
      <div style={{ position: 'fixed', bottom: -100, right: -100, width: 400, height: 400, borderRadius: '50%', background: 'var(--accent-purple-glow)', filter: 'blur(100px)', zIndex: -1 }} />
      <div style={{ position: 'fixed', top: -100, left: -100, width: 400, height: 400, borderRadius: '50%', background: 'var(--accent-cyan-glow)', filter: 'blur(100px)', zIndex: -1 }} />
    </div>
  );
}

export default App;
