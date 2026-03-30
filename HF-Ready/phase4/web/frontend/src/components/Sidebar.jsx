import React from 'react';
import { Activity, MessageSquare, Target, Settings, Brain } from 'lucide-react';

export const Sidebar = ({ currentView, onChangeView }) => {
  return (
    <div className="sidebar">
      <div className="brand">
        <Brain size={28} className="text-accent" style={{ color: "var(--text-accent)", filter: "drop-shadow(0 0 8px rgba(0,255,255,0.6))" }} />
        <h1>AN-RA AGI</h1>
      </div>
      
      <nav>
        <div 
          className={`nav-item ${currentView === 'dashboard' ? 'active' : ''}`}
          onClick={() => onChangeView('dashboard')}
        >
          <Activity size={20} />
          <span>System Status</span>
        </div>
        
        <div 
          className={`nav-item ${currentView === 'chat' ? 'active' : ''}`}
          onClick={() => onChangeView('chat')}
        >
          <MessageSquare size={20} />
          <span>Ouroboros Chat</span>
        </div>
        
        <div 
          className={`nav-item ${currentView === 'goals' ? 'active' : ''}`}
          onClick={() => onChangeView('goals')}
        >
          <Target size={20} />
          <span>Autonomous Goals</span>
        </div>
      </nav>
      
      <div style={{ marginTop: 'auto' }}>
        <div className="nav-item">
          <Settings size={20} />
          <span>Settings</span>
        </div>
      </div>
    </div>
  );
};
