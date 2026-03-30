import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { ChatConsole } from './components/ChatConsole';
import { GoalTracker } from './components/GoalTracker';
import './index.css';

function App() {
  const [currentView, setCurrentView] = useState('dashboard');

  return (
    <div className="app-container">
      <Sidebar 
        currentView={currentView} 
        onChangeView={(view) => setCurrentView(view)} 
      />
      <main className="main-content animate-fade-in">
        {currentView === 'dashboard' && <Dashboard />}
        {currentView === 'chat' && <ChatConsole />}
        {currentView === 'goals' && <GoalTracker />}
      </main>
    </div>
  );
}

export default App;
