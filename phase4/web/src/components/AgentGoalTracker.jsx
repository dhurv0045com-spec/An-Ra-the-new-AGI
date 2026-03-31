import React, { useState, useEffect } from 'react';
import { Target, CheckCircle2, Circle, Play, Plus, Loader2 } from 'lucide-react';

const AgentGoalTracker = () => {
  const [goals, setGoals] = useState([
    { id: 'initial', title: 'Initialize Neural Bridge', progress: 100, status: 'completed' }
  ]);
  const [newGoal, setNewGoal] = useState({ title: '', description: '' });
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    const fetchGoals = async () => {
      try {
        const response = await fetch('/api/goals');
        if (response.ok) {
          const data = await response.json();
          setGoals(data);
        }
      } catch (e) {
        console.error("Goal fetch error:", e);
      }
    };
    fetchGoals();
    const interval = setInterval(fetchGoals, 5000);
    return () => clearInterval(interval);
  }, []);

  const triggerGoal = async (e) => {
    e.preventDefault();
    if (!newGoal.title) return;
    setIsRunning(true);
    try {
      const response = await fetch('/api/goal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newGoal),
      });
      if (response.ok) {
        setNewGoal({ title: '', description: '' });
      }
    } catch (e) {
      console.error("Goal trigger error:", e);
    }
    setIsRunning(false);
  };

  return (
    <div className="goal-tracker glass-panel animate-in" style={{ animationDelay: '0.2s' }}>
      <div className="panel-header" style={{ padding: '20px', borderBottom: '1px solid var(--panel-border)', background: 'rgba(255,255,255,0.02)' }}>
        <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Target size={16} /> AGENT GOAL TRACKER
        </h2>
      </div>

      <div className="goal-content" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
        
        {/* Active Goals List */}
        <div className="goal-list" style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {goals.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              No active goals in queue.
            </div>
          ) : (
            goals.map(goal => (
              <div key={goal.id} style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', border: '1px solid var(--panel-border)', position: 'relative' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                  {goal.progress >= 100 ? <CheckCircle2 size={16} color="var(--accent-emerald)" /> : <Loader2 size={16} className="animate-spin" color="var(--accent-cyan)" />}
                  <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>{goal.title}</span>
                </div>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${goal.progress}%`, background: goal.progress >= 100 ? 'var(--accent-emerald)' : 'var(--accent-cyan)', transition: 'width 1s ease' }} />
                </div>
              </div>
            ))
          )}
        </div>

        {/* New Goal Input */}
        <form onSubmit={triggerGoal} style={{ marginTop: 'auto', borderTop: '1px solid var(--panel-border)', paddingTop: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
             <input 
               type="text" 
               placeholder="Enter new objective..." 
               value={newGoal.title}
               onChange={(e) => setNewGoal({ ...newGoal, title: e.target.value })}
               style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', borderRadius: '8px', padding: '10px 16px', flex: 1, color: 'white', outline: 'none' }}
             />
             <button 
               type="submit" 
               disabled={isRunning}
               style={{ background: 'var(--accent-cyan)', border: 'none', borderRadius: '8px', width: '42px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'var(--transition-smooth)' }}
             >
               {isRunning ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} fill="black" stroke="black" />}
             </button>
          </div>
          <p style={{ fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center' }}>
            AGENT LOOP: PHASE 2 (45k) ACTIVE
          </p>
        </form>

      </div>
    </div>
  );
};

export default AgentGoalTracker;
