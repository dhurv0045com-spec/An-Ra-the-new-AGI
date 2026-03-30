import React, { useState, useEffect } from 'react';
import { getActiveGoals, spawnGoal } from '../api';
import { Target, PlusCircle, ServerCog, RefreshCw } from 'lucide-react';

export const GoalTracker = () => {
  const [goals, setGoals] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState('medium');
  const [spawning, setSpawning] = useState(false);

  const fetchGoals = async () => {
    try {
      const data = await getActiveGoals();
      setGoals(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGoals();
    const interval = setInterval(fetchGoals, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSpawn = async (e) => {
    e.preventDefault();
    if (!title || !description) return;
    
    setSpawning(true);
    try {
      await spawnGoal(title, description, priority);
      setTitle('');
      setDescription('');
      setPriority('medium');
      await fetchGoals();
    } catch (e) {
      alert("Failed to spawn goal: " + e.message);
    } finally {
      setSpawning(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '2rem', height: '100%' }}>
      {/* Active Goals Panel */}
      <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <ServerCog size={24} style={{ color: 'var(--text-accent)' }} />
            <h3>Autonomous Matrix</h3>
          </div>
          <button onClick={fetchGoals} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
          </button>
        </div>

        <div className="goals-list" style={{ overflowY: 'auto', flex: 1 }}>
          {goals.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '2rem' }}>
              No active goals in the agent loop.
            </div>
          ) : (
            goals.map((g) => (
              <div key={g.id} className="goal-item animate-fade-in">
                <div>
                  <div className="goal-title">{g.title}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                    ID: {g.id}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span className="badge">{g.progress}% Complete</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* New Goal Panel */}
      <div className="glass-panel" style={{ width: '350px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
          <Target size={20} style={{ color: 'var(--text-accent)' }} />
          <h3>Spawn New Goal</h3>
        </div>

        <form onSubmit={handleSpawn}>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Director Title</label>
            <input 
              className="form-input" 
              placeholder="e.g. Audit entire codebase"
              value={title}
              onChange={e => setTitle(e.target.value)}
              required
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Execution Details</label>
            <textarea 
              className="form-input" 
              placeholder="Describe constraints..."
              rows={4}
              value={description}
              onChange={e => setDescription(e.target.value)}
              required
              style={{ resize: 'vertical' }}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Priority Level</label>
            <select 
              className="form-input"
              value={priority}
              onChange={e => setPriority(e.target.value)}
            >
              <option value="low">Low (Background)</option>
              <option value="medium">Medium (Standard)</option>
              <option value="high">High (Priority)</option>
              <option value="urgent">Urgent (Preempt)</option>
            </select>
          </div>

          <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={spawning}>
            {spawning ? (
              <><RefreshCw size={18} className="animate-spin" /> Spawning Agent...</>
            ) : (
              <><PlusCircle size={18} /> Inject to Pipeline</>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
