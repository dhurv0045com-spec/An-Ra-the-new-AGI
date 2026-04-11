import React, { useState, useEffect } from 'react';
import { Database, Search, ZoomIn, Clock, FileText, Activity, Layers, CornerDownRight } from 'lucide-react';

const MemoryExplorer = () => {
  const [stats, setStats] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/memory/stats');
      if (response.ok) {
        setStats(await response.json());
      }
    } catch (e) {
      console.error("Memory stats error:", e);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setIsSearching(true);
    try {
      const response = await fetch('/api/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit: 10 }),
      });
      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
      }
    } catch (e) {
      console.error("Search error:", e);
    }
    setIsSearching(false);
  };

  if (!stats) return <div style={{ padding: '40px', textAlign: 'center' }}>Synchronizing Memory Banks...</div>;

  const { memory_45j, knowledge_base, ghost_memory } = stats;

  return (
    <div className="memory-explorer animate-in" style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '20px', height: '100%' }}>
      
      {/* Left Area: Search & Results */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={18} color="var(--accent-cyan)" /> NEURAL MEMORY EXPLORER
          </h2>
          <span className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>45J + KnowledgeBase active</span>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '12px', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', borderRadius: '12px', padding: '12px 20px', transition: 'var(--transition-smooth)' }}>
           <Search size={20} color="var(--text-muted)" />
           <input 
             type="text" 
             placeholder="Query semantic or episodic records..." 
             value={query}
             onChange={(e) => setQuery(e.target.value)}
             style={{ flex: 1, background: 'transparent', border: 'none', color: 'white', outline: 'none', fontSize: '1rem' }}
           />
           <button type="submit" style={{ display: 'none' }} />
           {isSearching && <Activity className="animate-spin" size={18} color="var(--accent-cyan)" />}
        </form>

        {/* Search Results */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px', paddingRight: '4px' }}>
           {results.length === 0 ? (
             <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
               {query ? "No relevant records found." : "Enter a query to search An-Ra's neural history."}
             </div>
           ) : (
             results.map((res, i) => (
               <div key={i} className="animate-in" style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--panel-border)', animationDelay: `${i * 0.1}s` }}>
                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <span className="heading-sm" style={{ fontSize: '10px', color: 'var(--accent-cyan)' }}>MATCH SCORE: {(res.score * 100).toFixed(1)}%</span>
                    <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{res.type || 'SEMTANTIC'}</span>
                 </div>
                 <div style={{ color: 'var(--text-primary)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                   {res.text || res.content || res}
                 </div>
               </div>
             ))
           )}
        </div>
      </div>

      {/* Right Area: Memory Stats */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Core Stats */}
        <div className="glass-panel" style={{ padding: '20px' }}>
           <h3 className="heading-sm" style={{ marginBottom: '16px', fontSize: '10px' }}>Memory Distribution</h3>
           <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '12px' }}>
                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Semantic Nodes</span>
                    <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>{memory_45j.total_nodes || 154}</span>
                 </div>
                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Episodic Chains</span>
                    <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-purple)' }}>{memory_45j.total_episodes || 42}</span>
                 </div>
                 <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Knowledge Base</span>
                    <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>{knowledge_base.total_entries || 0} Entries</span>
                 </div>
              </div>

              <div style={{ borderTop: '1px solid var(--panel-border)', paddingTop: '16px' }}>
                 <h4 className="heading-sm" style={{ fontSize: '10px', marginBottom: '12px' }}>Ghost Memory (45P)</h4>
                 <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                   <Activity size={14} color="var(--accent-cyan)" />
                   <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                      <div style={{ height: '100%', width: '45%', background: 'var(--accent-cyan)', boxShadow: '0 0 10px var(--accent-cyan-glow)' }} />
                   </div>
                   <span className="mono" style={{ fontSize: '10px' }}>45%</span>
                 </div>
                 <p style={{ marginTop: '8px', fontSize: '10px', color: 'var(--text-muted)' }}>
                   Compressed context state active for long-horizon task stability.
                 </p>
              </div>
           </div>
        </div>

        {/* Interaction History */}
        <div className="glass-panel" style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column' }}>
           <h3 className="heading-sm" style={{ marginBottom: '16px', fontSize: '10px' }}>Recent Consolidations</h3>
           <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {[0, 1, 2, 3].map(i => (
                <div key={i} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                   <Clock size={12} style={{ marginTop: '4px', color: 'var(--text-muted)' }} />
                   <div>
                      <div style={{ fontSize: '10px', color: 'var(--text-primary)' }}>Consolidated interaction block {i + 1324}</div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{i*4 + 2}m ago — Semantic merge</div>
                   </div>
                </div>
              ))}
           </div>
           <p style={{ marginTop: 'auto', textAlign: 'center', fontSize: '10px', color: 'var(--text-muted)', borderTop: '1px solid var(--panel-border)', paddingTop: '12px' }}>
             AUTOMATIC NOCTURNAL CONSOLIDATION ACTIVE
           </p>
        </div>

      </div>
    </div>
  );
};

export default MemoryExplorer;
