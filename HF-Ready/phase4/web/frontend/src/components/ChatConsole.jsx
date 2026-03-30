import React, { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../api';
import { Send, Terminal } from 'lucide-react';

export const ChatConsole = () => {
  const [messages, setMessages] = useState([
    { role: 'an-ra', text: `An-Ra Phase 3 Online.\nIdentity: Active.\nOuroboros: Active.\nGhost Memory: Initialized.\n\nHow can I sequence your thoughts today?` }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  const scrollToBottom = () => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendChatMessage(userMsg);
      setMessages(prev => [...prev, { role: 'an-ra', text: res.response }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'an-ra', text: `[SYSTEM ERROR] Neural link failed: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Terminal size={20} className="text-accent" />
        <h3 style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>Ouroboros Chat Uplink</h3>
      </div>

      <div className="chat-history">
        {messages.map((msg, i) => (
          <div key={i} className={`message-wrapper ${msg.role === 'user' ? 'user' : 'an-ra'}`}>
            <div className="message-bubble" style={{ whiteSpace: 'pre-wrap' }}>
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message-wrapper an-ra">
            <div className="message-bubble" style={{ 
              fontStyle: 'italic', 
              color: 'var(--text-accent)',
              animation: 'pulseGlow 2s infinite'
            }}>
              Analyzing sequence...
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form onSubmit={handleSend} style={{ paddingTop: '1rem' }}>
        <div className="chat-input-wrapper">
          <input 
            type="text" 
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Initialize cognitive sequence..."
            disabled={loading}
          />
          <button type="submit" className="btn-icon" disabled={!input.trim() || loading}>
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
};
