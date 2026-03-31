import React, { useState, useEffect, useRef } from 'react';
import { Send, Terminal, User, Bot, Loader2, Minimize2, Maximize2 } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', text: "Neural Bridge established. Phase 3: Cognition online. I am An-Ra. How can I assist you today?", timestamp: new Date().toLocaleTimeString() }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, isTyping]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;

    const userMessage = { id: Date.now(), role: 'user', text: input, timestamp: new Date().toLocaleTimeString() };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', text: data.response, timestamp: new Date().toLocaleTimeString() }]);
      } else {
        setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', text: "Error connecting to MasterSystem backbone.", timestamp: new Date().toLocaleTimeString() }]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', text: "Connection refused. Ensure app.py is running.", timestamp: new Date().toLocaleTimeString() }]);
    }
    setIsTyping(false);
  };

  return (
    <div className="chat-panel glass-panel animate-in" style={{ animationDelay: '0.1s' }}>
      <div className="panel-header" style={{ padding: '20px', borderBottom: '1px solid var(--panel-border)', background: 'hsla(180, 100%, 50%, 0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="heading-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
          <Terminal size={16} /> NEURAL INTERFACE
        </h2>
        <div style={{ display: 'flex', gap: '8px' }}>
           <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-emerald)', boxShadow: '0 0 10px var(--accent-emerald)' }} />
           <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>ONLINE</span>
        </div>
      </div>

      <div className="chat-history" style={{ flex: 1, padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {messages.map((msg) => (
          <div key={msg.id} style={{ display: 'flex', gap: '16px', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: msg.role === 'user' ? 'var(--panel-border)' : 'var(--accent-cyan-glow)', border: '1px solid var(--panel-border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
               {msg.role === 'user' ? <User size={18} color="white" /> : <Bot size={18} color="var(--accent-cyan)" />}
            </div>
            <div style={{ maxWidth: '80%', display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start', gap: '6px' }}>
               <div style={{ padding: '12px 18px', borderRadius: '14px', background: msg.role === 'user' ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.05)', border: msg.role === 'user' ? 'none' : '1px solid var(--panel-border)', color: msg.role === 'user' ? 'black' : 'white', fontSize: '0.95rem', lineHeight: '1.5' }}>
                  {msg.text}
               </div>
               <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{msg.timestamp}</span>
            </div>
          </div>
        ))}
        {isTyping && (
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'var(--accent-cyan-glow)', border: '1px solid var(--panel-border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
               <Bot size={18} color="var(--accent-cyan)" />
            </div>
            <div style={{ padding: '12px 18px', borderRadius: '14px', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)' }}>
               <Loader2 size={16} className="animate-spin" color="var(--text-muted)" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSend} style={{ padding: '24px', borderTop: '1px solid var(--panel-border)', background: 'rgba(0,0,0,0.2)' }}>
        <div style={{ display: 'flex', gap: '12px', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', borderRadius: '12px', padding: '6px 12px', transition: 'var(--transition-smooth)' }}>
           <input 
             type="text" 
             placeholder="Type a message to An-Ra..." 
             value={input}
             onChange={(e) => setInput(e.target.value)}
             style={{ flex: 1, background: 'transparent', border: 'none', color: 'white', padding: '10px', outline: 'none', fontSize: '0.95rem' }}
           />
           <button 
             type="submit" 
             disabled={!input.trim() || isTyping}
             style={{ background: 'transparent', border: 'none', color: input.trim() ? 'var(--accent-cyan)' : 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'var(--transition-smooth)' }}
           >
             <Send size={20} fill={input.trim() ? 'var(--accent-cyan)' : 'none'} />
           </button>
        </div>
        <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between' }}>
           <div style={{ display: 'flex', gap: '12px' }}>
              <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Top-k: 50</span>
              <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Temp: 0.8</span>
           </div>
           <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>PHASE 3 COGNITION</span>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
