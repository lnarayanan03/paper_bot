import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { MessageCircle, Send, X } from 'lucide-react';
import MessageBubble from './MessageBubble.jsx';

function streamWords(text, onUpdate, onDone) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  let index = 0;
  const timer = window.setInterval(() => {
    index += 1;
    onUpdate(words.slice(0, index).join(' '));
    if (index >= words.length) {
      window.clearInterval(timer);
      onDone?.();
    }
  }, 28);

  if (!words.length) {
    window.clearInterval(timer);
    onUpdate('');
    onDone?.();
  }

  return () => window.clearInterval(timer);
}

export default function ChatWindow({ sessionId, launcherRef, onOpenImage }) {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'bot',
      type: 'text',
      text: 'Ask me about BERT, RoBERTa, sentiment analysis, or a figure from the indexed papers.',
      sources: [],
      hasSuggestionImage: false,
      hasSuggestionTable: false,
    },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesRef = useRef(null);

  useEffect(() => {
    messagesRef.current?.scrollTo({
      top: messagesRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isTyping]);

  const addMessage = (message) => {
    setMessages((current) => [...current, { id: crypto.randomUUID(), ...message }]);
  };

  const replaceMessage = (id, patch) => {
    setMessages((current) => current.map((message) => (message.id === id ? { ...message, ...patch } : message)));
  };

  const handleSend = async (text) => {
    const question = text.trim();
    if (!question || isTyping || !sessionId) return;

    addMessage({ role: 'user', type: 'text', text: question });
    setIsTyping(true);

    try {
      const response = await axios.post('/api/ask', { question, session_id: sessionId });
      const data = response.data || {};
      const botId = crypto.randomUUID();
      const seen = new Set();
      const uniqueSources = (data.sources || []).filter(s => {
        const key = `${s.source}:${s.page}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      setMessages((current) => [
        ...current,
        {
          id: botId,
          role: 'bot',
          type: 'text',
          text: '',
          sources: uniqueSources,
          hasSuggestionImage: !!data.has_relevant_image,
          hasSuggestionTable: !!data.has_relevant_table,
        },
      ]);
      const answerText = data.answer || 'I could not find an answer for that yet.';
      const isTable = (
        answerText.includes('| --- |') ||
        answerText.includes('|---|') ||
        answerText.includes('|-----|') ||
        answerText.includes('|----') ||
        /\|[-\s|]+\|/.test(answerText)
      );
      if (isTable) {
        // Table content — render immediately, no streaming
        replaceMessage(botId, { text: answerText });
      } else {
        streamWords(answerText, (text) => {
          replaceMessage(botId, { text });
        });
      }

      if (data.image_base64) {
        addMessage({
          role: 'bot',
          type: 'image',
          imageBase64: data.image_base64,
          caption: {
            source: data.image_source,
            page: data.image_page,
            text_preview: '',
          },
        });
      }
    } catch {
      addMessage({
        role: 'bot',
        type: 'text',
        text: 'PaperBot could not reach the API. Check that the backend is running on port 8000.',
        sources: [],
        hasSuggestionImage: false,
        hasSuggestionTable: false,
      });
    } finally {
      setIsTyping(false);
    }
  };

  const sendSuggestion = (suggestionText) => {
    if (isTyping || !sessionId) return;
    setInput('');
    handleSend(suggestionText);
  };

  const sendQuestion = async (event) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || isTyping || !sessionId) return;
    setInput('');
    await handleSend(question);
  };

  return (
    <>
      {isOpen ? (
        <section className="chat-panel" aria-label="PaperBot chat">
          <header className="chat-header">
            <div className="chat-title-row">
              <div className="chat-title">🔬 PaperBot</div>
              <span className="online-indicator">● online</span>
            </div>
            <button type="button" className="icon-btn" onClick={() => setIsOpen(false)} aria-label="Close chat">
              <X size={16} />
            </button>
          </header>

          <div className="messages" ref={messagesRef}>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onOpenImage={onOpenImage}
                onSuggestion={sendSuggestion}
              />
            ))}

            {isTyping ? <MessageBubble message={{ type: 'typing' }} /> : null}
          </div>

          <form className="chat-form" onSubmit={sendQuestion}>
            <input
              className="chat-input"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask a paper question..."
              aria-label="Ask PaperBot"
            />
            <button type="submit" className="send-btn" aria-label="Send question" disabled={!input.trim() || isTyping}>
              <Send size={16} />
            </button>
          </form>
        </section>
      ) : null}

      <button
        type="button"
        ref={launcherRef}
        className="chat-launcher"
        onClick={() => setIsOpen((value) => !value)}
        aria-label="Open PaperBot chat"
      >
        <MessageCircle size={24} />
      </button>
    </>
  );
}
