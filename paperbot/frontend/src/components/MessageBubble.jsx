import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SourcePanel from './SourcePanel.jsx';

export default function MessageBubble({ message, onOpenImage, onSuggestion }) {
  if (message.type === 'typing') {
    return (
      <div className="message-row bot">
        <div className="bubble">
          <div className="typing" aria-label="PaperBot is typing">
            <span />
            <span />
            <span />
          </div>
        </div>
      </div>
    );
  }
  if (message.type === 'image') {
    return (
      <div className="message-row bot">
        <div className="bubble">
          <img
            className="inline-image"
            src={message.imageBase64}
            alt={message.alt || 'Retrieved paper page'}
            onClick={() => onOpenImage?.(message.imageBase64)}
          />
          {message.caption ? <SourcePanel sources={[message.caption]} /> : null}
        </div>
      </div>
    );
  }

  const hasSuggestions = message.role === 'bot' && (message.hasSuggestionImage || message.hasSuggestionTable);

  return (
    <>
      <div className={`message-row ${message.role === 'user' ? 'user' : 'bot'}`}>
        <div className="bubble">
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
          </div>
          {message.role === 'bot' ? <SourcePanel sources={message.sources} /> : null}
        </div>
      </div>

      {hasSuggestions && (
        <div className="message-row bot suggestion-message">
          <div className="bubble suggestion-bubble">
            <span className="suggestion-label">
              💡 Related content available:
            </span>
          <div className="suggestion-row">
            {message.hasSuggestionImage && (
              <button
                type="button"
                className="suggestion-btn"
                onClick={() => onSuggestion?.('show me the figure for it')}
              >
                🖼️ Show figure
              </button>
            )}
            {message.hasSuggestionTable && (
              <button
                type="button"
                className="suggestion-btn"
                onClick={() => onSuggestion?.('give me the table for it')}
              >
                📊 Show table
              </button>
            )}
          </div>
          </div>
        </div>
      )}
    </>
  );
}
