import { FileText } from 'lucide-react';

export default function SourcePanel({ sources = [] }) {
  const visibleSources = sources.filter(Boolean);
  if (!visibleSources.length) return null;

  return (
    <div className="source-panel" aria-label="Sources">
      {visibleSources.map((source, index) => (
        <span className="meta-chip" key={`${source.source}-${source.page}-${index}`} title={source.text_preview || ''}>
          <FileText size={12} />
          {source.source || 'source'} · p.{source.page || '?'}
        </span>
      ))}
    </div>
  );
}
