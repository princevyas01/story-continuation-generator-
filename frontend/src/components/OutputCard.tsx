import React, { useState } from 'react';
import { Copy, Check, Download, ArrowRight, Clock, Cpu } from 'lucide-react';
import { GenerateResponse } from '../types';

interface OutputCardProps {
  result: GenerateResponse | null;
  onContinueFromOutput: (text: string) => void;
  isLoading: boolean;
}

export const OutputCard: React.FC<OutputCardProps> = ({
  result,
  onContinueFromOutput,
  isLoading,
}) => {
  const [copied, setCopied] = useState(false);

  if (isLoading) {
    return (
      <div className="card output-card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
        <div style={{ color: '#38bdf8', fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          Predicting Next Tokens with 2-Layer LSTM...
        </div>
        <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
          Sampling sequentially through softmax probability distribution...
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card output-card" style={{ textAlign: 'center', padding: '2.5rem 1.5rem', color: '#64748b' }}>
        Enter a story prompt above and click <strong>Generate Continuation</strong> to run local LSTM inference.
      </div>
    );
  }

  const handleCopy = () => {
    const fullText = `${result.prompt} ${result.generated_text}`.trim();
    navigator.clipboard.writeText(fullText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const fullText = `${result.prompt} ${result.generated_text}`.trim();
    const element = document.createElement('a');
    const file = new Blob([fullText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = 'generated_story.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const tokensGenerated = result.generated_text.trim()
    ? result.generated_text.trim().split(/\s+/).length
    : 0;
  const msPerToken = tokensGenerated > 0 ? (result.runtime_ms / tokensGenerated).toFixed(1) : '0';

  return (
    <div className="card output-card">
      <div className="card-title">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          Generated Story Continuation
        </span>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="preset-chip"
            onClick={handleCopy}
            title="Copy full story to clipboard"
          >
            {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            className="preset-chip"
            onClick={handleDownload}
            title="Download story as text file"
          >
            <Download size={14} /> Download
          </button>
          <button
            className="preset-chip"
            style={{ color: '#38bdf8', borderColor: '#38bdf8' }}
            onClick={() => onContinueFromOutput(`${result.prompt} ${result.generated_text}`.trim())}
            title="Append output into prompt and keep continuing"
          >
            <ArrowRight size={14} /> Continue Story
          </button>
        </div>
      </div>

      <div className="story-display">
        <span className="prompt-part">{result.prompt}</span>{' '}
        <span className="continuation-part">{result.generated_text}</span>
      </div>

      {/* Inference Diagnostic Footer */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1.25rem',
          marginTop: '1rem',
          fontSize: '0.8rem',
          color: '#94a3b8',
          borderTop: '1px solid #2e3c61',
          paddingTop: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <Clock size={14} color="#38bdf8" />
          <span>Runtime: <strong>{result.runtime_ms} ms</strong></span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <Cpu size={14} color="#818cf8" />
          <span>Tokens: <strong>{tokensGenerated}</strong> ({msPerToken} ms/tok)</span>
        </div>
        <div>
          <span>Temp: <strong>{result.settings.temperature}</strong> | Top-K: <strong>{result.settings.top_k}</strong></span>
        </div>
        <div style={{ marginLeft: 'auto', color: '#64748b' }}>
          Model: {result.model_version}
        </div>
      </div>
    </div>
  );
};
