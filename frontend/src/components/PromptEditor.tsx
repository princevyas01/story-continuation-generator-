import React from 'react';
import { PenTool, RotateCcw } from 'lucide-react';

interface PromptEditorProps {
  prompt: string;
  setPrompt: (value: string) => void;
  disabled: boolean;
}

const PRESETS = [
  'Once upon a time in a deep dark forest and',
  'The little wooden puppet wished with all his heart that',
  'In a kingdom far beyond the northern mountains there lived',
  'The brave young tailor took his needle and thread and',
  'A clever golden bird landed on the royal windowsill and whispered',
];

export const PromptEditor: React.FC<PromptEditorProps> = ({
  prompt,
  setPrompt,
  disabled,
}) => {
  const charCount = prompt.length;
  const wordCount = prompt.trim() ? prompt.trim().split(/\s+/).length : 0;

  return (
    <div className="card">
      <div className="card-title">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <PenTool size={18} color="#38bdf8" />
          Story Seed Prompt
        </span>
        <button
          onClick={() => setPrompt('')}
          disabled={disabled || !prompt}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#94a3b8',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            fontSize: '0.8rem',
          }}
          title="Clear Prompt"
        >
          <RotateCcw size={14} /> Clear
        </button>
      </div>

      <div className="textarea-container">
        <textarea
          className="story-textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter the beginning of your fairy tale or folklore story here..."
          disabled={disabled}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', fontSize: '0.75rem', color: '#64748b' }}>
          {wordCount} words • {charCount} chars
        </div>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.4rem' }}>
          Quick Starters:
        </div>
        <div className="presets-container">
          {PRESETS.map((preset, idx) => (
            <button
              key={idx}
              className="preset-chip"
              disabled={disabled}
              onClick={() => setPrompt(preset)}
            >
              {preset.slice(0, 32)}...
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
