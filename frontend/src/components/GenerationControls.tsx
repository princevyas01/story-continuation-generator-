import React from 'react';
import { Sliders, Zap } from 'lucide-react';
import { GenerationSettings } from '../types';

interface GenerationControlsProps {
  settings: GenerationSettings;
  setSettings: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  disabled: boolean;
}

export const GenerationControls: React.FC<GenerationControlsProps> = ({
  settings,
  setSettings,
  disabled,
}) => {
  const getTempHint = (t: number) => {
    if (t < 0.4) return 'Deterministic & repetitive (low entropy)';
    if (t < 0.7) return 'Focused & conservative';
    if (t < 1.0) return 'Balanced story flow (recommended)';
    if (t < 1.4) return 'Creative & adventurous';
    return 'High entropy / chaotic hallucinations';
  };

  const applyPreset = (presetName: 'conservative' | 'balanced' | 'creative') => {
    if (presetName === 'conservative') {
      setSettings((s) => ({ ...s, temperature: 0.4, top_k: 25, top_p: 0.85 }));
    } else if (presetName === 'balanced') {
      setSettings((s) => ({ ...s, temperature: 0.8, top_k: 40, top_p: 0.90 }));
    } else {
      setSettings((s) => ({ ...s, temperature: 1.2, top_k: 60, top_p: 0.95 }));
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sliders size={18} color="#818cf8" />
          Sampling Hyperparameters
        </span>
        <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Autoregressive LSTM</span>
      </div>

      {/* Preset Buttons */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem' }}>
        <button
          className="preset-chip"
          style={{ flex: 1, textAlign: 'center' }}
          disabled={disabled}
          onClick={() => applyPreset('conservative')}
        >
          Focused
        </button>
        <button
          className="preset-chip"
          style={{ flex: 1, textAlign: 'center' }}
          disabled={disabled}
          onClick={() => applyPreset('balanced')}
        >
          Balanced
        </button>
        <button
          className="preset-chip"
          style={{ flex: 1, textAlign: 'center' }}
          disabled={disabled}
          onClick={() => applyPreset('creative')}
        >
          Creative
        </button>
      </div>

      {/* Max Tokens Slider */}
      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Max New Tokens</span>
          <span className="control-value">{settings.max_new_tokens}</span>
        </div>
        <input
          type="range"
          min="10"
          max="150"
          step="5"
          className="range-slider"
          value={settings.max_new_tokens}
          disabled={disabled}
          onChange={(e) =>
            setSettings((s) => ({ ...s, max_new_tokens: Number(e.target.value) }))
          }
        />
        <div className="control-hint">Words to generate following the prompt</div>
      </div>

      {/* Temperature Slider */}
      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Temperature (\(T\))</span>
          <span className="control-value">{settings.temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.05"
          className="range-slider"
          value={settings.temperature}
          disabled={disabled}
          onChange={(e) =>
            setSettings((s) => ({ ...s, temperature: parseFloat(e.target.value) }))
          }
        />
        <div className="control-hint">{getTempHint(settings.temperature)}</div>
      </div>

      {/* Top-K Slider */}
      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Top-K Truncation</span>
          <span className="control-value">{settings.top_k}</span>
        </div>
        <input
          type="range"
          min="1"
          max="100"
          step="1"
          className="range-slider"
          value={settings.top_k}
          disabled={disabled}
          onChange={(e) =>
            setSettings((s) => ({ ...s, top_k: parseInt(e.target.value, 10) }))
          }
        />
        <div className="control-hint">Pool of highest probability candidate words</div>
      </div>

      {/* Top-P Slider */}
      <div className="control-group">
        <div className="control-header">
          <span className="control-label">Top-P (Nucleus)</span>
          <span className="control-value">{settings.top_p?.toFixed(2) || '1.00'}</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="1.0"
          step="0.05"
          className="range-slider"
          value={settings.top_p || 0.9}
          disabled={disabled}
          onChange={(e) =>
            setSettings((s) => ({ ...s, top_p: parseFloat(e.target.value) }))
          }
        />
        <div className="control-hint">Cumulative mass cutoff for vocabulary selection</div>
      </div>

      {/* Seed Input */}
      <div className="control-group" style={{ marginBottom: 0 }}>
        <div className="control-header">
          <span className="control-label">Random Seed (Optional)</span>
          <span className="control-value">{settings.seed !== undefined ? settings.seed : 'Random'}</span>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="number"
            placeholder="Random"
            style={{
              flex: 1,
              background: '#0a0f1d',
              border: '1px solid #2e3c61',
              borderRadius: '0.5rem',
              padding: '0.4rem 0.75rem',
              color: '#fff',
              fontSize: '0.85rem',
            }}
            value={settings.seed ?? ''}
            disabled={disabled}
            onChange={(e) => {
              const val = e.target.value;
              setSettings((s) => ({ ...s, seed: val === '' ? undefined : parseInt(val, 10) }));
            }}
          />
          <button
            className="preset-chip"
            disabled={disabled}
            onClick={() => setSettings((s) => ({ ...s, seed: Math.floor(Math.random() * 10000) }))}
            title="Pick a random seed"
          >
            <Zap size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};
