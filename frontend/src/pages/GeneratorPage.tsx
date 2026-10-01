import React, { useState, useRef } from 'react';
import { Sparkles, AlertCircle, XCircle } from 'lucide-react';
import { PromptEditor } from '../components/PromptEditor';
import { GenerationControls } from '../components/GenerationControls';
import { OutputCard } from '../components/OutputCard';
import { GenerationSettings, GenerateResponse } from '../types';
import { api, ApiError } from '../api/client';

export const GeneratorPage: React.FC = () => {
  const [prompt, setPrompt] = useState('Once upon a time in a deep dark forest and');
  const [settings, setSettings] = useState<GenerationSettings>({
    max_new_tokens: 50,
    temperature: 0.8,
    top_k: 40,
    top_p: 0.90,
    seed: undefined,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please provide a prompt before generating.');
      return;
    }

    setError(null);
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await api.generateStory(
        {
          prompt: prompt.trim(),
          max_new_tokens: settings.max_new_tokens,
          temperature: settings.temperature,
          top_k: settings.top_k,
          top_p: settings.top_p,
          seed: settings.seed,
        },
        controller.signal
      );
      setResult(response);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Generation request cancelled.');
      } else if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err.message || 'An unexpected error occurred during generation.');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  return (
    <div>
      {/* Error Banner */}
      {error && (
        <div
          style={{
            background: 'rgba(244, 63, 94, 0.15)',
            border: '1px solid rgba(244, 63, 94, 0.4)',
            color: '#fb7185',
            padding: '0.85rem 1.25rem',
            borderRadius: '0.75rem',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
          <button
            onClick={() => setError(null)}
            style={{ background: 'transparent', border: 'none', color: '#fb7185', cursor: 'pointer' }}
          >
            <XCircle size={16} />
          </button>
        </div>
      )}

      {/* Main Grid */}
      <div className="generator-grid">
        <div>
          <PromptEditor
            prompt={prompt}
            setPrompt={setPrompt}
            disabled={isLoading}
          />
          <div style={{ marginTop: '1.25rem' }}>
            {isLoading ? (
              <button
                className="btn-generate"
                style={{ background: '#475569' }}
                onClick={handleCancel}
              >
                <XCircle size={18} /> Cancel Generation
              </button>
            ) : (
              <button
                className="btn-generate"
                onClick={handleGenerate}
                disabled={!prompt.trim()}
              >
                <Sparkles size={18} /> Generate Continuation
              </button>
            )}
          </div>
        </div>

        <div>
          <GenerationControls
            settings={settings}
            setSettings={setSettings}
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Output Card */}
      <OutputCard
        result={result}
        onContinueFromOutput={(fullStory) => {
          setPrompt(fullStory + ' ');
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
        isLoading={isLoading}
      />
    </div>
  );
};
