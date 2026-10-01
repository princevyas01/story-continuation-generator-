import React from 'react';
import { Cpu, Layers, GitBranch, Terminal } from 'lucide-react';

export const HowItWorksPage: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Intro Card */}
      <div className="card">
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#38bdf8', marginBottom: '0.75rem' }}>
          How the Story Continuation Generator Works
        </h2>
        <p style={{ color: '#cbd5e1', lineHeight: 1.7 }}>
          This system is a real, locally trained <strong>Recurrent Neural Network</strong> with an
          <strong>LSTM (Long Short-Term Memory)</strong> architecture. Unlike cloud-hosted LLMs that send your text across the internet,
          every token in this application is predicted locally on your laptop's CPU using trained TensorFlow/Keras weights.
        </p>
      </div>

      {/* Pipeline Diagram */}
      <div className="card">
        <div className="card-title">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <GitBranch size={18} color="#818cf8" />
            End-to-End Inference Pipeline
          </span>
        </div>
        <div style={{ background: '#0a0f1d', border: '1px solid #2e3c61', borderRadius: '0.75rem', padding: '1.5rem', fontFamily: 'monospace', fontSize: '0.9rem', color: '#93c5fd', overflowX: 'auto', lineHeight: 1.8 }}>
          <div>[User Prompt Text]</div>
          <div>       │</div>
          <div>       ▼  (Regex Tokenizer & Normalization)</div>
          <div>[Word Token Sequence: &lt;w_1, w_2, ... w_t&gt;]</div>
          <div>       │</div>
          <div>       ▼  (Vocabulary Lookup: word2idx)</div>
          <div>[Integer Indices Tensor (1, 50)]</div>
          <div>       │</div>
          <div>       ▼</div>
          <div>┌───────────────────────────────────────────────────────────┐</div>
          <div>│ 1. Word Embedding Layer (Vocab: 5004, Dim: 128)           │</div>
          <div>│ 2. LSTM Layer 1 (Units: 256, Dropout: 0.20, Seq: True)     │</div>
          <div>│ 3. LSTM Layer 2 (Units: 256, Dropout: 0.20, Seq: False)    │</div>
          <div>│ 4. Dense Head (Units: 5004, Softmax Distribution)         │</div>
          <div>└───────────────────────────────────────────────────────────┘</div>
          <div>       │</div>
          <div>{'       ▼  Logits / Probability Distribution P(w_t+1)'}</div>
          <div>[Sampling Engine: Temperature Scaling ➔ Top-K ➔ Top-P]</div>
          <div>       │</div>
          <div>       ▼  (Sample next token ID ➔ Append to Context Window)</div>
          <div>[Generated Continuation Text]</div>
        </div>
      </div>

      {/* Architecture Deep Dive */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
        <div className="card">
          <div className="card-title">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={18} color="#38bdf8" />
              LSTM Cell Dynamics
            </span>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '0.75rem' }}>
            Standard RNNs suffer from vanishing gradients when learning dependencies across long sentences. LSTMs solve this via an internal cell state \(c_t\) and three regulatory gates:
          </p>
          <ul style={{ color: '#cbd5e1', fontSize: '0.85rem', paddingLeft: '1.25rem', lineHeight: 1.7 }}>
            <li><strong>Forget Gate (\(f_t\)):</strong> Decides which folklore context to discard.</li>
            <li><strong>Input Gate (\(i_t\)):</strong> Selects new information from the recent words to store in cell state.</li>
            <li><strong>Output Gate (\(o_t\)):</strong> Filters the cell state to produce the hidden activation vector \(h_t\).</li>
          </ul>
        </div>

        <div className="card">
          <div className="card-title">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Terminal size={18} color="#10b981" />
              Autoregressive Sampling
            </span>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '0.75rem' }}>
            Next tokens are predicted sequentially one by one:
          </p>
          <ul style={{ color: '#cbd5e1', fontSize: '0.85rem', paddingLeft: '1.25rem', lineHeight: 1.7 }}>
            <li><strong>Temperature (\(T\)):</strong> Divides logits by \(T\) before softmax. Lower temperatures focus on peak probabilities; higher temperatures increase entropy.</li>
            <li><strong>Top-K Truncation:</strong> Restricts sampling strictly to the top \(K\) highest-probability words.</li>
            <li><strong>Top-P (Nucleus):</strong> Dynamically chooses the smallest vocabulary subset whose cumulative probability reaches threshold \(P\).</li>
          </ul>
        </div>
      </div>

      {/* Resource & Hardware Specs */}
      <div className="card">
        <div className="card-title">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={18} color="#f59e0b" />
            Hardware & Efficiency Profile
          </span>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.6 }}>
          Designed and verified for the <strong>Intel Core Ultra 5 125H</strong> processor. Inference utilizes multi-threaded CPU instructions without requiring heavy GPU or CUDA runtimes. RAM footprint remains below 400 MB, allowing zero-latency story generation with high battery efficiency.
        </p>
      </div>
    </div>
  );
};
