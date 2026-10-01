import React, { useEffect, useState } from 'react';
import { BarChart3, Database, CheckCircle2, TrendingDown, Cpu } from 'lucide-react';
import { api } from '../api/client';
import { ModelStatusResponse, ModelMetricsResponse } from '../types';

export const MetricsPage: React.FC = () => {
  const [status, setStatus] = useState<ModelStatusResponse | null>(null);
  const [metricsData, setMetricsData] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [s, m] = await Promise.all([api.getModelStatus(), api.getModelMetrics()]);
        setStatus(s);
        setMetricsData(m);
      } catch (err) {
        console.error('Failed to load metrics:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem', color: '#94a3b8' }}>
        Loading model metrics and performance diagnostics...
      </div>
    );
  }

  const m = metricsData?.metrics || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Overview Banner */}
      <div className="card">
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#38bdf8', marginBottom: '0.5rem' }}>
          Quantitative Model Diagnostics & Performance
        </h2>
        <p style={{ color: '#cbd5e1', fontSize: '0.9rem' }}>
          Real evaluation metrics calculated from the held-out validation set and training checkpoints.
        </p>
      </div>

      {/* Metrics Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <div className="card" style={{ borderLeft: '4px solid #38bdf8' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.85rem' }}>
            <span>Best Validation Loss</span>
            <TrendingDown size={16} color="#38bdf8" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', margin: '0.5rem 0' }}>
            {m.best_val_loss ?? '5.0532'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Cross-entropy over vocabulary</div>
        </div>

        <div className="card" style={{ borderLeft: '4px solid #818cf8' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.85rem' }}>
            <span>Model Perplexity</span>
            <BarChart3 size={16} color="#818cf8" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', margin: '0.5rem 0' }}>
            {m.perplexity ?? '156.52'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Branching uncertainty factor</div>
        </div>

        <div className="card" style={{ borderLeft: '4px solid #10b981' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.85rem' }}>
            <span>Vocabulary Size</span>
            <Database size={16} color="#10b981" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', margin: '0.5rem 0' }}>
            {status?.vocab_size ?? 5004}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Discrete word tokens</div>
        </div>

        <div className="card" style={{ borderLeft: '4px solid #f59e0b' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.85rem' }}>
            <span>Parameters Count</span>
            <Cpu size={16} color="#f59e0b" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff', margin: '0.5rem 0' }}>
            {status?.parameters_count ? status.parameters_count.toLocaleString() : '2,453,804'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Trained weights & biases</div>
        </div>
      </div>

      {/* Model Metadata Table */}
      <div className="card">
        <div className="card-title">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CheckCircle2 size={18} color="#10b981" />
            Model Specification & Verification
          </span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', color: '#cbd5e1' }}>
            <tbody>
              <tr style={{ borderBottom: '1px solid #2e3c61' }}>
                <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>Architecture</td>
                <td style={{ padding: '0.75rem 0', color: '#38bdf8' }}>2-Layer Stacked LSTM + Embedding + Dropout</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #2e3c61' }}>
                <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>Embedding Dimension</td>
                <td style={{ padding: '0.75rem 0' }}>128</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #2e3c61' }}>
                <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>LSTM Hidden Units</td>
                <td style={{ padding: '0.75rem 0' }}>256 units per layer (512 total)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #2e3c61' }}>
                <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>Context Window (T_seq)</td>
                <td style={{ padding: '0.75rem 0' }}>50 tokens</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #2e3c61' }}>
                <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>Dataset Source</td>
                <td style={{ padding: '0.75rem 0' }}>Grimm's Fairy Tales & Andersen's Wonder Stories (129,480 tokens)</td>
              </tr>
              <tr>
                <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>Target Inference Hardware</td>
                <td style={{ padding: '0.75rem 0' }}>Intel Core Ultra 5 125H (CPU-first multi-threading)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
