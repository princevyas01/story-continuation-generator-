import React from 'react';
import { Sparkles, Cpu, BarChart2 } from 'lucide-react';

interface NavigationProps {
  activeTab: 'generator' | 'how-it-works' | 'metrics';
  setActiveTab: (tab: 'generator' | 'how-it-works' | 'metrics') => void;
}

export const Navigation: React.FC<NavigationProps> = ({ activeTab, setActiveTab }) => {
  return (
    <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
      <button
        className={`nav-btn ${activeTab === 'generator' ? 'active' : ''}`}
        onClick={() => setActiveTab('generator')}
        style={{ padding: '0.65rem 1.25rem' }}
      >
        <Sparkles size={16} />
        Story Generator
      </button>
      <button
        className={`nav-btn ${activeTab === 'how-it-works' ? 'active' : ''}`}
        onClick={() => setActiveTab('how-it-works')}
        style={{ padding: '0.65rem 1.25rem' }}
      >
        <Cpu size={16} />
        System & LSTM Architecture
      </button>
      <button
        className={`nav-btn ${activeTab === 'metrics' ? 'active' : ''}`}
        onClick={() => setActiveTab('metrics')}
        style={{ padding: '0.65rem 1.25rem' }}
      >
        <BarChart2 size={16} />
        Model Performance & Metrics
      </button>
    </div>
  );
};
