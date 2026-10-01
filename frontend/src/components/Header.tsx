import React from 'react';
import { BookOpen, Sparkles, Activity } from 'lucide-react';

interface HeaderProps {
  activeTab: 'generator' | 'how-it-works' | 'metrics';
  setActiveTab: (tab: 'generator' | 'how-it-works' | 'metrics') => void;
  isBackendHealthy: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  isBackendHealthy,
}) => {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="logo-group">
          <div className="logo-icon">
            <BookOpen size={24} />
          </div>
          <div>
            <div className="logo-title">Story Continuation Generator</div>
            <div className="logo-subtitle">Local 2-Layer LSTM • TensorFlow / Keras</div>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-btn ${activeTab === 'generator' ? 'active' : ''}`}
            onClick={() => setActiveTab('generator')}
          >
            <Sparkles size={16} />
            Generator
          </button>
          <button
            className={`nav-btn ${activeTab === 'how-it-works' ? 'active' : ''}`}
            onClick={() => setActiveTab('how-it-works')}
          >
            <BookOpen size={16} />
            Architecture
          </button>
          <button
            className={`nav-btn ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            <Activity size={16} />
            Metrics
          </button>

          <div
            className={`status-pill ${
              isBackendHealthy ? 'status-online' : 'status-offline'
            }`}
            title={isBackendHealthy ? 'FastAPI Backend Ready' : 'Backend Disconnected'}
          >
            <span className="status-dot"></span>
            {isBackendHealthy ? 'Model Ready' : 'Connecting...'}
          </div>
        </nav>
      </div>
    </header>
  );
};
