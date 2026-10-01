import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Navigation } from './components/Navigation';
import { Footer } from './components/Footer';
import { GeneratorPage } from './pages/GeneratorPage';
import { HowItWorksPage } from './pages/HowItWorksPage';
import { MetricsPage } from './pages/MetricsPage';
import { api } from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'generator' | 'how-it-works' | 'metrics'>('generator');
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const res = await api.checkHealth();
        if (isMounted) {
          setIsBackendHealthy(res.model_loaded);
        }
      } catch {
        if (isMounted) {
          setIsBackendHealthy(false);
        }
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendHealthy={isBackendHealthy}
      />

      <main className="main-content">
        <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

        {activeTab === 'generator' && <GeneratorPage />}
        {activeTab === 'how-it-works' && <HowItWorksPage />}
        {activeTab === 'metrics' && <MetricsPage />}
      </main>

      <Footer />
    </div>
  );
};

export default App;
