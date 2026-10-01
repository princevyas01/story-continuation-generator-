import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="footer">
      <p>
        <strong>Story Continuation Generator</strong> — Pure Local 2-Layer LSTM Language Model.
      </p>
      <p style={{ marginTop: '0.4rem', color: '#64748b' }}>
        Trained with TensorFlow/Keras on fairy tale corpora. Zero external generative APIs. Built for Intel Core Ultra 5 125H.
      </p>
    </footer>
  );
};
