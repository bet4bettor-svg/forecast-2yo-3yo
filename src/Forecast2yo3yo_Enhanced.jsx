import React, { useState } from 'react';

const Forecast2yo3yoEnhanced = () => {
  const [view, setView] = useState('predict');
  const [predictInputs, setPredictInputs] = useState({
    spsAvg: 2.30,
    slAvg: 7.5,
    distance2yo: 7.0,
    minSPS: 2.18,
    minSL: 7.2,
    maxSL: 7.8
  });

  // Model B coefficients (238 horses - updated model)
  const modelCoefficients = {
    intercept: 34.2380,
    spsAvg: -12.2353,
    slAvg: -0.6289,
    distance2yo: 0.9820
  };
  
  const modelRSquared = 0.777;
  const modelN = 238;

  // Inline styles object
  const styles = {
    container: {
      minHeight: '100vh',
      background: '#f5f5f5',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    },
    maxWidth: {
      maxWidth: '800px',
      margin: '0 auto'
    },
    header: {
      textAlign: 'center',
      marginBottom: '24px'
    },
    title: {
      fontSize: '32px',
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: '8px'
    },
    subtitle: {
      fontSize: '14px',
      color: '#6b7280'
    },
    buttonContainer: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '12px',
      marginBottom: '24px'
    },
    button: {
      padding: '12px 24px',
      border: 'none',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s'
    },
    buttonActive: {
      backgroundColor: '#10b981',
      color: 'white'
    },
    buttonInactive: {
      backgroundColor: 'white',
      color: '#6b7280',
      border: '2px solid #e5e7eb'
    },
    card: {
      backgroundColor: 'white',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      padding: '24px'
    },
    cardTitle: {
      fontSize: '20px',
      fontWeight: 'bold',
      marginBottom: '16px',
      color: '#1f2937'
    },
    inputContainer: {
      marginBottom: '24px'
    },
    inputGrid: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '12px',
      marginBottom: '16px'
    },
    inputFull: {
      marginBottom: '16px'
    },
    label: {
      display: 'block',
      fontSize: '14px',
      fontWeight: '600',
      marginBottom: '4px',
      color: '#374151'
    },
    input: {
      width: '100%',
      padding: '8px 12px',
      border: '2px solid #e5e7eb',
      borderRadius: '8px',
      fontSize: '14px',
      boxSizing: 'border-box'
    },
    resultBox: {
      background: '#e0f2fe',
      border: '3px solid #0ea5e9',
      borderRadius: '12px',
      padding: '24px',
      textAlign: 'center',
      marginBottom: '16px'
    },
    resultLabel: {
      fontSize: '14px',
      color: '#64748b',
      marginBottom: '8px'
    },
    resultValue: {
      fontSize: '72px',
      fontWeight: 'bold',
      color: '#0ea5e9',
      lineHeight: '1',
      marginBottom: '16px'
    },
    confidenceBox: {
      background: 'white',
      borderRadius: '8px',
      padding: '12px',
      textAlign: 'center'
    },
    confidenceLabel: {
      fontSize: '14px',
      color: '#64748b',
      marginBottom: '4px'
    },
    stars: {
      fontSize: '24px',
      marginBottom: '4px'
    },
    confidenceText: {
      fontSize: '12px',
      color: '#9ca3af'
    },
    footer: {
      textAlign: 'center',
      fontSize: '12px',
      color: '#9ca3af',
      marginTop: '16px'
    },
    statsGrid: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '16px',
      marginBottom: '24px'
    },
    statBox: {
      background: '#e0f2fe',
      border: '3px solid #0ea5e9',
      borderRadius: '12px',
      padding: '24px',
      textAlign: 'center'
    },
    statLabel: {
      fontSize: '14px',
      color: '#64748b',
      marginBottom: '8px'
    },
    statValue: {
      fontSize: '48px',
      fontWeight: 'bold',
      color: '#0ea5e9',
      lineHeight: '1'
    },
    infoText: {
      fontSize: '14px',
      color: '#4b5563',
      lineHeight: '1.6',
      marginBottom: '12px'
    }
  };

  // Calculate confidence rating
  const getConfidenceRating = (slRange, basePrediction) => {
    let category = 'middle';
    if (basePrediction < 7.1) category = 'sprint';
    else if (basePrediction > 10.0) category = 'staying';
    
    if (category === 'middle') {
      if (slRange < 0.85) return { stars: 5, label: 'Very High', color: 'green' };
      if (slRange < 1.00) return { stars: 4, label: 'High', color: 'blue' };
      if (slRange < 1.20) return { stars: 3, label: 'Moderate', color: 'yellow' };
      if (slRange < 1.35) return { stars: 2, label: 'Low', color: 'orange' };
      return { stars: 1, label: 'Very Low', color: 'red' };
    }
    
    if (category === 'sprint') {
      if (slRange < 0.85) return { stars: 5, label: 'Very High', color: 'green' };
      if (slRange < 1.00) return { stars: 4, label: 'High', color: 'blue' };
      if (slRange < 1.15) return { stars: 3, label: 'Moderate', color: 'yellow' };
      if (slRange < 1.30) return { stars: 2, label: 'Low', color: 'orange' };
      return { stars: 1, label: 'Very Low', color: 'red' };
    }
    
    if (slRange < 1.15) return { stars: 5, label: 'Very High', color: 'green' };
    if (slRange < 1.30) return { stars: 4, label: 'High', color: 'blue' };
    if (slRange < 1.50) return { stars: 3, label: 'Moderate', color: 'yellow' };
    if (slRange < 1.65) return { stars: 2, label: 'Low', color: 'orange' };
    return { stars: 1, label: 'Very Low', color: 'red' };
  };

  // Calculate Min SPS adjustment with SMOOTH gradients and 10.5f cutoff
  const getMinSPSAdjustment = (basePrediction, minSPS) => {
    // CUTOFF: Only apply adjustment for stamina predictions
    if (basePrediction < 10.5) {
      return 0;
    }
    
    let thresholds;
    if (basePrediction < 11) {
      thresholds = [
        [2.08, 3.0],
        [2.13, 1.8],
        [2.18, 1.0],
        [2.30, 0.5]
      ];
    } else if (basePrediction < 13) {
      thresholds = [
        [2.08, 1.8],
        [2.13, 1.0],
        [2.18, 0.7],
        [2.30, 0.3]
      ];
    } else {
      thresholds = [
        [2.08, 0.7],
        [2.13, 0.4],
        [2.18, 0.2],
        [2.30, 0.0]
      ];
    }
    
    if (minSPS <= thresholds[0][0]) {
      return thresholds[0][1];
    }
    
    if (minSPS >= thresholds[thresholds.length - 1][0]) {
      return thresholds[thresholds.length - 1][1];
    }
    
    for (let i = 0; i < thresholds.length - 1; i++) {
      const [spsLow, adjLow] = thresholds[i];
      const [spsHigh, adjHigh] = thresholds[i + 1];
      
      if (minSPS >= spsLow && minSPS <= spsHigh) {
        const proportion = (minSPS - spsLow) / (spsHigh - spsLow);
        const adjustment = adjLow + (adjHigh - adjLow) * proportion;
        return adjustment;
      }
    }
    
    return 0;
  };

  // Calculate prediction
  const basePrediction = modelCoefficients.intercept + 
                        modelCoefficients.spsAvg * predictInputs.spsAvg + 
                        modelCoefficients.slAvg * predictInputs.slAvg +
                        modelCoefficients.distance2yo * predictInputs.distance2yo;
  
  const slRange = predictInputs.maxSL - predictInputs.minSL;
  const confidence = getConfidenceRating(slRange, basePrediction);
  const minSPSAdjustment = getMinSPSAdjustment(basePrediction, predictInputs.minSPS);
  const finalPrediction = basePrediction + minSPSAdjustment;

  return (
    <div style={styles.container}>
      <div style={styles.maxWidth}>
        
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>2yo → 3yo Distance Predictor</h1>
          <p style={styles.subtitle}>
            Predict a horse's optimal 3-year-old distance based on 2-year-old stride data
          </p>
        </div>

        {/* Navigation Buttons */}
        <div style={styles.buttonContainer}>
          <button
            onClick={() => setView('predict')}
            style={{
              ...styles.button,
              ...(view === 'predict' ? styles.buttonActive : styles.buttonInactive)
            }}
          >
            🎯 Predict Distance
          </button>
          <button
            onClick={() => setView('stats')}
            style={{
              ...styles.button,
              ...(view === 'stats' ? styles.buttonActive : styles.buttonInactive)
            }}
          >
            📊 Model Stats
          </button>
        </div>

        {/* Model Stats View */}
        {view === 'stats' && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Model Performance</h2>
            
            <div style={styles.statsGrid}>
              <div style={styles.statBox}>
                <div style={styles.statLabel}>R² Value</div>
                <div style={styles.statValue}>{(modelRSquared * 100).toFixed(1)}%</div>
              </div>
              
              <div style={styles.statBox}>
                <div style={styles.statLabel}>Sample Size</div>
                <div style={styles.statValue}>{modelN}</div>
              </div>
            </div>

            <div>
              <p style={styles.infoText}>
                This model has been trained on {modelN} horses achieving {(modelRSquared * 100).toFixed(1)}% R² accuracy.
              </p>
            </div>
          </div>
        )}

        {/* Predict Distance View */}
        {view === 'predict' && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Enter 2-Year-Old Stride Data</h2>
            
            {/* Input Fields */}
            <div style={styles.inputContainer}>
              <div style={styles.inputGrid}>
                <div>
                  <label style={styles.label}>Average SPS (Hz)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.spsAvg}
                    onChange={(e) => setPredictInputs({...predictInputs, spsAvg: parseFloat(e.target.value) || 0})}
                    style={styles.input}
                  />
                </div>
                <div>
                  <label style={styles.label}>Average SL (m)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.slAvg}
                    onChange={(e) => setPredictInputs({...predictInputs, slAvg: parseFloat(e.target.value) || 0})}
                    style={styles.input}
                  />
                </div>
              </div>

              <div style={styles.inputFull}>
                <label style={styles.label}>2yo Optimal Distance (furlongs)</label>
                <input
                  type="number"
                  step="0.5"
                  value={predictInputs.distance2yo}
                  onChange={(e) => setPredictInputs({...predictInputs, distance2yo: parseFloat(e.target.value) || 0})}
                  style={styles.input}
                />
              </div>

              <div style={styles.inputGrid}>
                <div>
                  <label style={styles.label}>Min SPS (Hz)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.minSPS}
                    onChange={(e) => setPredictInputs({...predictInputs, minSPS: parseFloat(e.target.value) || 0})}
                    style={styles.input}
                  />
                </div>
                <div>
                  <label style={styles.label}>Min SL (m)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.minSL}
                    onChange={(e) => setPredictInputs({...predictInputs, minSL: parseFloat(e.target.value) || 0})}
                    style={styles.input}
                  />
                </div>
              </div>

              <div style={styles.inputFull}>
                <label style={styles.label}>Max SL (m)</label>
                <input
                  type="number"
                  step="0.01"
                  value={predictInputs.maxSL}
                  onChange={(e) => setPredictInputs({...predictInputs, maxSL: parseFloat(e.target.value) || 0})}
                  style={styles.input}
                />
              </div>
            </div>

            {/* Prediction Output */}
            <div style={styles.resultBox}>
              <div style={styles.resultLabel}>Predicted 3-Year-Old Optimal Distance</div>
              <div style={styles.resultValue}>{finalPrediction.toFixed(1)}f</div>
              
              <div style={styles.confidenceBox}>
                <div style={styles.confidenceLabel}>Prediction Confidence</div>
                <div style={styles.stars}>{'⭐'.repeat(confidence.stars)}</div>
                <div style={styles.confidenceText}>{confidence.label}</div>
              </div>
            </div>

            <div style={styles.footer}>
              <p>R²: {(modelRSquared * 100).toFixed(1)}% • Sample Size: {modelN} horses</p>
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={styles.footer}>
          <p>StridePredictor.com • Stride Biomechanics Analysis</p>
        </div>

      </div>
    </div>
  );
};

export default Forecast2yo3yoEnhanced;