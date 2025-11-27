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

  // Model B coefficients
  const modelCoefficients = {
    intercept: 42.7553,
    spsAvg: -14.5569,
    slAvg: -0.8917,
    distance2yo: 0.8062
  };
  
  const modelRSquared = 0.775;
  const modelN = 159;

  const convertToRaceDistance = (furlongs) => {
    const distances = [
      [5, '5f'], [5.5, '5.5f'], [6, '6f'], [6.5, '6.5f'], [7, '7f'],
      [8, '1m'], [8.5, '1m½f'], [9, '1m1f'], [10, '1m2f'], [11, '1m3f'],
      [12, '1m4f'], [13, '1m5f'], [14, '1m6f'], [15, '1m7f'], [16, '2m']
    ];
    
    let closest = distances[0];
    let minDiff = Math.abs(furlongs - distances[0][0]);
    
    for (let i = 1; i < distances.length; i++) {
      const diff = Math.abs(furlongs - distances[i][0]);
      if (diff < minDiff) {
        minDiff = diff;
        closest = distances[i];
      }
    }
    
    return closest[1];
  };

  const getConfidenceRating = (slRange, basePrediction) => {
    let category = 'middle';
    if (basePrediction < 7.1) category = 'sprint';
    else if (basePrediction > 10.0) category = 'staying';
    
    if (category === 'middle') {
      if (slRange < 0.85) return { stars: 5, label: 'Very High' };
      if (slRange < 1.00) return { stars: 4, label: 'High' };
      if (slRange < 1.20) return { stars: 3, label: 'Moderate' };
      if (slRange < 1.35) return { stars: 2, label: 'Low' };
      return { stars: 1, label: 'Very Low' };
    }
    
    if (category === 'sprint') {
      if (slRange < 0.85) return { stars: 5, label: 'Very High' };
      if (slRange < 1.00) return { stars: 4, label: 'High' };
      if (slRange < 1.15) return { stars: 3, label: 'Moderate' };
      if (slRange < 1.30) return { stars: 2, label: 'Low' };
      return { stars: 1, label: 'Very Low' };
    }
    
    if (slRange < 1.15) return { stars: 5, label: 'Very High' };
    if (slRange < 1.30) return { stars: 4, label: 'High' };
    if (slRange < 1.50) return { stars: 3, label: 'Moderate' };
    if (slRange < 1.65) return { stars: 2, label: 'Low' };
    return { stars: 1, label: 'Very Low' };
  };

  const getMinSPSAdjustment = (basePrediction, minSPS) => {
    if (basePrediction < 11) {
      if (minSPS < 2.08) return 4.5;
      if (minSPS < 2.13) return 2.5;
      if (minSPS < 2.18) return 1.5;
      return 0;
    }
    
    if (basePrediction < 13) {
      if (minSPS < 2.08) return 2.5;
      if (minSPS < 2.13) return 1.5;
      if (minSPS < 2.18) return 1.0;
      return 0;
    }
    
    if (minSPS < 2.08) return 1.0;
    if (minSPS < 2.13) return 0.5;
    if (minSPS < 2.18) return 0.25;
    return 0;
  };

  const basePrediction = modelCoefficients.intercept + 
                        modelCoefficients.spsAvg * predictInputs.spsAvg + 
                        modelCoefficients.slAvg * predictInputs.slAvg +
                        modelCoefficients.distance2yo * predictInputs.distance2yo;
  
  const slRange = predictInputs.maxSL - predictInputs.minSL;
  const confidence = getConfidenceRating(slRange, basePrediction);
  const minSPSAdjustment = getMinSPSAdjustment(basePrediction, predictInputs.minSPS);
  const finalPrediction = basePrediction + minSPSAdjustment;

  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: '#f5f5f5',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    },
    innerContainer: {
      maxWidth: '1200px',
      margin: '0 auto'
    },
    header: {
      textAlign: 'center',
      marginBottom: '30px',
      backgroundColor: 'white',
      padding: '30px',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    },
    title: {
      fontSize: '32px',
      fontWeight: 'bold',
      color: '#333',
      marginBottom: '10px'
    },
    subtitle: {
      fontSize: '16px',
      color: '#666'
    },
    buttonContainer: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '15px',
      marginBottom: '30px'
    },
    button: {
      padding: '15px 20px',
      fontSize: '16px',
      fontWeight: '600',
      border: '2px solid #ddd',
      borderRadius: '8px',
      backgroundColor: 'white',
      cursor: 'pointer',
      transition: 'all 0.2s',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px'
    },
    buttonActive: {
      backgroundColor: '#10b981',
      color: 'white',
      borderColor: '#10b981'
    },
    card: {
      backgroundColor: 'white',
      borderRadius: '8px',
      padding: '30px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    },
    cardTitle: {
      fontSize: '24px',
      fontWeight: 'bold',
      marginBottom: '20px',
      color: '#333'
    },
    inputGrid: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '15px',
      marginBottom: '20px'
    },
    inputGridThree: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr 1fr',
      gap: '15px',
      marginBottom: '20px'
    },
    inputGroup: {
      marginBottom: '20px'
    },
    label: {
      display: 'block',
      fontSize: '14px',
      fontWeight: '600',
      marginBottom: '8px',
      color: '#333'
    },
    input: {
      width: '100%',
      padding: '10px 12px',
      fontSize: '14px',
      border: '1px solid #ddd',
      borderRadius: '6px',
      boxSizing: 'border-box'
    },
    resultBox: {
      backgroundColor: '#e0f2fe',
      border: '2px solid #0ea5e9',
      borderRadius: '8px',
      padding: '30px',
      textAlign: 'center',
      marginTop: '20px'
    },
    resultLabel: {
      fontSize: '14px',
      color: '#666',
      marginBottom: '10px',
      textTransform: 'uppercase',
      letterSpacing: '1px'
    },
    resultValue: {
      fontSize: '72px',
      fontWeight: 'bold',
      color: '#0ea5e9',
      margin: '10px 0'
    },
    resultSubtext: {
      fontSize: '14px',
      color: '#666',
      marginTop: '15px'
    },
    confidenceBox: {
      textAlign: 'center',
      marginTop: '20px',
      padding: '15px',
      backgroundColor: '#f9fafb',
      borderRadius: '6px'
    },
    confidenceLabel: {
      fontSize: '14px',
      color: '#666',
      marginBottom: '8px'
    },
    stars: {
      fontSize: '24px',
      marginBottom: '5px'
    },
    confidenceText: {
      fontSize: '12px',
      color: '#666'
    },
    statsBox: {
      backgroundColor: '#e0f2fe',
      border: '2px solid #0ea5e9',
      borderRadius: '8px',
      padding: '40px',
      textAlign: 'center'
    },
    statsLabel: {
      fontSize: '16px',
      color: '#666',
      marginBottom: '15px'
    },
    statsValue: {
      fontSize: '96px',
      fontWeight: 'bold',
      color: '#0ea5e9',
      marginBottom: '10px'
    },
    statsDescription: {
      fontSize: '14px',
      color: '#666'
    },
    footer: {
      textAlign: 'center',
      marginTop: '30px',
      fontSize: '12px',
      color: '#999'
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.innerContainer}>
        
        <div style={styles.header}>
          <h1 style={styles.title}>2yo → 3yo Distance Predictor</h1>
          <p style={styles.subtitle}>
            Predict a horse's optimal 3-year-old distance based on 2-year-old stride data
          </p>
        </div>

        <div style={styles.buttonContainer}>
          <button
            onClick={() => setView('predict')}
            style={{
              ...styles.button,
              ...(view === 'predict' ? styles.buttonActive : {})
            }}
          >
            <span>🎯</span> Predict Distance
          </button>
          <button
            onClick={() => setView('stats')}
            style={{
              ...styles.button,
              ...(view === 'stats' ? styles.buttonActive : {})
            }}
          >
            <span>📊</span> Model Stats
          </button>
        </div>

        {view === 'stats' && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Model Performance</h2>
            
            <div style={styles.statsBox}>
              <p style={styles.statsLabel}>Model Accuracy</p>
              <p style={styles.statsValue}>{(modelRSquared * 100).toFixed(1)}%</p>
              <p style={styles.statsDescription}>R-squared (variance explained)</p>
            </div>

            <div style={{ marginTop: '20px', textAlign: 'center', color: '#666' }}>
              <p>Trained on {modelN} horses</p>
              <p style={{ marginTop: '10px' }}>
                This model predicts 3-year-old optimal distance from 2-year-old stride biomechanics
              </p>
            </div>
          </div>
        )}

        {view === 'predict' && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Enter 2-Year-Old Stride Data</h2>
            
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

            <div style={styles.inputGroup}>
              <label style={styles.label}>2yo Optimal Distance (furlongs)</label>
              <input
                type="number"
                step="0.5"
                value={predictInputs.distance2yo}
                onChange={(e) => setPredictInputs({...predictInputs, distance2yo: parseFloat(e.target.value) || 0})}
                style={styles.input}
              />
            </div>

            <div style={styles.inputGridThree}>
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
              <div>
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

            <div style={styles.resultBox}>
              <p style={styles.resultLabel}>Predicted 3-Year-Old Optimal Distance</p>
              <p style={styles.resultValue}>{finalPrediction.toFixed(1)}f ({convertToRaceDistance(finalPrediction)})</p>
              <p style={styles.resultSubtext}>
                R²: {(modelRSquared * 100).toFixed(1)}% • Sample Size: {modelN} horses
              </p>
            </div>

            <div style={styles.confidenceBox}>
              <p style={styles.confidenceLabel}>Prediction Confidence</p>
              <div style={styles.stars}>{'⭐'.repeat(confidence.stars)}</div>
              <p style={styles.confidenceText}>{confidence.label}</p>
            </div>
          </div>
        )}

        <div style={styles.footer}>
          <p>StridePredictor.com • Stride Biomechanics Analysis</p>
        </div>

      </div>
    </div>
  );
};

export default Forecast2yo3yoEnhanced;