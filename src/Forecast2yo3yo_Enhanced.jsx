import React, { useState } from 'react';
 
const Forecast2yo3yo = () => {
  const [view, setView] = useState('predict');
  const [predictInputs, setPredictInputs] = useState({
    spsAvg: 2.30,
    slAvg: 7.5,
    distance2yo: 7.0,
    minSPS: 2.18,
    goingGtS: false
  });
 
  // Pool 1 coefficients (n=382, R²=80.33%)
  const modelCoefficients = {
    intercept:    240.1830,
    spsAvg:      -189.1508,
    slAvg:         -0.6778,
    distance2yo:    0.9144,
    spsSquared:    38.1023
  };
 
  const modelRSquared = 0.8033;
  const modelN = 382;
  const gtsCorrection = -0.441;
 
  const styles = {
    container: {
      minHeight: '100vh',
      background: '#f5f5f5',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    },
    maxWidth: { maxWidth: '800px', margin: '0 auto' },
    header: { textAlign: 'center', marginBottom: '24px' },
    title: { fontSize: '32px', fontWeight: 'bold', color: '#1f2937', marginBottom: '8px' },
    subtitle: { fontSize: '14px', color: '#6b7280' },
    buttonContainer: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '24px' },
    button: { padding: '12px 24px', border: 'none', borderRadius: '8px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.2s' },
    buttonActive: { backgroundColor: '#2C3E50', color: 'white' },
    buttonInactive: { backgroundColor: 'white', color: '#6b7280', border: '2px solid #e5e7eb' },
    card: { backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', padding: '24px' },
    cardTitle: { fontSize: '20px', fontWeight: 'bold', marginBottom: '16px', color: '#1f2937' },
    inputGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' },
    inputFull: { marginBottom: '16px' },
    label: { display: 'block', fontSize: '14px', fontWeight: '600', marginBottom: '4px', color: '#374151' },
    input: { width: '100%', padding: '8px 12px', border: '2px solid #e5e7eb', borderRadius: '8px', fontSize: '14px', boxSizing: 'border-box' },
    select: { width: '100%', padding: '8px 12px', border: '2px solid #e5e7eb', borderRadius: '8px', fontSize: '14px', boxSizing: 'border-box', background: 'white' },
    resultBox: { background: '#e0f2fe', border: '3px solid #0ea5e9', borderRadius: '12px', padding: '24px', textAlign: 'center', marginBottom: '16px' },
    resultLabel: { fontSize: '14px', color: '#64748b', marginBottom: '8px' },
    resultValue: { fontSize: '72px', fontWeight: 'bold', color: '#0ea5e9', lineHeight: '1', marginBottom: '16px' },
    footer: { textAlign: 'center', fontSize: '12px', color: '#9ca3af', marginTop: '16px' },
    statsGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' },
    statBox: { background: '#e0f2fe', border: '3px solid #0ea5e9', borderRadius: '12px', padding: '24px', textAlign: 'center' },
    statLabel: { fontSize: '14px', color: '#64748b', marginBottom: '8px' },
    statValue: { fontSize: '48px', fontWeight: 'bold', color: '#0ea5e9', lineHeight: '1' },
    infoText: { fontSize: '14px', color: '#4b5563', lineHeight: '1.6', marginBottom: '12px' },
    correctionTag: { display: 'inline-block', fontSize: '12px', padding: '4px 10px', borderRadius: '12px', marginTop: '8px' },
    limitationBox: { background: '#fffbeb', border: '1px solid #f59e0b', borderRadius: '8px', padding: '16px', marginTop: '16px' },
    limitationTitle: { fontSize: '16px', fontWeight: '600', color: '#92400e', marginBottom: '8px' },
    limitationText: { fontSize: '13px', color: '#78350f', lineHeight: '1.6', margin: '4px 0' },
  };
 
  // Min SPS stayer correction — trigger at 10.5f
  const getMinSPSAdjustment = (basePrediction, minSPS) => {
    if (basePrediction < 10.5) return 0;
 
    let thresholds;
    if (basePrediction < 11) {
      thresholds = [[2.08, 2.1], [2.13, 1.26], [2.17, 0.7]];
    } else if (basePrediction < 13) {
      thresholds = [[2.08, 1.26], [2.13, 0.7], [2.17, 0.49]];
    } else {
      thresholds = [[2.08, 0.49], [2.13, 0.28], [2.17, 0.14]];
    }
 
    if (minSPS <= thresholds[0][0]) return thresholds[0][1];
    if (minSPS >= thresholds[thresholds.length - 1][0]) return 0;
 
    for (let i = 0; i < thresholds.length - 1; i++) {
      const [spsLow, adjLow] = thresholds[i];
      const [spsHigh, adjHigh] = thresholds[i + 1];
      if (minSPS >= spsLow && minSPS <= spsHigh) {
        const proportion = (minSPS - spsLow) / (spsHigh - spsLow);
        return adjLow + (adjHigh - adjLow) * proportion;
      }
    }
    return 0;
  };
 
  // Calculate prediction
  const basePrediction = modelCoefficients.intercept +
    modelCoefficients.spsAvg * predictInputs.spsAvg +
    modelCoefficients.slAvg * predictInputs.slAvg +
    modelCoefficients.distance2yo * predictInputs.distance2yo +
    modelCoefficients.spsSquared * predictInputs.spsAvg * predictInputs.spsAvg;
 
  const minSPSAdjustment = getMinSPSAdjustment(basePrediction, predictInputs.minSPS);
  const goingAdjustment = predictInputs.goingGtS ? gtsCorrection : 0;
  const finalPrediction = basePrediction + minSPSAdjustment + goingAdjustment;
 
  // Corrections applied labels
  const corrections = [];
  if (minSPSAdjustment !== 0) corrections.push(`Min SPS: ${minSPSAdjustment > 0 ? '+' : ''}${minSPSAdjustment.toFixed(2)}f`);
  if (goingAdjustment !== 0) corrections.push(`GtS: ${goingAdjustment.toFixed(2)}f`);
 
  return (
    <div style={styles.container}>
      <div style={styles.maxWidth}>
        <div style={styles.header}>
          <h1 style={styles.title}>2yo → 3yo Distance Predictor</h1>
          <p style={styles.subtitle}>
            Predict a horse's optimal 3-year-old distance from 2-year-old stride data
          </p>
        </div>
 
        <div style={styles.buttonContainer}>
          <button
            onClick={() => setView('predict')}
            style={{ ...styles.button, ...(view === 'predict' ? styles.buttonActive : styles.buttonInactive) }}
          >
            🎯 Predict Distance
          </button>
          <button
            onClick={() => setView('stats')}
            style={{ ...styles.button, ...(view === 'stats' ? styles.buttonActive : styles.buttonInactive) }}
          >
            📊 Model Stats
          </button>
        </div>
 
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
                Trained on {modelN} horses achieving {(modelRSquared * 100).toFixed(1)}% R² accuracy. 
                Predictions include automatic corrections for stayer pace profiles (Min SPS) and 
                Good to Soft ground conditions (−0.44f adjustment).
              </p>
            </div>
 
            <div style={styles.limitationBox}>
              <div style={styles.limitationTitle}>Known Limitations</div>
              <p style={styles.limitationText}>
                ⚑ Sprint predictions (≤6f optimal) over-predict by ~0.4f on average — this is structural to the polynomial and consistent across all data.
              </p>
              <p style={styles.limitationText}>
                ⚑ Stayers with moderate SPS profiles (base prediction 9.5–10.5f) may under-predict by 1–2f. The stride data cannot reliably separate a 10f horse from a 12f horse when both stride similarly as two-year-olds.
              </p>
              <p style={styles.limitationText}>
                ⚑ Extreme stayer predictions (13f+) carry wider error margins due to limited training observations at these distances.
              </p>
              <p style={styles.limitationText}>
                ⚑ Predictions from high-geometry tracks (Chester, Bath) or Kempton AW at 8f+ carry additional uncertainty from surface/track distortion of stride data.
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
                  type="number" step="0.01"
                  value={predictInputs.spsAvg}
                  onChange={(e) => setPredictInputs({...predictInputs, spsAvg: parseFloat(e.target.value) || 0})}
                  style={styles.input}
                />
              </div>
              <div>
                <label style={styles.label}>Average SL (m)</label>
                <input
                  type="number" step="0.01"
                  value={predictInputs.slAvg}
                  onChange={(e) => setPredictInputs({...predictInputs, slAvg: parseFloat(e.target.value) || 0})}
                  style={styles.input}
                />
              </div>
            </div>
 
            <div style={styles.inputGrid}>
              <div>
                <label style={styles.label}>2yo Race Distance (f)</label>
                <input
                  type="number" step="0.5"
                  value={predictInputs.distance2yo}
                  onChange={(e) => setPredictInputs({...predictInputs, distance2yo: parseFloat(e.target.value) || 0})}
                  style={styles.input}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: '4px' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '14px', fontWeight: '600', color: '#374151', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={predictInputs.goingGtS}
                    onChange={(e) => setPredictInputs({...predictInputs, goingGtS: e.target.checked})}
                    style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                  />
                  2yo race on Good to Soft?
                </label>
              </div>
            </div>
 
            <div style={{ marginBottom: '16px' }}>
              <label style={styles.label}>Min SPS (Hz)</label>
              <input
                type="number" step="0.01"
                value={predictInputs.minSPS}
                onChange={(e) => setPredictInputs({...predictInputs, minSPS: parseFloat(e.target.value) || 0})}
                style={{...styles.input, maxWidth: 'calc(50% - 6px)'}}
              />
            </div>
 
            <div style={styles.resultBox}>
              <div style={styles.resultLabel}>Predicted 3-Year-Old Optimal Distance</div>
              <div style={styles.resultValue}>{finalPrediction.toFixed(1)}f</div>
 
              {corrections.length > 0 && (
                <div style={{ marginBottom: '12px' }}>
                  {corrections.map((c, i) => (
                    <span key={i} style={{
                      ...styles.correctionTag,
                      background: c.includes('GtS') ? '#fef3c7' : '#dbeafe',
                      color: c.includes('GtS') ? '#92400e' : '#1e40af',
                      marginRight: '8px'
                    }}>
                      {c}
                    </span>
                  ))}
                </div>
              )}
 
            </div>
 
            <div style={styles.footer}>
              <p>R²: {(modelRSquared * 100).toFixed(1)}% • n={modelN} • Pool 1 (all tagged data)</p>
            </div>
          </div>
        )}
 
        <div style={styles.footer}>
          <p>StridePredictor.com • 2yo → 3yo Forward Projection Model</p>
        </div>
      </div>
    </div>
  );
};
 
export default Forecast2yo3yo;
 