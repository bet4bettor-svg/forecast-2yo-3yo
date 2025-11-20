import React, { useState, useMemo } from 'react';
import { defaultForecastData } from './data/trainingData';

const Forecast2yo3yo = () => {
  const [view, setView] = useState('predict');
  const [predictInputs, setPredictInputs] = useState({
    spsAvg2yo: 2.30,
    slAvg2yo: 7.5,
    distance2yo: 7.0
  });
  const [forecastData] = useState(defaultForecastData);

  const calculateForecastModelStats = (dataset) => {
    const n = dataset.length;
    if (n < 10) return null;
    
    const correlation = (x, y) => {
      const meanX = x.reduce((a, b) => a + b) / n;
      const meanY = y.reduce((a, b) => a + b) / n;
      const num = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
      const denX = Math.sqrt(x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0));
      const denY = Math.sqrt(y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0));
      return num / (denX * denY);
    };

    const distances = dataset.map(d => d.distance3yo);
    const spsAvgCorr = correlation(dataset.map(d => d.spsAvg2yo), distances);
    const slAvgCorr = correlation(dataset.map(d => d.slAvg2yo), distances);
    const distance2yoCorr = correlation(dataset.map(d => d.distance2yo), distances);

    const X = dataset.map(d => [1, d.spsAvg2yo, d.slAvg2yo, d.distance2yo]);
    const y = distances;
    
    const XtX = [
      [n, 
       X.reduce((s, row) => s + row[1], 0), 
       X.reduce((s, row) => s + row[2], 0),
       X.reduce((s, row) => s + row[3], 0)],
      [X.reduce((s, row) => s + row[1], 0), 
       X.reduce((s, row) => s + row[1] * row[1], 0), 
       X.reduce((s, row) => s + row[1] * row[2], 0),
       X.reduce((s, row) => s + row[1] * row[3], 0)],
      [X.reduce((s, row) => s + row[2], 0), 
       X.reduce((s, row) => s + row[1] * row[2], 0), 
       X.reduce((s, row) => s + row[2] * row[2], 0),
       X.reduce((s, row) => s + row[2] * row[3], 0)],
      [X.reduce((s, row) => s + row[3], 0),
       X.reduce((s, row) => s + row[1] * row[3], 0),
       X.reduce((s, row) => s + row[2] * row[3], 0),
       X.reduce((s, row) => s + row[3] * row[3], 0)]
    ];
    
    const Xty = [
      y.reduce((s, yi) => s + yi, 0),
      X.reduce((s, row, i) => s + row[1] * y[i], 0),
      X.reduce((s, row, i) => s + row[2] * y[i], 0),
      X.reduce((s, row, i) => s + row[3] * y[i], 0)
    ];

    const inv = invertMatrix4x4(XtX);
    
    const coefficients = [
      inv[0][0] * Xty[0] + inv[0][1] * Xty[1] + inv[0][2] * Xty[2] + inv[0][3] * Xty[3],
      inv[1][0] * Xty[0] + inv[1][1] * Xty[1] + inv[1][2] * Xty[2] + inv[1][3] * Xty[3],
      inv[2][0] * Xty[0] + inv[2][1] * Xty[1] + inv[2][2] * Xty[2] + inv[2][3] * Xty[3],
      inv[3][0] * Xty[0] + inv[3][1] * Xty[1] + inv[3][2] * Xty[2] + inv[3][3] * Xty[3]
    ];

    const predictions = X.map(row => coefficients[0] + coefficients[1] * row[1] + coefficients[2] * row[2] + coefficients[3] * row[3]);
    const meanY = y.reduce((a, b) => a + b) / n;
    const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - predictions[i], 2), 0);
    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0);
    const rSquared = 1 - (ssRes / ssTot);

    return { spsAvgCorr, slAvgCorr, distance2yoCorr, coefficients, rSquared, n };
  };

  const invertMatrix4x4 = (matrix) => {
    const m = matrix.map(row => [...row]);
    const inv = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];

    for (let i = 0; i < 4; i++) {
      let maxRow = i;
      for (let k = i + 1; k < 4; k++) {
        if (Math.abs(m[k][i]) > Math.abs(m[maxRow][i])) {
          maxRow = k;
        }
      }

      [m[i], m[maxRow]] = [m[maxRow], m[i]];
      [inv[i], inv[maxRow]] = [inv[maxRow], inv[i]];

      const pivot = m[i][i];
      for (let j = 0; j < 4; j++) {
        m[i][j] /= pivot;
        inv[i][j] /= pivot;
      }

      for (let k = 0; k < 4; k++) {
        if (k !== i) {
          const factor = m[k][i];
          for (let j = 0; j < 4; j++) {
            m[k][j] -= factor * m[i][j];
            inv[k][j] -= factor * inv[i][j];
          }
        }
      }
    }

    return inv;
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
const statsForecast = useMemo(() => forecastData ? calculateForecastModelStats(forecastData) : null, [forecastData]);

  const residualsForecast = useMemo(() => {
    if (!forecastData || !statsForecast) return [];
    return forecastData
      .map(horse => {
        const predicted = statsForecast.coefficients[0] + 
                         statsForecast.coefficients[1] * horse.spsAvg2yo + 
                         statsForecast.coefficients[2] * horse.slAvg2yo +
                         statsForecast.coefficients[3] * horse.distance2yo;
        const residual = horse.distance3yo - predicted;
        return { ...horse, predicted, residual, absResidual: Math.abs(residual) };
      })
      .sort((a, b) => b.absResidual - a.absResidual);
  }, [forecastData, statsForecast]);

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom right, #f0f4f8, #e8f5e9)', padding: '20px', fontFamily: 'Inter, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ background: 'white', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', padding: '32px', marginBottom: '24px' }}>
          <h1 style={{ fontSize: '32px', fontWeight: 'bold', textAlign: 'center', marginBottom: '8px', color: '#2C3E50', fontFamily: 'Work Sans, sans-serif' }}>
            2yo → 3yo Forecast
          </h1>
          <h2 style={{ fontSize: '16px', textAlign: 'center', marginBottom: '8px', color: '#3498DB', fontFamily: 'Work Sans, sans-serif' }}>
            Biomechanics + Race Context Analysis
          </h2>
          <p style={{ textAlign: 'center', color: '#666', fontSize: '14px' }}>
            Predict 3-year-old optimal distance from 2yo stride data and race context
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
          <button 
            onClick={() => setView('predict')} 
            style={{
              flex: 1,
              padding: '16px',
              borderRadius: '8px',
              border: view === 'predict' ? '2px solid #E67E22' : '2px solid #ddd',
              background: view === 'predict' ? '#FFF3E0' : '#f9f9f9',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#2C3E50' }}>🎯 Predict 2yo → 3yo Optimal Distance</div>
          </button>
          
          <button 
            onClick={() => setView('model-stats')} 
            style={{
              flex: 1,
              padding: '16px',
              borderRadius: '8px',
              border: view === 'model-stats' ? '2px solid #3498DB' : '2px solid #ddd',
              background: view === 'model-stats' ? '#E3F2FD' : '#f9f9f9',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#2C3E50' }}>📈 2yo → 3yo Model Stats</div>
          </button>
        </div>

        {view === 'model-stats' && (
          <div style={{ background: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', padding: '24px' }}>
            {statsForecast ? (
              <>
                <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px', textAlign: 'center', fontFamily: 'Work Sans, sans-serif' }}>2yo → 3yo Forecast Model</h2>
                
                <div style={{ background: 'linear-gradient(to right, #E3F2FD, #FCE4EC)', padding: '24px', borderRadius: '8px', border: '2px solid #3498DB', marginBottom: '16px' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '48px', fontWeight: 'bold', color: '#3498DB', marginBottom: '4px' }}>{(statsForecast.rSquared * 100).toFixed(1)}%</div>
                    <p style={{ fontSize: '14px', fontWeight: '600', color: '#2C3E50', marginBottom: '4px' }}>Model Accuracy R²</p>
                    <p style={{ fontSize: '12px', color: '#666' }}>Sample Size: {statsForecast.n} horses</p>
                  </div>
                </div>

                {residualsForecast.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: '14px', fontWeight: 'bold', color: '#2C3E50', marginBottom: '8px' }}>Top 5 Prediction Errors</h3>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ background: '#E3F2FD' }}>
                            <th style={{ border: '1px solid #3498DB', padding: '8px 4px', textAlign: 'left' }}>Horse</th>
                            <th style={{ border: '1px solid #3498DB', padding: '8px 4px', textAlign: 'right' }}>2yo Dist</th>
                            <th style={{ border: '1px solid #3498DB', padding: '8px 4px', textAlign: 'right' }}>Actual 3yo</th>
                            <th style={{ border: '1px solid #3498DB', padding: '8px 4px', textAlign: 'right' }}>Predicted</th>
                            <th style={{ border: '1px solid #3498DB', padding: '8px 4px', textAlign: 'right' }}>Error</th>
                          </tr>
                        </thead>
                        <tbody>
                          {residualsForecast.slice(0, 5).map((horse, idx) => (
                            <tr key={idx} style={{ background: horse.absResidual > 2 ? '#FFEBEE' : 'white' }}>
                              <td style={{ border: '1px solid #ddd', padding: '8px 4px', fontWeight: '500' }}>{horse.horse}</td>
                              <td style={{ border: '1px solid #ddd', padding: '8px 4px', textAlign: 'right' }}>{horse.distance2yo.toFixed(1)}f</td>
                              <td style={{ border: '1px solid #ddd', padding: '8px 4px', textAlign: 'right' }}>{horse.distance3yo.toFixed(1)}f</td>
                              <td style={{ border: '1px solid #ddd', padding: '8px 4px', textAlign: 'right' }}>{horse.predicted.toFixed(1)}f</td>
                              <td style={{ border: '1px solid #ddd', padding: '8px 4px', textAlign: 'right', fontWeight: 'bold' }}>
                                {horse.residual > 0 ? '+' : ''}{horse.residual.toFixed(2)}f
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px' }}>
                <p style={{ fontSize: '16px', color: '#666', marginBottom: '8px' }}>No Data Loaded</p>
                <p style={{ fontSize: '14px', color: '#999' }}>Import 2yo → 3yo data to see model statistics</p>
              </div>
            )}
          </div>
        )}

        {view === 'predict' && (
          <div style={{ background: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', padding: '24px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '12px', fontFamily: 'Work Sans, sans-serif' }}>Forecast 3yo Distance</h2>
            
            {statsForecast ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', marginBottom: '4px', color: '#2C3E50', minHeight: '32px' }}>2yo Average SPS (Hz)</label>
                    <input 
                      type="number" 
                      step="0.01" 
                      value={predictInputs.spsAvg2yo} 
                      onChange={(e) => setPredictInputs({...predictInputs, spsAvg2yo: parseFloat(e.target.value)})} 
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', fontSize: '14px' }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', marginBottom: '4px', color: '#2C3E50', minHeight: '32px' }}>2yo Average SL (m)</label>
                    <input 
                      type="number" 
                      step="0.01" 
                      value={predictInputs.slAvg2yo} 
                      onChange={(e) => setPredictInputs({...predictInputs, slAvg2yo: parseFloat(e.target.value)})} 
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', fontSize: '14px' }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', marginBottom: '4px', color: '#2C3E50', minHeight: '32px' }}>2yo Race Dist (f)</label>
                    <input 
                      type="number" 
                      step="0.5" 
                      value={predictInputs.distance2yo} 
                      onChange={(e) => setPredictInputs({...predictInputs, distance2yo: parseFloat(e.target.value)})} 
                      style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px', fontSize: '14px' }}
                    />
                  </div>
                </div>

                <div style={{ background: 'linear-gradient(to right, #E3F2FD, #FCE4EC)', padding: '24px', borderRadius: '8px', textAlign: 'center', border: '2px solid #3498DB' }}>
                  <p style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>PREDICTED 3YO DISTANCE</p>
                  <p style={{ fontSize: '48px', fontWeight: 'bold', color: '#3498DB', marginBottom: '4px' }}>
                    {(() => {
                      const prediction = statsForecast.coefficients[0] + 
                                       statsForecast.coefficients[1] * predictInputs.spsAvg2yo + 
                                       statsForecast.coefficients[2] * predictInputs.slAvg2yo +
                                       statsForecast.coefficients[3] * predictInputs.distance2yo;
                      return prediction.toFixed(1);
                    })()}f
                  </p>
                  <p style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>R²: {(statsForecast.rSquared * 100).toFixed(1)}% • Sample Size: {statsForecast.n} horses</p>
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '24px' }}>
                <p style={{ fontSize: '14px', color: '#666', marginBottom: '12px' }}>No model data loaded</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Forecast2yo3yo;
