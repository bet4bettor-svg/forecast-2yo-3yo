import React, { useState, useMemo } from 'react';
import { defaultForecastData } from './data/trainingData';

export default () => {
  const [view, setView] = useState('import');
  const [selectedAge, setSelectedAge] = useState('4plus');
  const [predictInputs, setPredictInputs] = useState({
    age: 3,
    spsAvg: 2.30,
    slAvg: 7.5,
    distance2yo: 7.0,
    ageTarget: 3,
    // NEW INPUTS for enhanced model
    minSPS: 2.18,
    minSL: 7.2,
    maxSL: 7.8
  });
  const [importText, setImportText] = useState('');
  const [customData, setCustomData] = useState(null);
  const [forecastImportText, setForecastImportText] = useState('');
  const [forecastData, setForecastData] = useState(defaultForecastData);

  const defaultData = [
    { horse: "Lambourn", distance: 12.03, age: 3, spsAvg: 2.13, slAvg: 6.87 },
    { horse: "Trefor", distance: 5.40, age: 4, spsAvg: 2.39, slAvg: 7.82 },
    { horse: "Squealer", distance: 5.10, age: 4, spsAvg: 2.55, slAvg: 7.54 },
    { horse: "The Man", distance: 5.00, age: 3, spsAvg: 2.42, slAvg: 7.73 },
    { horse: "Toca Madera", distance: 5.00, age: 3, spsAvg: 2.42, slAvg: 7.75 },
    { horse: "Vantheman", distance: 5.00, age: 3, spsAvg: 2.38, slAvg: 7.99 },
    { horse: "Jubilee Walk", distance: 5.00, age: 3, spsAvg: 2.39, slAvg: 7.99 },
    { horse: "Mon Na Slieve", distance: 5.00, age: 4, spsAvg: 2.45, slAvg: 7.58 },
    { horse: "Spring Is Sprung", distance: 5.00, age: 6, spsAvg: 2.30, slAvg: 8.26 },
    { horse: "Air Force One", distance: 5.00, age: 4, spsAvg: 2.47, slAvg: 7.42 }
  ];

  const data = customData || defaultData;
  const data3yo = useMemo(() => data.filter(h => h.age === 3), [data]);
  const data4plus = useMemo(() => data.filter(h => h.age >= 4), [data]);

  // NEW FUNCTION: Calculate confidence rating based on stride length range
  const getConfidenceRating = (slRange, basePrediction) => {
    // Determine distance category based on base prediction
    let category = 'middle'; // default
    if (basePrediction < 7.1) category = 'sprint';
    else if (basePrediction > 10.0) category = 'staying';
    
    // 2yo Middle Distance thresholds (most common for Derby prospects)
    if (category === 'middle') {
      if (slRange < 0.85) return { stars: 5, label: 'Very High', color: 'green' };
      if (slRange < 1.00) return { stars: 4, label: 'High', color: 'blue' };
      if (slRange < 1.20) return { stars: 3, label: 'Moderate', color: 'yellow' };
      if (slRange < 1.35) return { stars: 2, label: 'Low', color: 'orange' };
      return { stars: 1, label: 'Very Low', color: 'red' };
    }
    
    // Sprint thresholds
    if (category === 'sprint') {
      if (slRange < 0.85) return { stars: 5, label: 'Very High', color: 'green' };
      if (slRange < 1.00) return { stars: 4, label: 'High', color: 'blue' };
      if (slRange < 1.15) return { stars: 3, label: 'Moderate', color: 'yellow' };
      if (slRange < 1.30) return { stars: 2, label: 'Low', color: 'orange' };
      return { stars: 1, label: 'Very Low', color: 'red' };
    }
    
    // Staying thresholds
    if (slRange < 1.15) return { stars: 5, label: 'Very High', color: 'green' };
    if (slRange < 1.30) return { stars: 4, label: 'High', color: 'blue' };
    if (slRange < 1.50) return { stars: 3, label: 'Moderate', color: 'yellow' };
    if (slRange < 1.65) return { stars: 2, label: 'Low', color: 'orange' };
    return { stars: 1, label: 'Very Low', color: 'red' };
  };

  // NEW FUNCTION: Get stamina profile based on Min SPS
  const getStaminaProfile = (minSPS) => {
    if (minSPS < 2.08) return { level: 'Elite Stayer', icon: '🔥🔥', color: 'red' };
    if (minSPS < 2.13) return { level: 'Strong Stayer', icon: '🔥', color: 'orange' };
    if (minSPS < 2.18) return { level: 'Moderate Stamina', icon: '⚡', color: 'yellow' };
    return { level: 'Standard Profile', icon: '✓', color: 'gray' };
  };

  // NEW FUNCTION: Calculate Min SPS adjustment
  const getMinSPSAdjustment = (basePrediction, minSPS) => {
    // Base < 11f
    if (basePrediction < 11) {
      if (minSPS < 2.08) return 4.5;
      if (minSPS < 2.13) return 2.5;
      if (minSPS < 2.18) return 1.5;
      return 0; // Standard horses - no adjustment
    }
    
    // Base 11-13f
    if (basePrediction < 13) {
      if (minSPS < 2.08) return 2.5;
      if (minSPS < 2.13) return 1.5;
      if (minSPS < 2.18) return 1.0;
      return 0; // Standard horses - no adjustment
    }
    
    // Base > 13f
    if (minSPS < 2.08) return 1.0;
    if (minSPS < 2.13) return 0.5;
    if (minSPS < 2.18) return 0.25;
    return 0; // Standard horses - no adjustment
  };

  const handleImportData = () => {
    try {
      const lines = importText.trim().split('\n');
      const parsed = [];
      
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split('\t');
        if (parts.length >= 12) {
          parsed.push({
            horse: parts[1],
            distance: parseFloat(parts[3]),
            age: parseInt(parts[6]),
            spsAvg: parseFloat(parts[8]),
            slAvg: parseFloat(parts[11])
          });
        }
      }
      
      if (parsed.length > 0) {
        setCustomData(parsed);
        setView('comparison');
        setImportText('');
        alert(`Successfully imported ${parsed.length} horses!`);
      }
    } catch (error) {
      alert('Error parsing data.');
    }
  };

  const handleImportForecastData = () => {
    try {
      const lines = forecastImportText.trim().split('\n');
      const parsed = [];
      
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split('\t');
        if (parts.length >= 5) {
          parsed.push({
            horse: parts[0],
            spsAvg2yo: parseFloat(parts[1]),
            slAvg2yo: parseFloat(parts[2]),
            distance2yo: parseFloat(parts[3]),
            distance3yo: parseFloat(parts[4])
          });
        }
      }
      
      if (parsed.length > 0) {
        setForecastData(parsed);
        setView('forecast-model');
        setForecastImportText('');
        alert(`Successfully imported ${parsed.length} horses for Model B!`);
      }
    } catch (error) {
      alert('Error parsing forecast data.');
    }
  };

  const calculateModelStats = (dataset) => {
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

    const distances = dataset.map(d => d.distance);
    const spsAvgCorr = correlation(dataset.map(d => d.spsAvg), distances);
    const slAvgCorr = correlation(dataset.map(d => d.slAvg), distances);

    const X = dataset.map(d => [1, d.spsAvg, d.slAvg]);
    const y = distances;
    
    const XtX = [
      [n, X.reduce((s, row) => s + row[1], 0), X.reduce((s, row) => s + row[2], 0)],
      [X.reduce((s, row) => s + row[1], 0), X.reduce((s, row) => s + row[1] * row[1], 0), X.reduce((s, row) => s + row[1] * row[2], 0)],
      [X.reduce((s, row) => s + row[2], 0), X.reduce((s, row) => s + row[1] * row[2], 0), X.reduce((s, row) => s + row[2] * row[2], 0)]
    ];
    
    const Xty = [
      y.reduce((s, yi) => s + yi, 0),
      X.reduce((s, row, i) => s + row[1] * y[i], 0),
      X.reduce((s, row, i) => s + row[2] * y[i], 0)
    ];

    const det = XtX[0][0] * (XtX[1][1] * XtX[2][2] - XtX[1][2] * XtX[2][1]) -
                XtX[0][1] * (XtX[1][0] * XtX[2][2] - XtX[1][2] * XtX[2][0]) +
                XtX[0][2] * (XtX[1][0] * XtX[2][1] - XtX[1][1] * XtX[2][0]);

    const inv = [
      [(XtX[1][1] * XtX[2][2] - XtX[1][2] * XtX[2][1]) / det,
       -(XtX[0][1] * XtX[2][2] - XtX[0][2] * XtX[2][1]) / det,
       (XtX[0][1] * XtX[1][2] - XtX[0][2] * XtX[1][1]) / det],
      [-(XtX[1][0] * XtX[2][2] - XtX[1][2] * XtX[2][0]) / det,
       (XtX[0][0] * XtX[2][2] - XtX[0][2] * XtX[2][0]) / det,
       -(XtX[0][0] * XtX[1][2] - XtX[0][2] * XtX[1][0]) / det],
      [(XtX[1][0] * XtX[2][1] - XtX[1][1] * XtX[2][0]) / det,
       -(XtX[0][0] * XtX[2][1] - XtX[0][1] * XtX[2][0]) / det,
       (XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0]) / det]
    ];

    const coefficients = [
      inv[0][0] * Xty[0] + inv[0][1] * Xty[1] + inv[0][2] * Xty[2],
      inv[1][0] * Xty[0] + inv[1][1] * Xty[1] + inv[1][2] * Xty[2],
      inv[2][0] * Xty[0] + inv[2][1] * Xty[1] + inv[2][2] * Xty[2]
    ];

    const predictions = X.map(row => coefficients[0] + coefficients[1] * row[1] + coefficients[2] * row[2]);
    const meanY = y.reduce((a, b) => a + b) / n;
    const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - predictions[i], 2), 0);
    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0);
    const rSquared = 1 - (ssRes / ssTot);

    return { spsAvgCorr, slAvgCorr, coefficients, rSquared, n };
  };

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

    const distances3yo = dataset.map(d => d.distance3yo);
    const spsAvgCorr = correlation(dataset.map(d => d.spsAvg2yo), distances3yo);
    const slAvgCorr = correlation(dataset.map(d => d.slAvg2yo), distances3yo);
    const distance2yoCorr = correlation(dataset.map(d => d.distance2yo), distances3yo);

    const X = dataset.map(d => [1, d.spsAvg2yo, d.slAvg2yo, d.distance2yo]);
    const y = distances3yo;
    
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

    function matrixInverse4x4(m) {
      const A2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
      const A1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
      const A1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
      const A0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
      const A0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
      const A0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
      const A2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
      const A1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
      const A1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
      const A2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
      const A1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
      const A1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
      const A0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0];
      const A0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0];
      const A0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0];
      const A0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
      const A0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0];
      const A0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

      let det = m[0][0] * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223)
              - m[0][1] * (m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223)
              + m[0][2] * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123)
              - m[0][3] * (m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);

      det = 1 / det;

      return [
        [(m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223) * det,
         -(m[0][1] * A2323 - m[0][2] * A1323 + m[0][3] * A1223) * det,
         (m[0][1] * A2313 - m[0][2] * A1313 + m[0][3] * A1213) * det,
         -(m[0][1] * A2312 - m[0][2] * A1312 + m[0][3] * A1212) * det],
        [-(m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223) * det,
         (m[0][0] * A2323 - m[0][2] * A0323 + m[0][3] * A0223) * det,
         -(m[0][0] * A2313 - m[0][2] * A0313 + m[0][3] * A0213) * det,
         (m[0][0] * A2312 - m[0][2] * A0312 + m[0][3] * A0212) * det],
        [(m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123) * det,
         -(m[0][0] * A1323 - m[0][1] * A0323 + m[0][3] * A0123) * det,
         (m[0][0] * A1313 - m[0][1] * A0313 + m[0][3] * A0113) * det,
         -(m[0][0] * A1312 - m[0][1] * A0312 + m[0][3] * A0112) * det],
        [-(m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123) * det,
         (m[0][0] * A1223 - m[0][1] * A0223 + m[0][2] * A0123) * det,
         -(m[0][0] * A1213 - m[0][1] * A0213 + m[0][2] * A0113) * det,
         (m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112) * det]
      ];
    }

    const inv = matrixInverse4x4(XtX);

    const coefficients = [
      inv[0][0] * Xty[0] + inv[0][1] * Xty[1] + inv[0][2] * Xty[2] + inv[0][3] * Xty[3],
      inv[1][0] * Xty[0] + inv[1][1] * Xty[1] + inv[1][2] * Xty[2] + inv[1][3] * Xty[3],
      inv[2][0] * Xty[0] + inv[2][1] * Xty[1] + inv[2][2] * Xty[2] + inv[2][3] * Xty[3],
      inv[3][0] * Xty[0] + inv[3][1] * Xty[1] + inv[3][2] * Xty[2] + inv[3][3] * Xty[3]
    ];

    const predictions = X.map(row => 
      coefficients[0] + coefficients[1] * row[1] + coefficients[2] * row[2] + coefficients[3] * row[3]
    );
    const meanY = y.reduce((a, b) => a + b) / n;
    const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - predictions[i], 2), 0);
    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0);
    const rSquared = 1 - (ssRes / ssTot);

    return { spsAvgCorr, slAvgCorr, distance2yoCorr, coefficients, rSquared, n };
  };

  const stats3yo = useMemo(() => calculateModelStats(data3yo), [data3yo]);
  const stats4plus = useMemo(() => calculateModelStats(data4plus), [data4plus]);
  const statsForecast = useMemo(() => forecastData ? calculateForecastModelStats(forecastData) : null, [forecastData]);

  const convertToRaceDistance = (furlongs) => {
    const common = [
      { f: 5, label: '5f' },
      { f: 6, label: '6f' },
      { f: 7, label: '7f' },
      { f: 8, label: '1m' },
      { f: 9, label: '9f (1m1f)' },
      { f: 10, label: '1m2f' },
      { f: 11, label: '1m3f' },
      { f: 12, label: '1m4f (Derby)' },
      { f: 13, label: '1m5f' },
      { f: 14, label: '1m6f' },
      { f: 16, label: '2m' }
    ];
    
    let closest = common[0];
    let minDiff = Math.abs(furlongs - common[0].f);
    
    for (let i = 1; i < common.length; i++) {
      const diff = Math.abs(furlongs - common[i].f);
      if (diff < minDiff) {
        minDiff = diff;
        closest = common[i];
      }
    }
    
    return closest.label;
  };

  const residuals3yo = useMemo(() => {
    if (!stats3yo) return [];
    return data3yo.map(horse => {
      const predicted = stats3yo.coefficients[0] + 
                       stats3yo.coefficients[1] * horse.spsAvg + 
                       stats3yo.coefficients[2] * horse.slAvg;
      return {
        horse: horse.horse,
        distance: horse.distance,
        predicted,
        residual: horse.distance - predicted,
        absResidual: Math.abs(horse.distance - predicted)
      };
    }).sort((a, b) => b.absResidual - a.absResidual);
  }, [data3yo, stats3yo]);

  const residuals4plus = useMemo(() => {
    if (!stats4plus) return [];
    return data4plus.map(horse => {
      const predicted = stats4plus.coefficients[0] + 
                       stats4plus.coefficients[1] * horse.spsAvg + 
                       stats4plus.coefficients[2] * horse.slAvg;
      return {
        horse: horse.horse,
        distance: horse.distance,
        predicted,
        residual: horse.distance - predicted,
        absResidual: Math.abs(horse.distance - predicted)
      };
    }).sort((a, b) => b.absResidual - a.absResidual);
  }, [data4plus, stats4plus]);

  const residualsForecast = useMemo(() => {
    if (!statsForecast || !forecastData) return [];
    return forecastData.map(horse => {
      const predicted = statsForecast.coefficients[0] + 
                       statsForecast.coefficients[1] * horse.spsAvg2yo + 
                       statsForecast.coefficients[2] * horse.slAvg2yo +
                       statsForecast.coefficients[3] * horse.distance2yo;
      return {
        horse: horse.horse,
        distance2yo: horse.distance2yo,
        distance3yo: horse.distance3yo,
        predicted,
        residual: horse.distance3yo - predicted,
        absResidual: Math.abs(horse.distance3yo - predicted)
      };
    }).sort((a, b) => b.absResidual - a.absResidual);
  }, [forecastData, statsForecast]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">🐎 StridePredictor Model</h1>
        <p className="text-sm text-gray-600">Biomechanics-Based Distance Prediction</p>
      </div>

      <div className="flex flex-wrap gap-2 justify-center mb-6">
        <button onClick={() => setView('import')} className={`px-4 py-2 rounded-lg font-bold ${view === 'import' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600'}`}>
          Import Data
        </button>
        <button onClick={() => setView('comparison')} className={`px-4 py-2 rounded-lg font-bold ${view === 'comparison' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600'}`}>
          Model A (3yo & 4+yo)
        </button>
        <button onClick={() => setView('predictor')} className={`px-4 py-2 rounded-lg font-bold ${view === 'predictor' ? 'bg-green-600 text-white' : 'bg-white text-green-600'}`}>
          Predict (Model A)
        </button>
        <button onClick={() => setView('forecast-import')} className={`px-4 py-2 rounded-lg font-bold ${view === 'forecast-import' ? 'bg-purple-600 text-white' : 'bg-white text-purple-600'}`}>
          Import 2yo Data
        </button>
        <button onClick={() => setView('forecast-model')} className={`px-4 py-2 rounded-lg font-bold ${view === 'forecast-model' ? 'bg-purple-600 text-white' : 'bg-white text-purple-600'}`}>
          Model B (2yo→3yo)
        </button>
        <button onClick={() => setView('forecast-predictor')} className={`px-4 py-2 rounded-lg font-bold ${view === 'forecast-predictor' ? 'bg-pink-600 text-white' : 'bg-white text-pink-600'}`}>
          Predict (Model B) ⭐ NEW
        </button>
      </div>

      {view === 'import' && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-3">Import Horse Data (3yo & 4+yo)</h2>
          <p className="text-sm text-gray-600 mb-3">Paste tab-separated data from RaceIQ exports.</p>
          <textarea 
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            className="w-full h-64 px-3 py-2 border rounded text-sm font-mono"
            placeholder="Paste data here..."
          />
          <button onClick={handleImportData} className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg font-bold">
            Import Data
          </button>
        </div>
      )}

      {view === 'forecast-import' && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-3">Import 2yo → 3yo Forecast Data (Model B)</h2>
          <p className="text-sm text-gray-600 mb-3">Format: Horse Name, 2yo SPS, 2yo SL, 2yo Distance, Actual 3yo Distance</p>
          <textarea 
            value={forecastImportText}
            onChange={(e) => setForecastImportText(e.target.value)}
            className="w-full h-64 px-3 py-2 border rounded text-sm font-mono"
            placeholder="Paste forecast data here (tab-separated)..."
          />
          <button onClick={handleImportForecastData} className="mt-3 px-4 py-2 bg-purple-600 text-white rounded-lg font-bold">
            Import Forecast Data
          </button>
        </div>
      )}

      {view === 'comparison' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {(() => {
            const ageData = selectedAge === '3yo' ? data3yo : data4plus;
            const currentStats = selectedAge === '3yo' ? stats3yo : stats4plus;
            const currentResiduals = selectedAge === '3yo' ? residuals3yo : residuals4plus;

            return (
              <div className="lg:col-span-2">
                <div className="flex gap-2 mb-4 justify-center">
                  <button onClick={() => setSelectedAge('3yo')} className={`px-4 py-2 rounded-lg font-bold ${selectedAge === '3yo' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600'}`}>
                    3yo Model
                  </button>
                  <button onClick={() => setSelectedAge('4plus')} className={`px-4 py-2 rounded-lg font-bold ${selectedAge === '4plus' ? 'bg-green-600 text-white' : 'bg-white text-green-600'}`}>
                    4+yo Model
                  </button>
                </div>

                {!currentStats ? (
                  <div className="bg-white rounded-lg shadow p-8 text-center">
                    <p className="text-lg text-gray-600">No data loaded for {selectedAge}</p>
                  </div>
                ) : (
                  <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-xl font-bold mb-4 text-center">{selectedAge === '3yo' ? '3-Year-Olds' : '4+ Year-Olds'}</h2>
                    
                    <div className="bg-gradient-to-r from-blue-50 to-green-50 p-6 rounded-lg border-2 border-blue-300 mb-4">
                      <div className="text-center mb-4">
                        <div className="text-5xl font-bold text-blue-600 mb-1">{(currentStats.rSquared * 100).toFixed(1)}%</div>
                        <p className="text-sm font-semibold text-gray-700 mb-1">R² Accuracy</p>
                        <p className="text-xs text-gray-600">{currentStats.n} horses</p>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-center">
                        <div className="bg-white rounded p-2">
                          <p className="text-xs text-gray-600">SPS Correlation</p>
                          <p className="text-sm font-bold text-blue-700">{currentStats.spsAvgCorr.toFixed(3)}</p>
                        </div>
                        <div className="bg-white rounded p-2">
                          <p className="text-xs text-gray-600">SL Correlation</p>
                          <p className="text-sm font-bold text-green-700">{currentStats.slAvgCorr.toFixed(3)}</p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4 mb-4">
                      <h3 className="font-bold text-sm mb-3">🔢 Model Coefficients:</h3>
                      <div className="grid grid-cols-3 gap-3 text-xs">
                        <div className="bg-white rounded p-2">
                          <p className="text-gray-600">Intercept</p>
                          <p className="text-lg font-bold">{currentStats.coefficients[0].toFixed(4)}</p>
                        </div>
                        <div className="bg-white rounded p-2">
                          <p className="text-gray-600">Avg SPS</p>
                          <p className="text-lg font-bold text-blue-600">{currentStats.coefficients[1].toFixed(4)}</p>
                          <p className="text-xs text-gray-500">Per 0.1Hz: {(currentStats.coefficients[1] * 0.1).toFixed(2)}f</p>
                        </div>
                        <div className="bg-white rounded p-2">
                          <p className="text-gray-600">Avg SL</p>
                          <p className="text-lg font-bold text-green-600">{currentStats.coefficients[2].toFixed(4)}</p>
                          <p className="text-xs text-gray-500">Per 0.5m: {(currentStats.coefficients[2] * 0.5).toFixed(2)}f</p>
                        </div>
                      </div>
                    </div>

                    {currentResiduals.length > 0 && (
                      <div>
                        <h3 className="text-sm font-bold text-blue-700 mb-2">Top 5 Prediction Errors</h3>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs border-collapse">
                            <thead>
                              <tr className="bg-blue-100">
                                <th className="border border-blue-300 px-2 py-1 text-left">Horse</th>
                                <th className="border border-blue-300 px-2 py-1 text-right">Actual</th>
                                <th className="border border-blue-300 px-2 py-1 text-right">Predicted</th>
                                <th className="border border-blue-300 px-2 py-1 text-right">Error</th>
                              </tr>
                            </thead>
                            <tbody>
                              {currentResiduals.slice(0, 5).map((horse, idx) => (
                                <tr key={idx} className={horse.absResidual > 1 ? 'bg-red-50' : 'bg-white'}>
                                  <td className="border border-blue-200 px-2 py-1 font-medium">{horse.horse}</td>
                                  <td className="border border-blue-200 px-2 py-1 text-right">{horse.distance.toFixed(1)}f</td>
                                  <td className="border border-blue-200 px-2 py-1 text-right">{horse.predicted.toFixed(1)}f</td>
                                  <td className="border border-blue-200 px-2 py-1 text-right font-bold">
                                    {horse.residual > 0 ? '+' : ''}{horse.residual.toFixed(2)}f
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                <p className="text-xs text-gray-500 mt-2">R²: {(currentStats.rSquared * 100).toFixed(1)}%</p>
              </div>
            );
          })()}
        </div>
      )}

      {view === 'forecast-model' && (
        <div className="bg-white rounded-lg shadow p-6">
          {statsForecast ? (
            <>
              <h2 className="text-xl font-bold mb-4 text-center">Model B: 2yo → 3yo Forecast</h2>
              
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg border-2 border-purple-300 mb-4">
                <div className="text-center mb-4">
                  <div className="text-5xl font-bold text-purple-600 mb-1">{(statsForecast.rSquared * 100).toFixed(1)}%</div>
                  <p className="text-sm font-semibold text-gray-700 mb-1">R² Accuracy</p>
                  <p className="text-xs text-gray-600">{statsForecast.n} horses • With Race Context</p>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center mb-4">
                  <div className="bg-white rounded p-2">
                    <p className="text-xs text-gray-600">SPS Corr</p>
                    <p className="text-sm font-bold text-purple-700">{statsForecast.spsAvgCorr.toFixed(3)}</p>
                  </div>
                  <div className="bg-white rounded p-2">
                    <p className="text-xs text-gray-600">SL Corr</p>
                    <p className="text-sm font-bold text-purple-700">{statsForecast.slAvgCorr.toFixed(3)}</p>
                  </div>
                  <div className="bg-white rounded p-2">
                    <p className="text-xs text-gray-600">Dist Corr</p>
                    <p className="text-sm font-bold text-green-700">{statsForecast.distance2yoCorr.toFixed(3)}</p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4 mb-4">
                <h3 className="font-bold text-sm mb-3">🔢 Model B Coefficients:</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-white rounded p-2">
                    <p className="text-gray-600">Intercept</p>
                    <p className="text-lg font-bold">{statsForecast.coefficients[0].toFixed(4)}</p>
                  </div>
                  <div className="bg-white rounded p-2">
                    <p className="text-gray-600">2yo SPS</p>
                    <p className="text-lg font-bold text-blue-600">{statsForecast.coefficients[1].toFixed(4)}</p>
                    <p className="text-xs text-gray-500">Per 0.1Hz: {(statsForecast.coefficients[1] * 0.1).toFixed(2)}f</p>
                  </div>
                  <div className="bg-white rounded p-2">
                    <p className="text-gray-600">2yo SL ⭐</p>
                    <p className="text-lg font-bold text-green-600">{statsForecast.coefficients[2].toFixed(4)}</p>
                    <p className="text-xs text-gray-500">Per 0.5m: {(statsForecast.coefficients[2] * 0.5).toFixed(2)}f</p>
                  </div>
                  <div className="bg-white rounded p-2">
                    <p className="text-gray-600">2yo Distance ⭐</p>
                    <p className="text-lg font-bold text-orange-600">{statsForecast.coefficients[3].toFixed(4)}</p>
                    <p className="text-xs text-gray-500">Per furlong: {statsForecast.coefficients[3].toFixed(2)}f</p>
                  </div>
                </div>
              </div>

              {residualsForecast.length > 0 && (
                <div>
                  <h3 className="text-sm font-bold text-purple-700 mb-2">Top 5 Prediction Errors</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="bg-purple-100">
                          <th className="border border-purple-300 px-2 py-1 text-left">Horse</th>
                          <th className="border border-purple-300 px-2 py-1 text-right">2yo Dist</th>
                          <th className="border border-purple-300 px-2 py-1 text-right">Actual 3yo</th>
                          <th className="border border-purple-300 px-2 py-1 text-right">Predicted</th>
                          <th className="border border-purple-300 px-2 py-1 text-right">Error</th>
                        </tr>
                      </thead>
                      <tbody>
                        {residualsForecast.slice(0, 5).map((horse, idx) => (
                          <tr key={idx} className={horse.absResidual > 2 ? 'bg-red-50' : 'bg-white'}>
                            <td className="border border-purple-200 px-2 py-1 font-medium">{horse.horse}</td>
                            <td className="border border-purple-200 px-2 py-1 text-right">{horse.distance2yo.toFixed(1)}f</td>
                            <td className="border border-purple-200 px-2 py-1 text-right">{horse.distance3yo.toFixed(1)}f</td>
                            <td className="border border-purple-200 px-2 py-1 text-right">{horse.predicted.toFixed(1)}f</td>
                            <td className="border border-purple-200 px-2 py-1 text-right font-bold">
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
            <div className="text-center p-8">
              <p className="text-lg text-gray-600 mb-2">No Data Loaded</p>
              <p className="text-sm text-gray-500 mb-3">Import 2yo → 3yo data to see Model B</p>
              <button onClick={() => setView('forecast-import')} className="px-4 py-2 bg-purple-600 text-white rounded-lg font-bold">
                Import Data
              </button>
            </div>
          )}
        </div>
      )}

      {view === 'forecast-predictor' && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-3">🌟 Enhanced Forecast 3yo Distance (Model B)</h2>
          
          {statsForecast ? (
            <>
              {/* ENHANCED INPUT SECTION */}
              <div className="grid grid-cols-3 gap-3 mb-3">
                <div>
                  <label className="block text-xs mb-1 font-semibold">2yo Avg SPS (Hz)</label>
                  <input 
                    type="number" 
                    step="0.01" 
                    value={predictInputs.spsAvg} 
                    onChange={(e) => setPredictInputs({...predictInputs, spsAvg: parseFloat(e.target.value)})} 
                    className="w-full px-2 py-2 border rounded text-sm" 
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1 font-semibold">2yo Avg SL (m)</label>
                  <input 
                    type="number" 
                    step="0.01" 
                    value={predictInputs.slAvg} 
                    onChange={(e) => setPredictInputs({...predictInputs, slAvg: parseFloat(e.target.value)})} 
                    className="w-full px-2 py-2 border rounded text-sm" 
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1 font-semibold">2yo Race Dist (f)</label>
                  <input 
                    type="number" 
                    step="0.5" 
                    value={predictInputs.distance2yo} 
                    onChange={(e) => setPredictInputs({...predictInputs, distance2yo: parseFloat(e.target.value)})} 
                    className="w-full px-2 py-2 border rounded text-sm" 
                  />
                </div>
              </div>

              {/* NEW INPUT SECTION */}
              <div className="bg-yellow-50 border-2 border-yellow-300 rounded-lg p-3 mb-3">
                <p className="text-xs font-bold text-yellow-800 mb-2">⭐ NEW: Enhanced Analysis Inputs</p>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs mb-1 font-semibold">2yo Min SPS (Hz)</label>
                    <input 
                      type="number" 
                      step="0.01" 
                      value={predictInputs.minSPS} 
                      onChange={(e) => setPredictInputs({...predictInputs, minSPS: parseFloat(e.target.value)})} 
                      className="w-full px-2 py-2 border rounded text-sm" 
                    />
                  </div>
                  <div>
                    <label className="block text-xs mb-1 font-semibold">2yo Min SL (m)</label>
                    <input 
                      type="number" 
                      step="0.01" 
                      value={predictInputs.minSL} 
                      onChange={(e) => setPredictInputs({...predictInputs, minSL: parseFloat(e.target.value)})} 
                      className="w-full px-2 py-2 border rounded text-sm" 
                    />
                  </div>
                  <div>
                    <label className="block text-xs mb-1 font-semibold">2yo Max SL (m)</label>
                    <input 
                      type="number" 
                      step="0.01" 
                      value={predictInputs.maxSL} 
                      onChange={(e) => setPredictInputs({...predictInputs, maxSL: parseFloat(e.target.value)})} 
                      className="w-full px-2 py-2 border rounded text-sm" 
                    />
                  </div>
                </div>
              </div>

              {/* CALCULATION AND DISPLAY */}
              {(() => {
                // Base prediction (existing model)
                const basePrediction = statsForecast.coefficients[0] + 
                                     statsForecast.coefficients[1] * predictInputs.spsAvg + 
                                     statsForecast.coefficients[2] * predictInputs.slAvg +
                                     statsForecast.coefficients[3] * predictInputs.distance2yo;
                
                // NEW: Calculate stride length range
                const slRange = predictInputs.maxSL - predictInputs.minSL;
                
                // NEW: Get confidence rating
                const confidence = getConfidenceRating(slRange, basePrediction);
                
                // NEW: Get stamina profile
                const staminaProfile = getStaminaProfile(predictInputs.minSPS);
                
                // NEW: Calculate Min SPS adjustment
                const minSPSAdjustment = getMinSPSAdjustment(basePrediction, predictInputs.minSPS);
                
                // NEW: Final adjusted prediction
                const adjustedPrediction = basePrediction + minSPSAdjustment;
                
                return (
                  <>
                    {/* BASE PREDICTION */}
                    <div className="bg-gray-50 p-4 rounded-lg mb-3 border-2 border-gray-300">
                      <p className="text-xs text-gray-600 mb-1">BASE PREDICTION (Model B)</p>
                      <p className="text-2xl font-bold text-gray-700">
                        {basePrediction.toFixed(1)}f
                      </p>
                      <p className="text-sm text-gray-600">
                        {convertToRaceDistance(basePrediction)}
                      </p>
                    </div>

                    {/* DATA QUALITY */}
                    <div className={`bg-${confidence.color}-50 p-4 rounded-lg mb-3 border-2 border-${confidence.color}-300`}>
                      <p className="text-xs font-bold mb-1">📊 DATA QUALITY</p>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-lg font-bold">
                            {'⭐'.repeat(confidence.stars)}
                          </p>
                          <p className="text-sm font-semibold">{confidence.label} Confidence</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-600">SL Range</p>
                          <p className="text-lg font-bold">{slRange.toFixed(2)}M</p>
                        </div>
                      </div>
                    </div>

                    {/* STAMINA PROFILE */}
                    <div className={`bg-${staminaProfile.color}-50 p-4 rounded-lg mb-3 border-2 border-${staminaProfile.color}-300`}>
                      <p className="text-xs font-bold mb-1">🔥 STAMINA PROFILE</p>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-2xl mb-1">{staminaProfile.icon}</p>
                          <p className="text-sm font-semibold">{staminaProfile.level}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-600">Min SPS</p>
                          <p className="text-lg font-bold">{predictInputs.minSPS.toFixed(2)}</p>
                        </div>
                      </div>
                      {minSPSAdjustment > 0 && (
                        <p className="text-xs text-gray-600 mt-2">
                          Stamina Adjustment: +{minSPSAdjustment.toFixed(1)}f
                        </p>
                      )}
                    </div>

                    {/* FINAL ADJUSTED PREDICTION */}
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg text-center border-2 border-purple-300">
                      <p className="text-xs text-gray-600 mb-1">⭐ FINAL ADJUSTED PREDICTION ⭐</p>
                      <p className="text-xs text-gray-500 mb-2">(Base + Min SPS Adjustment)</p>
                      <p className="text-5xl font-bold text-purple-600 mb-1">
                        {adjustedPrediction.toFixed(1)}f
                      </p>
                      <p className="text-2xl text-gray-700 font-semibold mb-2">
                        {convertToRaceDistance(adjustedPrediction)}
                      </p>
                      <div className="grid grid-cols-2 gap-2 text-xs bg-white rounded p-2">
                        <div>
                          <p className="text-gray-600">Expected Range</p>
                          <p className="font-bold">{(adjustedPrediction - 1).toFixed(1)}f - {(adjustedPrediction + 1).toFixed(1)}f</p>
                        </div>
                        <div>
                          <p className="text-gray-600">Confidence</p>
                          <p className="font-bold">{'⭐'.repeat(confidence.stars)}</p>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        Model R²: {(statsForecast.rSquared * 100).toFixed(1)}% • {statsForecast.n} horses
                      </p>
                    </div>
                  </>
                );
              })()}
            </>
          ) : (
            <div className="text-center p-6">
              <p className="text-sm text-gray-600 mb-3">No Model B data loaded</p>
              <button onClick={() => setView('forecast-import')} className="px-4 py-2 bg-purple-600 text-white rounded-lg font-bold text-sm">
                Import Data
              </button>
            </div>
          )}
        </div>
      )}

      {view === 'predictor' && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-3">Predict Current Optimal Distance (Model A)</h2>
          
          <div className="mb-3">
            <label className="block text-xs mb-1">Age Group</label>
            <select value={predictInputs.ageTarget} onChange={(e) => setPredictInputs({...predictInputs, ageTarget: parseInt(e.target.value)})} className="w-full px-2 py-2 border rounded text-sm">
              <option value={3}>3-Year-Old</option>
              <option value={4}>4+ Year-Old</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <label className="block text-xs mb-1">Avg SPS (Hz)</label>
              <input type="number" step="0.01" value={predictInputs.spsAvg} onChange={(e) => setPredictInputs({...predictInputs, spsAvg: parseFloat(e.target.value)})} className="w-full px-2 py-2 border rounded text-sm" />
            </div>
            <div>
              <label className="block text-xs mb-1">Avg SL (m)</label>
              <input type="number" step="0.01" value={predictInputs.slAvg} onChange={(e) => setPredictInputs({...predictInputs, slAvg: parseFloat(e.target.value)})} className="w-full px-2 py-2 border rounded text-sm" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-green-50 p-6 rounded-lg text-center border-2 border-blue-300">
            <p className="text-xs text-gray-600 mb-1">PREDICTED OPTIMAL DISTANCE</p>
            <p className="text-xs text-gray-500 mb-2">(Model A: Current Capability)</p>
            <p className="text-4xl font-bold text-blue-600 mb-1">
              {(() => {
                const currentStats = predictInputs.ageTarget === 3 ? stats3yo : stats4plus;
                if (!currentStats) return 'N/A';
                const prediction = currentStats.coefficients[0] + 
                                 currentStats.coefficients[1] * predictInputs.spsAvg + 
                                 currentStats.coefficients[2] * predictInputs.slAvg;
                return prediction.toFixed(1);
              })()}f
            </p>
            <p className="text-xl text-gray-700 font-semibold">
              {(() => {
                const currentStats = predictInputs.ageTarget === 3 ? stats3yo : stats4plus;
                if (!currentStats) return 'N/A';
                const prediction = currentStats.coefficients[0] + 
                                 currentStats.coefficients[1] * predictInputs.spsAvg + 
                                 currentStats.coefficients[2] * predictInputs.slAvg;
                return convertToRaceDistance(prediction);
              })()}
            </p>
            <p className="text-xs text-gray-500 mt-2">
              {(() => {
                const currentStats = predictInputs.ageTarget === 3 ? stats3yo : stats4plus;
                return currentStats ? `R²: ${(currentStats.rSquared * 100).toFixed(1)}%` : '';
              })()}
            </p>
          </div>
        </div>
      )}
    </div>
  </div>
  );
};
