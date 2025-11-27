import React, { useState } from 'react';

export default () => {
  const [view, setView] = useState('predict');
  const [predictInputs, setPredictInputs] = useState({
    spsAvg: 2.30,
    slAvg: 7.5,
    distance2yo: 7.0,
    minSPS: 2.18,
    minSL: 7.2,
    maxSL: 7.8
  });

  // Model B coefficients (from your trained model)
  const modelCoefficients = {
    intercept: 42.7553,
    spsAvg: -14.5569,
    slAvg: -0.8917,
    distance2yo: 0.8062
  };
  
  const modelRSquared = 0.775; // Your Model B R²
  const modelN = 159; // Number of horses in training set

  // Helper function to convert furlongs to race distance
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

  // Calculate confidence rating based on stride length range
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

  // Calculate Min SPS adjustment
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-4">
      <div className="max-w-2xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            2yo → 3yo Distance Predictor
          </h1>
          <p className="text-sm text-gray-600">
            Predict a horse's optimal 3-year-old distance based on 2-year-old stride data
          </p>
        </div>

        {/* Navigation Buttons */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          <button
            onClick={() => setView('stats')}
            className={`py-3 px-4 rounded-lg font-bold text-sm transition ${
              view === 'stats' 
                ? 'bg-blue-600 text-white shadow-lg' 
                : 'bg-white text-gray-700 border-2 border-gray-300 hover:border-blue-300'
            }`}
          >
            📊 Model Stats
          </button>
          <button
            onClick={() => setView('predict')}
            className={`py-3 px-4 rounded-lg font-bold text-sm transition ${
              view === 'predict' 
                ? 'bg-purple-600 text-white shadow-lg' 
                : 'bg-white text-gray-700 border-2 border-gray-300 hover:border-purple-300'
            }`}
          >
            🎯 Predict Distance
          </button>
        </div>

        {/* Model Stats View */}
        {view === 'stats' && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold mb-4 text-gray-800">Model Performance</h2>
            
            <div className="bg-gradient-to-r from-blue-50 to-green-50 p-6 rounded-lg text-center border-2 border-blue-300">
              <p className="text-sm text-gray-600 mb-2">Model Accuracy</p>
              <p className="text-6xl font-bold text-blue-600 mb-2">
                {(modelRSquared * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-gray-600">
                R-squared (variance explained)
              </p>
            </div>

            <div className="mt-4 text-center text-sm text-gray-600">
              <p>Trained on {modelN} horses</p>
              <p className="mt-2">This model predicts 3-year-old optimal distance from 2-year-old stride biomechanics</p>
            </div>
          </div>
        )}

        {/* Predict Distance View */}
        {view === 'predict' && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold mb-4 text-gray-800">Enter 2-Year-Old Stride Data</h2>
            
            {/* Input Fields */}
            <div className="space-y-4 mb-6">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-semibold mb-1 text-gray-700">
                    Average SPS (Hz)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.spsAvg}
                    onChange={(e) => setPredictInputs({...predictInputs, spsAvg: parseFloat(e.target.value) || 0})}
                    className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg text-sm focus:border-purple-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1 text-gray-700">
                    Average SL (m)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.slAvg}
                    onChange={(e) => setPredictInputs({...predictInputs, slAvg: parseFloat(e.target.value) || 0})}
                    className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg text-sm focus:border-purple-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold mb-1 text-gray-700">
                  2yo Optimal Distance (furlongs)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={predictInputs.distance2yo}
                  onChange={(e) => setPredictInputs({...predictInputs, distance2yo: parseFloat(e.target.value) || 0})}
                  className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg text-sm focus:border-purple-500 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-sm font-semibold mb-1 text-gray-700">
                    Min SPS (Hz)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.minSPS}
                    onChange={(e) => setPredictInputs({...predictInputs, minSPS: parseFloat(e.target.value) || 0})}
                    className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg text-sm focus:border-purple-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1 text-gray-700">
                    Min SL (m)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.minSL}
                    onChange={(e) => setPredictInputs({...predictInputs, minSL: parseFloat(e.target.value) || 0})}
                    className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg text-sm focus:border-purple-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1 text-gray-700">
                    Max SL (m)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={predictInputs.maxSL}
                    onChange={(e) => setPredictInputs({...predictInputs, maxSL: parseFloat(e.target.value) || 0})}
                    className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg text-sm focus:border-purple-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>

            {/* Prediction Output */}
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg text-center border-2 border-purple-400">
              <p className="text-sm text-gray-600 mb-1">Predicted 3-Year-Old Optimal Distance</p>
              <p className="text-6xl font-bold text-purple-600 mb-2">
                {finalPrediction.toFixed(1)}f
              </p>
              <p className="text-2xl text-gray-700 font-semibold mb-4">
                {convertToRaceDistance(finalPrediction)}
              </p>
              
              <div className="text-center text-sm bg-white rounded-lg p-3">
                <p className="text-gray-600 mb-1">Prediction Confidence</p>
                <p className="font-bold text-gray-800">
                  {'⭐'.repeat(confidence.stars)}
                </p>
                <p className="text-xs text-gray-500">{confidence.label}</p>
              </div>
            </div>

            <div className="mt-4 text-center text-xs text-gray-500">
              <p>Model accuracy: {(modelRSquared * 100).toFixed(1)}% R² • Based on {modelN} horses</p>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-6 text-xs text-gray-500">
          <p>StridePredictor.com • Stride Biomechanics Analysis</p>
        </div>

      </div>
    </div>
  );
};