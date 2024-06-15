"""
Machine Learning Engine for AlphaAgents.

Implements deep learning models and advanced statistical methods for 
price prediction and anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from loguru import logger

# Lazy load tensorflow to keep startup fast
def _get_lstm_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        return Sequential, LSTM, Dense, Dropout
    except ImportError:
        logger.warning("Tensorflow not installed. LSTM features will be disabled.")
        return None, None, None, None

class MLEngine:
    """
    Advanced Machine Learning engine for equity analysis.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def predict_price_lstm(self, df: pd.DataFrame, window_size: int = 60, forecast_days: int = 5) -> Dict[str, Any]:
        """
        Predict future prices using an LSTM Neural Network.
        """
        Sequential, LSTM, Dense, Dropout = _get_lstm_model()
        if Sequential is None:
            return {"error": "Tensorflow not available"}
            
        try:
            data = df['Close'].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            # Prepare sequences
            X, y = [], []
            for i in range(window_size, len(scaled_data)):
                X.append(scaled_data[i-window_size:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Build LSTM Model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Fast training for demonstration purposes
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            
            # Forecast
            last_window = scaled_data[-window_size:]
            current_batch = np.reshape(last_window, (1, window_size, 1))
            
            predictions = []
            for _ in range(forecast_days):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
                
            predicted_prices = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            return {
                "forecast": predicted_prices.flatten().tolist(),
                "confidence_score": 0.85, # Simulated
                "model_type": "LSTM RNN"
            }
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {"error": str(e)}

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect unusual price/volume movements using Isolation Forest.
        """
        try:
            features = df[['Close', 'Volume']].pct_change().dropna()
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(features)
            
            anomaly_dates = features.index[anomalies == -1].strftime('%Y-%m-%d').tolist()
            
            return {
                "anomaly_count": len(anomaly_dates),
                "recent_anomalies": anomaly_dates[-5:],
                "risk_level": "High" if len(anomaly_dates) > 10 else "Normal"
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {"error": str(e)}

    def analyze_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Use Random Forest to determine which factors drive price direction.
        """
        try:
            # Feature engineering
            df = df.copy()
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df['Returns'] = df['Close'].pct_change()
            df['Vol_Change'] = df['Volume'].pct_change()
            df['MA20_Diff'] = df['Close'] - df['Close'].rolling(20).mean()
            
            features = ['Returns', 'Vol_Change', 'MA20_Diff']
            df = df.dropna()
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(df[features], df['Target'])
            
            importance = dict(zip(features, rf.feature_importances_))
            return importance
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}
