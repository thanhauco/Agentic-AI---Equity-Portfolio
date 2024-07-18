"""
Transformer-based Price Predictor for AlphaAgents.

Implements a Transformer Encoder architecture for time-series forecasting
using multi-head self-attention mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

def _get_tf_components():
    """Lazy load TensorFlow components."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization, 
            MultiHeadAttention, GlobalAveragePooling1D
        )
        return tf, Model, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    except ImportError:
        logger.warning("TensorFlow not installed. Transformer features will be disabled.")
        return None, None, None, None, None, None, None, None

class PositionalEncoding:
    """Sinusoidal positional encoding for sequence data."""
    
    def __init__(self, d_model: int, max_len: int = 500):
        self.d_model = d_model
        self.max_len = max_len
        self._encoding = None
        
    def get_encoding(self, seq_len: int) -> np.ndarray:
        if self._encoding is None or seq_len > self._encoding.shape[0]:
            position = np.arange(self.max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
            
            pe = np.zeros((self.max_len, self.d_model))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            self._encoding = pe
            
        return self._encoding[:seq_len]

class TransformerPredictor:
    """
    Transformer Encoder for stock price prediction.
    
    Uses multi-head self-attention to capture long-range dependencies
    in time-series data without recurrence.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.pos_encoding = PositionalEncoding(d_model)
        
    def _build_model(self, seq_len: int, n_features: int):
        """Build the Transformer Encoder model."""
        tf, Model, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D = _get_tf_components()
        
        if tf is None:
            return None
            
        inputs = Input(shape=(seq_len, n_features))
        
        # Project to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_enc = self.pos_encoding.get_encoding(seq_len)
        x = x + pos_enc
        
        # Transformer Encoder blocks
        for _ in range(self.n_layers):
            # Multi-Head Self-Attention
            attn_output = MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads
            )(x, x)
            attn_output = Dropout(self.dropout_rate)(attn_output)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed-Forward Network
            ff_output = Dense(self.ff_dim, activation='relu')(x)
            ff_output = Dense(self.d_model)(ff_output)
            ff_output = Dropout(self.dropout_rate)(ff_output)
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # Global average pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train_and_predict(
        self, 
        df: pd.DataFrame, 
        window_size: int = 30,
        forecast_days: int = 5,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Train the Transformer and predict future prices.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Number of days to look back
            forecast_days: Number of days to forecast
            epochs: Training epochs
        """
        tf, *_ = _get_tf_components()
        if tf is None:
            return {"error": "TensorFlow not available"}
            
        try:
            # Prepare features: Close, Volume, High-Low range
            features = df[['Close', 'Volume', 'High', 'Low']].copy()
            features['Range'] = features['High'] - features['Low']
            features = features[['Close', 'Volume', 'Range']]
            
            data = features.values
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(window_size, len(scaled_data)):
                X.append(scaled_data[i-window_size:i])
                y.append(scaled_data[i, 0])  # Predict Close price
                
            X, y = np.array(X), np.array(y)
            
            # Build and train model
            self.model = self._build_model(window_size, X.shape[2])
            if self.model is None:
                return {"error": "Failed to build model"}
                
            self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
            
            # Forecast
            last_window = scaled_data[-window_size:]
            predictions = []
            current_window = last_window.copy()
            
            for _ in range(forecast_days):
                input_seq = current_window.reshape(1, window_size, -1)
                pred = self.model.predict(input_seq, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Shift window and add prediction
                new_row = np.array([[pred, current_window[-1, 1], current_window[-1, 2]]])
                current_window = np.vstack([current_window[1:], new_row])
            
            # Inverse transform predictions
            pred_full = np.zeros((len(predictions), 3))
            pred_full[:, 0] = predictions
            predicted_prices = self.scaler.inverse_transform(pred_full)[:, 0]
            
            # Calculate attention weights for interpretability
            return {
                "forecast": predicted_prices.tolist(),
                "model_type": "Transformer Encoder",
                "architecture": {
                    "d_model": self.d_model,
                    "n_heads": self.n_heads,
                    "n_layers": self.n_layers,
                    "attention_mechanism": "Multi-Head Self-Attention"
                },
                "training_epochs": epochs,
                "window_size": window_size
            }
            
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return {"error": str(e)}

class GRUPredictor:
    """
    GRU (Gated Recurrent Unit) model for price prediction.
    
    Lighter alternative to LSTM with similar performance.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def train_and_predict(
        self, 
        df: pd.DataFrame,
        window_size: int = 60,
        forecast_days: int = 5
    ) -> Dict[str, Any]:
        """Train GRU and predict future prices."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GRU, Dense, Dropout
        except ImportError:
            return {"error": "TensorFlow not available"}
            
        try:
            data = df['Close'].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(window_size, len(scaled_data)):
                X.append(scaled_data[i-window_size:i, 0])
                y.append(scaled_data[i, 0])
                
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Build GRU model
            model = Sequential([
                GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                GRU(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mse')
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
                "model_type": "GRU (Gated Recurrent Unit)",
                "architecture": {
                    "layers": 2,
                    "units": 50,
                    "dropout": 0.2
                }
            }
            
        except Exception as e:
            logger.error(f"GRU prediction error: {e}")
            return {"error": str(e)}
