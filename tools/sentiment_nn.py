"""
Neural Sentiment Classifier for AlphaAgents.

Implements deep learning models for financial news sentiment analysis,
including a simplified BERT-style classifier and CNN-based text classifier.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
import re

# Keyword-based sentiment as fallback
BULLISH_KEYWORDS = [
    'surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'beat', 'upgrade',
    'outperform', 'buy', 'bullish', 'growth', 'profit', 'strong', 'record',
    'breakthrough', 'innovation', 'partnership', 'acquisition', 'dividend'
]

BEARISH_KEYWORDS = [
    'fall', 'drop', 'decline', 'plunge', 'crash', 'loss', 'miss', 'downgrade',
    'underperform', 'sell', 'bearish', 'weak', 'cut', 'layoff', 'lawsuit',
    'investigation', 'scandal', 'debt', 'bankruptcy', 'warning'
]

def _get_transformers():
    """Lazy load HuggingFace transformers."""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        return pipeline, AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        logger.warning("Transformers library not installed. Using fallback sentiment.")
        return None, None, None

class FinancialSentimentClassifier:
    """
    Neural network-based sentiment classifier for financial text.
    
    Uses pre-trained FinBERT or falls back to keyword-based analysis.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.pipeline = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of the NLP pipeline."""
        if self._initialized:
            return
            
        pipeline, _, _ = _get_transformers()
        if pipeline is not None:
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    truncation=True,
                    max_length=512
                )
                logger.info(f"Loaded {self.model_name} for sentiment analysis")
            except Exception as e:
                logger.warning(f"Failed to load {self.model_name}: {e}. Using fallback.")
                self.pipeline = None
                
        self._initialized = True
        
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Dictionary with label, score, and confidence
        """
        self._initialize()
        
        if self.pipeline is not None:
            try:
                result = self.pipeline(text[:512])[0]
                label = result['label'].lower()
                
                # Normalize FinBERT labels
                if label in ['positive', 'bullish']:
                    sentiment = 'bullish'
                elif label in ['negative', 'bearish']:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
                    
                return {
                    "sentiment": sentiment,
                    "confidence": result['score'],
                    "model": self.model_name,
                    "method": "transformer"
                }
            except Exception as e:
                logger.warning(f"Transformer inference failed: {e}")
                
        # Fallback to keyword-based
        return self._keyword_analysis(text)
    
    def _keyword_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based sentiment analysis."""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return {"sentiment": "neutral", "confidence": 0.5, "model": "keyword", "method": "rule-based"}
            
        if bullish_count > bearish_count:
            confidence = min(0.9, 0.5 + (bullish_count - bearish_count) * 0.1)
            return {"sentiment": "bullish", "confidence": confidence, "model": "keyword", "method": "rule-based"}
        elif bearish_count > bullish_count:
            confidence = min(0.9, 0.5 + (bearish_count - bullish_count) * 0.1)
            return {"sentiment": "bearish", "confidence": confidence, "model": "keyword", "method": "rule-based"}
        else:
            return {"sentiment": "neutral", "confidence": 0.5, "model": "keyword", "method": "rule-based"}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch."""
        return [self.analyze(text) for text in texts]
    
    def aggregate_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Aggregate sentiment across multiple texts.
        
        Returns weighted average sentiment score.
        """
        results = self.analyze_batch(texts)
        
        if not results:
            return {"aggregate_sentiment": "neutral", "score": 0.5, "sample_size": 0}
            
        bullish_score = 0
        bearish_score = 0
        neutral_score = 0
        
        for r in results:
            conf = r['confidence']
            if r['sentiment'] == 'bullish':
                bullish_score += conf
            elif r['sentiment'] == 'bearish':
                bearish_score += conf
            else:
                neutral_score += conf
                
        total = len(results)
        
        # Normalize
        bullish_pct = bullish_score / total
        bearish_pct = bearish_score / total
        
        # Net sentiment score: 1 = fully bullish, 0 = fully bearish, 0.5 = neutral
        net_score = 0.5 + (bullish_pct - bearish_pct) / 2
        
        if net_score > 0.6:
            aggregate = "bullish"
        elif net_score < 0.4:
            aggregate = "bearish"
        else:
            aggregate = "neutral"
            
        return {
            "aggregate_sentiment": aggregate,
            "score": net_score,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "sample_size": total,
            "model": results[0]['model'] if results else "unknown"
        }

class CNNTextClassifier:
    """
    Convolutional Neural Network for text classification.
    
    Uses 1D convolutions over word embeddings for sentiment detection.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, max_len: int = 200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.model = None
        self.tokenizer = None
        
    def _build_model(self):
        """Build the CNN text classifier."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
            )
            from tensorflow.keras.preprocessing.text import Tokenizer
        except ImportError:
            return None, None
            
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # 3 classes: bullish, neutral, bearish
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        
        return model, tokenizer
    
    def get_architecture(self) -> Dict[str, Any]:
        """Return model architecture details."""
        return {
            "model_type": "CNN Text Classifier",
            "layers": [
                "Embedding (10000 vocab, 128 dim)",
                "Conv1D (128 filters, kernel=5)",
                "GlobalMaxPooling1D",
                "Dense (64, ReLU)",
                "Dropout (0.5)",
                "Dense (3, Softmax)"
            ],
            "output_classes": ["bearish", "neutral", "bullish"]
        }
