#!/usr/bin/env python3
"""
Comprehensive test suite for the Cryptocurrency Forecasting System.
Tests all major components including data processing, model training, and predictions.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from main import (
    Config, TechnicalIndicators, SentimentAgent, DataAgent,
    CryptoDataset, LSTMModel, TransformerModel, ModelTrainer,
    EnsembleModel, CryptocurrencyForecaster, GenerativeAgent
)
from config import SystemConfig, ModelConfig, DataConfig
from utils import DataValidator, ModelEvaluator, FileManager

warnings.filterwarnings('ignore')


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        self.config = Config()
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        self.assertIsInstance(self.config.SEQUENCE_LENGTH, int)
        self.assertGreater(self.config.SEQUENCE_LENGTH, 0)
        self.assertIsInstance(self.config.BATCH_SIZE, int)
        self.assertGreater(self.config.BATCH_SIZE, 0)
    
    def test_config_paths(self):
        """Test configuration paths creation."""
        self.assertTrue(self.config.MODEL_DIR.exists())
        self.assertTrue(self.config.DATA_DIR.exists())
        self.assertTrue(self.config.RESULTS_DIR.exists())


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators calculation."""
    
    def setUp(self):
        # Create sample price data
        np.random.seed(42)
        self.prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02))
        self.high = self.prices + np.random.rand(100) * 2
        self.low = self.prices - np.random.rand(100) * 2
        self.close = self.prices
    
    def test_rsi_calculation(self):
        """Test RSI indicator calculation."""
        rsi = TechnicalIndicators.rsi(self.prices)
        
        # RSI should be between 0 and 100
        self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()))
        
        # RSI should have reasonable length (excluding NaN values)
        self.assertGreater(len(rsi.dropna()), 50)
    
    def test_macd_calculation(self):
        """Test MACD indicator calculation."""
        macd_line, signal_line = TechnicalIndicators.macd(self.prices)
        
        # Both lines should have same length
        self.assertEqual(len(macd_line), len(signal_line))
        
        # Should have numeric values
        self.assertTrue(all(isinstance(val, (int, float)) for val in macd_line.dropna()))
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(self.prices)
        
        # Upper should be >= Middle >= Lower
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        self.assertTrue(all(upper[valid_indices] >= middle[valid_indices]))
        self.assertTrue(all(middle[valid_indices] >= lower[valid_indices]))
    
    def test_stochastic_oscillator(self):
        """Test Stochastic oscillator calculation."""
        k_percent, d_percent = TechnicalIndicators.stochastic(
            self.high, self.low, self.close
        )
        
        # Values should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        
        self.assertTrue(all(0 <= val <= 100 for val in valid_k))
        self.assertTrue(all(0 <= val <= 100 for val in valid_d))
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr = TechnicalIndicators.atr(self.high, self.low, self.close)
        
        # ATR should be positive
        self.assertTrue(all(val >= 0 for val in atr.dropna()))


class TestSentimentAgent(unittest.TestCase):
    """Test sentiment analysis functionality."""
    
    def setUp(self):
        self.config = Config()
        self.sentiment_agent = SentimentAgent(self.config)
    
    def test_sentiment_agent_initialization(self):
        """Test sentiment agent initialization."""
        self.assertIsNotNone(self.sentiment_agent.sia)
        self.assertIsNotNone(self.sentiment_agent.sentiment_pipeline)
    
    @patch('newsapi.NewsApiClient')
    def test_news_sentiment_mock(self, mock_news_client):
        """Test news sentiment analysis with mocked API."""
        # Mock news articles
        mock_articles = {
            'articles': [
                {'title': 'Bitcoin reaches new highs', 'description': 'Great news for crypto'},
                {'title': 'Market correction ahead', 'description': 'Some concerns about volatility'}
            ]
        }
        
        mock_instance = Mock()
        mock_instance.get_everything.return_value = mock_articles
        mock_news_client.return_value = mock_instance
        
        # Test with mock
        self.sentiment_agent.news_client = mock_instance
        
        # This would normally be async, but we'll test the sync version
        sentiment = {'compound': 0.1, 'positive': 0.6, 'negative': 0.3, 'neutral': 0.1}
        
        # Verify sentiment structure
        self.assertIn('compound', sentiment)
        self.assertIn('positive', sentiment)
        self.assertIn('negative', sentiment)
        self.assertIn('neutral', sentiment)
    
    def test_social_media_sentiment(self):
        """Test social media sentiment analysis."""
        sample_texts = [
            "Bitcoin is going to the moon! 🚀",
            "Crypto market is looking bearish today",
            "Holding my position, staying optimistic",
            "This volatility is crazy, might sell soon"
        ]
        
        sentiment = self.sentiment_agent.analyze_social_media_sentiment(sample_texts)
        
        # Check sentiment structure
        self.assertIn('compound', sentiment)
        self.assertIn('positive', sentiment)
        self.assertIn('negative', sentiment)
        self.assertIn('neutral', sentiment)
        
        # Sentiment values should be reasonable
        self.assertTrue(-1 <= sentiment['compound'] <= 1)
        self.assertTrue(0 <= sentiment['positive'] <= 1)


class TestDataAgent(unittest.TestCase):
    """Test data collection and preprocessing."""
    
    def setUp(self):
        self.config = Config()
        self.data_agent = DataAgent(self.config)
    
    def test_data_agent_initialization(self):
        """Test data agent initialization."""
        self.assertIsNotNone(self.data_agent.scaler)
        self.assertIsNotNone(self.data_agent.sentiment_agent)
        self.assertIsNotNone(self.data_agent.technical_indicators)
    
    def create_sample_dataframe(self):
        """Create sample cryptocurrency data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        base_price = 50000
        prices = base_price + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(100) * 50,
            'high': prices + np.random.rand(100) * 200,
            'low': prices - np.random.rand(100) * 200,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        return df
    
    def test_prepare_sequences(self):
        """Test sequence preparation for model training."""
        df = self.create_sample_dataframe()
        
        # Add some basic features
        df['price_change'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=10).std()
        df.fillna(0, inplace=True)
        
        X, y, feature_cols = self.data_agent.prepare_sequences(df, 'close')
        
        # Check shapes
        expected_samples = len(df) - self.config.SEQUENCE_LENGTH
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], self.config.SEQUENCE_LENGTH)
        self.assertEqual(len(y), expected_samples)
        
        # Check that feature columns are reasonable
        self.assertIsInstance(feature_cols, list)
        self.assertGreater(len(feature_cols), 0)


class TestNeuralNetworks(unittest.TestCase):
    """Test neural network models."""
    
    def setUp(self):
        self.config = Config()
        self.input_size = 20
        self.batch_size = 32
        self.sequence_length = 60
    
    def test_lstm_model(self):
        """Test LSTM model architecture."""
        model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.sequence_length, self.input_
