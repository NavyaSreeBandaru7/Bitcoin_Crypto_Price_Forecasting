import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, AutoModel, pipeline,
    GPT2LMHeadModel, GPT2Tokenizer
)
import openai
from newsapi import NewsApiClient
import tweepy
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_forecast.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration management with environment variable support."""
    
    def __init__(self):
        # Model parameters
        self.SEQUENCE_LENGTH = 60
        self.BATCH_SIZE = 64
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.HIDDEN_SIZE = 128
        self.NUM_LAYERS = 3
        self.DROPOUT = 0.2
        
        # Data parameters
        self.TRAIN_SPLIT = 0.8
        self.VAL_SPLIT = 0.1
        self.TEST_SPLIT = 0.1
        
        # Feature engineering
        self.TECHNICAL_INDICATORS = True
        self.SENTIMENT_ANALYSIS = True
        self.MARKET_REGIME = True
        
        # API Keys (set via environment variables)
        self.NEWS_API_KEY = os.getenv('NEWS_API_KEY')
        self.TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        # Model paths
        self.MODEL_DIR = Path('models')
        self.DATA_DIR = Path('data')
        self.RESULTS_DIR = Path('results')
        
        # Create directories
        for dir_path in [self.MODEL_DIR, self.DATA_DIR, self.RESULTS_DIR]:
            dir_path.mkdir(exist_ok=True)


class TechnicalIndicators:
    """Advanced technical analysis indicators for feature engineering."""
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, ma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()


class SentimentAgent:
    """Intelligent agent for cryptocurrency sentiment analysis from multiple sources."""
    
    def __init__(self, config: Config):
        self.config = config
        self.sia = SentimentIntensityAnalyzer()
        self.news_client = None
        self.twitter_client = None
        
        # Initialize sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            return_all_scores=True
        )
        
        # Initialize API clients if keys are available
        if config.NEWS_API_KEY:
            self.news_client = NewsApiClient(api_key=config.NEWS_API_KEY)
        
        logger.info("Sentiment Agent initialized")
    
    async def get_news_sentiment(self, cryptocurrency: str = "bitcoin", days: int = 7) -> Dict:
        """Fetch and analyze news sentiment."""
        if not self.news_client:
            logger.warning("News API key not configured")
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Fetch news
            articles = self.news_client.get_everything(
                q=f"{cryptocurrency} OR crypto OR cryptocurrency",
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            sentiments = []
            for article in articles.get('articles', []):
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    sentiment = self.sia.polarity_scores(text)
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = {
                    'compound': np.mean([s['compound'] for s in sentiments]),
                    'positive': np.mean([s['pos'] for s in sentiments]),
                    'negative': np.mean([s['neg'] for s in sentiments]),
                    'neutral': np.mean([s['neu'] for s in sentiments])
                }
            else:
                avg_sentiment = {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
    
    def analyze_social_media_sentiment(self, text_data: List[str]) -> Dict:
        """Analyze sentiment from social media text data."""
        if not text_data:
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
        
        sentiments = []
        for text in text_data:
            if text.strip():
                sentiment = self.sia.polarity_scores(text)
                sentiments.append(sentiment)
        
        if sentiments:
            return {
                'compound': np.mean([s['compound'] for s in sentiments]),
                'positive': np.mean([s['pos'] for s in sentiments]),
                'negative': np.mean([s['neg'] for s in sentiments]),
                'neutral': np.mean([s['neu'] for s in sentiments])
            }
        
        return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}


class DataAgent:
    """Intelligent data collection and preprocessing agent."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.sentiment_agent = SentimentAgent(config)
        self.technical_indicators = TechnicalIndicators()
        
    async def fetch_cryptocurrency_data(self, symbol: str = "BTC-USD", period: str = "5y") -> pd.DataFrame:
        """Fetch cryptocurrency data with comprehensive feature engineering."""
        try:
            logger.info(f"Fetching data for {symbol}")
            
            # Fetch price data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            
            # Basic price features
            df['price_change'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_return'].rolling(window=30).std()
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Technical indicators
            if self.config.TECHNICAL_INDICATORS:
                df = self._add_technical_indicators(df)
            
            # Market regime detection
            if self.config.MARKET_REGIME:
                df = self._detect_market_regime(df)
            
            # Sentiment analysis
            if self.config.SENTIMENT_ANALYSIS:
                df = await self._add_sentiment_features(df, symbol)
            
            # Remove NaN values
            df.dropna(inplace=True)
            
            logger.info(f"Data fetching completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching cryptocurrency data: {e}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        # RSI
        df['rsi'] = self.technical_indicators.rsi(df['close'])
        
        # MACD
        macd_line, signal_line = self.technical_indicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = macd_line - signal_line
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_upper - bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        k_percent, d_percent = self.technical_indicators.stochastic(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        # ATR
        df['atr'] = self.technical_indicators.atr(df['high'], df['low'], df['close'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Price relative to moving averages
        df['price_vs_sma20'] = df['close'] / df['sma_20']
        df['price_vs_sma50'] = df['close'] / df['sma_50']
        
        return df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using statistical methods."""
        # Volatility regime
        df['vol_regime'] = np.where(
            df['volatility'] > df['volatility'].rolling(window=252).quantile(0.7),
            1,  # High volatility
            np.where(
                df['volatility'] < df['volatility'].rolling(window=252).quantile(0.3),
                -1,  # Low volatility
                0   # Medium volatility
            )
        )
        
        # Trend regime
        df['trend_regime'] = np.where(
            df['close'] > df['sma_50'],
            1,   # Uptrend
            -1   # Downtrend
        )
        
        return df
    
    async def _add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment analysis features."""
        try:
            # Get news sentiment (simplified for demo)
            crypto_name = symbol.replace('-USD', '').lower()
            sentiment_data = await self.sentiment_agent.get_news_sentiment(crypto_name)
            
            # Add sentiment features as constant values (in production, this would be time-series)
            df['news_sentiment'] = sentiment_data['compound']
            df['news_positive'] = sentiment_data['positive']
            df['news_negative'] = sentiment_data['negative']
            
            # Social sentiment (mock data for demo)
            df['social_sentiment'] = np.random.normal(0, 0.1, len(df))
            
        except Exception as e:
            logger.warning(f"Error adding sentiment features: {e}")
            df['news_sentiment'] = 0
            df['news_positive'] = 0
            df['news_negative'] = 0
            df['social_sentiment'] = 0
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare sequences for time series forecasting."""
        # Select features for modeling
        feature_cols = [col for col in df.columns if col not in ['date', target_col]]
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.config.SEQUENCE_LENGTH, len(features_scaled)):
            X.append(features_scaled[i-self.config.SEQUENCE_LENGTH:i])
            y.append(target[i])
        
        return np.array(X), np.array(y), feature_cols


class CryptoDataset(Dataset):
    """PyTorch dataset for cryptocurrency time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """Advanced LSTM model with attention mechanism."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        # Reshape for attention: (seq_len, batch_size, hidden_size)
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        output = attn_out[-1]  # (batch_size, hidden_size)
        
        # Fully connected layers
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer-based model for price prediction."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Global average pooling
        output = transformer_out.mean(dim=1)
        
        # Output layer
        output = self.dropout(output)
        output = self.fc(output)
        
        return output


class ModelTrainer:
    """Advanced model training with hyperparameter optimization and early stopping."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> Dict:
        """Train a model with early stopping and learning rate scheduling."""
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()
            
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.config.MODEL_DIR / f'{model_name}_best.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}/{self.config.EPOCHS}, '
                          f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        model.load_state_dict(torch.load(self.config.MODEL_DIR / f'{model_name}_best.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }


class EnsembleModel:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, models: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions."""
        predictions = []
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred * self.weights[name])
        
        return torch.sum(torch.stack(predictions), dim=0)


class CryptocurrencyForecaster:
    """Main forecasting system orchestrating all components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_agent = DataAgent(config)
        self.trainer = ModelTrainer(config)
        self.models = {}
        
    async def run_complete_pipeline(self, symbol: str = "BTC-USD") -> Dict:
        """Run the complete forecasting pipeline."""
        logger.info("Starting complete forecasting pipeline")
        
        try:
            # 1. Data collection and preprocessing
            df = await self.data_agent.fetch_cryptocurrency_data(symbol)
            
            # 2. Prepare sequences
            X, y, feature_names = self.data_agent.prepare_sequences(df)
            
            # 3. Split data
            train_size = int(len(X) * self.config.TRAIN_SPLIT)
            val_size = int(len(X) * self.config.VAL_SPLIT)
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
            
            # 4. Create data loaders
            train_dataset = CryptoDataset(X_train, y_train)
            val_dataset = CryptoDataset(X_val, y_val)
            test_dataset = CryptoDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE)
            
            # 5. Train models
            input_size = X.shape[2]
            
            # LSTM Model
            lstm_model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config.HIDDEN_SIZE,
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT
            )
            
            logger.info("Training LSTM model")
            lstm_results = self.trainer.train_model(lstm_model, train_loader, val_loader, "lstm")
            
            # Transformer Model
            transformer_model = TransformerModel(
                input_size=input_size,
                d_model=self.config.HIDDEN_SIZE,
                dropout=self.config.DROPOUT
            )
            
            logger.info("Training Transformer model")
            transformer_results = self.trainer.train_model(transformer_model, train_loader, val_loader, "transformer")
            
            # 6. Create ensemble
            ensemble = EnsembleModel({
                'lstm': lstm_model,
                'transformer': transformer_model
            })
            
            # 7. Evaluate models
            results = self._evaluate_models(
                {'lstm': lstm_model, 'transformer': transformer_model, 'ensemble': ensemble},
                test_loader, X_test, y_test
            )
            
            # 8. Generate predictions and visualizations
            self._generate_visualizations(df, results, symbol)
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _evaluate_models(self, models: Dict, test_loader: DataLoader, 
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all models and return metrics."""
        results = {}
        device = self.trainer.device
        
        for name, model in models.items():
            predictions = []
            actuals = []
            
            if name == 'ensemble':
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                pred = model.predict(X_test_tensor)
                predictions = pred.cpu().numpy().flatten()
                actuals = y_test
            else:
                model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)
                        predictions.extend(outputs.cpu().numpy().flatten())
                        actuals.extend(batch_y.numpy())
            
            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actuals, predictions)
            
            results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': predictions,
                'actuals': actuals
            }
            
            logger.info(f"{name.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return results
    
    def _generate_visualizations(self, df: pd.DataFrame, results: Dict, symbol: str):
        """Generate comprehensive visualizations."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Price history with technical indicators
        ax1 = axes[0, 0]
        ax1.plot(df['date'][-1000:], df['close'][-1000:], label='Close Price', linewidth=2)
        if 'sma_20' in df.columns:
            ax1.plot(df['date'][-1000:], df['sma_20'][-1000:], label='SMA 20', alpha=0.7)
            ax1.plot(df['date'][-1000:], df['sma_50'][-1000:], label='SMA 50', alpha=0.7)
        ax1.set_title(f'{symbol} Price History with Technical Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model predictions comparison
        ax2 = axes[0, 1]
        test_indices = range(len(results['lstm']['actuals']))
        ax2.plot(test_indices, results['lstm']['actuals'], label='Actual', linewidth=2, alpha=0.8)
        
        for model_name in ['lstm', 'transformer', 'ensemble']:
            if model_name in results:
                ax2.plot(test_indices, results[model_name]['predictions'], 
                        label=f'{model_name.upper()} Predictions', alpha=0.7)
        
        ax2.set_title('Model Predictions vs Actual Prices')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals analysis
        ax3 = axes[1, 0]
        for model_name in ['lstm', 'transformer', 'ensemble']:
            if model_name in results:
                residuals = np.array(results[model_name]['actuals']) - np.array(results[model_name]['predictions'])
                ax3.scatter(results[model_name]['predictions'], residuals, 
                           label=f'{model_name.upper()}', alpha=0.6, s=20)
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_title('Residuals Analysis')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model performance metrics
        ax4 = axes[1, 1]
        metrics = ['mae', 'rmse', 'r2']
        model_names = [name for name in results.keys() if name in ['lstm', 'transformer', 'ensemble']]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            values = [results[model_name][metric] for metric in metrics]
            ax4.bar(x + i*width, values, width, label=model_name.upper(), alpha=0.8)
        
        ax4.set_title('Model Performance Comparison')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Values')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([m.upper() for m in metrics])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_DIR / f'{symbol}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional volatility and sentiment analysis
        self._plot_advanced_analysis(df, symbol)
    
    def _plot_advanced_analysis(self, df: pd.DataFrame, symbol: str):
        """Generate advanced analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Volatility analysis
        ax1 = axes[0, 0]
        ax1.plot(df['date'][-1000:], df['volatility'][-1000:], color='red', linewidth=2)
        ax1.fill_between(df['date'][-1000:], df['volatility'][-1000:], alpha=0.3, color='red')
        ax1.set_title('Price Volatility Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volatility')
        ax1.grid(True, alpha=0.3)
        
        # RSI analysis
        if 'rsi' in df.columns:
            ax2 = axes[0, 1]
            ax2.plot(df['date'][-1000:], df['rsi'][-1000:], color='purple', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax2.fill_between(df['date'][-1000:], 30, 70, alpha=0.1, color='gray')
            ax2.set_title('RSI Analysis')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Sentiment analysis
        if 'news_sentiment' in df.columns:
            ax3 = axes[1, 0]
            sentiment_data = df['news_sentiment'][-1000:]
            ax3.bar(range(len(sentiment_data)), sentiment_data, 
                   color=['green' if x > 0 else 'red' for x in sentiment_data], alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax3.set_title('News Sentiment Analysis')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Sentiment Score')
            ax3.grid(True, alpha=0.3)
        
        # Price distribution
        ax4 = axes[1, 1]
        ax4.hist(df['price_change'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_title('Price Change Distribution')
        ax4.set_xlabel('Price Change (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_DIR / f'{symbol}_advanced_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


class GenerativeAgent:
    """AI agent for generating market insights and reports using LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize local LLM for insights (using transformers)
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load GPT-2 model: {e}")
            self.tokenizer = None
            self.model = None
    
    def generate_market_insights(self, results: Dict, df: pd.DataFrame, symbol: str) -> str:
        """Generate comprehensive market insights using AI."""
        
        # Extract key metrics
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'] if x != 'ensemble' else float('inf'))
        best_rmse = results[best_model]['rmse']
        best_r2 = results[best_model]['r2']
        
        # Market analysis
        current_price = df['close'].iloc[-1]
        price_change_24h = df['price_change'].iloc[-1] * 100
        volatility = df['volatility'].iloc[-1]
        
        # Technical indicators summary
        rsi_current = df['rsi'].iloc[-1] if 'rsi' in df.columns else None
        
        # Generate structured insights
        insights = f"""
# Cryptocurrency Market Analysis Report
## {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
- **Current Price**: ${current_price:,.2f}
- **24h Change**: {price_change_24h:+.2f}%
- **Current Volatility**: {volatility:.4f}
- **Best Model Performance**: {best_model.upper()} (RMSE: {best_rmse:.4f}, R²: {best_r2:.4f})

### Technical Analysis
"""
        
        if rsi_current:
            rsi_interpretation = "OVERBOUGHT" if rsi_current > 70 else "OVERSOLD" if rsi_current < 30 else "NEUTRAL"
            insights += f"- **RSI ({rsi_current:.1f})**: Market is currently {rsi_interpretation}\n"
        
        insights += f"""
### Model Performance Analysis
"""
        
        for model_name, metrics in results.items():
            insights += f"""
#### {model_name.upper()} Model
- **MAE**: {metrics['mae']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **R² Score**: {metrics['r2']:.4f}
"""
        
        # Risk assessment
        risk_level = "HIGH" if volatility > 0.05 else "MEDIUM" if volatility > 0.02 else "LOW"
        insights += f"""
### Risk Assessment
- **Volatility Level**: {risk_level}
- **Risk Score**: {volatility:.4f}

### Trading Recommendations
"""
        
        if price_change_24h > 2:
            insights += "- Strong upward momentum detected - Consider profit-taking strategies\n"
        elif price_change_24h < -2:
            insights += "- Significant downward pressure - Look for potential reversal signals\n"
        else:
            insights += "- Market consolidation phase - Range-bound trading opportunities\n"
        
        insights += f"""
### AI Model Insights
The ensemble model combining LSTM and Transformer architectures achieved superior 
performance with an R² score of {results.get('ensemble', {}).get('r2', 0):.4f}. 
This indicates {self._interpret_r2_score(results.get('ensemble', {}).get('r2', 0))} 
predictive capability for {symbol} price movements.

### Disclaimer
This analysis is generated by AI models and should not be considered as financial advice. 
Always conduct your own research and consult with financial professionals before making 
investment decisions.
"""
        
        return insights
    
    def _interpret_r2_score(self, r2: float) -> str:
        """Interpret R² score for non-technical users."""
        if r2 > 0.8:
            return "excellent"
        elif r2 > 0.6:
            return "good"
        elif r2 > 0.4:
            return "moderate"
        else:
            return "limited"
    
    def generate_trading_signals(self, df: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
        """Generate actionable trading signals."""
        signals = []
        
        # Simple signal generation based on predictions and technical indicators
        for i in range(-10, 0):  # Last 10 predictions
            if i >= -len(predictions):
                current_price = df['close'].iloc[i]
                predicted_price = predictions[i]
                price_change_pred = (predicted_price - current_price) / current_price * 100
                
                signal_type = "BUY" if price_change_pred > 1 else "SELL" if price_change_pred < -1 else "HOLD"
                confidence = min(abs(price_change_pred) * 10, 100)
                
                signals.append({
                    'timestamp': df['date'].iloc[i] if 'date' in df.columns else f"T{i}",
                    'signal': signal_type,
                    'confidence': confidence,
                    'predicted_change': price_change_pred,
                    'current_price': current_price,
                    'predicted_price': predicted_price
                })
        
        return signals


async def main():
    """Main execution function."""
    # Initialize configuration
    config = Config()
    
    # Create forecasting system
    forecaster = CryptocurrencyForecaster(config)
    
    # Initialize generative agent
    gen_agent = GenerativeAgent(config)
    
    try:
        # Run the complete pipeline
        results = await forecaster.run_complete_pipeline("BTC-USD")
        
        # Load the data for insights generation
        df = await forecaster.data_agent.fetch_cryptocurrency_data("BTC-USD")
        
        # Generate AI insights
        insights = gen_agent.generate_market_insights(results, df, "BTC-USD")
        
        # Save insights to file
        with open(config.RESULTS_DIR / 'market_insights.md', 'w') as f:
            f.write(insights)
        
        # Generate trading signals
        ensemble_predictions = results.get('ensemble', {}).get('predictions', [])
        if ensemble_predictions:
            signals = gen_agent.generate_trading_signals(df, ensemble_predictions)
            
            # Save signals to file
            signals_df = pd.DataFrame(signals)
            signals_df.to_csv(config.RESULTS_DIR / 'trading_signals.csv', index=False)
        
        logger.info("Analysis complete. Check the results directory for outputs.")
        print(insights)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    # Set up event loop for async execution
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
