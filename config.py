import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Deep learning model configuration."""
    sequence_length: int = 60
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    target_column: str = 'close'
    date_column: str = 'date'
    
    # Feature engineering flags
    technical_indicators: bool = True
    sentiment_analysis: bool = True
    market_regime_detection: bool = True
    volume_analysis: bool = True
    
    # Technical indicator parameters
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Moving average periods
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])


@dataclass
class APIConfig:
    """API keys and external service configuration."""
    news_api_key: str = field(default_factory=lambda: os.getenv('NEWS_API_KEY', ''))
    twitter_api_key: str = field(default_factory=lambda: os.getenv('TWITTER_API_KEY', ''))
    twitter_api_secret: str = field(default_factory=lambda: os.getenv('TWITTER_API_SECRET', ''))
    twitter_bearer_token: str = field(default_factory=lambda: os.getenv('TWITTER_BEARER_TOKEN', ''))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_KEY', ''))
    
    # Rate limiting
    api_rate_limit: int = 100  # requests per hour
    news_articles_limit: int = 100
    twitter_tweets_limit: int = 200


@dataclass
class PathConfig:
    """File system paths configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    model_dir: Path = field(default_factory=lambda: Path('models'))
    data_dir: Path = field(default_factory=lambda: Path('data'))
    results_dir: Path = field(default_factory=lambda: Path('results'))
    logs_dir: Path = field(default_factory=lambda: Path('logs'))
    cache_dir: Path = field(default_factory=lambda: Path('cache'))
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.model_dir, self.data_dir, self.results_dir, 
                    self.logs_dir, self.cache_dir]:
            path.mkdir(exist_ok=True, parents=True)


@dataclass
class TradingConfig:
    """Trading and signal generation configuration."""
    signal_threshold: float = 0.02  # 2% price change threshold
    confidence_threshold: float = 0.7
    risk_tolerance: str = 'medium'  # low, medium, high
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    # Signal timeframes
    signal_timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])


class SystemConfig:
    """Main system configuration class."""
    
    def __init__(self, config_file: str = None):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.paths = PathConfig()
        self.trading = TradingConfig()
        
        # System settings
        self.debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.device = 'cuda' if os.getenv('FORCE_CPU', 'False').lower() != 'true' else 'cpu'
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_file}: {e}")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        try:
            config_data = {
                'model': self.model.__dict__,
                'data': self.data.__dict__,
                'trading': self.trading.__dict__
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Model validation
        if self.model.sequence_length < 1:
            errors.append("Sequence length must be positive")
        
        if self.model.batch_size < 1:
            errors.append("Batch size must be positive")
        
        if not (0 < self.model.learning_rate < 1):
            errors.append("Learning rate must be between 0 and 1")
        
        # Data validation
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 0.001:
            errors.append(f"Data splits must sum to 1.0, got {total_split}")
        
        # Trading validation
        if not (0 < self.trading.max_position_size <= 1):
            errors.append("Max position size must be between 0 and 1")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_cryptocurrencies(self) -> List[Dict[str, Any]]:
        """Get list of supported cryptocurrencies."""
        return [
            {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'market_cap_rank': 1},
            {'symbol': 'ETH-USD', 'name': 'Ethereum', 'market_cap_rank': 2},
            {'symbol': 'BNB-USD', 'name': 'Binance Coin', 'market_cap_rank': 3},
            {'symbol': 'ADA-USD', 'name': 'Cardano', 'market_cap_rank': 4},
            {'symbol': 'SOL-USD', 'name': 'Solana', 'market_cap_rank': 5},
            {'symbol': 'XRP-USD', 'name': 'Ripple', 'market_cap_rank': 6},
            {'symbol': 'DOT-USD', 'name': 'Polkadot', 'market_cap_rank': 7},
            {'symbol': 'DOGE-USD', 'name': 'Dogecoin', 'market_cap_rank': 8},
            {'symbol': 'AVAX-USD', 'name': 'Avalanche', 'market_cap_rank': 9},
            {'symbol': 'SHIB-USD', 'name': 'Shiba Inu', 'market_cap_rank': 10}
        ]
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns based on configuration."""
        base_features = ['open', 'high', 'low', 'close', 'volume', 'price_change', 'log_return', 'volatility']
        
        if self.data.technical_indicators:
            technical_features = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'atr'
            ]
            
            # Add moving averages
            for period in self.data.ma_periods:
                technical_features.extend([f'sma_{period}', f'ema_{period}'])
            
            technical_features.extend(['price_vs_sma20', 'price_vs_sma50'])
            base_features.extend(technical_features)
        
        if self.data.market_regime_detection:
            base_features.extend(['vol_regime', 'trend_regime'])
        
        if self.data.sentiment_analysis:
            base_features.extend(['news_sentiment', 'news_positive', 'news_negative', 'social_sentiment'])
        
        if self.data.volume_analysis:
            base_features.extend(['volume_sma', 'volume_ratio'])
        
        return base_features


# Global configuration instance
config = SystemConfig()

# Environment setup
def setup_environment():
    """Setup environment variables and logging."""
    
    # Set up logging
    log_level = getattr(logging, config.log_level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.paths.logs_dir / 'system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create .env template if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        env_template = """# API Keys for Cryptocurrency Forecasting System
# News API (https://newsapi.org/)
NEWS_API_KEY=your_news_api_key_here

# Twitter API (https://developer.twitter.com/)
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# OpenAI API (https://openai.com/)
OPENAI_API_KEY=your_openai_api_key_here

# Alpha Vantage API (https://www.alphavantage.co/)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# System Configuration
DEBUG=False
LOG_LEVEL=INFO
FORCE_CPU=False
"""
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        logger.info("Created .env template file. Please add your API keys.")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed. Environment variables must be set manually.")


if __name__ == "__main__":
    setup_environment()
    
    # Validate configuration
    if config.validate():
        print("✅ Configuration is valid")
        
        # Print configuration summary
        print("\n📊 Configuration Summary:")
        print(f"Model: LSTM + Transformer ensemble")
        print(f"Sequence Length: {config.model.sequence_length}")
        print(f"Batch Size: {config.model.batch_size}")
        print(f"Learning Rate: {config.model.learning_rate}")
        print(f"Features: {len(config.get_feature_columns())} columns")
        print(f"Technical Indicators: {'✅' if config.data.technical_indicators else '❌'}")
        print(f"Sentiment Analysis: {'✅' if config.data.sentiment_analysis else '❌'}")
        print(f"Market Regime Detection: {'✅' if config.data.market_regime_detection else '❌'}")
        
    else:
        print("❌ Configuration validation failed")
