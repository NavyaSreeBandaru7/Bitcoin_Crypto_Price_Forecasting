import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataValidator:
    """Data quality validation and cleaning utilities."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate dataframe structure and content."""
        errors = []
        
        # Check if dataframe is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for excessive NaN values
        nan_threshold = 0.5  # 50% threshold
        for col in df.columns:
            if df[col].isna().sum() / len(df) > nan_threshold:
                errors.append(f"Column '{col}' has > {nan_threshold*100}% NaN values")
        
        # Check data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric")
        
        # Check for negative values in price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                errors.append(f"Column '{col}' contains negative values")
        
        # Check logical price relationships
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if (df['high'] < df['low']).any():
                errors.append("High prices are lower than low prices")
            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                errors.append("Close prices are outside high-low range")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess dataframe."""
        df_clean = df.copy()
        
        # Remove duplicate timestamps
        if 'date' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['date'])
        
        # Fill missing values with forward fill then backward fill
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method for price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Ensure logical price relationships
        if all(col in df_clean.columns for col in ['high', 'low']):
            df_clean['high'] = df_clean[['high', 'low']].max(axis=1)
            df_clean['low'] = df_clean[['high', 'low']].min(axis=1)
        
        return df_clean


class ModelEvaluator:
    """Model evaluation and performance analysis utilities."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Direction accuracy
        direction_accuracy = np.mean(
            np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
        ) * 100 if len(y_true) > 1 else 0
        
        # Volatility metrics
        true_volatility = np.std(np.diff(y_true))
        pred_volatility = np.std(np.diff(y_pred))
        volatility_ratio = pred_volatility / true_volatility if true_volatility > 0 else 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'volatility_ratio': volatility_ratio
        }
    
    @staticmethod
    def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                initial_capital: float = 10000) -> Dict[str, float]:
        """Calculate trading-specific performance metrics."""
        # Simple trading strategy: buy if prediction > current, sell otherwise
        returns = []
        capital = initial_capital
        
        for i in range(1, len(y_pred)):
            current_price = y_true[i-1]
            next_price = y_true[i]
            predicted_price = y_pred[i]
            
            if predicted_price > current_price:  # Buy signal
                return_pct = (next_price - current_price) / current_price
            else:  # Sell signal (short)
                return_pct = (current_price - next_price) / current_price
            
            returns.append(return_pct)
            capital *= (1 + return_pct)
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = ModelEvaluator._calculate_max_drawdown(returns)
        win_rate = np.sum(returns > 0) / len(returns) * 100 if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_capital': capital
        }
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) * 100


class VisualizationUtils:
    """Advanced visualization utilities for cryptocurrency analysis."""
    
    @staticmethod
    def plot_price_with_predictions(df: pd.DataFrame, predictions: Dict[str, np.ndarray],
                                  title: str = "Price Predictions", save_path: str = None):
        """Plot price history with model predictions."""
        plt.figure(figsize=(15, 8))
        
        # Plot actual prices
        plt.plot(df.index[-len(list(predictions.values())[0]):], 
                df['close'].iloc[-len(list(predictions.values())[0]):], 
                label='Actual Price', linewidth=2, alpha=0.8)
        
        # Plot predictions from different models
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            plt.plot(df.index[-len(pred):], pred, 
                    label=f'{model_name.upper()} Prediction', 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_technical_analysis(df: pd.DataFrame, title: str = "Technical Analysis", 
                              save_path: str = None):
        """Plot comprehensive technical analysis charts."""
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Price with moving averages
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=2)
        if 'sma_20' in df.columns:
            ax1.plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
        if 'sma_50' in df.columns:
            ax1.plot(df.index, df['sma_50'], label='SMA 50', alpha=0.7)
        ax1.set_title('Price with Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = axes[0, 1]
        ax2.bar(df.index, df['volume'], alpha=0.7, color='blue')
        ax2.set_title('Trading Volume')
        ax2.grid(True, alpha=0.3)
        
        # RSI
        ax3 = axes[1, 0]
        if 'rsi' in df.columns:
            ax3.plot(df.index, df['rsi'], color='purple', linewidth=2)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax3.set_title('RSI (Relative Strength Index)')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # MACD
        ax4 = axes[1, 1]
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            ax4.plot(df.index, df['macd'], label='MACD', linewidth=2)
            ax4.plot(df.index, df['macd_signal'], label='Signal', linewidth=2)
            if 'macd_histogram' in df.columns:
                ax4.bar(df.index, df['macd_histogram'], alpha=0.3, label='Histogram')
        ax4.set_title('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Bollinger Bands
        ax5 = axes[2, 0]
        ax5.plot(df.index, df['close'], label='Close Price', linewidth=2)
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax5.plot(df.index, df['bb_upper'], label='Upper Band', alpha=0.7)
            ax5.plot(df.index, df['bb_middle'], label='Middle Band', alpha=0.7)
            ax5.plot(df.index, df['bb_lower'], label='Lower Band', alpha=0.7)
            ax5.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.1)
        ax5.set_title('Bollinger Bands')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Volatility
        ax6 = axes[2, 1]
        if 'volatility' in df.columns:
            ax6.plot(df.index, df['volatility'], color='red', linewidth=2)
            ax6.fill_between(df.index, df['volatility'], alpha=0.3, color='red')
        ax6.set_title('Price Volatility')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict], title: str = "Model Comparison",
                            save_path: str = None):
        """Plot comprehensive model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(results.keys())
        metrics = ['mae', 'rmse', 'r2', 'mape']
        
        # Bar plot of metrics
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            ax1.bar(x + i*width, values, width, label=metric.upper())
        
        ax1.set_title('Model Performance Metrics')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Metric Values')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels([m.upper() for m in models])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Prediction accuracy scatter plot
        ax2 = axes[0, 1]
        for model in models:
            if 'predictions' in results[model] and 'actuals' in results[model]:
                ax2.scatter(results[model]['actuals'], results[model]['predictions'], 
                          label=model.upper(), alpha=0.6, s=20)
        
        # Perfect prediction line
        if results:
            all_actuals = np.concatenate([results[m]['actuals'] for m in models 
                                        if 'actuals' in results[m]])
            min_val, max_val = np.min(all_actuals), np.max(all_actuals)
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax2.set_title('Predicted vs Actual Values')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Residuals plot
        ax3 = axes[1, 0]
        for model in models:
            if 'predictions' in results[model] and 'actuals' in results[model]:
                residuals = np.array(results[model]['actuals']) - np.array(results[model]['predictions'])
                ax3.scatter(results[model]['predictions'], residuals, 
                          label=model.upper(), alpha=0.6, s=20)
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_title('Residuals Analysis')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # R² comparison
        ax4 = axes[1, 1]
        r2_values = [results[model].get('r2', 0) for model in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax4.bar(models, r2_values, color=colors, alpha=0.8)
        ax4.set_title('R² Score Comparison')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R² Score')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, r2_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FileManager:
    """File management utilities for saving and loading models and data."""
    
    @staticmethod
    def save_model(model: torch.nn.Module, filepath: str, metadata: Dict = None):
        """Save PyTorch model with metadata."""
        try:
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            torch.save(save_dict, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    @staticmethod
    def load_model(model: torch.nn.Module, filepath: str) -> Dict:
        """Load PyTorch model and return metadata."""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {filepath}")
            return checkpoint.get('metadata', {})
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {}
    
    @staticmethod
    def save_results(results: Dict, filepath: str):
        """Save results to JSON file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model, metrics in results.items():
                serializable_results[model] = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[model][key] = value.tolist()
                    else:
                        serializable_results[model][key] = value
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    @staticmethod
    def load_results(filepath: str) -> Dict:
        """Load results from JSON file."""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}


class PerformanceMonitor:
    """System performance monitoring utilities."""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = datetime.now()
        logger.info("Performance monitoring started")
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append({
                'stage': stage,
                'memory_mb': memory_mb,
                'timestamp': datetime.now()
            })
            logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")
        except ImportError:
            logger.warning("psutil not installed. Memory monitoring disabled.")
    
    def get_summary(self) -> Dict:
        """Get performance monitoring summary."""
        if not self.start_time:
            return {}
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        max_memory = max(self.memory_usage, key=lambda x: x['memory_mb'])['memory_mb'] if self.memory_usage else 0
        
        return {
            'total_execution_time': total_time,
            'max_memory_usage_mb': max_memory,
            'memory_stages': self.memory_usage
        }


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup comprehensive logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Suppress verbose logs from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'torch', 'transformers', 'yfinance', 'nltk', 'textblob'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        print(f"Please install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("All dependencies are satisfied")
    return True


if __name__ == "__main__":
    # Test utilities
    setup_logging('INFO')
    
    if check_dependencies():
        print("✅ All utilities are working correctly")
        
        # Test data validation
        test_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        validator = DataValidator()
        is_valid, errors = validator.validate_dataframe(test_df, ['open', 'high', 'low', 'close'])
        print(f"Data validation: {'✅' if is_valid else '❌'}")
        
        if errors:
            for error in errors:
                print(f"  - {error}")
    else:
        print("❌ Dependency check failed")
