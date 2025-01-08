# backend/__init__.py

from .data_collector import DataCollector
from .database import Database
from .technical_indicators import TechnicalIndicators
from .market_predictor import MarketPredictor
from .market_analyzer import MarketAnalyzer

# Version of the package
__version__ = '1.0.0'

# Export these classes so they can be imported directly from the package
__all__ = ['DataCollector', 'Database', 'TechnicalIndicators', 'MarketPredictor']