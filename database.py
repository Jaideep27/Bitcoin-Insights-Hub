# backend/database.py
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import Optional, List, Dict, Any
import logging

class Database:
    """Database class for handling all database operations"""
    _instance = None

    def __new__(cls, db_name: str = 'bitcoin_analysis.db'):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, db_name: str = 'bitcoin_analysis.db'):
        """Initialize database connection and create tables only once"""
        if not hasattr(self, 'initialized') or not self.initialized:
            self.db_name = db_name
            self.logger = self._setup_logger()
            self.create_tables()
            self.initialized = True

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('BitcoinDB')
        logger.setLevel(logging.INFO)
        
        # Check if handler already exists to avoid duplicate logs
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def create_tables(self):
        """Create all necessary database tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Create price data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                timestamp DATETIME PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            ''')
            
            # Create technical indicators table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                timestamp DATETIME PRIMARY KEY,
                rsi REAL,
                macd REAL,
                signal_line REAL,
                FOREIGN KEY (timestamp) REFERENCES price_data (timestamp)
            )
            ''')
            
            # Create news sentiment table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                title TEXT,
                content TEXT,
                sentiment_label TEXT,
                sentiment_score REAL
            )
            ''')
            
            # Create price predictions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                predictions TEXT,
                confidence_score REAL,
                technical_score REAL,
                pattern_score REAL,
                sentiment_score REAL,
                support_levels TEXT,
                resistance_levels TEXT,
                analysis TEXT
            )
            ''')
            
            # Create advanced technical indicators table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_indicators (
                timestamp DATETIME PRIMARY KEY,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                stoch_k REAL,
                stoch_d REAL,
                atr REAL,
                obv REAL,
                ichimoku_tenkan REAL,
                ichimoku_kijun REAL,
                ichimoku_senkou_a REAL,
                ichimoku_senkou_b REAL,
                FOREIGN KEY (timestamp) REFERENCES price_data (timestamp)
            )
            ''')
            
            conn.commit()
            self.logger.info("Successfully created database tables")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_price_data(self, data: pd.DataFrame):
        """Save price data to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            data.to_sql('price_data', conn, if_exists='replace', index=True)
            self.logger.info("Successfully saved price data")
        except Exception as e:
            self.logger.error(f"Error saving price data: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_technical_indicators(self, data: pd.DataFrame):
        """Save technical indicators to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            indicators = data[['RSI', 'MACD', 'Signal_Line']]
            indicators.to_sql('technical_indicators', conn, if_exists='replace', index=True)
            self.logger.info("Successfully saved technical indicators")
        except Exception as e:
            self.logger.error(f"Error saving technical indicators: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_advanced_indicators(self, data: pd.DataFrame):
        """Save advanced technical indicators to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            indicators = data[[
                'BB_upper', 'BB_middle', 'BB_lower',
                '%K', '%D', 'ATR', 'OBV',
                'Tenkan_sen', 'Kijun_sen',
                'Senkou_span_A', 'Senkou_span_B'
            ]]
            indicators.to_sql('advanced_indicators', conn, if_exists='replace', index=True)
            self.logger.info("Successfully saved advanced indicators")
        except Exception as e:
            self.logger.error(f"Error saving advanced indicators: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_news_sentiment(self, news_item: Dict[str, Any], sentiment: Dict[str, Any]):
        """Save news and sentiment data to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO news_sentiment 
            (timestamp, title, content, sentiment_label, sentiment_score)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                news_item.get('title', ''),
                news_item.get('body', ''),
                sentiment.get('label', 'neutral'),
                sentiment.get('score', 0.0)
            ))
            
            conn.commit()
            self.logger.info("Successfully saved news sentiment")
        except Exception as e:
            self.logger.error(f"Error saving news sentiment: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_prediction(self, prediction_data: Dict[str, Any]):
        """Save price prediction and analysis to database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO price_predictions 
            (timestamp, predictions, confidence_score, technical_score, 
             pattern_score, sentiment_score, support_levels, resistance_levels, analysis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                json.dumps(prediction_data.get('price_predictions', [])),
                prediction_data.get('confidence_score', 0.0),
                prediction_data.get('technical_score', 0.0),
                prediction_data.get('pattern_score', 0.0),
                prediction_data.get('sentiment_score', 0.0),
                json.dumps(prediction_data.get('support_levels', [])),
                json.dumps(prediction_data.get('resistance_levels', [])),
                prediction_data.get('analysis', 'No analysis available')
            ))
            
            conn.commit()
            self.logger.info("Successfully saved prediction data")
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_latest_price_data(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve latest price data"""
        try:
            conn = sqlite3.connect(self.db_name)
            query = f"SELECT * FROM price_data ORDER BY timestamp DESC LIMIT {limit}"
            data = pd.read_sql_query(query, conn, index_col='timestamp')
            return data
        finally:
            if conn:
                conn.close()

    def get_latest_indicators(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve latest technical indicators"""
        try:
            conn = sqlite3.connect(self.db_name)
            query = f"SELECT * FROM technical_indicators ORDER BY timestamp DESC LIMIT {limit}"
            data = pd.read_sql_query(query, conn, index_col='timestamp')
            return data
        finally:
            if conn:
                conn.close()

    def get_latest_advanced_indicators(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve latest advanced technical indicators"""
        try:
            conn = sqlite3.connect(self.db_name)
            query = f"SELECT * FROM advanced_indicators ORDER BY timestamp DESC LIMIT {limit}"
            data = pd.read_sql_query(query, conn, index_col='timestamp')
            return data
        finally:
            if conn:
                conn.close()

    def get_recent_news_sentiment(self, limit: int = 5) -> List[tuple]:
        """Retrieve recent news with sentiment"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM news_sentiment 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
        finally:
            if conn:
                conn.close()

    def get_latest_predictions(self, limit: int = 1) -> List[tuple]:
        """Retrieve latest price predictions"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM price_predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
        finally:
            if conn:
                conn.close()

    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            tables = ['price_data', 'technical_indicators', 'advanced_indicators', 
                     'news_sentiment', 'price_predictions']
            
            for table in tables:
                cursor.execute(f'''
                DELETE FROM {table} 
                WHERE timestamp < datetime('now', '-{days} days')
                ''')
            
            conn.commit()
            self.logger.info(f"Successfully cleaned up data older than {days} days")
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
        finally:
            if conn:
                conn.close()