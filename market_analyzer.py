import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import yfinance as yf
import requests
from typing import Dict, List, Tuple

class MarketAnalyzer:
    def __init__(self):
        self.support_resistance_window = 20
        self.pivot_point_threshold = 0.02  # 2% threshold for pivot points

    def get_historical_data(self, months: int = 6) -> pd.DataFrame:
        """Fetch 6 months of historical data"""
        try:
            btc = yf.Ticker("BTC-USD")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30)
            
            # Fetch daily data for pattern analysis
            data = btc.history(start=start_date, end=end_date, interval='1d')
            return data
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_historical_news(self, months: int = 6) -> List[Dict]:
        """Fetch historical news and significant events"""
        try:
            # Use CryptoCompare news API with historical data
            url = "https://min-api.cryptocompare.com/data/v2/news/?"
            params = {
                'lang': 'EN',
                'sortOrder': 'popular',
                'limit': 100  # Fetch more news to filter relevant ones
            }
            response = requests.get(url, params=params)
            all_news = response.json().get('Data', [])
            
            # Filter news by date and significance
            cutoff_date = datetime.now() - timedelta(days=months*30)
            significant_news = []
            
            for news in all_news:
                news_date = datetime.fromtimestamp(news['published_on'])
                if news_date >= cutoff_date:
                    # Filter for significant news based on categories/keywords
                    if any(keyword in news['categories'].lower() for keyword in 
                          ['regulation', 'adoption', 'etf', 'sec', 'federal', 'major']):
                        significant_news.append(news)
            
            return significant_news
        except Exception as e:
            print(f"Error fetching historical news: {e}")
            return []

    def identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify dynamic support and resistance levels using price action"""
        try:
            df = data.copy()
            
            # Calculate pivot points
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values
            
            # Find peaks for resistance levels
            resistance_idx, _ = find_peaks(highs, distance=self.support_resistance_window)
            resistance_levels = sorted(highs[resistance_idx])
            
            # Find troughs for support levels
            support_idx, _ = find_peaks(-lows, distance=self.support_resistance_window)
            support_levels = sorted(lows[support_idx])
            
            # Filter levels by significance (volume and retests)
            significant_levels = {
                'support': [],
                'resistance': []
            }
            
            current_price = closes[-1]
            
            # Find most relevant support levels below current price
            supports = [level for level in support_levels if level < current_price]
            if supports:
                # Get levels that have been tested multiple times
                support_tests = self._count_level_tests(df, supports)
                significant_levels['support'] = [
                    level for level, tests in support_tests.items() 
                    if tests >= 2  # Require at least 2 tests
                ][:3]  # Get top 3 most significant
            
            # Find most relevant resistance levels above current price
            resistances = [level for level in resistance_levels if level > current_price]
            if resistances:
                # Get levels that have been tested multiple times
                resistance_tests = self._count_level_tests(df, resistances)
                significant_levels['resistance'] = [
                    level for level, tests in resistance_tests.items() 
                    if tests >= 2  # Require at least 2 tests
                ][:3]  # Get top 3 most significant
            
            return significant_levels
            
        except Exception as e:
            print(f"Error identifying support/resistance: {e}")
            return {'support': [], 'resistance': []}

    def _count_level_tests(self, data: pd.DataFrame, levels: List[float]) -> Dict[float, int]:
        """Count how many times each price level has been tested"""
        level_tests = {}
        
        for level in levels:
            # Consider a price movement within 0.5% of level as a test
            threshold = level * 0.005
            tests = sum(1 for low, high in zip(data['Low'], data['High'])
                       if abs(low - level) <= threshold or abs(high - level) <= threshold)
            level_tests[level] = tests
            
        return level_tests

    def analyze_market_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze historical patterns and market behavior"""
        patterns = {
            'trend': self._identify_trend(data),
            'volatility': self._calculate_volatility(data),
            'momentum': self._calculate_momentum(data),
            'volume_trend': self._analyze_volume_trend(data),
            'key_levels': self.identify_support_resistance(data)
        }
        
        return patterns

    def _identify_trend(self, data: pd.DataFrame) -> Dict:
        """Identify current market trend using multiple timeframes"""
        df = data.copy()
        
        # Calculate EMAs for different timeframes
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA100'] = df['Close'].ewm(span=100).mean()
        
        current_price = df['Close'].iloc[-1]
        
        # Determine trend strength and direction
        trend = {
            'short_term': 'bullish' if current_price > df['EMA20'].iloc[-1] else 'bearish',
            'medium_term': 'bullish' if current_price > df['EMA50'].iloc[-1] else 'bearish',
            'long_term': 'bullish' if current_price > df['EMA100'].iloc[-1] else 'bearish'
        }
        
        # Calculate trend strength
        trend['strength'] = sum([
            1 if t == 'bullish' else -1 for t in trend.values() if isinstance(t, str)
        ]) / 3
        
        return trend

    def _calculate_volatility(self, data: pd.DataFrame) -> Dict:
        """Calculate market volatility metrics"""
        df = data.copy()
        
        # Calculate daily returns
        df['returns'] = df['Close'].pct_change()
        
        volatility = {
            'daily': df['returns'].std() * np.sqrt(252),  # Annualized volatility
            'recent': df['returns'].tail(7).std() * np.sqrt(252),  # Recent volatility
            'trend': 'increasing' if df['returns'].tail(7).std() > df['returns'].std() else 'decreasing'
        }
        
        return volatility

    def _calculate_momentum(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        df = data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        momentum = {
            'rsi': df['RSI'].iloc[-1],
            'rsi_trend': 'overbought' if df['RSI'].iloc[-1] > 70 else 'oversold' if df['RSI'].iloc[-1] < 30 else 'neutral',
            'price_momentum': df['Close'].pct_change(periods=7).iloc[-1]  # 7-day momentum
        }
        
        return momentum

    def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze volume trends"""
        df = data.copy()
        
        # Calculate volume moving averages
        df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
        
        volume_analysis = {
            'trend': 'increasing' if df['Volume'].iloc[-1] > df['volume_ma20'].iloc[-1] else 'decreasing',
            'average_volume': df['Volume'].mean(),
            'recent_volume': df['Volume'].tail(7).mean()
        }
        
        return volume_analysis

    def correlate_news_with_price(self, news_data: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Correlate significant news events with price movements"""
        correlated_events = []
        
        for news in news_data:
            news_date = datetime.fromtimestamp(news['published_on'])
            
            # Find price action around news
            try:
                price_before = price_data.loc[:news_date].iloc[-2:-1]['Close'].values[0]
                price_after = price_data.loc[news_date:].iloc[1:2]['Close'].values[0]
                
                price_change = ((price_after - price_before) / price_before) * 100
                
                if abs(price_change) >= 2:  # Significant price movement threshold
                    correlated_events.append({
                        'date': news_date,
                        'title': news['title'],
                        'price_change': price_change,
                        'impact': 'high' if abs(price_change) > 5 else 'medium'
                    })
            except:
                continue
        
        return correlated_events