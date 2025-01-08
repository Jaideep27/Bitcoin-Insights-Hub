# backend/technical_indicators.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Union, Optional,List


class TechnicalIndicators:
    """A comprehensive class for calculating various technical indicators"""

    @staticmethod
    def calculate_all(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given dataset"""
        df = data.copy()
        
        # Calculate all indicators
        df = TechnicalIndicators.calculate_rsi(df)
        df = TechnicalIndicators.calculate_macd(df)
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        df = TechnicalIndicators.calculate_stochastic_oscillator(df)
        df = TechnicalIndicators.calculate_atr(df)
        df = TechnicalIndicators.calculate_obv(df)
        df = TechnicalIndicators.calculate_ichimoku_cloud(df)
        
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate the Relative Strength Index (RSI)"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate the Moving Average Convergence Divergence (MACD)"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['BB_middle'] = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        df['BB_upper'] = df['BB_middle'] + (std * num_std)
        df['BB_lower'] = df['BB_middle'] - (std * num_std)  # Changed from BBL_20_2.0
        return df

    @staticmethod
    def calculate_stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_window).min()
        high_max = df['High'].rolling(window=k_window).max()
        
        df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['%D'] = df['%K'].rolling(window=d_window).mean()
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=window).mean()
        return df

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume (OBV)"""
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return df

    @staticmethod
    def calculate_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan_sen'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun_sen'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Senkou_span_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['Chikou_span'] = df['Close'].shift(-26)
        return df

    @staticmethod
    def get_market_summary(data: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """Generate market summary based on technical indicators"""
        if data is None or len(data) == 0:
            return None
            
        current_price = data['Close'].iloc[-1]
        try:
            price_change = ((current_price - data['Close'].iloc[-24]) / data['Close'].iloc[-24]) * 100
        except:
            price_change = 0.0
            
        # Get signals
        signals = TechnicalIndicators.get_signals(data)
        
        return {
            'current_price': current_price,
            'price_change_24h': price_change,
            'volume_24h': data['Volume'].iloc[-24:].sum(),
            'rsi': data['RSI'].iloc[-1],
            'macd_signal': signals['MACD'],
            'momentum': signals.get('RSI', 'Neutral'),
            'overall_trend': signals.get('BB', 'Neutral')
        }

    @staticmethod
    def get_signals(data: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        # RSI Signals
        if 'RSI' in data.columns:
            last_rsi = data['RSI'].iloc[-1]
            signals['RSI'] = 'Oversold' if last_rsi < 30 else 'Overbought' if last_rsi > 70 else 'Neutral'
        
        # MACD Signals
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            signals['MACD'] = 'Bullish' if data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1] else 'Bearish'
        
        # Bollinger Bands Signals
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            last_close = data['Close'].iloc[-1]
            if last_close > data['BB_upper'].iloc[-1]:
                signals['BB'] = 'Overbought'
            elif last_close < data['BB_lower'].iloc[-1]:
                signals['BB'] = 'Oversold'
            else:
                signals['BB'] = 'Neutral'
        
        # Stochastic Signals
        if '%K' in data.columns and '%D' in data.columns:
            last_k = data['%K'].iloc[-1]
            last_d = data['%D'].iloc[-1]
            signals['Stochastic'] = (
                'Oversold' if last_k < 20 and last_d < 20 
                else 'Overbought' if last_k > 80 and last_d > 80 
                else 'Neutral'
            )
        
        return signals

    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci Retracement Levels"""
        high = df['High'].max()
        low = df['Low'].min()
        diff = high - low
        
        df['Fib_0'] = low
        df['Fib_23.6'] = low + (diff * 0.236)
        df['Fib_38.2'] = low + (diff * 0.382)
        df['Fib_50'] = low + (diff * 0.5)
        df['Fib_61.8'] = low + (diff * 0.618)
        df['Fib_100'] = high
        return df

    @staticmethod
    def identify_patterns(df: pd.DataFrame, window: int = 20) -> Dict[str, List[int]]:
        """Identify common chart patterns"""
        patterns = {
            'double_tops': [],
            'double_bottoms': [],
            'head_and_shoulders': [],
            'triangles': []
        }
        
        # Find peaks and troughs
        peaks, _ = find_peaks(df['High'].values, distance=window)
        troughs, _ = find_peaks(-df['Low'].values, distance=window)
        
        # Identify double tops and bottoms
        for i in range(len(peaks)-1):
            if abs(df['High'].iloc[peaks[i]] - df['High'].iloc[peaks[i+1]]) < df['High'].std() * 0.1:
                patterns['double_tops'].append(peaks[i])
                
        for i in range(len(troughs)-1):
            if abs(df['Low'].iloc[troughs[i]] - df['Low'].iloc[troughs[i+1]]) < df['Low'].std() * 0.1:
                patterns['double_bottoms'].append(troughs[i])
        
        return patterns

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20, num_levels: int = 3) -> Dict[str, list]:
        """Calculate support and resistance levels"""
        levels = {
            'support': [],
            'resistance': []
        }
        
        # Find peaks and troughs
        peaks, _ = find_peaks(df['High'].values, distance=window)
        troughs, _ = find_peaks(-df['Low'].values, distance=window)
        
        # Get top resistance levels
        resistance_levels = df['High'].iloc[peaks].sort_values(ascending=False)[:num_levels]
        levels['resistance'] = resistance_levels.tolist()
        
        # Get bottom support levels
        support_levels = df['Low'].iloc[troughs].sort_values()[:num_levels]
        levels['support'] = support_levels.tolist()
        
        return levels