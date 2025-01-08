import numpy as np
import pandas as pd
from textblob import TextBlob
from scipy.signal import find_peaks
from datetime import datetime
import requests
from typing import Dict, List
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

class MarketPredictor:
    def __init__(self):
        try:
            # Gemini API Configuration
            self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
            if not self.gemini_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            genai.configure(api_key=self.gemini_api_key)
            
            # Create the Gemini model with specific configuration
            self.generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-thinking-exp-1219",
                generation_config=self.generation_config,
            )
            self.genai_available = True
            
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            self.genai_available = False
        
        # Support/Resistance parameters
        self.sr_window = 20
        self.pivot_point_threshold = 0.02
        self.analysis_window = 30
        self.significant_events = []

    def identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify dynamic support and resistance levels using price action"""
        try:
            df = data.copy()
            
            # Find peaks and troughs
            peaks, _ = find_peaks(df['High'].values, distance=self.sr_window)
            troughs, _ = find_peaks(-df['Low'].values, distance=self.sr_window)
            
            # Get resistance levels from peaks
            resistance_levels = df['High'].iloc[peaks].sort_values(ascending=False)[:3]
            
            # Get support levels from troughs
            support_levels = df['Low'].iloc[troughs].sort_values()[:3]
            
            # Filter significant levels
            current_price = df['Close'].iloc[-1]
            volume_profile = self._calculate_volume_profile(df)
            
            filtered_resistances = [
                level for level in resistance_levels 
                if level > current_price and self._is_significant_level(df, level, volume_profile)
            ]
            
            filtered_supports = [
                level for level in support_levels 
                if level < current_price and self._is_significant_level(df, level, volume_profile)
            ]
            
            return {
                'resistance_levels': filtered_resistances[:3],  # Top 3 resistance levels
                'support_levels': filtered_supports[:3],  # Top 3 support levels
            }
            
        except Exception as e:
            print(f"Error identifying support/resistance: {e}")
            return {'support_levels': [], 'resistance_levels': []}

    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[float, float]:
        """Calculate volume profile to identify significant price levels"""
        price_buckets = {}
        
        for price, volume in zip(data['Close'], data['Volume']):
            bucket = round(price, -1)  # Round to nearest 10
            price_buckets[bucket] = price_buckets.get(bucket, 0) + volume
            
        return price_buckets

    def _is_significant_level(self, data: pd.DataFrame, level: float, volume_profile: Dict[float, float]) -> bool:
        """Determine if a price level is significant based on historical interaction and volume"""
        bucket = round(level, -1)
        volume_threshold = np.mean(list(volume_profile.values())) * 1.5
        
        # Check if level has significant volume
        if volume_profile.get(bucket, 0) >= volume_threshold:
            return True
            
        # Check if level has been tested multiple times
        price_range = level * self.pivot_point_threshold
        tests = sum(1 for price in data['Close'] if abs(price - level) <= price_range)
        
        return tests >= 3  # Require at least 3 tests of the level

    def analyze_market_sentiment(self, news_data: List[Dict]) -> float:
        """Analyze market sentiment from news and technical indicators"""
        if not news_data:
            return 0.5
            
        sentiments = []
        for item in news_data:
            text = f"{item['title']} {item['body']}"
            blob = TextBlob(text)
            
            # Base sentiment
            sentiment = blob.sentiment.polarity
            
            # Adjust for crypto-specific keywords
            bullish_words = ['bullish', 'surge', 'rally', 'breakthrough', 'adoption', 'growth', 'upgrade', 'buy']
            bearish_words = ['bearish', 'crash', 'ban', 'regulation', 'sell-off', 'decline', 'risk', 'sell']
            
            text_lower = text.lower()
            
            # Weight recent news more heavily
            time_weight = 1.0
            if 'published_on' in item:
                news_age = datetime.now().timestamp() - item['published_on']
                if news_age < 3600:  # Less than 1 hour old
                    time_weight = 1.5
                elif news_age < 86400:  # Less than 24 hours old
                    time_weight = 1.2
            
            # Adjust sentiment based on keywords
            for word in bullish_words:
                if word in text_lower:
                    sentiment += 0.1 * time_weight
            for word in bearish_words:
                if word in text_lower:
                    sentiment -= 0.1 * time_weight
            
            sentiments.append(sentiment)
            
        # Normalize final sentiment score
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return min(max((avg_sentiment + 1) / 2, 0), 1)  # Convert to 0-1 scale

    def calculate_technical_score(self, data: pd.DataFrame) -> float:
        """Calculate technical analysis score focused on trend and momentum"""
        try:
            df = data.copy()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # Volume trend
            df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
            
            # Calculate scores
            rsi = df['RSI'].iloc[-1]
            rsi_score = (rsi - 30) / 40  # Normalize RSI score
            
            ma_trend = 1 if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] else 0
            price_above_ma = 1 if df['Close'].iloc[-1] > df['SMA20'].iloc[-1] else 0
            
            volume_trend = 1 if df['Volume'].iloc[-1] > df['Volume_SMA20'].iloc[-1] else 0
            
            # Combine scores with weightings
            technical_score = (
                rsi_score * 0.4 +
                ma_trend * 0.3 +
                price_above_ma * 0.2 +
                volume_trend * 0.1
            )
            
            return min(max(technical_score, 0), 1)
            
        except Exception as e:
            print(f"Error calculating technical score: {e}")
            return 0.5

    def predict_next_price_range(self, data: pd.DataFrame, news_data: List[Dict]) -> Dict:
        """Generate analysis focused on support/resistance and sentiment"""
        try:
             # Get LLM analysis first
            llm_analysis = self.get_market_analysis(data, news_data)
            # Get support and resistance levels
            key_levels = self.identify_support_resistance(data)
            
            # Calculate sentiment and technical scores
            sentiment_score = self.analyze_market_sentiment(news_data)
            technical_score = self.calculate_technical_score(data)
            
            # Combined market score
            combined_score = sentiment_score * 0.4 + technical_score * 0.6
            
            return {
                'support_levels': key_levels['support_levels'],
                'resistance_levels': key_levels['resistance_levels'],
                'sentiment_score': sentiment_score,
                'technical_score': technical_score,
                'combined_score': combined_score,
                'llm_analysis': llm_analysis.get('analysis', ''),
                'analysis': self._generate_analysis_summary(
                    data, 
                    key_levels,
                    sentiment_score,
                    technical_score,
                    news_data
                )
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._generate_fallback_analysis(data)

    def _generate_analysis_summary(self, data: pd.DataFrame, levels: Dict, 
                                 sentiment: float, technical: float, 
                                 news_data: List[Dict]) -> str:
        """Generate a comprehensive market analysis summary"""
        current_price = data['Close'].iloc[-1]
        
        # Determine market bias
        bias = "bullish" if sentiment > 0.6 else "bearish" if sentiment < 0.4 else "neutral"
        strength = "strong" if abs(sentiment - 0.5) > 0.3 else "moderate" if abs(sentiment - 0.5) > 0.1 else "weak"
        
        analysis = f"""Market Analysis Summary:

        Current Price: ${current_price:,.2f}
        Market Bias: {bias.capitalize()} ({strength})
        
        Technical Levels:
        """
        
        # Add resistance levels
        for i, level in enumerate(levels['resistance_levels'], 1):
            distance = ((level - current_price) / current_price) * 100
            analysis += f"\nResistance {i}: ${level:,.2f} ({distance:+.1f}%)"
            
        # Add support levels
        for i, level in enumerate(levels['support_levels'], 1):
            distance = ((level - current_price) / current_price) * 100
            analysis += f"\nSupport {i}: ${level:,.2f} ({distance:+.1f}%)"
        
        # Add sentiment analysis
        analysis += f"""
        
        Market Sentiment:
        - News Sentiment: {'Positive' if sentiment > 0.6 else 'Negative' if sentiment < 0.4 else 'Neutral'} ({sentiment:.1%})
        - Technical Score: {technical:.1%}
        """
        
        return analysis

    def _generate_fallback_analysis(self, data: pd.DataFrame) -> Dict:
        """Generate basic analysis when main analysis fails"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Simple support/resistance based on recent highs/lows
            highest_prices = data['High'].nlargest(3)
            lowest_prices = data['Low'].nsmallest(3)
            
            return {
                'support_levels': lowest_prices.tolist(),
                'resistance_levels': highest_prices.tolist(),
                'sentiment_score': 0.5,
                'technical_score': 0.5,
                'combined_score': 0.5,
                'analysis': "Unable to generate detailed analysis. Using basic price levels."
            }
            
        except Exception as e:
            print(f"Error in fallback analysis: {e}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'sentiment_score': 0.5,
                'technical_score': 0.5,
                'combined_score': 0.5,
                'analysis': "Analysis currently unavailable."
            }

    def get_market_analysis(self, data: pd.DataFrame, news_data: List[Dict]) -> Dict:
        """Get comprehensive market analysis using Gemini model"""
        try:
            # First check if Gemini is properly initialized
            if not hasattr(self, 'model') or not self.genai_available:
                print("Debug: Gemini model not properly initialized")
                return {
                    'analysis': self._generate_fallback_analysis(data)['analysis'],
                    'status': 'error',
                    'error': 'Gemini API not initialized'
                }

            # Print API key status (don't print the actual key!)
            print(f"Debug: API Key present: {bool(self.gemini_api_key)}")

            # Prepare market context
            market_context = self._prepare_market_context(data)
            print("Debug: Market context prepared successfully")

            # Create detailed prompt
            prompt = f"""As a senior cryptocurrency market analyst, provide a comprehensive analysis of Bitcoin's current market conditions based on the following data:

            Market Overview:
            - Current Price: ${market_context['current_price']:,.2f}
            - 24h Change: {market_context['price_change_24h']:+.2f}%
            - Volume: ${market_context['volume_24h']:,.0f}
            - RSI: {market_context['rsi']:.1f}
            - MACD: {'Bullish' if market_context['macd_signal'] > 0 else 'Bearish'}

            Technical Levels:
            Support Levels: {', '.join(f'${s:,.2f}' for s in market_context['support_levels'])}
            Resistance Levels: {', '.join(f'${r:,.2f}' for r in market_context['resistance_levels'])}

            Market Structure:
            - Trend Direction: {market_context['trend']}
            - Volume Profile: {market_context['volume_profile']}
            - Key Technical Events: {market_context['technical_events']}

            Recent News Impact:
            {market_context['news_summary']}

            Provide a detailed analysis in these sections:
            1. Market Structure Analysis
            - Current market phase
            - Support/resistance significance
            - Volume analysis

            2. Technical Analysis
            - Dominant trend
            - Key indicator signals
            - Pattern formations

            3. Risk Assessment
            - Key price levels
            - Potential scenarios
            - Risk factors

            4. Market Sentiment Analysis
            - Market psychology
            - News impact
            - Trading behavior"""

            print("Debug: Sending request to Gemini API")
            try:
                # Create a chat session and get response
                chat = self.model.start_chat(history=[])
                response = chat.send_message(prompt)
                
                print("Debug: Received response from Gemini API")
                print(f"Debug: Response type: {type(response)}")
                print(f"Debug: Response has text attribute: {hasattr(response, 'text')}")
                
                if hasattr(response, 'text') and response.text:
                    analysis = response.text
                    print("Debug: Successfully parsed response")
                    return {
                        'analysis': analysis,
                        'status': 'success',
                        'market_context': market_context
                    }
                else:
                    print("Debug: Empty response from API")
                    return {
                        'analysis': "Analysis currently unavailable. Empty response from API.",
                        'status': 'error',
                        'error': 'Empty response'
                    }
                    
            except Exception as e:
                print(f"Debug: Error in chat completion: {str(e)}")
                return {
                    'analysis': self._generate_fallback_analysis(data)['analysis'],
                    'status': 'error',
                    'error': str(e)
                }

        except Exception as e:
            print(f"Debug: Error in market analysis: {str(e)}")
            return {
                'analysis': self._generate_fallback_analysis(data)['analysis'],
                'status': 'error',
                'error': str(e)
            }


        

    def _prepare_market_context(self, data: pd.DataFrame) -> Dict:
        """Prepare detailed market context for analysis"""
        current_price = data['Close'].iloc[-1]
        
        # Calculate technical events
        technical_events = []
        
        # Check for golden/death cross
        sma20 = data['Close'].rolling(window=20).mean()
        sma50 = data['Close'].rolling(window=50).mean()
        
        if sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-2] <= sma50.iloc[-2]:
            technical_events.append("Golden Cross (20/50 MA)")
        elif sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-2] >= sma50.iloc[-2]:
            technical_events.append("Death Cross (20/50 MA)")
            
        # Check for oversold/overbought conditions
        if data['RSI'].iloc[-1] > 70:
            technical_events.append("Overbought (RSI)")
        elif data['RSI'].iloc[-1] < 30:
            technical_events.append("Oversold (RSI)")
            
        # Analyze volume profile
        recent_volume_avg = data['Volume'].tail(10).mean()
        historical_volume_avg = data['Volume'].mean()
        volume_status = (
            "Above average" if recent_volume_avg > historical_volume_avg * 1.2
            else "Below average" if recent_volume_avg < historical_volume_avg * 0.8
            else "Average"
        )
        
        # Determine trend
        trend = self._analyze_trend(data)
        
        return {
            'current_price': current_price,
            'price_change_24h': ((current_price - data['Close'].iloc[-24]) / data['Close'].iloc[-24]) * 100,
            'volume_24h': data['Volume'].tail(24).sum(),
            'rsi': data['RSI'].iloc[-1],
            'macd_signal': data['MACD'].iloc[-1] - data['Signal_Line'].iloc[-1],
            'support_levels': self.identify_support_resistance(data)['support_levels'],
            'resistance_levels': self.identify_support_resistance(data)['resistance_levels'],
            'trend': trend,
            'volume_profile': volume_status,
            'technical_events': ", ".join(technical_events) if technical_events else "No significant events",
            'news_summary': self._summarize_recent_news(data)
        }

    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """Analyze market trend using multiple timeframes"""
        trends = []
        
        # Short-term trend (20 periods)
        sma20 = data['Close'].rolling(window=20).mean()
        if data['Close'].iloc[-1] > sma20.iloc[-1]:
            trends.append(('short', 'bullish'))
        else:
            trends.append(('short', 'bearish'))
            
        # Medium-term trend (50 periods)
        sma50 = data['Close'].rolling(window=50).mean()
        if data['Close'].iloc[-1] > sma50.iloc[-1]:
            trends.append(('medium', 'bullish'))
        else:
            trends.append(('medium', 'bearish'))
            
        # Determine overall trend
        bull_count = sum(1 for _, trend in trends if trend == 'bullish')
        if bull_count == len(trends):
            return "Strong Uptrend"
        elif bull_count > len(trends) / 2:
            return "Moderate Uptrend"
        elif bull_count == 0:
            return "Strong Downtrend"
        else:
            return "Moderate Downtrend"

    def _update_significant_events(self, analysis: str):
        """Track significant market events from analysis"""
        # Extract key insights
        key_phrases = [
            "breakout", "breakdown", "reversal",
            "support broken", "resistance broken",
            "trend change", "accumulation", "distribution"
        ]
        
        for phrase in key_phrases:
            if phrase in analysis.lower():
                self.significant_events.append({
                    'timestamp': datetime.now(),
                    'event': phrase,
                    'price': self.current_price,
                    'context': analysis
                })
                
        # Keep only last 10 events
        self.significant_events = self.significant_events[-10:]

    def _summarize_recent_news(self, data: pd.DataFrame) -> str:
        """Summarize recent significant news"""
        if not self.significant_events:
            return "No significant market events recorded"
            
        summary = []
        for event in self.significant_events[-3:]:  # Last 3 events
            price_change = ((data['Close'].iloc[-1] - event['price']) / event['price']) * 100
            summary.append(f"- {event['event'].title()}: {price_change:+.2f}% change since event")
            
        return "\n".join(summary)

    def display_market_analysis(analysis_results, data):
        """Display clear and actionable market analysis"""
        st.subheader("🎯 Market Analysis Summary")
        
        # Get current price and daily change
        current_price = data['Close'].iloc[-1]
        daily_change = ((current_price - data['Close'].iloc[-24]) / data['Close'].iloc[-24]) * 100
        
        # Market Overview
        st.markdown("#### 📊 Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}", f"{daily_change:+.2f}%")
        with col2:
            st.metric("Technical Score", f"{analysis_results['technical_score']:.1%}", 
                    "Bullish" if analysis_results['technical_score'] > 0.5 else "Bearish")
        with col3:
            st.metric("Volume Trend", 
                    "Above Average" if data['Volume'].iloc[-1] > data['Volume'].mean() else "Below Average")

        # Key Technical Levels
        st.markdown("#### 🎯 Key Price Levels")
        
        # Format levels for display
        def format_levels(levels, price):
            return [f"${level:,.2f} ({((level - price)/price * 100):+.2f}%)" for level in levels]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Resistance Levels:**")
            for level in format_levels(analysis_results['resistance_levels'], current_price):
                st.markdown(f"• {level}")
        with col2:
            st.markdown("**Support Levels:**")
            for level in format_levels(analysis_results['support_levels'], current_price):
                st.markdown(f"• {level}")

        # Market Analysis
        st.markdown("#### 💡 Market Insights")
        
        # Technical Analysis
        with st.expander("Technical Analysis", expanded=True):
            st.markdown("""
            **Trend Analysis:**
            - Current Trend: Strong Uptrend
            - Volume Profile: Below average volume suggests caution
            - Key Indicators: RSI = {:.1f}, MACD shows bullish momentum
            """.format(data['RSI'].iloc[-1]))
            
            st.markdown("""
            **Key Observations:**
            - Price is testing major resistance zone
            - Volume needs to increase for trend confirmation
            - Watch for potential reversal signals at resistance
            """)
        
        # Risk Assessment
        with st.expander("Risk Assessment", expanded=True):
            st.markdown("""
            **Risk Factors:**
            - Primary resistance at ${:,.2f}
            - Low volume suggesting weak conviction
            - RSI approaching overbought conditions
            
            **Action Items:**
            - Set stops below ${:,.2f}
            - Watch for volume confirmation
            - Monitor for reversal patterns at resistance
            """.format(
                analysis_results['resistance_levels'][0] if analysis_results['resistance_levels'] else current_price,
                analysis_results['support_levels'][0] if analysis_results['support_levels'] else current_price
            ))

    def generate_market_analysis(data, levels, technical_score):
        """Generate clear and actionable market analysis"""
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        volume_trend = "Above Average" if data['Volume'].iloc[-1] > data['Volume'].mean() else "Below Average"
        
        analysis = f"""
    Market Structure Analysis:
    -------------------------
    • Current Market Phase: {'Bullish' if technical_score > 0.5 else 'Bearish'} Trend
    - Price Action: {'Testing resistance' if technical_score > 0.5 else 'Testing support'} at ${current_price:,.2f}
    - Volume Profile: {volume_trend}, suggesting {'strong' if volume_trend == 'Above Average' else 'weak'} conviction

    Technical Analysis:
    -----------------
    • Key Indicators:
    - RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
    - MACD: {'Bullish' if data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1] else 'Bearish'} momentum
    - Volume: {volume_trend} with {'strengthening' if data['Volume'].iloc[-1] > data['Volume'].iloc[-2] else 'weakening'} trend

    Risk Assessment:
    --------------
    • Critical Levels:
    - Key Resistance: ${levels['resistance_levels'][0]:,.2f} (+{((levels['resistance_levels'][0] - current_price)/current_price * 100):,.1f}%)
    - Key Support: ${levels['support_levels'][0]:,.2f} ({((levels['support_levels'][0] - current_price)/current_price * 100):,.1f}%)

    • Risk Factors:
    - {'Volume needs to increase for trend confirmation' if volume_trend == 'Below Average' else 'Strong volume supports current trend'}
    - {'RSI approaching overbought conditions' if rsi > 65 else 'RSI approaching oversold conditions' if rsi < 35 else 'RSI in neutral territory'}

    Trading Implications:
    ------------------
    • Short-term Outlook: {'Bullish' if technical_score > 0.6 else 'Bearish' if technical_score < 0.4 else 'Neutral'}
    • Key Action Points:
    - Monitor volume for trend confirmation
    - Watch for price action at ${levels['resistance_levels'][0]:,.2f}
    - Set stops below ${levels['support_levels'][0]:,.2f}
    """
        return analysis