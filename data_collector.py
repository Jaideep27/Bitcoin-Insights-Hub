# backend/data_collector.py
import yfinance as yf
import requests
from datetime import datetime, timedelta
import pandas as pd

class DataCollector:
    def __init__(self):
        self.crypto_url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

    def get_bitcoin_data(self, period="7d", interval="1h"):
        """Fetch Bitcoin price data"""
        try:
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching Bitcoin data: {e}")
            return None

    def get_crypto_news(self):
    # Fetch crypto news
        try:
            response = requests.get(self.crypto_url)
            if response.status_code == 200:
                news_data = response.json().get('Data', [])
                # Return only the first 5 items as a list
                return list(news_data)[:5] if news_data else []
            return []
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []