from typing import Dict, Any
import requests
import time
import streamlit as st

# Function outside the class to enable caching
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def fetch_bitcoin_data(base_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """Fetch Bitcoin data from CoinGecko"""
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(base_delay * (2 ** attempt))

            # Get basic price data
            simple_price_url = f"{base_url}/simple/price"
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }

            price_response = requests.get(simple_price_url, params=params, headers=headers)
            if price_response.status_code == 429:
                print(f"Rate limit hit (attempt {attempt + 1}), waiting...")
                continue

            price_response.raise_for_status()
            price_data = price_response.json().get('bitcoin', {})

            time.sleep(1)

            # Get detailed data
            detailed_url = f"{base_url}/coins/bitcoin"
            detailed_response = requests.get(detailed_url, headers=headers)
            if detailed_response.status_code == 429:
                print(f"Rate limit hit (attempt {attempt + 1}), waiting...")
                continue

            detailed_response.raise_for_status()
            detailed_data = detailed_response.json()
            market_data = detailed_data.get('market_data', {})

            return {
                'market_cap_rank': detailed_data.get('market_cap_rank', 1),
                'market_cap': price_data.get('usd_market_cap'),
                'volume_24h': price_data.get('usd_24h_vol'),
                'circulating_supply': market_data.get('circulating_supply'),
                'max_supply': 21_000_000,
                'current_price': price_data.get('usd'),
                'price_change_24h': price_data.get('usd_24h_change'),
                'roi_data': {
                    'percent_change_7d': market_data.get('price_change_percentage_7d'),
                    'percent_change_30d': market_data.get('price_change_percentage_30d')
                }
            }

        except requests.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {
                    'market_cap_rank': 1,
                    'market_cap': None,
                    'volume_24h': None,
                    'circulating_supply': None,
                    'max_supply': 21_000_000,
                    'current_price': None,
                    'price_change_24h': None,
                    'roi_data': {
                        'percent_change_7d': None,
                        'percent_change_30d': None
                    }
                }


class BitcoinInformationCollector:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {'accept': 'application/json'}

    def get_bitcoin_info(self) -> Dict[str, Any]:
        """Public method to fetch Bitcoin data using the cached function"""
        return fetch_bitcoin_data(self.base_url, self.headers)
