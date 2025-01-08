
# frontend/app.py
import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import re 

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Bitcoin Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .data-point {
        margin: 0.75rem 0;
        line-height: 1.6;
    }
    
    .label {
        color: #94A3B8;
        font-weight: 500;
    }
    
    .highlight {
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .highlight.trend {
        background-color: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
    }
    
    .highlight.technical {
        background-color: rgba(168, 85, 247, 0.1);
        color: #A855F7;
    }
    
    .highlight.price {
        background-color: rgba(34, 197, 94, 0.1);
        color: #22C55E;
    }
    
    .highlight.volume {
        background-color: rgba(249, 115, 22, 0.1);
        color: #F97316;
    }
    
    .highlight.pattern {
        background-color: rgba(236, 72, 153, 0.1);
        color: #EC4899;
    }
    
    .highlight.number {
        background-color: rgba(71, 85, 105, 0.1);
        color: #CBD5E1;
        font-family: 'Monaco', 'Consolas', monospace;
    }
    
    .analysis-section {
        background-color: rgba(17, 25, 40, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: #38BDF8;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .analysis-text {
        color: #CBD5E1;
        line-height: 1.8;
    }
    </style>
""", unsafe_allow_html=True)

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# Now import backend components
try:
    from backend.data_collector import DataCollector
    from backend.database import Database
    from backend.market_predictor import MarketPredictor
    from backend.technical_indicators import TechnicalIndicators
    from backend.bitcoin_information_collector import BitcoinInformationCollector
    from backend.futures_calculator import display_futures_calculator
except ImportError as e:
    st.error(f"""Error importing backend modules: {e}
    
    Current directory: {current_dir}
    Project root: {project_root}
    Python path: {sys.path}
    """)
    raise

# Initialize components
data_collector = DataCollector()
db = Database()
predictor = MarketPredictor()
bitcoin_collector = BitcoinInformationCollector()  # Changed name for consistency

# Initialize session state
if 'bitcoin_info' not in st.session_state:
    st.session_state.bitcoin_info = {}  # Initialize as empty dict instead of None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = None

def load_bitcoin_info():
    """Load Bitcoin information into session state"""
    try:
        info = bitcoin_collector.get_bitcoin_info()  # Use consistent name
        if info:
            st.session_state.bitcoin_info = info
        else:
            st.warning("Unable to fetch latest data, using cached data")
            if 'bitcoin_info' not in st.session_state:
                st.session_state.bitcoin_info = {}
    except Exception as e:
        st.error(f"Error loading Bitcoin data: {e}")
        if 'bitcoin_info' not in st.session_state:
            st.session_state.bitcoin_info = {}

def format_crypto_value(value, prefix="", suffix=""):
    """Format cryptocurrency values"""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, (int, float)):
            if value >= 1_000_000_000:
                return f"{prefix}{value/1_000_000_000:.2f}B{suffix}"
            elif value >= 1_000_000:
                return f"{prefix}{value/1_000_000:.2f}M{suffix}"
            return f"{prefix}{value:,.2f}{suffix}"
    except:
        return "N/A"
    return f"{prefix}{value}{suffix}"

def display_bitcoin_info():
    """Display Bitcoin information in sidebar"""
    info = st.session_state.get('bitcoin_info', {})
    
    # Market Stats with error handling
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Market Cap Rank</div>
            <div class="info-value">#{info.get('market_cap_rank', 'N/A')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Market Cap</div>
            <div class="info-value">{format_crypto_value(info.get('market_cap'), prefix='$')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">24h Volume</div>
            <div class="info-value">{format_crypto_value(info.get('volume_24h'), prefix='$')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Circulating Supply</div>
            <div class="info-value">{format_crypto_value(info.get('circulating_supply'), suffix=' BTC')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Max Supply</div>
            <div class="info-value">21,000,000 BTC</div>
        </div>""", 
        unsafe_allow_html=True
    )

# Load initial bitcoin info
load_bitcoin_info()


def display_sidebar_content():
    """Display sidebar content with Bitcoin information"""
    st.sidebar.title("Bitcoin Dashboard")
    
    # Refresh button with loading state
    if st.sidebar.button("🔄 Refresh Data", key="refresh_data_btn"):
        with st.sidebar:
            with st.spinner("Fetching latest data..."):
                load_bitcoin_info()
    
    # Bitcoin Market Info
    info = st.session_state.get('bitcoin_info', {})
    if info is None:
        info = {}  # Ensure info is a dictionary even if None
    
    # Market Stats with error handling
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Market Cap Rank</div>
            <div class="info-value">#{info.get('market_cap_rank', 'N/A')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Market Cap</div>
            <div class="info-value">{format_crypto_value(info.get('market_cap'), prefix='$')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">24h Volume</div>
            <div class="info-value">{format_crypto_value(info.get('volume_24h'), prefix='$')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Circulating Supply</div>
            <div class="info-value">{format_crypto_value(info.get('circulating_supply'), suffix=' BTC')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Max Supply</div>
            <div class="info-value">21,000,000 BTC</div>
        </div>""", 
        unsafe_allow_html=True
    )

    # Auto-refresh Settings
    st.sidebar.markdown("### Settings")
    auto_refresh = st.sidebar.checkbox('Enable Auto-refresh', value=False, key="auto_refresh_checkbox")
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            'Refresh Interval (seconds)',
            min_value=30,
            max_value=300,
            value=60,
            key="refresh_interval_slider"
        )
        time.sleep(refresh_interval)
        st.experimental_rerun()

def create_price_chart(data):
    """Create main price and volume chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='BTC'
        ),
        row=1, col=1
    )

    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_upper'],
            name='Upper BB',
            line=dict(color='gray', dash='dash'),
            opacity=0.5
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_lower'],
            name='Lower BB',
            line=dict(color='gray', dash='dash'),
            opacity=0.5,
            fill='tonexty'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['rgb(239, 83, 80)' if close < open else 'rgb(102, 187, 106)' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig


def create_technical_indicators_chart(data):
    """Create technical indicators chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RSI', 'MACD', 'Stochastic', 'OBV')
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name='MACD'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal'),
        row=1, col=2
    )

    # Stochastic
    fig.add_trace(
        go.Scatter(x=data.index, y=data['%K'], name='%K'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['%D'], name='%D'),
        row=2, col=1
    )

    # OBV
    fig.add_trace(
        go.Scatter(x=data.index, y=data['OBV'], name='OBV'),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True
    )

    return fig


    """Display combined market overview with technical and sentiment analysis"""
    st.subheader("Overall Market Sentiment")
    
    # Calculate combined score
    combined_score = (technical_score * 0.6) + (sentiment_score * 0.4)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=combined_score * 100,
        title={'text': "Combined Market Score", 'font': {'color': 'white'}},
        number={'suffix': "%", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "royalblue"},
            'steps': [
                {'range': [0, 40], 'color': "rgb(239, 83, 80)"},
                {'range': [40, 60], 'color': "rgb(128, 128, 128)"},
                {'range': [60, 100], 'color': "rgb(102, 187, 106)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display component scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Technical Analysis (60%)",
            f"{technical_score:.1%}",
            "Market Structure & Indicators",
            delta_color='inverse' if technical_score < 0.5 else 'normal'
        )
    with col2:
        st.metric(
            "Market Sentiment (40%)",
            f"{sentiment_score:.1%}",
            "News & Social Sentiment",
            delta_color='inverse' if sentiment_score < 0.5 else 'normal'
        )


def display_market_overview(technical_score, sentiment_score, analysis_results):
    """Display combined market overview with technical and sentiment analysis"""
    # Add custom CSS for better styling
    st.markdown("""
        <style>
            /* Base Analysis Styles */
            .analysis-header {
                color: #E2E8F0;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            /* Enhanced Text Formatting */
            .analysis-text {
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                line-height: 1.8;
                letter-spacing: 0.01em;
            }
            
            /* Category-specific Highlights */
            .highlight {
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-weight: 500;
            }
            
            .highlight.trend_terms {
                background-color: rgba(56, 189, 248, 0.1);
                color: #38BDF8;
            }
            
            .highlight.technical_terms {
                background-color: rgba(168, 85, 247, 0.1);
                color: #A855F7;
            }
            
            .highlight.price_levels {
                background-color: rgba(34, 197, 94, 0.1);
                color: #22C55E;
            }
            
            .highlight.volume_terms {
                background-color: rgba(249, 115, 22, 0.1);
                color: #F97316;
            }
            
            .highlight.indicators {
                background-color: rgba(236, 72, 153, 0.1);
                color: #EC4899;
            }
            
            .highlight.price {
                background-color: rgba(34, 197, 94, 0.1);
                color: #22C55E;
                font-family: 'Monaco', 'Consolas', monospace;
            }
            
            .highlight.percentage {
                background-color: rgba(56, 189, 248, 0.1);
                color: #38BDF8;
                font-family: 'Monaco', 'Consolas', monospace;
            }
            
            /* Improved Bullet Points */
            .bullet-point {
                display: flex;
                align-items: flex-start;
                margin: 0.75rem 0;
                line-height: 1.6;
            }
            
            .bullet {
                color: #38BDF8;
                margin-right: 0.75rem;
                font-size: 1.2em;
            }
            
            .bullet-content {
                flex: 1;
                color: #CBD5E1;
            }
            .analysis-section {
                background-color: rgba(17, 25, 40, 0.75);
                border: 1px solid rgba(56, 189, 248, 0.2);
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .section-title {
                color: #38BDF8;
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
            }
            .section-icon {
                margin-right: 0.5rem;
                color: #38BDF8;
            }
            .section-content {
                color: #CBD5E1;
                line-height: 1.8;
                font-size: 1.05rem;
            }
            .highlight {
                background-color: rgba(56, 189, 248, 0.1);
                color: #38BDF8;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-weight: 500;
            }
            .info-box {
                background-color: rgba(56, 189, 248, 0.1);
                border-left: 4px solid #38BDF8;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0 4px 4px 0;
            }
            .bullet-list {
                margin: 1rem 0;
                padding-left: 1.5rem;
            }
            .bullet-list li {
                margin-bottom: 0.75rem;
                line-height: 1.6;
            }
            .bullet-point::before {
                content: "•";
                color: #38BDF8;
                font-weight: bold;
                display: inline-block;
                width: 1em;
                margin-left: -1em;
            }
            .sentiment-gauge {
                margin: 2rem 0;
            }
            .timestamp {
                color: #64748B;
                font-size: 0.875rem;
                text-align: right;
                margin-top: 2rem;
                font-style: italic;
            }
        </style>
    """, unsafe_allow_html=True)

    # Market Sentiment Gauge
    st.markdown("<h2 class='analysis-header'>Market Overview</h2>", unsafe_allow_html=True)
    
    # Calculate combined score
    combined_score = (technical_score * 0.6) + (sentiment_score * 0.4)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=combined_score * 100,
        title={'text': "Combined Market Score", 'font': {'color': 'white', 'size': 24}},
        number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "royalblue"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(239, 83, 80, 0.7)"},
                {'range': [40, 60], 'color': "rgba(128, 128, 128, 0.7)"},
                {'range': [60, 100], 'color': "rgba(102, 187, 106, 0.7)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        margin=dict(t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Display component scores with improved layout
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Technical Analysis (60%)",
            f"{technical_score:.1%}",
            "Market Structure & Indicators",
            delta_color='normal' if technical_score > 0.5 else 'inverse'
        )
    with col2:
        st.metric(
            "Market Sentiment (40%)",
            f"{sentiment_score:.1%}",
            "News & Social Sentiment",
            delta_color='normal' if sentiment_score > 0.5 else 'inverse'
        )

    # Display LLM Analysis
    st.markdown("<h2 class='analysis-header'>Detailed Market Analysis</h2>", unsafe_allow_html=True)
    
    try:
        llm_analysis = analysis_results.get('llm_analysis', '')
        if llm_analysis:
            # Process and display each section
            sections = parse_llm_analysis(llm_analysis)
            section_icons = {
                "Market Structure": "📊",
                "Technical Analysis": "📈",
                "Risk Assessment": "⚠️",
                "Market Sentiment": "🔍"
            }
            
            for section in sections:
                with st.expander(section['title'], expanded=True):
                    icon = section_icons.get(section['title'].split()[0], "📝")
                    st.markdown(
                        f"<div class='analysis-section'>"
                        f"<div class='section-title'><span class='section-icon'>{icon}</span>{section['title']}</div>"
                        f"<div class='section-content'>{format_analysis_content(section['content'])}</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
            
            # Add timestamp
            st.markdown(
                f"<div class='timestamp'>Analysis generated on: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
                unsafe_allow_html=True
            )
        else:
            display_no_analysis_warning()
            
    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        st.button("Retry Analysis", on_click=st.experimental_rerun)


def display_technical_levels(data, analysis_results):
    """Display support and resistance levels with enhanced formatting"""
    st.markdown("""
        <style>
        .level-card {
            background-color: rgba(17, 25, 40, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .level-flex {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .level-label {
            color: #38BDF8;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .level-price {
            font-family: 'Monaco', 'Consolas', monospace;
            color: #E2E8F0;
            font-size: 1.1rem;
        }
        .level-distance {
            font-size: 0.9rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .distance-positive {
            color: #22C55E;
            background-color: rgba(34, 197, 94, 0.1);
        }
        .distance-negative {
            color: #EF4444;
            background-color: rgba(239, 68, 68, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Technical Levels")
    col1, col2 = st.columns(2)
    
    current_price = data['Close'].iloc[-1]
    
    with col1:
        st.markdown("#### Support Levels")
        for i, level in enumerate(analysis_results.get('support_levels', [])[:3], 1):
            distance = ((level - current_price) / current_price) * 100
            distance_class = "distance-positive" if distance > 0 else "distance-negative"
            
            st.markdown(f"""
            <div class="level-card">
                <div class="level-flex">
                    <span class="level-label">S{i}</span>
                    <span class="level-price">${level:,.2f}</span>
                    <span class="level-distance {distance_class}">
                        {'-' if distance < 0 else '+'}{abs(distance):.2f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Resistance Levels")
        for i, level in enumerate(analysis_results.get('resistance_levels', [])[:3], 1):
            distance = ((level - current_price) / current_price) * 100
            distance_class = "distance-positive" if distance > 0 else "distance-negative"
            
            st.markdown(f"""
            <div class="level-card">
                <div class="level-flex">
                    <span class="level-label">R{i}</span>
                    <span class="level-price">${level:,.2f}</span>
                    <span class="level-distance {distance_class}">
                        {'-' if distance < 0 else '+'}{abs(distance):.2f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Add description box
    st.markdown("""
        <div style="background-color: rgba(56, 189, 248, 0.1); border-left: 4px solid #38BDF8; 
                    padding: 1rem; margin-top: 1rem; border-radius: 0 4px 4px 0;">
            <div style="color: #38BDF8; font-weight: 600; margin-bottom: 0.5rem;">
                Understanding Technical Levels
            </div>
            <div style="color: #CBD5E1; line-height: 1.6;">
                • Support levels (S1-S3) indicate potential price floors where buying pressure may increase<br>
                • Resistance levels (R1-R3) show potential ceiling prices where selling pressure may increase<br>
                • Percentages show distance from current price - useful for risk/reward assessment
            </div>
        </div>
    """, unsafe_allow_html=True)

def parse_llm_analysis(text):
    """Parse LLM analysis with clean formatting and no duplicates"""
    if not text or not isinstance(text, str):
        return [{'title': 'Market Analysis', 'content': 'No analysis available', 'main_section': 'Analysis'}]

    try:
        # Define markers for sections to ignore
        skip_markers = [
            "Here's a breakdown",
            "The prompt provides",
            "Process Each Data Point",
            "Bitcoin Market Analysis",
            "Analyst:",
            "Review and Edit",
            "Summarize the key takeaways"
        ]

        # Define technical terms for highlighting
        term_categories = {
            'trend': ['bullish', 'bearish', 'uptrend', 'downtrend'],
            'technical': ['RSI', 'MACD', 'momentum', 'oversold', 'overbought'],
            'price': ['support', 'resistance', 'breakout', 'breakdown'],
            'volume': ['volume', 'liquidity', 'accumulation', 'distribution'],
            'pattern': ['consolidation', 'reversal', 'continuation', 'divergence']
        }

        # Clean up text and split into lines
        text = text.replace('**', '').strip()
        lines = []
        seen_content = set()

        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Skip template lines and duplicates
            if any(marker.lower() in line.lower() for marker in skip_markers):
                continue

            # Clean and normalize line
            cleaned_line = clean_content_line(line)
            if cleaned_line and cleaned_line not in seen_content:
                lines.append(cleaned_line)
                seen_content.add(cleaned_line)

        # Group into sections
        sections = []
        current_section = None
        current_content = []

        section_keywords = {
            'Market Structure': ['market structure', 'price action', 'volume analysis'],
            'Technical Analysis': ['technical', 'indicator', 'pattern', 'oscillator'],
            'Risk Assessment': ['risk', 'key levels', 'scenarios', 'caution'],
            'Market Sentiment': ['sentiment', 'outlook', 'bias', 'psychology']
        }

        for line in lines:
            # Check if line is a section header
            is_header = False
            section_type = 'Analysis'
            
            for section, keywords in section_keywords.items():
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    is_header = True
                    section_type = section
                    break

            # Data point formatting
            if ':' in line and not is_header:
                label, value = line.split(':', 1)
                if any(term.lower() in value.lower() for terms in term_categories.values() for term in terms):
                    formatted_line = format_data_point(label.strip(), value.strip(), term_categories)
                    current_content.append(formatted_line)
                else:
                    current_content.append(f"• {line}")
            else:
                current_content.append(line)

        # Create final sections
        if current_content:
            sections.append({
                'title': 'Market Analysis',
                'content': '\n'.join(current_content),
                'main_section': 'Analysis'
            })

        return sections

    except Exception as e:
        print(f"Error parsing analysis: {str(e)}")
        return [{'title': 'Market Analysis', 'content': text, 'main_section': 'Analysis'}]

def format_data_point(label, value, term_categories):
    """Format data points with proper highlighting"""
    formatted_value = value
    
    # Highlight technical terms
    for category, terms in term_categories.items():
        for term in terms:
            pattern = re.compile(f'\\b{term}\\b', re.IGNORECASE)
            formatted_value = pattern.sub(
                f'<span class="highlight {category}">{term}</span>',
                formatted_value
            )
    
    # Format numbers and percentages
    formatted_value = re.sub(
        r'(\$[\d,]+\.?\d*|[-+]?\d+\.?\d*%)',
        r'<span class="highlight number">\1</span>',
        formatted_value
    )
    
    return f'<div class="data-point"><span class="label">{label}:</span> {formatted_value}</div>'



def display_llm_analysis(analysis_results):
    """Display LLM analysis with enhanced error handling"""
    st.markdown("""
        <style>
            .analysis-section {
                background-color: rgba(17, 25, 40, 0.75);
                border: 1px solid rgba(56, 189, 248, 0.2);
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }
            .section-title {
                color: #38BDF8;
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            .analysis-text {
                color: #CBD5E1;
                line-height: 1.8;
                font-size: 1.05rem;
            }
            .highlight {
                background-color: rgba(56, 189, 248, 0.1);
                color: #38BDF8;
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-weight: 500;
            }
            .bullet-point {
                margin: 0.75rem 0;
                padding-left: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        llm_analysis = analysis_results.get('llm_analysis', '')
        if not llm_analysis:
            st.warning("No analysis data available.")
            return

        sections = parse_llm_analysis(llm_analysis)
        
        # Group sections by main section
        section_groups = {}
        for section in sections:
            main_section = section['main_section']
            if main_section not in section_groups:
                section_groups[main_section] = []
            section_groups[main_section].append(section)

        # Display each main section
        for main_section, subsections in section_groups.items():
            if main_section != 'Analysis':  # Skip the default group if there are others
                with st.expander(main_section, expanded=True):
                    for section in subsections:
                        st.markdown(
                            f'''
                            <div class="analysis-section">
                                <div class="section-title">{section["title"]}</div>
                                <div class="analysis-text">{format_analysis_content(section["content"])}</div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )

        # Show timestamp
        st.markdown(
            f'<div style="color: #64748B; font-size: 0.875rem; text-align: right; margin-top: 2rem;">'
            f'Analysis generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        st.code(f"Error details: {str(e)}")
        if st.button("Retry Analysis"):
            st.experimental_rerun()

def clean_title(title):
    """Clean section titles with error handling"""
    if not title:
        return "Analysis"
    
    try:
        # Remove special characters and formatting
        title = title.replace('**', '')
        title = re.sub(r'^\d+\.\s*', '', title)  # Remove leading numbers
        title = title.split(':', 1)[-1] if ':' in title else title  # Keep part after colon
        title = ' '.join(word.strip() for word in title.split())  # Clean up whitespace
        return title.strip() or "Analysis"
    except:
        return "Analysis"

def clean_content_line(line):
    """Clean content lines with error handling"""
    if not line:
        return ""
    
    try:
        # Remove formatting and normalize spacing
        line = line.replace('**', '')
        line = re.sub(r'\s+', ' ', line)  # Normalize spaces
        return line.strip()
    except:
        return line if line else ""

def format_analysis_content(content):
    """Format analysis content with error handling"""
    if not content:
        return "No content available"
        
    try:
        # Replace bullet points
        content = re.sub(r'^[-*]\s+', '• ', content, flags=re.MULTILINE)
        
        # Highlight key terms
        highlight_terms = [
            'bullish', 'bearish', 'support', 'resistance',
            'trend', 'breakout', 'breakdown', 'momentum',
            'volume', 'consolidation', 'RSI', 'MACD'
        ]
        
        for term in highlight_terms:
            content = re.sub(
                f'\\b{term}\\b',
                f'<span class="highlight">{term}</span>',
                content,
                flags=re.IGNORECASE
            )
            
        # Format bullet points
        content = '<br>'.join(
            f'<div class="bullet-point">• {line[2:]}</div>' if line.startswith('• ') else line
            for line in content.split('\n')
        )
        
        return content
    except:
        return content


def display_news_analysis(news_items, predictor):
    """Display news analysis with sentiment scoring and enhanced formatting"""
    if not news_items:
        st.warning("No news data available")
        return

    # Add custom CSS
    st.markdown("""
        <style>
        .news-card {
            background-color: rgba(17, 25, 40, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .news-title {
            color: #E2E8F0;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .news-content {
            color: #CBD5E1;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        .news-metadata {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            color: #64748B;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 1rem;
            margin-top: 1rem;
        }
        .sentiment-positive {
            color: #22C55E;
            background-color: rgba(34, 197, 94, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .sentiment-negative {
            color: #EF4444;
            background-color: rgba(239, 68, 68, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .sentiment-neutral {
            color: #94A3B8;
            background-color: rgba(148, 163, 184, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }
        .sentiment-gauge {
            margin: 2rem 0;
            padding: 1rem;
            background-color: rgba(17, 25, 40, 0.75);
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### Recent News Analysis")
    
    valid_sentiments = []

    for item in news_items:
        with st.expander(f"📄 {item['title']}", expanded=False):
            try:
                # Calculate sentiment
                sentiment = predictor.analyze_market_sentiment([item])
                sentiment_score = max(0.0, min(1.0, float(sentiment)))
                valid_sentiments.append(sentiment_score)
                
                # Determine sentiment label and style
                if sentiment_score > 0.6:
                    sentiment_label = "Positive"
                    sentiment_class = "sentiment-positive"
                elif sentiment_score < 0.4:
                    sentiment_label = "Negative"
                    sentiment_class = "sentiment-negative"
                else:
                    sentiment_label = "Neutral"
                    sentiment_class = "sentiment-neutral"
                
                # Display news card
                st.markdown(f"""
                    <div class="news-card">
                        <div class="news-title">{item['title']}</div>
                        <div class="news-content">{item['body']}</div>
                        <div class="news-metadata">
                            <div>
                                Source: {item.get('source', 'Unknown')} | 
                                Published: {datetime.fromtimestamp(item['published_on']).strftime('%Y-%m-%d %H:%M:%S')}
                            </div>
                            <div class="{sentiment_class}">
                                {sentiment_label} ({sentiment_score:.1%})
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show sentiment gauge
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(sentiment_score)
                with col2:
                    st.markdown(f'<span class="{sentiment_class}">{sentiment_label}</span>', 
                              unsafe_allow_html=True)
                
                # Save to database if available
                try:
                    if 'db' in globals():
                        db.save_news_sentiment(item, {
                            'label': sentiment_label,
                            'score': sentiment_score
                        })
                except Exception as e:
                    st.error(f"Error saving sentiment: {e}")
                    
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")

    # Display overall sentiment if we have valid data
    if valid_sentiments:
        avg_sentiment = sum(valid_sentiments) / len(valid_sentiments)
        avg_sentiment = max(0.0, min(1.0, avg_sentiment))
        
        st.markdown("### Overall News Sentiment")
        
        # Create gauge chart for overall sentiment
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_sentiment * 100,
            title={'text': "News Sentiment", 'font': {'color': 'white', 'size': 24}},
            number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.3)"},
                    {'range': [40, 60], 'color': "rgba(148, 163, 184, 0.3)"},
                    {'range': [60, 100], 'color': "rgba(34, 197, 94, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            template="plotly_dark",
            margin=dict(l=30, r=30, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def load_bitcoin_info():
    """Load Bitcoin information into session state"""
    try:
        st.session_state.bitcoin_info = btc_info.get_bitcoin_info()
    except Exception as e:
        st.error(f"Error loading Bitcoin data: {e}")
        st.session_state.bitcoin_info = None

def format_crypto_value(value, prefix="", suffix=""):
    """Format cryptocurrency values"""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, (int, float)):
            if value >= 1_000_000_000:
                return f"{prefix}{value/1_000_000_000:.2f}B{suffix}"
            elif value >= 1_000_000:
                return f"{prefix}{value/1_000_000:.2f}M{suffix}"
            return f"{prefix}{value:,.2f}{suffix}"
    except:
        return "N/A"
    return f"{prefix}{value}{suffix}"

def display_bitcoin_info():
    """Display Bitcoin information in sidebar"""
    info = st.session_state.get('bitcoin_info', {})
    
    st.sidebar.markdown("### Market Overview")
    
    # Market Cap Rank
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Market Cap Rank</div>
            <div class="info-value">#{info.get('market_cap_rank', 'N/A')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Market Cap
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Market Cap</div>
            <div class="info-value">{format_crypto_value(info.get('market_cap'), prefix='$')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    # 24h Volume
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">24h Volume</div>
            <div class="info-value">{format_crypto_value(info.get('volume_24h'), prefix='$')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Circulating Supply
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Circulating Supply</div>
            <div class="info-value">{format_crypto_value(info.get('circulating_supply'), suffix=' BTC')}</div>
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Max Supply
    st.sidebar.markdown(
        f"""<div class="info-card">
            <div class="info-label">Max Supply</div>
            <div class="info-value">21,000,000 BTC</div>
        </div>""", 
        unsafe_allow_html=True
    )



def main():

    api_key = os.getenv('GOOGLE_API_KEY')
    print(f"API Key present: {bool(api_key)}")

    st.title("Bitcoin Analysis Dashboard")
    
    # Controls section with unique keys
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        timeframe = st.selectbox(
            "Select Timeframe",
            ["24h", "7d", "30d"],
            index=1,
            key="main_timeframe_select"  # Updated unique key
        )
    with col2:
        interval = st.selectbox(
            "Select Interval",
            ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"],
            index=6,
            key="main_interval_select"  # Updated unique key
        )
    with col3:
        if st.button("🔄 Refresh Analysis", key="main_refresh_btn"):  # Updated unique key
            st.experimental_rerun()

    # Fetch data
    with st.spinner("Fetching market data..."):
        try:
            st.session_state.data = data_collector.get_bitcoin_data(
                period=timeframe,
                interval=interval
            )
            st.session_state.news_data = data_collector.get_crypto_news()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

    if st.session_state.data is None:
        st.error("Unable to fetch market data. Please try again later.")
        return

    # Calculate indicators
    data = TechnicalIndicators.calculate_all(st.session_state.data)
    analysis_results = predictor.predict_next_price_range(data, st.session_state.news_data)

    # Create tabs - Updated with new Futures Calculator tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Price Data",
        "📊 Technical Analysis",
        "📰 News Analysis",
        "🎯 Market Overview",
        "💹 Futures Calculator"
    ])

    with tab1:
        # Display current price
        current_price = data['Close'].iloc[-1]
        col1, col2 = st.columns([3, 1])
        with col2:
            st.metric(
                "Current BTC Price",
                f"${current_price:,.2f}",
                delta=f"{((current_price - data['Close'].iloc[-2])/data['Close'].iloc[-2]*100):.2f}%",
                delta_color="normal" if current_price >= data['Close'].iloc[-2] else "inverse"
            )
        
        

        # Price chart
        fig_price = create_price_chart(data)
        st.plotly_chart(fig_price, use_container_width=True)

        # Technical indicators
        st.subheader("Technical Indicators")
        fig_indicators = create_technical_indicators_chart(data)
        st.plotly_chart(fig_indicators, use_container_width=True)

    with tab2:
        # Technical Analysis
        st.metric(
            "Technical Score",
            f"{analysis_results['technical_score']:.1%}",
            delta="Based on technical indicators",
            delta_color='normal' if analysis_results['technical_score'] > 0.5 else 'inverse'
        )
        
        # Support and Resistance levels
        display_technical_levels(data, analysis_results)

    with tab3:
        display_news_analysis(st.session_state.news_data, predictor)

    with tab4:
        # Print debug info
        print("Debug - Analysis Results:", analysis_results)
        display_market_overview(
            analysis_results['technical_score'],
            analysis_results['sentiment_score'],
            analysis_results
        )
    
    with tab5:
        # New Futures Calculator tab
        display_futures_calculator()

    # Sidebar content
    display_sidebar_content()




def display_sidebar_content():
    """Display sidebar content with Bitcoin information"""
    st.sidebar.title("Bitcoin Dashboard")
    
    # Refresh button with loading state
    if st.sidebar.button("🔄 Refresh Data", key="refresh_data_btn"):
        with st.sidebar:
            with st.spinner("Fetching latest data..."):
                load_bitcoin_info()
    
    # Bitcoin Market Info
    info = st.session_state.bitcoin_info
    
    if info is None:
        load_bitcoin_info()
        info = st.session_state.bitcoin_info or {}

    # Market Stats with error handling
    display_bitcoin_info()

    # Auto-refresh Settings
    st.sidebar.markdown("### Settings")
    auto_refresh = st.sidebar.checkbox('Enable Auto-refresh', value=False, key="auto_refresh")
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            'Refresh Interval (seconds)',
            min_value=30,
            max_value=300,
            value=60,
            key="refresh_interval"
        )
        time.sleep(refresh_interval)
        st.experimental_rerun()

def display_llm_analysis(analysis_results):
    """Display enhanced LLM analysis with better formatting and structure"""
    st.markdown("""
        <style>
        .analysis-header {
            color: #E2E8F0;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .analysis-section {
            background-color: rgba(17, 25, 40, 0.75);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .section-title {
            color: #38BDF8;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .section-content {
            color: #CBD5E1;
            line-height: 1.6;
            font-size: 1rem;
        }
        .highlight {
            background-color: rgba(56, 189, 248, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            color: #38BDF8;
        }
        .analysis-metrics {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .bullet-point {
            color: #38BDF8;
            margin-right: 0.5rem;
        }
        .info-box {
            background-color: rgba(56, 189, 248, 0.1);
            border-left: 4px solid #38BDF8;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 4px 4px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='analysis-header'>Detailed Market Analysis</h2>", unsafe_allow_html=True)

    try:
        llm_analysis = analysis_results.get('llm_analysis', '')
        if llm_analysis:
            # Split analysis into sections
            sections = parse_analysis_sections(llm_analysis)
            
            # Process each section
            for section in sections:
                with st.expander(section['title'], expanded=True):
                    st.markdown(f"<div class='analysis-section'>", unsafe_allow_html=True)
                    
                    # Format content with bullet points and highlights
                    formatted_content = format_section_content(section['content'])
                    st.markdown(formatted_content, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

            # Add timestamp footer
            st.markdown(
                f"<div style='text-align: right; color: #64748B; font-size: 0.875rem; margin-top: 2rem;'>"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
                unsafe_allow_html=True
            )
        else:
            display_no_analysis_warning()
    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        if st.button("Retry Analysis"):
            st.experimental_rerun()

def parse_analysis_sections(analysis_text):
    """Parse analysis text into structured sections"""
    sections = []
    current_section = None
    current_content = []
    
    for line in analysis_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a section header
        if line.endswith(':') or line.startswith(('1.', '2.', '3.', '4.')):
            if current_section:
                sections.append({
                    'title': current_section,
                    'content': '\n'.join(current_content)
                })
            current_section = line.split('.', 1)[-1].strip(':').strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Add the last section
    if current_section:
        sections.append({
            'title': current_section,
            'content': '\n'.join(current_content)
        })
    
    return sections

def format_section_content(content):
    """Format section content with enhanced styling"""
    # Add bullet points to lines starting with dashes
    content = content.replace('\n- ', '\n• ')
    
    # Highlight key terms
    highlight_terms = [
        'bullish', 'bearish', 'overbought', 'oversold', 'resistance', 'support',
        'breakout', 'breakdown', 'trend', 'momentum', 'volume'
    ]
    
    formatted_content = content
    for term in highlight_terms:
        formatted_content = formatted_content.replace(
            f' {term} ',
            f' <span class="highlight">{term}</span> '
        )
    
    # Format numbers and percentages
    import re
    formatted_content = re.sub(
        r'(\$[\d,]+\.?\d*|[\d.]+%)',
        r'<span class="highlight">\1</span>',
        formatted_content
    )
    
    return f'<div class="section-content">{formatted_content}</div>'

def display_no_analysis_warning():
    """Display warning when no analysis is available"""
    st.warning("""
    ⚠️ No detailed analysis available at the moment. This might be due to:
    
    1. API key configuration issue
    2. Connection problems
    3. Service unavailability
    
    Please check your API key and try again.
    """)

    if st.button("Retry Analysis", key="retry_analysis_btn"):
        st.experimental_rerun()


# Error handling wrapper (at the end of the file)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        if st.button("Retry"):
            st.experimental_rerun()

# Error handling wrapper
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            if st.button("Retry"):
                st.experimental_rerun()
    return wrapper


# Main execution
if __name__ == "__main__":
    handle_errors(main)
