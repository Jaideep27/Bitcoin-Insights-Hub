# backend/futures_calculator.py

from typing import Dict, Any
import streamlit as st
import pandas as pd

def calculate_futures_results(
    usdt_amount: float,
    leverage: float,
    entry_price: float,
    exit_price: float,
    position_type: str,
    stop_loss: float = None
) -> Dict[str, Any]:
    """Calculate futures trading results"""
    try:
        # Calculate position size
        position_size = (usdt_amount * leverage) / entry_price
        
        # Calculate PNL
        if position_type == "Long":
            pnl = (exit_price - entry_price) * position_size
            liquidation_price = entry_price * (1 - (1 / leverage))
            if stop_loss:
                max_loss = (stop_loss - entry_price) * position_size
        else:  # Short
            pnl = (entry_price - exit_price) * position_size
            liquidation_price = entry_price * (1 + (1 / leverage))
            if stop_loss:
                max_loss = (entry_price - stop_loss) * position_size
        
        # Calculate ROI
        roi = (pnl / usdt_amount) * 100
        
        return {
            'position_size': position_size,
            'pnl': pnl,
            'roi': roi,
            'liquidation_price': liquidation_price,
            'max_loss': max_loss if stop_loss else None
        }
    except Exception as e:
        st.error(f"Error in calculations: {e}")
        return None

def display_futures_calculator():
    """Display futures calculator interface"""
    st.title("Futures Calculator")

    # Add styling
    st.markdown("""
        <style>
        .risk-warning {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #EF4444;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 4px 4px 0;
        }
        .metric-container {
            background-color: rgba(17, 25, 40, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize default values in session state if not present
    if 'futures_values' not in st.session_state:
        st.session_state.futures_values = {
            'usdt_amount': 100.0,
            'leverage': 10,
            'position_type': 'Long',
            'entry_price': 42000.0,
            'exit_price': 45000.0,
            'stop_loss': 40000.0
        }

    # Create form for inputs
    with st.form("futures_calculator_form"):
        col1, col2 = st.columns(2)

        with col1:
            usdt_amount = st.number_input(
                "USDT Amount",
                min_value=1.0,
                value=st.session_state.futures_values['usdt_amount'],
                step=10.0,
                help="Amount of USDT to invest"
            )

            leverage = st.select_slider(
                "Leverage",
                options=[1, 2, 3, 5, 10, 20, 50, 75, 100, 125],
                value=st.session_state.futures_values['leverage'],
                help="Select leverage multiplier"
            )

            position_type = st.radio(
                "Position Type",
                options=["Long", "Short"],
                index=0 if st.session_state.futures_values['position_type'] == 'Long' else 1,
                horizontal=True
            )

        with col2:
            entry_price = st.number_input(
                "Entry Price",
                min_value=0.1,
                value=st.session_state.futures_values['entry_price'],
                step=100.0,
                help="Entry price in USDT"
            )

            exit_price = st.number_input(
                "Exit Price",
                min_value=0.1,
                value=st.session_state.futures_values['exit_price'],
                step=100.0,
                help="Target exit price in USDT"
            )

            stop_loss = st.number_input(
                "Stop Loss (Optional)",
                min_value=0.0,
                value=st.session_state.futures_values['stop_loss'],
                step=100.0,
                help="Stop loss price in USDT (set to 0 to disable)"
            )

        # Calculate button
        submitted = st.form_submit_button("Calculate", use_container_width=True)

    # Update session state values
    if submitted:
        st.session_state.futures_values.update({
            'usdt_amount': usdt_amount,
            'leverage': leverage,
            'position_type': position_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss
        })

        # Validate inputs
        if entry_price <= 0 or exit_price <= 0:
            st.error("Entry and exit prices must be greater than 0")
            return

        if stop_loss > 0:
            if position_type == "Long" and stop_loss >= entry_price:
                st.error("Stop loss must be below entry price for long positions")
                return
            if position_type == "Short" and stop_loss <= entry_price:
                st.error("Stop loss must be above entry price for short positions")
                return

        # Calculate results
        try:
            results = calculate_futures_results(
                usdt_amount=usdt_amount,
                leverage=leverage,
                entry_price=entry_price,
                exit_price=exit_price,
                position_type=position_type,
                stop_loss=stop_loss if stop_loss > 0 else None
            )

            if results:
                # Display Results
                st.markdown("### Results")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Position Size",
                        f"{results['position_size']:.4f} BTC"
                    )

                with col2:
                    pnl_color = "normal" if results['pnl'] > 0 else "inverse"
                    st.metric(
                        "PnL",
                        f"${results['pnl']:.2f}",
                        f"{results['roi']:.2f}%",
                        delta_color=pnl_color
                    )

                with col3:
                    liq_distance = ((results['liquidation_price'] - entry_price) / entry_price * 100)
                    st.metric(
                        "Liquidation Price",
                        f"${results['liquidation_price']:.2f}",
                        f"{liq_distance:.2f}%",
                        delta_color="inverse"
                    )

                with col4:
                    if results.get('max_loss'):
                        st.metric(
                            "Max Loss",
                            f"${results['max_loss']:.2f}",
                            f"{(results['max_loss'] / usdt_amount * 100):.2f}%",
                            delta_color="inverse"
                        )

                # Risk Analysis
                st.markdown("### Risk Analysis")
                risk_to_liquidation = abs((entry_price - results['liquidation_price']) / entry_price * 100)

                if risk_to_liquidation < 10:
                    st.markdown(f"""
                        <div class="risk-warning">
                            <strong>⚠️ High Risk:</strong> Only {risk_to_liquidation:.1f}% to liquidation. Consider reducing leverage.
                        </div>
                    """, unsafe_allow_html=True)

                if results.get('max_loss') and abs(results['max_loss']) > usdt_amount:
                    st.markdown("""
                        <div class="risk-warning">
                            <strong>⚠️ Warning:</strong> Potential loss exceeds investment amount!
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error in calculations: {str(e)}")
            st.info("Please check your input values and try again.")