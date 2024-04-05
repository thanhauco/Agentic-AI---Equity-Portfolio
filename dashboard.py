"""
Interactive AlphaAgents Dashboard (Streamlit)
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestration import AlphaGroupChat
from portfolio import PortfolioBuilder
from portfolio.backtest import BacktestEngine

st.set_page_config(page_title="AlphaAgents Terminal", page_icon="🤖", layout="wide")

st.title("🤖 AlphaAgents: Multi-Agent Investment Terminal")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
risk_profile = st.sidebar.selectbox("Risk Profile", ["neutral", "averse"])
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,GOOGL")
tickers = [t.strip() for t in tickers_input.split(",")]

# Main UI
tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Portfolio View", "Backtesting"])

with tab1:
    st.header("Collaborative Stock Analysis")
    selected_ticker = st.selectbox("Select Ticker to Analyze", tickers)
    
    if st.button(f"Analyze {selected_ticker}"):
        with st.spinner(f"Agents are debating {selected_ticker}..."):
            chat = AlphaGroupChat(risk_profile=risk_profile)
            results = chat.analyze_stock(selected_ticker)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Fundamental Analyst")
                st.info(results.get("fundamental_analysis", "No data"))
            with col2:
                st.subheader("Sentiment Analyst")
                st.warning(results.get("sentiment_analysis", "No data"))
            with col3:
                st.subheader("Valuation Analyst")
                st.success(results.get("valuation_analysis", "No data"))

with tab2:
    st.header("Portfolio Construction")
    if st.button("Generate Recommendations"):
        st.write("Synthesizing multi-agent views into optimized weights...")
        # Mock logic for demo
        recs = [
            {"ticker": t, "rec": "BUY", "weight": f"{100/len(tickers):.1f}%"} for t in tickers
        ]
        st.table(recs)

with tab3:
    st.header("Performance Backtesting")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
    
    if st.button("Run Simulation"):
        engine = BacktestEngine()
        # Use simple equal weights for demo
        mock_recs = [{"ticker": t, "weight": 100/len(tickers)} for t in tickers]
        results = engine.run(mock_recs, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        st.metric("Total Return", f"{results['total_return_pct']:.2f}%")
        st.line_chart(results["history"])

st.sidebar.markdown("---")
st.sidebar.info("Built with AutoGen & GPT-4o")
