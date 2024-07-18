"""
Interactive AlphaAgents Dashboard (Streamlit)
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestration import AlphaGroupChat
from portfolio import PortfolioBuilder
from portfolio.backtest import BacktestEngine
from tools import get_sentiment_for_tickers, get_indicator_data

st.set_page_config(page_title="AlphaAgents Terminal", page_icon="🤖", layout="wide")

st.title("🤖 AlphaAgents: Multi-Agent Investment Terminal")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
risk_profile = st.sidebar.selectbox("Risk Profile", ["neutral", "averse"])
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,GOOGL,AMZN,TSLA,META")
tickers = [t.strip() for t in tickers_input.split(",")]

# Main UI
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stock Analysis", "Technical Charts", "Neural Analytics", "Portfolio View", "Market Sentiment"])

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
    st.header("Advanced Technical Analysis")
    chart_ticker = st.selectbox("Ticker for Charting", tickers, key="chart_ticker")
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"])
    
    if st.button("Generate Professional Chart"):
        with st.spinner("Processing technical indicators..."):
            df = yf.download(chart_ticker, period=period)
            df = get_indicator_data(df)
            
            # Create subplots: Price, MACD, RSI
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               row_heights=[0.6, 0.2, 0.2],
                               subplot_titles=("Price & Overlays", "MACD", "RSI"))
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                        low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # SMAs
            if "SMA_50" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange', width=1)), row=1, col=1)
            if "SMA_200" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="SMA 200", line=dict(color='red', width=1)), row=1, col=1)
            
            # Bollinger Bands
            if "BBU_20_2.0" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name="BB Upper", line=dict(dash='dash', color='gray', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name="BB Lower", line=dict(dash='dash', color='gray', width=1), fill='tonexty'), row=1, col=1)
            
            # MACD
            if "MACD_12_26_9" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name="MACD", line=dict(color='blue')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name="Signal", line=dict(color='orange')), row=2, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name="Histogram"), row=2, col=1)
            
            # RSI
            if "RSI" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), showlegend=False, line=dict(dash='dot', color='red')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), showlegend=False, line=dict(dash='dot', color='green')), row=3, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🧠 Neural Analytics & ML Predictions")
    ml_ticker = st.selectbox("Ticker for ML Prediction", tickers, key="ml_ticker")
    
    st.subheader("Model Selection")
    model_choice = st.radio("Select Model", ["LSTM RNN", "Transformer", "GRU", "Ensemble"], horizontal=True)
    
    if st.button("Run Neural Analysis"):
        from tools import MLEngine, TransformerPredictor, GRUPredictor, FinancialSentimentClassifier, get_stock_news
        
        with st.spinner("Training Neural Networks & Analyzing Patterns..."):
            df = yf.download(ml_ticker, period="2y")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Price Prediction ({model_choice})")
                
                if model_choice == "LSTM RNN":
                    engine = MLEngine()
                    pred_results = engine.predict_price_lstm(df)
                elif model_choice == "Transformer":
                    engine = TransformerPredictor()
                    pred_results = engine.train_and_predict(df)
                elif model_choice == "GRU":
                    engine = GRUPredictor()
                    pred_results = engine.train_and_predict(df)
                else:  # Ensemble
                    lstm_engine = MLEngine()
                    transformer = TransformerPredictor()
                    lstm_pred = lstm_engine.predict_price_lstm(df)
                    trans_pred = transformer.train_and_predict(df)
                    
                    if "error" not in lstm_pred and "error" not in trans_pred:
                        # Average ensemble
                        avg_forecast = [(l + t) / 2 for l, t in zip(lstm_pred['forecast'], trans_pred['forecast'])]
                        pred_results = {"forecast": avg_forecast, "model_type": "Ensemble (LSTM + Transformer)"}
                    else:
                        pred_results = lstm_pred if "error" not in lstm_pred else trans_pred
                
                if "error" not in pred_results:
                    st.success(f"5-Day Forecast: {[round(p, 2) for p in pred_results['forecast']]}")
                    st.info(f"Model: {pred_results.get('model_type', model_choice)}")
                    if "architecture" in pred_results:
                        st.json(pred_results["architecture"])
                else:
                    st.error(pred_results["error"])
            
            with col2:
                st.subheader("FinBERT Sentiment Analysis")
                classifier = FinancialSentimentClassifier()
                news = get_stock_news(ml_ticker, days=3)
                
                if news and "error" not in news[0]:
                    headlines = [n.get("title", "") for n in news[:10]]
                    agg_result = classifier.aggregate_sentiment(headlines)
                    
                    sentiment_color = "green" if agg_result["aggregate_sentiment"] == "bullish" else "red" if agg_result["aggregate_sentiment"] == "bearish" else "gray"
                    st.metric("News Sentiment", agg_result["aggregate_sentiment"].upper(), delta=f"{agg_result['score']:.2f}")
                    st.progress(agg_result["score"])
                    st.caption(f"Model: {agg_result['model']} | Sample: {agg_result['sample_size']} headlines")
                else:
                    st.warning("No news available for sentiment analysis")
            
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Anomaly Detection (Isolation Forest)")
                engine_ml = MLEngine()
                anomaly_results = engine_ml.detect_anomalies(df)
                if "error" not in anomaly_results:
                    st.write(f"Detected **{anomaly_results['anomaly_count']}** outliers")
                    st.warning(f"Recent anomalies: {anomaly_results['recent_anomalies']}")
                else:
                    st.error(anomaly_results["error"])
            
            with col4:
                st.subheader("Feature Importance (Random Forest)")
                importance = engine_ml.analyze_feature_importance(df)
                st.bar_chart(pd.Series(importance))

with tab4:
    st.header("Portfolio Construction")
    if st.button("Generate Recommendations"):
        st.write("Synthesizing multi-agent views into optimized weights...")
        # Mock logic for demo
        recs = [
            {"ticker": t, "rec": "BUY", "weight": f"{100/len(tickers):.1f}%"} for t in tickers
        ]
        st.table(recs)

with tab5:
    st.header("Global Market Sentiment Heatmap")
    if st.button("Refresh Sentiment Map"):
        with st.spinner("Calculating sentiment for universe..."):
            scores = get_sentiment_for_tickers(tickers)
            sentiment_df = pd.DataFrame([
                {"Ticker": k, "Sentiment": (v - 0.5) * 2, "Value": 1} for k, v in scores.items()
            ])
            
            fig = px.treemap(sentiment_df, 
                             path=['Ticker'], 
                             values='Value',
                             color='Sentiment',
                             color_continuous_scale='RdYlGn',
                             range_color=[-1, 1],
                             title="Stock Sentiment Heatmap (Red: Bearish, Green: Bullish)")
            st.plotly_chart(fig, use_container_width=True)

    st.header("Performance Backtesting")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
    
    if st.button("Run Simulation"):
        engine_bt = BacktestEngine()
        # Use simple equal weights for demo
        mock_recs = [{"ticker": t, "weight": 100/len(tickers)} for t in tickers]
        results = engine_bt.run(mock_recs, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        st.metric("Total Return", f"{results['total_return_pct']:.2f}%")
        st.line_chart(results["history"])

st.sidebar.markdown("---")
st.sidebar.info("Built with AutoGen & GPT-4o")
