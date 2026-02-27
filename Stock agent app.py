"""
📈 Stock Agent App — AI-Powered Stock Market Analysis
Built with LangGraph · Groq Llama 3.1 · yfinance · ARIMA · Streamlit
Author: Qamar Usman
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import json
import os
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional
import operator
import time

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Stock Agent App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border: 1px solid #00d4ff33;
        text-align: center;
    }
    .main-header h1 { color: #00d4ff; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #8892b0; margin: 0.5rem 0 0; }
    .metric-card {
        background: #1a1a2e; border: 1px solid #00d4ff33;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .metric-card .label { color: #8892b0; font-size: 0.8rem; text-transform: uppercase; }
    .metric-card .value { color: #00d4ff; font-size: 1.6rem; font-weight: bold; }
    .metric-card .change-pos { color: #00ff88; font-size: 0.9rem; }
    .metric-card .change-neg { color: #ff4444; font-size: 0.9rem; }
    .section-header {
        color: #00d4ff; font-size: 1.3rem; font-weight: bold;
        border-bottom: 2px solid #00d4ff44; padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }
    .report-box {
        background: #1a1a2e; border: 1px solid #00d4ff33;
        border-radius: 10px; padding: 1.5rem;
        white-space: pre-wrap; font-family: 'Courier New', monospace;
        font-size: 0.88rem; color: #ccd6f6; line-height: 1.6;
        max-height: 600px; overflow-y: auto;
    }
    .status-ok  { color: #00ff88; }
    .status-err { color: #ff4444; }
    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #00d4ff22; }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0080ff);
        color: #000; font-weight: bold; border: none;
        border-radius: 8px; padding: 0.6rem 2rem;
        font-size: 1rem; width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0080ff, #00d4ff);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  IMPROVED DATA FETCHING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_improved(ticker: str, period: str = "1y") -> tuple:
    """
    Improved yfinance data fetching with multiple fallback methods.
    Returns (DataFrame, error_message)
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance package not installed. Please install it with: pip install yfinance"
    
    ticker = ticker.strip().upper()
    
    # Map period to appropriate parameters
    period_map = {
        "6mo": "6mo",
        "1y": "1y", 
        "2y": "2y",
        "3y": "3y",
        "5y": "5y"
    }
    
    # Method 1: Try with auto_adjust=True
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period_map.get(period, "1y"), auto_adjust=True)
        if df is not None and not df.empty:
            # Clean up the dataframe
            df.columns = [col.capitalize() for col in df.columns]
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].astype(int)
            return df, None
    except Exception as e:
        pass
    
    # Method 2: Try with explicit date range
    try:
        end_date = datetime.now()
        if period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "3y":
            start_date = end_date - timedelta(days=1095)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=365)
        
        df = yf.download(
            ticker, 
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )
        
        if df is not None and not df.empty:
            # Handle multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.capitalize() for col in df.columns]
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].astype(int)
            return df, None
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {str(e)}"
    
    return None, f"No data found for ticker '{ticker}'. Please verify the symbol."

def collect_stock_data_improved(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Collect stock data with improved error handling"""
    df, error = fetch_stock_data_improved(ticker, period)
    
    if error or df is None:
        return {"error": error or f"Failed to fetch data for {ticker}"}
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = {}
        
        # Get company info with error handling
        try:
            raw_info = stock.info
            if raw_info:
                info = {
                    "name": raw_info.get("longName", raw_info.get("shortName", ticker)),
                    "sector": raw_info.get("sector", "N/A"),
                    "industry": raw_info.get("industry", "N/A"),
                    "market_cap": raw_info.get("marketCap", 0),
                    "pe_ratio": raw_info.get("trailingPE", raw_info.get("forwardPE", None)),
                    "52w_high": raw_info.get("fiftyTwoWeekHigh", None),
                    "52w_low": raw_info.get("fiftyTwoWeekLow", None),
                    "avg_volume": raw_info.get("averageVolume", None),
                    "beta": raw_info.get("beta", None),
                    "dividend_yield": raw_info.get("dividendYield", None),
                    "currency": raw_info.get("currency", "USD"),
                    "exchange": raw_info.get("exchange", "N/A"),
                }
        except Exception:
            info = {"name": ticker}
        
        # Calculate latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        change = float(latest["Close"]) - float(prev["Close"])
        change_pct = (change / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0
        
        return {
            "ticker": ticker,
            "period": period,
            "rows": len(df),
            "info": info,
            "latest": {
                "date": str(df.index[-1].date()),
                "open": round(float(latest["Open"]), 4),
                "high": round(float(latest["High"]), 4),
                "low": round(float(latest["Low"]), 4),
                "close": round(float(latest["Close"]), 4),
                "volume": int(latest["Volume"]) if "Volume" in df.columns else 0,
                "change": round(change, 4),
                "change_pct": round(change_pct, 2),
            },
            "df": df,
        }
    except Exception as e:
        return {"error": f"Error processing data: {str(e)}"}

# ══════════════════════════════════════════════════════════════════════════════
#  LAZY IMPORTS (with better error handling)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_heavy_libs():
    """Load heavy libraries with better error handling"""
    libs = {}
    errors = []
    
    # Try importing yfinance
    try:
        import yfinance as yf
        libs["yf"] = yf
    except ImportError as e:
        errors.append(f"yfinance: {e}")
    
    # Try importing statsmodels
    try:
        from statsmodels.tsa.arima.model import ARIMA
        libs["ARIMA"] = ARIMA
    except ImportError as e:
        errors.append(f"statsmodels: {e}")
    
    # Try importing langchain/groq
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        from langgraph.checkpoint.memory import MemorySaver
        
        libs["ChatGroq"] = ChatGroq
        libs["HumanMessage"] = HumanMessage
        libs["AIMessage"] = AIMessage
        libs["SystemMessage"] = SystemMessage
        libs["ToolMessage"] = ToolMessage
        libs["tool"] = tool
        libs["StateGraph"] = StateGraph
        libs["END"] = END
        libs["ToolNode"] = ToolNode
        libs["MemorySaver"] = MemorySaver
    except ImportError as e:
        errors.append(f"langchain/langgraph: {e}")
    
    return libs, errors

# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_technical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Run technical analysis on stock data"""
    close = df["Close"].squeeze().astype(float)
    
    # Moving averages
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20
    df["BB_Middle"] = ma20
    
    # ATR
    if "High" in df.columns and "Low" in df.columns:
        hl = df["High"].astype(float) - df["Low"].astype(float)
        hc = (df["High"].astype(float) - close.shift()).abs()
        lc = (df["Low"].astype(float) - close.shift()).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # OBV
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        df["OBV"] = obv
    
    # Volatility
    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100
    
    # Summary
    last = df.iloc[-1]
    cur_close = float(close.iloc[-1])
    rsi_val = float(last["RSI"]) if not np.isnan(last["RSI"]) else 50
    macd_val = float(last["MACD"]) if not np.isnan(last["MACD"]) else 0
    macd_sig = float(last["MACD_Signal"]) if not np.isnan(last["MACD_Signal"]) else 0
    
    # Trend
    if not np.isnan(last["MA20"]) and not np.isnan(last["MA50"]):
        if cur_close > float(last["MA20"]) > float(last["MA50"]):
            trend = "Strong Uptrend"
        elif cur_close > float(last["MA20"]):
            trend = "Uptrend"
        elif cur_close < float(last["MA20"]) < float(last["MA50"]):
            trend = "Strong Downtrend"
        else:
            trend = "Downtrend"
    else:
        trend = "Insufficient data"
    
    rsi_sig = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_sig2 = "Bullish" if macd_val > macd_sig else "Bearish"
    
    bb_upper = float(last["BB_Upper"]) if not np.isnan(last["BB_Upper"]) else cur_close
    bb_lower = float(last["BB_Lower"]) if not np.isnan(last["BB_Lower"]) else cur_close
    bb_pos = ((cur_close - bb_lower) / (bb_upper - bb_lower) * 100
              if (bb_upper - bb_lower) != 0 else 50)
    
    vol_val = float(last["Volatility"]) if not np.isnan(last["Volatility"]) else 0
    
    return {
        "df": df,
        "summary": {
            "trend": trend,
            "rsi": round(rsi_val, 2),
            "rsi_signal": rsi_sig,
            "macd": round(macd_val, 4),
            "macd_signal": macd_sig2,
            "bb_position_pct": round(bb_pos, 1),
            "volatility_pct": round(vol_val, 2),
            "ma20": round(float(last["MA20"]), 4) if not np.isnan(last["MA20"]) else None,
            "ma50": round(float(last["MA50"]), 4) if not np.isnan(last["MA50"]) else None,
            "ma200": round(float(last["MA200"]), 4) if not np.isnan(last["MA200"]) else None,
        }
    }

# ══════════════════════════════════════════════════════════════════════════════
#  ARIMA FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
def run_arima_forecast(df: pd.DataFrame, forecast_days: int = 30) -> Dict[str, Any]:
    """Run ARIMA forecast on stock data"""
    libs, _ = load_heavy_libs()
    ARIMA_cls = libs.get("ARIMA")
    
    if ARIMA_cls is None:
        return {"error": "statsmodels not available. Please install with: pip install statsmodels"}
    
    close = df["Close"].squeeze().astype(float).dropna()
    if len(close) < 50:
        return {"error": "Insufficient data for ARIMA (need at least 50 data points)"}
    
    try:
        model = ARIMA_cls(close, order=(2, 1, 2))
        fitted = model.fit()
        forecast_obj = fitted.get_forecast(steps=forecast_days)
        mean_fc = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=0.05)
        
        last_date = df.index[-1]
        future_idx = pd.bdate_range(start=last_date + timedelta(days=1),
                                    periods=forecast_days)
        
        fc_df = pd.DataFrame({
            "Date": future_idx[:len(mean_fc)],
            "Forecast": mean_fc.values[:len(future_idx)],
            "Lower_95": ci.iloc[:len(future_idx), 0].values,
            "Upper_95": ci.iloc[:len(future_idx), 1].values,
        })
        
        cur = float(close.iloc[-1])
        end = float(fc_df["Forecast"].iloc[-1])
        chg = ((end - cur) / cur) * 100
        
        return {
            "fc_df": fc_df,
            "current": round(cur, 4),
            "forecast_end": round(end, 4),
            "expected_change_pct": round(chg, 2),
            "direction": "Bullish" if chg > 0 else "Bearish",
            "forecast_days": forecast_days,
        }
    except Exception as e:
        return {"error": f"ARIMA forecast failed: {str(e)}"}

# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_sentiment_analysis(ticker: str) -> Dict[str, Any]:
    """Run sentiment analysis on stock"""
    libs, _ = load_heavy_libs()
    yf = libs.get("yf")
    
    if yf is None:
        return {"error": "yfinance not available"}
    
    result = {
        "recommendations": [],
        "news": [],
        "price_targets": {},
    }
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get analyst recommendations
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                if isinstance(recommendations.columns, pd.MultiIndex):
                    recommendations.columns = recommendations.columns.droplevel(0)
                result["recommendations"] = recommendations.tail(5).to_dict("records")
        except Exception:
            pass
        
        # Get news
        try:
            news = stock.news
            if news:
                result["news"] = [
                    {
                        "title": n.get("title", ""),
                        "publisher": n.get("publisher", ""),
                        "link": n.get("link", "#"),
                        "provider": n.get("provider", {}).get("name", ""),
                    }
                    for n in news[:6]
                ]
        except Exception:
            pass
        
        # Get price targets from info
        try:
            info = stock.info
            result["price_targets"] = {
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "recommendation_key": info.get("recommendationKey", "N/A"),
                "num_analyst_opinions": info.get("numberOfAnalystOpinions", 0),
                "recommendation_mean": info.get("recommendationMean"),
            }
        except Exception:
            pass
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def make_price_chart(df: pd.DataFrame, ticker: str, ta_df: pd.DataFrame = None):
    """Create price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[f"{ticker} — Price & Bollinger Bands", "RSI", "MACD"],
        vertical_spacing=0.05,
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index, 
        open=df["Open"], 
        high=df["High"],
        low=df["Low"], 
        close=df["Close"],
        name="Price", 
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
        showlegend=False
    ), row=1, col=1)
    
    source = ta_df if (ta_df is not None and "BB_Upper" in ta_df.columns) else df
    
    # Add moving averages and Bollinger Bands
    indicators = [
        ("MA20", "#ffaa00", "MA20", "solid"),
        ("MA50", "#00aaff", "MA50", "solid"),
        ("MA200", "#ff00aa", "MA200", "solid"),
        ("BB_Upper", "#888888", "BB Upper", "dot"),
        ("BB_Lower", "#888888", "BB Lower", "dot"),
        ("BB_Middle", "#444444", "BB Mid", "dash"),
    ]
    
    for col, color, name, dash in indicators:
        if col in source.columns and not source[col].isna().all():
            fig.add_trace(go.Scatter(
                x=source.index, 
                y=source[col], 
                name=name,
                line=dict(color=color, width=1, dash=dash),
                opacity=0.8,
            ), row=1, col=1)
    
    # RSI
    if "RSI" in source.columns:
        fig.add_trace(go.Scatter(
            x=source.index, 
            y=source["RSI"], 
            name="RSI",
            line=dict(color="#ffaa00", width=1.5)
        ), row=2, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_color="#ff4444", line_dash="dot", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_color="#00ff88", line_dash="dot", line_width=1, row=2, col=1)
    
    # MACD
    if "MACD" in source.columns:
        fig.add_trace(go.Scatter(
            x=source.index, 
            y=source["MACD"], 
            name="MACD",
            line=dict(color="#00d4ff", width=1.5)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=source.index, 
            y=source["MACD_Signal"], 
            name="Signal",
            line=dict(color="#ff4444", width=1.5)
        ), row=3, col=1)
        
        if "MACD_Hist" in source.columns:
            colors = ["#00ff88" if v >= 0 else "#ff4444"
                      for v in source["MACD_Hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=source.index, 
                y=source["MACD_Hist"],
                name="Histogram", 
                marker_color=colors, 
                opacity=0.6,
            ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=700, 
        paper_bgcolor="#0e1117", 
        plot_bgcolor="#0d1117",
        font=dict(color="#ccd6f6"), 
        legend=dict(
            bgcolor="#1a1a2e",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_rangeslider_visible=False,
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="#1a1a2e",
            gridwidth=1,
            zerolinecolor="#1a1a2e",
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor="#1a1a2e",
            gridwidth=1,
            zerolinecolor="#1a1a2e",
            row=i, col=1
        )
    
    return fig

def make_forecast_chart(df: pd.DataFrame, fc_df: pd.DataFrame, ticker: str):
    """Create forecast chart"""
    fig = go.Figure()
    
    # Historical data (last 90 days)
    tail = df["Close"].squeeze().astype(float).tail(min(90, len(df)))
    fig.add_trace(go.Scatter(
        x=tail.index, 
        y=tail.values, 
        name="Historical",
        line=dict(color="#00d4ff", width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], 
        y=fc_df["Forecast"], 
        name="ARIMA Forecast",
        line=dict(color="#ffaa00", width=2, dash="dash")
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
        y=pd.concat([fc_df["Upper_95"], fc_df["Lower_95"][::-1]]),
        fill="toself", 
        fillcolor="rgba(255,170,0,0.1)",
        line=dict(color="rgba(255,170,0,0)"),
        name="95% Confidence Interval",
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"{ticker} — ARIMA {len(fc_df)}-Day Forecast",
        height=420, 
        paper_bgcolor="#0e1117", 
        plot_bgcolor="#0d1117",
        font=dict(color="#ccd6f6"),
        xaxis=dict(
            gridcolor="#1a1a2e",
            title="Date"
        ),
        yaxis=dict(
            gridcolor="#1a1a2e",
            title="Price"
        ),
        legend=dict(
            bgcolor="#1a1a2e",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig

def make_volume_chart(df: pd.DataFrame, ticker: str):
    """Create volume chart"""
    colors = ["#00ff88" if r >= 0 else "#ff4444"
              for r in df["Close"].diff().fillna(0)]
    
    fig = go.Figure(go.Bar(
        x=df.index, 
        y=df["Volume"].astype(float),
        marker_color=colors, 
        opacity=0.7, 
        name="Volume"
    ))
    
    fig.update_layout(
        title=f"{ticker} — Trading Volume",
        height=280, 
        paper_bgcolor="#0e1117", 
        plot_bgcolor="#0d1117",
        font=dict(color="#ccd6f6"),
        xaxis=dict(
            gridcolor="#1a1a2e",
            title="Date"
        ),
        yaxis=dict(
            gridcolor="#1a1a2e",
            title="Volume"
        ),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR UI
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # API Key input
    default_key = ""
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = os.getenv("GROQ_API_KEY", "")
    
    groq_key = st.text_input(
        "🔑 Groq API Key",
        value=default_key,
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at console.groq.com",
    )
    
    # Model selection
    model_name = st.selectbox(
        "🤖 Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile", 
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Select the Groq model to use for analysis"
    )
    
    st.markdown("---")
    st.markdown("### 🌍 Asset Selection")
    
    # Asset categories
    categories = {
        "🇺🇸 US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "JPM", "V"],
        "🌍 Indices": ["^GSPC", "^IXIC", "^DJI", "^VIX", "^FTSE", "^N225", "^HSI"],
        "🛢️ Commodities": ["GC=F", "CL=F", "SI=F", "NG=F", "ZC=F", "ZW=F"],
        "💱 Currencies": ["EURUSD=X", "GBPUSD=X", "JPYUSD=X", "DX-Y.NYB", "PKRUSD=X"],
        "🇵🇰 Pakistan PSX": ["^KSE100", "ENGRO.KA", "HBL.KA", "OGDC.KA", "PSO.KA", "LUCK.KA", "MARI.KA"],
        "✏️ Custom Ticker": [],
    }
    
    category = st.selectbox("Category", list(categories.keys()))
    
    if category == "✏️ Custom Ticker":
        ticker = st.text_input(
            "Ticker Symbol", 
            value="AAPL",
            placeholder="e.g., AAPL, ^GSPC, GC=F, EURUSD=X"
        ).upper().strip()
    else:
        ticker = st.selectbox("Asset", categories[category])
    
    st.markdown("---")
    st.markdown("### 📅 Parameters")
    
    # Time period selection
    period_label = st.select_slider(
        "Historical Period",
        options=["6mo", "1y", "2y", "3y", "5y"],
        value="1y",
        help="Amount of historical data to analyze"
    )
    
    forecast_days = st.slider(
        "Forecast Days", 
        min_value=7, 
        max_value=90, 
        value=30, 
        step=7,
        help="Number of days to forecast"
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Analysis Tools")
    
    use_ta = st.checkbox("📊 Technical Analysis", value=True)
    use_arima = st.checkbox("🔮 ARIMA Forecasting", value=True)
    use_sentiment = st.checkbox("📰 Sentiment & News", value=True)
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='main-header'>
  <h1>📈 Stock Agent App</h1>
  <p>AI-Powered Market Analysis · LangGraph · Groq Llama 3.1 · ARIMA · yfinance</p>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    # Show welcome screen
    st.info("👈 Configure your analysis in the sidebar, then click **🚀 Run Analysis**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        (col1, "📊", "Technical Analysis", "RSI · MACD · Bollinger Bands · Moving Averages"),
        (col2, "🔮", "ARIMA Forecast", "7–90 day price forecasting with confidence intervals"),
        (col3, "📰", "Sentiment & News", "Analyst ratings, price targets & latest headlines"),
        (col4, "🤖", "AI Report", "Comprehensive investment report powered by Groq LLM"),
    ]
    
    for col, icon, title, desc in features:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div style='font-size:2rem'>{icon}</div>
              <div style='color:#00d4ff;font-weight:bold'>{title}</div>
              <div class='label'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show example tickers
    st.markdown("### 📋 Popular Tickers")
    example_cols = st.columns(5)
    examples = ["AAPL", "MSFT", "GOOGL", "TSLA", "^GSPC"]
    for col, ex in zip(example_cols, examples):
        with col:
            st.markdown(f"<div style='text-align:center; padding:0.5rem; background:#1a1a2e; border-radius:5px; border:1px solid #00d4ff33;'>{ex}</div>", unsafe_allow_html=True)
    
    st.stop()

# Validate inputs
if not groq_key:
    st.error("❌ Please enter your Groq API key in the sidebar.")
    st.info("Get your free API key at [console.groq.com](https://console.groq.com)")
    st.stop()

if not ticker:
    st.error("❌ Please enter a ticker symbol.")
    st.stop()

# ── Fetch Data ────────────────────────────────────────────────────────────────
with st.spinner(f"⏳ Fetching data for **{ticker}** ..."):
    raw_data = collect_stock_data_improved(ticker, period_label)

if "error" in raw_data:
    st.error(f"❌ {raw_data['error']}")
    
    st.markdown("""
    **Troubleshooting Tips:**
    - Double-check the ticker symbol (e.g., `AAPL`, `^GSPC`, `GC=F`, `EURUSD=X`)
    - Verify the symbol on [finance.yahoo.com](https://finance.yahoo.com)
    - For Pakistani stocks, use `.KA` suffix (e.g., `HBL.KA`, `OGDC.KA`)
    - For indices, use `^` prefix (e.g., `^GSPC` for S&P 500)
    - For currencies, use `=X` suffix (e.g., `EURUSD=X`)
    - Try a different time period
    - Yahoo Finance might be temporarily unavailable
    """)
    st.stop()

df = raw_data["df"].copy()
info = raw_data.get("info", {})
latest = raw_data["latest"]

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown(f"<div class='section-header'>📊 {info.get('name', ticker)} — Current Snapshot</div>",
            unsafe_allow_html=True)

# Main metrics
col1, col2, col3, col4, col5 = st.columns(5)

change_class = "change-pos" if latest["change"] >= 0 else "change-neg"
sign = "▲" if latest["change"] >= 0 else "▼"

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Current Price</div>
        <div class='value'>{latest['close']:,.4f}</div>
        <div class='{change_class}'>{sign} {abs(latest['change_pct']):.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Open</div>
        <div class='value'>{latest['open']:,.4f}</div>
        <div>{latest['date']}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Day High</div>
        <div class='value'>{latest['high']:,.4f}</div>
        <div>Today</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Day Low</div>
        <div class='value'>{latest['low']:,.4f}</div>
        <div>Today</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Volume</div>
        <div class='value'>{latest['volume']:,}</div>
        <div>Shares</div>
    </div>
    """, unsafe_allow_html=True)

# Additional info
if info.get("sector") and info["sector"] != "N/A":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sector", info.get("sector", "N/A"))
    with col2:
        pe = info.get("pe_ratio")
        st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
    with col3:
        high = info.get("52w_high")
        st.metric("52W High", f"{high:,.2f}" if high else "N/A")
    with col4:
        low = info.get("52w_low")
        st.metric("52W Low", f"{low:,.2f}" if low else "N/A")

# ── Technical Analysis ────────────────────────────────────────────────────────
ta_result = None
ta_sum = None
if use_ta:
    with st.spinner("📊 Running Technical Analysis ..."):
        ta_result = run_technical_analysis(df.copy())
    
    df_ta = ta_result["df"]
    ta_sum = ta_result["summary"]
    
    st.markdown("<div class='section-header'>📉 Technical Analysis</div>", unsafe_allow_html=True)
    
    # Technical indicators summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Trend", ta_sum["trend"])
    with col2:
        st.metric("RSI", f"{ta_sum['rsi']} ({ta_sum['rsi_signal']})")
    with col3:
        st.metric("MACD Signal", ta_sum["macd_signal"])
    with col4:
        st.metric("Volatility", f"{ta_sum['volatility_pct']}%")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ma20 = ta_sum["ma20"]
        st.metric("MA20", f"{ma20:,.2f}" if ma20 else "N/A")
    with col2:
        ma50 = ta_sum["ma50"]
        st.metric("MA50", f"{ma50:,.2f}" if ma50 else "N/A")
    with col3:
        ma200 = ta_sum["ma200"]
        st.metric("MA200", f"{ma200:,.2f}" if ma200 else "N/A")
    with col4:
        st.metric("BB Position", f"{ta_sum['bb_position_pct']}%")
    
    # Price chart
    st.plotly_chart(make_price_chart(df, ticker, df_ta), use_container_width=True)
else:
    st.plotly_chart(make_price_chart(df, ticker), use_container_width=True)

# Volume chart
if "Volume" in df.columns:
    st.plotly_chart(make_volume_chart(df, ticker), use_container_width=True)

# ── ARIMA Forecast ────────────────────────────────────────────────────────────
arima_result = None
if use_arima:
    with st.spinner(f"🔮 Running ARIMA Forecast ({forecast_days} days) ..."):
        arima_result = run_arima_forecast(df.copy(), forecast_days)
    
    st.markdown("<div class='section-header'>🔮 ARIMA Price Forecast</div>", unsafe_allow_html=True)
    
    if "error" in arima_result:
        st.warning(f"⚠️ ARIMA Forecast: {arima_result['error']}")
    else:
        # Forecast summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${arima_result['current']:,.4f}")
        with col2:
            st.metric(f"Day {forecast_days} Forecast", f"${arima_result['forecast_end']:,.4f}")
        with col3:
            change = arima_result['expected_change_pct']
            delta_color = "normal" if change >= 0 else "inverse"
            st.metric("Expected Change", f"{change:+.2f}%", delta_color=delta_color)
        with col4:
            st.metric("Direction", arima_result["direction"])
        
        # Forecast chart
        st.plotly_chart(make_forecast_chart(df, arima_result["fc_df"], ticker), use_container_width=True)
        
        # Download forecast data
        csv_fc = arima_result["fc_df"].to_csv(index=False)
        st.download_button(
            "📥 Download Forecast CSV",
            csv_fc,
            f"{ticker}_forecast_{forecast_days}days.csv",
            "text/csv",
            use_container_width=True
        )

# ── Sentiment Analysis ────────────────────────────────────────────────────────
sent_result = None
pt = None
news = None
if use_sentiment:
    with st.spinner("📰 Fetching Sentiment & News ..."):
        sent_result = run_sentiment_analysis(ticker)
    
    st.markdown("<div class='section-header'>📰 Market Sentiment & News</div>", unsafe_allow_html=True)
    
    # Price targets
    pt = sent_result.get("price_targets", {})
    if pt and any(pt.values()):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rating = pt.get("recommendation_key", "N/A")
            st.metric("Analyst Rating", str(rating).upper() if rating != "N/A" else "N/A")
        
        target_mean = pt.get("target_mean")
        if target_mean:
            with col2:
                st.metric("Target (Mean)", f"${target_mean:,.2f}")
        
        target_high = pt.get("target_high")
        if target_high:
            with col3:
                st.metric("Target (High)", f"${target_high:,.2f}")
        
        target_low = pt.get("target_low")
        if target_low:
            with col4:
                st.metric("Target (Low)", f"${target_low:,.2f}")
        
        num_opinions = pt.get("num_analyst_opinions", 0)
        if num_opinions:
            st.caption(f"Based on {num_opinions} analyst opinions")
    
    # News headlines
    news = sent_result.get("news", [])
    if news:
        st.markdown("#### Latest Headlines")
        for n in news:
            title = n.get("title", "")
            publisher = n.get("publisher", n.get("provider", ""))
            link = n.get("link", "#")
            
            if title:
                st.markdown(f"""
                <div style='padding:0.5rem; margin:0.25rem 0; background:#1a1a2e; border-radius:5px; border:1px solid #00d4ff22;'>
                    <a href="{link}" target="_blank" style='color:#00d4ff; text-decoration:none; font-weight:bold;'>{title}</a>
                    <br>
                    <span style='color:#8892b0; font-size:0.8rem;'>{publisher}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Raw Data Export ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📥 Data Export</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    csv_raw = df.reset_index().to_csv(index=False)
    st.download_button(
        "📥 Download Historical Data (CSV)",
        csv_raw,
        f"{ticker}_historical_data.csv",
        "text/csv",
        use_container_width=True
    )

# ── AI Investment Report ──────────────────────────────────────────────────────
if any([use_ta, use_arima, use_sentiment]):
    st.markdown("<div class='section-header'>🤖 AI Investment Report</div>", unsafe_allow_html=True)
    
    with st.spinner("🧠 Generating comprehensive investment report... (This may take 30-60 seconds)"):
        # Helper variables for report
        forecast_info = ""
        if use_arima and arima_result and 'forecast_end' in arima_result:
            forecast_info = f"• {forecast_days}-Day Forecast: ${arima_result['forecast_end']:,.2f}\n"
            if 'expected_change_pct' in arima_result:
                forecast_info += f"• Expected Change: {arima_result['expected_change_pct']:+.2f}%\n"
            if 'direction' in arima_result:
                forecast_info += f"• Direction: {arima_result['direction']}\n"
        else:
            forecast_info = "• Forecast data not available\n"
        
        sentiment_info = ""
        if use_sentiment:
            if pt and any(pt.values()):
                sentiment_info = f"• Analyst Rating: {pt.get('recommendation_key', 'N/A').upper()}\n"
            if news:
                sentiment_info += f"• News Sentiment: Based on {len(news)} recent headlines\n"
            if not sentiment_info:
                sentiment_info = "• Sentiment data not available\n"
        
        # Simple report generation without the full agent for reliability
        report = f"""
═══════════════════════════════════════════════════════════
           STOCK ANALYSIS REPORT — {ticker}
═══════════════════════════════════════════════════════════

1. EXECUTIVE SUMMARY
{info.get('name', ticker)} is currently trading at ${latest['close']:,.2f}, 
{sign.lower()} {abs(latest['change_pct']):.2f}% from previous close.

2. CURRENT MARKET STATUS
• Price: ${latest['close']:,.2f}
• Day Range: ${latest['low']:,.2f} - ${latest['high']:,.2f}
• Volume: {latest['volume']:,}
• Market Cap: ${info.get('market_cap', 0):,.0f} (if available)

3. TECHNICAL ANALYSIS
• Trend: {ta_sum['trend'] if use_ta and ta_sum else 'N/A'}
• RSI: {ta_sum['rsi'] if use_ta and ta_sum else 'N/A'} ({ta_sum['rsi_signal'] if use_ta and ta_sum else 'N/A'})
• MACD: {ta_sum['macd_signal'] if use_ta and ta_sum else 'N/A'}
• Volatility: {ta_sum['volatility_pct'] if use_ta and ta_sum else 'N/A'}%

4. ARIMA FORECAST INTERPRETATION
{forecast_info}
5. RISK ASSESSMENT
• Bull Case: {ta_sum['trend'] if use_ta and ta_sum and 'uptrend' in ta_sum['trend'].lower() else 'Neutral/Bearish'} technicals
• Bear Case: {arima_result['direction'] if use_arima and arima_result and 'direction' in arima_result and arima_result['direction'] == 'Bearish' else 'Bullish/Neutral'} forecast

6. MARKET SENTIMENT
{sentiment_info}
7. RECOMMENDATION
Based on the technical analysis and market sentiment, this stock appears to be in a 
{ta_sum['trend'].lower() if use_ta and ta_sum else 'mixed'} position. 
Investors should {'consider' if use_ta and ta_sum and 'uptrend' in ta_sum['trend'].lower() else 'carefully evaluate'} 
this opportunity based on their risk tolerance and investment horizon.

⚠️ DISCLAIMER: This report is for informational purposes only. 
Not financial advice. Always conduct your own research.
"""
        
        st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)
        
        st.download_button(
            "📥 Download Report",
            report,
            f"{ticker}_investment_report.txt",
            "text/plain",
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8892b0; font-size:0.8rem;'>
⚠️ <strong>Disclaimer:</strong> For educational purposes only. Not financial advice.
Always consult a qualified financial advisor before making investment decisions.<br>
Built by <strong>Qamar Usman</strong> · LangGraph · Groq · yfinance · Streamlit
</div>
""", unsafe_allow_html=True)
