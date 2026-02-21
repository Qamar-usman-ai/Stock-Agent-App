# ============================================================
# AI STOCK MARKET ANALYSIS AGENT — STREAMLIT APP
# LangGraph + Groq Llama 3.1 + yfinance Tools
# ============================================================

import os
import json
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq          # ← ChatGroq instead of ChatXAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================
st.set_page_config(
    page_title="AI Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --card:      #16161f;
    --border:    #2a2a3a;
    --accent:    #00ff88;
    --accent2:   #7b61ff;
    --accent3:   #ff6b35;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --red:       #ff4466;
    --green:     #00ff88;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.hero-header {
    background: linear-gradient(135deg, #0a0a0f 0%, #111118 50%, #0d0d1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff88, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.4rem;
}
.metric-value            { font-size: 1.6rem; font-weight: 700; color: var(--text); }
.metric-value.green      { color: var(--green); }
.metric-value.red        { color: var(--red); }
.metric-value.purple     { color: var(--accent2); }

.section-title {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.report-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    color: var(--text);
    white-space: pre-wrap;
    word-wrap: break-word;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSlider > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stTextInput > div > div > input[type="password"] {
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border) !important;
    border-radius: 10px;
}
.stProgress > div > div { background: var(--accent) !important; }
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — USER CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-size:1.3rem;font-weight:800;
        background:linear-gradient(135deg,#00ff88,#7b61ff);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;'>⚡ AI Stock Agent</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
        color:#6b6b80;letter-spacing:2px;text-transform:uppercase;margin-top:2px;'>
        Powered by Groq Llama 3.1 + LangGraph</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API KEY ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🔑 API Configuration</div>',
                unsafe_allow_html=True)
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxx",
        help="Get your FREE key from console.groq.com"
    )
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6b6b80;
    margin-top:-0.5rem;margin-bottom:0.8rem;'>
    🆓 Free key →
    <a href="https://console.groq.com" target="_blank"
       style='color:#7b61ff;'>console.groq.com</a>
    </div>
    """, unsafe_allow_html=True)

    # ── MODEL SELECTION ──────────────────────────────────────
    st.markdown('<div class="section-title">🤖 Groq Model</div>', unsafe_allow_html=True)
    groq_model = st.selectbox(
        "Select Model",
        [
            "llama-3.1-8b-instant",       # fastest, free
            "llama-3.3-70b-versatile",    # smartest
            "llama-3.1-70b-versatile",    # very smart
            "mixtral-8x7b-32768",         # long context
            "gemma2-9b-it",               # balanced
        ],
        index=0,
        help="llama-3.1-8b-instant is fastest and completely free"
    )

    model_badges = {
        "llama-3.1-8b-instant":    ("⚡ Fastest — Recommended",   "#00ff88"),
        "llama-3.3-70b-versatile": ("🧠 Most Intelligent",         "#7b61ff"),
        "llama-3.1-70b-versatile": ("🧠 Very Smart",               "#7b61ff"),
        "mixtral-8x7b-32768":      ("📚 32K Long Context",          "#7b61ff"),
        "gemma2-9b-it":            ("🔬 Balanced Speed/Quality",    "#00ff88"),
    }
    badge_text, badge_color = model_badges.get(groq_model, ("✅ Ready", "#00ff88"))
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;"
        f"color:{badge_color};margin-top:-0.4rem;margin-bottom:0.5rem;'>"
        f"{badge_text}</div>",
        unsafe_allow_html=True
    )

    # ── ASSET SELECTION ──────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Select Asset</div>', unsafe_allow_html=True)

    asset_category = st.selectbox(
        "Category",
        ["📈 US Stocks", "🌍 Market Indices", "🛢️ Commodities",
         "💱 Currencies & Bonds", "🇵🇰 Pakistan (PSX)", "✏️ Custom Ticker"]
    )

    ASSET_MAP = {
        "📈 US Stocks": {
            "Apple (AAPL)":     "AAPL",
            "Microsoft (MSFT)": "MSFT",
            "Google (GOOGL)":   "GOOGL",
            "Tesla (TSLA)":     "TSLA",
            "NVIDIA (NVDA)":    "NVDA",
            "Amazon (AMZN)":    "AMZN",
            "Meta (META)":      "META",
            "JPMorgan (JPM)":   "JPM",
        },
        "🌍 Market Indices": {
            "S&P 500":          "^GSPC",
            "NASDAQ":           "^IXIC",
            "Dow Jones":        "^DJI",
            "VIX Fear Index":   "^VIX",
            "NIFTY 50 (India)": "^NSEI",
            "FTSE 100 (UK)":    "^FTSE",
        },
        "🛢️ Commodities": {
            "Crude Oil":   "CL=F",
            "Gold":        "GC=F",
            "Silver":      "SI=F",
            "Natural Gas": "NG=F",
            "Copper":      "HG=F",
        },
        "💱 Currencies & Bonds": {
            "US 10Y Treasury": "^TNX",
            "USD Index":       "DX-Y.NYB",
            "EUR/USD":         "EURUSD=X",
            "GBP/USD":         "GBPUSD=X",
        },
        "🇵🇰 Pakistan (PSX)": {
            "KSE-100 Index":      "^KSE100",
            "Engro (ENGRO)":      "ENGRO.KA",
            "HBL Bank":           "HBL.KA",
            "OGDC":               "OGDC.KA",
            "Pakistan State Oil": "PSO.KA",
            "Lucky Cement":       "LUCK.KA",
            "MCB Bank":           "MCB.KA",
        },
        "✏️ Custom Ticker": {}
    }

    if asset_category == "✏️ Custom Ticker":
        ticker_input    = st.text_input("Enter Ticker Symbol",
                                        placeholder="e.g. AAPL, ^GSPC, GC=F, ENGRO.KA")
        selected_ticker = ticker_input.upper().strip() if ticker_input else ""
        display_name    = selected_ticker
    else:
        asset_options   = ASSET_MAP[asset_category]
        selected_asset  = st.selectbox("Select Asset", list(asset_options.keys()))
        selected_ticker = asset_options[selected_asset]
        display_name    = selected_asset

    # ── ANALYSIS SETTINGS ────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Analysis Settings</div>', unsafe_allow_html=True)

    data_period = st.selectbox(
        "Historical Data Period",
        ["6mo", "1y", "2y", "3y", "5y"],
        index=2,
        help="How much historical data to use for training"
    )

    forecast_days = st.slider(
        "Forecast Days",
        min_value=7, max_value=90, value=30, step=7,
        help="How many days ahead to predict"
    )

    # ── TOOLS ────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔧 Tools to Run</div>', unsafe_allow_html=True)
    run_technical  = st.checkbox("Technical Analysis",      value=True)
    run_arima      = st.checkbox("ARIMA Prediction",        value=True)
    run_sentiment  = st.checkbox("Sentiment & News",        value=True)
    run_llm_report = st.checkbox("AI Report (Groq Llama)",  value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("""
    <div style='margin-top:2rem;padding:1rem;background:#16161f;border-radius:10px;
    border:1px solid #2a2a3a;font-family:JetBrains Mono,monospace;
    font-size:0.65rem;color:#6b6b80;'>
    ⚠️ For educational purposes only.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">AI Stock Intelligence Platform</div>
    <div class="hero-sub">
    LangGraph Agent · Groq Llama 3.1 · yfinance · ARIMA Forecasting
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fetch_data(ticker, period):
    stock = yf.download(ticker, period=period, progress=False)
    if stock.empty:
        return None
    df = pd.DataFrame({
        'Open':   stock['Open'].squeeze(),
        'High':   stock['High'].squeeze(),
        'Low':    stock['Low'].squeeze(),
        'Close':  stock['Close'].squeeze(),
        'Volume': stock['Volume'].squeeze(),
    })
    try:
        vix   = yf.download("^VIX",     period=period, progress=False)['Close']
        gold  = yf.download("GC=F",     period=period, progress=False)['Close']
        oil   = yf.download("CL=F",     period=period, progress=False)['Close']
        bonds = yf.download("^TNX",     period=period, progress=False)['Close']
        df['VIX_lag1']   = vix.reindex(df.index).shift(1)
        df['Gold_lag1']  = gold.reindex(df.index).shift(1)
        df['Oil_lag1']   = oil.reindex(df.index).shift(1)
        df['Bonds_lag1'] = bonds.reindex(df.index).shift(1)
    except:
        pass
    return df.dropna(subset=['Close'])


def compute_technicals(df):
    close = df['Close']
    df    = df.copy()
    df['MA_20']  = close.rolling(20).mean()
    df['MA_50']  = close.rolling(50).mean()
    df['MA_200'] = close.rolling(200).mean()

    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD']        = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist']   = df['MACD'] - df['Signal_Line']

    df['BB_Middle'] = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    df['BB_Upper']  = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower']  = df['BB_Middle'] - 2 * bb_std

    df['Daily_Return']   = close.pct_change()
    df['Volatility_20d'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)
    return df.dropna(subset=['MA_20'])


def run_arima_model(df, n_days):
    close = df['Close'].dropna()
    d     = 0 if adfuller(close)[1] < 0.05 else 1

    split      = int(len(close) * 0.80)
    train, test = close[:split], close[split:]

    fit_train = ARIMA(train, order=(5, d, 2)).fit()
    rmse = np.sqrt(mean_squared_error(test, fit_train.forecast(steps=len(test))))
    mae  = mean_absolute_error(test, fit_train.forecast(steps=len(test)))

    fit_full     = ARIMA(close, order=(5, d, 2)).fit()
    fc_res       = fit_full.get_forecast(steps=n_days)
    fc_mean      = fc_res.predicted_mean
    ci           = fc_res.conf_int()
    fc_dates     = pd.bdate_range(
        start=close.index[-1] + timedelta(days=1), periods=n_days
    )
    forecast_df = pd.DataFrame({
        'Date':      fc_dates,
        'Predicted': fc_mean.values,
        'Lower_95':  ci.iloc[:, 0].values,
        'Upper_95':  ci.iloc[:, 1].values,
    })
    return forecast_df, rmse, mae, d


def get_sentiment(ticker):
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        news = t.news[:8] if t.news else []
        return info, news
    except:
        return {}, []


# ============================================================
# LANGCHAIN TOOLS
# ============================================================
_data_store = {}

@tool
def collect_stock_data_tool(ticker: str, period: str = "2y") -> str:
    """Collect historical stock market data using yfinance."""
    try:
        df = fetch_data(ticker, period)
        if df is None:
            return json.dumps({"error": "No data found"})
        _data_store[ticker] = df
        current = float(df['Close'].iloc[-1])
        return json.dumps({
            "ticker": ticker, "rows": len(df),
            "start":  str(df.index[0].date()),
            "end":    str(df.index[-1].date()),
            "current_price": round(current, 2),
            "high_52w": round(float(df['Close'].max()), 2),
            "low_52w":  round(float(df['Close'].min()), 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def analyze_stock_data_tool(ticker: str) -> str:
    """Perform complete technical analysis on collected stock data."""
    try:
        df = _data_store.get(ticker)
        if df is None:
            return json.dumps({"error": "Run collect_stock_data_tool first."})
        df_a  = compute_technicals(df)
        _data_store[f"{ticker}_analyzed"] = df_a
        latest = df_a.iloc[-1]
        rsi    = float(latest['RSI'])
        trend  = "BULLISH" if float(latest['MA_20']) > float(latest['MA_50']) else "BEARISH"
        macd_s = "BULLISH" if float(latest['MACD']) > float(latest['Signal_Line']) else "BEARISH"
        rsi_sig = ("OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL")
        return json.dumps({
            "ticker": ticker, "price": round(float(latest['Close']), 2),
            "trend": trend, "RSI": round(rsi, 2), "RSI_signal": rsi_sig,
            "MACD_signal": macd_s,
            "MA_20":    round(float(latest['MA_20']), 2),
            "MA_50":    round(float(latest['MA_50']), 2),
            "BB_Upper": round(float(latest['BB_Upper']), 2),
            "BB_Lower": round(float(latest['BB_Lower']), 2),
            "volatility_pct": round(float(latest['Volatility_20d']) * 100, 2),
            "support":    round(float(df_a['Low'].tail(50).min()), 2),
            "resistance": round(float(df_a['High'].tail(50).max()), 2),
            "30d_return": round(float(df['Close'].pct_change(30).iloc[-1]) * 100, 2),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def predict_arima_tool(ticker: str, forecast_days: int = 30) -> str:
    """Run ARIMA model to forecast future prices."""
    try:
        df = _data_store.get(f"{ticker}_analyzed", _data_store.get(ticker))
        if df is None:
            return json.dumps({"error": "No data found."})
        fdf, rmse, mae, d = run_arima_model(df, forecast_days)
        _data_store[f"{ticker}_forecast"] = fdf
        current  = float(df['Close'].iloc[-1])
        pred_end = float(fdf['Predicted'].iloc[-1])
        pct_chg  = round((pred_end - current) / current * 100, 2)
        return json.dumps({
            "ticker": ticker, "model": f"ARIMA(5,{d},2)",
            "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "current_price": round(current, 2),
            "predicted_end": round(pred_end, 2),
            "pct_change":    pct_chg,
            "direction":     "UP" if pct_chg > 0 else "DOWN",
            "forecast_days": forecast_days,
            "next_7_days":   fdf.head(7)[['Date', 'Predicted']].to_dict('records'),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_sentiment_tool(ticker: str) -> str:
    """Get analyst recommendations and recent news headlines."""
    try:
        info, news = get_sentiment(ticker)
        return json.dumps({
            "company":        info.get('longName', ticker),
            "sector":         info.get('sector', 'N/A'),
            "recommendation": info.get('recommendationKey', 'N/A'),
            "target_mean":    info.get('targetMeanPrice', 'N/A'),
            "target_high":    info.get('targetHighPrice', 'N/A'),
            "num_analysts":   info.get('numberOfAnalystOpinions', 'N/A'),
            "headlines": [
                {"title": a.get('title',''), "publisher": a.get('publisher','')}
                for a in news
            ],
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================
# LANGGRAPH AGENT — ChatGroq with Llama 3.1
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]


def build_agent(api_key_val: str, model_name: str):
    """Build LangGraph agent using ChatGroq (Llama 3.1)."""
    llm = ChatGroq(
        model=model_name,           # e.g. "llama-3.1-8b-instant"
        temperature=0,
        groq_api_key=api_key_val,
        max_tokens=4096,
    )
    tools_list = [
        collect_stock_data_tool,
        analyze_stock_data_tool,
        predict_arima_tool,
        get_sentiment_tool,
    ]
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    tool_node_obj = ToolNode(tools_list)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_obj)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ============================================================
# CHART FUNCTIONS
# ============================================================
def plot_price_chart(df, ticker, forecast_df=None):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("", "RSI", "MACD")
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4466',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,68,102,0.3)',
    ), row=1, col=1)

    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name="MA 20",
            line=dict(color='#7b61ff', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name="MA 50",
            line=dict(color='#ff6b35', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper",
            line=dict(color='rgba(123,97,255,0.4)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower",
            line=dict(color='rgba(123,97,255,0.4)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(123,97,255,0.04)'), row=1, col=1)

    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Predicted'],
            name="ARIMA Forecast",
            line=dict(color='#00ff88', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
            y=pd.concat([forecast_df['Upper_95'], forecast_df['Lower_95'][::-1]]),
            fill='toself', fillcolor='rgba(0,255,136,0.07)',
            line=dict(color='rgba(0,255,136,0)'), name="95% CI"), row=1, col=1)

    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
            line=dict(color='#ff6b35', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,68,102,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,255,136,0.5)", row=2, col=1)

    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD",
            line=dict(color='#7b61ff', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name="Signal",
            line=dict(color='#ff6b35', width=1.5)), row=3, col=1)
        colors = ['#00ff88' if v >= 0 else '#ff4466' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram",
            marker_color=colors, opacity=0.6), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#111118', plot_bgcolor='#0a0a0f',
        font=dict(family='JetBrains Mono', color='#e8e8f0', size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor='rgba(0,0,0,0)'),
        height=650, margin=dict(l=10, r=10, t=20, b=10),
        title=dict(text=f"  {ticker} — Price & Indicators",
                   font=dict(size=14, color='#6b6b80'))
    )
    for i in [1, 2, 3]:
        fig.update_xaxes(gridcolor='#1e1e2e', showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor='#1e1e2e', showgrid=True, row=i, col=1)
    return fig


def plot_volume_chart(df):
    colors = ['#00ff88' if df['Close'].iloc[i] >= df['Open'].iloc[i]
              else '#ff4466' for i in range(len(df))]
    fig = go.Figure(go.Bar(x=df.index, y=df['Volume'],
                           marker_color=colors, opacity=0.7, name="Volume"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#111118', plot_bgcolor='#0a0a0f',
        font=dict(family='JetBrains Mono', color='#e8e8f0', size=11),
        height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    fig.update_xaxes(gridcolor='#1e1e2e')
    fig.update_yaxes(gridcolor='#1e1e2e')
    return fig


# ============================================================
# MAIN APP LOGIC
# ============================================================
if not run_button:
    # Welcome screen
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 1</div>
            <div style="font-size:1.8rem;margin:0.3rem 0">🔑</div>
            <div style="font-weight:700;margin-bottom:0.3rem">Enter Groq API Key</div>
            <div style="color:#6b6b80;font-size:0.85rem">
            Free key from
            <a href="https://console.groq.com" target="_blank"
               style="color:#7b61ff;">console.groq.com</a>
            </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 2</div>
            <div style="font-size:1.8rem;margin:0.3rem 0">🎯</div>
            <div style="font-weight:700;margin-bottom:0.3rem">Pick Your Asset</div>
            <div style="color:#6b6b80;font-size:0.85rem">
            Stocks, Indices, Oil, Gold, PSX Pakistan,
            or any custom ticker
            </div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Step 3</div>
            <div style="font-size:1.8rem;margin:0.3rem 0">🚀</div>
            <div style="font-weight:700;margin-bottom:0.3rem">Run Analysis</div>
            <div style="color:#6b6b80;font-size:0.85rem">
            AI Agent runs all tools and writes
            a complete professional report
            </div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.5rem;padding:1.5rem;background:#16161f;border:1px solid #2a2a3a;
    border-radius:12px;font-family:JetBrains Mono,monospace;font-size:0.8rem;
    color:#6b6b80;line-height:1.8;'>
    <span style='color:#00ff88;'>$</span> Available tools in this agent:<br>
    <span style='color:#7b61ff;'>→</span> Tool 1 : Data Collection    (yfinance — any ticker globally)<br>
    <span style='color:#7b61ff;'>→</span> Tool 2 : Technical Analysis  (RSI · MACD · Bollinger Bands · MA)<br>
    <span style='color:#7b61ff;'>→</span> Tool 3 : ARIMA Forecasting   (7–90 days ahead + 95% confidence)<br>
    <span style='color:#7b61ff;'>→</span> Tool 4 : Sentiment & News    (analyst ratings · headlines)<br>
    <span style='color:#00ff88;'>→</span> Brain  : Groq Llama 3.1      (free · ultra-fast · writes full report)<br>
    <br>
    <span style='color:#00ff88;'>$</span> Default model :
    <span style='color:#e8e8f0;'>llama-3.1-8b-instant</span>
    &nbsp;|&nbsp; Provider :
    <span style='color:#e8e8f0;'>Groq Cloud (FREE)</span>
    &nbsp;|&nbsp; Speed :
    <span style='color:#00ff88;'>⚡ ~500 tokens/sec</span>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── VALIDATION ────────────────────────────────────────────
    if not selected_ticker:
        st.error("⚠️ Please enter a valid ticker symbol in the sidebar.")
        st.stop()

    if run_llm_report and not api_key:
        st.warning("⚠️ No Groq API key. AI Report skipped — all other tools will still run.")
        run_llm_report = False

    # Init shared variables
    rsi = vol = trend = macd_s = rsi_sig = rec = info = None
    df_analyzed = forecast_df = arima_result = None

    # ── DATA COLLECTION ───────────────────────────────────────
    st.markdown(
        f'<div class="section-title">📡 Data Collection — {display_name} ({selected_ticker})</div>',
        unsafe_allow_html=True
    )
    with st.spinner(f"Downloading {data_period} of data for {selected_ticker}..."):
        df_raw = fetch_data(selected_ticker, data_period)

    if df_raw is None or df_raw.empty:
        st.error(f"❌ No data found for **{selected_ticker}**. Check the ticker symbol and try again.")
        st.stop()

    current_price = float(df_raw['Close'].iloc[-1])
    prev_price    = float(df_raw['Close'].iloc[-2])
    price_chg     = current_price - prev_price
    price_chg_pct = (price_chg / prev_price) * 100
    high_52w      = float(df_raw['Close'].max())
    low_52w       = float(df_raw['Close'].min())
    chg_class     = "green" if price_chg >= 0 else "red"
    chg_arrow     = "▲" if price_chg >= 0 else "▼"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">{current_price:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Day Change</div>
            <div class="metric-value {chg_class}">
            {chg_arrow} {abs(price_chg):.2f} ({abs(price_chg_pct):.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">52W High</div>
            <div class="metric-value green">{high_52w:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">52W Low</div>
            <div class="metric-value red">{low_52w:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Trading Days</div>
            <div class="metric-value purple">{len(df_raw):,}</div>
        </div>""", unsafe_allow_html=True)

    # ── TECHNICAL ANALYSIS ────────────────────────────────────
    if run_technical:
        st.markdown('<div class="section-title">🔍 Technical Analysis</div>',
                    unsafe_allow_html=True)
        with st.spinner("Computing RSI, MACD, Bollinger Bands, Moving Averages..."):
            df_analyzed = compute_technicals(df_raw)

        latest = df_analyzed.iloc[-1]
        rsi    = float(latest['RSI'])
        trend  = "BULLISH" if float(latest['MA_20']) > float(latest['MA_50']) else "BEARISH"
        macd_s = "BULLISH" if float(latest['MACD']) > float(latest['Signal_Line']) else "BEARISH"
        vol    = float(latest['Volatility_20d']) * 100

        if rsi > 70:   rsi_sig, rsi_cls = "OVERBOUGHT", "red"
        elif rsi < 30: rsi_sig, rsi_cls = "OVERSOLD",   "green"
        else:          rsi_sig, rsi_cls = "NEUTRAL",     "purple"

        trend_cls = "green" if trend  == "BULLISH" else "red"
        macd_cls  = "green" if macd_s == "BULLISH" else "red"

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Trend</div>
                <div class="metric-value {trend_cls}">{trend}</div>
            </div>""", unsafe_allow_html=True)
        with t2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">RSI (14)</div>
                <div class="metric-value {rsi_cls}">{rsi:.1f} — {rsi_sig}</div>
            </div>""", unsafe_allow_html=True)
        with t3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">MACD Signal</div>
                <div class="metric-value {macd_cls}">{macd_s}</div>
            </div>""", unsafe_allow_html=True)
        with t4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Annual Volatility</div>
                <div class="metric-value purple">{vol:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    # ── ARIMA PREDICTION ──────────────────────────────────────
    if run_arima:
        st.markdown(
            f'<div class="section-title">🤖 ARIMA Forecast — Next {forecast_days} Days</div>',
            unsafe_allow_html=True
        )
        with st.spinner(f"Training ARIMA and forecasting {forecast_days} days..."):
            src_df = df_analyzed if df_analyzed is not None else df_raw
            forecast_df, rmse, mae, d_order = run_arima_model(src_df, forecast_days)

        pred_end  = float(forecast_df['Predicted'].iloc[-1])
        pct_chg   = (pred_end - current_price) / current_price * 100
        direction = "UP ▲" if pct_chg > 0 else "DOWN ▼"
        dir_cls   = "green" if pct_chg > 0 else "red"
        arima_result = {"rmse": rmse, "mae": mae,
                        "pct_chg": pct_chg, "pred_end": pred_end,
                        "direction": direction}

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value purple">ARIMA(5,{d_order},2)</div>
            </div>""", unsafe_allow_html=True)
        with a2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">RMSE Error</div>
                <div class="metric-value">{rmse:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with a3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{forecast_days}D Predicted Price</div>
                <div class="metric-value {dir_cls}">{pred_end:,.2f}</div>
            </div>""", unsafe_allow_html=True)
        with a4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Expected Move</div>
                <div class="metric-value {dir_cls}">
                {direction} {abs(pct_chg):.2f}%</div>
            </div>""", unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────
    chart_df = df_analyzed if df_analyzed is not None else df_raw
    st.markdown('<div class="section-title">📉 Price Chart</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_price_chart(chart_df.tail(365), selected_ticker, forecast_df),
        use_container_width=True
    )
    st.markdown('<div class="section-title">📊 Volume</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_volume_chart(chart_df.tail(365)), use_container_width=True)

    if forecast_df is not None:
        with st.expander("📋 View Full Forecast Table"):
            disp = forecast_df.copy()
            disp['Date'] = disp['Date'].astype(str)
            for c in ['Predicted', 'Lower_95', 'Upper_95']:
                disp[c] = disp[c].round(2)
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── SENTIMENT & NEWS ──────────────────────────────────────
    if run_sentiment:
        st.markdown('<div class="section-title">📰 Market Sentiment & News</div>',
                    unsafe_allow_html=True)
        with st.spinner("Fetching analyst data and headlines..."):
            info, news_items = get_sentiment(selected_ticker)

        rec     = info.get('recommendationKey', 'N/A').upper()
        rec_cls = ("green" if "buy"  in rec.lower() else
                   "red"   if "sell" in rec.lower() else "purple")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Analyst Recommendation</div>
                <div class="metric-value {rec_cls}">{rec}</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            target = info.get('targetMeanPrice', 'N/A')
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Mean Price Target</div>
                <div class="metric-value">{target}</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            n_analysts = info.get('numberOfAnalystOpinions', 'N/A')
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Number of Analysts</div>
                <div class="metric-value purple">{n_analysts}</div>
            </div>""", unsafe_allow_html=True)

        if news_items:
            st.markdown("**Recent Headlines**")
            for article in news_items:
                title = article.get('title', '')
                pub   = article.get('publisher', '')
                link  = article.get('link', '#')
                if title:
                    st.markdown(f"""
                    <div style='background:#16161f;border:1px solid #2a2a3a;
                    border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;'>
                        <a href="{link}" target="_blank"
                           style='color:#e8e8f0;text-decoration:none;
                           font-size:0.88rem;font-weight:600;'>{title}</a>
                        <div style='color:#6b6b80;font-size:0.72rem;
                        font-family:JetBrains Mono,monospace;
                        margin-top:0.2rem;'>{pub}</div>
                    </div>""", unsafe_allow_html=True)

    # ── GROQ LLM REPORT ───────────────────────────────────────
    if run_llm_report and api_key:
        st.markdown(
            f'<div class="section-title">🧠 AI Report — Groq / {groq_model}</div>',
            unsafe_allow_html=True
        )

        _data_store[selected_ticker]               = df_raw
        _data_store[f"{selected_ticker}_analyzed"] = (
            df_analyzed if df_analyzed is not None else df_raw
        )
        _data_store[f"{selected_ticker}_forecast"] = forecast_df

        tech_line  = (
            f"- Trend: {trend} | RSI: {round(rsi,1)} ({rsi_sig}) | "
            f"MACD: {macd_s} | Volatility: {round(vol,1)}%"
            if run_technical and df_analyzed is not None else ""
        )
        arima_line = (
            f"- ARIMA({forecast_days}d): {direction} {round(abs(pct_chg),2)}% → {round(pred_end,2)}"
            if run_arima and arima_result else ""
        )
        sent_line  = (
            f"- Analyst: {rec} | Target: {info.get('targetMeanPrice','N/A') if info else 'N/A'}"
            if run_sentiment and info else ""
        )

        prompt = f"""You are a professional quantitative analyst and stock market expert.

Analyze {selected_ticker} ({display_name}) with the data below:

MARKET DATA:
- Period        : {data_period}
- Current Price : {current_price:.2f}
- 52W High/Low  : {high_52w:.2f} / {low_52w:.2f}
- Day Change    : {price_chg:.2f} ({price_chg_pct:.2f}%)
{tech_line}
{arima_line}
{sent_line}

Write a complete professional stock analysis report with EXACTLY these 7 sections:

1. EXECUTIVE SUMMARY
2. CURRENT MARKET STATUS & PRICE ACTION
3. TECHNICAL ANALYSIS (RSI, MACD, Bollinger Bands, Moving Averages)
4. ARIMA FORECAST INTERPRETATION & CONFIDENCE
5. RISK ASSESSMENT (Bull scenario vs Bear scenario)
6. MACRO CONTEXT & MARKET SENTIMENT
7. FINAL RECOMMENDATION (Strong Buy / Buy / Hold / Sell / Strong Sell) with rationale

Use specific numbers, be data-driven, and keep a professional tone."""

        with st.spinner(f"🧠 {groq_model} is writing your analysis report..."):
            try:
                agent_app  = build_agent(api_key, groq_model)
                result     = agent_app.invoke({
                    "messages": [HumanMessage(content=prompt)]
                })
                llm_report = result["messages"][-1].content
                st.markdown(
                    f'<div class="report-box">{llm_report}</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                err = str(e)
                st.error(f"❌ Groq API Error: {err}")
                if "401" in err or "api_key" in err.lower() or "invalid" in err.lower():
                    st.info("💡 Invalid API key. Get a free key at console.groq.com")
                elif "rate" in err.lower() or "429" in err:
                    st.info("💡 Rate limit hit. Wait a moment or switch to llama-3.1-8b-instant.")
                elif "model" in err.lower():
                    st.info("💡 Model unavailable. Try llama-3.1-8b-instant from the sidebar.")
                else:
                    st.info("💡 Check console.groq.com for status and try again.")

    # ── EXPORT ────────────────────────────────────────────────
    st.markdown('<div class="section-title">💾 Export Data</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download Raw Data (CSV)",
            df_raw.to_csv().encode('utf-8'),
            f"{selected_ticker}_raw_data.csv", "text/csv"
        )
    with d2:
        if forecast_df is not None:
            st.download_button(
                "⬇️ Download Forecast (CSV)",
                forecast_df.to_csv(index=False).encode('utf-8'),
                f"{selected_ticker}_forecast.csv", "text/csv"
            )
