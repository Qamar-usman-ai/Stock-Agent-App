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
from typing import TypedDict, Annotated, Sequence
import operator

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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LAZY IMPORTS (avoid crash on Streamlit Cloud if a pkg is missing)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_heavy_libs():
    errors = []
    libs = {}
    try:
        import yfinance as yf
        libs["yf"] = yf
    except Exception as e:
        errors.append(f"yfinance: {e}")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        libs["ARIMA"] = ARIMA
    except Exception as e:
        errors.append(f"statsmodels: {e}")
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        libs["ChatGroq"] = ChatGroq
        libs["HumanMessage"] = HumanMessage
        libs["AIMessage"] = AIMessage
        libs["SystemMessage"] = SystemMessage
        libs["ToolMessage"] = ToolMessage
        libs["tool"] = tool
        libs["StateGraph"] = StateGraph
        libs["END"] = END
        libs["ToolNode"] = ToolNode
    except Exception as e:
        errors.append(f"langchain/langgraph: {e}")
    return libs, errors

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
def fetch_stock_data(ticker: str, period: str = "1y"):
    """Robust yfinance fetch with multiple fallbacks."""
    libs, _ = load_heavy_libs()
    yf = libs.get("yf")
    if yf is None:
        return None, "yfinance not available"

    ticker = ticker.strip().upper()

    # Attempt 1: Ticker.history()
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True, actions=False)
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df, None
    except Exception:
        pass

    # Attempt 2: yf.download()
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, show_errors=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df, None
    except Exception:
        pass

    # Attempt 3: date-range download
    try:
        days_map = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        days = days_map.get(period, 365)
        end = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False, show_errors=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df, None
    except Exception as e:
        return None, f"All fetch attempts failed for '{ticker}': {e}"

    return None, f"No data returned for '{ticker}'. Verify the ticker on finance.yahoo.com"


def collect_stock_data(ticker: str, period: str = "1y"):
    libs, _ = load_heavy_libs()
    yf = libs.get("yf")

    df, err = fetch_stock_data(ticker, period)
    if err:
        return {"error": err}

    info = {}
    if yf:
        try:
            t = yf.Ticker(ticker)
            raw = t.info or {}
            info = {
                "name":        raw.get("longName", ticker),
                "sector":      raw.get("sector", "N/A"),
                "industry":    raw.get("industry", "N/A"),
                "market_cap":  raw.get("marketCap", 0),
                "pe_ratio":    raw.get("trailingPE", None),
                "52w_high":    raw.get("fiftyTwoWeekHigh", None),
                "52w_low":     raw.get("fiftyTwoWeekLow", None),
                "avg_volume":  raw.get("averageVolume", None),
                "beta":        raw.get("beta", None),
                "dividend_yield": raw.get("dividendYield", None),
                "currency":    raw.get("currency", "USD"),
                "exchange":    raw.get("exchange", "N/A"),
            }
        except Exception:
            info = {"name": ticker}

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    change    = float(latest["Close"]) - float(prev["Close"])
    change_pct = (change / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0

    return {
        "ticker":     ticker,
        "period":     period,
        "rows":       len(df),
        "info":       info,
        "latest": {
            "date":    str(df.index[-1].date()),
            "open":    round(float(latest["Open"]),  4),
            "high":    round(float(latest["High"]),  4),
            "low":     round(float(latest["Low"]),   4),
            "close":   round(float(latest["Close"]), 4),
            "volume":  int(latest["Volume"]) if "Volume" in df.columns else 0,
            "change":  round(change, 4),
            "change_pct": round(change_pct, 2),
        },
        "df": df,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_technical_analysis(df: pd.DataFrame):
    close = df["Close"].squeeze().astype(float)

    # Moving averages
    df["MA20"]  = close.rolling(20).mean()
    df["MA50"]  = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

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
        lc = (df["Low"].astype(float)  - close.shift()).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # OBV
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        df["OBV"] = obv

    # Volatility (annualised)
    df["Returns"]    = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100

    # Summary
    last = df.iloc[-1]
    cur_close = float(close.iloc[-1])
    rsi_val   = float(last["RSI"])   if not np.isnan(last["RSI"])  else 50
    macd_val  = float(last["MACD"])  if not np.isnan(last["MACD"]) else 0
    macd_sig  = float(last["MACD_Signal"]) if not np.isnan(last["MACD_Signal"]) else 0

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

    rsi_sig   = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    macd_sig2 = "Bullish"    if macd_val > macd_sig else "Bearish"

    bb_upper = float(last["BB_Upper"]) if not np.isnan(last["BB_Upper"]) else cur_close
    bb_lower = float(last["BB_Lower"]) if not np.isnan(last["BB_Lower"]) else cur_close
    bb_pos = ((cur_close - bb_lower) / (bb_upper - bb_lower) * 100
              if (bb_upper - bb_lower) != 0 else 50)

    vol_val = float(last["Volatility"]) if not np.isnan(last["Volatility"]) else 0

    return {
        "df": df,
        "summary": {
            "trend": trend,
            "rsi":   round(rsi_val, 2),
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
#  TOOL 3 — ARIMA FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
def run_arima_forecast(df: pd.DataFrame, forecast_days: int = 30):
    libs, _ = load_heavy_libs()
    ARIMA_cls = libs.get("ARIMA")
    if ARIMA_cls is None:
        return {"error": "statsmodels not available"}

    close = df["Close"].squeeze().astype(float).dropna()
    if len(close) < 50:
        return {"error": "Insufficient data for ARIMA (need ≥ 50 rows)"}

    try:
        model  = ARIMA_cls(close, order=(2, 1, 2))
        fitted = model.fit()
        forecast_obj = fitted.get_forecast(steps=forecast_days)
        mean_fc = forecast_obj.predicted_mean
        ci      = forecast_obj.conf_int(alpha=0.05)

        last_date  = df.index[-1]
        future_idx = pd.bdate_range(start=last_date + timedelta(days=1),
                                    periods=forecast_days)

        fc_df = pd.DataFrame({
            "Date":      future_idx[:len(mean_fc)],
            "Forecast":  mean_fc.values[:len(future_idx)],
            "Lower_95":  ci.iloc[:len(future_idx), 0].values,
            "Upper_95":  ci.iloc[:len(future_idx), 1].values,
        })

        cur = float(close.iloc[-1])
        end = float(fc_df["Forecast"].iloc[-1])
        chg = ((end - cur) / cur) * 100

        return {
            "fc_df":    fc_df,
            "current":  round(cur, 4),
            "forecast_end": round(end, 4),
            "expected_change_pct": round(chg, 2),
            "direction": "Bullish" if chg > 0 else "Bearish",
            "forecast_days": forecast_days,
        }
    except Exception as e:
        return {"error": f"ARIMA failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 4 — SENTIMENT & NEWS
# ══════════════════════════════════════════════════════════════════════════════
def run_sentiment_analysis(ticker: str):
    libs, _ = load_heavy_libs()
    yf = libs.get("yf")
    if yf is None:
        return {"error": "yfinance not available"}

    result = {
        "recommendations": [],
        "news": [],
        "analyst_summary": {},
        "price_targets": {},
    }

    try:
        t = yf.Ticker(ticker)

        # Analyst recommendations
        try:
            rec = t.recommendations
            if rec is not None and not rec.empty:
                # Flatten multi-index if present
                if isinstance(rec.columns, pd.MultiIndex):
                    rec.columns = [" ".join(c).strip() for c in rec.columns]
                result["recommendations"] = rec.tail(5).to_dict("records")
        except Exception:
            pass

        # News headlines
        try:
            news = t.news or []
            result["news"] = [
                {"title": n.get("title", ""), "publisher": n.get("publisher", ""),
                 "link": n.get("link", "")}
                for n in news[:6]
            ]
        except Exception:
            pass

        # Analyst price target
        try:
            info = t.info or {}
            result["price_targets"] = {
                "target_mean":  info.get("targetMeanPrice"),
                "target_high":  info.get("targetHighPrice"),
                "target_low":   info.get("targetLowPrice"),
                "current_price": info.get("currentPrice"),
                "recommendation_key": info.get("recommendationKey", "N/A"),
                "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
            }
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  LangGraph AGENT
# ══════════════════════════════════════════════════════════════════════════════
def build_agent(groq_api_key: str, model_name: str):
    libs, errs = load_heavy_libs()
    required = ["ChatGroq", "HumanMessage", "SystemMessage", "tool",
                "StateGraph", "END", "ToolNode"]
    missing = [k for k in required if k not in libs]
    if missing:
        return None, f"Missing libraries: {missing}. Errors: {errs}"

    ChatGroq    = libs["ChatGroq"]
    HumanMessage = libs["HumanMessage"]
    SystemMessage = libs["SystemMessage"]
    tool_dec    = libs["tool"]
    StateGraph  = libs["StateGraph"]
    END         = libs["END"]
    ToolNode    = libs["ToolNode"]

    # ── Define LangChain tools ────────────────────────────────────────────────
    _data_store: dict = {}  # shared state between tools within one run

    @tool_dec
    def collect_data_tool(ticker: str, period: str = "1y") -> str:
        """Collect historical price data for a stock ticker."""
        res = collect_stock_data(ticker, period)
        if "error" in res:
            return f"ERROR: {res['error']}"
        _data_store["data"]   = res
        _data_store["ticker"] = ticker
        _data_store["period"] = period
        return json.dumps({k: v for k, v in res.items() if k != "df"}, default=str)

    @tool_dec
    def technical_analysis_tool(ticker: str) -> str:
        """Run technical analysis (RSI, MACD, Bollinger Bands, MAs) on the stock."""
        data = _data_store.get("data")
        if data is None:
            res = collect_stock_data(ticker, "1y")
            if "error" in res:
                return f"ERROR: {res['error']}"
            _data_store["data"] = res
        df  = _data_store["data"]["df"]
        res = run_technical_analysis(df.copy())
        _data_store["ta"] = res
        return json.dumps(res["summary"], default=str)

    @tool_dec
    def arima_forecast_tool(ticker: str, forecast_days: int = 30) -> str:
        """Run ARIMA forecast for the next N days."""
        data = _data_store.get("data")
        if data is None:
            res = collect_stock_data(ticker, "1y")
            if "error" in res:
                return f"ERROR: {res['error']}"
            _data_store["data"] = res
        df  = _data_store["data"]["df"]
        res = run_arima_forecast(df.copy(), forecast_days)
        if "error" in res:
            return f"ERROR: {res['error']}"
        _data_store["arima"] = res
        return json.dumps({k: v for k, v in res.items() if k != "fc_df"}, default=str)

    @tool_dec
    def sentiment_tool(ticker: str) -> str:
        """Fetch analyst recommendations, price targets and latest news headlines."""
        res = run_sentiment_analysis(ticker)
        _data_store["sentiment"] = res
        return json.dumps(res, default=str)

    tools_list = [collect_data_tool, technical_analysis_tool,
                  arima_forecast_tool, sentiment_tool]

    llm = ChatGroq(api_key=groq_api_key, model=model_name,
                   temperature=0.2, max_tokens=4096)
    llm_with_tools = llm.bind_tools(tools_list)

    # ── LangGraph state ───────────────────────────────────────────────────────
    class AgentState(TypedDict):
        messages: Annotated[Sequence, operator.add]

    tool_node = ToolNode(tools_list)

    def agent_node(state: AgentState):
        msgs = state["messages"]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    return compiled, _data_store


def run_agent(ticker: str, period: str, forecast_days: int,
              groq_api_key: str, model_name: str,
              use_ta: bool, use_arima: bool, use_sentiment: bool):

    compiled, store = build_agent(groq_api_key, model_name)
    if compiled is None:
        return None, store, "Agent build failed"   # store contains the error string here

    libs   = load_heavy_libs()[0]
    SM_cls = libs.get("SystemMessage")
    HM_cls = libs.get("HumanMessage")

    tools_requested = ["collect_data_tool"]
    if use_ta:        tools_requested.append("technical_analysis_tool")
    if use_arima:     tools_requested.append("arima_forecast_tool")
    if use_sentiment: tools_requested.append("sentiment_tool")

    system_prompt = f"""You are an expert financial analyst AI. Analyze the stock {ticker}.

Use these tools IN ORDER:
1. collect_data_tool — always first
{"2. technical_analysis_tool — run technical analysis" if use_ta else ""}
{"3. arima_forecast_tool — run ARIMA forecast for " + str(forecast_days) + " days" if use_arima else ""}
{"4. sentiment_tool — fetch news and analyst data" if use_sentiment else ""}

After ALL tools finish, write a PROFESSIONAL 7-SECTION INVESTMENT REPORT:

═══════════════════════════════════════════════════════════
              STOCK ANALYSIS REPORT — {ticker.upper()}
═══════════════════════════════════════════════════════════

1. EXECUTIVE SUMMARY
2. CURRENT MARKET STATUS & PRICE ACTION  
3. TECHNICAL ANALYSIS (RSI, MACD, Bollinger Bands, Moving Averages)
4. ARIMA FORECAST INTERPRETATION & CONFIDENCE
5. RISK ASSESSMENT (Bull vs Bear scenario)
6. MACRO CONTEXT & MARKET SENTIMENT
7. FINAL RECOMMENDATION (Strong Buy / Buy / Hold / Sell / Strong Sell)

Be data-driven, specific with numbers, and professional."""

    user_msg = (f"Analyze {ticker.upper()} for the past {period} "
                f"with a {forecast_days}-day forecast. "
                f"Produce the full investment report.")

    initial_state = {
        "messages": [SM_cls(content=system_prompt), HM_cls(content=user_msg)]
    }

    try:
        final_state = compiled.invoke(initial_state, config={"recursion_limit": 25})
        ai_msgs = [m for m in final_state["messages"]
                   if hasattr(m, "content") and not hasattr(m, "tool_calls")]
        report = ai_msgs[-1].content if ai_msgs else "No report generated."
        return report, store, None
    except Exception as e:
        return None, store, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def make_price_chart(df: pd.DataFrame, ticker: str, ta_df: pd.DataFrame = None):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[f"{ticker} — Price & Bollinger Bands", "RSI", "MACD"],
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
    ), row=1, col=1)

    source = ta_df if (ta_df is not None and "BB_Upper" in ta_df.columns) else df

    for col, color, name in [
        ("MA20",     "#ffaa00", "MA20"),
        ("MA50",     "#00aaff", "MA50"),
        ("MA200",    "#ff00aa", "MA200"),
        ("BB_Upper", "#888888", "BB Upper"),
        ("BB_Lower", "#888888", "BB Lower"),
        ("BB_Middle","#444444", "BB Mid"),
    ]:
        if col in source.columns:
            fig.add_trace(go.Scatter(
                x=source.index, y=source[col], name=name,
                line=dict(color=color, width=1,
                          dash="dot" if "BB" in col else "solid"),
                opacity=0.8,
            ), row=1, col=1)

    # RSI
    if "RSI" in source.columns:
        fig.add_trace(go.Scatter(
            x=source.index, y=source["RSI"], name="RSI",
            line=dict(color="#ffaa00", width=1.5)
        ), row=2, col=1)
        for lvl, col in [(70, "#ff4444"), (30, "#00ff88")]:
            fig.add_hline(y=lvl, line_color=col, line_dash="dot",
                          line_width=1, row=2, col=1)

    # MACD
    if "MACD" in source.columns:
        fig.add_trace(go.Scatter(
            x=source.index, y=source["MACD"], name="MACD",
            line=dict(color="#00d4ff", width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=source.index, y=source["MACD_Signal"], name="Signal",
            line=dict(color="#ff4444", width=1.5)
        ), row=3, col=1)
        if "MACD_Hist" in source.columns:
            colors = ["#00ff88" if v >= 0 else "#ff4444"
                      for v in source["MACD_Hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=source.index, y=source["MACD_Hist"],
                name="Histogram", marker_color=colors, opacity=0.6,
            ), row=3, col=1)

    fig.update_layout(
        height=700, paper_bgcolor="#0e1117", plot_bgcolor="#0d1117",
        font=dict(color="#ccd6f6"), legend=dict(bgcolor="#1a1a2e"),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1a1a2e", row=i, col=1)
        fig.update_yaxes(gridcolor="#1a1a2e", row=i, col=1)
    return fig


def make_forecast_chart(df: pd.DataFrame, fc_df: pd.DataFrame, ticker: str):
    fig = go.Figure()

    tail = df["Close"].squeeze().astype(float).tail(90)
    fig.add_trace(go.Scatter(
        x=tail.index, y=tail.values, name="Historical",
        line=dict(color="#00d4ff", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["Forecast"], name="ARIMA Forecast",
        line=dict(color="#ffaa00", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
        y=pd.concat([fc_df["Upper_95"], fc_df["Lower_95"][::-1]]),
        fill="toself", fillcolor="rgba(255,170,0,0.1)",
        line=dict(color="rgba(255,170,0,0)"),
        name="95% CI",
    ))
    fig.update_layout(
        title=f"{ticker} — ARIMA {len(fc_df)}-Day Forecast",
        height=420, paper_bgcolor="#0e1117", plot_bgcolor="#0d1117",
        font=dict(color="#ccd6f6"),
        xaxis=dict(gridcolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e"),
    )
    return fig


def make_volume_chart(df: pd.DataFrame, ticker: str):
    colors = ["#00ff88" if r >= 0 else "#ff4444"
              for r in df["Close"].diff().fillna(0)]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Volume"].astype(float),
        marker_color=colors, opacity=0.7, name="Volume"
    ))
    fig.update_layout(
        title=f"{ticker} — Volume",
        height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0d1117",
        font=dict(color="#ccd6f6"),
        xaxis=dict(gridcolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Groq API key (check secrets first, then env, then input)
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
        help="Get free key at console.groq.com",
    )

    model_name = st.selectbox("🤖 Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    st.markdown("---")
    st.markdown("### 🌍 Asset Selection")

    category = st.selectbox("Category", [
        "🇺🇸 US Stocks",
        "🌍 Indices",
        "🛢️ Commodities",
        "💱 Currencies",
        "🇵🇰 Pakistan PSX",
        "✏️ Custom Ticker",
    ])

    preset_map = {
        "🇺🇸 US Stocks":    ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","NFLX","JPM","V"],
        "🌍 Indices":        ["^GSPC","^IXIC","^DJI","^VIX","^FTSE","^N225"],
        "🛢️ Commodities":   ["GC=F","CL=F","SI=F","NG=F","ZC=F"],
        "💱 Currencies":     ["EURUSD=X","GBPUSD=X","JPYUSD=X","DX-Y.NYB","PKRUSD=X"],
        "🇵🇰 Pakistan PSX": ["^KSE100","ENGRO.KA","HBL.KA","OGDC.KA","PSO.KA","LUCK.KA"],
        "✏️ Custom Ticker":  [],
    }

    if category == "✏️ Custom Ticker":
        ticker = st.text_input("Ticker Symbol", value="AAPL",
                               placeholder="e.g. AAPL, ^GSPC, GC=F").upper().strip()
    else:
        options = preset_map[category]
        ticker = st.selectbox("Asset", options)

    st.markdown("---")
    st.markdown("### 📅 Parameters")

    period_label = st.select_slider("Historical Period", [
        "6mo","1y","2y","3y","5y"
    ], value="1y")

    forecast_days = st.slider("Forecast Days", 7, 90, 30, step=7)

    st.markdown("---")
    st.markdown("### 🔧 Tools")
    use_ta        = st.checkbox("Technical Analysis",   value=True)
    use_arima     = st.checkbox("ARIMA Forecasting",    value=True)
    use_sentiment = st.checkbox("Sentiment & News",     value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis")


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
    st.info("👈 Configure your analysis in the sidebar, then click **🚀 Run Analysis**")

    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, desc in [
        (col1, "📊", "Technical Analysis", "RSI · MACD · Bollinger Bands · MAs"),
        (col2, "🔮", "ARIMA Forecast",      "7–90 day price forecasting"),
        (col3, "📰", "Sentiment & News",    "Analyst ratings & headlines"),
        (col4, "🤖", "AI Report",           "Full Groq LLM investment report"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div style='font-size:2rem'>{icon}</div>
              <div style='color:#00d4ff;font-weight:bold'>{title}</div>
              <div class='label'>{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# ── Validate API key ──────────────────────────────────────────────────────────
if not groq_key:
    st.error("❌ Please enter your Groq API key in the sidebar.")
    st.stop()

if not ticker:
    st.error("❌ Please enter a ticker symbol.")
    st.stop()

# ── Step 1: Fetch data ────────────────────────────────────────────────────────
with st.spinner(f"⏳ Fetching data for **{ticker}** …"):
    raw = collect_stock_data(ticker, period_label)

if "error" in raw:
    st.error(f"❌ {raw['error']}")
    st.markdown("""
    **Troubleshooting tips:**
    - Double-check the ticker symbol (e.g. `AAPL`, `^GSPC`, `GC=F`)
    - Verify it on [finance.yahoo.com](https://finance.yahoo.com)
    - PSX tickers require `.KA` suffix (e.g. `HBL.KA`)
    """)
    st.stop()

df   = raw["df"].copy()
info = raw.get("info", {})
lat  = raw["latest"]

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown(f"<div class='section-header'>📊 {info.get('name', ticker)} — Live Snapshot</div>",
            unsafe_allow_html=True)

cols = st.columns(5)
change_cls = "change-pos" if lat["change"] >= 0 else "change-neg"
sign       = "▲" if lat["change"] >= 0 else "▼"

kpis = [
    ("Current Price",   f"{lat['close']:,.4f}",
     f"<span class='{change_cls}'>{sign} {abs(lat['change_pct']):.2f}%</span>"),
    ("Open",            f"{lat['open']:,.4f}",  lat["date"]),
    ("High",            f"{lat['high']:,.4f}",  "Today"),
    ("Low",             f"{lat['low']:,.4f}",   "Today"),
    ("Volume",          f"{lat['volume']:,}",   "Shares"),
]
for col, (label, val, sub) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>{label}</div>
          <div class='value'>{val}</div>
          <div>{sub}</div>
        </div>""", unsafe_allow_html=True)

# extra info row
if info.get("sector") and info["sector"] != "N/A":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sector",   info.get("sector","N/A"))
    c2.metric("P/E Ratio", info.get("pe_ratio","N/A"))
    if info.get("52w_high"):
        c3.metric("52W High", f"{info['52w_high']:,.2f}")
    if info.get("52w_low"):
        c4.metric("52W Low",  f"{info['52w_low']:,.2f}")

# ── Step 2: Technical Analysis ────────────────────────────────────────────────
ta_result = None
if use_ta:
    with st.spinner("🔧 Running Technical Analysis …"):
        ta_result = run_technical_analysis(df.copy())
    df_ta = ta_result["df"]
    ta_sum = ta_result["summary"]

    st.markdown("<div class='section-header'>📉 Technical Analysis</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend",        ta_sum["trend"])
    c2.metric("RSI",          f"{ta_sum['rsi']} ({ta_sum['rsi_signal']})")
    c3.metric("MACD Signal",  ta_sum["macd_signal"])
    c4.metric("Volatility",   f"{ta_sum['volatility_pct']}%")
    c1.metric("MA20",  f"{ta_sum['ma20']:,.2f}" if ta_sum["ma20"] else "N/A")
    c2.metric("MA50",  f"{ta_sum['ma50']:,.2f}" if ta_sum["ma50"] else "N/A")
    c3.metric("MA200", f"{ta_sum['ma200']:,.2f}" if ta_sum["ma200"] else "N/A")
    c4.metric("BB Position",  f"{ta_sum['bb_position_pct']}%")

    st.plotly_chart(make_price_chart(df, ticker, df_ta), use_container_width=True)
else:
    st.plotly_chart(make_price_chart(df, ticker), use_container_width=True)

# Volume chart
if "Volume" in df.columns:
    st.plotly_chart(make_volume_chart(df, ticker), use_container_width=True)

# ── Step 3: ARIMA ─────────────────────────────────────────────────────────────
arima_result = None
if use_arima:
    with st.spinner(f"🔮 Running ARIMA Forecast ({forecast_days} days) …"):
        arima_result = run_arima_forecast(df.copy(), forecast_days)

    st.markdown("<div class='section-header'>🔮 ARIMA Forecast</div>",
                unsafe_allow_html=True)

    if "error" in arima_result:
        st.warning(f"ARIMA: {arima_result['error']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",  f"{arima_result['current']:,.4f}")
        c2.metric(f"Day {forecast_days} Forecast",
                  f"{arima_result['forecast_end']:,.4f}")
        c3.metric("Expected Change", f"{arima_result['expected_change_pct']:+.2f}%")
        c4.metric("Direction",       arima_result["direction"])

        st.plotly_chart(
            make_forecast_chart(df, arima_result["fc_df"], ticker),
            use_container_width=True
        )

        csv_fc = arima_result["fc_df"].to_csv(index=False)
        st.download_button("⬇️ Download Forecast CSV", csv_fc,
                           f"{ticker}_forecast.csv", "text/csv")

# ── Step 4: Sentiment ─────────────────────────────────────────────────────────
sent_result = None
if use_sentiment:
    with st.spinner("📰 Fetching Sentiment & News …"):
        sent_result = run_sentiment_analysis(ticker)

    st.markdown("<div class='section-header'>📰 Market Sentiment & News</div>",
                unsafe_allow_html=True)

    pt = sent_result.get("price_targets", {})
    if pt:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyst Rating", str(pt.get("recommendation_key","N/A")).upper())
        if pt.get("target_mean"):
            c2.metric("Target (Mean)", f"{pt['target_mean']:,.2f}")
        if pt.get("target_high"):
            c3.metric("Target (High)", f"{pt['target_high']:,.2f}")
        if pt.get("target_low"):
            c4.metric("Target (Low)",  f"{pt['target_low']:,.2f}")

    news = sent_result.get("news", [])
    if news:
        st.markdown("**Latest Headlines:**")
        for n in news:
            title = n.get("title", "")
            pub   = n.get("publisher", "")
            link  = n.get("link", "#")
            if title:
                st.markdown(f"• [{title}]({link}) — *{pub}*")

# ── Raw data download ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📥 Raw Data Export</div>",
            unsafe_allow_html=True)
csv_raw = df.reset_index().to_csv(index=False)
st.download_button("⬇️ Download Historical Data CSV", csv_raw,
                   f"{ticker}_data.csv", "text/csv")

# ── Step 5: AI Report ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🤖 AI Investment Report (Groq LLM)</div>",
            unsafe_allow_html=True)

with st.spinner("🧠 Agent thinking and writing report … (may take 30–60s)"):
    report, store, agent_err = run_agent(
        ticker, period_label, forecast_days,
        groq_key, model_name,
        use_ta, use_arima, use_sentiment
    )

if agent_err:
    st.error(f"Agent error: {agent_err}")
    st.info("The charts and metrics above are still valid — only the LLM report failed.")
elif report:
    st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)
    st.download_button("⬇️ Download Report", report,
                       f"{ticker}_report.txt", "text/plain")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8892b0;font-size:0.8rem'>
⚠️ <b>Disclaimer:</b> For educational purposes only. Not financial advice.
Always consult a qualified financial advisor before making investment decisions.<br>
Built by <b>Qamar Usman</b> · LangGraph · Groq · yfinance · Streamlit
</div>
""", unsafe_allow_html=True)
