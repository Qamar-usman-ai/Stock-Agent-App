# 📈 Stock Agent App

> **AI-Powered Stock Market Analysis Agent**
> Built with LangGraph · Groq Llama 3.1 · yfinance · ARIMA · Streamlit

---

## 🌟 Overview

**Stock Agent App** is a production-grade AI agent that performs complete stock market analysis for any ticker symbol in the world. It combines real-time financial data collection, technical analysis, ARIMA time series forecasting, market sentiment analysis, and a Groq-powered LLM report — all inside a beautiful dark-themed Streamlit interface.

The agent is built using **LangGraph** which orchestrates multiple specialized tools, with **Groq's Llama 3.1** acting as the intelligent brain that decides what to analyze and writes the final professional report.

---

## 🚀 Features

- **Multi-Tool AI Agent** — LangGraph orchestrates 4 specialized tools automatically
- **Global Market Coverage** — US Stocks, Indices, Commodities, Currencies, Pakistan PSX, and any custom ticker
- **Technical Analysis** — RSI, MACD, Bollinger Bands, Moving Averages (20/50/200), Volatility
- **ARIMA Forecasting** — Predict 7–90 days into the future with 95% confidence intervals
- **Sentiment & News** — Analyst recommendations, price targets, and latest headlines
- **AI Professional Report** — Groq Llama 3.1 writes a complete 7-section investment report
- **Interactive Charts** — Candlestick, RSI, MACD, Volume charts powered by Plotly
- **CSV Export** — Download raw data and forecast data
- **Free to Run** — Uses Groq's free API tier

---

## 🏗️ Architecture

```
User (Streamlit UI)
        ↓
LangGraph Orchestrator
        ↓
Groq Llama 3.1 (Brain — decides which tools to call)
        ↓
┌───────────────────────────────────────────┐
│               4 TOOLS                     │
│                                           │
│  Tool 1 → Data Collection  (yfinance)     │
│  Tool 2 → Technical Analysis              │
│  Tool 3 → ARIMA Prediction                │
│  Tool 4 → Sentiment & News                │
└───────────────────────────────────────────┘
        ↓
Final Professional AI Report
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Qamar-usman-ai/stock-agent-app.git
cd stock-agent-app
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Your Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Go to **API Keys** → **Create API Key**
4. Copy your key (starts with `gsk_...`)

### 5. Run the App

```bash
streamlit run "Stock agent app · py.py"
```

The app will open in your browser at `http://localhost:8501`

---

## 🔑 Configuration

You do **not** need to create any `.env` file. Simply paste your Groq API key directly into the sidebar of the app when it opens.

If you prefer to set it as an environment variable:

```bash
# Windows
set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Mac / Linux
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🎯 How to Use

| Step | Action |
|------|--------|
| 1 | Paste your **Groq API key** in the sidebar |
| 2 | Select a **Groq Model** (default: llama-3.1-8b-instant) |
| 3 | Choose an **Asset Category** (US Stocks, Indices, Commodities, PSX, etc.) |
| 4 | Select the specific **Asset** or enter a custom ticker |
| 5 | Set **Historical Period** (6 months to 5 years) |
| 6 | Set **Forecast Days** (7 to 90 days) |
| 7 | Toggle which **Tools** to run |
| 8 | Click **🚀 Run Analysis** |

---

## 🤖 Supported Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `llama-3.1-8b-instant` | ⚡ Fastest | Good | Daily use, fast reports |
| `llama-3.3-70b-versatile` | Medium | Excellent | Deep analysis |
| `llama-3.1-70b-versatile` | Medium | Excellent | Complex reports |
| `mixtral-8x7b-32768` | Medium | Very Good | Long context |
| `gemma2-9b-it` | Fast | Good | Balanced |

---

## 🌍 Supported Markets

| Category | Examples |
|----------|---------|
| 🇺🇸 US Stocks | AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN |
| 🌍 Indices | ^GSPC (S&P500), ^IXIC (NASDAQ), ^DJI, ^VIX |
| 🛢️ Commodities | CL=F (Oil), GC=F (Gold), SI=F (Silver) |
| 💱 Currencies | EURUSD=X, GBPUSD=X, DX-Y.NYB |
| 🇵🇰 Pakistan PSX | ^KSE100, ENGRO.KA, HBL.KA, OGDC.KA |
| ✏️ Custom | Any valid Yahoo Finance ticker |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| `Streamlit` | Web UI framework |
| `LangGraph` | AI agent orchestration |
| `LangChain` | LLM tooling and abstractions |
| `ChatGroq` | Groq LLM integration |
| `Llama 3.1` | Language model (Meta / Groq) |
| `yfinance` | Financial data collection |
| `ARIMA` | Time series forecasting |
| `Plotly` | Interactive charts |
| `pandas / numpy` | Data processing |
| `scikit-learn` | Model evaluation metrics |
| `statsmodels` | Statistical modeling |

---

## 📊 Analysis Report Structure

The AI generates a 7-section professional report:

```
1. EXECUTIVE SUMMARY
2. CURRENT MARKET STATUS & PRICE ACTION
3. TECHNICAL ANALYSIS (RSI, MACD, Bollinger Bands, Moving Averages)
4. ARIMA FORECAST INTERPRETATION & CONFIDENCE
5. RISK ASSESSMENT (Bull scenario vs Bear scenario)
6. MACRO CONTEXT & MARKET SENTIMENT
7. FINAL RECOMMENDATION (Strong Buy / Buy / Hold / Sell / Strong Sell)
```

---

## 📁 Project Structure

```
stock-agent-app/
│
├── Stock agent app · py.py     # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions CI/CD
└── .gitignore                   # Git ignore rules
```

---

## ⚠️ Disclaimer

This application is built **for educational and research purposes only**. The analysis, predictions, and AI-generated reports produced by this tool do **not** constitute financial advice. Always conduct your own research and consult a qualified financial advisor before making any investment decisions. Past performance is not indicative of future results.

---

## 👨‍💻 Author

**Qamar Usman**
Machine Learning Engineer | Kaggle Expert (Top 0.4%) | LLMs · Time Series · Medical AI

- 🐙 GitHub: [github.com/Qamar-usman-ai](https://github.com/Qamar-usman-ai)
- 📊 Kaggle: [kaggle.com/qamarmath](https://www.kaggle.com/qamarmath)
- 💼 LinkedIn: [linkedin.com/in/qamar-usman-92a9752b7](https://www.linkedin.com/in/qamar-usman-92a9752b7)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
