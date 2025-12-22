import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EXCHANGE = "binance"

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
]

SCAN_TIMEFRAMES = ["1d", "4h", "1h"]

TIMEFRAME_CONFIRMATION_MAP = {
    "1M": "1d", "1w": "4h", "1d": "1h",
    "4h": "15m", "1h": "5m", "15m": "1m",
}

RSI_PERIOD = 14
RSI_LOOKBACK_CANDLES = 50
SWING_DETECTION_BARS = 5
PRICE_TOLERANCE_PERCENT = 0.5
RSI_TOLERANCE = 2.0
MIN_CONFIDENCE_THRESHOLD = 0.60
SCAN_INTERVAL = 300
ALERT_COOLDOWN = 1800
RAG_KNOWLEDGE_PATH = "rsi_divergence_ms_rag.json"

# Updated model name for new SDK
GEMINI_MODEL = "gemini-2.0-flash"
