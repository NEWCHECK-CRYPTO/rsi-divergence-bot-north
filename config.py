import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EXCHANGE = "bybit"
TIMEZONE = "Asia/Colombo"

SYMBOLS = []
TOP_COINS_COUNT = 100
QUOTE_CURRENCY = "USDT"

EXCLUDED_SYMBOLS = [
    "USDC/USDT", "BUSD/USDT", "TUSD/USDT", "USDP/USDT", "FDUSD/USDT",
    "DAI/USDT", "USDD/USDT", "EURC/USDT", "AEUR/USDT",
]

EXCLUDE_LEVERAGED = True

# V9: Removed 15m
SCAN_TIMEFRAMES = ["1h", "4h", "1d", "1w"]

TIMEFRAME_CONFIRMATION_MAP = {
    "1M": "1d",
    "1w": "4h",
    "1d": "1h",
    "4h": "15m",
    "1h": "5m",
}

LOOKBACK_CANDLES = 50
MIN_SWING_DISTANCE = 5
MIN_PRICE_MOVE_PCT = 0.3
SWING_STRENGTH_BARS = 3

# V7/V9: 3-candle confirmation
CONFIRMATION_CANDLES = 3
CONFIRMATION_THRESHOLD = 2

# V7/V9: Minimum confidence
MIN_CONFIDENCE = 0.70

# V9: Swing strength filtering (ignore weak middle swings <30%)
MIN_STRENGTH_RATIO = 0.30

RSI_PERIOD = 14

ALERT_COOLDOWN = 1800
SCAN_INTERVAL = 120

RAG_KNOWLEDGE_PATH = "rsi_divergence_ms_rag.json"
GEMINI_MODEL = "gemini-1.5-flash"
