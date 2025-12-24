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

# V10: Optimized timeframes (removed 1w until fully tested)
SCAN_TIMEFRAMES = ["1h", "4h", "1d"]

# V10: Multi-timeframe trend confirmation mapping
TREND_CONFIRMATION_MAP = {
    "1M": "1d",
    "1w": "4h",
    "1d": "1h",
    "4h": "15m",
    "1h": "5m",
}

# Alias for backward compatibility
TIMEFRAME_CONFIRMATION_MAP = TREND_CONFIRMATION_MAP

LOOKBACK_CANDLES = 50
MIN_SWING_DISTANCE = 5
MIN_PRICE_MOVE_PCT = 0.3
SWING_STRENGTH_BARS = 3

# V10: 2-candle confirmation (optimal!)
CONFIRMATION_CANDLES = 2
CONFIRMATION_THRESHOLD = 2  # Need both candles (2/2)

# V10: Minimum confidence (higher for reliability)
MIN_CONFIDENCE = 0.70

# V10: Recency check - max candles since Swing 2
MAX_CANDLES_SINCE_SWING2 = {
    "1h": 10,
    "4h": 8,
    "1d": 5,
    "1w": 3,
    "1M": 2,
}

# V10: Minimum ADX for trend confirmation
MIN_ADX_STRONG = 25
MIN_ADX_MODERATE = 20

RSI_PERIOD = 14

ALERT_COOLDOWN = 1800
SCAN_INTERVAL = 120

RAG_KNOWLEDGE_PATH = "rsi_divergence_ms_rag.json"
GEMINI_MODEL = "gemini-1.5-flash"
