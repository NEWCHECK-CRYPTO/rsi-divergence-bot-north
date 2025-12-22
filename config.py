import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EXCHANGE = "binance"

# Sri Lanka Timezone
TIMEZONE = "Asia/Colombo"  # UTC+5:30

# Dynamic symbols - will be fetched from Binance
# Set to empty list to auto-fetch top 100 by volume
SYMBOLS = []  # Auto-fetch top 100

# How many top coins to monitor (by 24h volume)
TOP_COINS_COUNT = 100

# Only USDT pairs
QUOTE_CURRENCY = "USDT"

# Exclude stablecoins and leveraged tokens
EXCLUDED_SYMBOLS = [
    "USDC/USDT", "BUSD/USDT", "TUSD/USDT", "USDP/USDT", "FDUSD/USDT",
    "DAI/USDT", "USDD/USDT", "EURC/USDT", "AEUR/USDT",
]

# Exclude leveraged tokens (contain UP, DOWN, BULL, BEAR)
EXCLUDE_LEVERAGED = True

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

# Scan interval (seconds) - 3 minutes for faster signals
SCAN_INTERVAL = 180

ALERT_COOLDOWN = 1800
RAG_KNOWLEDGE_PATH = "rsi_divergence_ms_rag.json"
GEMINI_MODEL = "gemini-1.5-flash"
