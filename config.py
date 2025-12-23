import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# =============================================================================
# EXCHANGE - Using Bybit (no geo-restrictions!)
# =============================================================================
EXCHANGE = "bybit"

# =============================================================================
# TIMEZONE
# =============================================================================
TIMEZONE = "Asia/Colombo"  # Sri Lanka (UTC+5:30)

# =============================================================================
# SYMBOLS CONFIGURATION
# =============================================================================
SYMBOLS = []  # Empty = auto-fetch top coins by volume
TOP_COINS_COUNT = 100  # Monitor top 100 coins!
QUOTE_CURRENCY = "USDT"

# Exclude stablecoins
EXCLUDED_SYMBOLS = [
    "USDC/USDT", "BUSD/USDT", "TUSD/USDT", "USDP/USDT", "FDUSD/USDT",
    "DAI/USDT", "USDD/USDT", "EURC/USDT", "AEUR/USDT",
]

# Exclude leveraged tokens
EXCLUDE_LEVERAGED = True

# =============================================================================
# TIMEFRAMES
# =============================================================================
SCAN_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]

# Confirmation timeframe mapping
TIMEFRAME_CONFIRMATION_MAP = {
    "1M": "1d",
    "1w": "4h",
    "1d": "1h",
    "4h": "15m",
    "1h": "5m",
    "15m": "1m",
}

# =============================================================================
# DIVERGENCE DETECTION SETTINGS - BALANCED
# =============================================================================

LOOKBACK_CANDLES = 50   # Look back 50 candles
MIN_SWING_DISTANCE = 5  # Swings must be 5+ candles apart
MIN_PRICE_MOVE_PCT = 0.3  # Swing must be 0.3% move (low for more signals)
SWING_STRENGTH_BARS = 3  # 3 candles each side

# =============================================================================
# RSI SETTINGS
# =============================================================================
RSI_PERIOD = 14

# =============================================================================
# ALERT SETTINGS
# =============================================================================
ALERT_COOLDOWN = 1800  # 30 min cooldown per symbol/timeframe
SCAN_INTERVAL = 120    # Scan every 2 minutes (faster!)

# =============================================================================
# RAG / GEMINI SETTINGS
# =============================================================================
RAG_KNOWLEDGE_PATH = "rsi_divergence_ms_rag.json"
GEMINI_MODEL = "gemini-1.5-flash"
