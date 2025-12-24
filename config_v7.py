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
# TIMEFRAMES - V7: REMOVED 15m (too noisy!)
# =============================================================================
SCAN_TIMEFRAMES = ["1h", "4h", "1d", "1w"]  # Removed 15m

# Confirmation timeframe mapping
TIMEFRAME_CONFIRMATION_MAP = {
    "1M": "1d",
    "1w": "4h",
    "1d": "1h",
    "4h": "15m",  # Keep for confirmation, not scanning
    "1h": "5m",   # Keep for confirmation, not scanning
}

# =============================================================================
# DIVERGENCE DETECTION SETTINGS - BALANCED
# =============================================================================

LOOKBACK_CANDLES = 50   # Look back 50 candles
MIN_SWING_DISTANCE = 5  # Swings must be 5+ candles apart
MIN_PRICE_MOVE_PCT = 0.3  # Swing must be 0.3% move (low for more signals)
SWING_STRENGTH_BARS = 3  # 3 candles each side

# =============================================================================
# V7 NEW: 3-CANDLE CONFIRMATION SETTINGS
# =============================================================================
CONFIRMATION_CANDLES = 3  # Check next 3 candles after swing2
CONFIRMATION_THRESHOLD = 2  # At least 2/3 candles must confirm

# =============================================================================
# V7 NEW: MINIMUM CONFIDENCE FILTER
# =============================================================================
MIN_CONFIDENCE = 0.70  # Only signals with 70%+ confidence

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
