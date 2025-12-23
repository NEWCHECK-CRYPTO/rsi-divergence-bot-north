import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EXCHANGE = "binance"

# =============================================================================
# TIMEZONE
# =============================================================================
TIMEZONE = "Asia/Colombo"  # Sri Lanka (UTC+5:30)

# =============================================================================
# SYMBOLS CONFIGURATION
# =============================================================================
SYMBOLS = []  # Empty = auto-fetch top coins by volume
TOP_COINS_COUNT = 50  # Monitor top 50 coins
QUOTE_CURRENCY = "USDT"

# Exclude stablecoins
EXCLUDED_SYMBOLS = [
    "USDC/USDT", "BUSD/USDT", "TUSD/USDT", "USDP/USDT", "FDUSD/USDT",
    "DAI/USDT", "USDD/USDT", "EURC/USDT", "AEUR/USDT",
]

# Exclude leveraged tokens (UP, DOWN, BULL, BEAR, etc.)
EXCLUDE_LEVERAGED = True

# =============================================================================
# TIMEFRAMES
# =============================================================================
SCAN_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]

# Confirmation timeframe mapping (signal TF -> lower TF for confirmation)
TIMEFRAME_CONFIRMATION_MAP = {
    "1M": "1d",
    "1w": "4h",
    "1d": "1h",
    "4h": "15m",
    "1h": "5m",
    "15m": "1m",
}

# =============================================================================
# DIVERGENCE DETECTION SETTINGS - ULTRA LOOSE FOR TESTING
# =============================================================================

# How many candles to look back
LOOKBACK_CANDLES = 40  # Short lookback

# Minimum candles between two swing points
MIN_SWING_DISTANCE = 3  # Very close swings allowed

# Minimum price move % to qualify as a swing
MIN_PRICE_MOVE_PCT = 0.1  # Tiny swings count (0.1%)

# How many candles on each side to confirm a swing
SWING_STRENGTH_BARS = 2  # Just 2 candles each side

# =============================================================================
# RSI SETTINGS
# =============================================================================
RSI_PERIOD = 14  # Standard RSI period (TradingView default)

# =============================================================================
# ALERT SETTINGS
# =============================================================================
# Cooldown between alerts for same symbol/timeframe (seconds)
ALERT_COOLDOWN = 1800  # 30 minutes

# Scan interval (seconds)
SCAN_INTERVAL = 180  # Every 3 minutes

# =============================================================================
# RAG / GEMINI SETTINGS
# =============================================================================
RAG_KNOWLEDGE_PATH = "rsi_divergence_ms_rag.json"
GEMINI_MODEL = "gemini-1.5-flash"


# =============================================================================
# SETTINGS EXPLANATION
# =============================================================================
"""
HOW THE NEW SETTINGS WORK:

1. LOOKBACK_CANDLES = 100
   - Searches last 100 candles for swing points
   - For 1H chart: ~4 days of data
   - For 4H chart: ~16 days of data
   - For 1D chart: ~3.5 months of data

2. MIN_SWING_DISTANCE = 15
   - Two swing points must be at least 15 candles apart
   - Prevents: [swing1]...[swing2] too close together
   - Ensures: Meaningful divergence pattern

3. MIN_PRICE_MOVE_PCT = 2.0
   - Swing must be at least 2% move from surrounding area
   - Filters out tiny wiggles that aren't real swings
   - Adjust higher (3-5%) for less signals, more accuracy

4. SWING_STRENGTH_BARS = 8
   - A swing high must be highest in 8 candles on each side (16 total)
   - A swing low must be lowest in 8 candles on each side
   - This ensures we find MAJOR swings, not small bumps

EXAMPLE:
   For a 4H chart with LOOKBACK=100, MIN_DISTANCE=15:
   - Looks at last 400 hours (~16 days)
   - Finds maybe 2-4 major swing highs
   - Finds maybe 2-4 major swing lows
   - Compares last 2 swings for divergence
   - Signal only if swings are 15+ candles (60+ hours) apart
"""
