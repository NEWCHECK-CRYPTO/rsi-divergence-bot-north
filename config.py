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

# Scan timeframes - includes all major timeframes
SCAN_TIMEFRAMES = ["1h", "4h", "1d", "1w", "1M"]

RSI_PERIOD = 14

ALERT_COOLDOWN = 1800  # 30 minutes cooldown per symbol/timeframe
SCAN_INTERVAL = 120    # Scan every 2 minutes

# Legacy settings (kept for compatibility but not used in simplified version)
LOOKBACK_CANDLES = 200
SWING_STRENGTH_BARS = 2
