import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EXCHANGE = "bybit"
TIMEZONE = "Asia/Colombo"

# Top 100 Coins by Market Cap (hardcoded for reliability)
# Updated list - excludes stablecoins and leveraged tokens
TOP_100_MARKET_CAP = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
    "DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "LINK/USDT",
    "SHIB/USDT", "TON/USDT", "DOT/USDT", "XLM/USDT", "BCH/USDT",
    "HBAR/USDT", "SUI/USDT", "UNI/USDT", "LTC/USDT", "PEPE/USDT",
    "NEAR/USDT", "APT/USDT", "AAVE/USDT", "ICP/USDT", "ETC/USDT",
    "CRO/USDT", "POL/USDT", "RENDER/USDT", "VET/USDT", "FET/USDT",
    "TAO/USDT", "ATOM/USDT", "OM/USDT", "FIL/USDT", "ARB/USDT",
    "STX/USDT", "ALGO/USDT", "KAS/USDT", "MNT/USDT", "OP/USDT",
    "INJ/USDT", "FTM/USDT", "GRT/USDT", "IMX/USDT", "BONK/USDT",
    "TIA/USDT", "WIF/USDT", "SEI/USDT", "THETA/USDT", "RUNE/USDT",
    "WLD/USDT", "FLOKI/USDT", "JASMY/USDT", "SAND/USDT", "LDO/USDT",
    "FLR/USDT", "GALA/USDT", "PYTH/USDT", "KAIA/USDT", "EOS/USDT",
    "JUP/USDT", "BEAM/USDT", "CORE/USDT", "FLOW/USDT", "ENS/USDT",
    "BTT/USDT", "BRETT/USDT", "QNT/USDT", "NEO/USDT", "IOTA/USDT",
    "BSV/USDT", "AIOZ/USDT", "XTZ/USDT", "AXS/USDT", "DYDX/USDT",
    "PENDLE/USDT", "STRK/USDT", "CFX/USDT", "AR/USDT", "W/USDT",
    "MANA/USDT", "AERO/USDT", "MKR/USDT", "CAKE/USDT", "APE/USDT",
    "EGLD/USDT", "ZRO/USDT", "SUPER/USDT", "XDC/USDT", "NOT/USDT",
    "CRV/USDT", "ORDI/USDT", "RON/USDT", "ZEC/USDT", "EIGEN/USDT",
    "MEME/USDT", "1INCH/USDT", "LUNC/USDT", "SNX/USDT", "CHZ/USDT"
]

# Number of coins to scan (use all 100 or subset)
TOP_COINS_COUNT = 100
QUOTE_CURRENCY = "USDT"

EXCLUDED_SYMBOLS = [
    "USDC/USDT", "BUSD/USDT", "TUSD/USDT", "USDP/USDT", "FDUSD/USDT",
    "DAI/USDT", "USDD/USDT", "EURC/USDT", "AEUR/USDT",
]

EXCLUDE_LEVERAGED = True

# Scan timeframes - 1h, 4h, 1d ONLY (as requested)
SCAN_TIMEFRAMES = ["1h", "4h", "1d"]

RSI_PERIOD = 14

# RSI Extreme Zones for valid divergence
RSI_OVERSOLD = 40     # Bullish: RSI must be below this at swing
RSI_OVERBOUGHT = 60   # Bearish: RSI must be above this at swing

# Swing Detection Settings
SWING_STRENGTH = 2    # Candles on each side to confirm swing
LOOKBACK_CANDLES = 200

# Minimum distance between swings (well-established standard: 5-25)
MIN_SWING_DISTANCE = 5   # Minimum 5 candles apart
MAX_SWING_DISTANCE = 50  # Maximum 50 candles apart

# Alert timing: 1 candle after swing confirmation for ALL timeframes
# This means we detect the swing and alert as soon as the next candle closes
CONFIRMATION_CANDLES = 1

# Recency: Maximum candles since Swing 2 for alert to be valid
# Since we want alert at exactly 1 candle after confirmation:
# Swing needs strength=2 (so confirmed at swing_idx + 2)
# Alert fires 1 candle later (swing_idx + 3 from swing point)
MAX_CANDLES_SINCE_SWING2 = {
    "1h": 3,   # Alert within 3 hours of swing
    "4h": 3,   # Alert within 12 hours of swing  
    "1d": 3,   # Alert within 3 days of swing
}

# Cooldown between alerts for same symbol/timeframe
ALERT_COOLDOWN = 3600  # 1 hour

# Scan interval - how often to check for new signals
# Aligned to candle closes for precision
SCAN_INTERVAL = 60  # Check every 1 minute for candle closes

# Max signal age to prevent stale alerts
MAX_SIGNAL_AGE = {
    "1h": 2 * 60 * 60,        # 2 hours
    "4h": 8 * 60 * 60,        # 8 hours  
    "1d": 2 * 24 * 60 * 60,   # 2 days
}
