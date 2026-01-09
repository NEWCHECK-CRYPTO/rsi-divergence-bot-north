"""
RSI Divergence Bot - UPDATED VERSION (1 CANDLE CONFIRMATION)
==============================================================
Changes from previous version:
1. 4h and 1d now use swing strength = 1 (was 2)
2. Alert timing: 1 candle after swing for ALL timeframes
3. Added verification logic to debug signal detection

Alert Timing (1 candle after swing):
- 4h: Alert ~4 hours after swing point
- 1d: Alert ~1 day after swing point  
- 1w: Alert ~1 week after swing point
- 1M: Alert ~1 month after swing point

CONDITIONS FOR VALID SIGNAL:
1. Valid swing points detected (strength=1 for all TFs)
2. Divergence pattern exists:
   - BULLISH: Price Lower Low + RSI Higher Low
   - BEARISH: Price Higher High + RSI Lower High
3. Swing 2 RSI in extreme zone
4. Swings are 3-50 candles apart
5. Pattern not invalidated between swings
6. Swing 2 is recent (within recency window)
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz
import time

from config import (
    EXCHANGE, SCAN_TIMEFRAMES, RSI_PERIOD, ALERT_COOLDOWN,
    TOP_COINS_COUNT, QUOTE_CURRENCY, EXCLUDED_SYMBOLS, 
    EXCLUDE_LEVERAGED, TIMEZONE, LOOKBACK_CANDLES,
    SWING_STRENGTH_BARS
)

SL_TZ = pytz.timezone(TIMEZONE)

# =============================================================================
# UPDATED: SWING STRENGTH = 1 FOR ALL TIMEFRAMES (1 candle confirmation)
# =============================================================================

# Swing Detection - Now 1 candle for all (faster signals)
SWING_STRENGTH_MAP = {
    "4h": 1,   # 1 candle each side - alert after 4 hours
    "1d": 1,   # 1 candle each side - alert after 1 day
    "1w": 1,   # 1 candle each side - alert after 1 week
    "1M": 1,   # 1 candle each side - alert after 1 month
}
SWING_STRENGTH = 1  # Default fallback

# Divergence Requirements
MIN_SWING_DISTANCE = 3   # Minimum candles between swings
MAX_SWING_DISTANCE = 50  # Maximum candles between swings

# RSI Extreme Zones (KEY FILTER)
RSI_OVERSOLD = 40    # Bullish: RSI must be below this
RSI_OVERBOUGHT = 60  # Bearish: RSI must be above this

# UPDATED: Recency per timeframe (adjusted for strength=1)
# With strength=1, swing is detected at index+1, so we have more window
MAX_CANDLES_SINCE_SWING2_MAP = {
    "4h": 3,   # 12 hours window (strength=1, detected at +1)
    "1d": 2,   # 2 days window (strength=1, detected at +1)
    "1w": 2,   # 2 weeks window (strength=1, detected at +1)
    "1M": 2,   # 2 months window (strength=1, detected at +1)
}
MAX_CANDLES_SINCE_SWING2 = 3  # Default fallback

# Max age in seconds - prevents stale/late signals
MAX_SIGNAL_AGE = {
    "4h": 16 * 60 * 60,          # 16 hours (4 candles)
    "1d": 2 * 24 * 60 * 60,      # 2 days
    "1w": 21 * 24 * 60 * 60,     # 3 weeks
    "1M": 60 * 24 * 60 * 60,     # 2 months
}

# =============================================================================
# MARKET REGIME & VOLATILITY (INFO ONLY - NOT A FILTER)
# =============================================================================

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX using Wilder's smoothing - matches TradingView"""
    if len(df) < period * 2:
        return 0.0
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.zeros(len(df))
    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    
    for i in range(1, len(df)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(len(df))
    smooth_plus_dm = np.zeros(len(df))
    smooth_minus_dm = np.zeros(len(df))
    
    atr[period] = np.sum(tr[1:period+1])
    smooth_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smooth_minus_dm[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period + 1, len(df)):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        smooth_plus_dm[i] = smooth_plus_dm[i-1] - (smooth_plus_dm[i-1] / period) + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i-1] - (smooth_minus_dm[i-1] / period) + minus_dm[i]
    
    plus_di = np.zeros(len(df))
    minus_di = np.zeros(len(df))
    dx = np.zeros(len(df))
    
    for i in range(period, len(df)):
        if atr[i] != 0:
            plus_di[i] = 100 * smooth_plus_dm[i] / atr[i]
            minus_di[i] = 100 * smooth_minus_dm[i] / atr[i]
        
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(len(df))
    first_adx_idx = period * 2
    if first_adx_idx < len(df):
        adx[first_adx_idx] = np.mean(dx[period+1:first_adx_idx+1])
        
        for i in range(first_adx_idx + 1, len(df)):
            adx[i] = ((adx[i-1] * (period - 1)) + dx[i]) / period
    
    return float(adx[-1]) if adx[-1] != 0 else 0.0


def calculate_atr(df: pd.DataFrame, period: int = 14) -> Tuple[float, float]:
    """Calculate ATR and average ATR for volatility detection"""
    if len(df) < period + 1:
        return 0.0, 0.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    avg_atr = float(atr.tail(50).mean()) if len(atr) >= 50 else current_atr
    
    return current_atr, avg_atr


def calculate_bollinger_bandwidth(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate Bollinger Band width for ranging detection"""
    if len(df) < period:
        return 0.0
    
    close = df['close']
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    
    bandwidth = ((upper - lower) / sma) * 100
    
    return float(bandwidth.iloc[-1]) if not pd.isna(bandwidth.iloc[-1]) else 0.0


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str
    regime_emoji: str
    regime_description: str
    adx_value: float
    divergence_rating: str
    confidence_adjustment: int


@dataclass 
class VolatilityStatus:
    """Volatility classification"""
    volatility_type: str
    volatility_emoji: str
    current_atr: float
    average_atr: float
    atr_ratio: float
    position_advice: str


def get_market_regime(df: pd.DataFrame) -> MarketRegime:
    """Classify market regime based on ADX and Bollinger Bandwidth"""
    adx = calculate_adx(df)
    bb_width = calculate_bollinger_bandwidth(df)
    
    if adx >= 40:
        return MarketRegime(
            regime_type="strong_trend",
            regime_emoji="🔥",
            regime_description="Strong Trend - SMC preferred",
            adx_value=adx,
            divergence_rating="RISKY",
            confidence_adjustment=-15
        )
    elif adx >= 25:
        return MarketRegime(
            regime_type="trending",
            regime_emoji="📈",
            regime_description="Trending - Use caution",
            adx_value=adx,
            divergence_rating="CAUTION",
            confidence_adjustment=-5
        )
    elif adx <= 20 and bb_width < 5:
        return MarketRegime(
            regime_type="ranging",
            regime_emoji="✅",
            regime_description="Ranging - IDEAL for divergence",
            adx_value=adx,
            divergence_rating="IDEAL",
            confidence_adjustment=10
        )
    else:
        return MarketRegime(
            regime_type="choppy",
            regime_emoji="⚠️",
            regime_description="Choppy - Mixed signals",
            adx_value=adx,
            divergence_rating="GOOD",
            confidence_adjustment=0
        )


def get_volatility_status(df: pd.DataFrame) -> VolatilityStatus:
    """Classify volatility based on ATR"""
    current_atr, avg_atr = calculate_atr(df)
    
    if avg_atr == 0:
        ratio = 1.0
    else:
        ratio = current_atr / avg_atr
    
    if ratio > 1.5:
        return VolatilityStatus(
            volatility_type="high",
            volatility_emoji="🌋",
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=ratio,
            position_advice="Reduce position size 50%"
        )
    elif ratio < 0.7:
        return VolatilityStatus(
            volatility_type="low",
            volatility_emoji="😴",
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=ratio,
            position_advice="Breakout may come soon"
        )
    else:
        return VolatilityStatus(
            volatility_type="normal",
            volatility_emoji="📊",
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=ratio,
            position_advice="Normal position size"
        )


# =============================================================================
# DATA CLASSES
# =============================================================================

class DivergenceType(Enum):
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"


class SignalStrength(Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    EARLY = "early"


@dataclass
class SwingPoint:
    index: int
    price: float
    rsi: float
    timestamp: datetime


@dataclass
class Divergence:
    divergence_type: DivergenceType
    swing1: SwingPoint
    swing2: SwingPoint
    current_price: float
    current_rsi: float
    candles_apart: int


@dataclass
class AlertSignal:
    symbol: str
    timeframe: str
    divergence: Divergence
    signal_strength: SignalStrength
    timestamp: datetime
    volume_rank: int
    tradingview_link: str
    candle_close_time: datetime
    why_valid: List[str]


def get_sl_time():
    """Get current Sri Lanka time"""
    return datetime.now(SL_TZ)


def format_sl_time(dt=None):
    """Format datetime in Sri Lanka time"""
    if dt is None:
        dt = get_sl_time()
    elif dt.tzinfo is None:
        dt = SL_TZ.localize(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S IST')


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing"""
    delta = close_prices.diff()
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    rsi = pd.Series(index=close_prices.index, dtype=float)
    
    if len(close_prices) < period + 1:
        return rsi
    
    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()
    
    if avg_loss == 0:
        rsi.iloc[period] = 100
    else:
        rsi.iloc[period] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    for i in range(period + 1, len(close_prices)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        
        if avg_loss == 0:
            rsi.iloc[i] = 100
        else:
            rsi.iloc[i] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    return rsi


def get_tradingview_link(symbol: str, timeframe: str) -> str:
    """Generate TradingView chart link"""
    exchange_map = {"bybit": "BYBIT", "binance": "BINANCE"}
    exchange_prefix = exchange_map.get(EXCHANGE.lower(), "BYBIT")
    clean_symbol = symbol.replace("/", "")
    
    tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M"}
    tv_interval = tf_map.get(timeframe, "240")
    
    return f"https://www.tradingview.com/chart/?symbol={exchange_prefix}:{clean_symbol}&interval={tv_interval}"


class DivergenceScanner:
    """Simplified Divergence Scanner - UPDATED WITH 1 CANDLE CONFIRMATION"""
    
    def __init__(self):
        print(f"[{format_sl_time()}] Initializing UPDATED Scanner (1 candle confirmation)...")
        print(f"[{format_sl_time()}] CONDITIONS:")
        print(f"  - Swing Strength: {SWING_STRENGTH_MAP}")
        print(f"  - Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles")
        print(f"  - Bullish RSI Zone: < {RSI_OVERSOLD}")
        print(f"  - Bearish RSI Zone: > {RSI_OVERBOUGHT}")
        print(f"  - Recency per TF: {MAX_CANDLES_SINCE_SWING2_MAP}")
        
        if EXCHANGE.lower() == "bybit":
            self.exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        elif EXCHANGE.lower() == "binance":
            self.exchange = ccxt.binance({'enableRateLimit': True})
        else:
            raise ValueError(f"Unsupported exchange: {EXCHANGE}")
        
        self.exchange.load_markets()
        self.alert_cooldowns = {}
        self.sent_divergences = {}
        self.volume_ranks = {}
        
        print(f"[{format_sl_time()}] Scanner ready!")
    
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        """Fetch top coins by 24h volume"""
        try:
            print(f"[{format_sl_time()}] Fetching top {count} coins...")
            
            valid_symbols = []
            for symbol, market in self.exchange.markets.items():
                if (market.get('quote') == 'USDT' and 
                    market.get('spot', False) and 
                    market.get('active', True) and
                    symbol not in EXCLUDED_SYMBOLS):
                    
                    base = market.get('base', '')
                    if EXCLUDE_LEVERAGED:
                        if any(x in base for x in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S', '5L', '5S']):
                            continue
                    if base in ['USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDD', 'USDP']:
                        continue
                    
                    valid_symbols.append(symbol)
            
            volume_data = {}
            try:
                if EXCHANGE.lower() == "bybit":
                    tickers = self.exchange.fetch_tickers(params={'category': 'spot'})
                else:
                    tickers = self.exchange.fetch_tickers()
                
                for symbol, ticker in tickers.items():
                    if symbol in valid_symbols:
                        vol = ticker.get('quoteVolume', 0)
                        if vol:
                            volume_data[symbol] = float(vol)
            except Exception as e:
                print(f"[{format_sl_time()}] Ticker fetch error: {e}")
                return self._get_fallback_symbols()
            
            sorted_pairs = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, _ in sorted_pairs[:count]]
            
            self.volume_ranks = {sym: rank for rank, sym in enumerate(top_symbols, 1)}
            
            print(f"[{format_sl_time()}] Got {len(top_symbols)} coins")
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] Error: {e}")
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback list of major coins"""
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
            'SHIB/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT', 'ATOM/USDT',
            'XLM/USDT', 'ETC/USDT', 'FIL/USDT', 'NEAR/USDT', 'APT/USDT',
            'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT', 'SEI/USDT',
            'TIA/USDT', 'PEPE/USDT', 'WIF/USDT', 'BONK/USDT', 'FLOKI/USDT'
        ]
        self.volume_ranks = {sym: rank for rank, sym in enumerate(symbols, 1)}
        return symbols
    
    def get_symbols_to_scan(self) -> List[str]:
        """Get symbols to scan"""
        symbols = self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
        if not symbols:
            symbols = self._get_fallback_symbols()
        return symbols
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with RSI"""
        try:
            if limit <= 200:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                all_ohlcv = []
                remaining = limit
                since = None
                
                while remaining > 0:
                    chunk_size = min(remaining, 200)
                    
                    if since is None:
                        ohlcv_chunk = self.exchange.fetch_ohlcv(symbol, timeframe, limit=chunk_size)
                    else:
                        ohlcv_chunk = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=chunk_size)
                    
                    if not ohlcv_chunk:
                        break
                    
                    all_ohlcv = ohlcv_chunk + all_ohlcv
                    oldest_ts = ohlcv_chunk[0][0]
                    
                    tf_ms = {
                        '1m': 60000, '5m': 300000, '15m': 900000,
                        '1h': 3600000, '4h': 14400000, '1d': 86400000, '1w': 604800000
                    }
                    step = tf_ms.get(timeframe, 3600000) * chunk_size
                    since = oldest_ts - step
                    remaining -= chunk_size
                    
                    if len(ohlcv_chunk) < chunk_size:
                        break
                
                ohlcv = all_ohlcv
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_swing_lows(self, df: pd.DataFrame, timeframe: str = "4h") -> List[SwingPoint]:
        """Find swing low points using CLOSE price"""
        swings = []
        strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
        
        for i in range(strength, len(df) - strength):
            is_swing_low = True
            
            for j in range(1, strength + 1):
                if df['close'].iloc[i] >= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] >= df['close'].iloc[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def find_swing_highs(self, df: pd.DataFrame, timeframe: str = "4h") -> List[SwingPoint]:
        """Find swing high points using CLOSE price"""
        swings = []
        strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
        
        for i in range(strength, len(df) - strength):
            is_swing_high = True
            
            for j in range(1, strength + 1):
                if df['close'].iloc[i] <= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] <= df['close'].iloc[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def check_pattern_validity(self, df: pd.DataFrame, swing1: SwingPoint, 
                                swing2: SwingPoint, is_bullish: bool) -> Tuple[bool, str]:
        """Check if pattern is not broken by middle candles"""
        start_idx = swing1.index + 1
        end_idx = swing2.index
        
        if start_idx >= end_idx:
            return True, "No middle candles"
        
        middle = df.iloc[start_idx:end_idx]
        
        if is_bullish:
            min_close = middle['close'].min()
            if min_close < swing2.price:
                return False, f"Price invalid: ${min_close:.2f} < ${swing2.price:.2f}"
        else:
            max_close = middle['close'].max()
            if max_close > swing2.price:
                return False, f"Price invalid: ${max_close:.2f} > ${swing2.price:.2f}"
        
        return True, "Pattern intact"
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        """Check if symbol/timeframe is on cooldown"""
        key = f"{symbol}_{timeframe}"
        if key in self.alert_cooldowns:
            elapsed = time.time() - self.alert_cooldowns[key]
            return elapsed < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        """Set cooldown for symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        self.alert_cooldowns[key] = time.time()
    
    def _is_duplicate(self, symbol: str, timeframe: str, swing2_ts: datetime) -> bool:
        """Check if we already sent this divergence"""
        key = f"{symbol}_{timeframe}_{swing2_ts.isoformat()}"
        return key in self.sent_divergences
    
    def _mark_sent(self, symbol: str, timeframe: str, swing2_ts: datetime):
        """Mark divergence as sent"""
        key = f"{symbol}_{timeframe}_{swing2_ts.isoformat()}"
        self.sent_divergences[key] = time.time()
        
        # Clean old entries (older than 24 hours)
        cutoff = time.time() - 86400
        self.sent_divergences = {k: v for k, v in self.sent_divergences.items() if v > cutoff}
    
    def verify_divergence_detection(self, symbol: str, df: pd.DataFrame, timeframe: str = "4h") -> Dict:
        """
        VERIFICATION FUNCTION: Debug divergence detection step by step
        Returns detailed info about what's found and why signals pass/fail
        """
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_candles": len(df),
            "swing_strength": SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH),
            "max_recency": MAX_CANDLES_SINCE_SWING2_MAP.get(timeframe, MAX_CANDLES_SINCE_SWING2),
            "swing_lows": [],
            "swing_highs": [],
            "potential_bullish": [],
            "potential_bearish": [],
            "valid_signals": [],
            "rejected_reasons": []
        }
        
        current_idx = len(df) - 1
        max_recency = result["max_recency"]
        
        # Find swings
        swing_lows = self.find_swing_lows(df, timeframe)
        swing_highs = self.find_swing_highs(df, timeframe)
        
        result["swing_lows"] = [{"idx": s.index, "price": s.price, "rsi": s.rsi, 
                                  "ts": s.timestamp.strftime('%Y-%m-%d %H:%M'),
                                  "candles_from_now": current_idx - s.index} for s in swing_lows[-10:]]
        result["swing_highs"] = [{"idx": s.index, "price": s.price, "rsi": s.rsi,
                                   "ts": s.timestamp.strftime('%Y-%m-%d %H:%M'),
                                   "candles_from_now": current_idx - s.index} for s in swing_highs[-10:]]
        
        # Check bullish divergences
        for i in range(len(swing_lows) - 1):
            swing1 = swing_lows[i]
            swing2 = swing_lows[i + 1]
            
            checks = {
                "swing1_idx": swing1.index,
                "swing2_idx": swing2.index,
                "swing1_price": swing1.price,
                "swing2_price": swing2.price,
                "swing1_rsi": swing1.rsi,
                "swing2_rsi": swing2.rsi,
                "candles_apart": swing2.index - swing1.index,
                "candles_since_swing2": current_idx - swing2.index,
                "checks": {}
            }
            
            # Check 1: Distance
            candles_apart = swing2.index - swing1.index
            dist_ok = MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE
            checks["checks"]["distance"] = {"value": candles_apart, "required": f"{MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE}", "pass": dist_ok}
            
            # Check 2: Price Lower Low
            price_ok = swing2.price < swing1.price
            checks["checks"]["price_ll"] = {"swing1": swing1.price, "swing2": swing2.price, "pass": price_ok}
            
            # Check 3: RSI Higher Low
            rsi_ok = swing2.rsi > swing1.rsi
            checks["checks"]["rsi_hl"] = {"swing1": swing1.rsi, "swing2": swing2.rsi, "pass": rsi_ok}
            
            # Check 4: RSI in oversold zone
            zone_ok = swing2.rsi < RSI_OVERSOLD
            checks["checks"]["rsi_zone"] = {"value": swing2.rsi, "required": f"< {RSI_OVERSOLD}", "pass": zone_ok}
            
            # Check 5: Recency
            candles_since = current_idx - swing2.index
            recency_ok = candles_since <= max_recency
            checks["checks"]["recency"] = {"value": candles_since, "required": f"<= {max_recency}", "pass": recency_ok}
            
            # Check 6: Pattern validity
            pattern_ok, pattern_msg = self.check_pattern_validity(df, swing1, swing2, True)
            checks["checks"]["pattern"] = {"message": pattern_msg, "pass": pattern_ok}
            
            all_pass = dist_ok and price_ok and rsi_ok and zone_ok and recency_ok and pattern_ok
            checks["all_pass"] = all_pass
            
            if price_ok and rsi_ok:  # Basic divergence exists
                result["potential_bullish"].append(checks)
                
                if all_pass:
                    result["valid_signals"].append({
                        "type": "BULLISH",
                        "swing1": {"idx": swing1.index, "price": swing1.price, "rsi": swing1.rsi},
                        "swing2": {"idx": swing2.index, "price": swing2.price, "rsi": swing2.rsi},
                        "candles_since": candles_since
                    })
                else:
                    failed = [k for k, v in checks["checks"].items() if not v["pass"]]
                    result["rejected_reasons"].append({
                        "type": "BULLISH",
                        "swing2_idx": swing2.index,
                        "failed_checks": failed
                    })
        
        # Check bearish divergences
        for i in range(len(swing_highs) - 1):
            swing1 = swing_highs[i]
            swing2 = swing_highs[i + 1]
            
            checks = {
                "swing1_idx": swing1.index,
                "swing2_idx": swing2.index,
                "swing1_price": swing1.price,
                "swing2_price": swing2.price,
                "swing1_rsi": swing1.rsi,
                "swing2_rsi": swing2.rsi,
                "candles_apart": swing2.index - swing1.index,
                "candles_since_swing2": current_idx - swing2.index,
                "checks": {}
            }
            
            # Check 1: Distance
            candles_apart = swing2.index - swing1.index
            dist_ok = MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE
            checks["checks"]["distance"] = {"value": candles_apart, "required": f"{MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE}", "pass": dist_ok}
            
            # Check 2: Price Higher High
            price_ok = swing2.price > swing1.price
            checks["checks"]["price_hh"] = {"swing1": swing1.price, "swing2": swing2.price, "pass": price_ok}
            
            # Check 3: RSI Lower High
            rsi_ok = swing2.rsi < swing1.rsi
            checks["checks"]["rsi_lh"] = {"swing1": swing1.rsi, "swing2": swing2.rsi, "pass": rsi_ok}
            
            # Check 4: RSI in overbought zone
            zone_ok = swing2.rsi > RSI_OVERBOUGHT
            checks["checks"]["rsi_zone"] = {"value": swing2.rsi, "required": f"> {RSI_OVERBOUGHT}", "pass": zone_ok}
            
            # Check 5: Recency
            candles_since = current_idx - swing2.index
            recency_ok = candles_since <= max_recency
            checks["checks"]["recency"] = {"value": candles_since, "required": f"<= {max_recency}", "pass": recency_ok}
            
            # Check 6: Pattern validity
            pattern_ok, pattern_msg = self.check_pattern_validity(df, swing1, swing2, False)
            checks["checks"]["pattern"] = {"message": pattern_msg, "pass": pattern_ok}
            
            all_pass = dist_ok and price_ok and rsi_ok and zone_ok and recency_ok and pattern_ok
            checks["all_pass"] = all_pass
            
            if price_ok and rsi_ok:  # Basic divergence exists
                result["potential_bearish"].append(checks)
                
                if all_pass:
                    result["valid_signals"].append({
                        "type": "BEARISH",
                        "swing1": {"idx": swing1.index, "price": swing1.price, "rsi": swing1.rsi},
                        "swing2": {"idx": swing2.index, "price": swing2.price, "rsi": swing2.rsi},
                        "candles_since": candles_since
                    })
                else:
                    failed = [k for k, v in checks["checks"].items() if not v["pass"]]
                    result["rejected_reasons"].append({
                        "type": "BEARISH",
                        "swing2_idx": swing2.index,
                        "failed_checks": failed
                    })
        
        return result
    
    def detect_divergences(self, symbol: str, df: pd.DataFrame, timeframe: str = "4h") -> List[AlertSignal]:
        """Main divergence detection - UPDATED WITH 1 CANDLE CONFIRMATION"""
        signals = []
        current_idx = len(df) - 1
        
        # Get timeframe-specific recency limit
        max_recency = MAX_CANDLES_SINCE_SWING2_MAP.get(timeframe, MAX_CANDLES_SINCE_SWING2)
        
        swing_lows = self.find_swing_lows(df, timeframe)
        swing_highs = self.find_swing_highs(df, timeframe)
        
        # === BULLISH DIVERGENCES ===
        for i in range(len(swing_lows) - 1):
            swing1 = swing_lows[i]
            swing2 = swing_lows[i + 1]
            
            why_valid = []
            
            # Condition 1: Swing distance
            candles_apart = swing2.index - swing1.index
            if not (MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE):
                continue
            why_valid.append(f"Distance: {candles_apart} candles")
            
            # Condition 2: Price makes LOWER LOW
            if swing2.price >= swing1.price:
                continue
            why_valid.append(f"Price: ${swing1.price:.4f} → ${swing2.price:.4f} (LL)")
            
            # Condition 3: RSI makes HIGHER LOW
            if swing2.rsi <= swing1.rsi:
                continue
            why_valid.append(f"RSI: {swing1.rsi:.1f} → {swing2.rsi:.1f} (HL)")
            
            # Condition 4: RSI in OVERSOLD zone
            if swing2.rsi >= RSI_OVERSOLD:
                continue
            why_valid.append(f"RSI Zone: {swing2.rsi:.1f} < {RSI_OVERSOLD}")
            
            # Condition 5: Pattern not broken
            pattern_valid, pattern_msg = self.check_pattern_validity(df, swing1, swing2, True)
            if not pattern_valid:
                continue
            why_valid.append(f"Pattern: Valid")
            
            # Condition 6: Recency - per timeframe
            candles_since = current_idx - swing2.index
            if candles_since > max_recency:
                continue
            why_valid.append(f"Recency: {candles_since}/{max_recency} candles")
            
            # Condition 7: Age check
            current_time = datetime.now(pytz.UTC)
            swing2_time = swing2.timestamp
            if swing2_time.tzinfo is None:
                swing2_time = pytz.UTC.localize(swing2_time)
            age_seconds = (current_time - swing2_time).total_seconds()
            max_age = MAX_SIGNAL_AGE.get(timeframe, 2 * 24 * 60 * 60)
            if age_seconds > max_age:
                continue
            why_valid.append(f"Age: {age_seconds/3600:.1f}h")
            
            # Check duplicate
            if self._is_duplicate(symbol, timeframe, swing2.timestamp):
                continue
            
            # Determine strength
            if swing2.rsi < 30:
                strength = SignalStrength.STRONG
            elif swing2.rsi < 35:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.EARLY
            
            signals.append(AlertSignal(
                symbol=symbol,
                timeframe=timeframe,
                divergence=Divergence(
                    divergence_type=DivergenceType.BULLISH_REGULAR,
                    swing1=swing1,
                    swing2=swing2,
                    current_price=df['close'].iloc[-1],
                    current_rsi=df['rsi'].iloc[-1],
                    candles_apart=candles_apart
                ),
                signal_strength=strength,
                timestamp=get_sl_time(),
                volume_rank=self.volume_ranks.get(symbol, 999),
                tradingview_link=get_tradingview_link(symbol, timeframe),
                candle_close_time=df['timestamp'].iloc[-1],
                why_valid=why_valid
            ))
        
        # === BEARISH DIVERGENCES ===
        for i in range(len(swing_highs) - 1):
            swing1 = swing_highs[i]
            swing2 = swing_highs[i + 1]
            
            why_valid = []
            
            # Condition 1: Swing distance
            candles_apart = swing2.index - swing1.index
            if not (MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE):
                continue
            why_valid.append(f"Distance: {candles_apart} candles")
            
            # Condition 2: Price makes HIGHER HIGH
            if swing2.price <= swing1.price:
                continue
            why_valid.append(f"Price: ${swing1.price:.4f} → ${swing2.price:.4f} (HH)")
            
            # Condition 3: RSI makes LOWER HIGH
            if swing2.rsi >= swing1.rsi:
                continue
            why_valid.append(f"RSI: {swing1.rsi:.1f} → {swing2.rsi:.1f} (LH)")
            
            # Condition 4: RSI in OVERBOUGHT zone
            if swing2.rsi <= RSI_OVERBOUGHT:
                continue
            why_valid.append(f"RSI Zone: {swing2.rsi:.1f} > {RSI_OVERBOUGHT}")
            
            # Condition 5: Pattern not broken
            pattern_valid, pattern_msg = self.check_pattern_validity(df, swing1, swing2, False)
            if not pattern_valid:
                continue
            why_valid.append(f"Pattern: Valid")
            
            # Condition 6: Recency
            candles_since = current_idx - swing2.index
            if candles_since > max_recency:
                continue
            why_valid.append(f"Recency: {candles_since}/{max_recency} candles")
            
            # Condition 7: Age check
            current_time = datetime.now(pytz.UTC)
            swing2_time = swing2.timestamp
            if swing2_time.tzinfo is None:
                swing2_time = pytz.UTC.localize(swing2_time)
            age_seconds = (current_time - swing2_time).total_seconds()
            max_age = MAX_SIGNAL_AGE.get(timeframe, 2 * 24 * 60 * 60)
            if age_seconds > max_age:
                continue
            why_valid.append(f"Age: {age_seconds/3600:.1f}h")
            
            # Check duplicate
            if self._is_duplicate(symbol, timeframe, swing2.timestamp):
                continue
            
            # Determine strength
            if swing2.rsi > 70:
                strength = SignalStrength.STRONG
            elif swing2.rsi > 65:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.EARLY
            
            signals.append(AlertSignal(
                symbol=symbol,
                timeframe=timeframe,
                divergence=Divergence(
                    divergence_type=DivergenceType.BEARISH_REGULAR,
                    swing1=swing1,
                    swing2=swing2,
                    current_price=df['close'].iloc[-1],
                    current_rsi=df['rsi'].iloc[-1],
                    candles_apart=candles_apart
                ),
                signal_strength=strength,
                timestamp=get_sl_time(),
                volume_rank=self.volume_ranks.get(symbol, 999),
                tradingview_link=get_tradingview_link(symbol, timeframe),
                candle_close_time=df['timestamp'].iloc[-1],
                why_valid=why_valid
            ))
        
        return signals
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
        """Scan a single symbol for divergences"""
        if self._is_on_cooldown(symbol, timeframe):
            return []
        
        df = self.fetch_ohlcv(symbol, timeframe, limit=LOOKBACK_CANDLES)
        if df is None or len(df) < 50:
            return []
        
        signals = self.detect_divergences(symbol, df, timeframe)
        
        # Mark sent and set cooldown for valid signals
        for signal in signals:
            self._mark_sent(symbol, timeframe, signal.divergence.swing2.timestamp)
            self._set_cooldown(symbol, timeframe)
            print(f"  🎯 SIGNAL: {symbol} {timeframe} {signal.divergence.divergence_type.value}")
        
        return signals
    
    def scan_all(self) -> List[AlertSignal]:
        """Scan all symbols across all timeframes"""
        all_signals = []
        symbols = self.get_symbols_to_scan()
        
        print(f"\n[{format_sl_time()}] 🔍 Scanning {len(symbols)} symbols across {SCAN_TIMEFRAMES}...")
        
        for symbol in symbols:
            for timeframe in SCAN_TIMEFRAMES:
                try:
                    signals = self.scan_symbol(symbol, timeframe)
                    all_signals.extend(signals)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        print(f"[{format_sl_time()}] ✅ Scan complete. Found {len(all_signals)} signals.")
        return all_signals
    
    def get_market_info(self, symbol: str, timeframe: str) -> Tuple[Optional[MarketRegime], Optional[VolatilityStatus]]:
        """Get market regime and volatility info"""
        df = self.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if df is None or len(df) < 50:
            return None, None
        
        regime = get_market_regime(df)
        volatility = get_volatility_status(df)
        
        return regime, volatility


class AlertFormatter:
    """Format alerts for Telegram"""
    
    @staticmethod
    def format_alert(alert: AlertSignal) -> str:
        """Format alert message"""
        div = alert.divergence
        is_bull = div.divergence_type == DivergenceType.BULLISH_REGULAR
        
        if alert.signal_strength == SignalStrength.STRONG:
            strength_icon = "🟢 STRONG"
        elif alert.signal_strength == SignalStrength.MEDIUM:
            strength_icon = "🟡 MEDIUM"
        else:
            strength_icon = "🔵 EARLY"
        
        direction = "🟢 BULLISH (Long)" if is_bull else "🔴 BEARISH (Short)"
        
        def fmt(p):
            if p >= 1000:
                return f"${p:,.2f}"
            elif p >= 1:
                return f"${p:.4f}"
            else:
                return f"${p:.6f}"
        
        conditions = "\n".join([f"  ✅ {c}" for c in alert.why_valid])
        
        if is_bull:
            entry = div.current_price
            sl = div.swing2.price * 0.99
            tp1 = entry * 1.02
            tp2 = entry * 1.05
        else:
            entry = div.current_price
            sl = div.swing2.price * 1.01
            tp1 = entry * 0.98
            tp2 = entry * 0.95
        
        msg = f"""{strength_icon} SIGNAL
{direction}

📊 {alert.symbol} | {alert.timeframe.upper()}
⏰ {format_sl_time(alert.candle_close_time)}

━━━━━━━━━━━━━━━━━━━━━━
📈 DIVERGENCE DETECTED
━━━━━━━━━━━━━━━━━━━━━━

Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
         {div.swing1.timestamp.strftime('%m-%d %H:%M')}

Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
         {div.swing2.timestamp.strftime('%m-%d %H:%M')}

Now:     {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})

━━━━━━━━━━━━━━━━━━━━━━
✅ WHY THIS IS VALID
━━━━━━━━━━━━━━━━━━━━━━
{conditions}

━━━━━━━━━━━━━━━━━━━━━━
🎯 TRADE IDEA
━━━━━━━━━━━━━━━━━━━━━━
Entry: {fmt(entry)} (current)
SL: {fmt(sl)}
TP1: {fmt(tp1)} (+2%)
TP2: {fmt(tp2)} (+5%)

📺 {alert.tradingview_link}

⚠️ DYOR - Not financial advice
🕐 {format_sl_time()}"""
        
        return msg
    
    @staticmethod
    def format_alert_with_regime(alert: AlertSignal, regime: MarketRegime, volatility: VolatilityStatus) -> str:
        """Format alert with market regime info"""
        div = alert.divergence
        is_bull = div.divergence_type == DivergenceType.BULLISH_REGULAR
        
        if alert.signal_strength == SignalStrength.STRONG:
            strength_icon = "🟢 STRONG"
        elif alert.signal_strength == SignalStrength.MEDIUM:
            strength_icon = "🟡 MEDIUM"
        else:
            strength_icon = "🔵 EARLY"
        
        direction = "🟢 BULLISH (Long)" if is_bull else "🔴 BEARISH (Short)"
        
        def fmt(p):
            if p >= 1000:
                return f"${p:,.2f}"
            elif p >= 1:
                return f"${p:.4f}"
            else:
                return f"${p:.6f}"
        
        conditions = "\n".join([f"  ✅ {c}" for c in alert.why_valid])
        
        if is_bull:
            entry = div.current_price
            sl = div.swing2.price * 0.99
            tp1 = entry * 1.02
            tp2 = entry * 1.05
        else:
            entry = div.current_price
            sl = div.swing2.price * 1.01
            tp1 = entry * 0.98
            tp2 = entry * 0.95
        
        msg = f"""{strength_icon} SIGNAL
{direction}

📊 {alert.symbol} | {alert.timeframe.upper()}
⏰ {format_sl_time(alert.candle_close_time)}

━━━━━━━━━━━━━━━━━━━━━━
📈 DIVERGENCE DETECTED
━━━━━━━━━━━━━━━━━━━━━━

Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
         {div.swing1.timestamp.strftime('%m-%d %H:%M')}

Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
         {div.swing2.timestamp.strftime('%m-%d %H:%M')}

Now:     {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})

━━━━━━━━━━━━━━━━━━━━━━
✅ WHY THIS IS VALID
━━━━━━━━━━━━━━━━━━━━━━
{conditions}

━━━━━━━━━━━━━━━━━━━━━━
📊 MARKET CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━
{regime.regime_emoji} Regime: {regime.regime_description}
   ADX: {regime.adx_value:.1f} | Rating: {regime.divergence_rating}

{volatility.volatility_emoji} Volatility: {volatility.volatility_type.upper()}
   ATR Ratio: {volatility.atr_ratio:.2f}x
   💡 {volatility.position_advice}

━━━━━━━━━━━━━━━━━━━━━━
🎯 TRADE IDEA
━━━━━━━━━━━━━━━━━━━━━━
Entry: {fmt(entry)} (current)
SL: {fmt(sl)}
TP1: {fmt(tp1)} (+2%)
TP2: {fmt(tp2)} (+5%)

📺 {alert.tradingview_link}

⚠️ DYOR - Not financial advice
🕐 {format_sl_time()}"""
        
        return msg
    
    @staticmethod
    def format_regime_info(symbol: str, timeframe: str, regime: MarketRegime, 
                           volatility: VolatilityStatus, current_price: float, 
                           current_rsi: float) -> str:
        """Format market regime info"""
        def fmt(p):
            if p >= 1000:
                return f"${p:,.2f}"
            elif p >= 1:
                return f"${p:.4f}"
            else:
                return f"${p:.6f}"
        
        msg = f"""*📊 Market Regime: {symbol} {timeframe.upper()}*

*Current:*
Price: {fmt(current_price)}
RSI: {current_rsi:.1f}

━━━━━━━━━━━━━━━━━━━━━━
*📈 TREND ANALYSIS*
━━━━━━━━━━━━━━━━━━━━━━
{regime.regime_emoji} *{regime.regime_description}*
ADX: {regime.adx_value:.1f}
Divergence Rating: *{regime.divergence_rating}*

━━━━━━━━━━━━━━━━━━━━━━
*📊 VOLATILITY*
━━━━━━━━━━━━━━━━━━━━━━
{volatility.volatility_emoji} *{volatility.volatility_type.upper()}*
ATR Ratio: {volatility.atr_ratio:.2f}x average
💡 {volatility.position_advice}

━━━━━━━━━━━━━━━━━━━━━━
*🎯 RECOMMENDATION*
━━━━━━━━━━━━━━━━━━━━━━"""
        
        if regime.divergence_rating == "IDEAL":
            msg += "\n✅ *IDEAL* conditions for divergence trading"
        elif regime.divergence_rating == "GOOD":
            msg += "\n🟡 *GOOD* conditions - proceed with caution"
        elif regime.divergence_rating == "CAUTION":
            msg += "\n⚠️ *CAUTION* - trending market, divergences risky"
        else:
            msg += "\n🔴 *RISKY* - strong trend, consider SMC/trend following"
        
        msg += f"\n\n🕐 {format_sl_time()}"
        
        return msg
    
    @staticmethod
    def format_simple(alert: AlertSignal) -> str:
        """Short format"""
        div = alert.divergence
        is_bull = div.divergence_type == DivergenceType.BULLISH_REGULAR
        icon = "🟢" if is_bull else "🔴"
        direction = "BULL" if is_bull else "BEAR"
        return f"{icon} {alert.symbol} {alert.timeframe} | {direction} | RSI: {div.swing2.rsi:.1f}"
