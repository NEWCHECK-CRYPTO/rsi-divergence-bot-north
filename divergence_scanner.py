"""
RSI Divergence Bot - SIMPLIFIED VERSION (Option C)
==================================================
Simple & Proven approach:
- Detect divergence pattern using CLOSE prices
- RSI must be in extreme zone
- Alert immediately when Swing 2 forms
- No confirmation candles - let trader decide entry

WHY CLOSE PRICES?
- RSI is calculated from CLOSE prices
- More consistent comparison
- Matches most TradingView indicators
- Ignores wick noise

CONDITIONS FOR VALID SIGNAL:
1. Valid swing points detected (strength = 3 candles each side, using CLOSE)
2. Divergence pattern exists:
   - BULLISH: Price Lower Low + RSI Higher Low (CLOSE prices)
   - BEARISH: Price Higher High + RSI Lower High (CLOSE prices)
3. Swing 2 RSI in extreme zone:
   - BULLISH: RSI < 40 (oversold territory)
   - BEARISH: RSI > 60 (overbought territory)
4. Swings are 5-50 candles apart
5. Pattern not invalidated between swings (PRICE ONLY):
   - BULLISH: No CLOSE below Swing2 CLOSE
   - BEARISH: No CLOSE above Swing2 CLOSE
   - RSI between swings is NOT checked (this is normal behavior)
6. Swing 2 is recent (within last 3 candles = just formed)
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
# SIMPLIFIED CONDITIONS - EASY TO UNDERSTAND
# =============================================================================

# Swing Detection - Strength per Timeframe
# Lower timeframes (1H, 4H): Strength 2 - more noisy, need confirmation
# Higher timeframes (1D, 1W, 1M): Strength 1 - faster signals, less noise
SWING_STRENGTH_MAP = {
    "1h": 2,   # 2 candles confirmation (2 hours delay)
    "4h": 2,   # 2 candles confirmation (8 hours delay)
    "1d": 1,   # 1 candle confirmation (1 day delay)
    "1w": 1,   # 1 candle confirmation (1 week delay)
    "1M": 1,   # 1 candle confirmation (1 month delay)
}
SWING_STRENGTH = 2  # Default fallback

# Divergence Requirements
MIN_SWING_DISTANCE = 3   # Minimum candles between swings (catch fast reversals)
MAX_SWING_DISTANCE = 50  # Maximum candles between swings

# RSI Extreme Zones (KEY FILTER)
RSI_OVERSOLD = 40    # Bullish divergence: RSI must be below this
RSI_OVERBOUGHT = 60  # Bearish divergence: RSI must be above this

# Recency - Alert only when Swing 2 just formed
MAX_CANDLES_SINCE_SWING2 = 2  # Swing 2 must be within last 2 candles

# Max age in seconds - prevents stale/late signals
MAX_SIGNAL_AGE = {
    "1h": 2 * 60 * 60,           # 2 hours
    "4h": 8 * 60 * 60,           # 8 hours
    "1d": 2 * 24 * 60 * 60,      # 2 days
    "1w": 14 * 24 * 60 * 60,     # 2 weeks (14 days)
    "1M": 60 * 24 * 60 * 60,     # 2 months (~60 days)
}

# =============================================================================


class DivergenceType(Enum):
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"


class SignalStrength(Enum):
    STRONG = "strong"      # RSI very extreme (< 30 or > 70)
    MEDIUM = "medium"      # RSI moderately extreme
    EARLY = "early"        # RSI at edge of extreme zone


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
    """Calculate RSI indicator"""
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


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX for trend strength info"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm_smooth = pd.Series(plus_dm).rolling(period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(period).mean()
    
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20.0


def get_tradingview_link(symbol: str, timeframe: str) -> str:
    """Generate TradingView chart link"""
    exchange_map = {"bybit": "BYBIT", "binance": "BINANCE"}
    exchange_prefix = exchange_map.get(EXCHANGE.lower(), "BYBIT")
    clean_symbol = symbol.replace("/", "")
    
    tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M"}
    tv_interval = tf_map.get(timeframe, "240")
    
    return f"https://www.tradingview.com/chart/?symbol={exchange_prefix}:{clean_symbol}&interval={tv_interval}"


class DivergenceScanner:
    """
    Simplified Divergence Scanner
    
    WHAT IT DOES:
    1. Finds swing highs and swing lows
    2. Checks if price and RSI are diverging
    3. Confirms RSI is in extreme zone
    4. Alerts immediately when pattern forms
    """
    
    def __init__(self):
        print(f"[{format_sl_time()}] Initializing Simplified Scanner...")
        print(f"[{format_sl_time()}] CONDITIONS:")
        print(f"  - Swing Strength: {SWING_STRENGTH} candles each side")
        print(f"  - Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles")
        print(f"  - Bullish RSI Zone: < {RSI_OVERSOLD}")
        print(f"  - Bearish RSI Zone: > {RSI_OVERBOUGHT}")
        print(f"  - Max candles since Swing2: {MAX_CANDLES_SINCE_SWING2}")
        
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
        self.alert_cooldowns = {}      # Time-based cooldown (backup)
        self.sent_divergences = {}     # Track sent divergences by Swing2 timestamp
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
                print(f"[{format_sl_time()}] Ticker fetch error: {e}, using fallback...")
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
        """Fetch OHLCV data with RSI - handles large requests with pagination"""
        try:
            # Bybit typically allows max 200-1000 candles per request
            # For safety, fetch in chunks of 200 if more than 200 requested
            if limit <= 200:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                # Fetch in chunks for large requests
                all_ohlcv = []
                remaining = limit
                since = None
                
                while remaining > 0:
                    chunk_size = min(remaining, 200)
                    
                    if since is None:
                        # First fetch - get most recent
                        ohlcv_chunk = self.exchange.fetch_ohlcv(symbol, timeframe, limit=chunk_size)
                    else:
                        # Subsequent fetches - get older data
                        ohlcv_chunk = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=chunk_size)
                    
                    if not ohlcv_chunk:
                        break
                    
                    all_ohlcv = ohlcv_chunk + all_ohlcv  # Prepend older data
                    
                    # Get timestamp of oldest candle for next fetch
                    oldest_ts = ohlcv_chunk[0][0]
                    
                    # Calculate time step based on timeframe
                    tf_ms = {
                        '1m': 60000, '5m': 300000, '15m': 900000,
                        '1h': 3600000, '4h': 14400000, '1d': 86400000, '1w': 604800000
                    }
                    step = tf_ms.get(timeframe, 3600000) * chunk_size
                    since = oldest_ts - step
                    
                    remaining -= chunk_size
                    time.sleep(0.1)  # Rate limit
                    
                    # Safety break if we got less than requested (no more data)
                    if len(ohlcv_chunk) < chunk_size:
                        break
                
                ohlcv = all_ohlcv[-limit:] if len(all_ohlcv) > limit else all_ohlcv
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_swing_lows(self, df: pd.DataFrame, timeframe: str = "4h") -> List[SwingPoint]:
        """
        Find swing low points using CLOSE price
        A swing low is a candle with lower CLOSE than surrounding candles
        """
        swings = []
        strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
        
        for i in range(strength, len(df) - strength):
            is_swing_low = True
            
            # Check if this candle's CLOSE is lower than surrounding candles' CLOSE
            for j in range(1, strength + 1):
                if df['close'].iloc[i] >= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] >= df['close'].iloc[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],  # Use CLOSE for swing lows
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def find_swing_highs(self, df: pd.DataFrame, timeframe: str = "4h") -> List[SwingPoint]:
        """
        Find swing high points using CLOSE price
        A swing high is a candle with higher CLOSE than surrounding candles
        """
        swings = []
        strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
        
        for i in range(strength, len(df) - strength):
            is_swing_high = True
            
            # Check if this candle's CLOSE is higher than surrounding candles' CLOSE
            for j in range(1, strength + 1):
                if df['close'].iloc[i] <= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] <= df['close'].iloc[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],  # Use CLOSE for swing highs
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def check_pattern_validity(self, df: pd.DataFrame, swing1: SwingPoint, 
                                swing2: SwingPoint, is_bullish: bool) -> Tuple[bool, str]:
        """
        Check if pattern is not broken by middle candles
        Using CLOSE prices - Only checks PRICE invalidation (not RSI)
        
        For BULLISH:
        - No candle CLOSE should go below Swing2 CLOSE (price invalidation)
        - RSI can do whatever it wants between swings (this is normal)
        
        For BEARISH:
        - No candle CLOSE should go above Swing2 CLOSE (price invalidation)
        - RSI can do whatever it wants between swings (this is normal)
        
        WHY NO RSI INVALIDATION?
        - RSI going more extreme then recovering IS the divergence
        - Most professional indicators don't check RSI between swings
        - Only the relationship between Swing1 and Swing2 RSI matters
        """
        start_idx = swing1.index + 1
        end_idx = swing2.index
        
        if start_idx >= end_idx:
            return True, "No middle candles"
        
        middle = df.iloc[start_idx:end_idx]
        
        if is_bullish:
            # PRICE CHECK ONLY: No lower close than Swing2
            min_close = middle['close'].min()
            if min_close < swing2.price:
                return False, f"Price invalid: Close ${min_close:.2f} < Swing2 ${swing2.price:.2f}"
        
        else:  # Bearish
            # PRICE CHECK ONLY: No higher close than Swing2
            max_close = middle['close'].max()
            if max_close > swing2.price:
                return False, f"Price invalid: Close ${max_close:.2f} > Swing2 ${swing2.price:.2f}"
        
        return True, "Pattern intact"
    
    def detect_divergences(self, symbol: str, df: pd.DataFrame, timeframe: str = "4h") -> List[AlertSignal]:
        """
        Main divergence detection - SIMPLIFIED
        
        BULLISH: Price Lower Low + RSI Higher Low + RSI < 40
        BEARISH: Price Higher High + RSI Lower High + RSI > 60
        """
        signals = []
        current_idx = len(df) - 1
        
        swing_lows = self.find_swing_lows(df, timeframe)
        swing_highs = self.find_swing_highs(df, timeframe)
        
        # === BULLISH DIVERGENCES ===
        for i in range(len(swing_lows) - 1):
            swing1 = swing_lows[i]
            swing2 = swing_lows[i + 1]
            
            why_valid = []
            
            # Condition 1: Swing distance (5-50 candles)
            candles_apart = swing2.index - swing1.index
            if not (MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE):
                continue
            why_valid.append(f"Distance: {candles_apart} candles")
            
            # Condition 2: Price makes LOWER LOW
            if swing2.price >= swing1.price:
                continue
            why_valid.append(f"Price: ${swing1.price:.2f} -> ${swing2.price:.2f} (LOWER LOW)")
            
            # Condition 3: RSI makes HIGHER LOW
            if swing2.rsi <= swing1.rsi:
                continue
            why_valid.append(f"RSI: {swing1.rsi:.1f} -> {swing2.rsi:.1f} (HIGHER LOW)")
            
            # Condition 4: RSI in OVERSOLD zone
            if swing2.rsi >= RSI_OVERSOLD:
                continue
            why_valid.append(f"RSI Zone: {swing2.rsi:.1f} < {RSI_OVERSOLD} (OVERSOLD)")
            
            # Condition 5: Pattern not broken
            pattern_valid, pattern_msg = self.check_pattern_validity(df, swing1, swing2, True)
            if not pattern_valid:
                continue
            why_valid.append(f"Pattern: {pattern_msg}")
            
            # Condition 6: Recency - Swing2 just formed
            candles_since = current_idx - swing2.index
            if candles_since > MAX_CANDLES_SINCE_SWING2:
                continue
            why_valid.append(f"Recency: {candles_since} candles ago")
            
            # Condition 7: Age check - prevent stale/late signals
            current_time = datetime.now(pytz.UTC)
            swing2_time = swing2.timestamp
            if swing2_time.tzinfo is None:
                swing2_time = pytz.UTC.localize(swing2_time)
            age_seconds = (current_time - swing2_time).total_seconds()
            max_age = MAX_SIGNAL_AGE.get(timeframe, 2 * 24 * 60 * 60)  # Default 2 days
            if age_seconds > max_age:
                continue
            why_valid.append(f"Age: {age_seconds/3600:.1f}h (max {max_age/3600:.0f}h)")
            
            # ALL PASSED - Determine strength
            if swing2.rsi < 30:
                strength = SignalStrength.STRONG
            elif swing2.rsi < 35:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.EARLY
            
            signals.append(AlertSignal(
                symbol=symbol,
                timeframe="",
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
                tradingview_link="",
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
            why_valid.append(f"Price: ${swing1.price:.2f} -> ${swing2.price:.2f} (HIGHER HIGH)")
            
            # Condition 3: RSI makes LOWER HIGH
            if swing2.rsi >= swing1.rsi:
                continue
            why_valid.append(f"RSI: {swing1.rsi:.1f} -> {swing2.rsi:.1f} (LOWER HIGH)")
            
            # Condition 4: RSI in OVERBOUGHT zone
            if swing2.rsi <= RSI_OVERBOUGHT:
                continue
            why_valid.append(f"RSI Zone: {swing2.rsi:.1f} > {RSI_OVERBOUGHT} (OVERBOUGHT)")
            
            # Condition 5: Pattern not broken
            pattern_valid, pattern_msg = self.check_pattern_validity(df, swing1, swing2, False)
            if not pattern_valid:
                continue
            why_valid.append(f"Pattern: {pattern_msg}")
            
            # Condition 6: Recency
            candles_since = current_idx - swing2.index
            if candles_since > MAX_CANDLES_SINCE_SWING2:
                continue
            why_valid.append(f"Recency: {candles_since} candles ago")
            
            # Condition 7: Age check - prevent stale/late signals
            current_time = datetime.now(pytz.UTC)
            swing2_time = swing2.timestamp
            if swing2_time.tzinfo is None:
                swing2_time = pytz.UTC.localize(swing2_time)
            age_seconds = (current_time - swing2_time).total_seconds()
            max_age = MAX_SIGNAL_AGE.get(timeframe, 2 * 24 * 60 * 60)  # Default 2 days
            if age_seconds > max_age:
                continue
            why_valid.append(f"Age: {age_seconds/3600:.1f}h (max {max_age/3600:.0f}h)")
            
            # Determine strength
            if swing2.rsi > 70:
                strength = SignalStrength.STRONG
            elif swing2.rsi > 65:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.EARLY
            
            signals.append(AlertSignal(
                symbol=symbol,
                timeframe="",
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
                tradingview_link="",
                candle_close_time=df['timestamp'].iloc[-1],
                why_valid=why_valid
            ))
        
        return signals
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        """Check time-based cooldown (backup protection)"""
        key = f"{symbol}_{timeframe}"
        if key in self.alert_cooldowns:
            elapsed = time.time() - self.alert_cooldowns[key]
            return elapsed < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        """Set time-based cooldown"""
        key = f"{symbol}_{timeframe}"
        self.alert_cooldowns[key] = time.time()
    
    def _is_divergence_already_sent(self, symbol: str, timeframe: str, swing2_timestamp: datetime) -> bool:
        """Check if this specific divergence was already sent"""
        key = f"{symbol}_{timeframe}_{swing2_timestamp.strftime('%Y-%m-%d_%H:%M')}"
        return key in self.sent_divergences
    
    def _mark_divergence_sent(self, symbol: str, timeframe: str, swing2_timestamp: datetime):
        """Mark this divergence as sent so we don't send it again"""
        key = f"{symbol}_{timeframe}_{swing2_timestamp.strftime('%Y-%m-%d_%H:%M')}"
        self.sent_divergences[key] = time.time()
        
        # Clean old entries (older than 7 days) to prevent memory buildup
        current_time = time.time()
        week_ago = current_time - (7 * 24 * 60 * 60)
        self.sent_divergences = {k: v for k, v in self.sent_divergences.items() if v > week_ago}
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
        """Scan a single symbol - only returns NEW divergences"""
        if self._is_on_cooldown(symbol, timeframe):
            return []
        
        df = self.fetch_ohlcv(symbol, timeframe, limit=200)
        if df is None or len(df) < 50:
            return []
        
        signals = self.detect_divergences(symbol, df, timeframe)
        
        # Filter out already-sent divergences
        new_signals = []
        for signal in signals:
            signal.timeframe = timeframe
            signal.tradingview_link = get_tradingview_link(symbol, timeframe)
            
            # Check if this specific divergence was already sent
            if not self._is_divergence_already_sent(symbol, timeframe, signal.divergence.swing2.timestamp):
                new_signals.append(signal)
                # Mark as sent
                self._mark_divergence_sent(symbol, timeframe, signal.divergence.swing2.timestamp)
        
        if new_signals:
            self._set_cooldown(symbol, timeframe)
        
        return new_signals
    
    def scan_all(self) -> List[AlertSignal]:
        """Scan all symbols"""
        all_signals = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] Scanning {len(symbols)} symbols on {SCAN_TIMEFRAMES}...")
        
        for symbol in symbols:
            for timeframe in SCAN_TIMEFRAMES:
                try:
                    signals = self.scan_symbol(symbol, timeframe)
                    all_signals.extend(signals)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        print(f"[{format_sl_time()}] Found {len(all_signals)} signals")
        return all_signals


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
    def format_simple(alert: AlertSignal) -> str:
        """Short format"""
        div = alert.divergence
        is_bull = div.divergence_type == DivergenceType.BULLISH_REGULAR
        icon = "🟢" if is_bull else "🔴"
        direction = "BULL" if is_bull else "BEAR"
        return f"{icon} {alert.symbol} {alert.timeframe} | {direction} | RSI: {div.swing2.rsi:.1f}"
