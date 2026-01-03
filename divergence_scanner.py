"""
RSI Divergence Bot - CANDLE-CLOSE ALIGNED VERSION
==================================================
Features:
1. Top 100 coins by market cap (hardcoded list)
2. Timeframes: 1h, 4h, 1d only
3. Alert fires 1 candle after swing confirmation
4. Candle-close aligned scanning for precise timing
5. Minimum 5 candles between swings (industry standard)

DIVERGENCE CONDITIONS:
- BULLISH: Price Lower Low + RSI Higher Low (RSI < 40)
- BEARISH: Price Higher High + RSI Lower High (RSI > 60)
- Swing strength: 2 candles each side
- Alert: 1 candle after swing confirmed
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
    TOP_100_MARKET_CAP, TOP_COINS_COUNT, QUOTE_CURRENCY, 
    EXCLUDED_SYMBOLS, EXCLUDE_LEVERAGED, TIMEZONE, LOOKBACK_CANDLES,
    SWING_STRENGTH, MIN_SWING_DISTANCE, MAX_SWING_DISTANCE,
    RSI_OVERSOLD, RSI_OVERBOUGHT, MAX_CANDLES_SINCE_SWING2,
    MAX_SIGNAL_AGE, CONFIRMATION_CANDLES
)

SL_TZ = pytz.timezone(TIMEZONE)


class DivergenceType(Enum):
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"


class SignalStrength(Enum):
    STRONG = "strong"    # RSI < 30 or > 70
    MEDIUM = "medium"    # RSI 30-35 or 65-70
    EARLY = "early"      # RSI 35-40 or 60-65


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
class MarketRegime:
    """Market regime classification - INFO ONLY, not a filter"""
    regime_type: str
    regime_emoji: str
    regime_description: str
    adx_value: float
    divergence_rating: str


@dataclass
class VolatilityStatus:
    """Volatility classification - INFO ONLY, not a filter"""
    volatility_type: str
    volatility_emoji: str
    current_atr: float
    average_atr: float
    atr_ratio: float
    position_advice: str


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
    seconds_after_close: float  # Timing metric


def get_sl_time() -> datetime:
    """Get current Sri Lanka time"""
    return datetime.now(SL_TZ)


def format_sl_time(dt: datetime = None) -> str:
    """Format datetime in Sri Lanka time"""
    if dt is None:
        dt = get_sl_time()
    elif dt.tzinfo is None:
        dt = SL_TZ.localize(dt)
    else:
        dt = dt.astimezone(SL_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S IST')


def get_timeframe_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    tf_map = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }
    return tf_map.get(timeframe, 3600)


def get_next_candle_close(timeframe: str) -> Tuple[datetime, float]:
    """Calculate next candle close time and seconds until then"""
    now = datetime.now(pytz.UTC)
    tf_seconds = get_timeframe_seconds(timeframe)
    
    # Calculate current candle start
    epoch_seconds = int(now.timestamp())
    candle_start = (epoch_seconds // tf_seconds) * tf_seconds
    candle_close = candle_start + tf_seconds
    
    next_close = datetime.fromtimestamp(candle_close, pytz.UTC)
    seconds_until = candle_close - epoch_seconds
    
    return next_close, seconds_until


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing - matches TradingView"""
    delta = close_prices.diff()
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    rsi = pd.Series(index=close_prices.index, dtype=float)
    
    if len(close_prices) < period + 1:
        return rsi
    
    # First RSI calculation
    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()
    
    if avg_loss == 0:
        rsi.iloc[period] = 100
    else:
        rsi.iloc[period] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    # Wilder's smoothing for subsequent values
    for i in range(period + 1, len(close_prices)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        
        if avg_loss == 0:
            rsi.iloc[i] = 100
        else:
            rsi.iloc[i] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    return rsi


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX using Wilder's smoothing"""
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
        
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
    
    # Wilder's smoothing
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
    """Calculate ATR and average ATR"""
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


def get_market_regime(df: pd.DataFrame) -> MarketRegime:
    """Classify market regime based on ADX - INFO ONLY"""
    adx = calculate_adx(df)
    
    if adx >= 40:
        return MarketRegime(
            regime_type="strong_trend",
            regime_emoji="🔥",
            regime_description="Strong Trend - Use caution",
            adx_value=adx,
            divergence_rating="RISKY"
        )
    elif adx >= 25:
        return MarketRegime(
            regime_type="trending",
            regime_emoji="📈",
            regime_description="Trending Market",
            adx_value=adx,
            divergence_rating="CAUTION"
        )
    elif adx <= 20:
        return MarketRegime(
            regime_type="ranging",
            regime_emoji="✅",
            regime_description="Ranging - IDEAL for divergence",
            adx_value=adx,
            divergence_rating="IDEAL"
        )
    else:
        return MarketRegime(
            regime_type="weak_trend",
            regime_emoji="⚠️",
            regime_description="Weak Trend",
            adx_value=adx,
            divergence_rating="OK"
        )


def get_volatility_status(df: pd.DataFrame) -> VolatilityStatus:
    """Classify volatility - INFO ONLY"""
    current_atr, avg_atr = calculate_atr(df)
    
    if avg_atr == 0:
        ratio = 1.0
    else:
        ratio = current_atr / avg_atr
    
    if ratio > 1.5:
        return VolatilityStatus(
            volatility_type="high",
            volatility_emoji="🔥",
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=ratio,
            position_advice="Reduce position size"
        )
    elif ratio < 0.7:
        return VolatilityStatus(
            volatility_type="low",
            volatility_emoji="😴",
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=ratio,
            position_advice="Normal position OK"
        )
    else:
        return VolatilityStatus(
            volatility_type="normal",
            volatility_emoji="✅",
            current_atr=current_atr,
            average_atr=avg_atr,
            atr_ratio=ratio,
            position_advice="Normal position OK"
        )


def get_tradingview_link(symbol: str, timeframe: str) -> str:
    """Generate TradingView chart link"""
    exchange_prefix = "BYBIT" if EXCHANGE.lower() == "bybit" else "BINANCE"
    clean_symbol = symbol.replace("/", "")
    
    tf_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D",
    }
    tv_interval = tf_map.get(timeframe, "240")
    
    return f"https://www.tradingview.com/chart/?symbol={exchange_prefix}:{clean_symbol}.P&interval={tv_interval}"


class DivergenceScanner:
    """RSI Divergence Scanner with candle-close aligned timing"""
    
    def __init__(self):
        print(f"[{format_sl_time()}] Initializing RSI Divergence Scanner...")
        
        # Initialize exchange
        if EXCHANGE.lower() == "bybit":
            self.exchange = ccxt.bybit()
        elif EXCHANGE.lower() == "binance":
            self.exchange = ccxt.binance()
        else:
            raise ValueError(f"Unsupported exchange: {EXCHANGE}")
        
        self.exchange.load_markets()
        
        self.alert_cooldowns = {}
        self.sent_signals = {}  # Track sent signals to prevent duplicates
        self.volume_ranks = {}
        
        # Validate available symbols
        self._validate_symbols()
        
        print(f"[{format_sl_time()}] Scanner initialized ({EXCHANGE.upper()})")
    
    def _validate_symbols(self):
        """Validate which symbols from our list are available on exchange"""
        available = set(self.exchange.symbols)
        valid_count = 0
        
        for symbol in TOP_100_MARKET_CAP[:TOP_COINS_COUNT]:
            if symbol in available:
                valid_count += 1
                self.volume_ranks[symbol] = valid_count
        
        print(f"[{format_sl_time()}] Validated {valid_count}/{TOP_COINS_COUNT} symbols")
    
    def get_symbols_to_scan(self) -> List[str]:
        """Get list of top 100 market cap symbols available on exchange"""
        available = set(self.exchange.symbols)
        symbols = []
        
        for symbol in TOP_100_MARKET_CAP[:TOP_COINS_COUNT]:
            if symbol in available and symbol not in EXCLUDED_SYMBOLS:
                symbols.append(symbol)
        
        return symbols
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and calculate RSI"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_swing_lows(self, df: pd.DataFrame, strength: int = SWING_STRENGTH) -> List[SwingPoint]:
        """Find swing low points using CLOSE prices"""
        swings = []
        
        for i in range(strength, len(df) - strength):
            is_low = True
            
            # Check if this candle's close is lower than surrounding closes
            for j in range(1, strength + 1):
                if df['close'].iloc[i] >= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] >= df['close'].iloc[i + j]:
                    is_low = False
                    break
            
            if is_low and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def find_swing_highs(self, df: pd.DataFrame, strength: int = SWING_STRENGTH) -> List[SwingPoint]:
        """Find swing high points using CLOSE prices"""
        swings = []
        
        for i in range(strength, len(df) - strength):
            is_high = True
            
            # Check if this candle's close is higher than surrounding closes
            for j in range(1, strength + 1):
                if df['close'].iloc[i] <= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] <= df['close'].iloc[i + j]:
                    is_high = False
                    break
            
            if is_high and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def check_price_invalidation(self, df: pd.DataFrame, swing1: SwingPoint, 
                                  swing2: SwingPoint, is_bullish: bool) -> bool:
        """Check if price between swings invalidates the pattern"""
        start_idx = swing1.index
        end_idx = swing2.index
        
        if end_idx <= start_idx + 1:
            return True  # No middle section to check
        
        middle_section = df.iloc[start_idx + 1:end_idx]
        
        if is_bullish:
            # For bullish divergence: no CLOSE should go below Swing 2
            min_close = middle_section['close'].min()
            if min_close < swing2.price:
                return False  # Invalidated
        else:
            # For bearish divergence: no CLOSE should go above Swing 2
            max_close = middle_section['close'].max()
            if max_close > swing2.price:
                return False  # Invalidated
        
        return True  # Valid
    
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
    
    def _get_signal_key(self, symbol: str, timeframe: str, swing2_ts: datetime) -> str:
        """Generate unique key for a signal to prevent duplicates"""
        return f"{symbol}_{timeframe}_{swing2_ts.isoformat()}"
    
    def _is_signal_sent(self, symbol: str, timeframe: str, swing2_ts: datetime) -> bool:
        """Check if this exact signal was already sent"""
        key = self._get_signal_key(symbol, timeframe, swing2_ts)
        return key in self.sent_signals
    
    def _mark_signal_sent(self, symbol: str, timeframe: str, swing2_ts: datetime):
        """Mark signal as sent"""
        key = self._get_signal_key(symbol, timeframe, swing2_ts)
        self.sent_signals[key] = time.time()
        
        # Clean old entries (older than 24 hours)
        cutoff = time.time() - 86400
        self.sent_signals = {k: v for k, v in self.sent_signals.items() if v > cutoff}
    
    def detect_divergences(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[AlertSignal]:
        """Detect valid divergences with all checks"""
        signals = []
        current_idx = len(df) - 1
        now = get_sl_time()
        
        # Get timeframe-specific recency limit
        max_recency = MAX_CANDLES_SINCE_SWING2.get(timeframe, 3)
        max_age_seconds = MAX_SIGNAL_AGE.get(timeframe, 7200)
        
        # Find swings
        swing_lows = self.find_swing_lows(df, SWING_STRENGTH)
        swing_highs = self.find_swing_highs(df, SWING_STRENGTH)
        
        # Get last candle close time
        last_candle_close = df['timestamp'].iloc[-1]
        
        # Calculate seconds since last candle close
        now_utc = datetime.now(pytz.UTC)
        seconds_since_close = (now_utc - last_candle_close).total_seconds()
        
        # ========== CHECK BULLISH DIVERGENCE ==========
        if len(swing_lows) >= 2:
            # Check most recent swing pairs first
            for i in range(len(swing_lows) - 1, 0, -1):
                swing1 = swing_lows[i - 1]
                swing2 = swing_lows[i]
                
                # Distance check
                candles_apart = swing2.index - swing1.index
                if candles_apart < MIN_SWING_DISTANCE or candles_apart > MAX_SWING_DISTANCE:
                    continue
                
                # Recency check - is Swing 2 recent enough?
                candles_since_swing2 = current_idx - swing2.index
                if candles_since_swing2 > max_recency:
                    continue
                
                # Age check - is the signal too old?
                swing2_age = (now_utc - swing2.timestamp).total_seconds()
                if swing2_age > max_age_seconds:
                    continue
                
                # Already sent check
                if self._is_signal_sent(symbol, timeframe, swing2.timestamp):
                    continue
                
                # BULLISH: Price Lower Low + RSI Higher Low
                if swing2.price < swing1.price and swing2.rsi > swing1.rsi:
                    # RSI must be in oversold zone
                    if swing2.rsi >= RSI_OVERSOLD:
                        continue
                    
                    # Price invalidation check
                    if not self.check_price_invalidation(df, swing1, swing2, True):
                        continue
                    
                    # Valid signal! Build why_valid list
                    why_valid = [
                        f"Price Lower Low: {swing2.price:.6f} < {swing1.price:.6f}",
                        f"RSI Higher Low: {swing2.rsi:.1f} > {swing1.rsi:.1f}",
                        f"RSI Oversold: {swing2.rsi:.1f} < {RSI_OVERSOLD}",
                        f"Distance: {candles_apart} candles (min {MIN_SWING_DISTANCE})",
                        f"Recency: {candles_since_swing2} candles ago (max {max_recency})",
                        f"Timing: {seconds_since_close:.0f}s after close"
                    ]
                    
                    # Determine signal strength
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
                        timestamp=now,
                        volume_rank=self.volume_ranks.get(symbol, 999),
                        tradingview_link=get_tradingview_link(symbol, timeframe),
                        candle_close_time=last_candle_close,
                        why_valid=why_valid,
                        seconds_after_close=seconds_since_close
                    ))
                    
                    # Only report best (most recent) bullish divergence
                    break
        
        # ========== CHECK BEARISH DIVERGENCE ==========
        if len(swing_highs) >= 2:
            for i in range(len(swing_highs) - 1, 0, -1):
                swing1 = swing_highs[i - 1]
                swing2 = swing_highs[i]
                
                # Distance check
                candles_apart = swing2.index - swing1.index
                if candles_apart < MIN_SWING_DISTANCE or candles_apart > MAX_SWING_DISTANCE:
                    continue
                
                # Recency check
                candles_since_swing2 = current_idx - swing2.index
                if candles_since_swing2 > max_recency:
                    continue
                
                # Age check
                swing2_age = (now_utc - swing2.timestamp).total_seconds()
                if swing2_age > max_age_seconds:
                    continue
                
                # Already sent check
                if self._is_signal_sent(symbol, timeframe, swing2.timestamp):
                    continue
                
                # BEARISH: Price Higher High + RSI Lower High
                if swing2.price > swing1.price and swing2.rsi < swing1.rsi:
                    # RSI must be in overbought zone
                    if swing2.rsi <= RSI_OVERBOUGHT:
                        continue
                    
                    # Price invalidation check
                    if not self.check_price_invalidation(df, swing1, swing2, False):
                        continue
                    
                    # Valid signal!
                    why_valid = [
                        f"Price Higher High: {swing2.price:.6f} > {swing1.price:.6f}",
                        f"RSI Lower High: {swing2.rsi:.1f} < {swing1.rsi:.1f}",
                        f"RSI Overbought: {swing2.rsi:.1f} > {RSI_OVERBOUGHT}",
                        f"Distance: {candles_apart} candles (min {MIN_SWING_DISTANCE})",
                        f"Recency: {candles_since_swing2} candles ago (max {max_recency})",
                        f"Timing: {seconds_since_close:.0f}s after close"
                    ]
                    
                    # Determine signal strength
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
                        timestamp=now,
                        volume_rank=self.volume_ranks.get(symbol, 999),
                        tradingview_link=get_tradingview_link(symbol, timeframe),
                        candle_close_time=last_candle_close,
                        why_valid=why_valid,
                        seconds_after_close=seconds_since_close
                    ))
                    
                    # Only report best bearish divergence
                    break
        
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
            self._mark_signal_sent(symbol, timeframe, signal.divergence.swing2.timestamp)
            self._set_cooldown(symbol, timeframe)
            print(f"  🎯 SIGNAL: {symbol} {timeframe} {signal.divergence.divergence_type.value} "
                  f"({signal.seconds_after_close:.0f}s after close)")
        
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
                    time.sleep(0.1)  # Rate limit
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        print(f"[{format_sl_time()}] ✅ Scan complete. Found {len(all_signals)} signals.")
        return all_signals
    
    def get_market_info(self, symbol: str, timeframe: str) -> Tuple[Optional[MarketRegime], Optional[VolatilityStatus]]:
        """Get market regime and volatility info for a symbol"""
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
        
        # Calculate trade levels
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

📊 {alert.symbol} (#{alert.volume_rank}) | {alert.timeframe.upper()}
⏰ Candle Close: {format_sl_time(alert.candle_close_time)}
⚡ Alert Delay: {alert.seconds_after_close:.0f} seconds

━━━━━━━━━━━━━━━━━━━━━━
📈 DIVERGENCE DETECTED
━━━━━━━━━━━━━━━━━━━━━━

Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
         {div.swing1.timestamp.strftime('%m-%d %H:%M')} UTC

Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
         {div.swing2.timestamp.strftime('%m-%d %H:%M')} UTC

Now:     {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})

━━━━━━━━━━━━━━━━━━━━━━
✅ WHY THIS IS VALID
━━━━━━━━━━━━━━━━━━━━━━
{conditions}

━━━━━━━━━━━━━━━━━━━━━━
🎯 TRADE IDEA
━━━━━━━━━━━━━━━━━━━━━━
Entry: {fmt(entry)} (current)
SL: {fmt(sl)} (below swing)
TP1: {fmt(tp1)} (+2%)
TP2: {fmt(tp2)} (+5%)

📺 {alert.tradingview_link}

⚠️ DYOR - Not financial advice
🕐 {format_sl_time()}"""
        
        return msg
    
    @staticmethod
    def format_alert_with_regime(alert: AlertSignal, regime: MarketRegime, 
                                  volatility: VolatilityStatus) -> str:
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

📊 {alert.symbol} (#{alert.volume_rank}) | {alert.timeframe.upper()}
⏰ Candle Close: {format_sl_time(alert.candle_close_time)}
⚡ Alert Delay: {alert.seconds_after_close:.0f} seconds

━━━━━━━━━━━━━━━━━━━━━━
📈 DIVERGENCE DETECTED
━━━━━━━━━━━━━━━━━━━━━━

Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
         {div.swing1.timestamp.strftime('%m-%d %H:%M')} UTC

Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
         {div.swing2.timestamp.strftime('%m-%d %H:%M')} UTC

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
                           volatility: VolatilityStatus, price: float, rsi: float) -> str:
        """Format market regime info"""
        
        def fmt(p):
            if p >= 1000:
                return f"${p:,.2f}"
            elif p >= 1:
                return f"${p:.4f}"
            else:
                return f"${p:.6f}"
        
        return f"""*📊 Market Regime Analysis*

*{symbol}* | {timeframe.upper()}
Price: {fmt(price)}
RSI: {rsi:.1f}

━━━━━━━━━━━━━━━━━━━━━━
*{regime.regime_emoji} Market Regime*
{regime.regime_description}
ADX: {regime.adx_value:.1f}
Divergence Rating: *{regime.divergence_rating}*

━━━━━━━━━━━━━━━━━━━━━━
*{volatility.volatility_emoji} Volatility*
Type: {volatility.volatility_type.upper()}
ATR Ratio: {volatility.atr_ratio:.2f}x
💡 {volatility.position_advice}

━━━━━━━━━━━━━━━━━━━━━━
🕐 {format_sl_time()}"""
    
    @staticmethod
    def format_simple(alert: AlertSignal) -> str:
        """Short format for quick view"""
        div = alert.divergence
        is_bull = div.divergence_type == DivergenceType.BULLISH_REGULAR
        icon = "🟢" if is_bull else "🔴"
        direction = "BULL" if is_bull else "BEAR"
        return f"{icon} {alert.symbol} {alert.timeframe} | {direction} | RSI: {div.swing2.rsi:.1f}"
