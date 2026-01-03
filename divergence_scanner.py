"""
RSI Divergence Bot V11 - COMPLETE IMPLEMENTATION
Features: 2-candle confirmation, MTF trend, recency checks, price movement filters
+ V11: Candle-close aligned scanning with signal delay tracking
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
    MIN_SWING_DISTANCE, MIN_PRICE_MOVE_PCT, SWING_STRENGTH_BARS,
    CONFIRMATION_CANDLES, CONFIRMATION_THRESHOLD, MIN_CONFIDENCE,
    MAX_CANDLES_SINCE_SWING2, TREND_CONFIRMATION_MAP, 
    MIN_ADX_STRONG, MIN_ADX_MODERATE
)

SL_TZ = pytz.timezone(TIMEZONE)


class DivergenceType(Enum):
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"


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
    confidence: float


@dataclass
class ConfirmationStatus:
    candles_checked: int
    rsi_rising_count: int
    price_rising_count: int
    is_confirmed: bool
    rsi_values: List[float]
    price_values: List[float]


@dataclass
class MomentumStatus:
    rsi_confirmed: bool
    rsi_direction: str
    price_confirmed: bool
    price_direction: str
    adx_value: float
    adx_direction: str


@dataclass
class MTFTrendStatus:
    confirmation_tf: str
    adx: float
    trend_direction: str
    price_trend: str
    rsi_trend: str
    is_confirmed: bool
    confidence_boost: float


@dataclass
class AlertSignal:
    symbol: str
    signal_tf: str
    divergence: Divergence
    confirmation: ConfirmationStatus
    momentum: MomentumStatus
    mtf_trend: Optional[MTFTrendStatus]
    signal_strength: SignalStrength
    total_confidence: float
    timestamp: datetime
    volume_rank: int
    tradingview_link: str
    candle_close_time: datetime
    tv_data: Optional[Dict]


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
    """Calculate RSI indicator using Wilder's smoothing"""
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
    """Calculate ADX - Average Directional Index"""
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
    
    return f"https://www.tradingview.com/chart/?symbol={exchange_prefix}:{clean_symbol}.P&interval={tv_interval}"


class DivergenceScanner:
    """V11 Complete Scanner"""
    
    def __init__(self):
        print(f"[{format_sl_time()}] Initializing V11 scanner...")
        
        if EXCHANGE.lower() == "bybit":
            self.exchange = ccxt.bybit()
        elif EXCHANGE.lower() == "binance":
            self.exchange = ccxt.binance()
        else:
            raise ValueError(f"Unsupported exchange: {EXCHANGE}")
        
        self.exchange.load_markets()
        self.alert_cooldowns = {}
        self.volume_ranks = {}  # Now stores market cap ranks
        self.marketcap_ranks = {}
        
        # Top 100 coins by market cap (updated periodically)
        # This list is used to filter and rank coins without external API calls
        self.TOP_MARKETCAP_COINS = [
            "BTC", "ETH", "XRP", "BNB", "SOL", "DOGE", "ADA", "TRX", "AVAX", "LINK",
            "XLM", "SUI", "HBAR", "DOT", "BCH", "LTC", "SHIB", "LEO", "UNI", "ATOM",
            "PEPE", "ETC", "NEAR", "XMR", "APT", "RENDER", "TAO", "VET", "ARB", "FIL",
            "AAVE", "ICP", "IMX", "OP", "INJ", "WIF", "KAS", "STX", "FTM", "BONK",
            "THETA", "SEI", "LDO", "JASMY", "FET", "GRT", "RUNE", "FLOKI", "ALGO", "TIA",
            "PYTH", "JUP", "SAND", "GALA", "ENA", "FLOW", "BEAM", "MANA", "AXS", "EGLD",
            "STRK", "CFX", "QNT", "IOTA", "BSV", "NEO", "KAVA", "XTZ", "AIOZ", "ENS",
            "DYDX", "W", "CHZ", "CAKE", "WLD", "APE", "PENDLE", "SNX", "ZRO", "ORDI",
            "CRV", "BLUR", "ONDO", "1INCH", "MINA", "ZEC", "COMP", "ROSE", "XEC", "OCEAN",
            "GMX", "ARKM", "SSV", "PIXEL", "SUPER", "MATIC", "LUNC", "PEOPLE", "MAGIC", "ENJ"
        ]
        
        print(f"[{format_sl_time()}] V11 Scanner initialized ({EXCHANGE.upper()})")
    
    def fetch_top_coins_by_marketcap(self, count: int = 100) -> List[str]:
        """Fetch top coins by market cap using predefined ranking"""
        try:
            print(f"[{format_sl_time()}] Fetching available pairs from {EXCHANGE.upper()}...")
            tickers = self.exchange.fetch_tickers()
            print(f"[{format_sl_time()}] Received {len(tickers)} tickers")
            
            # Filter for USDT pairs (handles both spot "BTC/USDT" and perps "BTC/USDT:USDT")
            usdt_pairs = {
                symbol: ticker for symbol, ticker in tickers.items()
                if f"/{QUOTE_CURRENCY}" in symbol 
                and symbol.split(':')[0] not in EXCLUDED_SYMBOLS  # Check base pair
            }
            
            # Exclude leveraged tokens
            if EXCLUDE_LEVERAGED:
                usdt_pairs = {
                    symbol: ticker for symbol, ticker in usdt_pairs.items()
                    if not any(x in symbol for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/', '3L/', '3S/'])
                }
            
            print(f"[{format_sl_time()}] Found {len(usdt_pairs)} tradeable USDT pairs")
            
            # Match exchange pairs with market cap ranking
            # Track seen bases to avoid duplicates (e.g., BTC/USDT:USDT and BTC/USDC:USDC)
            ranked_pairs = []
            seen_bases = set()
            
            for symbol in usdt_pairs.keys():
                # Extract base currency (e.g., "BTC" from "BTC/USDT:USDT")
                base = symbol.split('/')[0]
                
                # Skip if we already have this base currency
                if base in seen_bases:
                    continue
                
                if base in self.TOP_MARKETCAP_COINS:
                    rank = self.TOP_MARKETCAP_COINS.index(base) + 1
                    ranked_pairs.append((symbol, rank))
                    seen_bases.add(base)
            
            # Sort by market cap rank (lower = higher market cap)
            ranked_pairs.sort(key=lambda x: x[1])
            
            # Take top N
            top_symbols = [symbol for symbol, rank in ranked_pairs[:count]]
            
            # Store ranks
            for symbol, rank in ranked_pairs[:count]:
                self.volume_ranks[symbol] = rank
                self.marketcap_ranks[symbol] = rank
            
            print(f"[{format_sl_time()}] Selected top {len(top_symbols)} coins by market cap")
            
            # Log top 10 for verification
            if top_symbols:
                top_10 = [s.split('/')[0] for s in top_symbols[:10]]
                print(f"[{format_sl_time()}] Top 10: {', '.join(top_10)}")
            
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] ERROR fetching coins: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Keep old method name for backward compatibility
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        """Deprecated: Now fetches by market cap instead"""
        return self.fetch_top_coins_by_marketcap(count)
    
    def get_symbols_to_scan(self) -> List[str]:
        return self.fetch_top_coins_by_marketcap(TOP_COINS_COUNT)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and calculate RSI"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_swing_highs(self, df: pd.DataFrame, strength: int = 3) -> List[SwingPoint]:
        """Find swing high points"""
        swings = []
        for i in range(strength, len(df) - strength):
            is_high = True
            for j in range(1, strength + 1):
                if df['close'].iloc[i] <= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] <= df['close'].iloc[i + j]:
                    is_high = False
                    break
            if is_high and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(i, df['close'].iloc[i], df['rsi'].iloc[i], df['timestamp'].iloc[i]))
        return swings
    
    def find_swing_lows(self, df: pd.DataFrame, strength: int = 3) -> List[SwingPoint]:
        """Find swing low points"""
        swings = []
        for i in range(strength, len(df) - strength):
            is_low = True
            for j in range(1, strength + 1):
                if df['close'].iloc[i] >= df['close'].iloc[i - j] or \
                   df['close'].iloc[i] >= df['close'].iloc[i + j]:
                    is_low = False
                    break
            if is_low and not pd.isna(df['rsi'].iloc[i]):
                swings.append(SwingPoint(i, df['close'].iloc[i], df['rsi'].iloc[i], df['timestamp'].iloc[i]))
        return swings
    
    def check_price_invalidation(self, df: pd.DataFrame, swing1: SwingPoint, 
                                 swing2: SwingPoint, is_bullish: bool) -> bool:
        """Check if middle prices break the pattern"""
        middle = df.iloc[swing1.index + 1:swing2.index]
        if is_bullish:
            return middle['low'].min() >= swing2.price
        else:
            return middle['high'].max() <= swing2.price
    
    def check_rsi_invalidation(self, df: pd.DataFrame, swing1: SwingPoint,
                               swing2: SwingPoint, is_bullish: bool) -> bool:
        """Check if middle RSI breaks the pattern"""
        middle_rsi = df['rsi'].iloc[swing1.index + 1:swing2.index]
        if is_bullish:
            return middle_rsi.min() >= swing1.rsi
        else:
            return middle_rsi.max() <= swing1.rsi
    
    def check_recency(self, current_idx: int, swing2_idx: int, timeframe: str) -> bool:
        """Check if Swing 2 is recent enough"""
        candles_since = current_idx - swing2_idx
        max_allowed = MAX_CANDLES_SINCE_SWING2.get(timeframe, 10)
        return candles_since <= max_allowed
    
    def check_price_movement(self, current_price: float, swing2_price: float) -> Tuple[bool, float]:
        """Check price hasn't moved too much"""
        move_pct = abs(current_price - swing2_price) / swing2_price
        return move_pct <= 0.15, move_pct
    
    def check_2_candle_confirmation(self, df: pd.DataFrame, is_bullish: bool,
                                    swing2_idx: int) -> ConfirmationStatus:
        """2-candle confirmation check"""
        start_idx = swing2_idx + 1
        end_idx = swing2_idx + 3
        
        if end_idx > len(df):
            return ConfirmationStatus(0, 0, 0, False, [], [])
        
        rsi_vals = df['rsi'].iloc[start_idx:end_idx].tolist()
        price_vals = df['close'].iloc[start_idx:end_idx].tolist()
        
        if is_bullish:
            c1_rsi = rsi_vals[0] > df['rsi'].iloc[swing2_idx]
            c1_price = price_vals[0] > df['close'].iloc[swing2_idx]
            c2_rsi = rsi_vals[1] > rsi_vals[0]
            c2_price = price_vals[1] > price_vals[0]
        else:
            c1_rsi = rsi_vals[0] < df['rsi'].iloc[swing2_idx]
            c1_price = price_vals[0] < df['close'].iloc[swing2_idx]
            c2_rsi = rsi_vals[1] < rsi_vals[0]
            c2_price = price_vals[1] < price_vals[0]
        
        rsi_count = (1 if c1_rsi else 0) + (1 if c2_rsi else 0)
        price_count = (1 if c1_price else 0) + (1 if c2_price else 0)
        confirmed = rsi_count == 2 and price_count == 2
        
        return ConfirmationStatus(2, rsi_count, price_count, confirmed, rsi_vals, price_vals)
    
    def check_momentum_with_adx(self, df: pd.DataFrame, is_bullish: bool) -> MomentumStatus:
        """Check momentum using ADX"""
        adx = calculate_adx(df, period=14)
        
        if adx > MIN_ADX_STRONG:
            adx_dir = "Strong"
        elif adx > MIN_ADX_MODERATE:
            adx_dir = "Moderate"
        else:
            adx_dir = "Weak"
        
        rsi_vals = df['rsi'].tail(5)
        rsi_rising = all(rsi_vals.iloc[i] > rsi_vals.iloc[i-1] for i in range(1, len(rsi_vals)))
        rsi_falling = all(rsi_vals.iloc[i] < rsi_vals.iloc[i-1] for i in range(1, len(rsi_vals)))
        
        if rsi_rising:
            rsi_dir, rsi_conf = "Rising", is_bullish
        elif rsi_falling:
            rsi_dir, rsi_conf = "Falling", not is_bullish
        else:
            rsi_dir, rsi_conf = "Sideways", False
        
        price_vals = df['close'].tail(5)
        price_rising = all(price_vals.iloc[i] > price_vals.iloc[i-1] for i in range(1, len(price_vals)))
        price_falling = all(price_vals.iloc[i] < price_vals.iloc[i-1] for i in range(1, len(price_vals)))
        
        if price_rising:
            price_dir, price_conf = "Rising", is_bullish
        elif price_falling:
            price_dir, price_conf = "Falling", not is_bullish
        else:
            price_dir, price_conf = "Sideways", False
        
        return MomentumStatus(rsi_conf, rsi_dir, price_conf, price_dir, adx, adx_dir)
    
    def check_mtf_trend(self, symbol: str, signal_tf: str, is_bullish: bool) -> Optional[MTFTrendStatus]:
        """Multi-timeframe trend confirmation"""
        lower_tf = TREND_CONFIRMATION_MAP.get(signal_tf)
        if not lower_tf:
            return None
        
        df_lower = self.fetch_ohlcv(symbol, lower_tf, limit=50)
        if df_lower is None or len(df_lower) < 20:
            return None
        
        adx = calculate_adx(df_lower, period=14)
        
        price_vals = df_lower['close'].tail(5)
        p_rising = all(price_vals.iloc[i] > price_vals.iloc[i-1] for i in range(1, len(price_vals)))
        p_falling = all(price_vals.iloc[i] < price_vals.iloc[i-1] for i in range(1, len(price_vals)))
        price_trend = "Rising" if p_rising else ("Falling" if p_falling else "Sideways")
        
        rsi_vals = df_lower['rsi'].tail(5)
        r_rising = all(rsi_vals.iloc[i] > rsi_vals.iloc[i-1] for i in range(1, len(rsi_vals)))
        r_falling = all(rsi_vals.iloc[i] < rsi_vals.iloc[i-1] for i in range(1, len(rsi_vals)))
        rsi_trend = "Rising" if r_rising else ("Falling" if r_falling else "Sideways")
        
        strength = "Very Strong" if adx > MIN_ADX_STRONG else ("Strong" if adx > MIN_ADX_MODERATE else "Weak")
        
        if is_bullish and "Rising" in price_trend:
            trend_dir = f"{strength} Up"
        elif not is_bullish and "Falling" in price_trend:
            trend_dir = f"{strength} Down"
        else:
            trend_dir = f"{strength} (Conflicting)"
        
        if adx < MIN_ADX_MODERATE:
            confirmed, boost = False, -0.10
        elif (is_bullish and "Rising" in price_trend and "Rising" in rsi_trend) or \
             (not is_bullish and "Falling" in price_trend and "Falling" in rsi_trend):
            confirmed, boost = True, +0.15
        elif (is_bullish and "Rising" in price_trend) or (not is_bullish and "Falling" in price_trend):
            confirmed, boost = True, +0.10
        else:
            confirmed, boost = False, -0.05
        
        return MTFTrendStatus(lower_tf, adx, trend_dir, price_trend, rsi_trend, confirmed, boost)
    
    def detect_divergence(self, df: pd.DataFrame, swing_lows: List[SwingPoint],
                          swing_highs: List[SwingPoint]) -> Optional[Divergence]:
        """Detect divergences with all checks"""
        # Check bullish regular
        if len(swing_lows) >= 2:
            for i in range(len(swing_lows) - 1):
                s1, s2 = swing_lows[i], swing_lows[i + 1]
                candles = s2.index - s1.index
                if candles < MIN_SWING_DISTANCE or candles > 50:
                    continue
                if s2.price < s1.price and s2.rsi > s1.rsi:
                    if not self.check_price_invalidation(df, s1, s2, True):
                        continue
                    if not self.check_rsi_invalidation(df, s1, s2, True):
                        continue
                    return Divergence(DivergenceType.BULLISH_REGULAR, s1, s2, 
                                     df['close'].iloc[-1], df['rsi'].iloc[-1], candles, 0.75)
        
        # Check bearish regular
        if len(swing_highs) >= 2:
            for i in range(len(swing_highs) - 1):
                s1, s2 = swing_highs[i], swing_highs[i + 1]
                candles = s2.index - s1.index
                if candles < MIN_SWING_DISTANCE or candles > 50:
                    continue
                if s2.price > s1.price and s2.rsi < s1.rsi:
                    if not self.check_price_invalidation(df, s1, s2, False):
                        continue
                    if not self.check_rsi_invalidation(df, s1, s2, False):
                        continue
                    return Divergence(DivergenceType.BEARISH_REGULAR, s1, s2,
                                     df['close'].iloc[-1], df['rsi'].iloc[-1], candles, 0.75)
        
        return None
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        key = f"{symbol}_{timeframe}"
        if key in self.alert_cooldowns:
            return time.time() - self.alert_cooldowns[key] < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        self.alert_cooldowns[f"{symbol}_{timeframe}"] = time.time()
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
        """V11 Complete scan with all filters"""
        alerts = []
        
        if self._is_on_cooldown(symbol, timeframe):
            return alerts
        
        df = self.fetch_ohlcv(symbol, timeframe, LOOKBACK_CANDLES)
        if df is None or len(df) < 30:
            return alerts
        
        swing_highs = self.find_swing_highs(df, SWING_STRENGTH_BARS)
        swing_lows = self.find_swing_lows(df, SWING_STRENGTH_BARS)
        
        divergence = self.detect_divergence(df, swing_lows, swing_highs)
        if not divergence:
            return alerts
        
        is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
        current_idx = len(df) - 1
        swing2_idx = divergence.swing2.index
        
        # V10 Filters
        if not self.check_recency(current_idx, swing2_idx, timeframe):
            return alerts
        
        movement_ok, _ = self.check_price_movement(divergence.current_price, divergence.swing2.price)
        if not movement_ok:
            return alerts
        
        confirmation = self.check_2_candle_confirmation(df, is_bullish, swing2_idx)
        if not confirmation.is_confirmed:
            return alerts
        
        momentum = self.check_momentum_with_adx(df, is_bullish)
        
        mtf_trend = self.check_mtf_trend(symbol, timeframe, is_bullish)
        if mtf_trend and not mtf_trend.is_confirmed:
            return alerts
        
        # Calculate confidence
        confidence = divergence.confidence
        if confirmation.is_confirmed:
            confidence += 0.10
        if mtf_trend:
            confidence += mtf_trend.confidence_boost
        if momentum.adx_value > MIN_ADX_STRONG:
            confidence += 0.05
        confidence = min(confidence, 0.95)
        
        if confidence < MIN_CONFIDENCE:
            return alerts
        
        if confidence >= 0.85:
            strength = SignalStrength.STRONG
        elif confidence >= 0.75:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.EARLY
        
        self._set_cooldown(symbol, timeframe)
        
        alerts.append(AlertSignal(
            symbol=symbol,
            signal_tf=timeframe,
            divergence=divergence,
            confirmation=confirmation,
            momentum=momentum,
            mtf_trend=mtf_trend,
            signal_strength=strength,
            total_confidence=confidence,
            timestamp=get_sl_time(),
            volume_rank=self.volume_ranks.get(symbol, 999),
            tradingview_link=get_tradingview_link(symbol, timeframe),
            candle_close_time=df['timestamp'].iloc[-1],
            tv_data=None
        ))
        
        return alerts
    
    def scan_all(self) -> List[AlertSignal]:
        """Scan all symbols across all timeframes"""
        all_alerts = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] Scanning {len(symbols)} symbols...")
        
        for symbol in symbols:
            for timeframe in SCAN_TIMEFRAMES:
                try:
                    alerts = self.scan_symbol(symbol, timeframe)
                    all_alerts.extend(alerts)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        return all_alerts


class AlertFormatter:
    """Format V11 alerts for Telegram with timing info"""
    
    @staticmethod
    def format_alert(alert: AlertSignal) -> str:
        """Format complete V11 alert with signal delay tracking"""
        div = alert.divergence
        is_bull = "BULLISH" in div.divergence_type.value.upper()
        
        # Strength indicators
        if alert.signal_strength == SignalStrength.STRONG:
            emoji, label = "\U0001F7E2", "STRONG"
        elif alert.signal_strength == SignalStrength.MEDIUM:
            emoji, label = "\U0001F7E1", "MEDIUM"
        else:
            emoji, label = "\U0001F535", "EARLY"
        
        direction = "BULLISH \U0001F4C8" if is_bull else "BEARISH \U0001F4C9"
        
        def fmt(p):
            return f"${p:,.2f}" if p >= 1 else f"${p:.6f}"
        
        # V11: Calculate signal delay
        candle_close = alert.candle_close_time
        if candle_close.tzinfo is None:
            candle_close = SL_TZ.localize(candle_close)
        
        delay_secs = (alert.timestamp - candle_close).total_seconds()
        if delay_secs < 0:
            delay_secs = 0
        
        if delay_secs < 60:
            delay_str = f"\u26A1 {int(delay_secs)}s after close"
        elif delay_secs < 3600:
            delay_str = f"\u23F1 {int(delay_secs // 60)}m after close"
        else:
            h = int(delay_secs // 3600)
            m = int((delay_secs % 3600) // 60)
            delay_str = f"\u26A0\uFE0F {h}h {m}m after close"
        
        conf = alert.confirmation
        conf_emoji = "\u2705" if conf.is_confirmed else "\u23F3"
        conf_text = f"2-Candle Confirmed! RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2"
        
        # Pre-compute emojis to avoid backslash in f-string
        check_mark = "\u2705"
        cross_mark = "\u274C"
        separator = "\u2501" * 22
        
        mtf = alert.mtf_trend
        if mtf:
            mtf_emoji = check_mark if mtf.adx > MIN_ADX_MODERATE else cross_mark
            price_emoji_mtf = check_mark if 'Rising' in mtf.price_trend else cross_mark
            rsi_emoji_mtf = check_mark if 'Rising' in mtf.rsi_trend else cross_mark
            mtf_section = f"""{separator}
\U0001F50D Lower TF ({mtf.confirmation_tf.upper()}):
{mtf_emoji} Trend: {mtf.trend_direction} (ADX: {mtf.adx:.1f})
{price_emoji_mtf} Price: {mtf.price_trend}
{rsi_emoji_mtf} RSI: {mtf.rsi_trend}"""
        else:
            mtf_section = ""
        
        mom = alert.momentum
        rsi_emoji = check_mark if mom.rsi_confirmed else "\u23F3"
        price_emoji = check_mark if mom.price_confirmed else "\u23F3"
        adx_emoji = check_mark if mom.adx_value > MIN_ADX_MODERATE else "\u26A0\uFE0F"
        
        entry = div.current_price
        if is_bull:
            sl = min(div.swing1.price, div.swing2.price) * 0.99
            tp = entry * 1.04
            trade = "LONG"
        else:
            sl = max(div.swing1.price, div.swing2.price) * 1.01
            tp = entry * 0.96
            trade = "SHORT"
        
        msg = f"""{emoji} {label} SIGNAL - {direction}

\U0001F4CA {alert.symbol} (#{alert.volume_rank})
\u23F0 {alert.signal_tf.upper()} | Candle: {format_sl_time(alert.candle_close_time)}
\U0001F514 Sent: {format_sl_time(alert.timestamp)} ({delay_str})

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
\U0001F4C8 {div.divergence_type.value.replace('_', ' ').title()}

Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Now: {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})

\U0001F50D {div.candles_apart} candles apart
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
{conf_emoji} {conf_text}
{mtf_section}
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
{rsi_emoji} RSI ({alert.signal_tf}): {mom.rsi_direction}
{price_emoji} Price ({alert.signal_tf}): {mom.price_direction}
{adx_emoji} Trend: {mom.adx_direction} (ADX: {mom.adx_value:.1f})
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
\U0001F3AF {trade} | Entry: {fmt(entry)}
\U0001F6D1 SL: {fmt(sl)} | \U0001F3AF TP: {fmt(tp)}

\U0001F525 Confidence: {alert.total_confidence * 100:.0f}%
\U0001F4FA {alert.tradingview_link}

\u26A0\uFE0F DYOR | \U0001F1F1\U0001F1F0 {format_sl_time()}"""
        
        return msg
