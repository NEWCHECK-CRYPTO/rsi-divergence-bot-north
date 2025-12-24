"""
RSI Divergence Bot V10 - FIXED API CONNECTION
Enhanced error handling and multiple fallback methods
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
import traceback

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
    """V10 Scanner - FIXED API CONNECTION"""
    
    def __init__(self):
        print(f"[{format_sl_time()}] 🔧 Initializing V10 scanner with enhanced API connection...")
        
        # Try multiple exchange configurations
        self.exchange = None
        
        if EXCHANGE.lower() == "bybit":
            print(f"[{format_sl_time()}] 🔄 Attempting Bybit connection...")
            
            # Method 1: Try with proxy settings (sometimes helps with network issues)
            try:
                self.exchange = ccxt.bybit({
                    'enableRateLimit': True,
                    'timeout': 30000,  # 30 seconds timeout
                    'options': {
                        'defaultType': 'linear',
                        'recvWindow': 60000,
                    }
                })
                print(f"[{format_sl_time()}] ✅ Bybit exchange object created")
            except Exception as e:
                print(f"[{format_sl_time()}] ❌ Failed to create exchange: {e}")
                raise
                
        elif EXCHANGE.lower() == "binance":
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'future',
                }
            })
        else:
            raise ValueError(f"Unsupported exchange: {EXCHANGE}")
        
        # Load markets with retry
        print(f"[{format_sl_time()}] 🔄 Loading markets (this may take 10-20 seconds)...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.exchange.load_markets()
                print(f"[{format_sl_time()}] ✅ Markets loaded: {len(self.exchange.markets)} pairs")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{format_sl_time()}] ⚠️ Attempt {attempt + 1} failed, retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"[{format_sl_time()}] ❌ ERROR loading markets after {max_retries} attempts: {e}")
                    traceback.print_exc()
                    raise
        
        self.alert_cooldowns = {}
        self.volume_ranks = {}
        
        print(f"[{format_sl_time()}] ✅ V10 Scanner initialized ({EXCHANGE.upper()})")
    
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        """Fetch top coins - ENHANCED with multiple methods"""
        try:
            print(f"[{format_sl_time()}] 🔄 Method 1: Trying fetch_tickers()...")
            
            # Method 1: Standard fetch_tickers
            try:
                tickers = self.exchange.fetch_tickers()
                print(f"[{format_sl_time()}] 📊 Received {len(tickers)} tickers via fetch_tickers()")
                
                if len(tickers) > 0:
                    return self._process_tickers(tickers, count)
            except Exception as e:
                print(f"[{format_sl_time()}] ⚠️ fetch_tickers() failed: {e}")
            
            # Method 2: Fetch from markets list
            print(f"[{format_sl_time()}] 🔄 Method 2: Using markets list...")
            markets = self.exchange.markets
            usdt_symbols = [
                symbol for symbol in markets.keys() 
                if f"/{QUOTE_CURRENCY}" in symbol 
                and markets[symbol].get('active', True)
                and markets[symbol].get('type') == 'swap'
            ]
            
            print(f"[{format_sl_time()}] 📝 Found {len(usdt_symbols)} active {QUOTE_CURRENCY} perpetual pairs")
            
            if len(usdt_symbols) == 0:
                print(f"[{format_sl_time()}] ❌ No USDT perpetual pairs found!")
                return []
            
            # Fetch tickers individually with volume
            print(f"[{format_sl_time()}] 🔄 Fetching individual tickers for top coins...")
            tickers_with_volume = {}
            
            # Sort symbols alphabetically and get common ones first
            priority_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT', 'AVAX']
            priority_symbols = [f"{coin}/{QUOTE_CURRENCY}" for coin in priority_coins if f"{coin}/{QUOTE_CURRENCY}" in usdt_symbols]
            other_symbols = [s for s in usdt_symbols if s not in priority_symbols]
            
            # Fetch priority coins first
            for symbol in priority_symbols[:50]:  # Get top 50 priority
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    if ticker.get('quoteVolume'):
                        tickers_with_volume[symbol] = ticker
                    time.sleep(0.05)  # Small delay to avoid rate limits
                except Exception as e:
                    print(f"  ⚠️ Failed to fetch {symbol}: {str(e)[:50]}")
                    continue
            
            # If we still need more, fetch others
            if len(tickers_with_volume) < count:
                for symbol in other_symbols[:count]:
                    if len(tickers_with_volume) >= count:
                        break
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        if ticker.get('quoteVolume'):
                            tickers_with_volume[symbol] = ticker
                        time.sleep(0.05)
                    except:
                        continue
            
            print(f"[{format_sl_time()}] ✅ Fetched {len(tickers_with_volume)} tickers with volume data")
            
            if len(tickers_with_volume) > 0:
                return self._process_tickers(tickers_with_volume, count)
            
            print(f"[{format_sl_time()}] ❌ All methods failed!")
            return []
            
        except Exception as e:
            print(f"[{format_sl_time()}] ❌ CRITICAL ERROR fetching top coins:")
            print(f"[{format_sl_time()}] Error: {str(e)}")
            traceback.print_exc()
            return []
    
    def _process_tickers(self, tickers: dict, count: int) -> List[str]:
        """Process tickers and return top symbols by volume"""
        # Filter valid USDT pairs
        usdt_pairs = {}
        for symbol, ticker in tickers.items():
            if not symbol.endswith(f"/{QUOTE_CURRENCY}"):
                continue
            if symbol in EXCLUDED_SYMBOLS:
                continue
            if not ticker.get('quoteVolume'):
                continue
            usdt_pairs[symbol] = ticker
        
        print(f"[{format_sl_time()}] 🔍 Filtered to {len(usdt_pairs)} valid {QUOTE_CURRENCY} pairs")
        
        # Exclude leveraged
        if EXCLUDE_LEVERAGED:
            usdt_pairs = {
                symbol: ticker for symbol, ticker in usdt_pairs.items()
                if not any(x in symbol for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/', '3L/', '3S/', '2L/', '2S/'])
            }
            print(f"[{format_sl_time()}] 🚫 After excluding leveraged: {len(usdt_pairs)} pairs")
        
        # Sort by volume
        sorted_pairs = sorted(
            usdt_pairs.items(),
            key=lambda x: x[1].get('quoteVolume', 0),
            reverse=True
        )
        
        top_symbols = [symbol for symbol, _ in sorted_pairs[:count]]
        
        # Store volume ranks
        for rank, symbol in enumerate(top_symbols, 1):
            self.volume_ranks[symbol] = rank
        
        print(f"[{format_sl_time()}] ✅ Selected top {len(top_symbols)} coins by volume")
        
        # Show top 10
        if len(top_symbols) > 0:
            print(f"[{format_sl_time()}] 🏆 Top 10 coins:")
            for i, symbol in enumerate(top_symbols[:10], 1):
                vol = usdt_pairs[symbol].get('quoteVolume', 0)
                print(f"  {i}. {symbol} (Vol: ${vol:,.0f})")
        
        return top_symbols
    
    def get_symbols_to_scan(self) -> List[str]:
        """Get list of symbols to scan"""
        return self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and calculate RSI"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {symbol} {timeframe}: {e}")
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
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
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
                swings.append(SwingPoint(
                    index=i,
                    price=df['close'].iloc[i],
                    rsi=df['rsi'].iloc[i],
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swings
    
    def check_price_invalidation(self, df: pd.DataFrame, swing1: SwingPoint, 
                                 swing2: SwingPoint, is_bullish: bool) -> bool:
        """Check if middle troughs break the pattern"""
        start_idx = swing1.index
        end_idx = swing2.index
        
        middle_section = df.iloc[start_idx + 1:end_idx]
        
        if is_bullish:
            min_middle = middle_section['low'].min()
            if min_middle < swing2.price:
                return False
        else:
            max_middle = middle_section['high'].max()
            if max_middle > swing2.price:
                return False
        
        return True
    
    def check_rsi_invalidation(self, df: pd.DataFrame, swing1: SwingPoint,
                               swing2: SwingPoint, is_bullish: bool) -> bool:
        """Check if middle RSI values break the pattern"""
        start_idx = swing1.index
        end_idx = swing2.index
        
        middle_rsi = df['rsi'].iloc[start_idx + 1:end_idx]
        
        if is_bullish:
            min_middle_rsi = middle_rsi.min()
            if min_middle_rsi < swing1.rsi:
                return False
        else:
            max_middle_rsi = middle_rsi.max()
            if max_middle_rsi > swing1.rsi:
                return False
        
        return True
    
    def check_exact_2candle_timing(self, current_idx: int, swing2_idx: int) -> bool:
        """CRITICAL: Only alert if EXACTLY at 2nd confirmation candle"""
        candles_after_swing2 = current_idx - swing2_idx
        
        if candles_after_swing2 != 2:
            return False
        
        return True
    
    def check_2_candle_confirmation(self, df: pd.DataFrame, is_bullish: bool,
                                    swing2_idx: int) -> ConfirmationStatus:
        """2-candle confirmation check"""
        start_idx = swing2_idx + 1
        end_idx = swing2_idx + 3
        
        if end_idx > len(df):
            return ConfirmationStatus(
                candles_checked=0, rsi_rising_count=0, price_rising_count=0,
                is_confirmed=False, rsi_values=[], price_values=[]
            )
        
        rsi_values = df['rsi'].iloc[start_idx:end_idx].tolist()
        price_values = df['close'].iloc[start_idx:end_idx].tolist()
        
        if is_bullish:
            c1_rsi_rising = rsi_values[0] > df['rsi'].iloc[swing2_idx]
            c1_price_rising = price_values[0] > df['close'].iloc[swing2_idx]
            
            c2_rsi_rising = rsi_values[1] > rsi_values[0]
            c2_price_rising = price_values[1] > price_values[0]
            
            rsi_rising_count = (1 if c1_rsi_rising else 0) + (1 if c2_rsi_rising else 0)
            price_rising_count = (1 if c1_price_rising else 0) + (1 if c2_price_rising else 0)
        else:
            c1_rsi_falling = rsi_values[0] < df['rsi'].iloc[swing2_idx]
            c1_price_falling = price_values[0] < df['close'].iloc[swing2_idx]
            
            c2_rsi_falling = rsi_values[1] < rsi_values[0]
            c2_price_falling = price_values[1] < price_values[0]
            
            rsi_rising_count = (1 if c1_rsi_falling else 0) + (1 if c2_rsi_falling else 0)
            price_rising_count = (1 if c1_price_falling else 0) + (1 if c2_price_falling else 0)
        
        is_confirmed = (rsi_rising_count == 2 and price_rising_count == 2)
        
        return ConfirmationStatus(
            candles_checked=2,
            rsi_rising_count=rsi_rising_count,
            price_rising_count=price_rising_count,
            is_confirmed=is_confirmed,
            rsi_values=rsi_values,
            price_values=price_values
        )
    
    def check_momentum_with_adx(self, df: pd.DataFrame, is_bullish: bool) -> MomentumStatus:
        """Check momentum using ADX"""
        adx = calculate_adx(df, period=14)
        
        if adx > MIN_ADX_STRONG:
            adx_confirmed = True
            adx_direction = "Strong 💪"
        elif adx > MIN_ADX_MODERATE:
            adx_confirmed = True
            adx_direction = "Moderate 👍"
        else:
            adx_confirmed = False
            adx_direction = "Weak ⚠️"
        
        rsi_values = df['rsi'].tail(5)
        rsi_rising = all(rsi_values.iloc[i] > rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        rsi_falling = all(rsi_values.iloc[i] < rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        
        if rsi_rising:
            rsi_direction = "Rising ↗️"
            rsi_confirmed = is_bullish
        elif rsi_falling:
            rsi_direction = "Falling ↘️"
            rsi_confirmed = not is_bullish
        else:
            rsi_direction = "Sideways ↔️"
            rsi_confirmed = False
        
        price_values = df['close'].tail(5)
        price_rising = all(price_values.iloc[i] > price_values.iloc[i-1] for i in range(1, len(price_values)))
        price_falling = all(price_values.iloc[i] < price_values.iloc[i-1] for i in range(1, len(price_values)))
        
        if price_rising:
            price_direction = "Rising ↗️"
            price_confirmed = is_bullish
        elif price_falling:
            price_direction = "Falling ↘️"
            price_confirmed = not is_bullish
        else:
            price_direction = "Sideways ↔️"
            price_confirmed = False
        
        return MomentumStatus(
            rsi_confirmed=rsi_confirmed,
            rsi_direction=rsi_direction,
            price_confirmed=price_confirmed,
            price_direction=price_direction,
            adx_value=adx,
            adx_direction=adx_direction
        )
    
    def check_mtf_trend(self, symbol: str, signal_tf: str, is_bullish: bool) -> Optional[MTFTrendStatus]:
        """Multi-timeframe trend confirmation"""
        lower_tf = TREND_CONFIRMATION_MAP.get(signal_tf)
        if not lower_tf:
            return None
        
        df_lower = self.fetch_ohlcv(symbol, lower_tf, limit=50)
        if df_lower is None or len(df_lower) < 20:
            return None
        
        adx = calculate_adx(df_lower, period=14)
        
        price_values = df_lower['close'].tail(5)
        price_rising = all(price_values.iloc[i] > price_values.iloc[i-1] for i in range(1, len(price_values)))
        price_falling = all(price_values.iloc[i] < price_values.iloc[i-1] for i in range(1, len(price_values)))
        
        if price_rising:
            price_trend = "Rising ↗️"
        elif price_falling:
            price_trend = "Falling ↘️"
        else:
            price_trend = "Sideways ↔️"
        
        rsi_values = df_lower['rsi'].tail(5)
        rsi_rising = all(rsi_values.iloc[i] > rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        rsi_falling = all(rsi_values.iloc[i] < rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        
        if rsi_rising:
            rsi_trend = "Rising ↗️"
        elif rsi_falling:
            rsi_trend = "Falling ↘️"
        else:
            rsi_trend = "Sideways ↔️"
        
        if adx > MIN_ADX_STRONG:
            trend_strength = "Very Strong"
        elif adx > MIN_ADX_MODERATE:
            trend_strength = "Strong"
        else:
            trend_strength = "Weak"
        
        if is_bullish and "Rising" in price_trend:
            trend_direction = f"{trend_strength} Up 📈"
        elif not is_bullish and "Falling" in price_trend:
            trend_direction = f"{trend_strength} Down 📉"
        else:
            trend_direction = f"{trend_strength} (Conflicting)"
        
        if adx < MIN_ADX_MODERATE:
            is_confirmed = False
            confidence_boost = -0.10
        elif (is_bullish and "Rising" in price_trend and "Rising" in rsi_trend):
            is_confirmed = True
            confidence_boost = +0.15
        elif (not is_bullish and "Falling" in price_trend and "Falling" in rsi_trend):
            is_confirmed = True
            confidence_boost = +0.15
        elif (is_bullish and "Rising" in price_trend) or (not is_bullish and "Falling" in price_trend):
            is_confirmed = True
            confidence_boost = +0.10
        else:
            is_confirmed = False
            confidence_boost = -0.05
        
        return MTFTrendStatus(
            confirmation_tf=lower_tf,
            adx=adx,
            trend_direction=trend_direction,
            price_trend=price_trend,
            rsi_trend=rsi_trend,
            is_confirmed=is_confirmed,
            confidence_boost=confidence_boost
        )
    
    def detect_divergence(self, df: pd.DataFrame, swing_lows: List[SwingPoint],
                          swing_highs: List[SwingPoint]) -> Optional[Divergence]:
        """Detect divergences"""
        
        if len(swing_lows) >= 2:
            for i in range(len(swing_lows) - 1):
                swing1 = swing_lows[i]
                swing2 = swing_lows[i + 1]
                
                candles_apart = swing2.index - swing1.index
                if candles_apart < MIN_SWING_DISTANCE or candles_apart > 50:
                    continue
                
                if swing2.price < swing1.price and swing2.rsi > swing1.rsi:
                    if not self.check_price_invalidation(df, swing1, swing2, True):
                        continue
                    
                    if not self.check_rsi_invalidation(df, swing1, swing2, True):
                        continue
                    
                    return Divergence(
                        divergence_type=DivergenceType.BULLISH_REGULAR,
                        swing1=swing1,
                        swing2=swing2,
                        current_price=df['close'].iloc[-1],
                        current_rsi=df['rsi'].iloc[-1],
                        candles_apart=candles_apart,
                        confidence=0.75
                    )
        
        if len(swing_highs) >= 2:
            for i in range(len(swing_highs) - 1):
                swing1 = swing_highs[i]
                swing2 = swing_highs[i + 1]
                
                candles_apart = swing2.index - swing1.index
                if candles_apart < MIN_SWING_DISTANCE or candles_apart > 50:
                    continue
                
                if swing2.price > swing1.price and swing2.rsi < swing1.rsi:
                    if not self.check_price_invalidation(df, swing1, swing2, False):
                        continue
                    
                    if not self.check_rsi_invalidation(df, swing1, swing2, False):
                        continue
                    
                    return Divergence(
                        divergence_type=DivergenceType.BEARISH_REGULAR,
                        swing1=swing1,
                        swing2=swing2,
                        current_price=df['close'].iloc[-1],
                        current_rsi=df['rsi'].iloc[-1],
                        candles_apart=candles_apart,
                        confidence=0.75
                    )
        
        return None
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        """Check if symbol is on cooldown"""
        key = f"{symbol}_{timeframe}"
        if key in self.alert_cooldowns:
            elapsed = time.time() - self.alert_cooldowns[key]
            return elapsed < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        """Set cooldown for symbol"""
        key = f"{symbol}_{timeframe}"
        self.alert_cooldowns[key] = time.time()
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
        """EXACT 2-CANDLE SCAN"""
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
        
        if not self.check_exact_2candle_timing(current_idx, swing2_idx):
            return alerts
        
        confirmation = self.check_2_candle_confirmation(df, is_bullish, swing2_idx)
        if not confirmation.is_confirmed:
            return alerts
        
        momentum = self.check_momentum_with_adx(df, is_bullish)
        
        mtf_trend = self.check_mtf_trend(symbol, timeframe, is_bullish)
        if mtf_trend and not mtf_trend.is_confirmed:
            return alerts
        
        base_confidence = divergence.confidence
        
        if confirmation.is_confirmed:
            base_confidence += 0.10
        
        if mtf_trend:
            base_confidence += mtf_trend.confidence_boost
        
        if momentum.adx_value > MIN_ADX_STRONG:
            base_confidence += 0.05
        
        final_confidence = min(base_confidence, 0.95)
        
        if final_confidence < MIN_CONFIDENCE:
            return alerts
        
        if final_confidence >= 0.85:
            strength = SignalStrength.STRONG
        elif final_confidence >= 0.75:
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
            total_confidence=final_confidence,
            timestamp=get_sl_time(),
            volume_rank=self.volume_ranks.get(symbol, 999),
            tradingview_link=get_tradingview_link(symbol, timeframe),
            candle_close_time=df['timestamp'].iloc[-1],
            tv_data=None
        ))
        
        return alerts
    
    def scan_all(self) -> List[AlertSignal]:
        """Scan all symbols"""
        all_alerts = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] 🔍 Scanning {len(symbols)} symbols...")
        
        for symbol in symbols:
            for timeframe in SCAN_TIMEFRAMES:
                try:
                    alerts = self.scan_symbol(symbol, timeframe)
                    all_alerts.extend(alerts)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"❌ Error scanning {symbol} {timeframe}: {e}")
        
        return all_alerts


class AlertFormatter:
    """Format alerts"""
    
    @staticmethod
    def format_alert(alert: AlertSignal) -> str:
        """Format alert message"""
        div = alert.divergence
        is_bull = "BULLISH" in div.divergence_type.value.upper()
        
        if alert.signal_strength == SignalStrength.STRONG:
            emoji, label = "🟢", "STRONG"
        elif alert.signal_strength == SignalStrength.MEDIUM:
            emoji, label = "🟡", "MEDIUM"
        else:
            emoji, label = "🔴", "EARLY"
        
        direction = "BULLISH 📈" if is_bull else "BEARISH 📉"
        
        def fmt(p):
            return f"${p:,.2f}" if p >= 1 else f"${p:.6f}"
        
        conf = alert.confirmation
        conf_text = f"2-Candle Confirmed! RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2"
        
        mtf = alert.mtf_trend
        if mtf:
            mtf_section = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Lower TF ({mtf.confirmation_tf.upper()}): {mtf.trend_direction} (ADX: {mtf.adx:.1f})"""
        else:
            mtf_section = ""
        
        mom = alert.momentum
        
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

📊 {alert.symbol} (#{alert.volume_rank})
⏰ {alert.signal_tf.upper()} | {format_sl_time(alert.candle_close_time)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 {div.divergence_type.value.replace('_', ' ').title()}

Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Now: {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})

🔍 {div.candles_apart} candles apart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ {conf_text}
⚡ JUST CONFIRMED (2nd candle closed)
{mtf_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 {trade} | Entry: {fmt(entry)}
🛑 SL: {fmt(sl)} | 🎯 TP: {fmt(tp)}

🔥 Confidence: {alert.total_confidence * 100:.0f}%
📺 {alert.tradingview_link}

⚠️ DYOR | 🇱🇰 {format_sl_time()}"""
        
        return msg
