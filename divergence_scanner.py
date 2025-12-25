"""
RSI Divergence Bot V10 - FIXED VERSION
Fixed: Bybit ticker fetching issue
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
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Directional Movement
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
    """V10 Complete Scanner with all features - FIXED"""
    
    def __init__(self):
        print(f"[{format_sl_time()}] Initializing V10 scanner...")
        
        # Initialize exchange
        if EXCHANGE.lower() == "bybit":
            self.exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Use spot market
                }
            })
        elif EXCHANGE.lower() == "binance":
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
            })
        else:
            raise ValueError(f"Unsupported exchange: {EXCHANGE}")
        
        self.exchange.load_markets()
        
        self.alert_cooldowns = {}
        self.volume_ranks = {}
        
        print(f"[{format_sl_time()}] V10 Scanner initialized ({EXCHANGE.upper()})")
    
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        """Fetch top coins by 24h volume - FIXED VERSION"""
        try:
            print(f"[{format_sl_time()}] Fetching top coins from {EXCHANGE.upper()}...")
            
            # Get all valid USDT spot markets first
            valid_symbols = []
            for symbol, market in self.exchange.markets.items():
                # Only spot USDT pairs
                if (market.get('quote') == 'USDT' and 
                    market.get('spot', False) and 
                    market.get('active', True) and
                    symbol not in EXCLUDED_SYMBOLS):
                    
                    # Exclude leveraged tokens
                    if EXCLUDE_LEVERAGED:
                        base = market.get('base', '')
                        if any(x in base for x in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S']):
                            continue
                    
                    valid_symbols.append(symbol)
            
            print(f"[{format_sl_time()}] Found {len(valid_symbols)} valid USDT spot pairs")
            
            # Fetch tickers for valid symbols only (in batches to avoid rate limits)
            all_tickers = {}
            batch_size = 50
            
            for i in range(0, len(valid_symbols), batch_size):
                batch = valid_symbols[i:i+batch_size]
                try:
                    # Fetch individual tickers for each symbol
                    for symbol in batch:
                        try:
                            ticker = self.exchange.fetch_ticker(symbol)
                            if ticker and ticker.get('quoteVolume'):
                                all_tickers[symbol] = ticker
                        except Exception as e:
                            # Skip symbols that fail
                            continue
                    
                    time.sleep(0.1)  # Rate limit
                except Exception as e:
                    print(f"[{format_sl_time()}] Batch error: {e}")
                    continue
            
            print(f"[{format_sl_time()}] Fetched {len(all_tickers)} tickers successfully")
            
            # Sort by 24h quote volume
            sorted_pairs = sorted(
                all_tickers.items(),
                key=lambda x: float(x[1].get('quoteVolume', 0) or 0),
                reverse=True
            )
            
            # Get top N
            top_symbols = [symbol for symbol, _ in sorted_pairs[:count]]
            
            # Store volume ranks
            self.volume_ranks = {}
            for rank, symbol in enumerate(top_symbols, 1):
                self.volume_ranks[symbol] = rank
            
            print(f"[{format_sl_time()}] Selected top {len(top_symbols)} coins by volume")
            
            if top_symbols:
                print(f"[{format_sl_time()}] Top 5: {', '.join(top_symbols[:5])}")
            
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] ERROR fetching top coins: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def fetch_top_coins_by_volume_v2(self, count: int = 100) -> List[str]:
        """Alternative method using markets data directly"""
        try:
            print(f"[{format_sl_time()}] Fetching top coins (V2 method)...")
            
            # Reload markets to get fresh data
            self.exchange.load_markets()
            
            # Get USDT spot pairs with volume info from markets
            pairs_with_volume = []
            
            for symbol, market in self.exchange.markets.items():
                if (market.get('quote') == 'USDT' and 
                    market.get('spot', False) and 
                    market.get('active', True) and
                    symbol not in EXCLUDED_SYMBOLS):
                    
                    base = market.get('base', '')
                    
                    # Exclude leveraged/special tokens
                    if EXCLUDE_LEVERAGED:
                        if any(x in base for x in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S', '5L', '5S']):
                            continue
                    
                    # Exclude stablecoins
                    if base in ['USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDD', 'USDP']:
                        continue
                    
                    pairs_with_volume.append(symbol)
            
            print(f"[{format_sl_time()}] Found {len(pairs_with_volume)} valid pairs")
            
            # Now fetch volume for top pairs by fetching tickers one by one
            # First, let's try a batch approach with error handling
            volume_data = {}
            
            # Try fetching all tickers at once with proper error handling
            try:
                # For Bybit, we need to specify the market type
                if EXCHANGE.lower() == "bybit":
                    # Use the spot tickers endpoint directly
                    tickers = self.exchange.fetch_tickers(params={'category': 'spot'})
                    
                    for symbol, ticker in tickers.items():
                        if symbol in pairs_with_volume:
                            vol = ticker.get('quoteVolume', 0)
                            if vol:
                                volume_data[symbol] = float(vol)
                else:
                    tickers = self.exchange.fetch_tickers()
                    for symbol, ticker in tickers.items():
                        if symbol in pairs_with_volume:
                            vol = ticker.get('quoteVolume', 0)
                            if vol:
                                volume_data[symbol] = float(vol)
                                
            except Exception as e:
                print(f"[{format_sl_time()}] Batch ticker fetch failed: {e}")
                print(f"[{format_sl_time()}] Falling back to individual fetches...")
                
                # Fallback: fetch tickers individually for major pairs
                major_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
                              'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
                              'SHIB/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT', 'ATOM/USDT',
                              'XLM/USDT', 'ETC/USDT', 'FIL/USDT', 'NEAR/USDT', 'APT/USDT']
                
                for symbol in pairs_with_volume[:200]:  # Check first 200
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        vol = ticker.get('quoteVolume', 0)
                        if vol:
                            volume_data[symbol] = float(vol)
                        time.sleep(0.05)
                    except:
                        continue
            
            # Sort by volume
            sorted_pairs = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, _ in sorted_pairs[:count]]
            
            # Store ranks
            self.volume_ranks = {}
            for rank, symbol in enumerate(top_symbols, 1):
                self.volume_ranks[symbol] = rank
            
            print(f"[{format_sl_time()}] Got {len(top_symbols)} coins by volume")
            if top_symbols:
                print(f"[{format_sl_time()}] Top 5: {', '.join(top_symbols[:5])}")
            
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] V2 method error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_symbols_to_scan(self) -> List[str]:
        """Get list of symbols to scan"""
        # Try V2 method first (more reliable for Bybit)
        symbols = self.fetch_top_coins_by_volume_v2(TOP_COINS_COUNT)
        
        if not symbols:
            # Fallback to V1 if V2 fails
            symbols = self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
        
        if not symbols:
            # Ultimate fallback - use hardcoded major pairs
            print(f"[{format_sl_time()}] Using fallback coin list...")
            symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
                'SHIB/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT', 'ATOM/USDT',
                'XLM/USDT', 'ETC/USDT', 'FIL/USDT', 'NEAR/USDT', 'APT/USDT',
                'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'SUI/USDT', 'SEI/USDT',
                'TIA/USDT', 'PEPE/USDT', 'WIF/USDT', 'BONK/USDT', 'FLOKI/USDT'
            ]
            for rank, sym in enumerate(symbols, 1):
                self.volume_ranks[sym] = rank
        
        return symbols
    
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
    
    def check_recency(self, current_idx: int, swing2_idx: int, timeframe: str) -> bool:
        """Check if Swing 2 is recent enough"""
        candles_since = current_idx - swing2_idx
        max_allowed = MAX_CANDLES_SINCE_SWING2.get(timeframe, 10)
        
        if candles_since > max_allowed:
            return False
        
        return True
    
    def check_price_movement(self, current_price: float, swing2_price: float) -> Tuple[bool, float]:
        """Check if price hasn't moved too much already"""
        move_pct = abs(current_price - swing2_price) / swing2_price
        
        if move_pct > 0.15:
            return False, move_pct
        
        return True, move_pct
    
    def check_2_candle_confirmation(self, df: pd.DataFrame, is_bullish: bool,
                                    swing2_idx: int) -> ConfirmationStatus:
        """2-candle confirmation"""
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
            adx_direction = "Strong"
        elif adx > MIN_ADX_MODERATE:
            adx_confirmed = True
            adx_direction = "Moderate"
        else:
            adx_confirmed = False
            adx_direction = "Weak"
        
        rsi_values = df['rsi'].tail(5)
        rsi_rising = all(rsi_values.iloc[i] > rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        rsi_falling = all(rsi_values.iloc[i] < rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        
        if rsi_rising:
            rsi_direction = "Rising"
            rsi_confirmed = is_bullish
        elif rsi_falling:
            rsi_direction = "Falling"
            rsi_confirmed = not is_bullish
        else:
            rsi_direction = "Sideways"
            rsi_confirmed = False
        
        price_values = df['close'].tail(5)
        price_rising = all(price_values.iloc[i] > price_values.iloc[i-1] for i in range(1, len(price_values)))
        price_falling = all(price_values.iloc[i] < price_values.iloc[i-1] for i in range(1, len(price_values)))
        
        if price_rising:
            price_direction = "Rising"
            price_confirmed = is_bullish
        elif price_falling:
            price_direction = "Falling"
            price_confirmed = not is_bullish
        else:
            price_direction = "Sideways"
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
            price_trend = "Rising"
        elif price_falling:
            price_trend = "Falling"
        else:
            price_trend = "Sideways"
        
        rsi_values = df_lower['rsi'].tail(5)
        rsi_rising = all(rsi_values.iloc[i] > rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        rsi_falling = all(rsi_values.iloc[i] < rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        
        if rsi_rising:
            rsi_trend = "Rising"
        elif rsi_falling:
            rsi_trend = "Falling"
        else:
            rsi_trend = "Sideways"
        
        if adx > MIN_ADX_STRONG:
            trend_strength = "Very Strong"
        elif adx > MIN_ADX_MODERATE:
            trend_strength = "Strong"
        else:
            trend_strength = "Weak"
        
        if is_bullish and "Rising" in price_trend:
            trend_direction = f"{trend_strength} Up"
        elif not is_bullish and "Falling" in price_trend:
            trend_direction = f"{trend_strength} Down"
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
        """Detect divergences with all checks"""
        
        # Check bullish regular
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
        
        # Check bearish regular
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
        """Complete scan with all filters"""
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
        
        # Filter 1: Recency check
        if not self.check_recency(current_idx, swing2_idx, timeframe):
            return alerts
        
        # Filter 2: Price movement check
        movement_ok, move_pct = self.check_price_movement(divergence.current_price, divergence.swing2.price)
        if not movement_ok:
            return alerts
        
        # Filter 3: 2-candle confirmation
        confirmation = self.check_2_candle_confirmation(df, is_bullish, swing2_idx)
        if not confirmation.is_confirmed:
            return alerts
        
        # Filter 4: Momentum check (ADX)
        momentum = self.check_momentum_with_adx(df, is_bullish)
        
        # Filter 5: MTF trend check
        mtf_trend = self.check_mtf_trend(symbol, timeframe, is_bullish)
        if mtf_trend and not mtf_trend.is_confirmed:
            return alerts
        
        # Calculate final confidence
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
        
        # Determine signal strength
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
    """Format alerts for Telegram"""
    
    @staticmethod
    def format_alert(alert: AlertSignal) -> str:
        """Format complete alert"""
        div = alert.divergence
        is_bull = "BULLISH" in div.divergence_type.value.upper()
        
        # Strength
        if alert.signal_strength == SignalStrength.STRONG:
            emoji, label = "[STRONG]", "STRONG"
        elif alert.signal_strength == SignalStrength.MEDIUM:
            emoji, label = "[MEDIUM]", "MEDIUM"
        else:
            emoji, label = "[EARLY]", "EARLY"
        
        direction = "BULLISH (UP)" if is_bull else "BEARISH (DOWN)"
        
        # Format prices
        def fmt(p):
            return f"${p:,.2f}" if p >= 1 else f"${p:.6f}"
        
        # Confirmation
        conf = alert.confirmation
        conf_text = f"2-Candle: RSI {conf.rsi_rising_count}/2, Price {conf.price_rising_count}/2"
        
        # MTF Trend
        mtf = alert.mtf_trend
        if mtf:
            mtf_section = f"\nLower TF ({mtf.confirmation_tf}): {mtf.trend_direction} (ADX: {mtf.adx:.1f})"
        else:
            mtf_section = ""
        
        # Momentum
        mom = alert.momentum
        
        # Entry/SL/TP
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

{alert.symbol} (#{alert.volume_rank})
{alert.signal_tf.upper()} | {format_sl_time(alert.candle_close_time)}

--- DIVERGENCE ---
{div.divergence_type.value.replace('_', ' ').title()}
Swing 1: {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Now: {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})
{div.candles_apart} candles apart

--- CONFIRMATION ---
{conf_text}{mtf_section}

--- MOMENTUM ---
RSI: {mom.rsi_direction}
Price: {mom.price_direction}
Trend: {mom.adx_direction} (ADX: {mom.adx_value:.1f})

--- TRADE ---
{trade} | Entry: {fmt(entry)}
SL: {fmt(sl)} | TP: {fmt(tp)}

Confidence: {alert.total_confidence * 100:.0f}%
Chart: {alert.tradingview_link}

DYOR | {format_sl_time()}"""
        
        return msg
