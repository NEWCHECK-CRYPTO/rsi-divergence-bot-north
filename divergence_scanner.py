"""
Divergence Scanner Module - V3
- TradingView RSI Calculation (Wilder's Smoothing/RMA)
- Proper Candle Close Logic
- 3 Signal Levels: Strong, Medium, Early
- RSI Momentum + Price Action confirmation
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import time
import pytz

from config import (
    EXCHANGE, SYMBOLS, SCAN_TIMEFRAMES,
    TIMEFRAME_CONFIRMATION_MAP, RSI_PERIOD,
    RSI_LOOKBACK_CANDLES, SWING_DETECTION_BARS,
    PRICE_TOLERANCE_PERCENT, RSI_TOLERANCE,
    ALERT_COOLDOWN, TOP_COINS_COUNT, QUOTE_CURRENCY, 
    EXCLUDED_SYMBOLS, EXCLUDE_LEVERAGED, TIMEZONE
)

# Sri Lanka timezone
SL_TZ = pytz.timezone(TIMEZONE)


def get_sl_time() -> datetime:
    return datetime.now(SL_TZ)


def format_sl_time(dt: datetime = None) -> str:
    if dt is None:
        dt = get_sl_time()
    elif dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(SL_TZ)
    else:
        dt = dt.astimezone(SL_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S IST')


def calculate_rsi_tradingview(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI exactly like TradingView using Wilder's Smoothing Method (RMA)
    
    TradingView Formula:
    1. Calculate price changes (delta)
    2. Separate gains and losses
    3. First average: SMA of first N periods
    4. Subsequent: RMA = (prev_avg * (N-1) + current) / N
    5. RS = avg_gain / avg_loss
    6. RSI = 100 - (100 / (1 + RS))
    """
    # Calculate price changes
    delta = close_prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Initialize RSI series
    rsi = pd.Series(index=close_prices.index, dtype=float)
    
    # Need at least period + 1 values
    if len(close_prices) < period + 1:
        return rsi
    
    # First average: SMA of first N periods
    first_avg_gain = gains.iloc[1:period + 1].mean()
    first_avg_loss = losses.iloc[1:period + 1].mean()
    
    avg_gain = first_avg_gain
    avg_loss = first_avg_loss
    
    # Calculate first RSI value
    if avg_loss == 0:
        rsi.iloc[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi.iloc[period] = 100 - (100 / (1 + rs))
    
    # Calculate subsequent RSI values using Wilder's smoothing (RMA)
    for i in range(period + 1, len(close_prices)):
        current_gain = gains.iloc[i]
        current_loss = losses.iloc[i]
        
        # Wilder's smoothing: RMA = (prev_avg * (period - 1) + current) / period
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        
        if avg_loss == 0:
            rsi.iloc[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100 - (100 / (1 + rs))
    
    return rsi


class SignalStrength(Enum):
    STRONG = "strong"    # Divergence + MS + RSI + Price confirmed
    MEDIUM = "medium"    # Divergence + (RSI or Price) confirmed
    EARLY = "early"      # Divergence forming


class DivergenceType(Enum):
    STRONG_BULLISH = "strong_bullish"
    MEDIUM_BULLISH = "medium_bullish"
    WEAK_BULLISH = "weak_bullish"
    HIDDEN_BULLISH = "hidden_bullish"
    STRONG_BEARISH = "strong_bearish"
    MEDIUM_BEARISH = "medium_bearish"
    WEAK_BEARISH = "weak_bearish"
    HIDDEN_BEARISH = "hidden_bearish"


class StructureBreak(Enum):
    BULLISH_CHOCH = "Bullish CHoCH"
    BEARISH_CHOCH = "Bearish CHoCH"
    BULLISH_BOS = "Bullish BOS"
    BEARISH_BOS = "Bearish BOS"
    NONE = "None"


@dataclass
class MomentumStatus:
    rsi_confirmed: bool
    rsi_direction: str  # "Rising", "Falling", "Neutral"
    rsi_values: List[float]  # Last 3 RSI values (closed candles)
    price_confirmed: bool
    price_direction: str  # "Rising", "Falling", "Neutral"
    price_values: List[float]  # Last 3 close prices (closed candles)
    price_change_pct: float


@dataclass
class SwingPoint:
    index: int
    price: float  # Close price at swing
    rsi: float
    is_high: bool
    timestamp: datetime


@dataclass
class DivergenceSignal:
    symbol: str
    timeframe: str
    divergence_type: DivergenceType
    point1: SwingPoint
    point2: SwingPoint
    current_price: float
    confidence: float
    
    @property
    def is_bullish(self) -> bool:
        return "bullish" in self.divergence_type.value
    
    @property
    def confirmation_tf(self) -> str:
        return TIMEFRAME_CONFIRMATION_MAP.get(self.timeframe, "1h")
    
    @property
    def div_strength(self) -> str:
        if "strong" in self.divergence_type.value:
            return "STRONG"
        elif "medium" in self.divergence_type.value:
            return "MEDIUM"
        elif "weak" in self.divergence_type.value:
            return "WEAK"
        return "HIDDEN"


@dataclass
class MSConfirmation:
    timeframe: str
    structure_break: StructureBreak
    swing_high: float
    swing_low: float
    current_price: float
    confirmed: bool


@dataclass
class AlertSignal:
    symbol: str
    signal_tf: str
    confirm_tf: str
    divergence: DivergenceSignal
    ms_confirmation: Optional[MSConfirmation]
    signal_strength: SignalStrength
    momentum: MomentumStatus
    total_confidence: float
    timestamp: datetime
    volume_rank: int
    tradingview_link: str
    candle_close_time: datetime  # When the signal candle closed


def get_tradingview_link(symbol: str, timeframe: str) -> str:
    """Generate TradingView chart link"""
    tv_symbol = symbol.replace("/", "")
    tf_map = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W", "1M": "M"
    }
    tv_tf = tf_map.get(timeframe, "60")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}&interval={tv_tf}"


class DivergenceScanner:
    def __init__(self):
        exchange_class = getattr(ccxt, EXCHANGE)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.last_alerts: Dict[str, datetime] = {}
        self.symbols_cache: List[str] = []
        self.symbols_cache_time: datetime = None
        self.volume_ranks: Dict[str, int] = {}
    
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        """Fetch top coins by 24h trading volume from Binance"""
        
        # Cache for 5 minutes
        if (self.symbols_cache_time and 
            (datetime.now() - self.symbols_cache_time).total_seconds() < 300 and
            self.symbols_cache):
            return self.symbols_cache
        
        try:
            print(f"[{format_sl_time()}] Fetching top {count} coins by volume...")
            tickers = self.exchange.fetch_tickers()
            
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith(f'/{QUOTE_CURRENCY}'):
                    continue
                if symbol in EXCLUDED_SYMBOLS:
                    continue
                if EXCLUDE_LEVERAGED:
                    base = symbol.split('/')[0]
                    if any(x in base.upper() for x in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S']):
                        continue
                
                quote_volume = ticker.get('quoteVolume', 0) or 0
                if quote_volume > 0:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': quote_volume,
                        'price': ticker.get('last', 0)
                    })
            
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top_symbols = [p['symbol'] for p in usdt_pairs[:count]]
            self.volume_ranks = {p['symbol']: i+1 for i, p in enumerate(usdt_pairs[:count])}
            
            self.symbols_cache = top_symbols
            self.symbols_cache_time = datetime.now()
            
            print(f"[{format_sl_time()}] Loaded {len(top_symbols)} coins. Top 5: {top_symbols[:5]}")
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] Error fetching top coins: {e}")
            if self.symbols_cache:
                return self.symbols_cache
            return [
                "BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "BNB/USDT",
                "DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "LINK/USDT",
                "XLM/USDT", "SHIB/USDT", "DOT/USDT", "HBAR/USDT", "SUI/USDT",
                "BCH/USDT", "LTC/USDT", "UNI/USDT", "PEPE/USDT", "NEAR/USDT",
                "APT/USDT", "ICP/USDT", "ETC/USDT", "AAVE/USDT", "POL/USDT",
                "FIL/USDT", "ARB/USDT", "OP/USDT", "ATOM/USDT", "INJ/USDT",
                "RENDER/USDT", "FET/USDT", "IMX/USDT", "WIF/USDT", "BONK/USDT",
                "STX/USDT", "TAO/USDT", "SEI/USDT", "SAND/USDT", "MANA/USDT",
                "GALA/USDT", "AXS/USDT", "FTM/USDT", "ALGO/USDT", "THETA/USDT",
                "XTZ/USDT", "VET/USDT", "EGLD/USDT", "FLOW/USDT", "NEO/USDT"
            ]
    
    def get_symbols_to_scan(self) -> List[str]:
        if SYMBOLS and len(SYMBOLS) > 0:
            return SYMBOLS
        return self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    def _get_cooldown_key(self, symbol: str, timeframe: str, strength: SignalStrength) -> str:
        return f"{symbol}_{timeframe}_{strength.value}"
    
    def _is_on_cooldown(self, symbol: str, timeframe: str, strength: SignalStrength) -> bool:
        key = self._get_cooldown_key(symbol, timeframe, strength)
        if key in self.last_alerts:
            elapsed = (datetime.now() - self.last_alerts[key]).total_seconds()
            cooldown = ALERT_COOLDOWN if strength == SignalStrength.STRONG else ALERT_COOLDOWN // 2
            return elapsed < cooldown
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str, strength: SignalStrength):
        key = self._get_cooldown_key(symbol, timeframe, strength)
        self.last_alerts[key] = datetime.now()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data and calculate TradingView-style RSI
        Returns DataFrame with CLOSED candles only (excludes current forming candle)
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # IMPORTANT: Remove the last candle as it's still forming (not closed)
            # We only want to analyze CLOSED candles
            if len(df) > 1:
                df = df.iloc[:-1].copy()
            
            # Calculate RSI using TradingView method
            df['rsi'] = calculate_rsi_tradingview(df['close'], RSI_PERIOD)
            
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_swing_highs(self, df: pd.DataFrame, bars: int = None) -> List[SwingPoint]:
        """Find swing highs using CLOSE prices (not high prices) for consistency"""
        swing_highs = []
        n = bars or SWING_DETECTION_BARS
        
        for i in range(n, len(df) - n):
            # Use close prices for swing detection (more reliable)
            window_closes = df['close'].iloc[i-n:i+n+1]
            current_close = df['close'].iloc[i]
            
            # Also check if high is the highest in the window
            window_highs = df['high'].iloc[i-n:i+n+1]
            current_high = df['high'].iloc[i]
            
            # Swing high: current close is highest close AND current high is highest high
            if current_close == window_closes.max() and current_high == window_highs.max():
                rsi_val = df['rsi'].iloc[i]
                swing_highs.append(SwingPoint(
                    index=i,
                    price=current_close,  # Use close price
                    rsi=rsi_val if pd.notna(rsi_val) else 50,
                    is_high=True,
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swing_highs[-10:] if len(swing_highs) > 10 else swing_highs
    
    def find_swing_lows(self, df: pd.DataFrame, bars: int = None) -> List[SwingPoint]:
        """Find swing lows using CLOSE prices (not low prices) for consistency"""
        swing_lows = []
        n = bars or SWING_DETECTION_BARS
        
        for i in range(n, len(df) - n):
            # Use close prices for swing detection
            window_closes = df['close'].iloc[i-n:i+n+1]
            current_close = df['close'].iloc[i]
            
            # Also check if low is the lowest in the window
            window_lows = df['low'].iloc[i-n:i+n+1]
            current_low = df['low'].iloc[i]
            
            # Swing low: current close is lowest close AND current low is lowest low
            if current_close == window_closes.min() and current_low == window_lows.min():
                rsi_val = df['rsi'].iloc[i]
                swing_lows.append(SwingPoint(
                    index=i,
                    price=current_close,  # Use close price
                    rsi=rsi_val if pd.notna(rsi_val) else 50,
                    is_high=False,
                    timestamp=df['timestamp'].iloc[i]
                ))
        
        return swing_lows[-10:] if len(swing_lows) > 10 else swing_lows
    
    def check_momentum_on_closed_candles(self, df: pd.DataFrame, is_bullish: bool) -> MomentumStatus:
        """
        Check RSI and Price momentum on last 2-3 CLOSED candles
        
        For Bullish:
          - RSI should be rising (RSI[-1] > RSI[-2] > RSI[-3])
          - Price (close) should be rising
        
        For Bearish:
          - RSI should be falling (RSI[-1] < RSI[-2] < RSI[-3])
          - Price (close) should be falling
        """
        if len(df) < 3:
            return MomentumStatus(False, "Neutral", [], False, "Neutral", [], 0.0)
        
        # Get last 3 CLOSED candle RSI values
        rsi_values = []
        for i in range(-3, 0):
            rsi_val = df['rsi'].iloc[i]
            if pd.notna(rsi_val):
                rsi_values.append(round(rsi_val, 2))
        
        # Get last 3 CLOSED candle close prices
        price_values = [round(df['close'].iloc[i], 8) for i in range(-3, 0)]
        
        # Check RSI direction on closed candles
        rsi_confirmed = False
        rsi_direction = "Neutral"
        
        if len(rsi_values) >= 2:
            # Compare last 2 closed candles
            if rsi_values[-1] > rsi_values[-2]:
                rsi_direction = "Rising"
                if is_bullish:
                    rsi_confirmed = True
            elif rsi_values[-1] < rsi_values[-2]:
                rsi_direction = "Falling"
                if not is_bullish:
                    rsi_confirmed = True
            
            # Stronger confirmation: 3 consecutive candles
            if len(rsi_values) >= 3:
                if rsi_values[-1] > rsi_values[-2] > rsi_values[-3]:
                    rsi_direction = "Rising ↗️↗️"
                    if is_bullish:
                        rsi_confirmed = True
                elif rsi_values[-1] < rsi_values[-2] < rsi_values[-3]:
                    rsi_direction = "Falling ↘️↘️"
                    if not is_bullish:
                        rsi_confirmed = True
        
        # Check Price direction on closed candles
        price_confirmed = False
        price_direction = "Neutral"
        price_change_pct = 0.0
        
        if len(price_values) >= 2:
            # Calculate % change between last 2 closed candles
            price_change_pct = ((price_values[-1] - price_values[-2]) / price_values[-2]) * 100
            
            if price_values[-1] > price_values[-2]:
                price_direction = "Rising"
                if is_bullish:
                    price_confirmed = True
            elif price_values[-1] < price_values[-2]:
                price_direction = "Falling"
                if not is_bullish:
                    price_confirmed = True
            
            # Stronger confirmation: 3 consecutive candles
            if len(price_values) >= 3:
                if price_values[-1] > price_values[-2] > price_values[-3]:
                    price_direction = "Rising ↗️↗️"
                    if is_bullish:
                        price_confirmed = True
                elif price_values[-1] < price_values[-2] < price_values[-3]:
                    price_direction = "Falling ↘️↘️"
                    if not is_bullish:
                        price_confirmed = True
        
        return MomentumStatus(
            rsi_confirmed=rsi_confirmed,
            rsi_direction=rsi_direction,
            rsi_values=rsi_values,
            price_confirmed=price_confirmed,
            price_direction=price_direction,
            price_values=price_values,
            price_change_pct=round(price_change_pct, 4)
        )
    
    def detect_bullish_divergence(self, df: pd.DataFrame, swing_lows: List[SwingPoint], 
                                   current_price: float, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        if len(swing_lows) < 2:
            return None
        
        point1, point2 = swing_lows[-2], swing_lows[-1]
        if pd.isna(point1.rsi) or pd.isna(point2.rsi):
            return None
        
        price_pct = ((point2.price - point1.price) / point1.price) * 100
        rsi_change = point2.rsi - point1.rsi
        
        divergence_type, confidence = None, 0.0
        
        # Strong: Price Lower Low, RSI Higher Low
        if point2.price < point1.price and point2.rsi > point1.rsi:
            if abs(price_pct) > PRICE_TOLERANCE_PERCENT:
                divergence_type, confidence = DivergenceType.STRONG_BULLISH, 0.85
        
        # Medium: Price Equal (Double Bottom), RSI Higher Low
        elif abs(price_pct) <= PRICE_TOLERANCE_PERCENT and point2.rsi > point1.rsi:
            divergence_type, confidence = DivergenceType.MEDIUM_BULLISH, 0.75
        
        # Weak: Price Lower Low, RSI Equal
        elif point2.price < point1.price and abs(rsi_change) <= RSI_TOLERANCE:
            divergence_type, confidence = DivergenceType.WEAK_BULLISH, 0.60
        
        # Hidden: Price Higher Low, RSI Lower Low (trend continuation)
        elif point2.price > point1.price and point2.rsi < point1.rsi:
            divergence_type, confidence = DivergenceType.HIDDEN_BULLISH, 0.70
        
        if divergence_type:
            return DivergenceSignal(symbol, timeframe, divergence_type, point1, point2, current_price, confidence)
        return None
    
    def detect_bearish_divergence(self, df: pd.DataFrame, swing_highs: List[SwingPoint], 
                                   current_price: float, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        if len(swing_highs) < 2:
            return None
        
        point1, point2 = swing_highs[-2], swing_highs[-1]
        if pd.isna(point1.rsi) or pd.isna(point2.rsi):
            return None
        
        price_pct = ((point2.price - point1.price) / point1.price) * 100
        rsi_change = point2.rsi - point1.rsi
        
        divergence_type, confidence = None, 0.0
        
        # Strong: Price Higher High, RSI Lower High
        if point2.price > point1.price and point2.rsi < point1.rsi:
            if abs(price_pct) > PRICE_TOLERANCE_PERCENT:
                divergence_type, confidence = DivergenceType.STRONG_BEARISH, 0.85
        
        # Medium: Price Equal (Double Top), RSI Lower High
        elif abs(price_pct) <= PRICE_TOLERANCE_PERCENT and point2.rsi < point1.rsi:
            divergence_type, confidence = DivergenceType.MEDIUM_BEARISH, 0.75
        
        # Weak: Price Higher High, RSI Equal
        elif point2.price > point1.price and abs(rsi_change) <= RSI_TOLERANCE:
            divergence_type, confidence = DivergenceType.WEAK_BEARISH, 0.60
        
        # Hidden: Price Lower High, RSI Higher High (trend continuation)
        elif point2.price < point1.price and point2.rsi > point1.rsi:
            divergence_type, confidence = DivergenceType.HIDDEN_BEARISH, 0.70
        
        if divergence_type:
            return DivergenceSignal(symbol, timeframe, divergence_type, point1, point2, current_price, confidence)
        return None
    
    def check_ms_confirmation(self, symbol: str, confirmation_tf: str, is_bullish: bool) -> Optional[MSConfirmation]:
        df = self.fetch_ohlcv(symbol, confirmation_tf, limit=50)
        if df is None or len(df) < 20:
            return None
        
        swing_highs = self.find_swing_highs(df, bars=3)
        swing_lows = self.find_swing_lows(df, bars=3)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        # Use last CLOSED candle close price
        current_close = df['close'].iloc[-1]
        recent_high, prev_high = swing_highs[-1].price, swing_highs[-2].price
        recent_low, prev_low = swing_lows[-1].price, swing_lows[-2].price
        
        structure_break = StructureBreak.NONE
        
        if is_bullish:
            if current_close > recent_high:
                if recent_high > prev_high:
                    structure_break = StructureBreak.BULLISH_BOS
                else:
                    structure_break = StructureBreak.BULLISH_CHOCH
        else:
            if current_close < recent_low:
                if recent_low < prev_low:
                    structure_break = StructureBreak.BEARISH_BOS
                else:
                    structure_break = StructureBreak.BEARISH_CHOCH
        
        return MSConfirmation(
            confirmation_tf, structure_break, recent_high, recent_low, 
            current_close, structure_break != StructureBreak.NONE
        )
    
    def determine_signal_strength(self, divergence: DivergenceSignal, 
                                   ms_confirmed: bool, momentum: MomentumStatus) -> Tuple[SignalStrength, float]:
        """
        🟢 STRONG: Divergence + MS + (RSI or Price momentum)
        🟡 MEDIUM: Divergence + (RSI or Price momentum), no MS
        🔴 EARLY: Divergence only, no confirmations yet
        """
        base_confidence = divergence.confidence
        
        confirmations = 0
        if ms_confirmed:
            confirmations += 1
            base_confidence += 0.10
        if momentum.rsi_confirmed:
            confirmations += 1
            base_confidence += 0.05
        if momentum.price_confirmed:
            confirmations += 1
            base_confidence += 0.05
        
        if ms_confirmed and (momentum.rsi_confirmed or momentum.price_confirmed):
            return SignalStrength.STRONG, min(base_confidence, 0.95)
        elif momentum.rsi_confirmed or momentum.price_confirmed:
            return SignalStrength.MEDIUM, min(base_confidence, 0.85)
        else:
            return SignalStrength.EARLY, min(base_confidence, 0.70)
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
        """Scan a single symbol and return all signal levels"""
        alerts = []
        
        df = self.fetch_ohlcv(symbol, timeframe)
        if df is None or len(df) < RSI_LOOKBACK_CANDLES:
            return alerts
        
        # Current price is from the last CLOSED candle
        current_price = df['close'].iloc[-1]
        candle_close_time = df['timestamp'].iloc[-1]
        
        swing_highs = self.find_swing_highs(df)
        swing_lows = self.find_swing_lows(df)
        volume_rank = self.volume_ranks.get(symbol, 999)
        
        # Check for bullish divergence
        bullish_div = self.detect_bullish_divergence(df, swing_lows, current_price, symbol, timeframe)
        if bullish_div:
            momentum = self.check_momentum_on_closed_candles(df, is_bullish=True)
            ms_conf = self.check_ms_confirmation(symbol, bullish_div.confirmation_tf, True)
            ms_confirmed = ms_conf and ms_conf.confirmed
            
            signal_strength, confidence = self.determine_signal_strength(
                bullish_div, ms_confirmed, momentum
            )
            
            if not self._is_on_cooldown(symbol, timeframe, signal_strength):
                self._set_cooldown(symbol, timeframe, signal_strength)
                
                alerts.append(AlertSignal(
                    symbol=symbol,
                    signal_tf=timeframe,
                    confirm_tf=bullish_div.confirmation_tf,
                    divergence=bullish_div,
                    ms_confirmation=ms_conf,
                    signal_strength=signal_strength,
                    momentum=momentum,
                    total_confidence=confidence,
                    timestamp=get_sl_time(),
                    volume_rank=volume_rank,
                    tradingview_link=get_tradingview_link(symbol, timeframe),
                    candle_close_time=candle_close_time
                ))
        
        # Check for bearish divergence
        bearish_div = self.detect_bearish_divergence(df, swing_highs, current_price, symbol, timeframe)
        if bearish_div:
            momentum = self.check_momentum_on_closed_candles(df, is_bullish=False)
            ms_conf = self.check_ms_confirmation(symbol, bearish_div.confirmation_tf, False)
            ms_confirmed = ms_conf and ms_conf.confirmed
            
            signal_strength, confidence = self.determine_signal_strength(
                bearish_div, ms_confirmed, momentum
            )
            
            if not self._is_on_cooldown(symbol, timeframe, signal_strength):
                self._set_cooldown(symbol, timeframe, signal_strength)
                
                alerts.append(AlertSignal(
                    symbol=symbol,
                    signal_tf=timeframe,
                    confirm_tf=bearish_div.confirmation_tf,
                    divergence=bearish_div,
                    ms_confirmation=ms_conf,
                    signal_strength=signal_strength,
                    momentum=momentum,
                    total_confidence=confidence,
                    timestamp=get_sl_time(),
                    volume_rank=volume_rank,
                    tradingview_link=get_tradingview_link(symbol, timeframe),
                    candle_close_time=candle_close_time
                ))
        
        return alerts
    
    def scan_all(self, min_strength: SignalStrength = None) -> List[AlertSignal]:
        """Scan all symbols and timeframes"""
        all_alerts = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] Scanning {len(symbols)} coins across {len(SCAN_TIMEFRAMES)} timeframes...")
        
        for symbol in symbols:
            for timeframe in SCAN_TIMEFRAMES:
                try:
                    alerts = self.scan_symbol(symbol, timeframe)
                    
                    if min_strength:
                        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
                        min_level = strength_order.get(min_strength, 1)
                        alerts = [a for a in alerts if strength_order.get(a.signal_strength, 0) >= min_level]
                    
                    all_alerts.extend(alerts)
                    
                    for alert in alerts:
                        emoji = "🟢" if alert.signal_strength == SignalStrength.STRONG else "🟡" if alert.signal_strength == SignalStrength.MEDIUM else "🔴"
                        print(f"[{format_sl_time()}] {emoji} {alert.signal_strength.value.upper()}: {symbol} {timeframe}")
                    
                    time.sleep(0.3)
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
        all_alerts.sort(key=lambda x: (strength_order.get(x.signal_strength, 0), x.total_confidence), reverse=True)
        
        print(f"[{format_sl_time()}] Scan complete. Found {len(all_alerts)} signals.")
        return all_alerts


class AlertFormatter:
    DIV_NAMES = {
        DivergenceType.STRONG_BULLISH: "Strong Bullish",
        DivergenceType.MEDIUM_BULLISH: "Medium Bullish",
        DivergenceType.WEAK_BULLISH: "Weak Bullish",
        DivergenceType.HIDDEN_BULLISH: "Hidden Bullish",
        DivergenceType.STRONG_BEARISH: "Strong Bearish",
        DivergenceType.MEDIUM_BEARISH: "Medium Bearish",
        DivergenceType.WEAK_BEARISH: "Weak Bearish",
        DivergenceType.HIDDEN_BEARISH: "Hidden Bearish",
    }
    
    DIV_DESC = {
        DivergenceType.STRONG_BULLISH: "Price: LL | RSI: HL",
        DivergenceType.MEDIUM_BULLISH: "Price: DB | RSI: HL",
        DivergenceType.WEAK_BULLISH: "Price: LL | RSI: DB",
        DivergenceType.HIDDEN_BULLISH: "Price: HL | RSI: LL",
        DivergenceType.STRONG_BEARISH: "Price: HH | RSI: LH",
        DivergenceType.MEDIUM_BEARISH: "Price: DT | RSI: LH",
        DivergenceType.WEAK_BEARISH: "Price: HH | RSI: DT",
        DivergenceType.HIDDEN_BEARISH: "Price: LH | RSI: HH",
    }
    
    @staticmethod
    def fmt_price(p: float) -> str:
        if p >= 1000:
            return f"${p:,.2f}"
        elif p >= 1:
            return f"${p:.4f}"
        elif p >= 0.0001:
            return f"${p:.6f}"
        else:
            return f"${p:.8f}"
    
    @classmethod
    def format_alert(cls, signal: AlertSignal) -> str:
        is_bull = signal.divergence.is_bullish
        
        # Signal strength emoji and label
        if signal.signal_strength == SignalStrength.STRONG:
            strength_emoji = "🟢"
            strength_label = "STRONG SIGNAL"
        elif signal.signal_strength == SignalStrength.MEDIUM:
            strength_emoji = "🟡"
            strength_label = "MEDIUM SIGNAL"
        else:
            strength_emoji = "🔴"
            strength_label = "EARLY SIGNAL"
        
        direction = "BULLISH" if is_bull else "BEARISH"
        trade = "LONG" if is_bull else "SHORT"
        
        # Calculate trade levels
        if is_bull:
            stop = signal.ms_confirmation.swing_low * 0.995 if signal.ms_confirmation else signal.divergence.point2.price * 0.98
            target = signal.ms_confirmation.swing_high * 1.02 if signal.ms_confirmation else signal.divergence.current_price * 1.04
        else:
            stop = signal.ms_confirmation.swing_high * 1.005 if signal.ms_confirmation else signal.divergence.point2.price * 1.02
            target = signal.ms_confirmation.swing_low * 0.98 if signal.ms_confirmation else signal.divergence.current_price * 0.96
        
        # MS Confirmation status
        if signal.ms_confirmation and signal.ms_confirmation.confirmed:
            ms_status = f"✅ {signal.ms_confirmation.structure_break.value}"
        else:
            ms_status = "⏳ Waiting..."
        
        # RSI momentum
        rsi_emoji = "✅" if signal.momentum.rsi_confirmed else "⏳"
        rsi_vals = " → ".join([str(round(r, 1)) for r in signal.momentum.rsi_values])
        
        # Price momentum
        price_emoji = "✅" if signal.momentum.price_confirmed else "⏳"
        price_change = f"{signal.momentum.price_change_pct:+.2f}%"
        
        # Format candle close time in Sri Lanka time
        candle_time = signal.candle_close_time
        if candle_time.tzinfo is None:
            candle_time = pytz.utc.localize(candle_time).astimezone(SL_TZ)
        else:
            candle_time = candle_time.astimezone(SL_TZ)
        candle_time_str = candle_time.strftime('%Y-%m-%d %H:%M IST')
        
        return f"""{strength_emoji} {strength_label} - {direction}

📊 {signal.symbol} (Vol Rank #{signal.volume_rank})
⏰ Signal: {signal.signal_tf.upper()} | Confirm: {signal.confirm_tf.upper()}
🕐 Candle Close: {candle_time_str}

━━━━━━━━━━━━━━━━━━━━
📈 DIVERGENCE
━━━━━━━━━━━━━━━━━━━━
Type: {cls.DIV_NAMES[signal.divergence.divergence_type]}
Pattern: {cls.DIV_DESC[signal.divergence.divergence_type]}

Point 1: {cls.fmt_price(signal.divergence.point1.price)} (RSI: {signal.divergence.point1.rsi:.1f})
Point 2: {cls.fmt_price(signal.divergence.point2.price)} (RSI: {signal.divergence.point2.rsi:.1f})

━━━━━━━━━━━━━━━━━━━━
✅ CONFIRMATIONS
━━━━━━━━━━━━━━━━━━━━
MS ({signal.confirm_tf}): {ms_status}
{rsi_emoji} RSI: {signal.momentum.rsi_direction} ({rsi_vals})
{price_emoji} Price: {signal.momentum.price_direction} ({price_change})

━━━━━━━━━━━━━━━━━━━━
🎯 TRADE IDEA ({trade})
━━━━━━━━━━━━━━━━━━━━
Entry: {cls.fmt_price(signal.divergence.current_price)}
Stop: {cls.fmt_price(stop)}
Target: {cls.fmt_price(target)}

🔥 Confidence: {signal.total_confidence * 100:.0f}%

📺 TradingView: {signal.tradingview_link}

⚠️ Not financial advice. DYOR!
🇱🇰 {format_sl_time(signal.timestamp)}"""
