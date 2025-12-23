"""
Divergence Scanner Module - V5
- Uses TradingView-TA for RSI (exact TradingView values)
- CCXT for OHLCV price data only
- Divergence detection on TV RSI
- 3 Signal Levels
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

try:
    from tradingview_ta import TA_Handler, Interval
    HAS_TV_TA = True
except ImportError:
    HAS_TV_TA = False
    print("Warning: tradingview-ta not installed. Using fallback RSI calculation.")

from config import (
    EXCHANGE, SYMBOLS, SCAN_TIMEFRAMES,
    TIMEFRAME_CONFIRMATION_MAP, RSI_PERIOD,
    ALERT_COOLDOWN, TOP_COINS_COUNT, QUOTE_CURRENCY, 
    EXCLUDED_SYMBOLS, EXCLUDE_LEVERAGED, TIMEZONE,
    LOOKBACK_CANDLES, MIN_SWING_DISTANCE, MIN_PRICE_MOVE_PCT,
    SWING_STRENGTH_BARS
)

SL_TZ = pytz.timezone(TIMEZONE)

# TradingView interval mapping
TV_INTERVALS = {
    "1m": Interval.INTERVAL_1_MINUTE if HAS_TV_TA else None,
    "5m": Interval.INTERVAL_5_MINUTES if HAS_TV_TA else None,
    "15m": Interval.INTERVAL_15_MINUTES if HAS_TV_TA else None,
    "30m": Interval.INTERVAL_30_MINUTES if HAS_TV_TA else None,
    "1h": Interval.INTERVAL_1_HOUR if HAS_TV_TA else None,
    "2h": Interval.INTERVAL_2_HOURS if HAS_TV_TA else None,
    "4h": Interval.INTERVAL_4_HOURS if HAS_TV_TA else None,
    "1d": Interval.INTERVAL_1_DAY if HAS_TV_TA else None,
    "1w": Interval.INTERVAL_1_WEEK if HAS_TV_TA else None,
    "1M": Interval.INTERVAL_1_MONTH if HAS_TV_TA else None,
}


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


def calculate_rsi_fallback(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Fallback RSI calculation if TV-TA not available"""
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


class SignalStrength(Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    EARLY = "early"


class DivergenceType(Enum):
    BULLISH_REGULAR = "bullish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_REGULAR = "bearish_regular"
    BEARISH_HIDDEN = "bearish_hidden"


@dataclass
class MajorSwing:
    index: int
    price: float
    rsi: float
    is_high: bool
    timestamp: datetime
    strength: float


@dataclass
class DivergenceSignal:
    symbol: str
    timeframe: str
    divergence_type: DivergenceType
    swing1: MajorSwing
    swing2: MajorSwing
    current_price: float
    current_rsi: float  # Current RSI from TradingView
    price_change_pct: float
    rsi_change: float
    candles_apart: int
    confidence: float
    tv_recommendation: str  # TradingView recommendation


@dataclass
class MomentumStatus:
    rsi_confirmed: bool
    rsi_direction: str
    rsi_values: List[float]
    price_confirmed: bool
    price_direction: str
    price_change_pct: float


@dataclass
class AlertSignal:
    symbol: str
    signal_tf: str
    confirm_tf: str
    divergence: DivergenceSignal
    ms_confirmation: Optional[dict]
    signal_strength: SignalStrength
    momentum: MomentumStatus
    total_confidence: float
    timestamp: datetime
    volume_rank: int
    tradingview_link: str
    candle_close_time: datetime
    tv_data: dict  # Extra TradingView indicators


def get_tradingview_link(symbol: str, timeframe: str) -> str:
    tv_symbol = symbol.replace("/", "")
    tf_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M"
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
        self.tv_cache: Dict[str, dict] = {}  # Cache TV data
        self.tv_cache_time: Dict[str, datetime] = {}
    
    def get_tv_symbol(self, symbol: str) -> str:
        """Convert CCXT symbol to TradingView symbol"""
        # BTC/USDT -> BTCUSDT
        return symbol.replace("/", "")
    
    def fetch_tv_data(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Fetch RSI and other indicators from TradingView"""
        if not HAS_TV_TA:
            return None
        
        tv_interval = TV_INTERVALS.get(timeframe)
        if not tv_interval:
            return None
        
        # Check cache (valid for 60 seconds)
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.tv_cache:
            cache_age = (datetime.now() - self.tv_cache_time.get(cache_key, datetime.min)).total_seconds()
            if cache_age < 60:
                return self.tv_cache[cache_key]
        
        try:
            tv_symbol = self.get_tv_symbol(symbol)
            
            handler = TA_Handler(
                symbol=tv_symbol,
                exchange="BINANCE",
                screener="crypto",
                interval=tv_interval
            )
            
            analysis = handler.get_analysis()
            
            data = {
                "rsi": analysis.indicators.get("RSI", None),
                "rsi_prev": analysis.indicators.get("RSI[1]", None),  # Previous candle RSI
                "macd": analysis.indicators.get("MACD.macd", None),
                "macd_signal": analysis.indicators.get("MACD.signal", None),
                "stoch_k": analysis.indicators.get("Stoch.K", None),
                "stoch_d": analysis.indicators.get("Stoch.D", None),
                "ema20": analysis.indicators.get("EMA20", None),
                "ema50": analysis.indicators.get("EMA50", None),
                "bb_upper": analysis.indicators.get("BB.upper", None),
                "bb_lower": analysis.indicators.get("BB.lower", None),
                "atr": analysis.indicators.get("ATR", None),
                "adx": analysis.indicators.get("ADX", None),
                "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
                "buy_signals": analysis.summary.get("BUY", 0),
                "sell_signals": analysis.summary.get("SELL", 0),
                "neutral_signals": analysis.summary.get("NEUTRAL", 0),
            }
            
            # Cache it
            self.tv_cache[cache_key] = data
            self.tv_cache_time[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            print(f"[TV-TA] Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        """Fetch top coins by 24h trading volume"""
        if (self.symbols_cache_time and 
            (datetime.now() - self.symbols_cache_time).total_seconds() < 300 and
            self.symbols_cache):
            return self.symbols_cache
        
        try:
            print(f"[{format_sl_time()}] Fetching top {count} coins...")
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
                    usdt_pairs.append({'symbol': symbol, 'volume': quote_volume})
            
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top_symbols = [p['symbol'] for p in usdt_pairs[:count]]
            self.volume_ranks = {p['symbol']: i+1 for i, p in enumerate(usdt_pairs[:count])}
            
            self.symbols_cache = top_symbols
            self.symbols_cache_time = datetime.now()
            
            print(f"[{format_sl_time()}] Loaded {len(top_symbols)} coins")
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] Error fetching coins: {e}")
            if self.symbols_cache:
                return self.symbols_cache
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    
    def get_symbols_to_scan(self) -> List[str]:
        if SYMBOLS and len(SYMBOLS) > 0:
            return SYMBOLS
        return self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    def fetch_ohlcv_with_tv_rsi(self, symbol: str, timeframe: str, limit: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV from CCXT, then get RSI from TradingView
        TradingView only gives current RSI, so we calculate historical using fallback
        but validate current candle with TV RSI
        """
        try:
            fetch_limit = limit or LOOKBACK_CANDLES + 20
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Remove last candle (still forming)
            if len(df) > 1:
                df = df.iloc[:-1].copy()
            
            # Calculate RSI using our method for historical
            df['rsi'] = calculate_rsi_fallback(df['close'], RSI_PERIOD)
            
            # Get TradingView RSI for current/latest candle
            tv_data = self.fetch_tv_data(symbol, timeframe)
            if tv_data and tv_data.get('rsi') is not None:
                # Update the last candle's RSI with TradingView value
                df.loc[df.index[-1], 'rsi'] = tv_data['rsi']
                df.attrs['tv_data'] = tv_data  # Store TV data in dataframe
                
                # Log comparison
                our_rsi = calculate_rsi_fallback(df['close'], RSI_PERIOD).iloc[-1]
                tv_rsi = tv_data['rsi']
                diff = abs(our_rsi - tv_rsi) if pd.notna(our_rsi) else 0
                if diff > 2:
                    print(f"[RSI] {symbol} {timeframe}: Ours={our_rsi:.1f} TV={tv_rsi:.1f} (diff={diff:.1f})")
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_major_swing_highs(self, df: pd.DataFrame) -> List[MajorSwing]:
        """Find major swing highs"""
        swings = []
        n = SWING_STRENGTH_BARS
        
        for i in range(n, len(df) - n):
            current_high = df['high'].iloc[i]
            current_close = df['close'].iloc[i]
            
            window_highs = df['high'].iloc[i-n:i+n+1]
            if current_high != window_highs.max():
                continue
            
            surrounding_lows = df['low'].iloc[i-n:i+n+1].min()
            swing_strength = ((current_high - surrounding_lows) / surrounding_lows) * 100
            
            if swing_strength < MIN_PRICE_MOVE_PCT:
                continue
            
            rsi_val = df['rsi'].iloc[i]
            if pd.isna(rsi_val):
                continue
            
            swings.append(MajorSwing(
                index=i,
                price=current_close,
                rsi=rsi_val,
                is_high=True,
                timestamp=df['timestamp'].iloc[i],
                strength=swing_strength
            ))
        
        return swings
    
    def find_major_swing_lows(self, df: pd.DataFrame) -> List[MajorSwing]:
        """Find major swing lows"""
        swings = []
        n = SWING_STRENGTH_BARS
        
        for i in range(n, len(df) - n):
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            
            window_lows = df['low'].iloc[i-n:i+n+1]
            if current_low != window_lows.min():
                continue
            
            surrounding_highs = df['high'].iloc[i-n:i+n+1].max()
            swing_strength = ((surrounding_highs - current_low) / current_low) * 100
            
            if swing_strength < MIN_PRICE_MOVE_PCT:
                continue
            
            rsi_val = df['rsi'].iloc[i]
            if pd.isna(rsi_val):
                continue
            
            swings.append(MajorSwing(
                index=i,
                price=current_close,
                rsi=rsi_val,
                is_high=False,
                timestamp=df['timestamp'].iloc[i],
                strength=swing_strength
            ))
        
        return swings
    
    def filter_significant_swings(self, swings: List[MajorSwing], current_idx: int) -> List[MajorSwing]:
        """Filter swings by distance and lookback"""
        if not swings:
            return []
        
        valid_swings = [s for s in swings if current_idx - LOOKBACK_CANDLES <= s.index < current_idx - 2]
        
        if len(valid_swings) < 2:
            return valid_swings
        
        valid_swings.sort(key=lambda x: x.index)
        
        filtered = [valid_swings[0]]
        for swing in valid_swings[1:]:
            if swing.index - filtered[-1].index >= MIN_SWING_DISTANCE:
                filtered.append(swing)
        
        return filtered[-3:] if len(filtered) > 3 else filtered
    
    def detect_divergence(self, df: pd.DataFrame, idx: int, 
                          swing_lows: List[MajorSwing], 
                          swing_highs: List[MajorSwing],
                          symbol: str, timeframe: str,
                          tv_data: dict = None) -> Optional[DivergenceSignal]:
        """Detect divergence"""
        current_price = df['close'].iloc[idx]
        current_rsi = df['rsi'].iloc[idx]
        tv_recommendation = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
        
        # Check BULLISH divergence
        valid_lows = self.filter_significant_swings(swing_lows, idx)
        if len(valid_lows) >= 2:
            swing1, swing2 = valid_lows[-2], valid_lows[-1]
            
            price_pct = ((swing2.price - swing1.price) / swing1.price) * 100
            rsi_change = swing2.rsi - swing1.rsi
            candles_apart = swing2.index - swing1.index
            
            div_type = None
            confidence = 0.0
            
            # Bullish Regular: Price LL, RSI HL
            if swing2.price < swing1.price and swing2.rsi > swing1.rsi:
                div_type = DivergenceType.BULLISH_REGULAR
                confidence = min(0.9, 0.6 + abs(price_pct) * 0.02 + abs(rsi_change) * 0.01)
                
                # Boost confidence if TV agrees
                if tv_recommendation in ["BUY", "STRONG_BUY"]:
                    confidence = min(0.95, confidence + 0.05)
            
            # Bullish Hidden: Price HL, RSI LL
            elif swing2.price > swing1.price and swing2.rsi < swing1.rsi:
                div_type = DivergenceType.BULLISH_HIDDEN
                confidence = min(0.85, 0.55 + abs(price_pct) * 0.02 + abs(rsi_change) * 0.01)
            
            if div_type:
                return DivergenceSignal(
                    symbol=symbol, timeframe=timeframe, divergence_type=div_type,
                    swing1=swing1, swing2=swing2, current_price=current_price,
                    current_rsi=current_rsi, price_change_pct=price_pct,
                    rsi_change=rsi_change, candles_apart=candles_apart,
                    confidence=confidence, tv_recommendation=tv_recommendation
                )
        
        # Check BEARISH divergence
        valid_highs = self.filter_significant_swings(swing_highs, idx)
        if len(valid_highs) >= 2:
            swing1, swing2 = valid_highs[-2], valid_highs[-1]
            
            price_pct = ((swing2.price - swing1.price) / swing1.price) * 100
            rsi_change = swing2.rsi - swing1.rsi
            candles_apart = swing2.index - swing1.index
            
            div_type = None
            confidence = 0.0
            
            # Bearish Regular: Price HH, RSI LH
            if swing2.price > swing1.price and swing2.rsi < swing1.rsi:
                div_type = DivergenceType.BEARISH_REGULAR
                confidence = min(0.9, 0.6 + abs(price_pct) * 0.02 + abs(rsi_change) * 0.01)
                
                if tv_recommendation in ["SELL", "STRONG_SELL"]:
                    confidence = min(0.95, confidence + 0.05)
            
            # Bearish Hidden: Price LH, RSI HH
            elif swing2.price < swing1.price and swing2.rsi > swing1.rsi:
                div_type = DivergenceType.BEARISH_HIDDEN
                confidence = min(0.85, 0.55 + abs(price_pct) * 0.02 + abs(rsi_change) * 0.01)
            
            if div_type:
                return DivergenceSignal(
                    symbol=symbol, timeframe=timeframe, divergence_type=div_type,
                    swing1=swing1, swing2=swing2, current_price=current_price,
                    current_rsi=current_rsi, price_change_pct=price_pct,
                    rsi_change=rsi_change, candles_apart=candles_apart,
                    confidence=confidence, tv_recommendation=tv_recommendation
                )
        
        return None
    
    def check_momentum(self, df: pd.DataFrame, is_bullish: bool, tv_data: dict = None) -> MomentumStatus:
        """Check momentum using TV RSI data"""
        if len(df) < 3:
            return MomentumStatus(False, "Neutral", [], False, "Neutral", 0.0)
        
        # Get RSI values
        rsi_current = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2]
        
        # If we have TV data with previous RSI, use it
        if tv_data and tv_data.get('rsi_prev') is not None:
            rsi_prev = tv_data['rsi_prev']
        
        rsi_values = [rsi_prev, rsi_current]
        rsi_values = [r for r in rsi_values if pd.notna(r)]
        
        price_values = [df['close'].iloc[-2], df['close'].iloc[-1]]
        price_change_pct = ((price_values[-1] - price_values[-2]) / price_values[-2]) * 100
        
        rsi_confirmed = False
        rsi_direction = "Neutral"
        
        if len(rsi_values) >= 2:
            if rsi_values[-1] > rsi_values[-2]:
                rsi_direction = "Rising ↗️"
                if is_bullish:
                    rsi_confirmed = True
            elif rsi_values[-1] < rsi_values[-2]:
                rsi_direction = "Falling ↘️"
                if not is_bullish:
                    rsi_confirmed = True
        
        price_confirmed = False
        price_direction = "Neutral"
        
        if price_values[-1] > price_values[-2]:
            price_direction = "Rising ↗️"
            if is_bullish:
                price_confirmed = True
        elif price_values[-1] < price_values[-2]:
            price_direction = "Falling ↘️"
            if not is_bullish:
                price_confirmed = True
        
        return MomentumStatus(
            rsi_confirmed=rsi_confirmed,
            rsi_direction=rsi_direction,
            rsi_values=[round(r, 1) for r in rsi_values],
            price_confirmed=price_confirmed,
            price_direction=price_direction,
            price_change_pct=round(price_change_pct, 2)
        )
    
    def determine_signal_strength(self, divergence: DivergenceSignal, 
                                   momentum: MomentumStatus,
                                   tv_data: dict = None) -> Tuple[SignalStrength, float]:
        """Determine signal strength with TV confirmation"""
        base_confidence = divergence.confidence
        
        # Extra boost if TV recommendation aligns
        is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
        tv_rec = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
        
        tv_agrees = False
        if is_bullish and tv_rec in ["BUY", "STRONG_BUY"]:
            tv_agrees = True
        elif not is_bullish and tv_rec in ["SELL", "STRONG_SELL"]:
            tv_agrees = True
        
        if momentum.rsi_confirmed and momentum.price_confirmed:
            if tv_agrees:
                return SignalStrength.STRONG, min(base_confidence + 0.15, 0.98)
            return SignalStrength.STRONG, min(base_confidence + 0.10, 0.95)
        elif momentum.rsi_confirmed or momentum.price_confirmed:
            if tv_agrees:
                return SignalStrength.STRONG, min(base_confidence + 0.10, 0.92)
            return SignalStrength.MEDIUM, min(base_confidence + 0.05, 0.85)
        else:
            if tv_agrees:
                return SignalStrength.MEDIUM, min(base_confidence + 0.05, 0.80)
            return SignalStrength.EARLY, min(base_confidence, 0.70)
    
    def _get_cooldown_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        key = self._get_cooldown_key(symbol, timeframe)
        if key in self.last_alerts:
            elapsed = (datetime.now() - self.last_alerts[key]).total_seconds()
            return elapsed < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        key = self._get_cooldown_key(symbol, timeframe)
        self.last_alerts[key] = datetime.now()
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
        """Scan a single symbol using TV-TA RSI"""
        alerts = []
        
        if self._is_on_cooldown(symbol, timeframe):
            return alerts
        
        df = self.fetch_ohlcv_with_tv_rsi(symbol, timeframe)
        if df is None or len(df) < LOOKBACK_CANDLES:
            return alerts
        
        # Get TV data
        tv_data = df.attrs.get('tv_data', None) or self.fetch_tv_data(symbol, timeframe)
        
        # Find swings
        swing_highs = self.find_major_swing_highs(df)
        swing_lows = self.find_major_swing_lows(df)
        
        idx = len(df) - 1
        current_price = df['close'].iloc[idx]
        candle_close_time = df['timestamp'].iloc[idx]
        volume_rank = self.volume_ranks.get(symbol, 999)
        
        # Detect divergence
        divergence = self.detect_divergence(df, idx, swing_lows, swing_highs, symbol, timeframe, tv_data)
        
        if divergence:
            is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
            momentum = self.check_momentum(df, is_bullish, tv_data)
            signal_strength, confidence = self.determine_signal_strength(divergence, momentum, tv_data)
            
            confirm_tf = TIMEFRAME_CONFIRMATION_MAP.get(timeframe, "1h")
            
            self._set_cooldown(symbol, timeframe)
            
            alerts.append(AlertSignal(
                symbol=symbol,
                signal_tf=timeframe,
                confirm_tf=confirm_tf,
                divergence=divergence,
                ms_confirmation=None,
                signal_strength=signal_strength,
                momentum=momentum,
                total_confidence=confidence,
                timestamp=get_sl_time(),
                volume_rank=volume_rank,
                tradingview_link=get_tradingview_link(symbol, timeframe),
                candle_close_time=candle_close_time,
                tv_data=tv_data or {}
            ))
        
        return alerts
    
    def scan_all(self, min_strength: SignalStrength = None) -> List[AlertSignal]:
        """Scan all symbols"""
        all_alerts = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] Scanning {len(symbols)} coins × {len(SCAN_TIMEFRAMES)} TFs")
        print(f"[{format_sl_time()}] Using TradingView RSI: {HAS_TV_TA}")
        
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
                        tv_status = f"TV:{alert.divergence.tv_recommendation}" if alert.divergence.tv_recommendation else ""
                        print(f"[{format_sl_time()}] {emoji} {alert.signal_strength.value.upper()}: {symbol} {timeframe} RSI={alert.divergence.current_rsi:.1f} {tv_status}")
                    
                    time.sleep(0.15)  # Slightly faster with caching
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
        all_alerts.sort(key=lambda x: (strength_order.get(x.signal_strength, 0), x.total_confidence), reverse=True)
        
        print(f"[{format_sl_time()}] Scan complete. Found {len(all_alerts)} signals.")
        return all_alerts


class AlertFormatter:
    DIV_NAMES = {
        DivergenceType.BULLISH_REGULAR: "📈 Bullish Regular",
        DivergenceType.BULLISH_HIDDEN: "📈 Bullish Hidden",
        DivergenceType.BEARISH_REGULAR: "📉 Bearish Regular",
        DivergenceType.BEARISH_HIDDEN: "📉 Bearish Hidden",
    }
    
    DIV_DESC = {
        DivergenceType.BULLISH_REGULAR: "Price: Lower Low | RSI: Higher Low",
        DivergenceType.BULLISH_HIDDEN: "Price: Higher Low | RSI: Lower Low",
        DivergenceType.BEARISH_REGULAR: "Price: Higher High | RSI: Lower High",
        DivergenceType.BEARISH_HIDDEN: "Price: Lower High | RSI: Higher High",
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
        is_bull = "BULLISH" in signal.divergence.divergence_type.value.upper()
        
        if signal.signal_strength == SignalStrength.STRONG:
            strength_emoji = "🟢"
            strength_label = "STRONG"
        elif signal.signal_strength == SignalStrength.MEDIUM:
            strength_emoji = "🟡"
            strength_label = "MEDIUM"
        else:
            strength_emoji = "🔴"
            strength_label = "EARLY"
        
        direction = "BULLISH 📈" if is_bull else "BEARISH 📉"
        trade = "LONG" if is_bull else "SHORT"
        
        entry = signal.divergence.current_price
        if is_bull:
            stop = min(signal.divergence.swing1.price, signal.divergence.swing2.price) * 0.99
            target = entry * 1.04
        else:
            stop = max(signal.divergence.swing1.price, signal.divergence.swing2.price) * 1.01
            target = entry * 0.96
        
        rsi_emoji = "✅" if signal.momentum.rsi_confirmed else "⏳"
        price_emoji = "✅" if signal.momentum.price_confirmed else "⏳"
        
        candle_time = signal.candle_close_time
        if candle_time.tzinfo is None:
            candle_time = pytz.utc.localize(candle_time).astimezone(SL_TZ)
        candle_time_str = candle_time.strftime('%Y-%m-%d %H:%M IST')
        
        div = signal.divergence
        tv = signal.tv_data
        
        # TV Recommendation styling
        tv_rec = div.tv_recommendation
        if tv_rec in ["STRONG_BUY", "BUY"]:
            tv_emoji = "🟢"
        elif tv_rec in ["STRONG_SELL", "SELL"]:
            tv_emoji = "🔴"
        else:
            tv_emoji = "⚪"
        
        return f"""{strength_emoji} {strength_label} SIGNAL - {direction}

📊 {signal.symbol} (Vol Rank #{signal.volume_rank})
⏰ Timeframe: {signal.signal_tf.upper()}
🕐 Candle Close: {candle_time_str}

{'━'*30}
📈 DIVERGENCE DETECTED
{'━'*30}
{cls.DIV_NAMES[div.divergence_type]}
{cls.DIV_DESC[div.divergence_type]}

Swing 1: {cls.fmt_price(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {cls.fmt_price(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Current: {cls.fmt_price(div.current_price)} (RSI: {div.current_rsi:.1f})

📏 Distance: {div.candles_apart} candles
📊 Price: {div.price_change_pct:+.2f}% | RSI: {div.rsi_change:+.1f}

{'━'*30}
📺 TRADINGVIEW DATA
{'━'*30}
{tv_emoji} Recommendation: {tv_rec}
RSI: {tv.get('rsi', 'N/A'):.1f if tv.get('rsi') else 'N/A'}
MACD: {tv.get('macd', 'N/A'):.2f if tv.get('macd') else 'N/A'}
Stoch K: {tv.get('stoch_k', 'N/A'):.1f if tv.get('stoch_k') else 'N/A'}

{'━'*30}
✅ MOMENTUM CHECK
{'━'*30}
{rsi_emoji} RSI: {signal.momentum.rsi_direction}
{price_emoji} Price: {signal.momentum.price_direction} ({signal.momentum.price_change_pct:+.2f}%)

{'━'*30}
🎯 TRADE IDEA ({trade})
{'━'*30}
Entry: {cls.fmt_price(entry)}
Stop Loss: {cls.fmt_price(stop)}
Target: {cls.fmt_price(target)}

🔥 Confidence: {signal.total_confidence * 100:.0f}%

📺 {signal.tradingview_link}

⚠️ DYOR - Not financial advice!
🇱🇰 {format_sl_time(signal.timestamp)}"""
