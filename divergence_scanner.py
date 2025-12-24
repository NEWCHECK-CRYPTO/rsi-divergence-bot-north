"""
V9 Complete Divergence Scanner with AlertFormatter
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

from config import (
    EXCHANGE, SYMBOLS, SCAN_TIMEFRAMES,
    TIMEFRAME_CONFIRMATION_MAP, RSI_PERIOD,
    ALERT_COOLDOWN, TOP_COINS_COUNT, QUOTE_CURRENCY, 
    EXCLUDED_SYMBOLS, EXCLUDE_LEVERAGED, TIMEZONE,
    LOOKBACK_CANDLES, MIN_SWING_DISTANCE, MIN_PRICE_MOVE_PCT,
    SWING_STRENGTH_BARS
)

SL_TZ = pytz.timezone(TIMEZONE)
MIN_STRENGTH_RATIO = 0.30

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

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
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
    current_rsi: float
    price_change_pct: float
    rsi_change: float
    candles_apart: int
    confidence: float
    tv_recommendation: str

@dataclass
class ConfirmationStatus:
    candles_checked: int
    rsi_rising_count: int
    price_rising_count: int
    volume_rising_count: int
    is_confirmed: bool
    rsi_values: List[float]
    price_values: List[float]
    volume_values: List[float]

@dataclass
class MomentumStatus:
    rsi_confirmed: bool
    rsi_direction: str
    rsi_values: List[float]
    price_confirmed: bool
    price_direction: str
    price_change_pct: float
    volume_confirmed: bool
    volume_direction: str
    volume_change_pct: float

@dataclass
class AlertSignal:
    symbol: str
    signal_tf: str
    confirm_tf: str
    divergence: DivergenceSignal
    ms_confirmation: Optional[dict]
    signal_strength: SignalStrength
    momentum: MomentumStatus
    confirmation: ConfirmationStatus
    total_confidence: float
    timestamp: datetime
    volume_rank: int
    tradingview_link: str
    candle_close_time: datetime
    tv_data: dict

def get_tradingview_link(symbol: str, timeframe: str) -> str:
    tv_symbol = symbol.replace("/", "")
    tf_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M"
    }
    tv_tf = tf_map.get(timeframe, "60")
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{tv_symbol}.P&interval={tv_tf}"

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
        self.tv_cache: Dict[str, dict] = {}
        self.tv_cache_time: Dict[str, datetime] = {}
        
        print(f"[{format_sl_time()}] V9 Scanner initialized ({EXCHANGE.upper()})")
    
    def get_tv_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "")
    
    def fetch_tv_data(self, symbol: str, timeframe: str) -> Optional[dict]:
        if not HAS_TV_TA:
            return None
        
        tv_interval = TV_INTERVALS.get(timeframe)
        if not tv_interval:
            return None
        
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.tv_cache:
            cache_age = (datetime.now() - self.tv_cache_time.get(cache_key, datetime.min)).total_seconds()
            if cache_age < 60:
                return self.tv_cache[cache_key]
        
        try:
            tv_symbol = self.get_tv_symbol(symbol)
            
            for exchange in ["BYBIT", "BINANCE"]:
                try:
                    handler = TA_Handler(
                        symbol=tv_symbol,
                        exchange=exchange,
                        screener="crypto",
                        interval=tv_interval
                    )
                    analysis = handler.get_analysis()
                    
                    data = {
                        "rsi": analysis.indicators.get("RSI", None),
                        "rsi_prev": analysis.indicators.get("RSI[1]", None),
                        "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
                        "buy_signals": analysis.summary.get("BUY", 0),
                        "sell_signals": analysis.summary.get("SELL", 0),
                    }
                    
                    self.tv_cache[cache_key] = data
                    self.tv_cache_time[cache_key] = datetime.now()
                    return data
                    
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def fetch_top_coins_by_volume(self, count: int = 100) -> List[str]:
        if (self.symbols_cache_time and 
            (datetime.now() - self.symbols_cache_time).total_seconds() < 300 and
            self.symbols_cache):
            return self.symbols_cache
        
        try:
            print(f"[{format_sl_time()}] Fetching top {count} coins from {EXCHANGE}...")
            tickers = self.exchange.fetch_tickers()
            
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith(f'/{QUOTE_CURRENCY}'):
                    continue
                if symbol in EXCLUDED_SYMBOLS:
                    continue
                if EXCLUDE_LEVERAGED:
                    base = symbol.split('/')[0]
                    if any(x in base.upper() for x in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S', '2X', '3X']):
                        continue
                
                quote_volume = ticker.get('quoteVolume', 0) or 0
                if quote_volume > 0:
                    usdt_pairs.append({'symbol': symbol, 'volume': quote_volume})
            
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top_symbols = [p['symbol'] for p in usdt_pairs[:count]]
            
            self.volume_ranks = {s: i+1 for i, s in enumerate(top_symbols)}
            self.symbols_cache = top_symbols
            self.symbols_cache_time = datetime.now()
            
            print(f"[{format_sl_time()}] Loaded {len(top_symbols)} symbols")
            return top_symbols
            
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return []
    
    def get_symbols_to_scan(self) -> List[str]:
        if SYMBOLS:
            return SYMBOLS
        
        now = datetime.now()
        if self.symbols_cache and self.symbols_cache_time:
            age = (now - self.symbols_cache_time).total_seconds()
            if age < 3600:
                return self.symbols_cache
        
        symbols = self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
        return symbols
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 20:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            tv_data = self.fetch_tv_data(symbol, timeframe)
            if tv_data and tv_data.get('rsi') is not None:
                df.loc[df.index[-1], 'rsi'] = tv_data['rsi']
            
            df.attrs['tv_data'] = tv_data
            return df
            
        except Exception as e:
            print(f"[{format_sl_time()}] Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_major_swing_highs(self, df: pd.DataFrame) -> List[MajorSwing]:
        swings = []
        n = SWING_STRENGTH_BARS
        
        for i in range(n, len(df) - n):
            current_high = df['high'].iloc[i]
            
            window_highs = df['high'].iloc[i-n:i+n+1]
            if current_high != window_highs.max():
                continue
            
            surrounding_lows = df['low'].iloc[i-n:i+n+1].min()
            price_move_pct = ((current_high - surrounding_lows) / surrounding_lows) * 100
            
            if price_move_pct < MIN_PRICE_MOVE_PCT:
                continue
            
            rsi_val = df['rsi'].iloc[i]
            if pd.isna(rsi_val):
                continue
            
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            volume = df['volume'].iloc[i]
            strength = candle_range * volume
            
            swings.append(MajorSwing(
                index=i,
                price=df['close'].iloc[i],
                rsi=rsi_val,
                is_high=True,
                timestamp=df['timestamp'].iloc[i],
                strength=strength
            ))
        
        return swings
    
    def find_major_swing_lows(self, df: pd.DataFrame) -> List[MajorSwing]:
        swings = []
        n = SWING_STRENGTH_BARS
        
        for i in range(n, len(df) - n):
            current_low = df['low'].iloc[i]
            
            window_lows = df['low'].iloc[i-n:i+n+1]
            if current_low != window_lows.min():
                continue
            
            surrounding_highs = df['high'].iloc[i-n:i+n+1].max()
            price_move_pct = ((surrounding_highs - current_low) / current_low) * 100
            
            if price_move_pct < MIN_PRICE_MOVE_PCT:
                continue
            
            rsi_val = df['rsi'].iloc[i]
            if pd.isna(rsi_val):
                continue
            
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            volume = df['volume'].iloc[i]
            strength = candle_range * volume
            
            swings.append(MajorSwing(
                index=i,
                price=df['close'].iloc[i],
                rsi=rsi_val,
                is_high=False,
                timestamp=df['timestamp'].iloc[i],
                strength=strength
            ))
        
        return swings
    
    def filter_significant_swings(self, swings: List[MajorSwing], current_idx: int) -> List[MajorSwing]:
        if not swings:
            return []
        
        filtered = []
        for swing in swings:
            if current_idx - swing.index > LOOKBACK_CANDLES:
                continue
            if current_idx - swing.index < MIN_SWING_DISTANCE:
                continue
            filtered.append(swing)
        
        if len(filtered) < 2:
            return filtered
        
        final = []
        for i, swing in enumerate(filtered):
            if i == 0:
                final.append(swing)
                continue
            
            prev_swing = final[-1]
            distance = swing.index - prev_swing.index
            
            if distance >= MIN_SWING_DISTANCE:
                price_diff = abs(swing.price - prev_swing.price) / prev_swing.price * 100
                if price_diff >= MIN_PRICE_MOVE_PCT:
                    final.append(swing)
        
        return final
    
    def _validate_bearish_regular(self, peak_a: MajorSwing, peak_b: MajorSwing, 
                                    all_peaks: List[MajorSwing]) -> Tuple[bool, str]:
        middles = [p for p in all_peaks if peak_a.index < p.index < peak_b.index]
        if not middles:
            return True, "No middles"
        
        avg_strength = (peak_a.strength + peak_b.strength) / 2
        sig_middles = [m for m in middles if m.strength >= avg_strength * MIN_STRENGTH_RATIO]
        if not sig_middles:
            return True, f"{len(middles)} middles too weak"
        
        for m in sig_middles:
            if m.price > peak_b.price:
                return False, f"Middle ${m.price:.4f} > Peak B ${peak_b.price:.4f}"
        
        for m in sig_middles:
            if m.rsi > peak_a.rsi:
                return False, f"Middle RSI {m.rsi:.1f} > Peak A RSI {peak_a.rsi:.1f}"
        
        return True, f"{len(sig_middles)} middles OK"
    
    def _validate_bullish_regular(self, trough_a: MajorSwing, trough_b: MajorSwing,
                                    all_troughs: List[MajorSwing]) -> Tuple[bool, str]:
        middles = [t for t in all_troughs if trough_a.index < t.index < trough_b.index]
        if not middles:
            return True, "No middles"
        
        avg_strength = (trough_a.strength + trough_b.strength) / 2
        sig_middles = [m for m in middles if m.strength >= avg_strength * MIN_STRENGTH_RATIO]
        if not sig_middles:
            return True, f"{len(middles)} middles too weak"
        
        for m in sig_middles:
            if m.price < trough_b.price:
                return False, f"Middle ${m.price:.4f} < Trough B ${trough_b.price:.4f}"
        
        for m in sig_middles:
            if m.rsi < trough_a.rsi:
                return False, f"Middle RSI {m.rsi:.1f} < Trough A RSI {trough_a.rsi:.1f}"
        
        return True, f"{len(sig_middles)} middles OK"
    
    def _validate_bearish_hidden(self, peak_a: MajorSwing, peak_b: MajorSwing,
                                   all_peaks: List[MajorSwing]) -> Tuple[bool, str]:
        middles = [p for p in all_peaks if peak_a.index < p.index < peak_b.index]
        if not middles:
            return True, "No middles"
        
        avg_strength = (peak_a.strength + peak_b.strength) / 2
        sig_middles = [m for m in middles if m.strength >= avg_strength * MIN_STRENGTH_RATIO]
        if not sig_middles:
            return True, f"{len(middles)} middles too weak"
        
        for m in sig_middles:
            if m.price > peak_a.price:
                return False, f"Middle ${m.price:.4f} > Peak A ${peak_a.price:.4f}"
            if m.price < peak_b.price:
                return False, f"Middle ${m.price:.4f} < Peak B ${peak_b.price:.4f}"
        
        for m in sig_middles:
            if m.rsi < peak_a.rsi:
                return False, f"Middle RSI {m.rsi:.1f} < Peak A RSI {peak_a.rsi:.1f}"
            if m.rsi > peak_b.rsi:
                return False, f"Middle RSI {m.rsi:.1f} > Peak B RSI {peak_b.rsi:.1f}"
        
        return True, f"{len(sig_middles)} middles OK"
    
    def _validate_bullish_hidden(self, trough_a: MajorSwing, trough_b: MajorSwing,
                                   all_troughs: List[MajorSwing]) -> Tuple[bool, str]:
        middles = [t for t in all_troughs if trough_a.index < t.index < trough_b.index]
        if not middles:
            return True, "No middles"
        
        avg_strength = (trough_a.strength + trough_b.strength) / 2
        sig_middles = [m for m in middles if m.strength >= avg_strength * MIN_STRENGTH_RATIO]
        if not sig_middles:
            return True, f"{len(middles)} middles too weak"
        
        for m in sig_middles:
            if m.price < trough_a.price:
                return False, f"Middle ${m.price:.4f} < Trough A ${trough_a.price:.4f}"
            if m.price > trough_b.price:
                return False, f"Middle ${m.price:.4f} > Trough B ${trough_b.price:.4f}"
        
        for m in sig_middles:
            if m.rsi > trough_a.rsi:
                return False, f"Middle RSI {m.rsi:.1f} > Trough A RSI {trough_a.rsi:.1f}"
            if m.rsi < trough_b.rsi:
                return False, f"Middle RSI {m.rsi:.1f} < Trough B RSI {trough_b.rsi:.1f}"
        
        return True, f"{len(sig_middles)} middles OK"
    
    def detect_divergence(self, df: pd.DataFrame, idx: int, swing_lows: List[MajorSwing],
                          swing_highs: List[MajorSwing], symbol: str, timeframe: str,
                          tv_data: dict = None) -> Optional[DivergenceSignal]:
        current_price = df['close'].iloc[idx]
        current_rsi = df['rsi'].iloc[idx]
        
        if pd.isna(current_rsi):
            return None
        
        valid_lows = self.filter_significant_swings(swing_lows, idx)
        valid_highs = self.filter_significant_swings(swing_highs, idx)
        
        if len(valid_lows) >= 2:
            p1, p2 = valid_lows[-2], valid_lows[-1]
            
            price_change_pct = ((p2.price - p1.price) / p1.price) * 100
            rsi_change = p2.rsi - p1.rsi
            candles_apart = p2.index - p1.index
            
            if p2.price < p1.price and p2.rsi > p1.rsi:
                is_valid, reason = self._validate_bullish_regular(p1, p2, valid_lows)
                
                if not is_valid:
                    print(f"[{symbol} {timeframe}] Bullish Regular INVALID: {reason}")
                else:
                    confidence = min(0.75 + (abs(price_change_pct) / 10) * 0.1 + (rsi_change / 20) * 0.1, 0.95)
                    tv_rec = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
                    
                    return DivergenceSignal(
                        symbol=symbol, timeframe=timeframe,
                        divergence_type=DivergenceType.BULLISH_REGULAR,
                        swing1=p1, swing2=p2,
                        current_price=current_price, current_rsi=current_rsi,
                        price_change_pct=price_change_pct, rsi_change=rsi_change,
                        candles_apart=candles_apart, confidence=confidence,
                        tv_recommendation=tv_rec
                    )
            
            if p2.price > p1.price and p2.rsi < p1.rsi:
                is_valid, reason = self._validate_bullish_hidden(p1, p2, valid_lows)
                
                if not is_valid:
                    print(f"[{symbol} {timeframe}] Bullish Hidden INVALID: {reason}")
                else:
                    confidence = min(0.65 + (abs(price_change_pct) / 10) * 0.1 + (abs(rsi_change) / 20) * 0.05, 0.85)
                    tv_rec = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
                    
                    return DivergenceSignal(
                        symbol=symbol, timeframe=timeframe,
                        divergence_type=DivergenceType.BULLISH_HIDDEN,
                        swing1=p1, swing2=p2,
                        current_price=current_price, current_rsi=current_rsi,
                        price_change_pct=price_change_pct, rsi_change=rsi_change,
                        candles_apart=candles_apart, confidence=confidence,
                        tv_recommendation=tv_rec
                    )
        
        if len(valid_highs) >= 2:
            p1, p2 = valid_highs[-2], valid_highs[-1]
            
            price_change_pct = ((p2.price - p1.price) / p1.price) * 100
            rsi_change = p2.rsi - p1.rsi
            candles_apart = p2.index - p1.index
            
            if p2.price > p1.price and p2.rsi < p1.rsi:
                is_valid, reason = self._validate_bearish_regular(p1, p2, valid_highs)
                
                if not is_valid:
                    print(f"[{symbol} {timeframe}] Bearish Regular INVALID: {reason}")
                else:
                    confidence = min(0.75 + (abs(price_change_pct) / 10) * 0.1 + (abs(rsi_change) / 20) * 0.1, 0.95)
                    tv_rec = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
                    
                    return DivergenceSignal(
                        symbol=symbol, timeframe=timeframe,
                        divergence_type=DivergenceType.BEARISH_REGULAR,
                        swing1=p1, swing2=p2,
                        current_price=current_price, current_rsi=current_rsi,
                        price_change_pct=price_change_pct, rsi_change=rsi_change,
                        candles_apart=candles_apart, confidence=confidence,
                        tv_recommendation=tv_rec
                    )
            
            if p2.price < p1.price and p2.rsi > p1.rsi:
                is_valid, reason = self._validate_bearish_hidden(p1, p2, valid_highs)
                
                if not is_valid:
                    print(f"[{symbol} {timeframe}] Bearish Hidden INVALID: {reason}")
                else:
                    confidence = min(0.65 + (abs(price_change_pct) / 10) * 0.1 + (rsi_change / 20) * 0.05, 0.85)
                    tv_rec = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
                    
                    return DivergenceSignal(
                        symbol=symbol, timeframe=timeframe,
                        divergence_type=DivergenceType.BEARISH_HIDDEN,
                        swing1=p1, swing2=p2,
                        current_price=current_price, current_rsi=current_rsi,
                        price_change_pct=price_change_pct, rsi_change=rsi_change,
                        candles_apart=candles_apart, confidence=confidence,
                        tv_recommendation=tv_rec
                    )
        
        return None
    
    def check_3_candle_confirmation(self, df: pd.DataFrame, is_bullish: bool, 
                                     swing2_idx: int) -> ConfirmationStatus:
        start_idx = swing2_idx + 1
        end_idx = swing2_idx + 4
        
        if end_idx > len(df):
            return ConfirmationStatus(
                candles_checked=0, rsi_rising_count=0,
                price_rising_count=0, volume_rising_count=0,
                is_confirmed=False, rsi_values=[], price_values=[], volume_values=[]
            )
        
        rsi_values = df['rsi'].iloc[start_idx:end_idx].tolist()
        price_values = df['close'].iloc[start_idx:end_idx].tolist()
        volume_values = df['volume'].iloc[start_idx:end_idx].tolist()
        
        rsi_rising_count = price_rising_count = volume_rising_count = 0
        
        for i in range(len(rsi_values) - 1):
            if is_bullish:
                if rsi_values[i+1] > rsi_values[i]: rsi_rising_count += 1
                if price_values[i+1] > price_values[i]: price_rising_count += 1
                if volume_values[i+1] > volume_values[i]: volume_rising_count += 1
            else:
                if rsi_values[i+1] < rsi_values[i]: rsi_rising_count += 1
                if price_values[i+1] < price_values[i]: price_rising_count += 1
                if volume_values[i+1] > volume_values[i]: volume_rising_count += 1
        
        is_confirmed = (rsi_rising_count >= 2 and price_rising_count >= 2 and volume_rising_count >= 2)
        
        return ConfirmationStatus(
            candles_checked=3, rsi_rising_count=rsi_rising_count,
            price_rising_count=price_rising_count, volume_rising_count=volume_rising_count,
            is_confirmed=is_confirmed, rsi_values=rsi_values,
            price_values=price_values, volume_values=volume_values
        )
    
    def check_momentum(self, df: pd.DataFrame, is_bullish: bool, tv_data: dict = None) -> MomentumStatus:
        if len(df) < 3:
            return MomentumStatus(
                rsi_confirmed=False, rsi_direction="Unknown", rsi_values=[],
                price_confirmed=False, price_direction="Unknown", price_change_pct=0,
                volume_confirmed=False, volume_direction="Unknown", volume_change_pct=0
            )
        
        rsi_values = df['rsi'].iloc[-3:].tolist()
        price_values = df['close'].iloc[-3:].tolist()
        volume_values = df['volume'].iloc[-3:].tolist()
        
        rsi_rising = rsi_values[-1] > rsi_values[-2] > rsi_values[-3]
        rsi_falling = rsi_values[-1] < rsi_values[-2] < rsi_values[-3]
        
        if rsi_rising:
            rsi_direction = "Rising ↗️"
            rsi_confirmed = is_bullish
        elif rsi_falling:
            rsi_direction = "Falling ↘️"
            rsi_confirmed = not is_bullish
        else:
            rsi_direction = "Sideways ↔️"
            rsi_confirmed = False
        
        price_rising = price_values[-1] > price_values[-2] > price_values[-3]
        price_falling = price_values[-1] < price_values[-2] < price_values[-3]
        price_change_pct = ((price_values[-1] - price_values[-3]) / price_values[-3]) * 100
        
        if price_rising:
            price_direction = "Rising ↗️"
            price_confirmed = is_bullish
        elif price_falling:
            price_direction = "Falling ↘️"
            price_confirmed = not is_bullish
        else:
            price_direction = "Sideways ↔️"
            price_confirmed = False
        
        volume_rising = volume_values[-1] > volume_values[-2] > volume_values[-3]
        volume_falling = volume_values[-1] < volume_values[-2] < volume_values[-3]
        volume_change_pct = ((volume_values[-1] - volume_values[-3]) / volume_values[-3]) * 100
        
        if volume_rising:
            volume_direction = "Rising ↗️"
            volume_confirmed = True
        elif volume_falling:
            volume_direction = "Falling ↘️"
            volume_confirmed = False
        else:
            volume_direction = "Sideways ↔️"
            volume_confirmed = False
        
        return MomentumStatus(
            rsi_confirmed=rsi_confirmed, rsi_direction=rsi_direction, rsi_values=rsi_values,
            price_confirmed=price_confirmed, price_direction=price_direction,
            price_change_pct=round(price_change_pct, 2),
            volume_confirmed=volume_confirmed, volume_direction=volume_direction,
            volume_change_pct=round(volume_change_pct, 2)
        )
    
    def determine_signal_strength(self, divergence: DivergenceSignal, 
                                   momentum: MomentumStatus,
                                   confirmation: ConfirmationStatus,
                                   tv_data: dict = None) -> Tuple[SignalStrength, float]:
        base_confidence = divergence.confidence
        
        is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
        tv_rec = tv_data.get('recommendation', 'NEUTRAL') if tv_data else 'NEUTRAL'
        
        tv_agrees = False
        if is_bullish and tv_rec in ["BUY", "STRONG_BUY"]:
            tv_agrees = True
        elif not is_bullish and tv_rec in ["SELL", "STRONG_SELL"]:
            tv_agrees = True
        
        if confirmation.is_confirmed:
            if momentum.rsi_confirmed and momentum.price_confirmed and momentum.volume_confirmed:
                if tv_agrees:
                    return SignalStrength.STRONG, min(base_confidence + 0.20, 0.98)
                return SignalStrength.STRONG, min(base_confidence + 0.15, 0.95)
            return SignalStrength.STRONG, min(base_confidence + 0.10, 0.90)
        
        elif momentum.rsi_confirmed and momentum.price_confirmed and momentum.volume_confirmed:
            if tv_agrees:
                return SignalStrength.MEDIUM, min(base_confidence + 0.10, 0.85)
            return SignalStrength.MEDIUM, min(base_confidence + 0.05, 0.80)
        
        else:
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
        alerts = []
        
        if self._is_on_cooldown(symbol, timeframe):
            return alerts
        
        df = self.fetch_ohlcv(symbol, timeframe, limit=250)
        if df is None or len(df) < 20:
            return alerts
        
        tv_data = df.attrs.get('tv_data', None)
        
        swing_highs = self.find_major_swing_highs(df)
        swing_lows = self.find_major_swing_lows(df)
        
        idx = len(df) - 1
        candle_close_time = df['timestamp'].iloc[idx]
        volume_rank = self.volume_ranks.get(symbol, 999)
        
        divergence = self.detect_divergence(df, idx, swing_lows, swing_highs, symbol, timeframe, tv_data)
        
        if divergence:
            is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
            
            confirmation = self.check_3_candle_confirmation(df, is_bullish, divergence.swing2.index)
            momentum = self.check_momentum(df, is_bullish, tv_data)
            signal_strength, confidence = self.determine_signal_strength(divergence, momentum, confirmation, tv_data)
            
            if confidence < 0.70:
                return alerts
            
            confirm_tf = TIMEFRAME_CONFIRMATION_MAP.get(timeframe, "1h")
            
            self._set_cooldown(symbol, timeframe)
            
            alerts.append(AlertSignal(
                symbol=symbol, signal_tf=timeframe, confirm_tf=confirm_tf,
                divergence=divergence, ms_confirmation=None,
                signal_strength=signal_strength, momentum=momentum,
                confirmation=confirmation, total_confidence=confidence,
                timestamp=get_sl_time(), volume_rank=volume_rank,
                tradingview_link=get_tradingview_link(symbol, timeframe),
                candle_close_time=candle_close_time, tv_data=tv_data or {}
            ))
        
        return alerts
    
    def scan_all(self, min_strength: SignalStrength = None) -> List[AlertSignal]:
        all_alerts = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] 🔍 Scanning {len(symbols)} coins × {len(SCAN_TIMEFRAMES)} TFs")
        
        scanned = 0
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
                        conf_emoji = "✅" if alert.confirmation.is_confirmed else "⏳"
                        print(f"[{format_sl_time()}] {emoji}{conf_emoji} {symbol} {timeframe}: {alert.divergence.divergence_type.value} ({alert.total_confidence*100:.0f}%)")
                    
                    scanned += 1
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
        all_alerts.sort(key=lambda x: (strength_order.get(x.signal_strength, 0), x.total_confidence), reverse=True)
        
        print(f"[{format_sl_time()}] ✅ Scanned {scanned} pairs. Found {len(all_alerts)} signals (70%+ confidence).")
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
        volume_emoji = "✅" if signal.momentum.volume_confirmed else "⏳"
        
        conf = signal.confirmation
        if conf.is_confirmed:
            conf_emoji = "✅"
            conf_text = f"3-Candle Confirmed! RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2 Vol:{conf.volume_rising_count}/2"
        elif conf.candles_checked > 0:
            conf_emoji = "⏳"
            conf_text = f"Waiting... RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2 Vol:{conf.volume_rising_count}/2"
        else:
            conf_emoji = "⏳"
            conf_text = "Waiting for confirmation candles..."
        
        candle_time = signal.candle_close_time
        if candle_time.tzinfo is None:
            candle_time = pytz.utc.localize(candle_time).astimezone(SL_TZ)
        candle_time_str = candle_time.strftime('%Y-%m-%d %H:%M IST')
        
        div = signal.divergence
        
        tv_rec = div.tv_recommendation
        if tv_rec in ["STRONG_BUY", "BUY"]:
            tv_emoji = "🟢"
        elif tv_rec in ["STRONG_SELL", "SELL"]:
            tv_emoji = "🔴"
        else:
            tv_emoji = "⚪"
        
        return f"""{strength_emoji} {strength_label} SIGNAL - {direction}

📊 {signal.symbol} (#{signal.volume_rank})
⏰ {signal.signal_tf.upper()} | {candle_time_str}

{'━'*28}
{cls.DIV_NAMES[div.divergence_type]}
{cls.DIV_DESC[div.divergence_type]}

Swing 1: {cls.fmt_price(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {cls.fmt_price(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Now: {cls.fmt_price(div.current_price)} (RSI: {div.current_rsi:.1f})

📏 {div.candles_apart} candles apart
{'━'*28}
{conf_emoji} {conf_text}
{'━'*28}
{tv_emoji} TradingView: {tv_rec}
{rsi_emoji} RSI: {signal.momentum.rsi_direction}
{price_emoji} Price: {signal.momentum.price_direction}
{volume_emoji} Volume: {signal.momentum.volume_direction} ({signal.momentum.volume_change_pct:+.1f}%)
{'━'*28}
🎯 {trade} | Entry: {cls.fmt_price(entry)}
🛑 SL: {cls.fmt_price(stop)} | 🎯 TP: {cls.fmt_price(target)}

🔥 Confidence: {signal.total_confidence * 100:.0f}%
📺 {signal.tradingview_link}

⚠️ DYOR | 🇱🇰 {format_sl_time(signal.timestamp)}"""
