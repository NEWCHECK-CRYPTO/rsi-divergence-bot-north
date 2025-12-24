"""
SIMPLIFIED Scanner - Back to Basics (Like Your Old Version)
Just fetch markets and scan - no fancy volume sorting
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import pytz
import time

from config import (
    EXCHANGE, SCAN_TIMEFRAMES, RSI_PERIOD, ALERT_COOLDOWN,
    QUOTE_CURRENCY, EXCLUDED_SYMBOLS, EXCLUDE_LEVERAGED, TIMEZONE,
    LOOKBACK_CANDLES, MIN_SWING_DISTANCE, SWING_STRENGTH_BARS,
    MIN_CONFIDENCE, MIN_ADX_STRONG, MIN_ADX_MODERATE,
    TREND_CONFIRMATION_MAP
)

SL_TZ = pytz.timezone(TIMEZONE)

# SIMPLE COIN LIST - Top coins that definitely exist
SIMPLE_COINS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "BCH/USDT", "XLM/USDT", "NEAR/USDT", "ALGO/USDT", "ICP/USDT",
    "FIL/USDT", "TRX/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
    "INJ/USDT", "SUI/USDT", "SEI/USDT", "HBAR/USDT", "GRT/USDT",
    "IMX/USDT", "RUNE/USDT", "FTM/USDT", "AAVE/USDT", "MKR/USDT",
    "SNX/USDT", "CRV/USDT", "LDO/USDT", "BLUR/USDT", "PEPE/USDT",
    "SHIB/USDT", "WLD/USDT", "ORDI/USDT", "STX/USDT", "TIA/USDT",
    "PYTH/USDT", "JUP/USDT", "ONDO/USDT", "METIS/USDT", "MINA/USDT",
    "ROSE/USDT", "EGLD/USDT", "FLOW/USDT", "XTZ/USDT", "SAND/USDT",
    "MANA/USDT", "AXS/USDT", "CHZ/USDT", "GALA/USDT", "APE/USDT",
    "GMT/USDT", "GAL/USDT", "LRC/USDT", "ZIL/USDT", "ONE/USDT",
    "ZEC/USDT", "DASH/USDT", "WAVES/USDT", "QTUM/USDT", "ZRX/USDT",
    "BAT/USDT", "OMG/USDT", "ENJ/USDT", "MEME/USDT", "FLOKI/USDT",
    "WEN/USDT", "BONK/USDT", "DYDX/USDT", "GMX/USDT", "RNDR/USDT",
    "FET/USDT", "AGIX/USDT", "OCEAN/USDT", "ANKR/USDT", "CKB/USDT",
    "CELO/USDT", "KAVA/USDT", "ALPHA/USDT", "YFI/USDT", "COMP/USDT",
    "SUSHI/USDT", "BAL/USDT", "SKL/USDT", "BNT/USDT", "KSM/USDT",
    "COTI/USDT", "JASMY/USDT", "RSR/USDT", "AUDIO/USDT", "CTSI/USDT"
]


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
    date_str: str  # Exact date for TradingView verification


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
    return datetime.now(SL_TZ)


def format_sl_time(dt=None):
    if dt is None:
        dt = get_sl_time()
    elif dt.tzinfo is None:
        dt = SL_TZ.localize(dt)
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


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
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
    exchange_map = {"bybit": "BYBIT", "binance": "BINANCE"}
    exchange_prefix = exchange_map.get(EXCHANGE.lower(), "BYBIT")
    clean_symbol = symbol.replace("/", "")
    
    tf_map = {"1h": "60", "4h": "240", "1d": "D"}
    tv_interval = tf_map.get(timeframe, "240")
    
    return f"https://www.tradingview.com/chart/?symbol={exchange_prefix}:{clean_symbol}.P&interval={tv_interval}"


class SimpleDivergenceScanner:
    """SIMPLE Scanner - Like Your Old Version"""
    
    def __init__(self):
        print(f"[{format_sl_time()}] 🔧 Initializing SIMPLE scanner...")
        
        # SIMPLE connection - no fancy options
        if EXCHANGE.lower() == "bybit":
            self.exchange = ccxt.bybit()
        elif EXCHANGE.lower() == "binance":
            self.exchange = ccxt.binance()
        else:
            raise ValueError(f"Unsupported exchange: {EXCHANGE}")
        
        self.exchange.enableRateLimit = True
        
        print(f"[{format_sl_time()}] ✅ Scanner ready ({EXCHANGE.upper()})")
        
        self.alert_cooldowns = {}
        self.symbols_to_scan = SIMPLE_COINS
        
        print(f"[{format_sl_time()}] 📊 Loaded {len(self.symbols_to_scan)} coins")
        print(f"[{format_sl_time()}] 🏆 Top 10: {self.symbols_to_scan[:10]}")
    
    def get_symbols_to_scan(self) -> List[str]:
        """Return simple coin list"""
        return self.symbols_to_scan
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            return df
            
        except Exception as e:
            return None
    
    def find_swing_highs(self, df: pd.DataFrame, strength: int = 3) -> List[SwingPoint]:
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
                    timestamp=df['timestamp'].iloc[i],
                    date_str=df['timestamp'].iloc[i].strftime('%Y-%m-%d')
                ))
        
        return swings
    
    def find_swing_lows(self, df: pd.DataFrame, strength: int = 3) -> List[SwingPoint]:
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
                    timestamp=df['timestamp'].iloc[i],
                    date_str=df['timestamp'].iloc[i].strftime('%Y-%m-%d')
                ))
        
        return swings
    
    def check_price_invalidation(self, df: pd.DataFrame, swing1: SwingPoint, 
                                 swing2: SwingPoint, is_bullish: bool) -> bool:
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
        candles_after_swing2 = current_idx - swing2_idx
        return candles_after_swing2 == 2
    
    def check_2_candle_confirmation(self, df: pd.DataFrame, is_bullish: bool,
                                    swing2_idx: int) -> ConfirmationStatus:
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
        adx = calculate_adx(df, period=14)
        
        if adx > MIN_ADX_STRONG:
            adx_direction = "Strong 💪"
        elif adx > MIN_ADX_MODERATE:
            adx_direction = "Moderate 👍"
        else:
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
        lower_tf = TREND_CONFIRMATION_MAP.get(signal_tf)
        if not lower_tf:
            return None
        
        df_lower = self.fetch_ohlcv(symbol, lower_tf, limit=50)
        if df_lower is None or len(df_lower) < 20:
            return None
        
        adx = calculate_adx(df_lower, period=14)
        
        if adx < MIN_ADX_MODERATE:
            is_confirmed = False
            confidence_boost = -0.10
        else:
            is_confirmed = True
            confidence_boost = +0.10
        
        return MTFTrendStatus(
            confirmation_tf=lower_tf,
            adx=adx,
            trend_direction="Confirmed" if is_confirmed else "Weak",
            price_trend="",
            rsi_trend="",
            is_confirmed=is_confirmed,
            confidence_boost=confidence_boost
        )
    
    def detect_divergence(self, df: pd.DataFrame, swing_lows: List[SwingPoint],
                          swing_highs: List[SwingPoint]) -> Optional[Divergence]:
        
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
        key = f"{symbol}_{timeframe}"
        if key in self.alert_cooldowns:
            elapsed = time.time() - self.alert_cooldowns[key]
            return elapsed < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        key = f"{symbol}_{timeframe}"
        self.alert_cooldowns[key] = time.time()
    
    def scan_symbol(self, symbol: str, timeframe: str) -> List[AlertSignal]:
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
        
        base_confidence = divergence.confidence + 0.10
        if mtf_trend:
            base_confidence += mtf_trend.confidence_boost
        
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
        
        # Get rank
        try:
            rank = self.symbols_to_scan.index(symbol) + 1
        except:
            rank = 999
        
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
            volume_rank=rank,
            tradingview_link=get_tradingview_link(symbol, timeframe),
            candle_close_time=df['timestamp'].iloc[-1],
            tv_data=None
        ))
        
        return alerts
    
    def scan_all(self) -> List[AlertSignal]:
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
                    pass
        
        return all_alerts


# Use simple scanner
DivergenceScanner = SimpleDivergenceScanner


class AlertFormatter:
    @staticmethod
    def format_alert(alert: AlertSignal) -> str:
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

Swing 1: {div.swing1.date_str} @ {fmt(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {div.swing2.date_str} @ {fmt(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Current: {fmt(div.current_price)} (RSI: {div.current_rsi:.1f})

🔍 {div.candles_apart} candles apart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 2-Candle Confirmed! RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2
⚡ JUST CONFIRMED (2nd candle closed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 {trade} | Entry: {fmt(entry)}
🛑 SL: {fmt(sl)} | 🎯 TP: {fmt(tp)}

🔥 Confidence: {alert.total_confidence * 100:.0f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ VERIFY BEFORE TRADING:
1. Open TradingView BYBIT:{alert.symbol.replace('/', '')}
2. Set to {alert.signal_tf.upper()} timeframe
3. Check {div.swing1.date_str} = {fmt(div.swing1.price)}
4. Check {div.swing2.date_str} = {fmt(div.swing2.price)}
5. Prices match (±$10)? → Trade ✅
6. Prices differ? → DO NOT TRADE ❌

📺 {alert.tradingview_link}

🇱🇰 {format_sl_time()}"""
        
        return msg
