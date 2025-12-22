"""
Divergence Scanner Module
Fetches top 100 coins by volume from Binance
All times in Sri Lanka timezone (Asia/Colombo)
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
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
    MIN_CONFIDENCE_THRESHOLD, ALERT_COOLDOWN,
    TOP_COINS_COUNT, QUOTE_CURRENCY, EXCLUDED_SYMBOLS,
    EXCLUDE_LEVERAGED, TIMEZONE
)

# Sri Lanka timezone
SL_TZ = pytz.timezone(TIMEZONE)


def get_sl_time() -> datetime:
    """Get current Sri Lanka time"""
    return datetime.now(SL_TZ)


def format_sl_time(dt: datetime = None) -> str:
    """Format datetime in Sri Lanka time"""
    if dt is None:
        dt = get_sl_time()
    elif dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(SL_TZ)
    else:
        dt = dt.astimezone(SL_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S IST')


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
class SwingPoint:
    index: int
    price: float
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
    def strength(self) -> str:
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
    confirmation: MSConfirmation
    total_confidence: float
    timestamp: datetime
    volume_rank: int  # Rank by 24h volume


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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
            
            # Fetch all tickers
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                # Only USDT pairs
                if not symbol.endswith(f'/{QUOTE_CURRENCY}'):
                    continue
                
                # Exclude stablecoins
                if symbol in EXCLUDED_SYMBOLS:
                    continue
                
                # Exclude leveraged tokens
                if EXCLUDE_LEVERAGED:
                    base = symbol.split('/')[0]
                    if any(x in base.upper() for x in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '2L', '2S']):
                        continue
                
                # Get 24h quote volume (in USDT)
                quote_volume = ticker.get('quoteVolume', 0) or 0
                
                if quote_volume > 0:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': quote_volume,
                        'price': ticker.get('last', 0)
                    })
            
            # Sort by volume (highest first)
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            
            # Take top N
            top_symbols = [p['symbol'] for p in usdt_pairs[:count]]
            
            # Store volume ranks
            self.volume_ranks = {p['symbol']: i+1 for i, p in enumerate(usdt_pairs[:count])}
            
            # Cache results
            self.symbols_cache = top_symbols
            self.symbols_cache_time = datetime.now()
            
            print(f"[{format_sl_time()}] Top 5 by volume: {top_symbols[:5]}")
            
            return top_symbols
            
        except Exception as e:
            print(f"[{format_sl_time()}] Error fetching top coins: {e}")
            # Return cached or default top 50
            if self.symbols_cache:
                return self.symbols_cache
            # Default top 50 coins by typical volume
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
        """Get list of symbols to scan"""
        if SYMBOLS and len(SYMBOLS) > 0:
            return SYMBOLS
        return self.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    def _is_on_cooldown(self, symbol: str, timeframe: str) -> bool:
        key = f"{symbol}_{timeframe}"
        if key in self.last_alerts:
            elapsed = (datetime.now() - self.last_alerts[key]).total_seconds()
            return elapsed < ALERT_COOLDOWN
        return False
    
    def _set_cooldown(self, symbol: str, timeframe: str):
        self.last_alerts[f"{symbol}_{timeframe}"] = datetime.now()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        swing_highs = []
        n = SWING_DETECTION_BARS
        for i in range(n, len(df) - n):
            window = df['high'].iloc[i-n:i+n+1]
            if df['high'].iloc[i] == window.max():
                swing_highs.append(SwingPoint(
                    index=i,
                    price=df['high'].iloc[i],
                    rsi=df['rsi'].iloc[i] if pd.notna(df['rsi'].iloc[i]) else 50,
                    is_high=True,
                    timestamp=df['timestamp'].iloc[i]
                ))
        return swing_highs[-10:] if len(swing_highs) > 10 else swing_highs
    
    def find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        swing_lows = []
        n = SWING_DETECTION_BARS
        for i in range(n, len(df) - n):
            window = df['low'].iloc[i-n:i+n+1]
            if df['low'].iloc[i] == window.min():
                swing_lows.append(SwingPoint(
                    index=i,
                    price=df['low'].iloc[i],
                    rsi=df['rsi'].iloc[i] if pd.notna(df['rsi'].iloc[i]) else 50,
                    is_high=False,
                    timestamp=df['timestamp'].iloc[i]
                ))
        return swing_lows[-10:] if len(swing_lows) > 10 else swing_lows
    
    def detect_bullish_divergence(self, swing_lows: List[SwingPoint], current_price: float, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        if len(swing_lows) < 2:
            return None
        point1, point2 = swing_lows[-2], swing_lows[-1]
        if pd.isna(point1.rsi) or pd.isna(point2.rsi):
            return None
        
        price_pct = ((point2.price - point1.price) / point1.price) * 100
        rsi_change = point2.rsi - point1.rsi
        
        divergence_type, confidence = None, 0.0
        
        if point2.price < point1.price and point2.rsi > point1.rsi:
            if abs(price_pct) > PRICE_TOLERANCE_PERCENT:
                divergence_type, confidence = DivergenceType.STRONG_BULLISH, 0.85
        elif abs(price_pct) <= PRICE_TOLERANCE_PERCENT and point2.rsi > point1.rsi:
            divergence_type, confidence = DivergenceType.MEDIUM_BULLISH, 0.75
        elif point2.price < point1.price and abs(rsi_change) <= RSI_TOLERANCE:
            divergence_type, confidence = DivergenceType.WEAK_BULLISH, 0.60
        elif point2.price > point1.price and point2.rsi < point1.rsi:
            divergence_type, confidence = DivergenceType.HIDDEN_BULLISH, 0.70
        
        if divergence_type:
            return DivergenceSignal(symbol, timeframe, divergence_type, point1, point2, current_price, confidence)
        return None
    
    def detect_bearish_divergence(self, swing_highs: List[SwingPoint], current_price: float, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        if len(swing_highs) < 2:
            return None
        point1, point2 = swing_highs[-2], swing_highs[-1]
        if pd.isna(point1.rsi) or pd.isna(point2.rsi):
            return None
        
        price_pct = ((point2.price - point1.price) / point1.price) * 100
        rsi_change = point2.rsi - point1.rsi
        
        divergence_type, confidence = None, 0.0
        
        if point2.price > point1.price and point2.rsi < point1.rsi:
            if abs(price_pct) > PRICE_TOLERANCE_PERCENT:
                divergence_type, confidence = DivergenceType.STRONG_BEARISH, 0.85
        elif abs(price_pct) <= PRICE_TOLERANCE_PERCENT and point2.rsi < point1.rsi:
            divergence_type, confidence = DivergenceType.MEDIUM_BEARISH, 0.75
        elif point2.price > point1.price and abs(rsi_change) <= RSI_TOLERANCE:
            divergence_type, confidence = DivergenceType.WEAK_BEARISH, 0.60
        elif point2.price < point1.price and point2.rsi > point1.rsi:
            divergence_type, confidence = DivergenceType.HIDDEN_BEARISH, 0.70
        
        if divergence_type:
            return DivergenceSignal(symbol, timeframe, divergence_type, point1, point2, current_price, confidence)
        return None
    
    def check_ms_confirmation(self, symbol: str, confirmation_tf: str, is_bullish: bool) -> Optional[MSConfirmation]:
        df = self.fetch_ohlcv(symbol, confirmation_tf, limit=50)
        if df is None or len(df) < 20:
            return None
        
        swing_highs = self.find_swing_highs(df)
        swing_lows = self.find_swing_lows(df)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        current_close = df['close'].iloc[-1]
        recent_high, prev_high = swing_highs[-1].price, swing_highs[-2].price
        recent_low, prev_low = swing_lows[-1].price, swing_lows[-2].price
        
        structure_break = StructureBreak.NONE
        if is_bullish:
            if current_close > recent_high:
                structure_break = StructureBreak.BULLISH_BOS if recent_high > prev_high else StructureBreak.BULLISH_CHOCH
        else:
            if current_close < recent_low:
                structure_break = StructureBreak.BEARISH_BOS if recent_low < prev_low else StructureBreak.BEARISH_CHOCH
        
        return MSConfirmation(confirmation_tf, structure_break, recent_high, recent_low, current_close, structure_break != StructureBreak.NONE)
    
    def scan_symbol(self, symbol: str, timeframe: str) -> Optional[AlertSignal]:
        if self._is_on_cooldown(symbol, timeframe):
            return None
        
        df = self.fetch_ohlcv(symbol, timeframe)
        if df is None or len(df) < RSI_LOOKBACK_CANDLES:
            return None
        
        current_price = df['close'].iloc[-1]
        swing_highs = self.find_swing_highs(df)
        swing_lows = self.find_swing_lows(df)
        
        volume_rank = self.volume_ranks.get(symbol, 999)
        
        # Check bullish
        bullish_div = self.detect_bullish_divergence(swing_lows, current_price, symbol, timeframe)
        if bullish_div and bullish_div.confidence >= MIN_CONFIDENCE_THRESHOLD:
            confirmation = self.check_ms_confirmation(symbol, bullish_div.confirmation_tf, True)
            if confirmation and confirmation.confirmed:
                self._set_cooldown(symbol, timeframe)
                total_conf = min(bullish_div.confidence + (0.10 if "BOS" in confirmation.structure_break.value else 0), 0.95)
                return AlertSignal(symbol, timeframe, bullish_div.confirmation_tf, bullish_div, confirmation, total_conf, get_sl_time(), volume_rank)
        
        # Check bearish
        bearish_div = self.detect_bearish_divergence(swing_highs, current_price, symbol, timeframe)
        if bearish_div and bearish_div.confidence >= MIN_CONFIDENCE_THRESHOLD:
            confirmation = self.check_ms_confirmation(symbol, bearish_div.confirmation_tf, False)
            if confirmation and confirmation.confirmed:
                self._set_cooldown(symbol, timeframe)
                total_conf = min(bearish_div.confidence + (0.10 if "BOS" in confirmation.structure_break.value else 0), 0.95)
                return AlertSignal(symbol, timeframe, bearish_div.confirmation_tf, bearish_div, confirmation, total_conf, get_sl_time(), volume_rank)
        
        return None
    
    def scan_all(self) -> List[AlertSignal]:
        alerts = []
        symbols = self.get_symbols_to_scan()
        
        print(f"[{format_sl_time()}] Scanning {len(symbols)} symbols...")
        
        for symbol in symbols:
            for timeframe in SCAN_TIMEFRAMES:
                try:
                    alert = self.scan_symbol(symbol, timeframe)
                    if alert:
                        alerts.append(alert)
                        print(f"[{format_sl_time()}] 🚨 Signal: {symbol} {timeframe}")
                    time.sleep(0.3)  # Rate limiting
                except Exception as e:
                    print(f"Error scanning {symbol} {timeframe}: {e}")
        
        print(f"[{format_sl_time()}] Scan complete. Found {len(alerts)} signals.")
        return alerts


class AlertFormatter:
    NAMES = {
        DivergenceType.STRONG_BULLISH: "Strong Bullish",
        DivergenceType.MEDIUM_BULLISH: "Medium Bullish",
        DivergenceType.WEAK_BULLISH: "Weak Bullish",
        DivergenceType.HIDDEN_BULLISH: "Hidden Bullish",
        DivergenceType.STRONG_BEARISH: "Strong Bearish",
        DivergenceType.MEDIUM_BEARISH: "Medium Bearish",
        DivergenceType.WEAK_BEARISH: "Weak Bearish",
        DivergenceType.HIDDEN_BEARISH: "Hidden Bearish",
    }
    
    DESC = {
        DivergenceType.STRONG_BULLISH: "Price: LL | RSI: HL",
        DivergenceType.MEDIUM_BULLISH: "Price: DB | RSI: HL",
        DivergenceType.WEAK_BULLISH: "Price: LL | RSI: DB",
        DivergenceType.HIDDEN_BULLISH: "Price: HL | RSI: LL",
        DivergenceType.STRONG_BEARISH: "Price: HH | RSI: LH",
        DivergenceType.MEDIUM_BEARISH: "Price: DT | RSI: LH",
        DivergenceType.WEAK_BEARISH: "Price: HH | RSI: DT",
        DivergenceType.HIDDEN_BEARISH: "Price: LH | RSI: HH",
    }
    
    @classmethod
    def format_alert(cls, signal: AlertSignal) -> str:
        is_bull = signal.divergence.is_bullish
        emoji = "🟢" if is_bull else "🔴"
        direction = "BULLISH" if is_bull else "BEARISH"
        trade = "LONG" if is_bull else "SHORT"
        
        stop = signal.confirmation.swing_low * 0.995 if is_bull else signal.confirmation.swing_high * 1.005
        target = signal.confirmation.swing_high * 1.01 if is_bull else signal.confirmation.swing_low * 0.99
        
        # Format price based on value
        def fmt_price(p):
            if p >= 1000:
                return f"${p:,.2f}"
            elif p >= 1:
                return f"${p:.4f}"
            else:
                return f"${p:.8f}"
        
        return f"""{emoji} {direction} DIVERGENCE CONFIRMED

📊 {signal.symbol} (Vol Rank #{signal.volume_rank})
⏰ Signal: {signal.signal_tf.upper()} | Confirm: {signal.confirm_tf.upper()}

━━━━━━━━━━━━━━━━━━━━
📈 DIVERGENCE
━━━━━━━━━━━━━━━━━━━━
Type: {cls.NAMES[signal.divergence.divergence_type]}
Pattern: {cls.DESC[signal.divergence.divergence_type]}
Strength: {signal.divergence.strength}

Point 1: {fmt_price(signal.divergence.point1.price)} (RSI: {signal.divergence.point1.rsi:.1f})
Point 2: {fmt_price(signal.divergence.point2.price)} (RSI: {signal.divergence.point2.rsi:.1f})

━━━━━━━━━━━━━━━━━━━━
✅ MS CONFIRMATION
━━━━━━━━━━━━━━━━━━━━
Break: {signal.confirmation.structure_break.value}
Swing High: {fmt_price(signal.confirmation.swing_high)}
Swing Low: {fmt_price(signal.confirmation.swing_low)}

━━━━━━━━━━━━━━━━━━━━
🎯 TRADE IDEA ({trade})
━━━━━━━━━━━━━━━━━━━━
Price: {fmt_price(signal.divergence.current_price)}
Stop: {fmt_price(stop)}
Target: {fmt_price(target)}

🔥 Confidence: {signal.total_confidence * 100:.0f}%

⚠️ Not financial advice. DYOR!
🇱🇰 {format_sl_time(signal.timestamp)}"""
