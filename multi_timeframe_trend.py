"""
Multi-Timeframe Trend Confirmation
Professional approach: Check trend on lower timeframe
"""

import ccxt
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass

# Multi-timeframe mapping for trend confirmation
TREND_CONFIRMATION_MAP = {
    "1M": "1d",    # Monthly divergence → Check daily trend
    "1w": "4h",    # Weekly divergence → Check 4h trend
    "1d": "1h",    # Daily divergence → Check 1h trend
    "4h": "15m",   # 4h divergence → Check 15m trend
    "1h": "5m",    # 1h divergence → Check 5m trend
}


@dataclass
class TrendConfirmation:
    """Results from lower timeframe trend check"""
    confirmation_tf: str      # Which TF was checked
    adx: float               # Trend strength (0-100)
    trend_direction: str     # "Strong Up", "Strong Down", "Weak", "Sideways"
    price_trend: str         # "Rising", "Falling", "Sideways"
    rsi_trend: str          # "Rising", "Falling", "Sideways"
    is_confirmed: bool      # Does lower TF confirm the divergence?
    confidence_boost: float  # How much to boost/reduce confidence


class MultiTimeframeTrendChecker:
    """
    Checks trend strength on lower timeframe
    Confirms if divergence is playing out in real-time
    """
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
    
    def check_trend_on_lower_tf(self, symbol: str, signal_tf: str, 
                                 is_bullish: bool) -> TrendConfirmation:
        """
        Check trend on lower timeframe to confirm divergence
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            signal_tf: Timeframe where divergence detected (e.g., "1d")
            is_bullish: True for bullish divergence, False for bearish
        
        Returns:
            TrendConfirmation with analysis results
        """
        # Get confirmation timeframe
        confirmation_tf = TREND_CONFIRMATION_MAP.get(signal_tf, "1h")
        
        # Fetch lower timeframe data
        df = self._fetch_data(symbol, confirmation_tf, limit=50)
        
        if df is None or len(df) < 20:
            return self._default_confirmation(confirmation_tf)
        
        # Calculate indicators on lower TF
        adx = self._calculate_adx(df)
        price_trend = self._get_price_trend(df)
        rsi_trend = self._get_rsi_trend(df)
        
        # Determine trend direction
        trend_direction = self._classify_trend(adx, price_trend, is_bullish)
        
        # Check if confirmed
        is_confirmed, confidence_boost = self._evaluate_confirmation(
            adx, price_trend, rsi_trend, is_bullish
        )
        
        return TrendConfirmation(
            confirmation_tf=confirmation_tf,
            adx=adx,
            trend_direction=trend_direction,
            price_trend=price_trend,
            rsi_trend=rsi_trend,
            is_confirmed=is_confirmed,
            confidence_boost=confidence_boost
        )
    
    def _fetch_data(self, symbol: str, timeframe: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for lower timeframe"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate RSI
            df['rsi'] = self._calculate_rsi(df['close'])
            
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX - Trend Strength"""
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
        
        import numpy as np
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm_smooth = pd.Series(plus_dm).rolling(period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(period).mean()
        
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20.0
    
    def _calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
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
    
    def _get_price_trend(self, df: pd.DataFrame, periods: int = 5) -> str:
        """Analyze price trend on lower TF"""
        closes = df['close'].tail(periods)
        
        # Check if consistently rising/falling
        rising = all(closes.iloc[i] > closes.iloc[i-1] for i in range(1, len(closes)))
        falling = all(closes.iloc[i] < closes.iloc[i-1] for i in range(1, len(closes)))
        
        if rising:
            return "Rising ↗️"
        elif falling:
            return "Falling ↘️"
        else:
            return "Sideways ↔️"
    
    def _get_rsi_trend(self, df: pd.DataFrame, periods: int = 5) -> str:
        """Analyze RSI trend on lower TF"""
        rsi_values = df['rsi'].tail(periods)
        
        # Check if consistently rising/falling
        rising = all(rsi_values.iloc[i] > rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        falling = all(rsi_values.iloc[i] < rsi_values.iloc[i-1] for i in range(1, len(rsi_values)))
        
        if rising:
            return "Rising ↗️"
        elif falling:
            return "Falling ↘️"
        else:
            return "Sideways ↔️"
    
    def _classify_trend(self, adx: float, price_trend: str, is_bullish: bool) -> str:
        """Classify trend strength and direction"""
        if adx > 30:
            strength = "Very Strong"
        elif adx > 25:
            strength = "Strong"
        elif adx > 20:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Check if trend matches divergence direction
        if is_bullish and "Rising" in price_trend:
            return f"{strength} Up 📈"
        elif not is_bullish and "Falling" in price_trend:
            return f"{strength} Down 📉"
        else:
            return f"{strength} (Conflicting)"
    
    def _evaluate_confirmation(self, adx: float, price_trend: str, 
                               rsi_trend: str, is_bullish: bool) -> Tuple[bool, float]:
        """
        Evaluate if lower TF confirms the divergence
        
        Returns: (is_confirmed, confidence_boost)
        """
        # Check trend strength
        if adx < 20:
            # Weak trend on lower TF = not confirmed
            return False, -0.10
        
        # Check price trend alignment
        if is_bullish:
            # Bullish divergence should show rising price on lower TF
            if "Rising" in price_trend:
                price_aligned = True
            else:
                price_aligned = False
        else:
            # Bearish divergence should show falling price on lower TF
            if "Falling" in price_trend:
                price_aligned = True
            else:
                price_aligned = False
        
        # Check RSI trend alignment
        if is_bullish:
            rsi_aligned = "Rising" in rsi_trend
        else:
            rsi_aligned = "Falling" in rsi_trend
        
        # Evaluate confirmation
        if price_aligned and rsi_aligned and adx >= 25:
            # Perfect confirmation
            return True, +0.15
        elif price_aligned and adx >= 20:
            # Good confirmation
            return True, +0.10
        elif price_aligned or rsi_aligned:
            # Partial confirmation
            return True, +0.05
        else:
            # No confirmation
            return False, -0.05
    
    def _default_confirmation(self, confirmation_tf: str) -> TrendConfirmation:
        """Return default if data fetch fails"""
        return TrendConfirmation(
            confirmation_tf=confirmation_tf,
            adx=20.0,
            trend_direction="Unknown",
            price_trend="Unknown",
            rsi_trend="Unknown",
            is_confirmed=False,
            confidence_boost=0.0
        )


# ============================================================================
# USAGE IN DIVERGENCE SCANNER
# ============================================================================

def scan_symbol_with_mtf_trend(self, symbol: str, timeframe: str) -> List[AlertSignal]:
    """
    Enhanced scan with multi-timeframe trend confirmation
    """
    # ... [existing code to detect divergence] ...
    
    if divergence:
        is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
        
        # ========================================
        # NEW: Check trend on lower timeframe
        # ========================================
        mtf_checker = MultiTimeframeTrendChecker(self.exchange)
        trend_confirmation = mtf_checker.check_trend_on_lower_tf(
            symbol, 
            timeframe, 
            is_bullish
        )
        
        # Get 3-candle confirmation (on signal TF)
        confirmation = self.check_3_candle_confirmation(df, is_bullish, swing2_idx)
        
        # Get momentum (on signal TF)
        momentum = self.check_momentum(df, is_bullish, tv_data)
        
        # ========================================
        # Adjust confidence based on MTF trend
        # ========================================
        base_confidence = divergence.confidence
        
        # Apply MTF trend boost/penalty
        base_confidence += trend_confirmation.confidence_boost
        
        # Determine signal strength
        signal_strength, final_confidence = self.determine_signal_strength(
            divergence, momentum, confirmation, tv_data
        )
        
        # ========================================
        # CRITICAL: Reject if MTF trend not confirmed
        # ========================================
        if not trend_confirmation.is_confirmed:
            print(f"[{symbol} {timeframe}] REJECTED: Lower TF ({trend_confirmation.confirmation_tf}) trend not confirmed")
            return []
        
        # Check minimum confidence
        if final_confidence < 0.70:
            return []
        
        # Create alert with MTF trend info
        alert = AlertSignal(
            # ... existing fields ...
            mtf_trend=trend_confirmation,  # NEW FIELD
            # ...
        )
        
        return [alert]


# ============================================================================
# UPDATED TELEGRAM MESSAGE FORMAT
# ============================================================================

def format_alert_with_mtf(alert: AlertSignal) -> str:
    """Format alert with multi-timeframe trend info"""
    
    mtf = alert.mtf_trend
    
    # ADX emoji
    if mtf.adx > 25:
        adx_emoji = "✅"
    elif mtf.adx > 20:
        adx_emoji = "⚠️"
    else:
        adx_emoji = "❌"
    
    return f"""{strength_emoji} {strength_label} SIGNAL - {direction}

📊 {alert.symbol} (#{alert.volume_rank})
⏰ {alert.signal_tf.upper()} | {candle_time_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{DIV_NAMES[div.divergence_type]}
{DIV_DESC[div.divergence_type]}

Swing 1: {fmt_price(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {fmt_price(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Now: {fmt_price(div.current_price)} (RSI: {div.current_rsi:.1f})

📏 {div.candles_apart} candles apart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conf_emoji} {conf_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Lower TF Trend ({mtf.confirmation_tf.upper()}):
{adx_emoji} Trend: {mtf.trend_direction} (ADX: {mtf.adx:.1f})
{'✅' if 'Rising' in mtf.price_trend else '❌'} Price: {mtf.price_trend}
{'✅' if 'Rising' in mtf.rsi_trend else '❌'} RSI: {mtf.rsi_trend}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tv_emoji} TradingView: {tv_rec}
{rsi_emoji} RSI ({alert.signal_tf}): {alert.momentum.rsi_direction}
{price_emoji} Price ({alert.signal_tf}): {alert.momentum.price_direction}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 {trade} | Entry: {fmt_price(entry)}
🛑 SL: {fmt_price(stop)} | 🎯 TP: {fmt_price(target)}

🔥 Confidence: {alert.total_confidence * 100:.0f}%
📺 {alert.tradingview_link}

⚠️ DYOR | 🇱🇰 {format_sl_time(alert.timestamp)}"""
