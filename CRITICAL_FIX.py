"""
CRITICAL FIX for V9 Bot

PROBLEMS IDENTIFIED:
1. Sends signals for OLD divergences (Swing 2 happened weeks ago)
2. Doesn't properly wait for 3-candle confirmation
3. Sends signals below 70% confidence
4. Volume falling but still sends signal

SOLUTIONS:
1. Check recency: Swing 2 must be recent (< 10 candles ago)
2. Actually wait for 3 candles AFTER Swing 2
3. Enforce 70% minimum strictly
4. Replace volume with trend strength (ADX)
"""

def scan_symbol_FIXED(self, symbol: str, timeframe: str) -> List[AlertSignal]:
    """
    FIXED VERSION - Filters old divergences and enforces confirmation
    """
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
        # ========================================
        # FIX 1: CHECK RECENCY
        # ========================================
        swing2_idx = divergence.swing2.index
        candles_since_swing2 = idx - swing2_idx
        
        # CRITICAL: Swing 2 must be recent!
        MAX_CANDLES_AGO = {
            '1h': 10,   # 10 hours ago max
            '4h': 8,    # 32 hours ago max
            '1d': 5,    # 5 days ago max
            '1w': 3,    # 3 weeks ago max (for weekly)
        }
        
        max_age = MAX_CANDLES_AGO.get(timeframe, 10)
        
        if candles_since_swing2 > max_age:
            print(f"[{symbol} {timeframe}] REJECTED: Swing 2 is {candles_since_swing2} candles old (max {max_age})")
            return alerts  # TOO OLD, SKIP!
        
        # ========================================
        # FIX 2: CHECK 3-CANDLE CONFIRMATION
        # ========================================
        is_bullish = "BULLISH" in divergence.divergence_type.value.upper()
        
        confirmation = self.check_3_candle_confirmation(df, is_bullish, swing2_idx)
        
        # CRITICAL: Must have AT LEAST 3 candles after Swing 2
        if confirmation.candles_checked < 3:
            print(f"[{symbol} {timeframe}] REJECTED: Not enough candles after Swing 2 (need 3, have {candles_since_swing2})")
            return alerts  # NOT ENOUGH CANDLES YET
        
        # ========================================
        # FIX 3: CHECK TREND STRENGTH (REPLACES VOLUME)
        # ========================================
        momentum = self.check_momentum_with_trend_strength(df, is_bullish, tv_data)
        
        signal_strength, confidence = self.determine_signal_strength(divergence, momentum, confirmation, tv_data)
        
        # ========================================
        # FIX 4: ENFORCE 70% MINIMUM STRICTLY
        # ========================================
        if confidence < 0.70:
            print(f"[{symbol} {timeframe}] REJECTED: Confidence {confidence*100:.0f}% < 70%")
            return alerts  # BELOW MINIMUM
        
        # ========================================
        # FIX 5: REQUIRE CONFIRMATION FOR WEEKLY
        # ========================================
        if timeframe in ['1w', '1M']:
            if not confirmation.is_confirmed:
                print(f"[{symbol} {timeframe}] REJECTED: Weekly/Monthly requires full confirmation")
                return alerts  # WEEKLY NEEDS CONFIRMATION
        
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


def check_momentum_with_trend_strength(self, df: pd.DataFrame, is_bullish: bool, tv_data: dict = None) -> MomentumStatus:
    """
    IMPROVED: Uses TREND STRENGTH (ADX) instead of volume
    
    Why ADX is better than volume:
    - Volume can be manipulated
    - Volume falling doesn't mean trend is weak
    - ADX measures actual trend strength (0-100)
    - ADX > 25 = strong trend (reliable)
    """
    if len(df) < 3:
        return MomentumStatus(
            rsi_confirmed=False, rsi_direction="Unknown", rsi_values=[],
            price_confirmed=False, price_direction="Unknown", price_change_pct=0,
            volume_confirmed=False, volume_direction="Unknown", volume_change_pct=0
        )
    
    # Get last 3 candles
    rsi_values = df['rsi'].iloc[-3:].tolist()
    price_values = df['close'].iloc[-3:].tolist()
    
    # RSI direction
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
    
    # Price direction
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
    
    # TREND STRENGTH (ADX) - REPLACES VOLUME
    adx = self._calculate_adx(df, period=14)
    
    if adx > 25:
        trend_direction = "Strong 💪"
        trend_confirmed = True  # Strong trend = good
        trend_value = adx
    elif adx > 20:
        trend_direction = "Moderate 👍"
        trend_confirmed = True
        trend_value = adx
    else:
        trend_direction = "Weak ⚠️"
        trend_confirmed = False  # Weak trend = risky
        trend_value = adx
    
    return MomentumStatus(
        rsi_confirmed=rsi_confirmed,
        rsi_direction=rsi_direction,
        rsi_values=rsi_values,
        price_confirmed=price_confirmed,
        price_direction=price_direction,
        price_change_pct=round(price_change_pct, 2),
        volume_confirmed=trend_confirmed,  # Using ADX now
        volume_direction=trend_direction,  # "Strong/Moderate/Weak"
        volume_change_pct=round(trend_value, 1)  # ADX value
    )


def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate ADX (Average Directional Index)
    Measures trend strength (0-100)
    
    > 25 = Strong trend
    20-25 = Moderate trend
    < 20 = Weak/no trend
    """
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
    
    return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20


# ============================================================================
# UPDATED ALERT MESSAGE FORMAT
# ============================================================================

def format_alert_FIXED(alert: AlertSignal) -> str:
    """Shows ADX instead of Volume"""
    
    is_bull = "BULLISH" in alert.divergence.divergence_type.value.upper()
    
    # ... [same emoji/direction logic] ...
    
    # ADX emoji (replaces volume)
    adx_value = alert.momentum.volume_change_pct  # This is ADX now
    if adx_value > 25:
        adx_emoji = "✅"
    elif adx_value > 20:
        adx_emoji = "⚠️"
    else:
        adx_emoji = "❌"
    
    # Confirmation
    conf = alert.confirmation
    if conf.is_confirmed:
        conf_emoji = "✅"
        conf_text = f"3-Candle Confirmed! RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2"
    else:
        conf_emoji = "⏳"
        conf_text = f"Waiting... RSI:{conf.rsi_rising_count}/2 Price:{conf.price_rising_count}/2"
    
    return f"""{strength_emoji} {strength_label} SIGNAL - {direction}

📊 {alert.symbol} (#{alert.volume_rank})
⏰ {alert.signal_tf.upper()} | {candle_time_str}

{'━'*28}
{DIV_NAMES[div.divergence_type]}
{DIV_DESC[div.divergence_type]}

Swing 1: {fmt_price(div.swing1.price)} (RSI: {div.swing1.rsi:.1f})
Swing 2: {fmt_price(div.swing2.price)} (RSI: {div.swing2.rsi:.1f})
Now: {fmt_price(div.current_price)} (RSI: {div.current_rsi:.1f})

📏 {div.candles_apart} candles apart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conf_emoji} {conf_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tv_emoji} TradingView: {tv_rec}
{rsi_emoji} RSI: {alert.momentum.rsi_direction}
{price_emoji} Price: {alert.momentum.price_direction}
{adx_emoji} Trend: {alert.momentum.volume_direction} (ADX: {adx_value:.1f})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 {trade} | Entry: {fmt_price(entry)}
🛑 SL: {fmt_price(stop)} | 🎯 TP: {fmt_price(target)}

🔥 Confidence: {alert.total_confidence * 100:.0f}%
📺 {alert.tradingview_link}

⚠️ DYOR | 🇱🇰 {format_sl_time(alert.timestamp)}"""
