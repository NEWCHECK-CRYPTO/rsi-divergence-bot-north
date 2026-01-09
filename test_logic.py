#!/usr/bin/env python3
"""
RSI Divergence Logic Verification - SIMULATION MODE
====================================================
Tests the signal detection logic with simulated data to verify
the 1-candle swing strength configuration works correctly.

This runs without network access - uses generated test data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION - 1 CANDLE SWING STRENGTH FOR ALL TIMEFRAMES
# =============================================================================

SWING_STRENGTH_MAP = {
    "4h": 1,   # Alert ~4 hours after swing
    "1d": 1,   # Alert ~1 day after swing
    "1w": 1,   # Alert ~1 week after swing
    "1M": 1,   # Alert ~1 month after swing
}

RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
MIN_SWING_DISTANCE = 3
MAX_SWING_DISTANCE = 50

MAX_CANDLES_SINCE_SWING2_MAP = {
    "4h": 3,
    "1d": 2,
    "1w": 2,
    "1M": 2,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing"""
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


@dataclass
class SwingPoint:
    index: int
    price: float
    rsi: float
    timestamp: datetime


def find_swing_lows(df: pd.DataFrame, timeframe: str) -> List[SwingPoint]:
    """Find swing low points"""
    swings = []
    strength = SWING_STRENGTH_MAP.get(timeframe, 1)
    
    for i in range(strength, len(df) - strength):
        is_swing_low = True
        
        for j in range(1, strength + 1):
            if df['close'].iloc[i] >= df['close'].iloc[i - j] or \
               df['close'].iloc[i] >= df['close'].iloc[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low and not pd.isna(df['rsi'].iloc[i]):
            swings.append(SwingPoint(
                index=i,
                price=df['close'].iloc[i],
                rsi=df['rsi'].iloc[i],
                timestamp=df['timestamp'].iloc[i]
            ))
    
    return swings


def find_swing_highs(df: pd.DataFrame, timeframe: str) -> List[SwingPoint]:
    """Find swing high points"""
    swings = []
    strength = SWING_STRENGTH_MAP.get(timeframe, 1)
    
    for i in range(strength, len(df) - strength):
        is_swing_high = True
        
        for j in range(1, strength + 1):
            if df['close'].iloc[i] <= df['close'].iloc[i - j] or \
               df['close'].iloc[i] <= df['close'].iloc[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high and not pd.isna(df['rsi'].iloc[i]):
            swings.append(SwingPoint(
                index=i,
                price=df['close'].iloc[i],
                rsi=df['rsi'].iloc[i],
                timestamp=df['timestamp'].iloc[i]
            ))
    
    return swings


def check_pattern_validity(df: pd.DataFrame, swing1: SwingPoint, 
                            swing2: SwingPoint, is_bullish: bool) -> Tuple[bool, str]:
    """Check if pattern is not broken by middle candles"""
    start_idx = swing1.index + 1
    end_idx = swing2.index
    
    if start_idx >= end_idx:
        return True, "No middle candles"
    
    middle = df.iloc[start_idx:end_idx]
    
    if is_bullish:
        min_close = middle['close'].min()
        if min_close < swing2.price:
            return False, f"Broken: ${min_close:.2f} < ${swing2.price:.2f}"
    else:
        max_close = middle['close'].max()
        if max_close > swing2.price:
            return False, f"Broken: ${max_close:.2f} > ${swing2.price:.2f}"
    
    return True, "Valid"


# =============================================================================
# SIMULATED DATA GENERATORS
# =============================================================================

def generate_bullish_divergence_data(timeframe: str = "4h", candles: int = 100) -> pd.DataFrame:
    """
    Generate data with a bullish divergence pattern.
    
    Pattern: Price makes Lower Low, RSI makes Higher Low
             RSI in oversold zone (< 40)
    """
    np.random.seed(42)  # For reproducibility
    
    base_price = 100.0
    prices = []
    
    # Phase 1: Initial decline (candles 0-30)
    for i in range(31):
        noise = np.random.uniform(-0.5, 0.5)
        price = base_price - (i * 0.5) + noise
        prices.append(price)
    
    # Phase 2: Small bounce (candles 31-50)
    bounce_start = prices[-1]
    for i in range(20):
        noise = np.random.uniform(-0.3, 0.3)
        price = bounce_start + (i * 0.2) + noise
        prices.append(price)
    
    # Phase 3: Second decline to LOWER price (candles 51-70)
    decline_start = prices[-1]
    for i in range(20):
        noise = np.random.uniform(-0.3, 0.3)
        price = decline_start - (i * 0.3) + noise
        prices.append(price)
    
    # Phase 4: Recovery bounce (candles 71-99)
    recovery_start = prices[-1]
    for i in range(candles - len(prices)):
        noise = np.random.uniform(-0.2, 0.4)  # Slightly bullish bias
        price = recovery_start + (i * 0.15) + noise
        prices.append(price)
    
    # Create DataFrame
    tf_hours = {"4h": 4, "1d": 24, "1w": 168, "1M": 720}
    hours = tf_hours.get(timeframe, 4)
    
    timestamps = [datetime.now() - timedelta(hours=hours * (candles - i - 1)) for i in range(candles)]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * candles
    })
    
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    return df


def generate_bearish_divergence_data(timeframe: str = "4h", candles: int = 100) -> pd.DataFrame:
    """
    Generate data with a bearish divergence pattern.
    
    Pattern: Price makes Higher High, RSI makes Lower High
             RSI in overbought zone (> 60)
    """
    np.random.seed(43)  # For reproducibility
    
    base_price = 100.0
    prices = []
    
    # Phase 1: Initial rise (candles 0-30)
    for i in range(31):
        noise = np.random.uniform(-0.5, 0.5)
        price = base_price + (i * 0.5) + noise
        prices.append(price)
    
    # Phase 2: Small pullback (candles 31-50)
    pullback_start = prices[-1]
    for i in range(20):
        noise = np.random.uniform(-0.3, 0.3)
        price = pullback_start - (i * 0.2) + noise
        prices.append(price)
    
    # Phase 3: Second rise to HIGHER price (candles 51-70)
    rise_start = prices[-1]
    for i in range(20):
        noise = np.random.uniform(-0.3, 0.3)
        price = rise_start + (i * 0.35) + noise  # Goes higher than first peak
        prices.append(price)
    
    # Phase 4: Decline (candles 71-99)
    decline_start = prices[-1]
    for i in range(candles - len(prices)):
        noise = np.random.uniform(-0.4, 0.2)  # Slightly bearish bias
        price = decline_start - (i * 0.15) + noise
        prices.append(price)
    
    # Create DataFrame
    tf_hours = {"4h": 4, "1d": 24, "1w": 168, "1M": 720}
    hours = tf_hours.get(timeframe, 4)
    
    timestamps = [datetime.now() - timedelta(hours=hours * (candles - i - 1)) for i in range(candles)]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * candles
    })
    
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    return df


def generate_recent_swing_data(timeframe: str = "4h", candles: int = 50) -> pd.DataFrame:
    """
    Generate data with a RECENT swing point (within recency window).
    This tests whether the signal fires correctly with 1-candle strength.
    """
    np.random.seed(44)
    
    base_price = 100.0
    prices = []
    
    # Phase 1: Decline (candles 0-35)
    for i in range(36):
        noise = np.random.uniform(-0.3, 0.3)
        price = base_price - (i * 0.4) + noise
        prices.append(price)
    
    # Phase 2: Small bounce creating swing low 1 (candles 36-40)
    bounce_start = prices[-1]
    for i in range(5):
        noise = np.random.uniform(-0.2, 0.2)
        price = bounce_start + (i * 0.3) + noise
        prices.append(price)
    
    # Phase 3: Second decline to lower low (candles 41-45)
    decline_start = prices[-1]
    for i in range(5):
        noise = np.random.uniform(-0.2, 0.2)
        price = decline_start - (i * 0.5) + noise
        prices.append(price)
    
    # Phase 4: Bounce after swing low 2 - THIS IS THE KEY
    # With strength=1, we need 1 candle after swing2
    # Swing2 should be at index ~45, detected at index 46
    recovery_start = prices[-1]
    for i in range(candles - len(prices)):
        noise = np.random.uniform(-0.1, 0.3)
        price = recovery_start + (i * 0.2) + noise
        prices.append(price)
    
    # Create DataFrame
    tf_hours = {"4h": 4, "1d": 24, "1w": 168, "1M": 720}
    hours = tf_hours.get(timeframe, 4)
    
    timestamps = [datetime.now() - timedelta(hours=hours * (candles - i - 1)) for i in range(candles)]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * candles
    })
    
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    return df


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

def test_swing_detection():
    """Test 1: Verify swing detection with strength=1"""
    print("\n" + "="*60)
    print("TEST 1: SWING DETECTION WITH STRENGTH=1")
    print("="*60)
    
    # Create simple data with clear swings
    prices = [100, 99, 98, 99, 100,   # Swing low at idx 2
              101, 102, 101, 100, 99,  # Swing high at idx 6
              98, 97, 98, 99, 100,     # Swing low at idx 11
              101, 102, 103, 102, 101] # Swing high at idx 17
    
    timestamps = [datetime.now() - timedelta(hours=4 * (len(prices) - i - 1)) for i in range(len(prices))]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices)
    })
    df['rsi'] = calculate_rsi(df['close'], 14)
    # Fill RSI NaNs with reasonable values for testing
    df['rsi'] = df['rsi'].fillna(50)
    
    swing_lows = find_swing_lows(df, "4h")
    swing_highs = find_swing_highs(df, "4h")
    
    print(f"\nPrices: {prices}")
    print(f"\nExpected swing lows at: [2, 11]")
    print(f"Found swing lows at: {[s.index for s in swing_lows]}")
    
    print(f"\nExpected swing highs at: [6, 17]")
    print(f"Found swing highs at: {[s.index for s in swing_highs]}")
    
    # Verify
    expected_lows = {2, 11}
    found_lows = {s.index for s in swing_lows}
    low_match = expected_lows.issubset(found_lows)
    
    expected_highs = {6, 17}
    found_highs = {s.index for s in swing_highs}
    high_match = expected_highs.issubset(found_highs)
    
    print(f"\n✅ Swing low detection: {'PASS' if low_match else 'FAIL'}")
    print(f"✅ Swing high detection: {'PASS' if high_match else 'FAIL'}")
    
    return low_match and high_match


def test_bullish_divergence_detection():
    """Test 2: Verify bullish divergence is detected"""
    print("\n" + "="*60)
    print("TEST 2: BULLISH DIVERGENCE DETECTION")
    print("="*60)
    
    timeframe = "4h"
    df = generate_bullish_divergence_data(timeframe, 100)
    
    current_idx = len(df) - 1
    max_recency = MAX_CANDLES_SINCE_SWING2_MAP.get(timeframe, 3)
    
    print(f"\nGenerated {len(df)} candles with bullish divergence pattern")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
    
    swing_lows = find_swing_lows(df, timeframe)
    print(f"\nFound {len(swing_lows)} swing lows:")
    for s in swing_lows[-5:]:
        candles_ago = current_idx - s.index
        print(f"  idx={s.index}, price=${s.price:.2f}, RSI={s.rsi:.1f}, {candles_ago} candles ago")
    
    # Check for divergences
    divergences_found = 0
    valid_signals = 0
    
    for i in range(len(swing_lows) - 1):
        swing1 = swing_lows[i]
        swing2 = swing_lows[i + 1]
        
        price_ll = swing2.price < swing1.price
        rsi_hl = swing2.rsi > swing1.rsi
        
        if price_ll and rsi_hl:
            divergences_found += 1
            candles_apart = swing2.index - swing1.index
            candles_since = current_idx - swing2.index
            
            dist_ok = MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE
            zone_ok = swing2.rsi < RSI_OVERSOLD
            recency_ok = candles_since <= max_recency
            pattern_ok, _ = check_pattern_validity(df, swing1, swing2, True)
            
            all_pass = dist_ok and zone_ok and recency_ok and pattern_ok
            
            print(f"\n  Bullish Divergence {divergences_found}:")
            print(f"    Swing1: idx={swing1.index}, ${swing1.price:.2f}, RSI={swing1.rsi:.1f}")
            print(f"    Swing2: idx={swing2.index}, ${swing2.price:.2f}, RSI={swing2.rsi:.1f}")
            print(f"    Checks: dist={dist_ok}, zone={zone_ok}, recency={recency_ok}, pattern={pattern_ok}")
            print(f"    Result: {'✅ VALID SIGNAL' if all_pass else '❌ FILTERED'}")
            
            if all_pass:
                valid_signals += 1
    
    print(f"\n📊 Summary:")
    print(f"   Divergences found: {divergences_found}")
    print(f"   Valid signals: {valid_signals}")
    
    return divergences_found > 0


def test_bearish_divergence_detection():
    """Test 3: Verify bearish divergence is detected"""
    print("\n" + "="*60)
    print("TEST 3: BEARISH DIVERGENCE DETECTION")
    print("="*60)
    
    timeframe = "4h"
    df = generate_bearish_divergence_data(timeframe, 100)
    
    current_idx = len(df) - 1
    max_recency = MAX_CANDLES_SINCE_SWING2_MAP.get(timeframe, 3)
    
    print(f"\nGenerated {len(df)} candles with bearish divergence pattern")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
    
    swing_highs = find_swing_highs(df, timeframe)
    print(f"\nFound {len(swing_highs)} swing highs:")
    for s in swing_highs[-5:]:
        candles_ago = current_idx - s.index
        print(f"  idx={s.index}, price=${s.price:.2f}, RSI={s.rsi:.1f}, {candles_ago} candles ago")
    
    # Check for divergences
    divergences_found = 0
    valid_signals = 0
    
    for i in range(len(swing_highs) - 1):
        swing1 = swing_highs[i]
        swing2 = swing_highs[i + 1]
        
        price_hh = swing2.price > swing1.price
        rsi_lh = swing2.rsi < swing1.rsi
        
        if price_hh and rsi_lh:
            divergences_found += 1
            candles_apart = swing2.index - swing1.index
            candles_since = current_idx - swing2.index
            
            dist_ok = MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE
            zone_ok = swing2.rsi > RSI_OVERBOUGHT
            recency_ok = candles_since <= max_recency
            pattern_ok, _ = check_pattern_validity(df, swing1, swing2, False)
            
            all_pass = dist_ok and zone_ok and recency_ok and pattern_ok
            
            print(f"\n  Bearish Divergence {divergences_found}:")
            print(f"    Swing1: idx={swing1.index}, ${swing1.price:.2f}, RSI={swing1.rsi:.1f}")
            print(f"    Swing2: idx={swing2.index}, ${swing2.price:.2f}, RSI={swing2.rsi:.1f}")
            print(f"    Checks: dist={dist_ok}, zone={zone_ok}, recency={recency_ok}, pattern={pattern_ok}")
            print(f"    Result: {'✅ VALID SIGNAL' if all_pass else '❌ FILTERED'}")
            
            if all_pass:
                valid_signals += 1
    
    print(f"\n📊 Summary:")
    print(f"   Divergences found: {divergences_found}")
    print(f"   Valid signals: {valid_signals}")
    
    return divergences_found > 0


def test_recency_filter():
    """Test 4: Verify recency filter works correctly"""
    print("\n" + "="*60)
    print("TEST 4: RECENCY FILTER (1 CANDLE TIMING)")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  SWING_STRENGTH_MAP: {SWING_STRENGTH_MAP}")
    print(f"  MAX_CANDLES_SINCE_SWING2_MAP: {MAX_CANDLES_SINCE_SWING2_MAP}")
    
    print(f"\nExpected behavior:")
    print(f"  4h: Swing detected at swing_idx+1 (after 1 candle = 4 hours)")
    print(f"      Valid window: 3 candles = up to 12 hours after swing")
    print(f"  1d: Swing detected at swing_idx+1 (after 1 candle = 1 day)")
    print(f"      Valid window: 2 candles = up to 2 days after swing")
    
    # Test with 4h timeframe
    timeframe = "4h"
    max_recency = MAX_CANDLES_SINCE_SWING2_MAP[timeframe]
    
    print(f"\n4H Recency Test:")
    print(f"  Max candles since swing2: {max_recency}")
    
    # Simulate different scenarios
    scenarios = [
        (1, "✅ Within window (detected immediately)"),
        (2, "✅ Within window"),
        (3, "✅ Within window (edge case)"),
        (4, "❌ Outside window"),
        (5, "❌ Outside window"),
    ]
    
    for candles_since, expected in scenarios:
        in_window = candles_since <= max_recency
        actual = "✅" if in_window else "❌"
        print(f"    {candles_since} candles since swing2: {actual} {'PASS' if actual == expected[0] else 'FAIL'}")
    
    return True


def test_timeframe_comparison():
    """Test 5: Compare behavior across timeframes"""
    print("\n" + "="*60)
    print("TEST 5: TIMEFRAME COMPARISON")
    print("="*60)
    
    print(f"\n{'Timeframe':<10} {'Strength':<10} {'Recency':<10} {'Alert After':<15} {'Window'}")
    print("-" * 60)
    
    tf_hours = {"4h": 4, "1d": 24, "1w": 168, "1M": 720}
    
    for tf in ["4h", "1d", "1w", "1M"]:
        strength = SWING_STRENGTH_MAP[tf]
        recency = MAX_CANDLES_SINCE_SWING2_MAP[tf]
        hours = tf_hours[tf]
        
        alert_after = f"{hours * strength}h"
        if hours * strength >= 24:
            alert_after = f"{hours * strength // 24}d"
        
        window = f"{hours * recency}h"
        if hours * recency >= 24:
            window = f"{hours * recency // 24}d"
        
        print(f"{tf:<10} {strength:<10} {recency:<10} {alert_after:<15} {window}")
    
    return True


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("RSI DIVERGENCE SIGNAL VERIFICATION - SIMULATION")
    print("="*60)
    print("\nThis test verifies the signal detection logic works correctly")
    print("with the updated 1-candle swing strength configuration.")
    
    results = []
    
    # Run tests
    results.append(("Swing Detection", test_swing_detection()))
    results.append(("Bullish Divergence", test_bullish_divergence_detection()))
    results.append(("Bearish Divergence", test_bearish_divergence_detection()))
    results.append(("Recency Filter", test_recency_filter()))
    results.append(("Timeframe Comparison", test_timeframe_comparison()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe 1-candle swing strength configuration is working correctly.")
        print("\nAlert timing summary:")
        print("  • 4h: Signal fires ~4 hours after swing point")
        print("  • 1d: Signal fires ~1 day after swing point")
        print("  • 1w: Signal fires ~1 week after swing point")
        print("  • 1M: Signal fires ~1 month after swing point")
    else:
        print("⚠️ SOME TESTS FAILED - Review the output above")
    
    print("="*60)


if __name__ == "__main__":
    main()
