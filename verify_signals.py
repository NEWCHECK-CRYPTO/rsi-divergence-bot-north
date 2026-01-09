#!/usr/bin/env python3
"""
RSI Divergence Signal Verification Script
==========================================
Tests whether signals are being generated correctly with the new
1-candle swing strength configuration.

Usage:
    python verify_signals.py [SYMBOL] [TIMEFRAME]
    
Examples:
    python verify_signals.py              # Test BTC/USDT on all timeframes
    python verify_signals.py ETH/USDT 4h  # Test specific symbol/timeframe
    python verify_signals.py BTC 1d       # Test BTC/USDT on 1d
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock config if not available
try:
    from config import *
except ImportError:
    # Default config values
    EXCHANGE = "bybit"
    TIMEZONE = "Asia/Colombo"
    SCAN_TIMEFRAMES = ["4h", "1d", "1w", "1M"]
    RSI_PERIOD = 14
    ALERT_COOLDOWN = 1800
    TOP_COINS_COUNT = 200
    QUOTE_CURRENCY = "USDT"
    EXCLUDED_SYMBOLS = []
    EXCLUDE_LEVERAGED = True
    LOOKBACK_CANDLES = 200
    SWING_STRENGTH_BARS = 2

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import pytz

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
            return False, f"Price invalid: ${min_close:.4f} < ${swing2.price:.4f}"
    else:
        max_close = middle['close'].max()
        if max_close > swing2.price:
            return False, f"Price invalid: ${max_close:.4f} > ${swing2.price:.4f}"
    
    return True, "Pattern intact"


# =============================================================================
# MAIN VERIFICATION FUNCTION
# =============================================================================

def verify_signals(symbol: str, timeframe: str, limit: int = 200) -> Dict:
    """
    Verify signal detection for a symbol/timeframe combination.
    Returns detailed analysis of what signals are found and why.
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING: {symbol} {timeframe.upper()}")
    print(f"{'='*60}")
    
    # Configuration
    strength = SWING_STRENGTH_MAP.get(timeframe, 1)
    max_recency = MAX_CANDLES_SINCE_SWING2_MAP.get(timeframe, 3)
    
    print(f"\n📋 CONFIGURATION:")
    print(f"   Swing Strength: {strength} candle(s) each side")
    print(f"   Max Recency: {max_recency} candles from current")
    print(f"   RSI Oversold: < {RSI_OVERSOLD}")
    print(f"   RSI Overbought: > {RSI_OVERBOUGHT}")
    print(f"   Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles")
    
    # Fetch data
    print(f"\n📊 Fetching {limit} candles...")
    
    try:
        # Try binance first (more likely to work with network restrictions)
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        exchange.load_markets()
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print("❌ No data returned")
            return {"error": "No data"}
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        
        print(f"   ✓ Got {len(df)} candles")
        print(f"   First: {df['timestamp'].iloc[0]}")
        print(f"   Last:  {df['timestamp'].iloc[-1]}")
        print(f"   Current Price: ${df['close'].iloc[-1]:,.4f}")
        print(f"   Current RSI: {df['rsi'].iloc[-1]:.1f}")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return {"error": str(e)}
    
    current_idx = len(df) - 1
    
    # Find swing points
    print(f"\n🔍 FINDING SWING POINTS (strength={strength})...")
    
    swing_lows = find_swing_lows(df, timeframe)
    swing_highs = find_swing_highs(df, timeframe)
    
    print(f"   Found {len(swing_lows)} swing lows")
    print(f"   Found {len(swing_highs)} swing highs")
    
    # Show recent swing lows
    print(f"\n📉 RECENT SWING LOWS (last 10):")
    print(f"   {'Idx':<6} {'Candles Ago':<12} {'Price':<14} {'RSI':<8} {'Timestamp'}")
    print(f"   {'-'*65}")
    for s in swing_lows[-10:]:
        candles_ago = current_idx - s.index
        in_window = "✓" if candles_ago <= max_recency else ""
        print(f"   {s.index:<6} {candles_ago:<12} ${s.price:<12.4f} {s.rsi:<8.1f} {s.timestamp} {in_window}")
    
    # Show recent swing highs
    print(f"\n📈 RECENT SWING HIGHS (last 10):")
    print(f"   {'Idx':<6} {'Candles Ago':<12} {'Price':<14} {'RSI':<8} {'Timestamp'}")
    print(f"   {'-'*65}")
    for s in swing_highs[-10:]:
        candles_ago = current_idx - s.index
        in_window = "✓" if candles_ago <= max_recency else ""
        print(f"   {s.index:<6} {candles_ago:<12} ${s.price:<12.4f} {s.rsi:<8.1f} {s.timestamp} {in_window}")
    
    # Check for divergences
    print(f"\n🔄 CHECKING DIVERGENCES...")
    
    results = {
        "symbol": symbol,
        "timeframe": timeframe,
        "config": {
            "swing_strength": strength,
            "max_recency": max_recency,
            "rsi_oversold": RSI_OVERSOLD,
            "rsi_overbought": RSI_OVERBOUGHT
        },
        "data": {
            "candles": len(df),
            "swing_lows": len(swing_lows),
            "swing_highs": len(swing_highs)
        },
        "bullish_divergences": [],
        "bearish_divergences": [],
        "valid_signals": []
    }
    
    # Check bullish divergences
    print(f"\n🟢 BULLISH DIVERGENCE CHECK:")
    print(f"   Looking for: Price LL + RSI HL, RSI < {RSI_OVERSOLD}")
    
    bullish_count = 0
    for i in range(len(swing_lows) - 1):
        swing1 = swing_lows[i]
        swing2 = swing_lows[i + 1]
        
        # Check for basic divergence pattern
        price_ll = swing2.price < swing1.price
        rsi_hl = swing2.rsi > swing1.rsi
        
        if price_ll and rsi_hl:
            bullish_count += 1
            candles_apart = swing2.index - swing1.index
            candles_since = current_idx - swing2.index
            
            # Check all conditions
            checks = {
                "distance": MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE,
                "price_ll": price_ll,
                "rsi_hl": rsi_hl,
                "rsi_zone": swing2.rsi < RSI_OVERSOLD,
                "recency": candles_since <= max_recency,
                "pattern": check_pattern_validity(df, swing1, swing2, True)[0]
            }
            
            all_pass = all(checks.values())
            
            div_info = {
                "swing1": {"idx": swing1.index, "price": swing1.price, "rsi": swing1.rsi, "ts": str(swing1.timestamp)},
                "swing2": {"idx": swing2.index, "price": swing2.price, "rsi": swing2.rsi, "ts": str(swing2.timestamp)},
                "candles_apart": candles_apart,
                "candles_since": candles_since,
                "checks": checks,
                "all_pass": all_pass
            }
            
            results["bullish_divergences"].append(div_info)
            
            if all_pass:
                results["valid_signals"].append({"type": "BULLISH", **div_info})
            
            # Print details
            status = "✅ VALID" if all_pass else "❌ FILTERED"
            print(f"\n   {status} Divergence #{bullish_count}:")
            print(f"      Swing1: idx={swing1.index}, ${swing1.price:.4f}, RSI={swing1.rsi:.1f}")
            print(f"      Swing2: idx={swing2.index}, ${swing2.price:.4f}, RSI={swing2.rsi:.1f}")
            print(f"      Candles apart: {candles_apart}")
            print(f"      Candles since swing2: {candles_since} (max={max_recency})")
            print(f"      Checks:")
            for check_name, passed in checks.items():
                icon = "✓" if passed else "✗"
                print(f"         {icon} {check_name}")
    
    if bullish_count == 0:
        print("   No bullish divergences found")
    
    # Check bearish divergences
    print(f"\n🔴 BEARISH DIVERGENCE CHECK:")
    print(f"   Looking for: Price HH + RSI LH, RSI > {RSI_OVERBOUGHT}")
    
    bearish_count = 0
    for i in range(len(swing_highs) - 1):
        swing1 = swing_highs[i]
        swing2 = swing_highs[i + 1]
        
        # Check for basic divergence pattern
        price_hh = swing2.price > swing1.price
        rsi_lh = swing2.rsi < swing1.rsi
        
        if price_hh and rsi_lh:
            bearish_count += 1
            candles_apart = swing2.index - swing1.index
            candles_since = current_idx - swing2.index
            
            # Check all conditions
            checks = {
                "distance": MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE,
                "price_hh": price_hh,
                "rsi_lh": rsi_lh,
                "rsi_zone": swing2.rsi > RSI_OVERBOUGHT,
                "recency": candles_since <= max_recency,
                "pattern": check_pattern_validity(df, swing1, swing2, False)[0]
            }
            
            all_pass = all(checks.values())
            
            div_info = {
                "swing1": {"idx": swing1.index, "price": swing1.price, "rsi": swing1.rsi, "ts": str(swing1.timestamp)},
                "swing2": {"idx": swing2.index, "price": swing2.price, "rsi": swing2.rsi, "ts": str(swing2.timestamp)},
                "candles_apart": candles_apart,
                "candles_since": candles_since,
                "checks": checks,
                "all_pass": all_pass
            }
            
            results["bearish_divergences"].append(div_info)
            
            if all_pass:
                results["valid_signals"].append({"type": "BEARISH", **div_info})
            
            # Print details
            status = "✅ VALID" if all_pass else "❌ FILTERED"
            print(f"\n   {status} Divergence #{bearish_count}:")
            print(f"      Swing1: idx={swing1.index}, ${swing1.price:.4f}, RSI={swing1.rsi:.1f}")
            print(f"      Swing2: idx={swing2.index}, ${swing2.price:.4f}, RSI={swing2.rsi:.1f}")
            print(f"      Candles apart: {candles_apart}")
            print(f"      Candles since swing2: {candles_since} (max={max_recency})")
            print(f"      Checks:")
            for check_name, passed in checks.items():
                icon = "✓" if passed else "✗"
                print(f"         {icon} {check_name}")
    
    if bearish_count == 0:
        print("   No bearish divergences found")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {symbol} {timeframe.upper()}")
    print(f"{'='*60}")
    print(f"   Total swing lows: {len(swing_lows)}")
    print(f"   Total swing highs: {len(swing_highs)}")
    print(f"   Bullish divergences found: {bullish_count}")
    print(f"   Bearish divergences found: {bearish_count}")
    print(f"   Valid signals (all checks pass): {len(results['valid_signals'])}")
    
    if results['valid_signals']:
        print(f"\n   🎯 VALID SIGNALS:")
        for sig in results['valid_signals']:
            print(f"      {sig['type']}: Swing2 at idx {sig['swing2']['idx']}, "
                  f"RSI={sig['swing2']['rsi']:.1f}, {sig['candles_since']} candles ago")
    
    return results


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("RSI DIVERGENCE SIGNAL VERIFICATION")
    print("="*60)
    print(f"\nConfiguration (1 candle confirmation):")
    print(f"  SWING_STRENGTH_MAP: {SWING_STRENGTH_MAP}")
    print(f"  MAX_CANDLES_SINCE_SWING2_MAP: {MAX_CANDLES_SINCE_SWING2_MAP}")
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if len(args) >= 2:
        # Specific symbol and timeframe
        symbol = args[0].upper()
        if '/' not in symbol:
            symbol = symbol + '/USDT'
        timeframe = args[1].lower()
        if timeframe == "1m":
            timeframe = "1M"
        elif timeframe == "1w":
            timeframe = "1w"
        
        verify_signals(symbol, timeframe)
    
    elif len(args) == 1:
        # Specific symbol, all timeframes
        symbol = args[0].upper()
        if '/' not in symbol:
            symbol = symbol + '/USDT'
        
        for tf in ["4h", "1d", "1w", "1M"]:
            verify_signals(symbol, tf)
    
    else:
        # Default: Test BTC/USDT on 4h and 1d (the main timeframes you care about)
        print("\nTesting BTC/USDT on 4h and 1d timeframes...")
        
        for tf in ["4h", "1d"]:
            verify_signals("BTC/USDT", tf)
        
        print("\n" + "="*60)
        print("Additional tests you can run:")
        print("="*60)
        print("  python verify_signals.py ETH/USDT 4h")
        print("  python verify_signals.py SOL 1d")
        print("  python verify_signals.py BTC")  # all timeframes


if __name__ == "__main__":
    main()
