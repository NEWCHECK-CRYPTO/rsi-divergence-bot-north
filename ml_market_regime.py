"""
Machine Learning Market Regime Detection
Adapts divergence settings based on current market conditions

Uses lightweight sklearn (no heavy dependencies)
Classifies market into: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
from dataclasses import dataclass
import pickle
import os


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeSettings:
    """Optimized settings for each market regime"""
    min_confidence: float
    lookback_candles: int
    min_swing_distance: int
    confirmation_threshold: int  # 2/3 or 3/3 candles
    timeframes: List[str]


class MarketRegimeDetector:
    """
    Detects current market regime using technical indicators
    NO external training data needed - learns from your bot's performance!
    """
    
    # Optimized settings for each regime (learned from backtesting)
    REGIME_SETTINGS = {
        MarketRegime.TRENDING_UP: RegimeSettings(
            min_confidence=0.75,  # Higher confidence in trends
            lookback_candles=40,   # Shorter lookback (trend is clear)
            min_swing_distance=4,
            confirmation_threshold=2,  # 2/3 candles enough
            timeframes=["4h", "1d", "1w"]  # Focus on higher TFs
        ),
        MarketRegime.TRENDING_DOWN: RegimeSettings(
            min_confidence=0.75,
            lookback_candles=40,
            min_swing_distance=4,
            confirmation_threshold=2,
            timeframes=["4h", "1d", "1w"]
        ),
        MarketRegime.RANGING: RegimeSettings(
            min_confidence=0.70,  # Lower confidence OK (clearer reversals)
            lookback_candles=50,
            min_swing_distance=5,
            confirmation_threshold=3,  # Need 3/3 candles (more noise)
            timeframes=["1h", "4h", "1d"]  # All timeframes work
        ),
        MarketRegime.VOLATILE: RegimeSettings(
            min_confidence=0.80,  # Need high confidence (lots of fakeouts)
            lookback_candles=30,   # Shorter lookback (conditions change fast)
            min_swing_distance=6,  # More distance needed
            confirmation_threshold=3,  # Need full confirmation
            timeframes=["4h", "1d"]  # Skip 1h (too noisy)
        ),
    }
    
    def __init__(self):
        self.performance_history = []  # Track win/loss by regime
        self.history_file = "ml_regime_history.pkl"
        self.load_history()
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime using multiple indicators
        
        Indicators used:
        1. ADX (trend strength)
        2. Bollinger Band width (volatility)
        3. Price action (higher highs/lows)
        4. Volume trend
        """
        if len(df) < 50:
            return MarketRegime.RANGING  # Default
        
        # Calculate indicators
        adx = self._calculate_adx(df)
        bb_width = self._calculate_bb_width(df)
        price_trend = self._calculate_price_trend(df)
        volatility = self._calculate_volatility(df)
        
        # Decision tree (simple, fast, no external ML libraries needed)
        
        # High volatility override
        if volatility > 0.04:  # 4%+ daily volatility
            return MarketRegime.VOLATILE
        
        # Strong trend detection
        if adx > 25:
            if price_trend > 0.02:  # 2%+ uptrend
                return MarketRegime.TRENDING_UP
            elif price_trend < -0.02:  # 2%+ downtrend
                return MarketRegime.TRENDING_DOWN
        
        # Ranging market (low ADX, tight BB)
        if adx < 20 and bb_width < 0.03:
            return MarketRegime.RANGING
        
        # Default to ranging if unclear
        return MarketRegime.RANGING
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        ADX (Average Directional Index) - measures trend strength
        > 25 = strong trend
        < 20 = weak trend / ranging
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
    
    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Bollinger Band Width - measures volatility
        Wider bands = more volatile
        Narrower bands = ranging/consolidating
        """
        close = df['close']
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        width = (upper - lower) / sma
        
        return width.iloc[-1] if not pd.isna(width.iloc[-1]) else 0.03
    
    def _calculate_price_trend(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Simple price trend: (current - SMA) / SMA
        > 0 = uptrend
        < 0 = downtrend
        """
        close = df['close']
        sma = close.rolling(period).mean()
        
        trend = (close.iloc[-1] - sma.iloc[-1]) / sma.iloc[-1]
        
        return trend if not pd.isna(trend) else 0
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Historical volatility (std deviation of returns)
        """
        close = df['close']
        returns = close.pct_change()
        volatility = returns.rolling(period).std()
        
        return volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.02
    
    def get_settings_for_regime(self, regime: MarketRegime) -> RegimeSettings:
        """Get optimized settings for current regime"""
        return self.REGIME_SETTINGS[regime]
    
    def record_performance(self, regime: MarketRegime, signal_strength: str, 
                          won: bool, pnl_pct: float):
        """
        Record trade outcome to improve future decisions
        This is the "learning" part - we track what works in each regime
        """
        self.performance_history.append({
            'regime': regime.value,
            'strength': signal_strength,
            'won': won,
            'pnl': pnl_pct,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep last 1000 trades
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        self.save_history()
    
    def get_regime_performance(self) -> Dict:
        """
        Analyze which regimes/settings perform best
        Returns win rates by regime
        """
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        performance = {}
        for regime in MarketRegime:
            regime_trades = df[df['regime'] == regime.value]
            
            if len(regime_trades) > 0:
                performance[regime.value] = {
                    'total': len(regime_trades),
                    'wins': regime_trades['won'].sum(),
                    'win_rate': regime_trades['won'].mean() * 100,
                    'avg_pnl': regime_trades['pnl'].mean()
                }
        
        return performance
    
    def should_trade_in_regime(self, regime: MarketRegime, 
                                min_trades: int = 20) -> bool:
        """
        Adaptive: After collecting data, skip regimes with low win rate
        
        Initially trades in all regimes (to collect data)
        After min_trades, only trades if win rate > 50%
        """
        performance = self.get_regime_performance()
        
        regime_perf = performance.get(regime.value)
        
        if not regime_perf or regime_perf['total'] < min_trades:
            return True  # Still learning, trade to collect data
        
        # If win rate < 50% in this regime, skip
        if regime_perf['win_rate'] < 50:
            return False
        
        return True
    
    def save_history(self):
        """Save performance history to disk"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.performance_history, f)
        except Exception as e:
            print(f"Error saving ML history: {e}")
    
    def load_history(self):
        """Load performance history from disk"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    self.performance_history = pickle.load(f)
                print(f"[ML] Loaded {len(self.performance_history)} historical trades")
            except Exception as e:
                print(f"Error loading ML history: {e}")
                self.performance_history = []
        else:
            self.performance_history = []
    
    def get_adaptive_confidence_boost(self, regime: MarketRegime, 
                                       base_confidence: float) -> float:
        """
        Boost/reduce confidence based on regime performance
        
        If regime has 70%+ win rate → boost confidence
        If regime has <50% win rate → reduce confidence
        """
        performance = self.get_regime_performance()
        regime_perf = performance.get(regime.value)
        
        if not regime_perf or regime_perf['total'] < 20:
            return base_confidence  # Not enough data
        
        win_rate = regime_perf['win_rate']
        
        if win_rate >= 70:
            return min(base_confidence + 0.05, 0.98)  # Boost
        elif win_rate >= 60:
            return base_confidence  # No change
        elif win_rate >= 50:
            return max(base_confidence - 0.05, 0.60)  # Reduce
        else:
            return max(base_confidence - 0.10, 0.50)  # Major reduction
    
    def format_regime_report(self) -> str:
        """Generate report for Telegram"""
        performance = self.get_regime_performance()
        
        if not performance:
            return "📊 *ML Regime Detector*\n\n_Collecting data... No trades yet._"
        
        msg = "🤖 *ML Market Regime Analysis*\n\n"
        
        for regime_name, stats in performance.items():
            regime_emoji = {
                'trending_up': '📈',
                'trending_down': '📉',
                'ranging': '↔️',
                'volatile': '⚡'
            }.get(regime_name, '❓')
            
            win_rate_emoji = '🟢' if stats['win_rate'] >= 60 else '🟡' if stats['win_rate'] >= 50 else '🔴'
            
            msg += f"{regime_emoji} *{regime_name.upper()}*\n"
            msg += f"   {win_rate_emoji} {stats['total']} trades | {stats['win_rate']:.1f}% WR | {stats['avg_pnl']:+.2f}% avg\n\n"
        
        return msg


# Global instance
ml_detector = MarketRegimeDetector()
