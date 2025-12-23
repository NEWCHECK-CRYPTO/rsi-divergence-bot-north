"""
Trade Journal Module
- Stores all signals sent
- Tracks outcomes (TP hit, SL hit, or manual close)
- Provides RAG-powered analysis of performance
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pytz

from config import TIMEZONE

SL_TZ = pytz.timezone(TIMEZONE)

# Journal file path
JOURNAL_FILE = "trade_journal.json"


class TradeOutcome(Enum):
    OPEN = "open"           # Still waiting
    WIN_TP = "win_tp"       # Hit take profit
    WIN_MANUAL = "win_manual"  # Closed manually in profit
    LOSS_SL = "loss_sl"     # Hit stop loss
    LOSS_MANUAL = "loss_manual"  # Closed manually in loss
    EXPIRED = "expired"     # Signal expired without action


@dataclass
class JournalEntry:
    id: str                     # Unique ID
    timestamp: str              # When signal was generated
    symbol: str
    timeframe: str
    direction: str              # LONG or SHORT
    signal_strength: str        # strong, medium, early
    divergence_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    
    # Tracking fields
    outcome: str = "open"       # TradeOutcome value
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl_percent: Optional[float] = None
    notes: Optional[str] = None
    
    # Analysis fields
    swings_apart: int = 0
    price_change_pct: float = 0.0
    rsi_change: float = 0.0


class TradeJournal:
    def __init__(self):
        self.entries: List[JournalEntry] = []
        self.load()
    
    def load(self):
        """Load journal from file"""
        if os.path.exists(JOURNAL_FILE):
            try:
                with open(JOURNAL_FILE, 'r') as f:
                    data = json.load(f)
                    self.entries = [JournalEntry(**e) for e in data]
                print(f"[Journal] Loaded {len(self.entries)} entries")
            except Exception as e:
                print(f"[Journal] Error loading: {e}")
                self.entries = []
        else:
            self.entries = []
    
    def save(self):
        """Save journal to file"""
        try:
            with open(JOURNAL_FILE, 'w') as f:
                json.dump([asdict(e) for e in self.entries], f, indent=2)
        except Exception as e:
            print(f"[Journal] Error saving: {e}")
    
    def generate_id(self) -> str:
        """Generate unique trade ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_entry(self, signal) -> str:
        """Add a new signal to journal"""
        is_bullish = "BULLISH" in signal.divergence.divergence_type.value.upper()
        
        entry = JournalEntry(
            id=self.generate_id(),
            timestamp=signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            symbol=signal.symbol,
            timeframe=signal.signal_tf,
            direction="LONG" if is_bullish else "SHORT",
            signal_strength=signal.signal_strength.value,
            divergence_type=signal.divergence.divergence_type.value,
            entry_price=signal.divergence.current_price,
            stop_loss=self._calc_sl(signal),
            take_profit=self._calc_tp(signal),
            confidence=signal.total_confidence,
            swings_apart=signal.divergence.candles_apart,
            price_change_pct=signal.divergence.price_change_pct,
            rsi_change=signal.divergence.rsi_change
        )
        
        self.entries.append(entry)
        self.save()
        
        return entry.id
    
    def _calc_sl(self, signal) -> float:
        """Calculate stop loss"""
        is_bullish = "BULLISH" in signal.divergence.divergence_type.value.upper()
        price = signal.divergence.current_price
        swing_price = min(signal.divergence.swing1.price, signal.divergence.swing2.price) if is_bullish else max(signal.divergence.swing1.price, signal.divergence.swing2.price)
        
        if is_bullish:
            return swing_price * 0.99
        else:
            return swing_price * 1.01
    
    def _calc_tp(self, signal) -> float:
        """Calculate take profit"""
        is_bullish = "BULLISH" in signal.divergence.divergence_type.value.upper()
        price = signal.divergence.current_price
        
        if is_bullish:
            return price * 1.04
        else:
            return price * 0.96
    
    def update_outcome(self, trade_id: str, outcome: str, exit_price: float = None, notes: str = None) -> bool:
        """Update trade outcome"""
        for entry in self.entries:
            if entry.id == trade_id:
                entry.outcome = outcome
                entry.exit_time = datetime.now(SL_TZ).strftime("%Y-%m-%d %H:%M:%S")
                
                if exit_price:
                    entry.exit_price = exit_price
                    if entry.direction == "LONG":
                        entry.pnl_percent = ((exit_price - entry.entry_price) / entry.entry_price) * 100
                    else:
                        entry.pnl_percent = ((entry.entry_price - exit_price) / entry.entry_price) * 100
                elif outcome == "win_tp":
                    entry.exit_price = entry.take_profit
                    entry.pnl_percent = 4.0 if entry.direction == "LONG" else 4.0
                elif outcome == "loss_sl":
                    entry.exit_price = entry.stop_loss
                    entry.pnl_percent = -2.0
                
                if notes:
                    entry.notes = notes
                
                self.save()
                return True
        return False
    
    def get_open_trades(self) -> List[JournalEntry]:
        """Get all open trades"""
        return [e for e in self.entries if e.outcome == "open"]
    
    def get_closed_trades(self) -> List[JournalEntry]:
        """Get all closed trades"""
        return [e for e in self.entries if e.outcome != "open"]
    
    def get_recent_trades(self, days: int = 30) -> List[JournalEntry]:
        """Get trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self.entries if datetime.strptime(e.timestamp, "%Y-%m-%d %H:%M:%S") > cutoff]
    
    def get_stats(self, trades: List[JournalEntry] = None) -> Dict:
        """Calculate statistics"""
        if trades is None:
            trades = self.get_closed_trades()
        
        if not trades:
            return {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "best_trade": 0,
                "worst_trade": 0
            }
        
        wins = [t for t in trades if t.outcome in ["win_tp", "win_manual"]]
        losses = [t for t in trades if t.outcome in ["loss_sl", "loss_manual"]]
        
        total_win_pnl = sum(t.pnl_percent or 0 for t in wins)
        total_loss_pnl = abs(sum(t.pnl_percent or 0 for t in losses))
        
        all_pnl = [t.pnl_percent for t in trades if t.pnl_percent is not None]
        
        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades) * 100) if trades else 0,
            "total_pnl": sum(all_pnl),
            "avg_win": (total_win_pnl / len(wins)) if wins else 0,
            "avg_loss": (total_loss_pnl / len(losses)) if losses else 0,
            "profit_factor": (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else float('inf'),
            "best_trade": max(all_pnl) if all_pnl else 0,
            "worst_trade": min(all_pnl) if all_pnl else 0
        }
    
    def get_stats_by_strength(self) -> Dict:
        """Get stats broken down by signal strength"""
        closed = self.get_closed_trades()
        
        result = {}
        for strength in ["strong", "medium", "early"]:
            trades = [t for t in closed if t.signal_strength == strength]
            result[strength] = self.get_stats(trades)
        
        return result
    
    def get_stats_by_direction(self) -> Dict:
        """Get stats broken down by direction"""
        closed = self.get_closed_trades()
        
        result = {}
        for direction in ["LONG", "SHORT"]:
            trades = [t for t in closed if t.direction == direction]
            result[direction] = self.get_stats(trades)
        
        return result
    
    def get_stats_by_timeframe(self) -> Dict:
        """Get stats broken down by timeframe"""
        closed = self.get_closed_trades()
        
        timeframes = list(set(t.timeframe for t in closed))
        result = {}
        for tf in timeframes:
            trades = [t for t in closed if t.timeframe == tf]
            result[tf] = self.get_stats(trades)
        
        return result
    
    def get_stats_by_symbol(self) -> Dict:
        """Get stats broken down by symbol"""
        closed = self.get_closed_trades()
        
        symbols = list(set(t.symbol for t in closed))
        result = {}
        for symbol in symbols:
            trades = [t for t in closed if t.symbol == symbol]
            result[symbol] = self.get_stats(trades)
        
        # Sort by number of trades
        result = dict(sorted(result.items(), key=lambda x: x[1]["total"], reverse=True))
        
        return result
    
    def generate_analysis_prompt(self) -> str:
        """Generate prompt for RAG analysis"""
        stats = self.get_stats()
        by_strength = self.get_stats_by_strength()
        by_direction = self.get_stats_by_direction()
        by_tf = self.get_stats_by_timeframe()
        
        recent = self.get_recent_trades(30)
        recent_stats = self.get_stats([t for t in recent if t.outcome != "open"])
        
        prompt = f"""Analyze this trading bot's performance and provide actionable recommendations.

OVERALL STATISTICS:
- Total Closed Trades: {stats['total']}
- Win Rate: {stats['win_rate']:.1f}%
- Total P&L: {stats['total_pnl']:.2f}%
- Profit Factor: {stats['profit_factor']:.2f}
- Average Win: {stats['avg_win']:.2f}%
- Average Loss: {stats['avg_loss']:.2f}%
- Best Trade: {stats['best_trade']:.2f}%
- Worst Trade: {stats['worst_trade']:.2f}%

BY SIGNAL STRENGTH:
- 🟢 STRONG: {by_strength['strong']['total']} trades, {by_strength['strong']['win_rate']:.1f}% WR, {by_strength['strong']['total_pnl']:.2f}% P&L
- 🟡 MEDIUM: {by_strength['medium']['total']} trades, {by_strength['medium']['win_rate']:.1f}% WR, {by_strength['medium']['total_pnl']:.2f}% P&L
- 🔴 EARLY: {by_strength['early']['total']} trades, {by_strength['early']['win_rate']:.1f}% WR, {by_strength['early']['total_pnl']:.2f}% P&L

BY DIRECTION:
- LONG: {by_direction.get('LONG', {}).get('total', 0)} trades, {by_direction.get('LONG', {}).get('win_rate', 0):.1f}% WR
- SHORT: {by_direction.get('SHORT', {}).get('total', 0)} trades, {by_direction.get('SHORT', {}).get('win_rate', 0):.1f}% WR

BY TIMEFRAME:
"""
        for tf, tf_stats in by_tf.items():
            prompt += f"- {tf}: {tf_stats['total']} trades, {tf_stats['win_rate']:.1f}% WR\n"
        
        prompt += f"""
LAST 30 DAYS:
- Trades: {recent_stats['total']}
- Win Rate: {recent_stats['win_rate']:.1f}%
- P&L: {recent_stats['total_pnl']:.2f}%

Please provide:
1. Overall assessment (is this profitable?)
2. Which signal strength performs best?
3. Should we trade LONG, SHORT, or both?
4. Which timeframes work best?
5. Top 3 specific recommendations to improve
6. Any patterns or concerns you notice
"""
        return prompt
    
    def format_stats_message(self) -> str:
        """Format stats for Telegram message"""
        stats = self.get_stats()
        open_trades = self.get_open_trades()
        
        if stats['total'] == 0 and len(open_trades) == 0:
            return "📊 *Trade Journal*\n\n_No trades recorded yet. Signals will be automatically logged._"
        
        by_strength = self.get_stats_by_strength()
        by_direction = self.get_stats_by_direction()
        
        msg = f"""📊 *Trade Journal Statistics*

{'━'*30}
📈 OVERALL PERFORMANCE
{'━'*30}
Total Trades: {stats['total']} closed, {len(open_trades)} open
Win Rate: *{stats['win_rate']:.1f}%*
Total P&L: *{stats['total_pnl']:+.2f}%*
Profit Factor: *{stats['profit_factor']:.2f}*

Avg Win: +{stats['avg_win']:.2f}%
Avg Loss: {stats['avg_loss']:.2f}%
Best: {stats['best_trade']:+.2f}% | Worst: {stats['worst_trade']:+.2f}%

{'━'*30}
🎯 BY SIGNAL STRENGTH
{'━'*30}
🟢 STRONG: {by_strength['strong']['total']} trades | {by_strength['strong']['win_rate']:.0f}% WR | {by_strength['strong']['total_pnl']:+.1f}%
🟡 MEDIUM: {by_strength['medium']['total']} trades | {by_strength['medium']['win_rate']:.0f}% WR | {by_strength['medium']['total_pnl']:+.1f}%
🔴 EARLY: {by_strength['early']['total']} trades | {by_strength['early']['win_rate']:.0f}% WR | {by_strength['early']['total_pnl']:+.1f}%

{'━'*30}
📈 BY DIRECTION
{'━'*30}
LONG: {by_direction.get('LONG', {}).get('total', 0)} trades | {by_direction.get('LONG', {}).get('win_rate', 0):.0f}% WR
SHORT: {by_direction.get('SHORT', {}).get('total', 0)} trades | {by_direction.get('SHORT', {}).get('win_rate', 0):.0f}% WR
"""
        
        return msg
    
    def format_open_trades(self) -> str:
        """Format open trades for Telegram"""
        open_trades = self.get_open_trades()
        
        if not open_trades:
            return "📭 *No Open Trades*\n\n_All signals have been resolved._"
        
        msg = f"📊 *Open Trades ({len(open_trades)})*\n\n"
        
        for i, t in enumerate(open_trades[-10:], 1):  # Show last 10
            emoji = "🟢" if t.signal_strength == "strong" else "🟡" if t.signal_strength == "medium" else "🔴"
            direction = "📈" if t.direction == "LONG" else "📉"
            
            msg += f"{emoji} `{t.id}` {direction} {t.symbol} {t.timeframe}\n"
            msg += f"   Entry: ${t.entry_price:,.2f} | SL: ${t.stop_loss:,.2f} | TP: ${t.take_profit:,.2f}\n\n"
        
        if len(open_trades) > 10:
            msg += f"_...and {len(open_trades) - 10} more_\n"
        
        msg += "\n_Use /close <id> <win|loss> to record outcome_"
        
        return msg


# Global journal instance
journal = TradeJournal()
