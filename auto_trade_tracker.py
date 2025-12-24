"""
Automatic Trade Outcome Tracker
Monitors open trades and automatically records when TP/SL is hit
"""

import asyncio
import time
from typing import Dict, List
from datetime import datetime
import ccxt

from trade_journal import journal, TradeOutcome
from ml_market_regime import ml_detector, MarketRegime
from config import EXCHANGE, TIMEZONE
import pytz

SL_TZ = pytz.timezone(TIMEZONE)


class AutoTradeTracker:
    """
    Monitors all open trades in background
    Automatically updates journal when TP/SL hit
    Feeds results to ML for learning
    """
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.monitoring = True
        self.check_interval = 60  # Check every 60 seconds
        
    async def start_monitoring(self, telegram_bot):
        """
        Background task that runs forever
        Checks open trades every minute
        """
        print(f"[{self._now()}] 🤖 Auto-tracker started")
        
        while self.monitoring:
            try:
                await self.check_all_trades(telegram_bot)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"[{self._now()}] Tracker error: {e}")
                await asyncio.sleep(60)
    
    async def check_all_trades(self, telegram_bot):
        """Check all open trades for TP/SL hits"""
        open_trades = journal.get_open_trades()
        
        if not open_trades:
            return
        
        print(f"[{self._now()}] 📊 Monitoring {len(open_trades)} open trades...")
        
        for trade in open_trades:
            try:
                # Get current price
                current_price = await self._get_current_price(trade.symbol)
                
                if current_price is None:
                    continue
                
                # Check if TP or SL hit
                outcome = self._check_outcome(trade, current_price)
                
                if outcome:
                    # Update journal
                    exit_price = current_price
                    journal.update_outcome(
                        trade.id, 
                        outcome.value,
                        exit_price,
                        f"Auto-closed at {self._now()}"
                    )
                    
                    # Record for ML learning
                    won = outcome in [TradeOutcome.WIN_TP, TradeOutcome.WIN_MANUAL]
                    pnl = trade.pnl_percent if trade.pnl_percent else self._calc_pnl(trade, exit_price)
                    
                    if hasattr(trade, 'regime') and trade.regime != 'unknown':
                        ml_detector.record_performance(
                            MarketRegime(trade.regime),
                            trade.signal_strength,
                            won,
                            pnl
                        )
                    
                    # Notify user on Telegram
                    await self._send_notification(telegram_bot, trade, outcome, current_price)
                    
                    print(f"[{self._now()}] ✅ Trade {trade.id} closed: {outcome.value}")
                
            except Exception as e:
                print(f"[{self._now()}] Error checking {trade.id}: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Fetch current price from exchange"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def _check_outcome(self, trade, current_price: float) -> TradeOutcome:
        """
        Check if trade hit TP or SL
        
        Returns:
        - WIN_TP if TP hit
        - LOSS_SL if SL hit  
        - None if still open
        """
        
        if trade.direction == "LONG":
            # Long trade
            if current_price >= trade.take_profit:
                return TradeOutcome.WIN_TP
            elif current_price <= trade.stop_loss:
                return TradeOutcome.LOSS_SL
        else:
            # Short trade
            if current_price <= trade.take_profit:
                return TradeOutcome.WIN_TP
            elif current_price >= trade.stop_loss:
                return TradeOutcome.LOSS_SL
        
        return None
    
    def _calc_pnl(self, trade, exit_price: float) -> float:
        """Calculate P&L percentage"""
        if trade.direction == "LONG":
            return ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            return ((trade.entry_price - exit_price) / trade.entry_price) * 100
    
    async def _send_notification(self, telegram_bot, trade, outcome: TradeOutcome, exit_price: float):
        """Send Telegram notification when trade closes"""
        
        pnl = self._calc_pnl(trade, exit_price)
        
        if outcome in [TradeOutcome.WIN_TP, TradeOutcome.WIN_MANUAL]:
            emoji = "🎉"
            result = "WIN"
        else:
            emoji = "❌"
            result = "LOSS"
        
        msg = f"""{emoji} *Trade Closed - {result}*

📝 ID: `{trade.id}`
📊 {trade.symbol} {trade.timeframe}
{'📈' if trade.direction == 'LONG' else '📉'} {trade.direction}

💰 Entry: ${trade.entry_price:.4f}
💰 Exit: ${exit_price:.4f}
📊 P&L: {pnl:+.2f}%

🎯 TP: ${trade.take_profit:.4f}
🛑 SL: ${trade.stop_loss:.4f}

⏰ {self._now()}
"""
        
        # Send to all subscribers
        from main import subscribers
        for chat_id in subscribers.keys():
            try:
                await telegram_bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode='Markdown'
                )
            except Exception as e:
                print(f"Error sending notification to {chat_id}: {e}")
    
    def _now(self) -> str:
        """Get current time in SL timezone"""
        return datetime.now(SL_TZ).strftime('%Y-%m-%d %H:%M:%S IST')
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        print(f"[{self._now()}] 🛑 Auto-tracker stopped")


# Global tracker instance
tracker = None

def get_tracker(exchange: ccxt.Exchange) -> AutoTradeTracker:
    """Get or create tracker instance"""
    global tracker
    if tracker is None:
        tracker = AutoTradeTracker(exchange)
    return tracker
