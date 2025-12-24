"""
RSI Divergence Bot V10 - WITH ML AUTO-TRACKING
Main file with ML integration and automatic trade monitoring
"""

import asyncio
import logging
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import pytz

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_TOKEN, SCAN_TIMEFRAMES, SCAN_INTERVAL, TOP_COINS_COUNT, TIMEZONE,
    LOOKBACK_CANDLES, MIN_SWING_DISTANCE, MIN_PRICE_MOVE_PCT, SWING_STRENGTH_BARS,
    GEMINI_API_KEY, EXCHANGE
)
from divergence_scanner import (
    DivergenceScanner, AlertFormatter, SignalStrength, get_sl_time, format_sl_time
)
from trade_journal import journal
from ml_market_regime import ml_detector  # ML MODULE
from auto_trade_tracker import get_tracker  # AUTO-TRACKING MODULE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

scanner = DivergenceScanner()
rag = None
subscribers = {}
SL_TZ = pytz.timezone(TIMEZONE)
tracker = get_tracker(scanner.exchange)  # INITIALIZE TRACKER


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        symbols = scanner.get_symbols_to_scan()
        stats = journal.get_stats()
        open_trades = len(journal.get_open_trades())
        ml_perf = ml_detector.get_regime_performance()
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot V10 ML</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}.green{{color:#3fb950}}.stat{{background:#161b22;padding:20px;border-radius:10px;margin:10px 0}}</style></head>
<body>
<h1>🤖 RSI Divergence Bot V10 ML</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b> | 🇱🇰 {format_sl_time()}</p>
<div class="stat">
<p>📊 Coins: <b>{len(symbols)}</b> | ⏰ TFs: {len(SCAN_TIMEFRAMES)} | 👥 Subs: {len(subscribers)}</p>
<p>📝 Trades: {stats['total']} closed, {open_trades} open | Win Rate: <span class="green">{stats['win_rate']:.0f}%</span></p>
</div>
<div class="stat">
<p>🤖 ML: {len(ml_perf)} regimes tracked</p>
<p>⚙️ Lookback: {LOOKBACK_CANDLES} | Min Dist: {MIN_SWING_DISTANCE} | Min Move: {MIN_PRICE_MOVE_PCT}%</p>
</div>
</body></html>"""
        self.wfile.write(html.encode())
    def log_message(self, f, *a): pass

def run_web_server():
    server = HTTPServer(('0.0.0.0', int(os.environ.get('PORT', 8080))), HealthHandler)
    server.serve_forever()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = journal.get_stats()
    symbols = scanner.get_symbols_to_scan()
    open_trades = len(journal.get_open_trades())
    
    await update.message.reply_text(
        f"🤖 *RSI Divergence Bot V10 ML*\n\n"
        f"📊 *{len(symbols)}* coins | ⏰ {', '.join(SCAN_TIMEFRAMES)}\n"
        f"🦎 Exchange: *{EXCHANGE.upper()}*\n\n"
        f"*Features:*\n"
        f"✅ 100 top coins by volume\n"
        f"✅ TradingView RSI validation\n"
        f"✅ Trade Journal\n"
        f"✅ 🤖 ML Market Regime Detection\n"
        f"✅ 🎯 Auto TP/SL Tracking\n"
        f"✅ Scans every {SCAN_INTERVAL//60} min\n\n"
        f"*Journal:* {stats['total']} trades | {stats['win_rate']:.0f}% WR\n"
        f"*Open:* {open_trades} trades being monitored\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get alerts\n"
        f"/scan - Manual scan\n"
        f"/debug <symbol> - Test single coin\n"
        f"/journal - Statistics\n"
        f"/ml - ML regime performance\n"
        f"/open - Show open trades\n"
        f"/coins - List coins",
        parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    min_strength = SignalStrength.EARLY
    if context.args:
        if context.args[0].lower() == "strong": min_strength = SignalStrength.STRONG
        elif context.args[0].lower() == "medium": min_strength = SignalStrength.MEDIUM
    subscribers[chat_id] = {"min_strength": min_strength}
    
    strength_text = {
        SignalStrength.STRONG: "🟢 Strong only",
        SignalStrength.MEDIUM: "🟡 Medium+",
        SignalStrength.EARLY: "🔴🟡🟢 All signals"
    }
    
    await update.message.reply_text(
        f"✅ *Subscribed to V10 ML!*\n\n"
        f"🦎 Exchange: {EXCHANGE.upper()}\n"
        f"📊 {strength_text[min_strength]}\n"
        f"🔄 Scan every {SCAN_INTERVAL//60} min\n"
        f"🤖 ML adapts to market conditions\n"
        f"🎯 Auto-tracks TP/SL hits\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown')


async def show_ml_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show ML regime performance"""
    msg = ml_detector.format_regime_report()
    await update.message.reply_text(msg, parse_mode='Markdown')


async def show_open_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show open trades being monitored"""
    msg = journal.format_open_trades()
    await update.message.reply_text(msg, parse_mode='Markdown')


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Unsubscribed")


async def show_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Fetching coins...")
    symbols = scanner.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    msg = f"📊 *Top {len(symbols)} Coins on {EXCHANGE.upper()}*\n\n"
    for i, s in enumerate(symbols[:30], 1):
        msg += f"{i}. {s}\n"
    
    if len(symbols) > 30:
        msg += f"\n_...and {len(symbols) - 30} more_"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def show_journal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = journal.format_stats_message()
    await update.message.reply_text(msg, parse_mode='Markdown')


async def debug_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug a single symbol"""
    symbol = context.args[0].upper() if context.args else "BTC"
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    timeframe = context.args[1] if len(context.args) > 1 else "4h"
    
    await update.message.reply_text(f"🔍 Debug: {symbol} {timeframe}...")
    
    try:
        markets = scanner.exchange.load_markets()
        await update.message.reply_text(f"✅ {EXCHANGE.upper()} connected! {len(markets)} markets")
        
        if symbol not in markets:
            similar = [s for s in markets if symbol.split('/')[0] in s][:5]
            await update.message.reply_text(f"❌ {symbol} not found.\n\nTry: {', '.join(similar)}")
            return
        
        df = scanner.fetch_ohlcv(symbol, timeframe)
        if df is None or len(df) < 10:
            await update.message.reply_text(f"❌ Could not fetch {symbol} data")
            return
        
        # Detect market regime
        regime = ml_detector.detect_regime(df)
        regime_settings = ml_detector.get_settings_for_regime(regime)
        
        swing_highs = scanner.find_major_swing_highs(df)
        swing_lows = scanner.find_major_swing_lows(df)
        
        msg = f"🔍 *Debug: {symbol} {timeframe}*\n\n"
        msg += f"📊 Candles: {len(df)}\n"
        msg += f"💰 Price: ${df['close'].iloc[-1]:,.4f}\n"
        msg += f"📈 RSI: {df['rsi'].iloc[-1]:.1f}\n\n"
        
        msg += f"🤖 *ML Regime: {regime.value.upper()}*\n"
        msg += f"Min Confidence: {regime_settings.min_confidence*100:.0f}%\n"
        msg += f"Lookback: {regime_settings.lookback_candles}\n\n"
        
        msg += f"📈 Swing Highs: {len(swing_highs)}\n"
        msg += f"📉 Swing Lows: {len(swing_lows)}\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        import traceback
        await update.message.reply_text(f"❌ Error: {e}\n\n{traceback.format_exc()[-500:]}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"🔍 *Scanning {len(symbols)} coins...*\n\n"
        f"🦎 Exchange: {EXCHANGE.upper()}\n"
        f"⏰ TFs: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"⏳ This takes 3-5 minutes...",
        parse_mode='Markdown')
    
    try:
        alerts = scanner.scan_all()
        
        if alerts:
            strong = len([a for a in alerts if a.signal_strength == SignalStrength.STRONG])
            medium = len([a for a in alerts if a.signal_strength == SignalStrength.MEDIUM])
            early = len([a for a in alerts if a.signal_strength == SignalStrength.EARLY])
            
            await update.message.reply_text(
                f"✅ *Found {len(alerts)} signals!*\n\n"
                f"🟢 Strong: {strong}\n"
                f"🟡 Medium: {medium}\n"
                f"🔴 Early: {early}",
                parse_mode='Markdown')
            
            for alert in alerts[:10]:
                trade_id = journal.add_entry(alert)
                msg = AlertFormatter.format_alert(alert)
                msg += f"\n\n📝 ID: `{trade_id}`"
                await update.message.reply_text(msg)
                await asyncio.sleep(0.5)
            
            if len(alerts) > 10:
                await update.message.reply_text(f"_...and {len(alerts) - 10} more signals_", parse_mode='Markdown')
        else:
            await update.message.reply_text(
                f"🔭 *No divergences found*\n\n"
                f"Market may be ranging or trending strongly.\n"
                f"Try again in 1-2 hours.\n\n"
                f"🇱🇰 {format_sl_time()}",
                parse_mode='Markdown')
                
    except Exception as e:
        await update.message.reply_text(f"❌ Scan error: {e}")


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers:
        return
    
    logger.info(f"[{format_sl_time()}] Scheduled scan for {len(subscribers)} subs")
    
    try:
        alerts = scanner.scan_all()
        
        if not alerts:
            logger.info(f"[{format_sl_time()}] No signals found")
            return
        
        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
        
        for alert in alerts:
            trade_id = journal.add_entry(alert)
            logger.info(f"[{format_sl_time()}] Signal: {alert.symbol} {alert.signal_tf} ({trade_id})")
        
        for chat_id, prefs in subscribers.items():
            min_level = strength_order.get(prefs.get("min_strength", SignalStrength.EARLY), 1)
            user_alerts = [a for a in alerts if strength_order.get(a.signal_strength, 0) >= min_level]
            
            for alert in user_alerts[:5]:
                try:
                    entry = next((e for e in journal.entries[-len(alerts):] if e.symbol == alert.symbol), None)
                    msg = AlertFormatter.format_alert(alert)
                    if entry:
                        msg += f"\n\n📝 ID: `{entry.id}`"
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} alerts")
        
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")


def main():
    global rag
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Web server started")
    
    # Init RAG
    try:
        if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
            from rag_module import TradingKnowledgeRAG
            rag = TradingKnowledgeRAG()
            logger.info(f"[{format_sl_time()}] RAG ready")
    except: pass
    
    # Load symbols
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] {EXCHANGE.upper()}: {len(symbols)} coins loaded")
    
    # Journal stats
    stats = journal.get_stats()
    logger.info(f"[{format_sl_time()}] Journal: {stats['total']} trades")
    
    # ML stats
    ml_perf = ml_detector.get_regime_performance()
    logger.info(f"[{format_sl_time()}] ML: {len(ml_perf)} regimes tracked")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("debug", debug_scan))
    app.add_handler(CommandHandler("journal", show_journal))
    app.add_handler(CommandHandler("ml", show_ml_stats))
    app.add_handler(CommandHandler("open", show_open_trades))
    app.add_handler(CommandHandler("coins", show_coins))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    # START AUTO-TRACKER (runs in background)
    asyncio.create_task(tracker.start_monitoring(app.bot))
    logger.info(f"[{format_sl_time()}] 🎯 Auto-tracker monitoring started")
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot V10 ML ({EXCHANGE.upper()}) starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
