"""
RSI Divergence Alert Bot - V4 with Trade Journal
- Proper Lookback Range
- Significant Swing Detection  
- Trade Journal & Tracking
- RAG-Powered Performance Analysis
"""

import asyncio
import logging
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import pytz

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import (
    TELEGRAM_TOKEN, SCAN_TIMEFRAMES, SCAN_INTERVAL, TOP_COINS_COUNT, TIMEZONE,
    LOOKBACK_CANDLES, MIN_SWING_DISTANCE, MIN_PRICE_MOVE_PCT, SWING_STRENGTH_BARS,
    GEMINI_API_KEY
)
from divergence_scanner import (
    DivergenceScanner, AlertFormatter, SignalStrength, get_sl_time, format_sl_time
)
from trade_journal import journal

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

scanner = DivergenceScanner()
rag = None
subscribers = {}
SL_TZ = pytz.timezone(TIMEZONE)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        symbols = scanner.get_symbols_to_scan()
        stats = journal.get_stats()
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot V4</title>
<style>body{{font-family:system-ui;background:#1a1a2e;color:#fff;padding:40px}}</style></head>
<body><h1>🤖 RSI Bot V4 + Journal</h1>
<p>🇱🇰 {format_sl_time()}</p>
<p>📊 {len(symbols)} coins | 👥 {len(subscribers)} subs | 📝 {stats['total']} trades | {stats['win_rate']:.0f}% WR</p>
</body></html>"""
        self.wfile.write(html.encode())
    def log_message(self, f, *a): pass

def run_web_server():
    server = HTTPServer(('0.0.0.0', int(os.environ.get('PORT', 8080))), HealthHandler)
    server.serve_forever()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = journal.get_stats()
    symbols = scanner.get_symbols_to_scan()
    
    try:
        from tradingview_ta import TA_Handler
        tv_status = "✅ TradingView-TA Active"
    except:
        tv_status = "⚠️ TradingView-TA Fallback"
    
    await update.message.reply_text(
        f"🤖 *RSI Divergence Bot V5*\n\n"
        f"📊 {len(symbols)} coins | ⏰ {', '.join(SCAN_TIMEFRAMES)}\n\n"
        f"*V5 Features:*\n"
        f"📺 {tv_status}\n"
        f"📝 Trade Journal\n"
        f"🤖 AI Analysis\n\n"
        f"*Journal:* {stats['total']} trades | {stats['win_rate']:.0f}% WR\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get alerts\n"
        f"/journal - Statistics\n"
        f"/open - Open trades\n"
        f"/close <id> win|loss - Record result\n"
        f"/analyze - AI analysis\n"
        f"/ask <question> - Ask anything\n"
        f"/scan - Manual scan\n"
        f"/debug <symbol> <tf> - Debug scan",
        parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    min_strength = SignalStrength.EARLY
    if context.args:
        if context.args[0].lower() == "strong": min_strength = SignalStrength.STRONG
        elif context.args[0].lower() == "medium": min_strength = SignalStrength.MEDIUM
    subscribers[chat_id] = {"min_strength": min_strength}
    await update.message.reply_text(f"✅ Subscribed! All signals logged to journal.\n🇱🇰 {format_sl_time()}")


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Unsubscribed")


async def show_journal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = journal.format_stats_message()
    keyboard = [[
        InlineKeyboardButton("📊 By TF", callback_data="stats_tf"),
        InlineKeyboardButton("💰 By Symbol", callback_data="stats_symbol"),
        InlineKeyboardButton("🤖 Analyze", callback_data="stats_analyze")
    ]]
    await update.message.reply_text(msg, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))


async def show_open_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(journal.format_open_trades(), parse_mode='Markdown')


async def close_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: `/close <id> win|loss [price]`\nUse /open to see IDs", parse_mode='Markdown')
        return
    
    trade_id, outcome_str = context.args[0], context.args[1].lower()
    exit_price = float(context.args[2]) if len(context.args) > 2 else None
    
    outcome = ("win_tp" if outcome_str == "win" else "loss_sl") if not exit_price else ("win_manual" if outcome_str == "win" else "loss_manual")
    
    if journal.update_outcome(trade_id, outcome, exit_price):
        trade = next((t for t in journal.entries if t.id == trade_id), None)
        emoji = "✅" if "win" in outcome else "❌"
        await update.message.reply_text(f"{emoji} Trade `{trade_id}` closed: {trade.pnl_percent:+.2f}%", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"❌ Trade `{trade_id}` not found", parse_mode='Markdown')


async def analyze_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = journal.get_stats()
    if stats['total'] < 3:
        await update.message.reply_text(f"📊 Need at least 3 closed trades. You have {stats['total']}.")
        return
    
    await update.message.reply_text("🤖 Analyzing...")
    
    try:
        if rag:
            analysis = rag.query(journal.generate_analysis_prompt())
            await update.message.reply_text(f"🤖 *AI Analysis*\n\n{analysis}", parse_mode='Markdown')
        else:
            by_str = journal.get_stats_by_strength()
            by_dir = journal.get_stats_by_direction()
            best_str = max(by_str.items(), key=lambda x: x[1]['win_rate'] if x[1]['total'] > 0 else 0)
            
            await update.message.reply_text(
                f"🤖 *Analysis*\n\n"
                f"{'✅ Profitable' if stats['total_pnl'] > 0 else '❌ Losing'}\n"
                f"Win Rate: {stats['win_rate']:.0f}%\n"
                f"Profit Factor: {stats['profit_factor']:.2f}\n\n"
                f"Best: {best_str[0].upper()} ({best_str[1]['win_rate']:.0f}% WR)\n\n"
                f"💡 {'Focus on ' + best_str[0].upper() + ' signals' if best_str[1]['total'] > 2 else 'Need more data'}",
                parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def stats_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "stats_tf":
        by_tf = journal.get_stats_by_timeframe()
        msg = "📊 *By Timeframe*\n\n" + "\n".join([f"• {tf}: {s['total']} trades, {s['win_rate']:.0f}% WR" for tf, s in by_tf.items()]) if by_tf else "No data"
        await query.edit_message_text(msg, parse_mode='Markdown')
    elif query.data == "stats_symbol":
        by_sym = journal.get_stats_by_symbol()
        msg = "📊 *By Symbol (Top 10)*\n\n" + "\n".join([f"• {s}: {st['total']} trades, {st['win_rate']:.0f}% WR" for s, st in list(by_sym.items())[:10]]) if by_sym else "No data"
        await query.edit_message_text(msg, parse_mode='Markdown')
    elif query.data == "stats_analyze":
        await query.edit_message_text("Use /analyze for full AI analysis")


async def ask_rag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ask <your question>")
        return
    question = " ".join(context.args)
    await update.message.reply_text("🤔 Thinking...")
    try:
        if rag:
            await update.message.reply_text(f"🤖 {rag.query(question)}")
        else:
            await update.message.reply_text("⚠️ Gemini not configured")
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"🔍 *Scanning {len(symbols)} coins...*\n\n"
        f"⚙️ Settings:\n"
        f"• Lookback: {LOOKBACK_CANDLES} candles\n"
        f"• Min distance: {MIN_SWING_DISTANCE} candles\n"
        f"• Min move: {MIN_PRICE_MOVE_PCT}%\n"
        f"• Swing strength: {SWING_STRENGTH_BARS} bars\n\n"
        f"⏳ This may take 3-5 minutes...",
        parse_mode='Markdown')
    
    try:
        alerts = scanner.scan_all()
        if alerts:
            await update.message.reply_text(f"✅ Found {len(alerts)} signals!")
            for alert in alerts:
                trade_id = journal.add_entry(alert)
                msg = AlertFormatter.format_alert(alert) + f"\n\n📝 ID: `{trade_id}`"
                await update.message.reply_text(msg)
                await asyncio.sleep(0.5)
        else:
            # Give helpful feedback
            await update.message.reply_text(
                f"📭 *No signals found*\n\n"
                f"Possible reasons:\n"
                f"• Market ranging (no clear divergences)\n"
                f"• Swings not significant enough\n"
                f"• Need to wait for patterns to form\n\n"
                f"💡 Try `/scan` again in 1-2 hours, or\n"
                f"adjust settings in config.py\n\n"
                f"🇱🇰 {format_sl_time()}",
                parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}\n\nCheck Northflank logs for details.")


async def show_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.fetch_top_coins_by_volume(20)
    msg = "🔥 *Top 20*\n\n" + "\n".join([f"#{i+1} {s}" for i, s in enumerate(symbols)])
    await update.message.reply_text(msg, parse_mode='Markdown')


async def debug_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug: Scan one symbol and show what's found"""
    symbol = context.args[0] if context.args else "BTC/USDT"
    timeframe = context.args[1] if len(context.args) > 1 else "4h"
    
    await update.message.reply_text(f"🔍 Debug scan: {symbol} {timeframe}...\n\nStep 1: Testing exchange connection...")
    
    try:
        # Step 1: Test basic exchange connection
        try:
            markets = scanner.exchange.load_markets()
            await update.message.reply_text(f"✅ Exchange connected! {len(markets)} markets loaded.")
        except Exception as e:
            await update.message.reply_text(f"❌ Exchange connection failed: {e}")
            return
        
        # Step 2: Check if symbol exists
        if symbol not in markets:
            await update.message.reply_text(f"❌ Symbol {symbol} not found. Try: BTC/USDT, ETH/USDT")
            return
        
        await update.message.reply_text(f"✅ Symbol {symbol} found. Fetching OHLCV...")
        
        # Step 3: Fetch OHLCV directly (without TV)
        try:
            ohlcv = scanner.exchange.fetch_ohlcv(symbol, timeframe, limit=50)
            await update.message.reply_text(f"✅ Got {len(ohlcv)} candles from Binance")
        except Exception as e:
            await update.message.reply_text(f"❌ OHLCV fetch failed: {e}")
            return
        
        # Step 4: Process data
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Step 5: Calculate RSI manually (skip TV-TA for now)
        from divergence_scanner import calculate_rsi_fallback
        df['rsi'] = calculate_rsi_fallback(df['close'], 14)
        
        # Step 6: Find swings
        swing_highs = scanner.find_major_swing_highs(df)
        swing_lows = scanner.find_major_swing_lows(df)
        
        msg = f"🔍 *Debug: {symbol} {timeframe}*\n\n"
        msg += f"📊 Candles: {len(df)}\n"
        msg += f"💰 Price: ${df['close'].iloc[-1]:,.2f}\n"
        msg += f"📈 RSI: {df['rsi'].iloc[-1]:.1f}\n\n"
        
        msg += f"⚙️ *Settings:*\n"
        msg += f"• Lookback: {LOOKBACK_CANDLES}\n"
        msg += f"• Min Distance: {MIN_SWING_DISTANCE}\n"
        msg += f"• Min Move: {MIN_PRICE_MOVE_PCT}%\n"
        msg += f"• Swing Bars: {SWING_STRENGTH_BARS}\n\n"
        
        msg += f"📈 Swing Highs: {len(swing_highs)}\n"
        msg += f"📉 Swing Lows: {len(swing_lows)}\n\n"
        
        if swing_highs:
            msg += f"*Last Swing Highs:*\n"
            for sh in swing_highs[-3:]:
                msg += f"• ${sh.price:,.2f} RSI:{sh.rsi:.1f}\n"
        
        if swing_lows:
            msg += f"\n*Last Swing Lows:*\n"
            for sl in swing_lows[-3:]:
                msg += f"• ${sl.price:,.2f} RSI:{sl.rsi:.1f}\n"
        
        # Check divergence
        idx = len(df) - 1
        valid_lows = scanner.filter_significant_swings(swing_lows, idx)
        valid_highs = scanner.filter_significant_swings(swing_highs, idx)
        
        msg += f"\n*Filtered:* {len(valid_lows)} lows, {len(valid_highs)} highs\n"
        
        if len(valid_lows) >= 2:
            p1, p2 = valid_lows[-2], valid_lows[-1]
            if p2.price < p1.price and p2.rsi > p1.rsi:
                msg += "\n✅ *BULLISH DIVERGENCE!*\n"
            else:
                msg += "\n❌ No bullish div\n"
        
        if len(valid_highs) >= 2:
            p1, p2 = valid_highs[-2], valid_highs[-1]
            if p2.price > p1.price and p2.rsi < p1.rsi:
                msg += "✅ *BEARISH DIVERGENCE!*\n"
            else:
                msg += "❌ No bearish div\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        await update.message.reply_text(f"❌ Error: {e}\n\n```{error[-1000:]}```", parse_mode='Markdown')


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers: return
    logger.info(f"[{format_sl_time()}] Scanning...")
    
    try:
        alerts = scanner.scan_all()
        if not alerts: return
        
        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
        
        for alert in alerts:
            trade_id = journal.add_entry(alert)
            logger.info(f"Logged {trade_id}: {alert.symbol}")
        
        for chat_id, prefs in subscribers.items():
            min_level = strength_order.get(prefs.get("min_strength", SignalStrength.EARLY), 1)
            for alert in [a for a in alerts if strength_order.get(a.signal_strength, 0) >= min_level]:
                try:
                    entry = next((e for e in journal.entries[-len(alerts):] if e.symbol == alert.symbol), None)
                    msg = AlertFormatter.format_alert(alert) + f"\n\n📝 ID: `{entry.id if entry else 'N/A'}`"
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                except: pass
    except Exception as e:
        logger.error(f"Scan error: {e}")


def main():
    global rag
    threading.Thread(target=run_web_server, daemon=True).start()
    
    try:
        if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
            from rag_module import TradingKnowledgeRAG
            rag = TradingKnowledgeRAG()
            logger.info("RAG ready")
    except: pass
    
    logger.info(f"Journal: {journal.get_stats()['total']} trades")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("journal", show_journal))
    app.add_handler(CommandHandler("open", show_open_trades))
    app.add_handler(CommandHandler("close", close_trade))
    app.add_handler(CommandHandler("analyze", analyze_performance))
    app.add_handler(CommandHandler("ask", ask_rag))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("top", show_top))
    app.add_handler(CommandHandler("debug", debug_scan))
    app.add_handler(CallbackQueryHandler(stats_callback, pattern="^stats_"))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"🤖 Bot V5 + TradingView-TA starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
