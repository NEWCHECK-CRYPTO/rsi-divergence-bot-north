"""
RSI Divergence Alert Bot - V3
- 3 Signal Levels: Strong, Medium, Early
- TradingView RSI Calculation
- Proper Candle Close Logic
- Sri Lanka Time
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

from config import TELEGRAM_TOKEN, SCAN_TIMEFRAMES, SCAN_INTERVAL, TOP_COINS_COUNT, TIMEZONE
from divergence_scanner import (
    DivergenceScanner, AlertFormatter, SignalStrength,
    get_sl_time, format_sl_time
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
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
        top_5 = symbols[:5] if symbols else []
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot V3</title><meta http-equiv="refresh" content="30">
<style>body{{font-family:sans-serif;background:#1a1a2e;color:#fff;padding:40px}}
.card{{background:rgba(255,255,255,0.05);border-radius:16px;padding:24px;margin:20px 0}}
.status{{background:#00c853;padding:8px 16px;border-radius:20px;color:#000}}</style></head>
<body><h1>🤖 RSI Divergence Bot V3</h1>
<div class="card"><span class="status">✅ Running</span><p>🇱🇰 {format_sl_time()}</p>
<p>📊 {len(symbols)} coins | 👥 {len(subscribers)} subscribers | ⏰ {SCAN_INTERVAL//60}m interval</p></div>
<div class="card"><h3>Signal Levels</h3><p>🟢 STRONG | 🟡 MEDIUM | 🔴 EARLY</p></div>
<div class="card"><h3>Top 5</h3><p>{', '.join(top_5)}</p></div>
<div class="card"><h3>Timeframes</h3><p>{', '.join(SCAN_TIMEFRAMES)}</p></div></body></html>"""
        self.wfile.write(html.encode())
    def log_message(self, format, *args): pass


def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f"Health server on port {port}")
    server.serve_forever()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"🤖 *RSI Divergence Bot V3*\n\n"
        f"📊 *{len(symbols)}* coins | ⏰ {', '.join(SCAN_TIMEFRAMES)}\n"
        f"📈 TradingView RSI | 🇱🇰 Sri Lanka time\n\n"
        f"*Signal Levels:*\n"
        f"🟢 STRONG - Div + MS + Momentum\n"
        f"🟡 MEDIUM - Div + Momentum\n"
        f"🔴 EARLY - Divergence forming\n\n"
        f"*Commands:*\n"
        f"/subscribe - All signals\n"
        f"/subscribe strong - Strong only\n"
        f"/subscribe medium - Medium+\n"
        f"/scan - Manual scan\n"
        f"/status - Bot status\n"
        f"/top - Top 20 coins\n"
        f"/rules - Trading rules",
        parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    min_strength = SignalStrength.EARLY
    if context.args:
        level = context.args[0].lower()
        if level == "strong": min_strength = SignalStrength.STRONG
        elif level == "medium": min_strength = SignalStrength.MEDIUM
    
    subscribers[chat_id] = {"min_strength": min_strength}
    strength_text = {
        SignalStrength.STRONG: "🟢 Strong only",
        SignalStrength.MEDIUM: "🟡 Medium+",
        SignalStrength.EARLY: "🔴🟡🟢 All signals"
    }
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"✅ *Subscribed!*\n\n"
        f"📊 {len(symbols)} coins | ⏰ {', '.join(SCAN_TIMEFRAMES)}\n"
        f"🔔 {strength_text[min_strength]}\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown')


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id in subscribers:
        del subscribers[update.effective_chat.id]
    await update.message.reply_text("❌ Unsubscribed")


async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"📊 *Status*\n\n"
        f"🪙 {len(symbols)} coins\n"
        f"⏰ {', '.join(SCAN_TIMEFRAMES)}\n"
        f"👥 {len(subscribers)} subscribers\n"
        f"🇱🇰 {format_sl_time()}\n\n"
        f"*Top 10:*\n" + "\n".join([f"#{i+1} {s}" for i, s in enumerate(symbols[:10])]),
        parse_mode='Markdown')


async def show_top_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Fetching...")
    try:
        symbols = scanner.fetch_top_coins_by_volume(TOP_COINS_COUNT)
        lines = ["🔥 *Top 20 by Volume*\n"]
        for i, s in enumerate(symbols[:20], 1):
            medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else f"#{i}"
            lines.append(f"{medal} {s}")
        lines.append(f"\n🇱🇰 {format_sl_time()}")
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(f"🔍 Scanning {len(symbols)} coins...\n⏳ 3-5 minutes\n🇱🇰 {format_sl_time()}")
    try:
        alerts = scanner.scan_all()
        if alerts:
            strong = len([a for a in alerts if a.signal_strength == SignalStrength.STRONG])
            medium = len([a for a in alerts if a.signal_strength == SignalStrength.MEDIUM])
            early = len([a for a in alerts if a.signal_strength == SignalStrength.EARLY])
            await update.message.reply_text(f"✅ *Found {len(alerts)} signals!*\n🟢 {strong} | 🟡 {medium} | 🔴 {early}", parse_mode='Markdown')
            for alert in alerts:
                await update.message.reply_text(AlertFormatter.format_alert(alert))
                await asyncio.sleep(0.5)
        else:
            await update.message.reply_text(f"📭 No signals\n🇱🇰 {format_sl_time()}")
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def show_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🟢 Bullish", callback_data="rules_bullish"),
                 InlineKeyboardButton("🔴 Bearish", callback_data="rules_bearish")],
                [InlineKeyboardButton("📈 MS", callback_data="rules_ms"),
                 InlineKeyboardButton("📊 Signals", callback_data="rules_signals")]]
    await update.message.reply_text("📚 *Trading Rules*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')


async def rules_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    rules = {
        "bullish": "🟢 *BULLISH*\n• Strong: Price LL + RSI HL\n• Medium: Price DB + RSI HL\n• Hidden: Price HL + RSI LL\n\n✅ RSI rising\n✅ Price rising",
        "bearish": "🔴 *BEARISH*\n• Strong: Price HH + RSI LH\n• Medium: Price DT + RSI LH\n• Hidden: Price LH + RSI HH\n\n✅ RSI falling\n✅ Price falling",
        "ms": "📈 *MARKET STRUCTURE*\n• BOS = Break of Structure\n• CHoCH = Change of Character",
        "signals": "📊 *SIGNALS*\n🟢 STRONG: Div+MS+Momentum\n🟡 MEDIUM: Div+Momentum\n🔴 EARLY: Div forming"
    }
    await query.edit_message_text(rules.get(query.data.replace("rules_",""), "N/A"), parse_mode='Markdown')


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers: return
    logger.info(f"[{format_sl_time()}] Scanning for {len(subscribers)} subscribers")
    try:
        alerts = scanner.scan_all()
        if not alerts: return
        strength_order = {SignalStrength.STRONG: 3, SignalStrength.MEDIUM: 2, SignalStrength.EARLY: 1}
        for chat_id, prefs in subscribers.items():
            min_level = strength_order.get(prefs.get("min_strength", SignalStrength.EARLY), 1)
            user_alerts = [a for a in alerts if strength_order.get(a.signal_strength, 0) >= min_level]
            for alert in user_alerts:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=AlertFormatter.format_alert(alert))
                except Exception as e:
                    logger.error(f"Send error: {e}")
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} alerts")
    except Exception as e:
        logger.error(f"Scan error: {e}")


def main():
    global rag
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Health server started")
    
    try:
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
            from rag_module import TradingKnowledgeRAG
            rag = TradingKnowledgeRAG()
            logger.info(f"[{format_sl_time()}] Gemini RAG ready")
    except: pass
    
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] Loaded {len(symbols)} symbols")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("top", show_top_coins))
    app.add_handler(CommandHandler("status", show_status))
    app.add_handler(CommandHandler("rules", show_rules))
    app.add_handler(CallbackQueryHandler(rules_callback, pattern="^rules_"))
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot V3 starting | {len(symbols)} coins | {', '.join(SCAN_TIMEFRAMES)}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
