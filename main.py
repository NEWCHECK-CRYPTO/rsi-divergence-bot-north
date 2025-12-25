"""
Main Telegram Bot - V11 RELAXED VERSION
"""

import asyncio
import logging
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import threading
import pytz

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_TOKEN, SCAN_TIMEFRAMES, SCAN_INTERVAL, 
    TOP_COINS_COUNT, TIMEZONE, EXCHANGE
)
from divergence_scanner import (
    DivergenceScanner, AlertFormatter, SignalStrength, SignalQuality,
    get_sl_time, format_sl_time
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

scanner = DivergenceScanner()
subscribers = {}
SL_TZ = pytz.timezone(TIMEZONE)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        symbols = scanner.get_symbols_to_scan()
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot V11</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}.badge{{background:#238636;padding:4px 8px;border-radius:6px}}</style></head>
<body>
<h1>🤖 RSI Divergence Bot - V11 Relaxed</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{len(symbols)}</b></p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p>Features: <span class="badge">Hidden Divergences</span> <span class="badge">Quality Tiers</span></p>
<p>Status: <span style="color:#3fb950">✅ RUNNING</span></p>
<p>Time: {format_sl_time()}</p>
</body></html>"""
        
        self.wfile.write(html.encode())
    
    def log_message(self, f, *a):
        pass


def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    server.serve_forever()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    
    await update.message.reply_text(
        f"🤖 *RSI Divergence Bot - V11 Relaxed*\n\n"
        f"📊 *{len(symbols)}* coins\n"
        f"⏰ {', '.join(SCAN_TIMEFRAMES)}\n"
        f"🦎 Exchange: *{EXCHANGE.upper()}*\n\n"
        f"✨ *Features:*\n"
        f"• Regular + Hidden Divergences\n"
        f"• Quality Tiers (Premium/Standard/Exploratory)\n"
        f"• Optional MTF Confirmation\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get all signals (65%+)\n"
        f"/premium - Only premium signals (85%+)\n"
        f"/standard - Standard+ signals (70%+)\n"
        f"/scan - Manual scan\n"
        f"/coins - List coins\n"
        f"/mystatus - Check your settings\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {
        "min_confidence": 0.65,
        "quality_filter": None
    }
    
    await update.message.reply_text(
        f"✅ *Subscribed!*\n\n"
        f"Getting: *All Signals* (65%+)\n"
        f"Quality: All tiers 💎⭐🔍\n\n"
        f"🦎 Exchange: {EXCHANGE.upper()}\n"
        f"🔄 Scan every {SCAN_INTERVAL//60} min\n\n"
        f"Change filter:\n"
        f"/premium - Only 85%+ signals\n"
        f"/standard - Only 70%+ signals\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def premium_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in subscribers:
        await update.message.reply_text("❌ Please /subscribe first")
        return
    
    subscribers[chat_id]["min_confidence"] = 0.85
    subscribers[chat_id]["quality_filter"] = "premium"
    
    await update.message.reply_text(
        f"💎 *Premium Signals Only*\n\n"
        f"Minimum confidence: 85%\n"
        f"Quality: Premium tier only\n\n"
        f"Expected: 5-10 signals/day\n"
        f"Win rate: ~75%\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def standard_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in subscribers:
        await update.message.reply_text("❌ Please /subscribe first")
        return
    
    subscribers[chat_id]["min_confidence"] = 0.70
    subscribers[chat_id]["quality_filter"] = "standard"
    
    await update.message.reply_text(
        f"⭐ *Standard+ Signals*\n\n"
        f"Minimum confidence: 70%\n"
        f"Quality: Premium + Standard\n\n"
        f"Expected: 15-25 signals/day\n"
        f"Win rate: ~65%\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def my_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    if chat_id not in subscribers:
        await update.message.reply_text("❌ Not subscribed. Use /subscribe")
        return
    
    settings = subscribers[chat_id]
    min_conf = settings["min_confidence"]
    quality = settings["quality_filter"]
    
    if quality == "premium":
        quality_text = "💎 Premium only"
    elif quality == "standard":
        quality_text = "⭐ Premium + Standard"
    else:
        quality_text = "💎⭐🔍 All tiers"
    
    await update.message.reply_text(
        f"📊 *Your Settings*\n\n"
        f"Min Confidence: {min_conf*100:.0f}%\n"
        f"Quality Filter: {quality_text}\n\n"
        f"Change:\n"
        f"/subscribe - All signals (65%+)\n"
        f"/premium - Premium only (85%+)\n"
        f"/standard - Standard+ (70%+)\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


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


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"🔍 *Scanning {len(symbols)} coins...*\n\n"
        f"⏰ TFs: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"⏳ This takes 3-5 minutes...",
        parse_mode='Markdown'
    )
    
    try:
        alerts = scanner.scan_all()
        
        if alerts:
            premium = [a for a in alerts if a.signal_quality == SignalQuality.PREMIUM]
            standard = [a for a in alerts if a.signal_quality == SignalQuality.STANDARD]
            exploratory = [a for a in alerts if a.signal_quality == SignalQuality.EXPLORATORY]
            
            await update.message.reply_text(
                f"✅ *Found {len(alerts)} signals!*\n\n"
                f"💎 Premium: {len(premium)}\n"
                f"⭐ Standard: {len(standard)}\n"
                f"🔍 Exploratory: {len(exploratory)}",
                parse_mode='Markdown'
            )
            
            for alert in alerts[:10]:
                msg = AlertFormatter.format_alert(alert)
                await update.message.reply_text(msg)
                await asyncio.sleep(0.5)
            
            if len(alerts) > 10:
                await update.message.reply_text(
                    f"_...and {len(alerts) - 10} more signals_",
                    parse_mode='Markdown'
                )
        else:
            await update.message.reply_text(
                f"🔭 *No divergences found*\n\n"
                f"Try again in 1-2 hours.\n\n"
                f"🇱🇰 {format_sl_time()}",
                parse_mode='Markdown'
            )
    
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
        
        for alert in alerts:
            logger.info(f"[{format_sl_time()}] Signal: {alert.symbol} {alert.signal_tf} - {alert.signal_quality.value} ({alert.total_confidence*100:.0f}%)")
        
        for chat_id, settings in subscribers.items():
            min_conf = settings["min_confidence"]
            quality_filter = settings["quality_filter"]
            
            filtered_alerts = []
            for alert in alerts:
                if alert.total_confidence < min_conf:
                    continue
                
                if quality_filter == "premium" and alert.signal_quality != SignalQuality.PREMIUM:
                    continue
                elif quality_filter == "standard" and alert.signal_quality == SignalQuality.EXPLORATORY:
                    continue
                
                filtered_alerts.append(alert)
            
            if not filtered_alerts:
                continue
            
            for alert in filtered_alerts[:5]:
                try:
                    msg = AlertFormatter.format_alert(alert)
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} alerts")
    
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")


def main():
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Web server started")
    
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] {EXCHANGE.upper()}: {len(symbols)} coins loaded")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("premium", premium_only))
    app.add_handler(CommandHandler("standard", standard_mode))
    app.add_handler(CommandHandler("mystatus", my_status))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("coins", show_coins))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot V11 Relaxed ({EXCHANGE.upper()}) starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
