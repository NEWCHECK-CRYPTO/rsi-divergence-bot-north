"""
Main Telegram Bot - V10 with DEBUG mode
EXACT 2-CANDLE ALERTS ONLY
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
    SimpleDivergenceScanner as DivergenceScanner, 
    AlertFormatter, SignalStrength, 
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
        num_coins = len(symbols) if symbols else 0
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot Working</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}</style></head>
<body>
<h1>🤖 RSI Divergence Bot - V10</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{num_coins}</b></p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p>Status: <span style="color:#3fb950">✅ RUNNING</span></p>
<p>Time: {format_sl_time()}</p>
<p style="color:#58a6ff">⚡ EXACT 2-CANDLE MODE</p>
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
    num_coins = len(symbols) if symbols else 0
    
    await update.message.reply_text(
        f"🤖 *RSI Divergence Bot - V10*\n\n"
        f"📊 *{num_coins}* coins loaded\n"
        f"⏰ {', '.join(SCAN_TIMEFRAMES)}\n"
        f"🦎 Exchange: *{EXCHANGE.upper()}*\n"
        f"⚡ Mode: *EXACT 2-CANDLE*\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get alerts\n"
        f"/scan - Manual scan\n"
        f"/coins - List coins\n"
        f"/debug - System check ⚙️\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug command to check system"""
    await update.message.reply_text("🔧 Running system check...")
    
    try:
        # Test getting symbols
        await update.message.reply_text("🔄 Checking coin list...")
        symbols = scanner.get_symbols_to_scan()
        
        if symbols and len(symbols) > 0:
            coins_list = "\n".join([f"{i}. {s}" for i, s in enumerate(symbols[:10], 1)])
            await update.message.reply_text(
                f"✅ Bot has {len(symbols)} coins loaded!\n\n"
                f"Top 10:\n{coins_list}\n\n"
                f"...and {len(symbols) - 10} more"
            )
            
            # Test scanning one symbol
            await update.message.reply_text(f"🔍 Testing scan on {symbols[0]}...")
            test_alerts = scanner.scan_symbol(symbols[0], "4h")
            
            if test_alerts:
                await update.message.reply_text(f"🎯 Found signal on {symbols[0]}!")
                msg = AlertFormatter.format_alert(test_alerts[0])
                await update.message.reply_text(msg)
            else:
                await update.message.reply_text(
                    f"✅ Scanner working! (no signals on {symbols[0]} right now)\n\n"
                    f"⚡ Note: Alerts only when 2nd confirmation candle JUST closes\n\n"
                    f"Bot is ready to scan {len(symbols)} coins!"
                )
                
            # Test exchange connection
            await update.message.reply_text("🔄 Testing exchange connection...")
            test_df = scanner.fetch_ohlcv(symbols[0], "1h", limit=10)
            
            if test_df is not None:
                await update.message.reply_text(
                    f"✅ Exchange working!\n"
                    f"Successfully fetched {len(test_df)} candles for {symbols[0]}\n\n"
                    f"🎉 Everything is working perfectly!"
                )
            else:
                await update.message.reply_text(
                    f"⚠️ Exchange connection issue\n"
                    f"Could not fetch chart data for {symbols[0]}\n\n"
                    f"Try:\n"
                    f"1. Change EXCHANGE to 'binance' in config.py\n"
                    f"2. Check your internet connection\n"
                    f"3. Use VPN if exchange is blocked"
                )
        else:
            await update.message.reply_text(
                "❌ ERROR: No coins loaded!\n\n"
                "This shouldn't happen with the simple scanner.\n"
                "Check if divergence_scanner.py has SIMPLE_COINS list."
            )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await update.message.reply_text(
            f"❌ Debug failed:\n\n"
            f"Error: {str(e)}\n\n"
            f"Details:\n{error_trace[:500]}"
        )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {"min_strength": SignalStrength.EARLY}
    
    symbols = scanner.get_symbols_to_scan()
    num_coins = len(symbols) if symbols else 0
    
    await update.message.reply_text(
        f"✅ *Subscribed!*\n\n"
        f"🦎 Exchange: {EXCHANGE.upper()}\n"
        f"📊 Tracking: {num_coins} coins\n"
        f"🔄 Scan every {SCAN_INTERVAL//60} min\n"
        f"⚡ Alert: When 2nd candle JUST closes\n\n"
        f"You'll get FRESH signals only!\n"
        f"No old/late alerts 🚫\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Unsubscribed")


async def show_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Getting coin list...")
    symbols = scanner.get_symbols_to_scan()
    
    if not symbols or len(symbols) == 0:
        await update.message.reply_text(
            "❌ No coins loaded!\n"
            "Try /debug to check system status."
        )
        return
    
    msg = f"📊 *{len(symbols)} Coins Loaded*\n\n"
    for i, s in enumerate(symbols[:30], 1):
        msg += f"{i}. {s}\n"
    
    if len(symbols) > 30:
        msg += f"\n_...and {len(symbols) - 30} more_"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    
    if not symbols or len(symbols) == 0:
        await update.message.reply_text(
            "❌ No coins loaded!\n"
            "Try /debug to check system status."
        )
        return
    
    await update.message.reply_text(
        f"🔍 *Scanning {len(symbols)} coins...*\n\n"
        f"⏰ TFs: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"⚡ Looking for FRESH 2-candle confirms\n"
        f"⏳ This takes 3-5 minutes...",
        parse_mode='Markdown'
    )
    
    try:
        alerts = scanner.scan_all()
        
        if alerts:
            await update.message.reply_text(
                f"✅ *Found {len(alerts)} FRESH signals!*\n"
                f"⚡ All 2nd candle JUST closed!",
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
                f"🔭 *No FRESH divergences found*\n\n"
                f"Scanned {len(symbols)} coins across {len(SCAN_TIMEFRAMES)} timeframes.\n\n"
                f"⚡ Only alerts when 2nd candle JUST closes\n"
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
            logger.info(f"[{format_sl_time()}] No FRESH signals found")
            return
        
        for alert in alerts:
            logger.info(f"[{format_sl_time()}] FRESH Signal: {alert.symbol} {alert.signal_tf}")
        
        for chat_id in subscribers.keys():
            for alert in alerts[:5]:
                try:
                    msg = AlertFormatter.format_alert(alert)
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} FRESH alerts")
    
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")


def main():
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Web server started")
    
    symbols = scanner.get_symbols_to_scan()
    num_coins = len(symbols) if symbols else 0
    logger.info(f"[{format_sl_time()}] {EXCHANGE.upper()}: {num_coins} coins loaded")
    logger.info(f"[{format_sl_time()}] ⚡ EXACT 2-CANDLE MODE - Fresh signals only!")
    
    if num_coins == 0:
        logger.error(f"[{format_sl_time()}] WARNING: No coins loaded! Check divergence_scanner.py")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("coins", show_coins))
    app.add_handler(CommandHandler("debug", debug))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot V10 ({EXCHANGE.upper()}) starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
