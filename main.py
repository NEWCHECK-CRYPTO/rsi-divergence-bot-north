"""
Main Telegram Bot - V11 with Candle-Close Aligned Scanning
Signals fire immediately when confirmation candles close
"""

import asyncio
import logging
from datetime import datetime, timedelta
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
    DivergenceScanner, AlertFormatter, SignalStrength, 
    get_sl_time, format_sl_time
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

scanner = DivergenceScanner()
subscribers = {}
SL_TZ = pytz.timezone(TIMEZONE)

# Track last scan time per timeframe to avoid duplicate scans
last_scan_times = {}


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        symbols = scanner.get_symbols_to_scan()
        now = get_sl_time()
        
        # Show next candle close times
        next_closes = []
        for tf in SCAN_TIMEFRAMES:
            next_close = get_next_candle_close(tf, now)
            next_closes.append(f"{tf}: {next_close.strftime('%H:%M')}")
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot V11</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}.green{{color:#3fb950}}</style></head>
<body>
<h1>🤖 RSI Divergence Bot - V11</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{len(symbols)}</b></p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p>Status: <span class="green">✅ RUNNING</span></p>
<p>Time: {format_sl_time()}</p>
<hr>
<h3>⏰ Next Candle Closes (IST)</h3>
<p>{' | '.join(next_closes)}</p>
<p><small>Scans trigger within 30s of candle close</small></p>
</body></html>"""
        
        self.wfile.write(html.encode())
    
    def log_message(self, f, *a):
        pass


def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    server.serve_forever()


def get_timeframe_seconds(tf: str) -> int:
    """Convert timeframe to seconds"""
    tf_map = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
    }
    return tf_map.get(tf, 3600)


def get_next_candle_close(tf: str, now: datetime = None) -> datetime:
    """Calculate when the next candle closes for a timeframe"""
    if now is None:
        now = get_sl_time()
    
    # Convert to UTC for calculation
    now_utc = now.astimezone(pytz.UTC)
    
    tf_seconds = get_timeframe_seconds(tf)
    
    if tf == "1w":
        # Weekly candles close Sunday midnight UTC
        days_until_sunday = (6 - now_utc.weekday()) % 7
        if days_until_sunday == 0 and now_utc.hour == 0 and now_utc.minute == 0:
            days_until_sunday = 7
        next_close = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        next_close += timedelta(days=days_until_sunday if days_until_sunday > 0 else 7)
    elif tf == "1d":
        # Daily candles close at midnight UTC
        next_close = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        if next_close <= now_utc:
            next_close += timedelta(days=1)
    else:
        # Intraday - calculate based on period
        timestamp = now_utc.timestamp()
        current_period_start = (timestamp // tf_seconds) * tf_seconds
        next_close_timestamp = current_period_start + tf_seconds
        next_close = datetime.fromtimestamp(next_close_timestamp, pytz.UTC)
    
    # Convert back to Sri Lanka time
    return next_close.astimezone(SL_TZ)


def get_seconds_until_candle_close(tf: str) -> float:
    """Get seconds until next candle closes"""
    now = get_sl_time()
    next_close = get_next_candle_close(tf, now)
    delta = (next_close - now).total_seconds()
    return max(0, delta)


def should_scan_timeframe(tf: str) -> bool:
    """Check if we should scan this timeframe (just after candle close)"""
    global last_scan_times
    
    now = get_sl_time()
    tf_seconds = get_timeframe_seconds(tf)
    
    # Get the most recent candle close time
    now_utc = now.astimezone(pytz.UTC)
    timestamp = now_utc.timestamp()
    last_close_timestamp = (timestamp // tf_seconds) * tf_seconds
    last_close = datetime.fromtimestamp(last_close_timestamp, pytz.UTC).astimezone(SL_TZ)
    
    # Check if we already scanned this candle
    last_scan = last_scan_times.get(tf)
    if last_scan and last_scan >= last_close:
        return False
    
    # Check if we're within 2 minutes of candle close (scan window)
    seconds_since_close = (now - last_close).total_seconds()
    if 0 <= seconds_since_close <= 120:  # Within 2 minutes of close
        last_scan_times[tf] = now
        return True
    
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    now = get_sl_time()
    
    # Calculate next candle closes
    next_closes = []
    for tf in SCAN_TIMEFRAMES:
        next_close = get_next_candle_close(tf, now)
        time_until = get_seconds_until_candle_close(tf)
        mins = int(time_until // 60)
        next_closes.append(f"  • {tf.upper()}: {next_close.strftime('%H:%M')} ({mins}m)")
    
    await update.message.reply_text(
        f"🤖 *RSI Divergence Bot - V11*\n\n"
        f"📊 *{len(symbols)}* coins\n"
        f"⏰ Timeframes: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"🦎 Exchange: *{EXCHANGE.upper()}*\n\n"
        f"*Next Candle Closes (IST):*\n"
        f"{chr(10).join(next_closes)}\n\n"
        f"⚡ Signals fire within 30s of candle close!\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get alerts\n"
        f"/scan - Manual scan\n"
        f"/status - Next candle times\n"
        f"/coins - List coins\n\n"
        f"🇱🇰 {format_sl_time()}",
        parse_mode='Markdown'
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show next candle close times"""
    now = get_sl_time()
    
    lines = ["⏰ *Next Candle Closes (IST)*\n"]
    
    for tf in SCAN_TIMEFRAMES:
        next_close = get_next_candle_close(tf, now)
        time_until = get_seconds_until_candle_close(tf)
        
        if time_until < 60:
            time_str = f"{int(time_until)}s"
        elif time_until < 3600:
            time_str = f"{int(time_until // 60)}m"
        else:
            hours = int(time_until // 3600)
            mins = int((time_until % 3600) // 60)
            time_str = f"{hours}h {mins}m"
        
        lines.append(f"  • *{tf.upper()}*: {next_close.strftime('%H:%M')} (in {time_str})")
    
    lines.append(f"\n⚡ Scans trigger within 30s of close")
    lines.append(f"📊 Subscribers: {len(subscribers)}")
    lines.append(f"\n🕐 Now: {format_sl_time()}")
    
    await update.message.reply_text("\n".join(lines), parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {"min_strength": SignalStrength.EARLY}
    
    now = get_sl_time()
    next_closes = []
    for tf in SCAN_TIMEFRAMES:
        next_close = get_next_candle_close(tf, now)
        next_closes.append(f"{tf}: {next_close.strftime('%H:%M')}")
    
    await update.message.reply_text(
        f"✅ *Subscribed to V11!*\n\n"
        f"🦎 Exchange: {EXCHANGE.upper()}\n"
        f"⚡ Signals fire at candle close!\n\n"
        f"*Next alerts possible at:*\n"
        f"{' | '.join(next_closes)}\n\n"
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
            await update.message.reply_text(
                f"✅ *Found {len(alerts)} signals!*",
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
                f"Try again after next candle close.\n\n"
                f"🇱🇰 {format_sl_time()}",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await update.message.reply_text(f"❌ Scan error: {e}")


async def smart_scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    """
    Smart scanning - only scan timeframes that just had a candle close
    This ensures signals fire within seconds of confirmation
    """
    if not subscribers:
        return
    
    now = get_sl_time()
    timeframes_to_scan = []
    
    # Check which timeframes just had a candle close
    for tf in SCAN_TIMEFRAMES:
        if should_scan_timeframe(tf):
            timeframes_to_scan.append(tf)
            logger.info(f"[{format_sl_time()}] ⏰ {tf} candle just closed - scanning!")
    
    if not timeframes_to_scan:
        return  # No candles just closed, skip
    
    logger.info(f"[{format_sl_time()}] Scanning TFs: {timeframes_to_scan} for {len(subscribers)} subs")
    
    try:
        # Only scan the timeframes that just closed
        all_alerts = []
        symbols = scanner.get_symbols_to_scan()
        
        for symbol in symbols:
            for tf in timeframes_to_scan:
                try:
                    alerts = scanner.scan_symbol(symbol, tf)
                    all_alerts.extend(alerts)
                    await asyncio.sleep(0.05)  # Small delay to avoid rate limits
                except Exception as e:
                    logger.error(f"Error scanning {symbol} {tf}: {e}")
        
        if not all_alerts:
            logger.info(f"[{format_sl_time()}] No signals for {timeframes_to_scan}")
            return
        
        # Send alerts immediately
        for alert in all_alerts:
            logger.info(f"[{format_sl_time()}] 🎯 Signal: {alert.symbol} {alert.signal_tf}")
            
            for chat_id in subscribers.keys():
                try:
                    msg = AlertFormatter.format_alert(alert)
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Send error to {chat_id}: {e}")
        
        logger.info(f"[{format_sl_time()}] ✅ Sent {len(all_alerts)} alerts")
    
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")
        import traceback
        traceback.print_exc()


async def full_scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    """
    Full scan of all timeframes - runs less frequently as backup
    """
    if not subscribers:
        return
    
    logger.info(f"[{format_sl_time()}] Running full backup scan...")
    
    try:
        alerts = scanner.scan_all()
        
        if not alerts:
            logger.info(f"[{format_sl_time()}] Full scan: No signals")
            return
        
        for chat_id in subscribers.keys():
            for alert in alerts[:5]:
                try:
                    msg = AlertFormatter.format_alert(alert)
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Full scan sent {len(alerts)} alerts")
    
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Full scan error: {e}")


def main():
    # Start web server
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Web server started")
    
    # Load symbols
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] {EXCHANGE.upper()}: {len(symbols)} coins loaded")
    
    # Show next candle closes
    now = get_sl_time()
    for tf in SCAN_TIMEFRAMES:
        next_close = get_next_candle_close(tf, now)
        secs = get_seconds_until_candle_close(tf)
        logger.info(f"  {tf}: Next close at {next_close.strftime('%H:%M:%S')} ({int(secs)}s)")
    
    # Build application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("coins", show_coins))
    app.add_handler(CommandHandler("status", status))
    
    # Smart scan every 30 seconds - checks if any candle just closed
    app.job_queue.run_repeating(smart_scheduled_scan, interval=30, first=10)
    
    # Full backup scan every 30 minutes
    app.job_queue.run_repeating(full_scheduled_scan, interval=1800, first=300)
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot V11 ({EXCHANGE.upper()}) starting...")
    logger.info(f"[{format_sl_time()}] ⚡ Smart scanning enabled - signals fire at candle close!")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
