"""
RSI Divergence Bot - Main Telegram Bot
======================================
Features:
- Top 100 coins by market cap
- 1h, 4h, 1d timeframes
- Candle-close aligned scanning
- Alert 1 candle after swing confirmation
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
    TOP_COINS_COUNT, TIMEZONE, EXCHANGE, TOP_100_MARKET_CAP,
    SWING_STRENGTH, MIN_SWING_DISTANCE, MAX_SWING_DISTANCE,
    RSI_OVERSOLD, RSI_OVERBOUGHT, MAX_CANDLES_SINCE_SWING2,
    CONFIRMATION_CANDLES
)
from divergence_scanner import (
    DivergenceScanner, AlertFormatter, SignalStrength, DivergenceType,
    get_sl_time, format_sl_time, get_next_candle_close,
    get_market_regime, get_volatility_status, MarketRegime, VolatilityStatus
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
        
        # Calculate next candle closes
        candle_info = ""
        for tf in SCAN_TIMEFRAMES:
            next_close, secs = get_next_candle_close(tf)
            mins = int(secs // 60)
            secs_rem = int(secs % 60)
            candle_info += f"<p>{tf.upper()}: {mins}m {secs_rem}s</p>"
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Divergence Bot</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}h2{{color:#7ee787}}p{{color:#8b949e}}.ok{{color:#3fb950}}</style></head>
<body>
<h1>🤖 RSI Divergence Bot</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{len(symbols)} / {TOP_COINS_COUNT}</b> (Top Market Cap)</p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p class="ok">Status: ✅ RUNNING</p>
<p>Time: {format_sl_time()}</p>
<hr>
<h2>⏰ Next Candle Closes</h2>
{candle_info}
<hr>
<h2>📋 Conditions</h2>
<p>Swing Strength: {SWING_STRENGTH} candles each side</p>
<p>Alert Timing: {CONFIRMATION_CANDLES} candle after swing</p>
<p>Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles</p>
<p>Bullish RSI: &lt; {RSI_OVERSOLD}</p>
<p>Bearish RSI: &gt; {RSI_OVERBOUGHT}</p>
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
    
    msg = f"""*🤖 RSI Divergence Bot*

📊 *{len(symbols)}* top market cap coins
⏰ Timeframes: *{', '.join(tf.upper() for tf in SCAN_TIMEFRAMES)}*
🦎 Exchange: *{EXCHANGE.upper()}*

*Signal Conditions:*
• Swing Strength: {SWING_STRENGTH} candles each side
• Alert: {CONFIRMATION_CANDLES} candle after swing confirmed
• Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
• Bullish RSI: < {RSI_OVERSOLD} (oversold)
• Bearish RSI: > {RSI_OVERBOUGHT} (overbought)

*Alert Timing:*
• 1h: Alert ~1 hour after swing
• 4h: Alert ~4 hours after swing
• 1d: Alert ~1 day after swing

*Commands:*
/subscribe - Get alerts
/scan - Manual scan
/coins - List coins
/verify SYMBOL TF - Check divergences
/regime SYMBOL TF - Check market conditions
/conditions - Show all conditions
/timing - Show next candle closes

Time: {format_sl_time()}"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def conditions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current conditions"""
    msg = f"""*📋 Signal Conditions*

*Swing Detection:*
• Strength: {SWING_STRENGTH} candles each side
• A swing low/high is confirmed when there are {SWING_STRENGTH} lower/higher closes on each side

*Alert Timing:*
• Alert fires {CONFIRMATION_CANDLES} candle after swing is confirmed
• 1h: Alert within ~1-2 hours of swing
• 4h: Alert within ~4-8 hours of swing
• 1d: Alert within ~1-2 days of swing

*For BULLISH Divergence:*
1. Price makes LOWER LOW (Swing2 < Swing1)
2. RSI makes HIGHER LOW (Swing2 RSI > Swing1 RSI)
3. RSI is OVERSOLD (< {RSI_OVERSOLD})
4. Swings are {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
5. No candle CLOSE below Swing2 between swings
6. Swing2 recently confirmed

*For BEARISH Divergence:*
1. Price makes HIGHER HIGH (Swing2 > Swing1)
2. RSI makes LOWER HIGH (Swing2 RSI < Swing1 RSI)
3. RSI is OVERBOUGHT (> {RSI_OVERBOUGHT})
4. Swings are {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
5. No candle CLOSE above Swing2 between swings
6. Swing2 recently confirmed

*Signal Strength:*
🟢 STRONG: RSI < 30 (bull) or > 70 (bear)
🟡 MEDIUM: RSI 30-35 (bull) or 65-70 (bear)
🔵 EARLY: RSI 35-40 (bull) or 60-65 (bear)"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def timing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show next candle close times"""
    msg = "*⏰ Next Candle Closes*\n\n"
    
    for tf in SCAN_TIMEFRAMES:
        next_close, secs = get_next_candle_close(tf)
        mins = int(secs // 60)
        secs_rem = int(secs % 60)
        next_close_sl = next_close.astimezone(SL_TZ)
        msg += f"*{tf.upper()}*: {mins}m {secs_rem}s\n"
        msg += f"   → {next_close_sl.strftime('%H:%M:%S')} IST\n\n"
    
    msg += f"Current: {format_sl_time()}"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {"subscribed": True}
    
    await update.message.reply_text(
        f"✅ *Subscribed!*\n\n"
        f"You'll receive alerts when divergences are detected.\n"
        f"Timeframes: {', '.join(tf.upper() for tf in SCAN_TIMEFRAMES)}\n"
        f"Scan interval: {SCAN_INTERVAL} seconds\n\n"
        f"Time: {format_sl_time()}",
        parse_mode='Markdown'
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Unsubscribed from alerts")


async def show_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show list of coins being scanned"""
    symbols = scanner.get_symbols_to_scan()
    
    msg = f"*📊 Top {len(symbols)} Coins by Market Cap*\n\n"
    
    for i, s in enumerate(symbols[:50], 1):
        clean = s.replace('/USDT', '')
        msg += f"{i}. {clean}\n"
    
    if len(symbols) > 50:
        msg += f"\n_...and {len(symbols) - 50} more_"
    
    msg += f"\n\nExchange: {EXCHANGE.upper()}"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Verify divergences on a symbol"""
    args = context.args
    
    if not args or len(args) < 2:
        await update.message.reply_text(
            "*🔍 Divergence Verification Tool*\n\n"
            "Usage: `/verify SYMBOL TIMEFRAME`\n\n"
            "Examples:\n"
            "`/verify BTC/USDT 4h`\n"
            "`/verify ETHUSDT 1h`\n"
            "`/verify SOL/USDT 1d`\n\n"
            "Timeframes: 1h, 4h, 1d",
            parse_mode='Markdown'
        )
        return
    
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol = symbol + '/USDT'
    
    timeframe = args[1].lower()
    if timeframe not in SCAN_TIMEFRAMES:
        await update.message.reply_text(
            f"❌ Invalid timeframe. Use: {', '.join(SCAN_TIMEFRAMES)}"
        )
        return
    
    await update.message.reply_text(
        f"🔍 *Analyzing {symbol} {timeframe.upper()}*\n"
        f"Looking for divergences...",
        parse_mode='Markdown'
    )
    
    try:
        df = scanner.fetch_ohlcv(symbol, timeframe, limit=200)
        
        if df is None or len(df) < 50:
            await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
            return
        
        # Find swings
        swing_lows = scanner.find_swing_lows(df, SWING_STRENGTH)
        swing_highs = scanner.find_swing_highs(df, SWING_STRENGTH)
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_idx = len(df) - 1
        
        report = []
        report.append(f"📊 {symbol} {timeframe.upper()} Analysis")
        report.append("=" * 50)
        report.append(f"Current Price: ${current_price:.6f}")
        report.append(f"Current RSI: {current_rsi:.1f}")
        report.append(f"Candles: {len(df)}")
        report.append(f"Swing Lows: {len(swing_lows)}")
        report.append(f"Swing Highs: {len(swing_highs)}")
        report.append("")
        
        # Check for bullish divergences
        bullish_found = []
        if len(swing_lows) >= 2:
            for i in range(len(swing_lows) - 1):
                s1 = swing_lows[i]
                s2 = swing_lows[i + 1]
                
                candles_apart = s2.index - s1.index
                if candles_apart < MIN_SWING_DISTANCE or candles_apart > MAX_SWING_DISTANCE:
                    continue
                
                # Price LL, RSI HL
                if s2.price < s1.price and s2.rsi > s1.rsi and s2.rsi < RSI_OVERSOLD:
                    recency = current_idx - s2.index
                    max_rec = MAX_CANDLES_SINCE_SWING2.get(timeframe, 3)
                    status = "✅ RECENT" if recency <= max_rec else f"⏰ OLD ({recency} candles)"
                    
                    bullish_found.append({
                        's1': s1, 's2': s2,
                        'apart': candles_apart,
                        'recency': recency,
                        'status': status
                    })
        
        # Check for bearish divergences
        bearish_found = []
        if len(swing_highs) >= 2:
            for i in range(len(swing_highs) - 1):
                s1 = swing_highs[i]
                s2 = swing_highs[i + 1]
                
                candles_apart = s2.index - s1.index
                if candles_apart < MIN_SWING_DISTANCE or candles_apart > MAX_SWING_DISTANCE:
                    continue
                
                # Price HH, RSI LH
                if s2.price > s1.price and s2.rsi < s1.rsi and s2.rsi > RSI_OVERBOUGHT:
                    recency = current_idx - s2.index
                    max_rec = MAX_CANDLES_SINCE_SWING2.get(timeframe, 3)
                    status = "✅ RECENT" if recency <= max_rec else f"⏰ OLD ({recency} candles)"
                    
                    bearish_found.append({
                        's1': s1, 's2': s2,
                        'apart': candles_apart,
                        'recency': recency,
                        'status': status
                    })
        
        # Report bullish divergences
        report.append("🟢 BULLISH DIVERGENCES")
        report.append("-" * 50)
        if bullish_found:
            for b in bullish_found[-5:]:
                report.append(f"{b['status']}")
                report.append(f"  Swing 1: ${b['s1'].price:.6f} RSI:{b['s1'].rsi:.1f}")
                report.append(f"  Swing 2: ${b['s2'].price:.6f} RSI:{b['s2'].rsi:.1f}")
                report.append(f"  Distance: {b['apart']} candles")
                report.append("")
        else:
            report.append("  None found")
            report.append("")
        
        # Report bearish divergences
        report.append("🔴 BEARISH DIVERGENCES")
        report.append("-" * 50)
        if bearish_found:
            for b in bearish_found[-5:]:
                report.append(f"{b['status']}")
                report.append(f"  Swing 1: ${b['s1'].price:.6f} RSI:{b['s1'].rsi:.1f}")
                report.append(f"  Swing 2: ${b['s2'].price:.6f} RSI:{b['s2'].rsi:.1f}")
                report.append(f"  Distance: {b['apart']} candles")
                report.append("")
        else:
            report.append("  None found")
            report.append("")
        
        # Recent swing points
        report.append("📍 RECENT SWING LOWS")
        report.append("-" * 50)
        for sl in swing_lows[-5:]:
            report.append(f"  {sl.timestamp.strftime('%m-%d %H:%M')} | ${sl.price:.6f} | RSI: {sl.rsi:.1f}")
        
        report.append("")
        report.append("📍 RECENT SWING HIGHS")
        report.append("-" * 50)
        for sh in swing_highs[-5:]:
            report.append(f"  {sh.timestamp.strftime('%m-%d %H:%M')} | ${sh.price:.6f} | RSI: {sh.rsi:.1f}")
        
        # Send report
        full_report = "\n".join(report)
        
        # Split if too long
        if len(full_report) > 4000:
            chunks = [full_report[i:i+4000] for i in range(0, len(full_report), 4000)]
            for chunk in chunks:
                await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
                await asyncio.sleep(0.5)
        else:
            await update.message.reply_text(f"```\n{full_report}\n```", parse_mode='Markdown')
        
    except Exception as e:
        import traceback
        await update.message.reply_text(f"❌ Error: {e}\n\n{traceback.format_exc()[:500]}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run a manual scan"""
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"🔍 *Scanning {len(symbols)} coins...*\n"
        f"Timeframes: {', '.join(tf.upper() for tf in SCAN_TIMEFRAMES)}\n"
        f"This may take a few minutes...",
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
                # Get market regime info
                regime, volatility = scanner.get_market_info(alert.symbol, alert.timeframe)
                
                if regime and volatility:
                    msg = AlertFormatter.format_alert_with_regime(alert, regime, volatility)
                else:
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
                f"*📭 No divergences found right now*\n\n"
                f"This is normal - divergences are rare.\n"
                f"The bot will alert you when one forms.\n\n"
                f"Time: {format_sl_time()}",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"❌ Scan error: {e}")


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    """Run scheduled scan aligned to candle closes"""
    if not subscribers:
        return
    
    logger.info(f"[{format_sl_time()}] Scheduled scan for {len(subscribers)} subscribers")
    
    try:
        alerts = scanner.scan_all()
        
        if not alerts:
            logger.info(f"[{format_sl_time()}] No signals found")
            return
        
        for alert in alerts:
            logger.info(f"[{format_sl_time()}] Signal: {alert.symbol} {alert.timeframe} "
                       f"({alert.seconds_after_close:.0f}s after close)")
        
        for chat_id in subscribers.keys():
            for alert in alerts[:5]:  # Max 5 alerts per scan
                try:
                    regime, volatility = scanner.get_market_info(alert.symbol, alert.timeframe)
                    
                    if regime and volatility:
                        msg = AlertFormatter.format_alert_with_regime(alert, regime, volatility)
                    else:
                        msg = AlertFormatter.format_alert(alert)
                    
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} alerts")
    
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")


async def regime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check market regime for a symbol"""
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "*📊 Market Regime Checker*\n\n"
            "Check if market conditions are good for divergence trading.\n\n"
            "*Usage:*\n"
            "`/regime BTCUSDT`\n"
            "`/regime BTC/USDT`\n"
            "`/regime BTC/USDT 4h`\n\n"
            "*Timeframes:* 1h, 4h, 1d\n"
            "*Default:* 4h",
            parse_mode='Markdown'
        )
        return
    
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol = symbol + '/USDT'
    
    timeframe = args[1].lower() if len(args) > 1 else "4h"
    if timeframe not in SCAN_TIMEFRAMES:
        timeframe = "4h"
    
    await update.message.reply_text(f"📊 Analyzing {symbol} {timeframe.upper()}...")
    
    try:
        df = scanner.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if df is None or len(df) < 50:
            await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
            return
        
        regime_info = get_market_regime(df)
        volatility_info = get_volatility_status(df)
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        msg = AlertFormatter.format_regime_info(
            symbol, timeframe, regime_info, volatility_info,
            current_price, current_rsi
        )
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        import traceback
        await update.message.reply_text(f"❌ Error: {e}\n\n{traceback.format_exc()[:500]}")


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick debug info"""
    symbols = scanner.get_symbols_to_scan()
    
    # Next candle closes
    candle_info = ""
    for tf in SCAN_TIMEFRAMES:
        next_close, secs = get_next_candle_close(tf)
        mins = int(secs // 60)
        candle_info += f"• {tf.upper()}: {mins}m\n"
    
    msg = f"""*🔧 Debug Info*

Exchange: {EXCHANGE.upper()}
Coins loaded: {len(symbols)}
Subscribers: {len(subscribers)}
Cooldowns active: {len(scanner.alert_cooldowns)}
Signals tracked: {len(scanner.sent_signals)}

*Swing Settings:*
• Strength: {SWING_STRENGTH} candles
• Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE}

*Recency Windows:*
• 1h: {MAX_CANDLES_SINCE_SWING2.get('1h', 3)} candles
• 4h: {MAX_CANDLES_SINCE_SWING2.get('4h', 3)} candles
• 1d: {MAX_CANDLES_SINCE_SWING2.get('1d', 3)} candles

*RSI Zones:*
• Bullish: < {RSI_OVERSOLD}
• Bearish: > {RSI_OVERBOUGHT}

*Next Candle Closes:*
{candle_info}
Time: {format_sl_time()}"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


def main():
    # Start web server for health checks
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Web server started")
    
    # Load symbols
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] {EXCHANGE.upper()}: {len(symbols)} coins loaded")
    
    # Build bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("coins", show_coins))
    app.add_handler(CommandHandler("verify", verify))
    app.add_handler(CommandHandler("conditions", conditions))
    app.add_handler(CommandHandler("timing", timing))
    app.add_handler(CommandHandler("regime", regime))
    app.add_handler(CommandHandler("debug", debug))
    
    # Schedule scans every minute to catch candle closes
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=30)
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot starting...")
    logger.info(f"[{format_sl_time()}] Timeframes: {SCAN_TIMEFRAMES}")
    logger.info(f"[{format_sl_time()}] Swing Strength: {SWING_STRENGTH}")
    logger.info(f"[{format_sl_time()}] Confirmation: {CONFIRMATION_CANDLES} candle after swing")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
