"""
RSI Divergence Bot - SIMPLIFIED VERSION
Main Telegram Bot with Swing Strength = 2
"""

import asyncio
import logging
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import threading
import pytz
import pandas as pd

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_TOKEN, SCAN_TIMEFRAMES, SCAN_INTERVAL, 
    TOP_COINS_COUNT, TIMEZONE, EXCHANGE
)
from divergence_scanner import (
    DivergenceScanner, AlertFormatter, SignalStrength, DivergenceType,
    get_sl_time, format_sl_time,
    SWING_STRENGTH_MAP, SWING_STRENGTH, MIN_SWING_DISTANCE, MAX_SWING_DISTANCE,
    RSI_OVERSOLD, RSI_OVERBOUGHT, MAX_CANDLES_SINCE_SWING2
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
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Divergence Bot</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}.ok{{color:#3fb950}}</style></head>
<body>
<h1>RSI Divergence Bot - Simplified</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{len(symbols)}</b></p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p class="ok">Status: RUNNING</p>
<p>Time: {format_sl_time()}</p>
<hr>
<h3>Conditions:</h3>
<p>Swing Strength: 2 (all timeframes)</p>
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
    
    msg = f"""*RSI Divergence Bot - Simplified*

📊 *{len(symbols)}* coins loaded
⏰ Timeframes: {', '.join(SCAN_TIMEFRAMES)}
🦎 Exchange: *{EXCHANGE.upper()}*

*Signal Conditions:*
• Swing Strength: 2 (optimal for all TFs)
• Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles
• Bullish RSI: < {RSI_OVERSOLD} (oversold)
• Bearish RSI: > {RSI_OVERBOUGHT} (overbought)
• Price-based invalidation only
• Recency: Within {MAX_CANDLES_SINCE_SWING2} candles

*Commands:*
/subscribe - Get alerts
/scan - Manual scan
/coins - List coins
/verify SYMBOL TF - Check divergences
/conditions - Show conditions

Time: {format_sl_time()}"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def conditions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current conditions"""
    msg = f"""*📋 Signal Conditions (CLOSE-based, Strength=2)*

*Why these settings?*
• CLOSE prices match RSI calculation
• Swing Strength 2 = industry standard
• Price-only invalidation (no RSI check between swings)

*For BULLISH Divergence:*
1. Find Swing LOW (lowest CLOSE with 2 higher closes each side)
2. Price makes LOWER LOW (Swing2 < Swing1)
3. RSI makes HIGHER LOW (Swing2 RSI > Swing1 RSI)
4. RSI is OVERSOLD (< {RSI_OVERSOLD})
5. Swings are {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
6. No candle CLOSE below Swing2 between swings
7. Swing2 formed within last {MAX_CANDLES_SINCE_SWING2} candles

*For BEARISH Divergence:*
1. Find Swing HIGH (highest CLOSE with 2 lower closes each side)
2. Price makes HIGHER HIGH (Swing2 > Swing1)
3. RSI makes LOWER HIGH (Swing2 RSI < Swing1 RSI)
4. RSI is OVERBOUGHT (> {RSI_OVERBOUGHT})
5. Swings are {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
6. No candle CLOSE above Swing2 between swings
7. Swing2 formed within last {MAX_CANDLES_SINCE_SWING2} candles

*Signal Strength:*
🟢 STRONG: RSI < 30 (bull) or > 70 (bear)
🟡 MEDIUM: RSI 30-35 (bull) or 65-70 (bear)
🔵 EARLY: RSI 35-40 (bull) or 60-65 (bear)"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {"subscribed": True}
    
    await update.message.reply_text(
        f"✅ *Subscribed!*\n\n"
        f"You'll receive alerts when divergences are detected.\n"
        f"Timeframes: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"Scan interval: {SCAN_INTERVAL//60} minutes\n\n"
        f"Time: {format_sl_time()}",
        parse_mode='Markdown'
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.pop(update.effective_chat.id, None)
    await update.message.reply_text("❌ Unsubscribed from alerts")


async def show_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Fetching coins...")
    symbols = scanner.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    msg = f"*Top {len(symbols)} Coins on {EXCHANGE.upper()}*\n\n"
    for i, s in enumerate(symbols[:30], 1):
        msg += f"{i}. {s}\n"
    
    if len(symbols) > 30:
        msg += f"\n_...and {len(symbols) - 30} more_"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Verify divergences on a symbol - shows ALL divergences found"""
    args = context.args
    
    if not args or len(args) < 2:
        await update.message.reply_text(
            "*Divergence Verification Tool*\n\n"
            "Usage: `/verify SYMBOL TIMEFRAME [CANDLES]`\n\n"
            "Examples:\n"
            "`/verify BTC/USDT 4h`\n"
            "`/verify ETH/USDT 1h 500`\n"
            "`/verify SOL/USDT 1d 1000`\n"
            "`/verify BTC/USDT 1M 100`\n\n"
            "Default: 200 candles (max: 1000)\n"
            "Timeframes: 1h, 4h, 1d, 1M\n"
            "Swing Strength: 2 (all timeframes)",
            parse_mode='Markdown'
        )
        return
    
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol = symbol + '/USDT'
    
    timeframe = args[1]
    # Handle monthly timeframe
    if timeframe.upper() == "1M":
        timeframe = "1M"
    else:
        timeframe = timeframe.lower()
    
    candles = int(args[2]) if len(args) > 2 else 200
    candles = min(candles, 1000)
    
    # Get swing strength for this timeframe
    strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
    
    await update.message.reply_text(
        f"*Analyzing {symbol} {timeframe.upper()}*\n"
        f"Scanning {candles} candles...\n"
        f"Swing Strength: {strength}",
        parse_mode='Markdown'
    )
    
    try:
        df = scanner.fetch_ohlcv(symbol, timeframe, limit=candles)
        
        if df is None or len(df) < 50:
            await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
            return
        
        # Get swing points with timeframe
        swing_lows = scanner.find_swing_lows(df, timeframe)
        swing_highs = scanner.find_swing_highs(df, timeframe)
        
        # Build report
        report = []
        report.append("=" * 50)
        report.append(f"DIVERGENCE ANALYSIS: {symbol} {timeframe.upper()}")
        report.append(f"Candles: {len(df)} | Swing Strength: {strength}")
        report.append(f"Time: {format_sl_time()}")
        report.append("=" * 50)
        report.append("")
        
        # Current state
        current = df.iloc[-1]
        report.append(f"Current Price: ${current['close']:.2f}")
        report.append(f"Current RSI: {current['rsi']:.1f}")
        report.append(f"Data: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        report.append("")
        
        report.append(f"Swing Lows Found: {len(swing_lows)}")
        report.append(f"Swing Highs Found: {len(swing_highs)}")
        report.append("")
        
        # BULLISH SCAN
        report.append("=" * 50)
        report.append("BULLISH DIVERGENCE SCAN")
        report.append(f"(Price Lower Low + RSI Higher Low + RSI < {RSI_OVERSOLD})")
        report.append("=" * 50)
        
        bullish_found = 0
        bullish_valid = 0
        current_idx = len(df) - 1
        
        for i in range(len(swing_lows) - 1):
            s1 = swing_lows[i]
            s2 = swing_lows[i + 1]
            
            if s2.price < s1.price and s2.rsi > s1.rsi:
                bullish_found += 1
                candles_apart = s2.index - s1.index
                candles_since = current_idx - s2.index
                
                checks = []
                all_pass = True
                
                if MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE:
                    checks.append(f"[OK] Distance: {candles_apart} candles")
                else:
                    checks.append(f"[FAIL] Distance: {candles_apart} (need {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE})")
                    all_pass = False
                
                checks.append(f"[OK] Price: ${s1.price:.2f} -> ${s2.price:.2f} (LOWER LOW)")
                checks.append(f"[OK] RSI: {s1.rsi:.1f} -> {s2.rsi:.1f} (HIGHER LOW)")
                
                if s2.rsi < RSI_OVERSOLD:
                    checks.append(f"[OK] RSI Zone: {s2.rsi:.1f} < {RSI_OVERSOLD} (OVERSOLD)")
                else:
                    checks.append(f"[FAIL] RSI Zone: {s2.rsi:.1f} >= {RSI_OVERSOLD} (NOT oversold)")
                    all_pass = False
                
                valid, msg = scanner.check_pattern_validity(df, s1, s2, True)
                if valid:
                    checks.append(f"[OK] Pattern: {msg}")
                else:
                    checks.append(f"[FAIL] Pattern: {msg}")
                    all_pass = False
                
                if candles_since <= MAX_CANDLES_SINCE_SWING2:
                    checks.append(f"[OK] Recency: {candles_since} candles ago (RECENT)")
                else:
                    checks.append(f"[--] Recency: {candles_since} candles ago (historical)")
                
                report.append("")
                status = ">>> VALID SIGNAL <<<" if all_pass and candles_since <= MAX_CANDLES_SINCE_SWING2 else "(pattern only)" if all_pass else "(failed)"
                report.append(f"BULLISH #{bullish_found}: {s1.timestamp.strftime('%Y-%m-%d')} -> {s2.timestamp.strftime('%Y-%m-%d')} {status}")
                for c in checks:
                    report.append(f"  {c}")
                
                if all_pass and candles_since <= MAX_CANDLES_SINCE_SWING2:
                    bullish_valid += 1
        
        if bullish_found == 0:
            report.append("")
            report.append("No bullish divergence patterns found")
        
        # BEARISH SCAN
        report.append("")
        report.append("=" * 50)
        report.append("BEARISH DIVERGENCE SCAN")
        report.append(f"(Price Higher High + RSI Lower High + RSI > {RSI_OVERBOUGHT})")
        report.append("=" * 50)
        
        bearish_found = 0
        bearish_valid = 0
        
        for i in range(len(swing_highs) - 1):
            s1 = swing_highs[i]
            s2 = swing_highs[i + 1]
            
            if s2.price > s1.price and s2.rsi < s1.rsi:
                bearish_found += 1
                candles_apart = s2.index - s1.index
                candles_since = current_idx - s2.index
                
                checks = []
                all_pass = True
                
                if MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE:
                    checks.append(f"[OK] Distance: {candles_apart} candles")
                else:
                    checks.append(f"[FAIL] Distance: {candles_apart} (need {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE})")
                    all_pass = False
                
                checks.append(f"[OK] Price: ${s1.price:.2f} -> ${s2.price:.2f} (HIGHER HIGH)")
                checks.append(f"[OK] RSI: {s1.rsi:.1f} -> {s2.rsi:.1f} (LOWER HIGH)")
                
                if s2.rsi > RSI_OVERBOUGHT:
                    checks.append(f"[OK] RSI Zone: {s2.rsi:.1f} > {RSI_OVERBOUGHT} (OVERBOUGHT)")
                else:
                    checks.append(f"[FAIL] RSI Zone: {s2.rsi:.1f} <= {RSI_OVERBOUGHT} (NOT overbought)")
                    all_pass = False
                
                valid, msg = scanner.check_pattern_validity(df, s1, s2, False)
                if valid:
                    checks.append(f"[OK] Pattern: {msg}")
                else:
                    checks.append(f"[FAIL] Pattern: {msg}")
                    all_pass = False
                
                if candles_since <= MAX_CANDLES_SINCE_SWING2:
                    checks.append(f"[OK] Recency: {candles_since} candles ago (RECENT)")
                else:
                    checks.append(f"[--] Recency: {candles_since} candles ago (historical)")
                
                report.append("")
                status = ">>> VALID SIGNAL <<<" if all_pass and candles_since <= MAX_CANDLES_SINCE_SWING2 else "(pattern only)" if all_pass else "(failed)"
                report.append(f"BEARISH #{bearish_found}: {s1.timestamp.strftime('%Y-%m-%d')} -> {s2.timestamp.strftime('%Y-%m-%d')} {status}")
                for c in checks:
                    report.append(f"  {c}")
                
                if all_pass and candles_since <= MAX_CANDLES_SINCE_SWING2:
                    bearish_valid += 1
        
        if bearish_found == 0:
            report.append("")
            report.append("No bearish divergence patterns found")
        
        # Summary
        report.append("")
        report.append("=" * 50)
        report.append("SUMMARY")
        report.append("=" * 50)
        report.append(f"Bullish patterns: {bullish_found} found, {bullish_valid} valid signals")
        report.append(f"Bearish patterns: {bearish_found} found, {bearish_valid} valid signals")
        report.append("")
        
        # TradingView link
        tv_link = f"https://www.tradingview.com/chart/?symbol=BYBIT:{symbol.replace('/', '')}&interval="
        tf_map = {"1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M"}
        tv_link += tf_map.get(timeframe, "240")
        report.append(f"TradingView: {tv_link}")
        
        # Send in chunks
        chunks = []
        current_chunk = ""
        for line in report:
            if len(current_chunk) + len(line) + 1 > 3900:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        
        for chunk in chunks:
            await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
            await asyncio.sleep(0.5)
        
        await update.message.reply_text(f"[Open TradingView Chart]({tv_link})", parse_mode='Markdown')
        
    except Exception as e:
        import traceback
        await update.message.reply_text(f"❌ Error: {e}\n\n{traceback.format_exc()[:500]}")


async def debug_swings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug swing points for a specific month"""
    args = context.args
    
    if not args or len(args) < 3:
        await update.message.reply_text(
            "*Debug Swing Points*\n\n"
            "Usage: `/debug_swings SYMBOL TIMEFRAME MONTH`\n\n"
            "Examples:\n"
            "`/debug_swings SOL/USDT 1d 02` (February)\n"
            "`/debug_swings BTC/USDT 4h 09` (September)",
            parse_mode='Markdown'
        )
        return
    
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol = symbol + '/USDT'
    
    timeframe = args[1]
    if timeframe.upper() == "1M":
        timeframe = "1M"
    else:
        timeframe = timeframe.lower()
    
    month = args[2]
    strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
    
    await update.message.reply_text(f"*Debugging {symbol} {timeframe} month {month}...*\nSwing Strength: {strength}", parse_mode='Markdown')
    
    try:
        df = scanner.fetch_ohlcv(symbol, timeframe, limit=1000)
        
        if df is None:
            await update.message.reply_text("❌ Could not fetch data")
            return
        
        df['month'] = df['timestamp'].dt.strftime('%m')
        month_df = df[df['month'] == month].copy()
        
        if len(month_df) == 0:
            await update.message.reply_text(f"❌ No data for month {month}")
            return
        
        swing_lows = scanner.find_swing_lows(df, timeframe)
        swing_highs = scanner.find_swing_highs(df, timeframe)
        
        swing_lows_month = [s for s in swing_lows if s.timestamp.strftime('%m') == month]
        swing_highs_month = [s for s in swing_highs if s.timestamp.strftime('%m') == month]
        
        report = []
        report.append("=" * 55)
        report.append(f"DEBUG: {symbol} {timeframe} - Month {month}")
        report.append(f"Swing Strength: {strength}")
        report.append("=" * 55)
        report.append("")
        
        report.append(f"[CANDLE DATA FOR MONTH {month}]")
        report.append("-" * 55)
        report.append(f"{'Date':<12} {'Close':<12} {'RSI':<8}")
        report.append("-" * 55)
        
        for _, row in month_df.iterrows():
            date_str = row['timestamp'].strftime('%m-%d')
            rsi_str = f"{row['rsi']:.1f}" if not pd.isna(row['rsi']) else "N/A"
            report.append(f"{date_str:<12} ${row['close']:<11.2f} {rsi_str:<8}")
        
        report.append("")
        report.append(f"[SWING LOWS IN MONTH {month}] (Total: {len(swing_lows_month)})")
        report.append("-" * 55)
        if swing_lows_month:
            for sl in swing_lows_month:
                report.append(f"  {sl.timestamp.strftime('%Y-%m-%d')} | ${sl.price:.2f} | RSI: {sl.rsi:.1f}")
        else:
            report.append("  No swing lows detected")
        
        report.append("")
        report.append(f"[SWING HIGHS IN MONTH {month}] (Total: {len(swing_highs_month)})")
        report.append("-" * 55)
        if swing_highs_month:
            for sh in swing_highs_month:
                report.append(f"  {sh.timestamp.strftime('%Y-%m-%d')} | ${sh.price:.2f} | RSI: {sh.rsi:.1f}")
        else:
            report.append("  No swing highs detected")
        
        # Send
        chunks = []
        current_chunk = ""
        for line in report:
            if len(current_chunk) + len(line) + 1 > 3900:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        
        for chunk in chunks:
            await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
            await asyncio.sleep(0.5)
        
    except Exception as e:
        import traceback
        await update.message.reply_text(f"❌ Error: {e}\n\n{traceback.format_exc()[:500]}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"*Scanning {len(symbols)} coins...*\n"
        f"Timeframes: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"Swing Strength: 2\n"
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
                f"*No divergences found right now*\n\n"
                f"This is normal - divergences are rare.\n"
                f"The bot will alert you when one forms.\n\n"
                f"Time: {format_sl_time()}",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"❌ Scan error: {e}")


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers:
        return
    
    logger.info(f"[{format_sl_time()}] Scheduled scan for {len(subscribers)} subscribers")
    
    try:
        alerts = scanner.scan_all()
        
        if not alerts:
            logger.info(f"[{format_sl_time()}] No signals found")
            return
        
        for alert in alerts:
            logger.info(f"[{format_sl_time()}] Signal: {alert.symbol} {alert.timeframe}")
        
        for chat_id in subscribers.keys():
            for alert in alerts[:5]:
                try:
                    msg = AlertFormatter.format_alert(alert)
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} alerts")
    
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick debug info"""
    symbols = scanner.get_symbols_to_scan()
    
    msg = f"""*Debug Info*

Exchange: {EXCHANGE.upper()}
Coins loaded: {len(symbols)}
Subscribers: {len(subscribers)}
Cooldowns active: {len(scanner.alert_cooldowns)}

*Conditions:*
Swing Strength: 2 (all TFs)
Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE}
Bullish RSI: < {RSI_OVERSOLD}
Bearish RSI: > {RSI_OVERBOUGHT}
Recency: {MAX_CANDLES_SINCE_SWING2} candles

Time: {format_sl_time()}"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


def main():
    threading.Thread(target=run_web_server, daemon=True).start()
    logger.info(f"[{format_sl_time()}] Web server started")
    
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] {EXCHANGE.upper()}: {len(symbols)} coins loaded")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("coins", show_coins))
    app.add_handler(CommandHandler("verify", verify))
    app.add_handler(CommandHandler("conditions", conditions))
    app.add_handler(CommandHandler("debug", debug))
    app.add_handler(CommandHandler("debug_swings", debug_swings))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] Bot starting...")
    logger.info(f"[{format_sl_time()}] Timeframes: {SCAN_TIMEFRAMES}")
    logger.info(f"[{format_sl_time()}] Swing Strength: 2")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
