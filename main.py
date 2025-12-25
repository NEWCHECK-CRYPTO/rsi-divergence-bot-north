"""
Main Telegram Bot - V10 with Debug Commands
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
    DivergenceScanner, AlertFormatter, SignalStrength, 
    get_sl_time, format_sl_time, calculate_adx
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
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Bot Working</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}</style></head>
<body>
<h1>RSI Divergence Bot - V10</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{len(symbols)}</b></p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p>Status: <span style="color:#3fb950">RUNNING</span></p>
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
        f"*RSI Divergence Bot - V10*\n\n"
        f"*{len(symbols)}* coins loaded\n"
        f"Timeframes: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"Exchange: *{EXCHANGE.upper()}*\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get alerts\n"
        f"/scan - Manual scan\n"
        f"/coins - List coins\n"
        f"/debug - Full diagnostic\n"
        f"/debugraw - Raw API data\n\n"
        f"Time: {format_sl_time()}",
        parse_mode='Markdown'
    )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {"min_strength": SignalStrength.EARLY}
    
    await update.message.reply_text(
        f"*Subscribed!*\n\n"
        f"Exchange: {EXCHANGE.upper()}\n"
        f"Scan every {SCAN_INTERVAL//60} min\n"
        f"Time: {format_sl_time()}",
        parse_mode='Markdown'
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.pop(update.effective_chat.id, None)
    await update.message.reply_text("Unsubscribed")


async def show_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching coins...")
    symbols = scanner.fetch_top_coins_by_volume(TOP_COINS_COUNT)
    
    msg = f"*Top {len(symbols)} Coins on {EXCHANGE.upper()}*\n\n"
    for i, s in enumerate(symbols[:30], 1):
        msg += f"{i}. {s}\n"
    
    if len(symbols) > 30:
        msg += f"\n_...and {len(symbols) - 30} more_"
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comprehensive debug command to test all bot components"""
    await update.message.reply_text("*Running Full Diagnostic...*", parse_mode='Markdown')
    
    debug_report = []
    debug_report.append("=" * 40)
    debug_report.append("RSI DIVERGENCE BOT - DIAGNOSTIC REPORT")
    debug_report.append("=" * 40)
    debug_report.append(f"Time: {format_sl_time()}")
    debug_report.append(f"Exchange: {EXCHANGE.upper()}")
    debug_report.append("")
    
    # TEST 1: Exchange Connection
    debug_report.append("[TEST 1] Exchange Connection")
    debug_report.append("-" * 40)
    try:
        markets = scanner.exchange.load_markets()
        total_markets = len(markets)
        usdt_markets = len([m for m in markets if m.endswith('/USDT')])
        debug_report.append(f"[OK] Connected to {EXCHANGE.upper()}")
        debug_report.append(f"     Total markets: {total_markets}")
        debug_report.append(f"     USDT pairs: {usdt_markets}")
    except Exception as e:
        debug_report.append(f"[FAIL] Connection FAILED: {e}")
    debug_report.append("")
    
    # TEST 2: Fetch Top Coins
    debug_report.append("[TEST 2] Fetch Top Coins by Volume")
    debug_report.append("-" * 40)
    try:
        symbols = scanner.fetch_top_coins_by_volume(TOP_COINS_COUNT)
        debug_report.append(f"[OK] Fetched {len(symbols)} coins")
        if len(symbols) > 0:
            debug_report.append(f"     Top 5: {', '.join(symbols[:5])}")
            debug_report.append(f"     #50: {symbols[49] if len(symbols) >= 50 else 'N/A'}")
            debug_report.append(f"     #100: {symbols[99] if len(symbols) >= 100 else 'N/A'}")
        else:
            debug_report.append(f"[WARN] 0 coins returned!")
    except Exception as e:
        debug_report.append(f"[FAIL] Fetch FAILED: {e}")
        symbols = []
    debug_report.append("")
    
    # TEST 3: OHLCV Data Fetch
    debug_report.append("[TEST 3] OHLCV Data Fetch")
    debug_report.append("-" * 40)
    test_symbol = symbols[0] if symbols else "BTC/USDT"
    try:
        for tf in SCAN_TIMEFRAMES:
            df = scanner.fetch_ohlcv(test_symbol, tf, limit=50)
            if df is not None and len(df) > 0:
                debug_report.append(f"[OK] {tf}: {len(df)} candles")
                debug_report.append(f"     Latest: O={df['open'].iloc[-1]:.2f} H={df['high'].iloc[-1]:.2f}")
                debug_report.append(f"             L={df['low'].iloc[-1]:.2f} C={df['close'].iloc[-1]:.2f}")
                debug_report.append(f"     Time: {df['timestamp'].iloc[-1]}")
            else:
                debug_report.append(f"[FAIL] {tf}: No data returned")
    except Exception as e:
        debug_report.append(f"[FAIL] OHLCV FAILED: {e}")
    debug_report.append("")
    
    # TEST 4: RSI Calculation
    debug_report.append("[TEST 4] RSI Calculation")
    debug_report.append("-" * 40)
    try:
        df = scanner.fetch_ohlcv(test_symbol, "4h", limit=100)
        if df is not None:
            rsi_values = df['rsi'].dropna()
            if len(rsi_values) > 0:
                debug_report.append(f"[OK] RSI calculated for {test_symbol}")
                debug_report.append(f"     Current RSI: {rsi_values.iloc[-1]:.2f}")
                debug_report.append(f"     RSI Range: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
                debug_report.append(f"     RSI Mean: {rsi_values.mean():.2f}")
                
                # Show last 5 RSI values
                last5 = rsi_values.tail(5).tolist()
                debug_report.append(f"     Last 5: {', '.join([f'{r:.1f}' for r in last5])}")
            else:
                debug_report.append(f"[FAIL] RSI calculation returned empty")
    except Exception as e:
        debug_report.append(f"[FAIL] RSI FAILED: {e}")
    debug_report.append("")
    
    # TEST 5: Swing Detection
    debug_report.append("[TEST 5] Swing Point Detection")
    debug_report.append("-" * 40)
    try:
        df = scanner.fetch_ohlcv(test_symbol, "4h", limit=100)
        if df is not None:
            swing_highs = scanner.find_swing_highs(df, 3)
            swing_lows = scanner.find_swing_lows(df, 3)
            
            debug_report.append(f"[OK] Swing detection working")
            debug_report.append(f"     Swing Highs found: {len(swing_highs)}")
            debug_report.append(f"     Swing Lows found: {len(swing_lows)}")
            
            if swing_highs:
                latest_high = swing_highs[-1]
                debug_report.append(f"     Latest High: ${latest_high.price:.2f} (RSI: {latest_high.rsi:.1f})")
            if swing_lows:
                latest_low = swing_lows[-1]
                debug_report.append(f"     Latest Low: ${latest_low.price:.2f} (RSI: {latest_low.rsi:.1f})")
    except Exception as e:
        debug_report.append(f"[FAIL] Swing Detection FAILED: {e}")
    debug_report.append("")
    
    # TEST 6: ADX Calculation
    debug_report.append("[TEST 6] ADX Calculation")
    debug_report.append("-" * 40)
    try:
        df = scanner.fetch_ohlcv(test_symbol, "4h", limit=100)
        if df is not None:
            adx = calculate_adx(df, period=14)
            debug_report.append(f"[OK] ADX calculated: {adx:.2f}")
            if adx > 25:
                debug_report.append(f"     Trend: STRONG")
            elif adx > 20:
                debug_report.append(f"     Trend: MODERATE")
            else:
                debug_report.append(f"     Trend: WEAK")
    except Exception as e:
        debug_report.append(f"[FAIL] ADX FAILED: {e}")
    debug_report.append("")
    
    # TEST 7: Divergence Detection (Quick scan on 5 coins)
    debug_report.append("[TEST 7] Divergence Detection")
    debug_report.append("-" * 40)
    try:
        test_coins = symbols[:5] if len(symbols) >= 5 else symbols
        divergences_found = 0
        
        for sym in test_coins:
            for tf in ["4h"]:
                df = scanner.fetch_ohlcv(sym, tf, limit=100)
                if df is not None:
                    swing_highs = scanner.find_swing_highs(df, 3)
                    swing_lows = scanner.find_swing_lows(df, 3)
                    div = scanner.detect_divergence(df, swing_lows, swing_highs)
                    if div:
                        divergences_found += 1
                        debug_report.append(f"     Found: {sym}: {div.divergence_type.value}")
        
        debug_report.append(f"[OK] Divergence detection working")
        debug_report.append(f"     Scanned: {len(test_coins)} coins on 4h")
        debug_report.append(f"     Raw divergences: {divergences_found}")
    except Exception as e:
        debug_report.append(f"[FAIL] Divergence Detection FAILED: {e}")
    debug_report.append("")
    
    # TEST 8: Full Signal Pipeline (1 coin)
    debug_report.append("[TEST 8] Full Signal Pipeline")
    debug_report.append("-" * 40)
    try:
        # Temporarily disable cooldown for test
        original_cooldowns = scanner.alert_cooldowns.copy()
        scanner.alert_cooldowns = {}
        
        test_coin = symbols[0] if symbols else "BTC/USDT"
        alerts = scanner.scan_symbol(test_coin, "4h")
        
        scanner.alert_cooldowns = original_cooldowns
        
        if alerts:
            debug_report.append(f"[OK] Full pipeline working - Signal found!")
            debug_report.append(f"     {test_coin} 4h: {alerts[0].divergence.divergence_type.value}")
            debug_report.append(f"     Confidence: {alerts[0].total_confidence*100:.0f}%")
        else:
            debug_report.append(f"[OK] Full pipeline working - No signal (normal)")
            debug_report.append(f"     {test_coin} 4h: No divergence meeting criteria")
    except Exception as e:
        debug_report.append(f"[FAIL] Full Pipeline FAILED: {e}")
        import traceback
        debug_report.append(f"     {traceback.format_exc()[:200]}")
    debug_report.append("")
    
    # TEST 9: Volume Ranking
    debug_report.append("[TEST 9] Volume Ranking")
    debug_report.append("-" * 40)
    try:
        if scanner.volume_ranks:
            debug_report.append(f"[OK] Volume ranks stored: {len(scanner.volume_ranks)}")
            top3 = list(scanner.volume_ranks.items())[:3]
            for sym, rank in top3:
                debug_report.append(f"     #{rank}: {sym}")
        else:
            debug_report.append(f"[WARN] No volume ranks stored")
    except Exception as e:
        debug_report.append(f"[FAIL] Volume Ranking FAILED: {e}")
    debug_report.append("")
    
    # TEST 10: Bot Status
    debug_report.append("[TEST 10] Bot Status")
    debug_report.append("-" * 40)
    debug_report.append(f"[OK] Subscribers: {len(subscribers)}")
    debug_report.append(f"[OK] Cooldowns active: {len(scanner.alert_cooldowns)}")
    debug_report.append(f"[OK] Scan interval: {SCAN_INTERVAL}s ({SCAN_INTERVAL//60}min)")
    debug_report.append(f"[OK] Timeframes: {', '.join(SCAN_TIMEFRAMES)}")
    debug_report.append("")
    
    # SUMMARY
    debug_report.append("=" * 40)
    debug_report.append("DIAGNOSTIC SUMMARY")
    debug_report.append("=" * 40)
    
    all_passed = len(symbols) > 0
    if all_passed:
        debug_report.append("[OK] All critical tests PASSED")
        debug_report.append("[OK] Bot is OPERATIONAL")
    else:
        debug_report.append("[FAIL] Some tests FAILED")
        debug_report.append("[WARN] Check logs for details")
    
    debug_report.append("")
    debug_report.append(f"Report generated: {format_sl_time()}")
    
    # Send report (split if too long)
    full_report = "\n".join(debug_report)
    
    if len(full_report) > 4000:
        # Split into chunks
        chunks = [full_report[i:i+4000] for i in range(0, len(full_report), 4000)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
            await asyncio.sleep(0.3)
    else:
        await update.message.reply_text(f"```\n{full_report}\n```", parse_mode='Markdown')


async def debug_raw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Raw data debug - shows actual API responses"""
    await update.message.reply_text("*Fetching Raw Data from Bybit...*", parse_mode='Markdown')
    
    try:
        # Raw ticker fetch
        import ccxt
        exchange = ccxt.bybit()
        
        # Test 1: Raw markets
        markets = exchange.load_markets()
        usdt_futures = [m for m in markets.keys() if m.endswith('/USDT') and ':USDT' not in m]
        
        msg = f"*Raw Bybit Data*\n\n"
        msg += f"Total markets: {len(markets)}\n"
        msg += f"USDT Spot pairs: {len(usdt_futures)}\n\n"
        
        # Test 2: Raw tickers
        tickers = exchange.fetch_tickers()
        usdt_tickers = {k: v for k, v in tickers.items() if k.endswith('/USDT')}
        
        msg += f"Tickers received: {len(tickers)}\n"
        msg += f"USDT tickers: {len(usdt_tickers)}\n\n"
        
        # Sort by volume
        sorted_tickers = sorted(
            usdt_tickers.items(),
            key=lambda x: x[1].get('quoteVolume', 0) or 0,
            reverse=True
        )
        
        msg += f"*Top 10 by Volume:*\n"
        for i, (sym, ticker) in enumerate(sorted_tickers[:10], 1):
            vol = ticker.get('quoteVolume', 0) or 0
            price = ticker.get('last', 0) or 0
            msg += f"{i}. {sym}: ${vol/1e6:.1f}M (${price:.4f})\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
        # Test 3: Raw OHLCV
        test_sym = sorted_tickers[0][0] if sorted_tickers else "BTC/USDT"
        ohlcv = exchange.fetch_ohlcv(test_sym, '4h', limit=5)
        
        msg2 = f"*Raw OHLCV for {test_sym}*\n\n"
        msg2 += "```\n"
        msg2 += "Time                 O        H        L        C\n"
        msg2 += "-" * 55 + "\n"
        for candle in ohlcv:
            ts = datetime.fromtimestamp(candle[0]/1000)
            msg2 += f"{ts.strftime('%m-%d %H:%M')}  {candle[1]:<8.2f} {candle[2]:<8.2f} {candle[3]:<8.2f} {candle[4]:<8.2f}\n"
        msg2 += "```"
        
        await update.message.reply_text(msg2, parse_mode='Markdown')
        
    except Exception as e:
        import traceback
        await update.message.reply_text(f"Error: {e}\n\n{traceback.format_exc()[:500]}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"*Scanning {len(symbols)} coins...*\n\n"
        f"Timeframes: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"This takes 3-5 minutes...",
        parse_mode='Markdown'
    )
    
    try:
        alerts = scanner.scan_all()
        
        if alerts:
            await update.message.reply_text(
                f"*Found {len(alerts)} signals!*",
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
                f"*No divergences found*\n\n"
                f"Try again in 1-2 hours.\n\n"
                f"Time: {format_sl_time()}",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"Scan error: {e}")


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
            logger.info(f"[{format_sl_time()}] Signal: {alert.symbol} {alert.signal_tf}")
        
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
    app.add_handler(CommandHandler("debug", debug))
    app.add_handler(CommandHandler("debugraw", debug_raw))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] Bot V10 ({EXCHANGE.upper()}) starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
