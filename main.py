"""
RSI Divergence Bot - STRONG SIGNALS ONLY VERSION
Main Telegram Bot - Only sends STRONG signals
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
    RSI_OVERSOLD, RSI_OVERBOUGHT, MAX_CANDLES_SINCE_SWING2_MAP, MAX_CANDLES_SINCE_SWING2,
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
        
        html = f"""<!DOCTYPE html><html><head><title>RSI Divergence Bot - STRONG SIGNALS</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#fff;padding:40px}}
h1{{color:#58a6ff}}p{{color:#8b949e}}.ok{{color:#3fb950}}.strong{{color:#ffd700}}</style></head>
<body>
<h1>RSI Divergence Bot - STRONG SIGNALS ONLY</h1>
<p>Exchange: <b>{EXCHANGE.upper()}</b></p>
<p>Coins: <b>{len(symbols)}</b></p>
<p>Timeframes: <b>{', '.join(SCAN_TIMEFRAMES)}</b></p>
<p>Subscribers: <b>{len(subscribers)}</b></p>
<p class="ok">Status: RUNNING</p>
<p>Time: {format_sl_time()}</p>
<hr>
<h3>Conditions:</h3>
<p>Swing Strength: 4h/1d=2, 1w/1M=1</p>
<p>Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles</p>
<p>Bullish RSI: &lt; {RSI_OVERSOLD}</p>
<p>Bearish RSI: &gt; {RSI_OVERBOUGHT}</p>
<p class="strong">⚡ STRONG ONLY: Bull RSI &lt; 30, Bear RSI &gt; 70</p>
<h3>Alert Timing (1 candle after swing):</h3>
<p>4h: 4 hours | 1d: 1 day | 1w: 1 week | 1M: 1 month</p>
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
    
    msg = f"""*RSI Divergence Bot - STRONG SIGNALS ONLY*

📊 *{len(symbols)}* coins loaded
⏰ Timeframes: {', '.join(SCAN_TIMEFRAMES)}
🦎 Exchange: *{EXCHANGE.upper()}*

*Signal Conditions:*
• Swing Strength: 4h/1d=2, 1w/1M=1
• Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles
• Bullish RSI: < {RSI_OVERSOLD} (oversold)
• Bearish RSI: > {RSI_OVERBOUGHT} (overbought)
• Price-based invalidation only

*⚡ STRONG SIGNALS ONLY:*
• Bullish: RSI < 30 (deeply oversold)
• Bearish: RSI > 70 (deeply overbought)

*Alert Timing (1 candle after swing):*
• 4h: 4 hours after swing
• 1d: 1 day after swing
• 1w: 1 week after swing
• 1M: 1 month after swing

*Commands:*
/subscribe - Get alerts
/scan - Manual scan
/coins - List coins
/verify SYMBOL TF - Check divergences
/regime SYMBOL TF - Check market conditions
/conditions - Show conditions

Time: {format_sl_time()}"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def conditions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current conditions"""
    msg = f"""*📋 Signal Conditions - STRONG ONLY*

*Swing Strength per Timeframe:*
• 4h: 2 candles each side
• 1d: 2 candles each side
• 1w: 1 candle each side
• 1M: 1 candle each side

*Alert Timing (1 candle after swing):*
• 4h: Alert after 4 hours
• 1d: Alert after 1 day
• 1w: Alert after 1 week
• 1M: Alert after 1 month

*For BULLISH Divergence:*
1. Price makes LOWER LOW (Swing2 < Swing1)
2. RSI makes HIGHER LOW (Swing2 RSI > Swing1 RSI)
3. RSI is OVERSOLD (< {RSI_OVERSOLD})
4. Swings are {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
5. No candle CLOSE below Swing2 between swings
6. Swing2 is 1 candle old (just confirmed)
7. ⚡ RSI < 30 (STRONG only)

*For BEARISH Divergence:*
1. Price makes HIGHER HIGH (Swing2 > Swing1)
2. RSI makes LOWER HIGH (Swing2 RSI < Swing1 RSI)
3. RSI is OVERBOUGHT (> {RSI_OVERBOUGHT})
4. Swings are {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles apart
5. No candle CLOSE above Swing2 between swings
6. Swing2 is 1 candle old (just confirmed)
7. ⚡ RSI > 70 (STRONG only)

*Signal Filter:*
🟢 STRONG ONLY: RSI < 30 (bull) or > 70 (bear)
❌ MEDIUM & EARLY signals are filtered out"""
    
    await update.message.reply_text(msg, parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers[chat_id] = {"subscribed": True}
    
    await update.message.reply_text(
        f"✅ *Subscribed to STRONG signals only!*\n\n"
        f"You'll receive alerts only when STRONG divergences are detected.\n"
        f"• Bullish: RSI < 30\n"
        f"• Bearish: RSI > 70\n\n"
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
            "`/verify ETH/USDT 1d 500`\n"
            "`/verify SOL/USDT 1w 200`\n"
            "`/verify BTC/USDT 1M 100`\n\n"
            "Default: 200 candles (max: 1000)\n"
            "Timeframes: 4h, 1d, 1w, 1M\n\n"
            "*Swing Strength:*\n"
            "• 4h, 1d: 2 candles each side\n"
            "• 1w, 1M: 1 candle each side\n\n"
            "*Note:* Verify shows ALL divergences,\n"
            "but only STRONG ones are sent to channel.",
            parse_mode='Markdown'
        )
        return
    
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol = symbol + '/USDT'
    
    timeframe = args[1]
    # Handle weekly and monthly timeframes
    if timeframe.upper() == "1M":
        timeframe = "1M"
    elif timeframe.upper() == "1W":
        timeframe = "1w"
    else:
        timeframe = timeframe.lower()
    
    candles = int(args[2]) if len(args) > 2 else 200
    candles = min(candles, 1000)
    
    # Get swing strength for this timeframe
    strength = SWING_STRENGTH_MAP.get(timeframe, SWING_STRENGTH)
    
    # Get recency for this timeframe
    max_recency = MAX_CANDLES_SINCE_SWING2_MAP.get(timeframe, MAX_CANDLES_SINCE_SWING2)
    
    await update.message.reply_text(
        f"*Analyzing {symbol} {timeframe.upper()}*\n"
        f"Scanning {candles} candles...\n"
        f"Swing Strength: {strength}\n"
        f"Recency Window: {max_recency} candle(s)\n"
        f"STRONG filter: RSI < 30 (bull), > 70 (bear)",
        parse_mode='Markdown'
    )
    
    try:
        df = scanner.fetch_ohlcv(symbol, timeframe, limit=candles)
        
        if df is None or len(df) < 50:
            await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
            return
        
        current_idx = len(df) - 1
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        swing_lows = scanner.find_swing_lows(df, timeframe)
        swing_highs = scanner.find_swing_highs(df, timeframe)
        
        # Build verification report
        report = []
        report.append("=" * 60)
        report.append(f"DIVERGENCE VERIFICATION: {symbol} {timeframe.upper()}")
        report.append("=" * 60)
        report.append(f"Candles analyzed: {len(df)}")
        report.append(f"Current Price: ${current_price:.4f}")
        report.append(f"Current RSI: {current_rsi:.1f}")
        report.append(f"Swing Strength: {strength}")
        report.append(f"Recency: {max_recency} candle(s)")
        report.append(f"Swing Distance: {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE}")
        report.append("")
        report.append(f"Swing Lows Found: {len(swing_lows)}")
        report.append(f"Swing Highs Found: {len(swing_highs)}")
        report.append("")
        
        # Check for bullish divergences (show all, mark strong)
        bull_divs = []
        for i in range(len(swing_lows) - 1):
            sw1 = swing_lows[i]
            sw2 = swing_lows[i + 1]
            
            candles_apart = sw2.index - sw1.index
            if not (MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE):
                continue
            
            if sw2.price >= sw1.price:  # Not lower low
                continue
                
            if sw2.rsi <= sw1.rsi:  # Not higher RSI
                continue
                
            if sw2.rsi >= RSI_OVERSOLD:  # Not oversold
                continue
            
            # Check pattern validity
            pattern_valid, _ = scanner.check_pattern_validity(df, sw1, sw2, True)
            if not pattern_valid:
                continue
            
            candles_since = current_idx - sw2.index
            is_recent = candles_since <= max_recency
            is_strong = sw2.rsi < 30
            
            bull_divs.append({
                'sw1': sw1,
                'sw2': sw2,
                'candles_apart': candles_apart,
                'candles_since': candles_since,
                'is_recent': is_recent,
                'is_strong': is_strong
            })
        
        # Check for bearish divergences (show all, mark strong)
        bear_divs = []
        for i in range(len(swing_highs) - 1):
            sw1 = swing_highs[i]
            sw2 = swing_highs[i + 1]
            
            candles_apart = sw2.index - sw1.index
            if not (MIN_SWING_DISTANCE <= candles_apart <= MAX_SWING_DISTANCE):
                continue
            
            if sw2.price <= sw1.price:  # Not higher high
                continue
                
            if sw2.rsi >= sw1.rsi:  # Not lower RSI
                continue
                
            if sw2.rsi <= RSI_OVERBOUGHT:  # Not overbought
                continue
            
            # Check pattern validity
            pattern_valid, _ = scanner.check_pattern_validity(df, sw1, sw2, False)
            if not pattern_valid:
                continue
            
            candles_since = current_idx - sw2.index
            is_recent = candles_since <= max_recency
            is_strong = sw2.rsi > 70
            
            bear_divs.append({
                'sw1': sw1,
                'sw2': sw2,
                'candles_apart': candles_apart,
                'candles_since': candles_since,
                'is_recent': is_recent,
                'is_strong': is_strong
            })
        
        # Report bullish divergences
        report.append("=" * 60)
        report.append("BULLISH DIVERGENCES (Price LL + RSI HL)")
        report.append("=" * 60)
        
        if bull_divs:
            for d in bull_divs:
                sw1, sw2 = d['sw1'], d['sw2']
                status = []
                if d['is_recent']:
                    status.append("RECENT")
                if d['is_strong']:
                    status.append("⚡STRONG")
                status_str = " | ".join(status) if status else "historical"
                
                alert_status = "✅ WOULD ALERT" if (d['is_recent'] and d['is_strong']) else "❌ filtered"
                
                report.append(f"\n[{status_str}] {alert_status}")
                report.append(f"  Swing 1: {sw1.timestamp.strftime('%Y-%m-%d %H:%M')}")
                report.append(f"    Price: ${sw1.price:.4f} | RSI: {sw1.rsi:.1f}")
                report.append(f"  Swing 2: {sw2.timestamp.strftime('%Y-%m-%d %H:%M')}")
                report.append(f"    Price: ${sw2.price:.4f} | RSI: {sw2.rsi:.1f}")
                report.append(f"  Distance: {d['candles_apart']} candles")
                report.append(f"  Age: {d['candles_since']} candles ago")
        else:
            report.append("\nNo bullish divergences found.")
        
        # Report bearish divergences
        report.append("")
        report.append("=" * 60)
        report.append("BEARISH DIVERGENCES (Price HH + RSI LH)")
        report.append("=" * 60)
        
        if bear_divs:
            for d in bear_divs:
                sw1, sw2 = d['sw1'], d['sw2']
                status = []
                if d['is_recent']:
                    status.append("RECENT")
                if d['is_strong']:
                    status.append("⚡STRONG")
                status_str = " | ".join(status) if status else "historical"
                
                alert_status = "✅ WOULD ALERT" if (d['is_recent'] and d['is_strong']) else "❌ filtered"
                
                report.append(f"\n[{status_str}] {alert_status}")
                report.append(f"  Swing 1: {sw1.timestamp.strftime('%Y-%m-%d %H:%M')}")
                report.append(f"    Price: ${sw1.price:.4f} | RSI: {sw1.rsi:.1f}")
                report.append(f"  Swing 2: {sw2.timestamp.strftime('%Y-%m-%d %H:%M')}")
                report.append(f"    Price: ${sw2.price:.4f} | RSI: {sw2.rsi:.1f}")
                report.append(f"  Distance: {d['candles_apart']} candles")
                report.append(f"  Age: {d['candles_since']} candles ago")
        else:
            report.append("\nNo bearish divergences found.")
        
        # Summary
        report.append("")
        report.append("=" * 60)
        report.append("SUMMARY")
        report.append("=" * 60)
        strong_recent_bull = sum(1 for d in bull_divs if d['is_recent'] and d['is_strong'])
        strong_recent_bear = sum(1 for d in bear_divs if d['is_recent'] and d['is_strong'])
        report.append(f"Total Bullish: {len(bull_divs)} | STRONG & Recent: {strong_recent_bull}")
        report.append(f"Total Bearish: {len(bear_divs)} | STRONG & Recent: {strong_recent_bear}")
        report.append(f"Alertable NOW: {strong_recent_bull + strong_recent_bear}")
        
        # Recent swing points for reference
        report.append("")
        report.append("[RECENT SWING LOWS]")
        report.append("-" * 55)
        for sl in swing_lows[-15:]:
            strong_mark = "⚡" if sl.rsi < 30 else "  "
            report.append(f"{strong_mark} {sl.timestamp.strftime('%Y-%m-%d %H:%M')} | ${sl.price:.4f} | RSI: {sl.rsi:.1f}")
        
        report.append("")
        report.append("[RECENT SWING HIGHS]")
        report.append("-" * 55)
        for sh in swing_highs[-15:]:
            strong_mark = "⚡" if sh.rsi > 70 else "  "
            report.append(f"{strong_mark} {sh.timestamp.strftime('%Y-%m-%d %H:%M')} | ${sh.price:.4f} | RSI: {sh.rsi:.1f}")
        
        # Send report in chunks
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
        f"*Scanning {len(symbols)} coins for STRONG signals...*\n"
        f"Timeframes: {', '.join(SCAN_TIMEFRAMES)}\n"
        f"Swing Strength: 4h/1d=2, 1w/1M=1\n"
        f"⚡ STRONG only: RSI < 30 (bull), > 70 (bear)\n"
        f"This may take a few minutes...",
        parse_mode='Markdown'
    )
    
    try:
        alerts = scanner.scan_all()
        
        if alerts:
            await update.message.reply_text(
                f"✅ *Found {len(alerts)} STRONG signals!*",
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
                    f"_...and {len(alerts) - 10} more STRONG signals_",
                    parse_mode='Markdown'
                )
        else:
            await update.message.reply_text(
                f"*No STRONG divergences found right now*\n\n"
                f"STRONG signals require:\n"
                f"• Bullish: RSI < 30\n"
                f"• Bearish: RSI > 70\n\n"
                f"The bot will alert you when one forms.\n\n"
                f"Time: {format_sl_time()}",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        await update.message.reply_text(f"❌ Scan error: {e}")


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers:
        return
    
    logger.info(f"[{format_sl_time()}] Scheduled scan for {len(subscribers)} subscribers (STRONG only)")
    
    try:
        alerts = scanner.scan_all()
        
        if not alerts:
            logger.info(f"[{format_sl_time()}] No STRONG signals found")
            return
        
        for alert in alerts:
            logger.info(f"[{format_sl_time()}] STRONG Signal: {alert.symbol} {alert.timeframe}")
        
        for chat_id in subscribers.keys():
            for alert in alerts[:5]:
                try:
                    # Get market regime info
                    regime, volatility = scanner.get_market_info(alert.symbol, alert.timeframe)
                    
                    if regime and volatility:
                        msg = AlertFormatter.format_alert_with_regime(alert, regime, volatility)
                    else:
                        msg = AlertFormatter.format_alert(alert)
                    
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        logger.info(f"[{format_sl_time()}] Sent {len(alerts)} STRONG alerts")
    
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
            "*Timeframes:* 4h, 1d, 1w, 1M\n"
            "*Default:* 4h",
            parse_mode='Markdown'
        )
        return
    
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol = symbol + '/USDT'
    
    timeframe = "4h"
    if len(args) > 1:
        tf = args[1]
        if tf.upper() == "1M":
            timeframe = "1M"
        elif tf.upper() == "1W":
            timeframe = "1w"
        else:
            timeframe = tf.lower()
    
    await update.message.reply_text(f"📊 Analyzing {symbol} {timeframe.upper()}...")
    
    try:
        # Fetch data
        df = scanner.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if df is None or len(df) < 50:
            await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
            return
        
        # Get regime and volatility
        regime_info = get_market_regime(df)
        volatility_info = get_volatility_status(df)
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        # Format and send
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
    
    msg = f"""*Debug Info - STRONG SIGNALS ONLY*

Exchange: {EXCHANGE.upper()}
Coins loaded: {len(symbols)}
Subscribers: {len(subscribers)}
Cooldowns active: {len(scanner.alert_cooldowns)}

*Swing Strength:*
• 4h: {SWING_STRENGTH_MAP.get('4h', 2)} candles
• 1d: {SWING_STRENGTH_MAP.get('1d', 2)} candles
• 1w: {SWING_STRENGTH_MAP.get('1w', 1)} candle
• 1M: {SWING_STRENGTH_MAP.get('1M', 1)} candle

*Recency Windows (all = 1 candle):*
• 4h: {MAX_CANDLES_SINCE_SWING2_MAP.get('4h', 1)} candle (4 hours)
• 1d: {MAX_CANDLES_SINCE_SWING2_MAP.get('1d', 1)} candle (1 day)
• 1w: {MAX_CANDLES_SINCE_SWING2_MAP.get('1w', 1)} candle (1 week)
• 1M: {MAX_CANDLES_SINCE_SWING2_MAP.get('1M', 1)} candle (1 month)

*Swing Distance:* {MIN_SWING_DISTANCE}-{MAX_SWING_DISTANCE} candles

*RSI Zones:*
Bullish: < {RSI_OVERSOLD}
Bearish: > {RSI_OVERBOUGHT}

*⚡ STRONG Filter:*
Bullish: RSI < 30
Bearish: RSI > 70

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
    app.add_handler(CommandHandler("regime", regime))
    app.add_handler(CommandHandler("debug", debug))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] Bot starting - STRONG SIGNALS ONLY...")
    logger.info(f"[{format_sl_time()}] Timeframes: {SCAN_TIMEFRAMES}")
    logger.info(f"[{format_sl_time()}] Swing Strength: 4h/1d=2, 1w/1M=1")
    logger.info(f"[{format_sl_time()}] Alert Timing: 1 candle after swing (all TFs)")
    logger.info(f"[{format_sl_time()}] STRONG filter: RSI < 30 (bull), > 70 (bear)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
