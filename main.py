"""
RSI Divergence Alert Bot - Top 100 Binance Coins
All times in Sri Lanka timezone (Asia/Colombo)
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
from divergence_scanner import DivergenceScanner, AlertFormatter, get_sl_time, format_sl_time

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

scanner = DivergenceScanner()
rag = None
subscribers = set()

SL_TZ = pytz.timezone(TIMEZONE)


# =============================================================================
# WEB SERVER
# =============================================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        symbols = scanner.get_symbols_to_scan()
        top_5 = symbols[:5] if symbols else []
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RSI Divergence Bot</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #fff;
            padding: 40px 20px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .card {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        }}
        .status {{ 
            display: inline-block;
            padding: 8px 16px;
            background: #00c853;
            border-radius: 20px;
            font-weight: 600;
        }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 20px; }}
        .stat {{ background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: 700; }}
        .stat-label {{ opacity: 0.8; font-size: 0.85rem; }}
        .top-coins {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }}
        .coin {{ background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-weight: 500; }}
        code {{ background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 6px; }}
        ul {{ list-style: none; }}
        li {{ padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        .time {{ font-size: 1.2rem; margin-top: 10px; opacity: 0.9; }}
        .flag {{ font-size: 1.5rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RSI Divergence Alert Bot</h1>
        <p style="opacity:0.8; margin-bottom:30px;">Top {TOP_COINS_COUNT} Binance Coins by Volume | EmperorBTC Methodology</p>
        
        <div class="card">
            <span class="status">✅ Running</span>
            <p class="time"><span class="flag">🇱🇰</span> Sri Lanka Time: <strong>{format_sl_time()}</strong></p>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(symbols)}</div>
                    <div class="stat-label">Coins</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(subscribers)}</div>
                    <div class="stat-label">Subscribers</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{SCAN_INTERVAL // 60}m</div>
                    <div class="stat-label">Scan Interval</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(SCAN_TIMEFRAMES)}</div>
                    <div class="stat-label">Timeframes</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom:15px;">🔥 Top 5 by Volume</h3>
            <div class="top-coins">
                {"".join([f'<span class="coin">#{i+1} {s}</span>' for i, s in enumerate(top_5)])}
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom:15px;">💬 Commands</h3>
            <ul>
                <li><code>/start</code> - Welcome</li>
                <li><code>/subscribe</code> - Get alerts</li>
                <li><code>/unsubscribe</code> - Stop alerts</li>
                <li><code>/scan</code> - Manual scan</li>
                <li><code>/top</code> - View top 20 coins</li>
                <li><code>/rules</code> - Trading rules</li>
                <li><code>/time</code> - Current Sri Lanka time</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        pass


def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f"Health server on port {port}")
    server.serve_forever()


# =============================================================================
# TELEGRAM HANDLERS
# =============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🤖 *RSI Divergence Alert Bot*\n\n"
        f"📊 Monitoring *Top {TOP_COINS_COUNT}* Binance coins by volume\n"
        f"🇱🇰 All times in Sri Lanka timezone\n\n"
        f"*Commands:*\n"
        f"/subscribe - Get live alerts\n"
        f"/unsubscribe - Stop alerts\n"
        f"/scan - Manual scan now\n"
        f"/top - View top 20 coins\n"
        f"/rules - Trading rules\n"
        f"/time - Current SL time\n\n"
        f"⚠️ Not financial advice!",
        parse_mode='Markdown'
    )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in subscribers:
        await update.message.reply_text("✅ Already subscribed!")
        return
    subscribers.add(chat_id)
    
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"✅ *Subscribed to Live Alerts!*\n\n"
        f"📊 Monitoring: *{len(symbols)}* coins\n"
        f"⏰ Scan interval: *{SCAN_INTERVAL // 60}* minutes\n"
        f"🇱🇰 Time: {format_sl_time()}\n\n"
        f"You'll receive alerts when divergences are confirmed!",
        parse_mode='Markdown'
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.discard(update.effective_chat.id)
    await update.message.reply_text("❌ Unsubscribed from alerts")


async def show_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🇱🇰 *Sri Lanka Time*\n\n"
        f"📅 {format_sl_time()}\n"
        f"🌐 Timezone: Asia/Colombo (UTC+5:30)",
        parse_mode='Markdown'
    )


async def show_top_coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Fetching top coins by volume...")
    
    try:
        symbols = scanner.fetch_top_coins_by_volume(TOP_COINS_COUNT)
        top_20 = symbols[:20]
        
        lines = [f"🔥 *Top 20 Coins by 24h Volume*\n"]
        for i, symbol in enumerate(top_20, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            lines.append(f"{medal} {symbol}")
        
        lines.append(f"\n🇱🇰 Updated: {format_sl_time()}")
        
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = scanner.get_symbols_to_scan()
    await update.message.reply_text(
        f"🔍 Scanning {len(symbols)} coins...\n"
        f"⏳ This takes 2-3 minutes\n"
        f"🇱🇰 Started: {format_sl_time()}"
    )
    
    try:
        alerts = scanner.scan_all()
        
        if alerts:
            await update.message.reply_text(f"✅ Found {len(alerts)} signal(s)!")
            for alert in alerts:
                await update.message.reply_text(AlertFormatter.format_alert(alert))
        else:
            await update.message.reply_text(
                f"📭 No confirmed signals right now.\n\n"
                f"🇱🇰 Completed: {format_sl_time()}"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def show_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("🟢 Bullish", callback_data="rules_bullish"),
            InlineKeyboardButton("🔴 Bearish", callback_data="rules_bearish"),
        ],
        [
            InlineKeyboardButton("📈 MS", callback_data="rules_ms"),
            InlineKeyboardButton("✅ Confirm", callback_data="rules_confirm"),
        ]
    ]
    await update.message.reply_text(
        "📚 *Trading Rules* - Select:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def rules_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    rules = {
        "bullish": "🟢 *BULLISH DIVERGENCE*\n_(Look at LOWS)_\n\n• *Strong:* Price LL + RSI HL → 85%\n• *Medium:* Price DB + RSI HL → 75%\n• *Weak:* Price LL + RSI DB → 60%\n• *Hidden:* Price HL + RSI LL → 70%",
        "bearish": "🔴 *BEARISH DIVERGENCE*\n_(Look at HIGHS)_\n\n• *Strong:* Price HH + RSI LH → 85%\n• *Medium:* Price DT + RSI LH → 75%\n• *Weak:* Price HH + RSI DT → 60%\n• *Hidden:* Price LH + RSI HH → 70%",
        "ms": "📈 *MARKET STRUCTURE*\n\n• HH = Higher High\n• HL = Higher Low\n• LL = Lower Low\n• LH = Lower High\n• BOS = Break of Structure\n• CHoCH = Change of Character\n\n_Bullish:_ HH + HL\n_Bearish:_ LL + LH",
        "confirm": "✅ *CONFIRMATION*\n\n*LONG:*\n1. Bullish div on higher TF\n2. Bullish CHoCH/BOS on lower TF\n3. Entry above CHoCH\n\n*SHORT:* Opposite"
    }
    
    rule_type = query.data.replace("rules_", "")
    await query.edit_message_text(rules.get(rule_type, "Not found"), parse_mode='Markdown')


async def ask_rag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ask <question>")
        return
    
    question = " ".join(context.args)
    await update.message.reply_text("🤔 Thinking...")
    
    try:
        if rag:
            answer = rag.query(question)
            await update.message.reply_text(answer)
        else:
            await update.message.reply_text("⚠️ Gemini not configured. Use /rules")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers:
        logger.info(f"[{format_sl_time()}] No subscribers, skipping scan")
        return
    
    logger.info(f"[{format_sl_time()}] Starting scheduled scan for {len(subscribers)} subscribers")
    
    try:
        alerts = scanner.scan_all()
        
        for alert in alerts:
            message = AlertFormatter.format_alert(alert)
            for chat_id in subscribers:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=message)
                except Exception as e:
                    logger.error(f"Send error: {e}")
        
        if alerts:
            logger.info(f"[{format_sl_time()}] Sent {len(alerts)} alerts")
            
    except Exception as e:
        logger.error(f"[{format_sl_time()}] Scan error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global rag
    
    # Start web server
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logger.info(f"[{format_sl_time()}] Health server started")
    
    # Initialize RAG
    try:
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
            from rag_module import TradingKnowledgeRAG
            rag = TradingKnowledgeRAG()
            logger.info(f"[{format_sl_time()}] Gemini RAG ready")
    except Exception as e:
        logger.warning(f"[{format_sl_time()}] RAG init failed: {e}")
    
    # Pre-fetch top coins
    logger.info(f"[{format_sl_time()}] Fetching top {TOP_COINS_COUNT} coins...")
    symbols = scanner.get_symbols_to_scan()
    logger.info(f"[{format_sl_time()}] Loaded {len(symbols)} symbols")
    
    # Telegram bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("top", show_top_coins))
    app.add_handler(CommandHandler("rules", show_rules))
    app.add_handler(CommandHandler("time", show_time))
    app.add_handler(CommandHandler("ask", ask_rag))
    app.add_handler(CallbackQueryHandler(rules_callback, pattern="^rules_"))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info(f"[{format_sl_time()}] 🤖 Bot starting...")
    logger.info(f"[{format_sl_time()}] 📊 Monitoring {len(symbols)} coins")
    logger.info(f"[{format_sl_time()}] ⏰ Scan interval: {SCAN_INTERVAL}s")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
