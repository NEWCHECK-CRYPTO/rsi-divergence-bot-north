"""
RSI Divergence Alert Bot - Northflank Version
"""

import asyncio
import logging
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import TELEGRAM_TOKEN, SYMBOLS, SCAN_TIMEFRAMES, SCAN_INTERVAL
from divergence_scanner import DivergenceScanner, AlertFormatter
from rag_module import TradingKnowledgeRAG, SimpleKnowledgeBase

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

scanner = DivergenceScanner()
rag = None
simple_kb = SimpleKnowledgeBase()
subscribers = set()


# =============================================================================
# WEB SERVER (Health Check for Northflank)
# =============================================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
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
        .container {{ max-width: 800px; margin: 0 auto; }}
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
        code {{ background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 6px; }}
        ul {{ list-style: none; }}
        li {{ padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RSI Divergence Alert Bot</h1>
        <p style="opacity:0.8; margin-bottom:30px;">EmperorBTC Methodology | Northflank</p>
        
        <div class="card">
            <span class="status">✅ Running</span>
            <p style="margin-top:15px; opacity:0.8;">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(SYMBOLS)}</div>
                    <div class="stat-label">Symbols</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(subscribers)}</div>
                    <div class="stat-label">Subscribers</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{SCAN_INTERVAL // 60}m</div>
                    <div class="stat-label">Interval</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom:15px;">💬 Commands</h3>
            <ul>
                <li><code>/start</code> - Welcome</li>
                <li><code>/subscribe</code> - Get alerts</li>
                <li><code>/unsubscribe</code> - Stop alerts</li>
                <li><code>/scan</code> - Manual scan</li>
                <li><code>/rules</code> - Trading rules</li>
                <li><code>/ask [q]</code> - Ask Gemini AI</li>
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
    # Northflank sets PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f"Health check server on port {port}")
    server.serve_forever()


# =============================================================================
# TELEGRAM HANDLERS
# =============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *RSI Divergence Alert Bot*\n\n"
        "*Commands:*\n"
        "/subscribe - Get alerts\n"
        "/unsubscribe - Stop alerts\n"
        "/scan - Manual scan\n"
        "/symbols - View symbols\n"
        "/rules - Trading rules\n"
        "/ask - Ask Gemini AI\n\n"
        "⚠️ Not financial advice!",
        parse_mode='Markdown'
    )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in subscribers:
        await update.message.reply_text("✅ Already subscribed!")
        return
    subscribers.add(chat_id)
    await update.message.reply_text(
        f"✅ *Subscribed!*\n\n"
        f"📊 Monitoring {len(SYMBOLS)} symbols\n"
        f"⏰ Scanning every {SCAN_INTERVAL//60} min",
        parse_mode='Markdown'
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.discard(update.effective_chat.id)
    await update.message.reply_text("❌ Unsubscribed")


async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning... (takes ~1 min)")
    try:
        alerts = scanner.scan_all()
        if alerts:
            await update.message.reply_text(f"✅ Found {len(alerts)} signal(s)!")
            for alert in alerts:
                await update.message.reply_text(AlertFormatter.format_alert(alert))
        else:
            await update.message.reply_text("📭 No confirmed signals right now.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def show_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols_list = "\n".join([f"• {s}" for s in SYMBOLS])
    await update.message.reply_text(f"📊 *Symbols:*\n\n{symbols_list}", parse_mode='Markdown')


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
        return
    logger.info(f"Scanning for {len(subscribers)} subscribers")
    try:
        alerts = scanner.scan_all()
        for alert in alerts:
            message = AlertFormatter.format_alert(alert)
            for chat_id in subscribers:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=message)
                except Exception as e:
                    logger.error(f"Send error: {e}")
    except Exception as e:
        logger.error(f"Scan error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global rag
    
    # Start web server for health checks
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logger.info("Health server started")
    
    # Initialize RAG
    try:
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
            rag = TradingKnowledgeRAG()
            logger.info("Gemini RAG ready")
    except Exception as e:
        logger.warning(f"RAG init failed: {e}")
    
    # Telegram bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("symbols", show_symbols))
    app.add_handler(CommandHandler("rules", show_rules))
    app.add_handler(CommandHandler("ask", ask_rag))
    app.add_handler(CallbackQueryHandler(rules_callback, pattern="^rules_"))
    
    app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL, first=60)
    
    logger.info("🤖 Bot starting on Northflank...")
    logger.info(f"📊 Monitoring {len(SYMBOLS)} symbols")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
