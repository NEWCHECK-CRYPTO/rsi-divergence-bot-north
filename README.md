# RSI Divergence Trading Bot V10 - EXACT 2-CANDLE MODE

🤖 Telegram bot that scans top 100 cryptocurrencies for RSI divergences and sends **FRESH alerts only** - exactly when the 2nd confirmation candle closes!

## ✨ Key Features

- **⚡ EXACT 2-CANDLE TIMING**: Alerts ONLY when 2nd confirmation candle JUST closed (no old/late signals)
- **📊 Top 100 Coins**: Automatically scans highest volume USDT pairs on Bybit
- **🔍 Multi-Timeframe**: Scans 1h, 4h, and 1d charts
- **✅ Multi-Layer Filtering**: 
  - 2-candle confirmation (both must confirm)
  - Multi-timeframe trend alignment
  - ADX momentum verification
  - Pattern invalidation checks
- **🎯 High-Quality Signals**: 70%+ confidence minimum
- **🔔 Real-Time Alerts**: Get notified immediately via Telegram

## 🚨 How It Works

### Signal Timeline (Example: Daily Chart)

```
Day 1: Swing 2 forms (divergence detected) ❌ NO ALERT
Day 2: Confirmation Candle 1 closes       ❌ NO ALERT  
Day 3: Confirmation Candle 2 closes       ✅ ALERT SENT!
Day 4: Too late                           ❌ NO ALERT
Day 5: Too late                           ❌ NO ALERT
```

**You only get alerts when signals are FRESH - the moment the 2nd candle closes!**

## 📋 Setup Instructions

### 1. Get a Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy your bot token

### 2. Install Locally (Development)

```bash
# Clone/download this folder
cd rsi_divergence_bot_v10

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env and add your Telegram token
# TELEGRAM_TOKEN=your_token_here

# Run the bot
python main.py
```

### 3. Deploy to Cloud (Railway/Render)

#### Option A: Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select this repository
4. Add environment variable:
   - `TELEGRAM_TOKEN` = your bot token
5. Deploy!

#### Option B: Render

1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Environment Variables**: Add `TELEGRAM_TOKEN`
5. Deploy!

## 🎮 Bot Commands

- `/start` - Show bot info and commands
- `/subscribe` - Start receiving alerts
- `/unsubscribe` - Stop receiving alerts
- `/scan` - Manual scan (3-5 minutes)
- `/coins` - List top 100 coins being tracked
- `/debug` - Check system status (troubleshooting)

## 📊 Alert Example

```
🟢 STRONG SIGNAL - BULLISH 📈

📊 BTC/USDT (#1)
⏰ 4H | 2025-12-24 20:00:00 IST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Bullish Regular

Swing 1: $95,234.50 (RSI: 32.4)
Swing 2: $94,156.20 (RSI: 38.7)
Now: $96,450.30 (RSI: 42.1)

🔍 18 candles apart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 2-Candle Confirmed! RSI:2/2 Price:2/2
⚡ JUST CONFIRMED (2nd candle closed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Lower TF Trend (1H):
✅ Trend: Strong Up 📈 (ADX: 28.3)
✅ Price: Rising ↗️
✅ RSI: Rising ↗️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ RSI (4h): Rising ↗️
✅ Price (4h): Rising ↗️
✅ Trend: Strong 💪 (ADX: 26.8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 LONG | Entry: $96,450.30
🛑 SL: $93,214.64 | 🎯 TP: $100,308.31

🔥 Confidence: 87%
📺 [TradingView Chart Link]

⚠️ DYOR | 🇱🇰 2025-12-24 20:15:23 IST
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
EXCHANGE = "bybit"           # Exchange to use
TOP_COINS_COUNT = 100        # Number of top coins to scan
SCAN_TIMEFRAMES = ["1h", "4h", "1d"]  # Timeframes
SCAN_INTERVAL = 120          # Scan every 2 minutes
MIN_CONFIDENCE = 0.70        # Minimum 70% confidence
ALERT_COOLDOWN = 1800        # 30min cooldown per symbol
```

## 🔍 What is RSI Divergence?

**RSI Divergence** happens when:
- **Bullish**: Price makes lower lows BUT RSI makes higher lows → Potential reversal UP 📈
- **Bearish**: Price makes higher highs BUT RSI makes lower highs → Potential reversal DOWN 📉

This bot finds these patterns and waits for **2 candles to confirm** before alerting you!

## 🎯 Signal Quality

The bot uses multiple filters to ensure high-quality signals:

1. **Swing Detection**: Finds proper swing highs/lows
2. **Pattern Validation**: Checks middle price action doesn't invalidate
3. **2-Candle Confirmation**: BOTH candles must confirm (2/2)
4. **Timing Check**: EXACTLY 2 candles after Swing 2 (not before, not after)
5. **MTF Trend**: Lower timeframe must align
6. **ADX Momentum**: Trend strength must be adequate
7. **Confidence Score**: Minimum 70% required

**Result**: You get fewer signals, but they're much more reliable!

## 📈 Expected Signal Frequency

- **Strong signals (🟢)**: 1-3 per day
- **Medium signals (🟡)**: 3-7 per day  
- **Early signals (🔴)**: 5-10 per day

Most signals are filtered out - only the best pass!

## ⚠️ Important Notes

- **Not Financial Advice**: Always do your own research (DYOR)
- **Use Stop Losses**: The bot provides suggested SL levels
- **Paper Trade First**: Test the signals before using real money
- **Market Conditions**: Works best in trending markets
- **Fresh Signals Only**: No old/late alerts - only when 2nd candle JUST closes!

## 🐛 Troubleshooting

### No coins loading (shows 0 coins)
```bash
# Use /debug command in Telegram
# OR check logs for errors
# Common issues:
# - Network/firewall blocking exchange API
# - Exchange API down
# - Rate limits reached
```

### Not receiving alerts
```bash
# Make sure you're subscribed: /subscribe
# Check if bot is scanning: /debug
# Verify signals exist: /scan
```

### Bot crashes/restarts
```bash
# Check your cloud platform logs
# Verify TELEGRAM_TOKEN is set correctly
# Ensure requirements.txt is installed
```

## 📝 Files

- `main.py` - Telegram bot interface
- `divergence_scanner.py` - Core scanning logic with EXACT 2-candle timing
- `config.py` - Configuration settings
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker container config
- `.env.example` - Environment variables template

## 🔄 Updates

**V10 Features:**
- ⚡ **EXACT 2-CANDLE MODE**: Alerts only when 2nd candle JUST closed
- 🚫 No old/late signals
- ✅ Fresh, actionable signals only
- 📊 Top 100 coins by volume
- 🔍 Multi-layer filtering
- 💪 High confidence threshold (70%+)

## 📞 Support

Issues? Questions?
1. Use `/debug` command in the bot
2. Check this README
3. Review logs on your hosting platform

## ⚖️ License

For educational purposes only. Trade at your own risk.

---

**Made with ❤️ for crypto traders who want FRESH, quality signals - not spam!**

🇱🇰 Timezone: Asia/Colombo (IST)
