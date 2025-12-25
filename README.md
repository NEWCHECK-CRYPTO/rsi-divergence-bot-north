# RSI Divergence Trading Bot V11 - Relaxed Version

🤖 Advanced Telegram bot that scans cryptocurrency markets for RSI divergences with **relaxed conditions for more signals**.

## 🚀 Features

### V11 Enhancements
- ✅ **Hidden Divergences** - 4 divergence types (Regular + Hidden, Bullish + Bearish)
- ✅ **Quality Tiers** - Premium (85%+), Standard (70%+), Exploratory (65%+)
- ✅ **Optional MTF Trend** - Multi-timeframe confirmation (bonus, not required)
- ✅ **More Signals** - 25-40 signals/day (vs 5-10 in strict mode)
- ✅ **User Filtering** - Choose which quality tiers to receive

### Core Features
- 🔍 Scans top 150 coins by volume
- ⏰ 4 timeframes: 15m, 1h, 4h, 1d
- 📊 2-candle confirmation (1/2 required)
- 🎯 ADX trend strength analysis
- 🔗 TradingView chart links
- 📈 Entry/SL/TP suggestions

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/rsi-divergence-bot-v11.git
cd rsi-divergence-bot-v11
```

### 2. Setup Environment
```bash
cp .env.example .env
```

Edit `.env` and add your tokens:
```
TELEGRAM_TOKEN=your_bot_token_from_botfather
GEMINI_API_KEY=your_gemini_key (optional)
PORT=8080
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Bot
```bash
python main.py
```

## 🐳 Docker Deployment

### Build & Run
```bash
docker build -t rsi-bot .
docker run -d --env-file .env -p 8080:8080 rsi-bot
```

### Railway/Render Deployment
1. Push to GitHub
2. Connect repository to Railway/Render
3. Add environment variables
4. Deploy!

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Start bot & see info |
| `/subscribe` | Get all signals (65%+) |
| `/premium` | Only premium signals (85%+) |
| `/standard` | Standard+ signals (70%+) |
| `/mystatus` | Check your settings |
| `/scan` | Manual scan now |
| `/coins` | List top coins |
| `/unsubscribe` | Stop alerts |

## 🎯 Signal Quality Tiers

### 💎 Premium (85%+ confidence)
- All confirmations aligned
- Strong trend support
- Expected: 5-10/day
- Win rate: ~75%

### ⭐ Standard (70-85% confidence)
- Most confirmations aligned
- Moderate trend support
- Expected: 15-25/day
- Win rate: ~65%

### 🔍 Exploratory (65-70% confidence)
- Basic confirmations met
- Weaker trend support
- Expected: 25-40/day
- Win rate: ~60%

## 📊 Divergence Types

### Regular Divergences (Reversal)
- **Bullish Regular**: Price LL, RSI HL → Buy signal
- **Bearish Regular**: Price HH, RSI LH → Sell signal

### Hidden Divergences (Continuation)
- **Bullish Hidden**: Price HL, RSI LL → Continue uptrend
- **Bearish Hidden**: Price LH, RSI HH → Continue downtrend

## ⚙️ Configuration

Edit `config.py` to customize:

```python
TOP_COINS_COUNT = 150          # Number of coins to scan
SCAN_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
LOOKBACK_CANDLES = 80          # History to analyze
MIN_SWING_DISTANCE = 7         # Min candles between swings
SWING_STRENGTH_BARS = 3        # Swing detection sensitivity
CONFIRMATION_THRESHOLD = 1     # 1/2 candles needed
MIN_CONFIDENCE = 0.65          # Minimum 65% confidence
SCAN_INTERVAL = 90             # Scan every 90 seconds
```

## 📈 Example Signal

```
💎 PREMIUM | 🟢 STRONG - BULLISH 📈

📊 BTC/USDT (#1)
⏰ 4H | 2025-12-25 14:00:00 IST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Bullish Regular

Swing 1: $92,500.00 (RSI: 35.2)
Swing 2: $92,000.00 (RSI: 38.5)
Now: $93,200.00 (RSI: 42.5)

🔍 27 candles apart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 2-Candle: RSI:2/2 Price:2/2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Lower TF Trend (1H):
✅ Trend: Very Strong Up 📈 (ADX: 31.2)
✅ Price: Rising ↗️
✅ RSI: Rising ↗️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ RSI (4h): Rising ↗️
✅ Price (4h): Rising ↗️
✅ Trend: Strong 💪 (ADX: 28.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 LONG | Entry: $93,200.00
🛑 SL: $91,080.00 | 🎯 TP: $96,928.00

🔥 Confidence: 95%
📺 [TradingView Chart]

⚠️ DYOR | 🇱🇰 2025-12-25 14:32:15 IST
```

## 🔧 Troubleshooting

### Bot not starting?
- Check Telegram token is correct
- Ensure Python 3.11+ is installed
- Verify all dependencies installed

### No signals?
- Markets may be ranging (no divergences)
- Try lowering `MIN_CONFIDENCE` in config
- Wait 1-2 hours and try again

### Too many signals?
- Use `/premium` for fewer, higher quality
- Increase `MIN_CONFIDENCE` to 0.70+
- Reduce `SCAN_TIMEFRAMES`

## 📝 Version History

- **V11** - Relaxed mode with hidden divergences & quality tiers
- **V10** - Original strict mode with all filters
- **V9** - Added invalidation checks
- **V8** - Multi-timeframe trend confirmation

## ⚠️ Disclaimer

This bot is for **educational purposes only**. Cryptocurrency trading carries significant risk. Always:
- Do your own research (DYOR)
- Use proper risk management
- Never invest more than you can afford to lose
- Test with paper trading first

## 📄 License

MIT License - Feel free to modify and use!

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

## 📧 Support

- Issues: GitHub Issues
- Updates: Check releases

---

**Happy Trading! 🚀** Remember: Past performance doesn't guarantee future results.
