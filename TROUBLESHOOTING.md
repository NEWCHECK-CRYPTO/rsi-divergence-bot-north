# 🔧 TROUBLESHOOTING GUIDE

## Problem: "0 coins loaded" Error

### Quick Diagnosis

Run this in your terminal where the bot is installed:

```bash
python test_connection.py
```

This will tell you exactly what's wrong.

---

## Common Issues & Solutions

### ❌ Issue 1: Network/Firewall Blocking

**Symptoms:**
```
❌ ERROR: Could not fetch any coins!
[2025-12-24] WARNING: No coins loaded!
```

**Causes:**
- Your network/ISP blocks cryptocurrency exchange APIs
- Cloud platform (Railway/Render/Heroku) blocks crypto APIs
- Firewall blocking outbound connections
- Country-level restrictions on crypto exchanges

**Solutions:**

1. **Switch to Binance** (often more accessible):
   ```python
   # In config.py, change:
   EXCHANGE = "binance"  # Instead of "bybit"
   ```

2. **Use Fallback Mode** (Already built-in!):
   - The bot automatically uses hardcoded top 100 coins if API fails
   - You'll see: `🔄 Using FALLBACK coin list...`
   - This lets the bot work even without API access for coin list

3. **Deploy to Different Platform:**
   - **Works**: Vercel, Netlify, your own VPS
   - **Might Block**: Railway, some Render regions
   - **Usually Works**: DigitalOcean, AWS, Google Cloud

4. **Use VPN:**
   - Install VPN on your server/computer
   - Connect to unrestricted country (US, UK, Singapore)
   - Restart bot

---

### ❌ Issue 2: API Rate Limiting

**Symptoms:**
```
429 Too Many Requests
Rate limit exceeded
```

**Solutions:**
- Bot already has 0.05s delays - this is automatic
- Increase `SCAN_INTERVAL` in config.py to 300 (5 minutes)
- Reduce `TOP_COINS_COUNT` to 50

---

### ❌ Issue 3: Exchange API Down

**Symptoms:**
```
Timeout after 30000ms
Connection refused
```

**Solutions:**
1. Check exchange status:
   - Bybit: https://bybit.com
   - Binance: https://binance.com

2. Wait 10-30 minutes and retry

3. Switch exchanges in config.py

---

### ❌ Issue 4: Invalid API Credentials (If using authenticated endpoints)

**Note:** This bot doesn't need API keys for basic scanning, but if you added them:

**Symptoms:**
```
Invalid API key
Authentication failed
```

**Solution:**
- Remove API keys from .env file
- Bot works fine without authentication for public data

---

## ✅ Working Configurations

### Configuration A: With API Access (Best)
```python
# config.py
EXCHANGE = "bybit"  # or "binance"
TOP_COINS_COUNT = 100
```
- Full functionality
- Real-time volume ranking
- Most accurate signals

### Configuration B: Fallback Mode (When API Blocked)
```python
# config.py  
EXCHANGE = "bybit"
TOP_COINS_COUNT = 100
```
- Uses hardcoded top 100 coins
- Still scans charts (this works even when ticker API fails)
- Slightly less accurate ranking
- **Still 100% functional for alerts!**

---

## 🔍 How to Check What's Wrong

### Step 1: Test Connection
```bash
cd rsi_divergence_bot_v10
python test_connection.py
```

You'll see one of these:

✅ **Success:**
```
✅ BYBIT CONNECTION SUCCESSFUL!
Your bot should work with Bybit.
```

⚠️ **Fallback:**
```
❌ Bybit failed
✅ BINANCE CONNECTION SUCCESSFUL!
RECOMMENDATION: Change EXCHANGE = 'binance' in config.py
```

❌ **Both Failed:**
```
❌ BOTH EXCHANGES FAILED
Possible causes:
1. No internet connection
2. Firewall blocking
3. Country restrictions
```

### Step 2: Check Bot Logs

Look for these messages:

✅ **Good:**
```
[2025-12-24] ✅ Markets loaded: 450 pairs
[2025-12-24] ✅ Selected top 100 coins by volume
```

⚠️ **Fallback (Still OK):**
```
[2025-12-24] ❌ fetch_tickers() failed
[2025-12-24] 🔄 Using FALLBACK coin list...
[2025-12-24] ✅ Using 100 coins from fallback list
```

❌ **Bad:**
```
[2025-12-24] ❌ All methods failed!
[2025-12-24] WARNING: No coins loaded!
```

---

## 💡 Understanding Fallback Mode

When the bot can't fetch volume data from the exchange, it uses a hardcoded list of top 100 coins (BTC, ETH, SOL, etc.). 

**What still works:**
- ✅ Chart data fetching (OHLCV)
- ✅ RSI calculations
- ✅ Divergence detection
- ✅ All filters and confirmations
- ✅ Telegram alerts

**What changes:**
- ⚠️ Volume ranking is fixed (not real-time)
- ⚠️ Can't detect newly trending coins

**Bottom line:** You still get 100% functional divergence alerts!

---

## 🌐 Network Requirements

The bot needs to access:
- `api.bybit.com` (for Bybit)
- `api.binance.com` (for Binance)  
- `api.telegram.org` (for Telegram bot)

If ANY of these are blocked, the bot won't work properly.

**Test manually:**
```bash
curl https://api.bybit.com/v5/market/tickers?category=linear
```

If you get data → API accessible  
If you get error → API blocked

---

## 🚀 Recommended Setup

1. **Local development**: Bybit or Binance (both work)
2. **Cloud deployment**: 
   - Try Binance first (more widely accessible)
   - Use VPS (DigitalOcean, AWS) for guaranteed access
   - Avoid shared hosting platforms that block crypto APIs

3. **Best practices**:
   ```python
   EXCHANGE = "binance"  # More reliable globally
   TOP_COINS_COUNT = 100
   SCAN_INTERVAL = 180  # 3 minutes (easier on API)
   ```

---

## 📞 Still Not Working?

1. Run `/debug` in Telegram bot
2. Check the output - it will tell you exactly what's failing
3. Copy the error message
4. Check against this guide

**Emergency fallback**: The bot will work with fallback coin list even if API is completely blocked!
