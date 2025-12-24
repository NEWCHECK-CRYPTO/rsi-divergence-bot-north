# 🚀 SETUP GUIDE - Step by Step

## ✅ What's Included:

```
rsi_bot_clean/
├── config.py                    # Settings (exchange, timeframes, etc.)
├── main.py                      # Telegram bot
├── divergence_scanner.py        # Core scanning logic (SIMPLE version)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # For cloud deployment
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── README.md                    # Full documentation
├── TROUBLESHOOTING.md           # Troubleshooting guide
├── test_connection.py           # Test exchange connection
└── verify_files.py              # Verify all files are correct
```

---

## 📋 Step-by-Step Setup

### Step 1: Extract Files
```bash
unzip rsi_bot_CLEAN.zip
cd rsi_bot_clean
```

### Step 2: Verify Files
```bash
python verify_files.py
```

You should see:
```
✅✅✅ ALL CHECKS PASSED!
```

If you see any errors, STOP and check which files are missing.

### Step 3: Test Exchange Connection (Optional but Recommended)
```bash
python test_connection.py
```

Expected output:
```
✅✅✅ BYBIT WORKS! Use: EXCHANGE='bybit'
```

OR

```
✅✅✅ BINANCE WORKS! Use: EXCHANGE='binance'
```

If both fail, see TROUBLESHOOTING.md

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Create .env File
```bash
cp .env.example .env
```

Edit .env:
```bash
nano .env  # or use any text editor
```

Add your Telegram bot token:
```
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

Save and exit.

### Step 6: Run the Bot
```bash
python main.py
```

You should see:
```
[2025-12-24 22:45:00 IST] 🔧 Initializing SIMPLE scanner...
[2025-12-24 22:45:00 IST] ✅ Scanner ready (BYBIT)
[2025-12-24 22:45:00 IST] 📊 Loaded 100 coins
[2025-12-24 22:45:00 IST] 🏆 Top 10: ['BTC/USDT', 'ETH/USDT', ...]
[2025-12-24 22:45:01 IST] 🤖 Bot V10 (BYBIT) starting...
```

**100 coins loaded** = ✅ SUCCESS!

### Step 7: Start Using in Telegram
1. Open Telegram
2. Search for your bot
3. Send `/start`
4. Send `/subscribe`

Done! 🎉

---

## 🔧 Configuration

### Change Exchange (if Bybit doesn't work)

Edit `config.py`:
```python
EXCHANGE = "binance"  # Change from "bybit" to "binance"
```

### Adjust Scan Settings

Edit `config.py`:
```python
SCAN_TIMEFRAMES = ["1h", "4h", "1d"]  # Add/remove timeframes
SCAN_INTERVAL = 120  # Scan every 2 minutes (increase if needed)
MIN_CONFIDENCE = 0.70  # Minimum 70% confidence
```

---

## ☁️ Cloud Deployment

### Railway
1. Create new project
2. Connect GitHub repo
3. Add environment variable: `TELEGRAM_TOKEN=your_token`
4. Deploy!

### Render
1. New Web Service
2. Connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `python main.py`
5. Environment variables: `TELEGRAM_TOKEN=your_token`

---

## ❓ Troubleshooting

### Problem: "0 coins loaded"
**Solution:** The bot uses a hardcoded list of 100 top coins. This message should NOT appear with this version. If it does:
```bash
python verify_files.py
```

Check if `divergence_scanner.py` exists and has `SIMPLE_COINS` list.

### Problem: Import errors
```
ModuleNotFoundError: No module named 'divergence_scanner'
```

**Solution:**
```bash
python verify_files.py
```

Make sure the file is named `divergence_scanner.py` (not `divergence_scanner_simple.py`).

### Problem: Exchange connection fails
```bash
python test_connection.py
```

Follow recommendations from the output.

---

## 📁 File Naming - IMPORTANT!

**Correct names:**
- ✅ `divergence_scanner.py` (main scanner file)
- ✅ `main.py` (imports from divergence_scanner)
- ✅ `config.py`

**Wrong names (will cause errors):**
- ❌ `divergence_scanner_simple.py`
- ❌ `divergence_scanner_fixed.py`
- ❌ `scanner.py`

If you see wrong names, rename:
```bash
mv divergence_scanner_simple.py divergence_scanner.py
```

---

## ✅ Verification Checklist

Before running:
- [ ] All 10 required files present
- [ ] `verify_files.py` passes all checks
- [ ] `.env` file created with TELEGRAM_TOKEN
- [ ] `requirements.txt` installed
- [ ] `test_connection.py` shows exchange works

After running:
- [ ] Bot shows "100 coins loaded"
- [ ] `/start` command works in Telegram
- [ ] `/subscribe` command works
- [ ] `/scan` command runs without errors

---

## 🎯 Quick Start Commands

```bash
# 1. Verify everything
python verify_files.py

# 2. Test connection
python test_connection.py

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env
cp .env.example .env
# Edit .env with your token

# 5. Run bot
python main.py
```

---

## 💡 Success Indicators

**You're good to go if you see:**
```
✅ Scanner ready (BYBIT)
✅ Loaded 100 coins
✅ Bot V10 (BYBIT) starting...
```

**Something's wrong if you see:**
```
❌ 0 coins loaded
❌ ModuleNotFoundError
❌ File not found
```

Use `verify_files.py` to diagnose!

---

## 📞 Need Help?

1. Run `python verify_files.py`
2. Run `python test_connection.py`
3. Check `TROUBLESHOOTING.md`
4. All files are correctly named and imported

---

**Made simple, made to work!** 🚀
