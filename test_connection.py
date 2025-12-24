"""
Test Exchange Connection
Run this to diagnose API connection issues
"""

print("\n" + "="*60)
print("EXCHANGE CONNECTION DIAGNOSTIC TEST")
print("="*60 + "\n")

try:
    import ccxt
    print("✅ ccxt library installed")
except ImportError:
    print("❌ ccxt not installed!")
    print("Run: pip install ccxt")
    exit(1)

# Test Bybit
print("\n📡 Testing BYBIT connection...")
print("-" * 60)

try:
    bybit = ccxt.bybit({'enableRateLimit': True, 'timeout': 30000})
    print("  ✅ Exchange object created")
    
    bybit.load_markets()
    print(f"  ✅ Connected! {len(bybit.markets)} markets loaded")
    
    usdt_pairs = [s for s in bybit.markets.keys() if '/USDT' in s]
    print(f"  ✅ Found {len(usdt_pairs)} USDT pairs")
    
    if len(usdt_pairs) > 0:
        test_symbol = usdt_pairs[0]
        ticker = bybit.fetch_ticker(test_symbol)
        print(f"  ✅ Ticker test successful ({test_symbol}: ${ticker['last']})")
        print("\n" + "="*60)
        print("✅✅✅ BYBIT WORKS! Use: EXCHANGE='bybit'")
        print("="*60)
        exit(0)
        
except Exception as e:
    print(f"  ❌ Bybit failed: {e}")

# Test Binance
print("\n📡 Testing BINANCE connection...")
print("-" * 60)

try:
    binance = ccxt.binance({
        'enableRateLimit': True,
        'timeout': 30000,
        'options': {'defaultType': 'future'}
    })
    print("  ✅ Exchange object created")
    
    binance.load_markets()
    print(f"  ✅ Connected! {len(binance.markets)} markets loaded")
    
    usdt_pairs = [s for s in binance.markets.keys() if '/USDT' in s]
    print(f"  ✅ Found {len(usdt_pairs)} USDT pairs")
    
    if len(usdt_pairs) > 0:
        test_symbol = usdt_pairs[0]
        ticker = binance.fetch_ticker(test_symbol)
        print(f"  ✅ Ticker test successful ({test_symbol}: ${ticker['last']})")
        print("\n" + "="*60)
        print("✅✅✅ BINANCE WORKS! Use: EXCHANGE='binance'")
        print("="*60)
        print("RECOMMENDED: Change EXCHANGE='binance' in config.py")
        exit(0)
        
except Exception as e:
    print(f"  ❌ Binance failed: {e}")

# Both failed
print("\n" + "="*60)
print("❌❌❌ BOTH EXCHANGES FAILED")
print("="*60)
print("\n🔍 DIAGNOSIS:")
print("  • Internet connection: Blocked or no access")
print("  • Firewall: May be blocking crypto exchange APIs")
print("  • Location: Country/ISP may block crypto exchanges")
print("  • Cloud platform: May block these domains")

print("\n💡 SOLUTIONS:")
print("  1. Try different network/WiFi")
print("  2. Use VPN")
print("  3. Deploy to different cloud (DigitalOcean, AWS)")
print("  4. Use FALLBACK MODE (bot will still work!)")

print("\n📝 FALLBACK MODE:")
print("  The bot has 100 hardcoded top coins.")
print("  Even if API fails, you'll still get divergence alerts!")
print("  Just run the bot - it will use fallback automatically.")

print("\n" + "="*60)
