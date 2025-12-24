#!/usr/bin/env python3
"""
Verify all files are correctly named and imports work
"""

import os
import sys

print("=" * 60)
print("FILE VERIFICATION CHECK")
print("=" * 60)

# Required files
required_files = [
    'config.py',
    'main.py',
    'divergence_scanner.py',
    'requirements.txt',
    'Dockerfile',
    '.env.example',
    '.gitignore',
    'README.md',
    'TROUBLESHOOTING.md',
    'test_connection.py'
]

print("\n📁 Checking required files...")
all_exist = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✅ {file} ({size:,} bytes)")
    else:
        print(f"  ❌ {file} MISSING!")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing!")
    sys.exit(1)

print("\n📝 Checking imports...")

# Check main.py imports
try:
    with open('main.py', 'r') as f:
        content = f.read()
        if 'from divergence_scanner import' in content:
            print("  ✅ main.py imports from 'divergence_scanner' (correct)")
        elif 'from divergence_scanner_simple import' in content:
            print("  ❌ main.py imports from 'divergence_scanner_simple' (wrong!)")
            print("     File should be renamed to divergence_scanner.py")
            sys.exit(1)
        else:
            print("  ⚠️  main.py import not found or unusual")
except Exception as e:
    print(f"  ❌ Error reading main.py: {e}")
    sys.exit(1)

# Check divergence_scanner.py exists
if not os.path.exists('divergence_scanner.py'):
    print("  ❌ divergence_scanner.py does not exist!")
    sys.exit(1)
else:
    print("  ✅ divergence_scanner.py exists")

# Check for duplicate/old files
old_files = [
    'divergence_scanner_simple.py',
    'divergence_scanner_fixed.py',
    'fallback_coins.py'
]

print("\n🗑️  Checking for old/duplicate files...")
has_old = False
for file in old_files:
    if os.path.exists(file):
        print(f"  ⚠️  {file} exists (can be deleted)")
        has_old = True

if not has_old:
    print("  ✅ No old files found")

# Test import
print("\n🧪 Testing Python imports...")
try:
    sys.path.insert(0, os.getcwd())
    from config import EXCHANGE, SCAN_TIMEFRAMES
    print(f"  ✅ config.py imports work")
    print(f"     EXCHANGE={EXCHANGE}, TIMEFRAMES={SCAN_TIMEFRAMES}")
except Exception as e:
    print(f"  ❌ config.py import failed: {e}")
    sys.exit(1)

try:
    from divergence_scanner import SimpleDivergenceScanner, AlertFormatter
    print(f"  ✅ divergence_scanner.py imports work")
except Exception as e:
    print(f"  ❌ divergence_scanner.py import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅✅✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\n🚀 Your bot is ready to run:")
print("   python main.py")
print("\n📝 Make sure to:")
print("   1. Create .env file with TELEGRAM_TOKEN")
print("   2. Install requirements: pip install -r requirements.txt")
print("=" * 60)
