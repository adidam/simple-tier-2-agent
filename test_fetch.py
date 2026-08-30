#!/usr/bin/env python
"""Quick test script to verify get_stock_info and get_price_history functions."""

from tools import get_stock_info, get_price_history

print("=" * 60)
print("Testing get_stock_info()")
print("=" * 60)
symbols = ['AAPL', 'GOOGL', 'TCS', 'TCS.NS', 'INFY']
for sym in symbols:
    data = get_stock_info(sym)
    print(f'{sym}:')
    print(f'  Price: {data["price"]} {data["currency"]}')
    print(f'  Ratios (count): {len(data["key_ratios"])}')
    if data["key_ratios"]:
        print(f'  Sample: PE={data["key_ratios"].get("trailingPE")}, Market Cap={data["key_ratios"].get("marketCap")}')
    print(f'  Error: {data["error"]}')
    print()

print("=" * 60)
print("Testing get_price_history()")
print("=" * 60)
test_symbols = [('AAPL', '6mo'), ('GOOGL', '1y'), ('TCS', '3mo'), ('INFY', '6mo')]
for sym, period in test_symbols:
    hist = get_price_history(sym, period)
    print(f'{sym} ({period}):')
    print(f'  Start: ${hist["start_price"]:.2f} → End: ${hist["end_price"]:.2f}')
    print(f'  Change: {hist["percent_change"]}%')
    print(f'  High: ${hist["high"]:.2f}, Low: ${hist["low"]:.2f}')
    print(f'  Error: {hist["error"]}')
    print()
