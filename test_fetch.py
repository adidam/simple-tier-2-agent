#!/usr/bin/env python
"""Quick test script to verify fetch_stock_data works."""

from tools import fetch_stock_data

symbols = ['AAPL', 'GOOGL', 'TCS', 'TCS.NS', 'INFY']
for sym in symbols:
    data = fetch_stock_data(sym)
    print(f'{sym}:')
    print(f'  Price: {data["price"]} {data["currency"]}')
    print(f'  Ratios (count): {len(data["key_ratios"])}')
    if data["key_ratios"]:
        print(f'  Sample: PE={data["key_ratios"].get("trailingPE")}, Market Cap={data["key_ratios"].get("marketCap")}')
    print(f'  Error: {data["error"]}')
    print()
