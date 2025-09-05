#!/usr/bin/env python3
"""
API Client Module for DEX and CEX Data Fetching
Handles all external API calls for price and volume data
"""

import requests
import json
import ccxt
import pandas as pd
from datetime import datetime, timezone
from auth import BITQUERY_TOKEN

def get_dex_prices():
    """Fetch DEX prices from Bitquery"""
    print("Fetching DEX prices from Bitquery...")
    
    url = "https://streaming.bitquery.io/graphql"
    payload = json.dumps({
        "query": "{\n  Trading {\n    Currencies(\n      where: {\n        Currency: { Id: { in:[ \"bid:solana\",\"bid:eth\",\"bid:bitcoin\"]} },\n        Interval: { Time: { Duration: { eq: 60 } } }\n      },\n      limit: { count: 5000 },\n      orderBy: { descending: Interval_Time_End }\n    ) {\n      Currency {\n        Id\n        Name\n        Symbol\n      }\n      Block {\n        Date\n        Time\n        Timestamp\n      }\n      Interval {\n        Time {\n          Start\n          Duration\n          End\n        }\n      }\n      Volume {\n        Base\n        BaseAttributedToUsd\n        Quote\n        Usd\n      }\n      Price {\n        IsQuotedInUsd\n        Ohlc {\n          Open\n          High\n          Low\n          Close\n        }\n        Average {\n          Estimate\n          ExponentialMoving\n          Mean\n          SimpleMoving\n          WeightedSimpleMoving\n        }\n      }\n    }\n  }\n}\n",
        "variables": "{}"
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BITQUERY_TOKEN}'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    
    # Extract price data
    dex_prices = []
    if 'data' in data and 'Trading' in data['data'] and 'Currencies' in data['data']['Trading']:
        for item in data['data']['Trading']['Currencies']:
            try:
                timestamp = datetime.fromisoformat(item['Block']['Time'].replace('Z', '+00:00'))
                close_price = float(item['Price']['Ohlc']['Close'])
                volume = float(item['Volume']['Usd']) if item['Volume']['Usd'] else 0
                currency_id = item['Currency']['Id']
                currency_symbol = item['Currency']['Symbol']
                
                # Map currency IDs to symbols
                if currency_id == 'bid:bitcoin':
                    symbol = 'BTC'
                elif currency_id == 'bid:eth':
                    symbol = 'ETH'
                elif currency_id == 'bid:solana':
                    symbol = 'SOL'  # Native SOL token on Solana
                else:
                    symbol = currency_symbol
                
                dex_prices.append({
                    'timestamp': timestamp,
                    'close': close_price,
                    'volume': volume,
                    'currency': symbol,
                    'currency_id': currency_id,
                    'source': 'DEX'
                })
            except (KeyError, ValueError, TypeError) as e:
                continue
    
    return sorted(dex_prices, key=lambda x: x['timestamp'])

def get_cex_prices():
    """Fetch CEX prices from multiple exchanges"""
    print("Fetching CEX prices from multiple exchanges...")
    
    cex_prices = []
    exchanges = [
        ('binance', ccxt.binance()),
        ('coinbase', ccxt.coinbase())
    ]
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for exchange_name, exchange in exchanges:
        print(f"   Fetching from {exchange_name.upper()}...")
        try:
            exchange.load_markets()
            
            for symbol in symbols:
                print(f"     Fetching {symbol} data...")
                try:
                    # Get OHLCV data for each symbol
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=2000)
                    
                    for candle in ohlcv:
                        # Convert from UTC timestamp to UTC datetime
                        timestamp = datetime.fromtimestamp(candle[0] / 1000, timezone.utc)
                        close_price = float(candle[4])
                        volume = float(candle[5])
                        currency = symbol.split('/')[0]  # Extract BTC or ETH
                        
                        cex_prices.append({
                            'timestamp': timestamp,
                            'close': close_price,
                            'volume': volume,
                            'currency': currency,
                            'exchange': exchange_name,
                            'source': 'CEX'
                        })
                except Exception as e:
                    print(f"     Error fetching {symbol} from {exchange_name}: {e}")
                    continue
        except Exception as e:
            print(f"   Error initializing {exchange_name}: {e}")
            continue
    
    return sorted(cex_prices, key=lambda x: x['timestamp'])

def get_cex_volume_analysis(symbol='SOL/USDT', exchange_name='binance', limit=10000):
    """Fetch detailed volume analysis for a specific symbol and exchange"""
    print(f"Fetching volume analysis for {symbol} from {exchange_name}...")
    
    try:
        if exchange_name.lower() == 'binance':
            exchange = ccxt.binance()
        elif exchange_name.lower() == 'coinbase':
            exchange = ccxt.coinbase()
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
        
    except Exception as e:
        print(f"Error fetching volume data: {e}")
        return None
