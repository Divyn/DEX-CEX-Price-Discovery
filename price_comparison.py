#!/usr/bin/env python3
"""
Comprehensive DEX vs CEX Price Comparison Analysis
Compares Bitcoin prices from Bitquery (DEX) and Binance (CEX) to analyze:
- Price deviations and correlations
- Lead-lag relationships
- Mathematical statistics
"""

import requests
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import secrets
from auth import BITQUERY_TOKEN

def get_dex_prices():
    """Fetch DEX prices from Bitquery"""
    print("Fetching DEX prices from Bitquery...")
    
    url = "https://streaming.bitquery.io/graphql"
    payload = json.dumps({
        "query": "{\n  Trading {\n    Currencies(\n      where: {Currency: {Id: {in: [\"bid:bitcoin\", \"bid:eth\"]}}, Interval: {Time: {Duration: {eq: 60}}}}\n      limit: {count: 2000}\n      orderBy: {descending: Interval_Time_End}\n    ) {\n      Currency {\n        Id\n        Name\n        Symbol\n      }\n      Block {\n        Date\n        Time\n        Timestamp\n      }\n      Interval {\n        Time {\n          Start\n          Duration\n          End\n        }\n      }\n      Volume {\n        Base\n        BaseAttributedToUsd\n        Quote\n        Usd\n      }\n      Price {\n        IsQuotedInUsd\n        Ohlc {\n          Open\n          High\n          Low\n          Close\n        }\n        Average {\n          Estimate\n          ExponentialMoving\n          Mean\n          SimpleMoving\n          WeightedSimpleMoving\n        }\n      }\n    }\n  }\n}\n",
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
                # Keep as UTC timezone-aware datetime
                close_price = float(item['Price']['Ohlc']['Close'])
                volume = float(item['Volume']['Usd']) if item['Volume']['Usd'] else 0
                currency_id = item['Currency']['Id']
                currency_symbol = item['Currency']['Symbol']
                
                # Map currency IDs to symbols
                if currency_id == 'bid:bitcoin':
                    symbol = 'BTC'
                elif currency_id == 'bid:eth':
                    symbol = 'ETH'
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
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    for exchange_name, exchange in exchanges:
        print(f"   Fetching from {exchange_name.upper()}...")
        try:
            exchange.load_markets()
            
            for symbol in symbols:
                print(f"     Fetching {symbol} data...")
                try:
                    # Get OHLCV data for each symbol
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=20000)
                    
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

def align_timestamps(dex_data, cex_data, tolerance_minutes=30):
    """Align DEX and CEX data by timestamp with tolerance, grouped by currency and exchange"""
    print("Aligning timestamps...")
    
    # Group data by currency
    dex_by_currency = {}
    cex_by_currency = {}
    
    for item in dex_data:
        currency = item['currency']
        if currency not in dex_by_currency:
            dex_by_currency[currency] = []
        dex_by_currency[currency].append(item)
    
    for item in cex_data:
        currency = item['currency']
        if currency not in cex_by_currency:
            cex_by_currency[currency] = {}
        exchange = item.get('exchange', 'unknown')
        if exchange not in cex_by_currency[currency]:
            cex_by_currency[currency][exchange] = []
        cex_by_currency[currency][exchange].append(item)
    
    all_aligned_data = {}
    total_matched = 0
    total_dex_unmatched = 0
    total_cex_unmatched = 0
    
    # Process each currency and exchange combination
    for currency in set(list(dex_by_currency.keys()) + list(cex_by_currency.keys())):
        print(f"\n   Processing {currency}...")
        
        dex_currency_data = dex_by_currency.get(currency, [])
        cex_currency_exchanges = cex_by_currency.get(currency, {})
        
        if not dex_currency_data or not cex_currency_exchanges:
            print(f"   No data for {currency} in one or both sources")
            continue
        
        # Process each exchange separately
        for exchange_name, cex_currency_data in cex_currency_exchanges.items():
            print(f"     Processing {currency} from {exchange_name.upper()}...")
            
            if not dex_currency_data or not cex_currency_data:
                print(f"     No data for {currency} from {exchange_name}")
                continue
        
        aligned_data = []
        dex_idx = 0
        cex_idx = 0
        dex_matched = 0
        cex_matched = 0
        dex_unmatched = 0
        cex_unmatched = 0
        
        # Get time ranges for context
        dex_start = dex_currency_data[0]['timestamp'] if dex_currency_data else None
        dex_end = dex_currency_data[-1]['timestamp'] if dex_currency_data else None
        cex_start = cex_currency_data[0]['timestamp'] if cex_currency_data else None
        cex_end = cex_currency_data[-1]['timestamp'] if cex_currency_data else None
        
        print(f"     DEX {currency} range: {dex_start} to {dex_end}")
        print(f"     CEX {currency} range: {cex_start} to {cex_end}")
        
        # # Debug: Show sample timestamps
        # if dex_currency_data and cex_currency_data:
        #     print(f"     Sample DEX timestamps: {dex_currency_data[0]['timestamp']}, {dex_currency_data[1]['timestamp']}")
        #     print(f"     Sample CEX timestamps: {cex_currency_data[0]['timestamp']}, {cex_currency_data[1]['timestamp']}")
            
        #     # Check if there's any overlap
        #     dex_times = [d['timestamp'] for d in dex_currency_data]
        #     cex_times = [c['timestamp'] for c in cex_currency_data]
            
        #     dex_min, dex_max = min(dex_times), max(dex_times)
        #     cex_min, cex_max = min(cex_times), max(cex_times)
            
        #     overlap_start = max(dex_min, cex_min)
        #     overlap_end = min(dex_max, cex_max)
            
        #     if overlap_start < overlap_end:
        #         print(f"     Overlap period: {overlap_start} to {overlap_end}")
        #     else:
        #         print(f"     NO OVERLAP! DEX: {dex_min} to {dex_max}, CEX: {cex_min} to {cex_max}")
        
        while dex_idx < len(dex_currency_data) and cex_idx < len(cex_currency_data):
            dex_time = dex_currency_data[dex_idx]['timestamp']
            cex_time = cex_currency_data[cex_idx]['timestamp']
            
            time_diff = abs((dex_time - cex_time).total_seconds() / 60)  # difference in minutes
            
            if time_diff <= tolerance_minutes:
                aligned_data.append({
                    'timestamp': dex_time,
                    'currency': currency,
                    'dex_price': dex_currency_data[dex_idx]['close'],
                    'cex_price': cex_currency_data[cex_idx]['close'],
                    'dex_volume': dex_currency_data[dex_idx]['volume'],
                    'cex_volume': cex_currency_data[cex_idx]['volume'],
                    'time_diff_minutes': time_diff
                })
                dex_matched += 1
                cex_matched += 1
                dex_idx += 1
                cex_idx += 1
            elif dex_time < cex_time:
                dex_unmatched += 1
                dex_idx += 1
            else:
                cex_unmatched += 1
                cex_idx += 1
        
        # Count remaining unmatched records
        dex_unmatched += len(dex_currency_data) - dex_idx
        cex_unmatched += len(cex_currency_data) - cex_idx
        
        print(f"     {currency} matched records: {len(aligned_data)}")
        print(f"     DEX {currency}: {dex_matched} matched, {dex_unmatched} unmatched")
        print(f"     CEX {currency}: {cex_matched} matched, {cex_unmatched} unmatched")
        
        if aligned_data:
            avg_time_diff = sum(d['time_diff_minutes'] for d in aligned_data) / len(aligned_data)
            max_time_diff = max(d['time_diff_minutes'] for d in aligned_data)
            print(f"     Average time difference: {avg_time_diff:.1f} minutes")
            print(f"     Maximum time difference: {max_time_diff:.1f} minutes")
        
        all_aligned_data[currency] = aligned_data
        total_matched += len(aligned_data)
        total_dex_unmatched += dex_unmatched
        total_cex_unmatched += cex_unmatched
    
    print(f"\n   Total matched records across all currencies: {total_matched}")
    print(f"   Total DEX unmatched: {total_dex_unmatched}")
    print(f"   Total CEX unmatched: {total_cex_unmatched}")
    
    return all_aligned_data

def calculate_deviations(aligned_data):
    """Calculate various price deviation metrics"""
    print("Calculating price deviations...")
    
    if not aligned_data:
        return {}
    
    dex_prices = np.array([d['dex_price'] for d in aligned_data])
    cex_prices = np.array([d['cex_price'] for d in aligned_data])
    
    # Basic statistics
    dex_mean = np.mean(dex_prices)
    cex_mean = np.mean(cex_prices)
    
    # Price differences
    price_diff = dex_prices - cex_prices
    abs_price_diff = np.abs(price_diff)
    
    # Percentage deviations
    pct_deviation = (price_diff / cex_prices) * 100
    abs_pct_deviation = np.abs(pct_deviation)
    
    # Statistical measures
    mean_deviation = np.mean(price_diff)
    median_deviation = np.median(price_diff)
    std_deviation = np.std(price_diff)
    
    mean_abs_deviation = np.mean(abs_price_diff)
    median_abs_deviation = np.median(abs_price_diff)
    
    mean_pct_deviation = np.mean(pct_deviation)
    median_pct_deviation = np.median(pct_deviation)
    std_pct_deviation = np.std(pct_deviation)
    
    mean_abs_pct_deviation = np.mean(abs_pct_deviation)
    median_abs_pct_deviation = np.median(abs_pct_deviation)
    
    # Correlation
    correlation, p_value = pearsonr(dex_prices, cex_prices)
    
    # Maximum deviations
    max_positive_deviation = np.max(price_diff)
    max_negative_deviation = np.min(price_diff)
    max_abs_deviation = np.max(abs_price_diff)
    
    max_positive_pct_deviation = np.max(pct_deviation)
    max_negative_pct_deviation = np.min(pct_deviation)
    max_abs_pct_deviation = np.max(abs_pct_deviation)
    
    return {
        'count': len(aligned_data),
        'dex_mean_price': dex_mean,
        'cex_mean_price': cex_mean,
        'price_difference': cex_mean - dex_mean,
        'mean_deviation': mean_deviation,
        'median_deviation': median_deviation,
        'std_deviation': std_deviation,
        'mean_abs_deviation': mean_abs_deviation,
        'median_abs_deviation': median_abs_deviation,
        'mean_pct_deviation': mean_pct_deviation,
        'median_pct_deviation': median_pct_deviation,
        'std_pct_deviation': std_pct_deviation,
        'mean_abs_pct_deviation': mean_abs_pct_deviation,
        'median_abs_pct_deviation': median_abs_pct_deviation,
        'correlation': correlation,
        'correlation_p_value': p_value,
        'max_positive_deviation': max_positive_deviation,
        'max_negative_deviation': max_negative_deviation,
        'max_abs_deviation': max_abs_deviation,
        'max_positive_pct_deviation': max_positive_pct_deviation,
        'max_negative_pct_deviation': max_negative_pct_deviation,
        'max_abs_pct_deviation': max_abs_pct_deviation
    }

def analyze_lead_lag(aligned_data, max_lag_hours=24):
    """Analyze lead-lag relationships between DEX and CEX"""
    print("Analyzing lead-lag relationships...")
    
    if len(aligned_data) < max_lag_hours:
        return {}
    
    dex_prices = np.array([d['dex_price'] for d in aligned_data])
    cex_prices = np.array([d['cex_price'] for d in aligned_data])
    
    correlations = {}
    
    # Test different lags
    for lag in range(-max_lag_hours, max_lag_hours + 1):
        if lag == 0:
            continue
            
        if lag > 0:
            # DEX leads CEX
            if len(dex_prices) > lag:
                dex_shifted = dex_prices[:-lag] if lag > 0 else dex_prices
                cex_shifted = cex_prices[lag:] if lag > 0 else cex_prices
            else:
                continue
        else:
            # CEX leads DEX
            if len(cex_prices) > abs(lag):
                dex_shifted = dex_prices[abs(lag):] if lag < 0 else dex_prices
                cex_shifted = cex_prices[:-abs(lag)] if lag < 0 else cex_prices
            else:
                continue
        
        if len(dex_shifted) > 10 and len(cex_shifted) > 10:
            min_len = min(len(dex_shifted), len(cex_shifted))
            dex_shifted = dex_shifted[:min_len]
            cex_shifted = cex_shifted[:min_len]
            
            try:
                corr, p_val = pearsonr(dex_shifted, cex_shifted)
                correlations[lag] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'sample_size': min_len
                }
            except:
                continue
    
    # Find best correlation
    if correlations:
        best_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]['correlation']))
        best_corr = correlations[best_lag]
        
        return {
            'best_lag_hours': best_lag, #DEX prices move first - When you look at DEX prices at time T, they are most highly correlated with CEX prices at time T+4 hours (e.g. 4 hours later)
            'best_correlation': best_corr['correlation'],
            'best_p_value': best_corr['p_value'],
            'all_correlations': correlations
        }
    
    return {}

def print_analysis_results(deviations, lead_lag, currency="BTC"):
    """Print comprehensive analysis results"""
    print("\n" + "="*80)
    print(f"DEX vs CEX PRICE COMPARISON ANALYSIS - {currency}")
    print("="*80)
    
    if not deviations:
        print("No data available for analysis.")
        return
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   • Total aligned data points: {deviations['count']:,}")
    print(f"   • DEX average price: ${deviations['dex_mean_price']:,.2f}")
    print(f"   • CEX average price: ${deviations['cex_mean_price']:,.2f}")
    print(f"   • Average price difference (CEX - DEX): ${deviations['price_difference']:,.2f}")
    
    print(f"\n💰 PRICE DEVIATIONS:")
    print(f"   • Mean deviation: ${deviations['mean_deviation']:,.2f}")
    print(f"   • Median deviation: ${deviations['median_deviation']:,.2f}")
    print(f"   • Standard deviation: ${deviations['std_deviation']:,.2f}")
    print(f"   • Mean absolute deviation: ${deviations['mean_abs_deviation']:,.2f}")
    print(f"   • Median absolute deviation: ${deviations['median_abs_deviation']:,.2f}")
    
    print(f"\n📈 PERCENTAGE DEVIATIONS:")
    print(f"   • Mean % deviation: {deviations['mean_pct_deviation']:.4f}%")
    print(f"   • Median % deviation: {deviations['median_pct_deviation']:.4f}%")
    print(f"   • Standard deviation %: {deviations['std_pct_deviation']:.4f}%")
    print(f"   • Mean absolute % deviation: {deviations['mean_abs_pct_deviation']:.4f}%")
    print(f"   • Median absolute % deviation: {deviations['median_abs_pct_deviation']:.4f}%")
    
    print(f"\n🔗 CORRELATION ANALYSIS:")
    print(f"   • Price correlation: {deviations['correlation']:.6f}")
    print(f"   • Correlation p-value: {deviations['correlation_p_value']:.6f}")
    if deviations['correlation_p_value'] < 0.05:
        print(f"   • Correlation is statistically significant (p < 0.05)")
    else:
        print(f"   • Correlation is not statistically significant (p >= 0.05)")
    
    print(f"\n📊 EXTREME DEVIATIONS:")
    print(f"   • Maximum positive deviation: ${deviations['max_positive_deviation']:,.2f}")
    print(f"   • Maximum negative deviation: ${deviations['max_negative_deviation']:,.2f}")
    print(f"   • Maximum absolute deviation: ${deviations['max_abs_deviation']:,.2f}")
    print(f"   • Max positive % deviation: {deviations['max_positive_pct_deviation']:.4f}%")
    print(f"   • Max negative % deviation: {deviations['max_negative_pct_deviation']:.4f}%")
    print(f"   • Max absolute % deviation: {deviations['max_abs_pct_deviation']:.4f}%")
    
    if lead_lag:
        print(f"\n⏰ LEAD-LAG ANALYSIS:")
        print(f"   • Best correlation lag: {lead_lag['best_lag_hours']} hours")
        print(f"   • Best correlation: {lead_lag['best_correlation']:.6f}")
        print(f"   • Best correlation p-value: {lead_lag['best_p_value']:.6f}")
        
        if lead_lag['best_lag_hours'] > 0:
            print(f"   • DEX prices lead CEX prices by {lead_lag['best_lag_hours']} hours")
        elif lead_lag['best_lag_hours'] < 0:
            print(f"   • CEX prices lead DEX prices by {abs(lead_lag['best_lag_hours'])} hours")
        else:
            print(f"   • No significant lead-lag relationship found")
    
    print(f"\n🎯 INTERPRETATION:")
    if deviations['mean_pct_deviation'] > 0:
        print(f"   • On average, DEX prices are {deviations['mean_pct_deviation']:.4f}% higher than CEX prices")
    else:
        print(f"   • On average, DEX prices are {abs(deviations['mean_pct_deviation']):.4f}% lower than CEX prices")
    
    if deviations['correlation'] > 0.9:
        print(f"   • Very strong positive correlation between DEX and CEX prices")
    elif deviations['correlation'] > 0.7:
        print(f"   • Strong positive correlation between DEX and CEX prices")
    elif deviations['correlation'] > 0.5:
        print(f"   • Moderate positive correlation between DEX and CEX prices")
    elif deviations['correlation'] > 0.3:
        print(f"   • Weak positive correlation between DEX and CEX prices")
    else:
        print(f"   • Very weak or no correlation between DEX and CEX prices")
    
    if deviations['mean_abs_pct_deviation'] < 0.1:
        print(f"   • Prices are very close (mean absolute deviation < 0.1%)")
    elif deviations['mean_abs_pct_deviation'] < 0.5:
        print(f"   • Prices are reasonably close (mean absolute deviation < 0.5%)")
    elif deviations['mean_abs_pct_deviation'] < 1.0:
        print(f"   • Prices show moderate deviation (mean absolute deviation < 1%)")
    else:
        print(f"   • Prices show significant deviation (mean absolute deviation >= 1%)")

def main():
    """Main analysis function"""
    print("Starting DEX vs CEX Price Comparison Analysis...")
    
    try:
        # Fetch data
        dex_data = get_dex_prices()
        cex_data = get_cex_prices()
        
        print(f"Fetched {len(dex_data)} DEX data points")
        print(f"Fetched {len(cex_data)} CEX data points")
        
        if dex_data:
            print(f"DEX time range: {dex_data[0]['timestamp']} to {dex_data[-1]['timestamp']}")
        if cex_data:
            print(f"CEX time range: {cex_data[0]['timestamp']} to {cex_data[-1]['timestamp']}")
        
        if not dex_data or not cex_data:
            print("Error: No data available from one or both sources")
            return
        
        # Align timestamps
        aligned_data_by_currency = align_timestamps(dex_data, cex_data, tolerance_minutes=60)
        
        if not aligned_data_by_currency:
            print("Error: No data points could be aligned")
            return
        
        # Process each currency
        all_results = {}
        all_data_for_csv = []
        
        for currency, aligned_data in aligned_data_by_currency.items():
            print(f"\n{'='*60}")
            print(f"ANALYZING {currency}")
            print(f"{'='*60}")
            
            if not aligned_data:
                print(f"No aligned data for {currency}")
                continue
            
            print(f"Aligned {len(aligned_data)} {currency} data points")
            
            # Calculate deviations
            deviations = calculate_deviations(aligned_data)
            
            # Analyze lead-lag relationships
            lead_lag = analyze_lead_lag(aligned_data)
            
            # Print results
            print_analysis_results(deviations, lead_lag, currency)
            
            # Store results
            all_results[currency] = {
                'deviations': deviations,
                'lead_lag': lead_lag
            }
            
            # Add to CSV data
            all_data_for_csv.extend(aligned_data)
        
        # Save detailed data to CSV
        if all_data_for_csv:
            df = pd.DataFrame(all_data_for_csv)
            df.to_csv('dex_cex_comparison.csv', index=False)
            print(f"\n💾 Detailed data saved to 'dex_cex_comparison.csv'")
            
            # Print summary across all currencies
            print(f"\n{'='*80}")
            print("SUMMARY ACROSS ALL CURRENCIES")
            print(f"{'='*80}")
            
            for currency, results in all_results.items():
                if results['deviations']:
                    print(f"\n{currency}:")
                    print(f"  • Correlation: {results['deviations']['correlation']:.6f}")
                    print(f"  • Mean % deviation: {results['deviations']['mean_pct_deviation']:.4f}%")
                    print(f"  • Mean abs % deviation: {results['deviations']['mean_abs_pct_deviation']:.4f}%")
                    if results['lead_lag'] and 'best_lag_hours' in results['lead_lag']:
                        lag = results['lead_lag']['best_lag_hours']
                        if lag > 0:
                            print(f"  • DEX leads CEX by {lag} hours")
                        elif lag < 0:
                            print(f"  • CEX leads DEX by {abs(lag)} hours")
                        else:
                            print(f"  • No significant lead-lag relationship")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
