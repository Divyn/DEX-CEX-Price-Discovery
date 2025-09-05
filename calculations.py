#!/usr/bin/env python3
"""
Calculations Module for DEX vs CEX Analysis
Contains all statistical analysis functions for price and volume comparison
"""

import numpy as np
from scipy.stats import pearsonr

def align_timestamps_simple(dex_data, cex_data, tolerance_minutes=30):
    """Simple alignment for DEX vs single CEX exchange"""
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
            cex_by_currency[currency] = []
        cex_by_currency[currency].append(item)
    
    all_aligned_data = {}
    total_matched = 0
    
    # Process each currency separately
    for currency in set(list(dex_by_currency.keys()) + list(cex_by_currency.keys())):
        print(f"\n   Processing {currency}...")
        
        if currency not in dex_by_currency or currency not in cex_by_currency:
            print(f"     Skipping {currency} - missing data from one source")
            continue
        
        dex_currency_data = sorted(dex_by_currency[currency], key=lambda x: x['timestamp'])
        cex_currency_data = sorted(cex_by_currency[currency], key=lambda x: x['timestamp'])
        
        print(f"     DEX {currency} range: {dex_currency_data[0]['timestamp']} to {dex_currency_data[-1]['timestamp']}")
        print(f"     CEX {currency} range: {cex_currency_data[0]['timestamp']} to {cex_currency_data[-1]['timestamp']}")
        
        aligned_data = []
        dex_matched = 0
        cex_matched = 0
        
        # Simple alignment: find closest CEX timestamp for each DEX timestamp
        for dex_item in dex_currency_data:
            dex_timestamp = dex_item['timestamp']
            best_cex_item = None
            best_time_diff = float('inf')
            
            for cex_item in cex_currency_data:
                time_diff = abs((dex_timestamp - cex_item['timestamp']).total_seconds() / 60)
                if time_diff < best_time_diff and time_diff <= tolerance_minutes:
                    best_time_diff = time_diff
                    best_cex_item = cex_item
            
            if best_cex_item is not None:
                aligned_data.append({
                    'timestamp': dex_timestamp,
                    'dex_price': dex_item['close'],
                    'dex_volume': dex_item['volume'],
                    'cex_price': best_cex_item['close'],
                    'cex_volume': best_cex_item['volume'],
                    'time_diff_minutes': best_time_diff
                })
                dex_matched += 1
                cex_matched += 1
        
        if aligned_data:
            time_diffs = [item['time_diff_minutes'] for item in aligned_data]
            avg_time_diff = np.mean(time_diffs)
            max_time_diff = np.max(time_diffs)
            
            print(f"     {currency} matched records: {len(aligned_data)}")
            print(f"     DEX {currency}: {dex_matched} matched, {len(dex_currency_data) - dex_matched} unmatched")
            print(f"     CEX {currency}: {cex_matched} matched, {len(cex_currency_data) - cex_matched} unmatched")
            print(f"     Average time difference: {avg_time_diff:.1f} minutes")
            print(f"     Maximum time difference: {max_time_diff:.1f} minutes")
            
            all_aligned_data[currency] = aligned_data
            total_matched += len(aligned_data)
    
    print(f"\n   Total matched records across all currencies: {total_matched}")
    return all_aligned_data

def calculate_deviations(aligned_data):
    """Calculate price deviations between DEX and CEX"""
    if not aligned_data:
        return {}
    
    dex_prices = np.array([d['dex_price'] for d in aligned_data])
    cex_prices = np.array([d['cex_price'] for d in aligned_data])
    
    # Basic statistics
    dex_mean = np.mean(dex_prices)
    cex_mean = np.mean(cex_prices)
    
    # Price differences
    price_diff = cex_prices - dex_prices
    abs_price_diff = np.abs(price_diff)
    
    # Percentage differences
    pct_deviation = (price_diff / dex_prices) * 100
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

def analyze_volume_comparison(aligned_data):
    """Analyze volume patterns between DEX and CEX markets"""
    print("Analyzing volume patterns...")
    
    if len(aligned_data) < 10:
        return {}
    
    # Extract volume data - use same indices for both to maintain alignment
    valid_indices = [i for i, d in enumerate(aligned_data) if d['dex_volume'] > 0 and d['cex_volume'] > 0]
    
    if len(valid_indices) == 0:
        print("   Warning: No valid volume data found")
        return {}
    
    # Extract volumes and convert CEX volume to USD
    dex_volumes = np.array([aligned_data[i]['dex_volume'] for i in valid_indices])  # Already in USD
    cex_volumes_base = np.array([aligned_data[i]['cex_volume'] for i in valid_indices])  # Base currency
    cex_prices = np.array([aligned_data[i]['cex_price'] for i in valid_indices])  # Price in USD
    
    # Convert CEX volume to USD by multiplying by price
    cex_volumes = cex_volumes_base * cex_prices
    
    print(f"   Volume analysis: {len(dex_volumes)} valid volume pairs")
    print(f"   DEX total volume (USD): ${np.sum(dex_volumes):,.2f}")
    print(f"   CEX total volume (USD): ${np.sum(cex_volumes):,.2f}")
    print(f"   DEX volume range: ${np.min(dex_volumes):,.2f} - ${np.max(dex_volumes):,.2f}")
    print(f"   CEX volume range: ${np.min(cex_volumes):,.2f} - ${np.max(cex_volumes):,.2f}")
    
    # Basic volume statistics
    dex_total_volume = np.sum(dex_volumes)
    cex_total_volume = np.sum(cex_volumes)
    
    dex_mean_volume = np.mean(dex_volumes)
    cex_mean_volume = np.mean(cex_volumes)
    
    dex_median_volume = np.median(dex_volumes)
    cex_median_volume = np.median(cex_volumes)
    
    dex_std_volume = np.std(dex_volumes)
    cex_std_volume = np.std(cex_volumes)
    
    # Volume ratios
    volume_ratio = cex_total_volume / dex_total_volume if dex_total_volume > 0 else 0
    mean_volume_ratio = cex_mean_volume / dex_mean_volume if dex_mean_volume > 0 else 0
    
    # Volume correlation
    volume_correlation, volume_p_value = pearsonr(dex_volumes, cex_volumes)
    
    # Volume percentiles
    dex_volume_percentiles = {
        '25th': np.percentile(dex_volumes, 25),
        '75th': np.percentile(dex_volumes, 75),
        '90th': np.percentile(dex_volumes, 90),
        '95th': np.percentile(dex_volumes, 95)
    }
    
    cex_volume_percentiles = {
        '25th': np.percentile(cex_volumes, 25),
        '75th': np.percentile(cex_volumes, 75),
        '90th': np.percentile(cex_volumes, 90),
        '95th': np.percentile(cex_volumes, 95)
    }
    
    # High volume periods analysis (top 10%)
    dex_high_volume_threshold = np.percentile(dex_volumes, 90)
    cex_high_volume_threshold = np.percentile(cex_volumes, 90)
    
    dex_high_volume_periods = np.sum(dex_volumes >= dex_high_volume_threshold)
    cex_high_volume_periods = np.sum(cex_volumes >= cex_high_volume_threshold)
    
    # Volume-weighted average price (VWAP) comparison
    dex_prices = np.array([d['dex_price'] for d in aligned_data if d['dex_volume'] > 0])
    cex_prices = np.array([d['cex_price'] for d in aligned_data if d['cex_volume'] > 0])
    
    if len(dex_prices) > 0 and len(cex_prices) > 0:
        dex_vwap = np.sum(dex_prices * dex_volumes) / np.sum(dex_volumes)
        cex_vwap = np.sum(cex_prices * cex_volumes) / np.sum(cex_volumes)
        vwap_deviation = ((cex_vwap - dex_vwap) / dex_vwap * 100) if dex_vwap > 0 else 0
    else:
        dex_vwap = cex_vwap = vwap_deviation = 0
    
    return {
        'dex_total_volume': dex_total_volume,
        'cex_total_volume': cex_total_volume,
        'dex_mean_volume': dex_mean_volume,
        'cex_mean_volume': cex_mean_volume,
        'dex_median_volume': dex_median_volume,
        'cex_median_volume': cex_median_volume,
        'dex_std_volume': dex_std_volume,
        'cex_std_volume': cex_std_volume,
        'volume_ratio': volume_ratio,
        'mean_volume_ratio': mean_volume_ratio,
        'volume_correlation': volume_correlation,
        'volume_p_value': volume_p_value,
        'dex_volume_percentiles': dex_volume_percentiles,
        'cex_volume_percentiles': cex_volume_percentiles,
        'dex_high_volume_periods': dex_high_volume_periods,
        'cex_high_volume_periods': cex_high_volume_periods,
        'dex_vwap': dex_vwap,
        'cex_vwap': cex_vwap,
        'vwap_deviation': vwap_deviation
    }

def analyze_lead_lag(aligned_data, max_lag_minutes=60):
    """Analyze lead-lag relationships between DEX and CEX using actual time differences"""
    print("Analyzing lead-lag relationships...")
    
    if len(aligned_data) < 10:
        return {}
    
    # Convert to log returns to remove trend effects
    dex_prices = np.array([d['dex_price'] for d in aligned_data])
    cex_prices = np.array([d['cex_price'] for d in aligned_data])
    
    # Calculate log returns
    dex_returns = np.diff(np.log(dex_prices))
    cex_returns = np.diff(np.log(cex_prices))
    
    # Get time differences for proper lag calculation
    timestamps = [d['timestamp'] for d in aligned_data]
    time_diffs = []
    for i in range(1, len(timestamps)):
        diff_minutes = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
        time_diffs.append(diff_minutes)
    
    # Calculate cumulative time differences
    cumulative_time = np.cumsum([0] + time_diffs)
    
    correlations = {}
    
    # Test different lags in minutes
    for lag_minutes in range(-max_lag_minutes, max_lag_minutes + 1, 5):  # Test every 5 minutes
        if lag_minutes == 0:
            continue
        
        # Find indices that correspond to the desired time lag
        if lag_minutes > 0:
            # DEX leads CEX - find where time difference is closest to lag_minutes
            valid_pairs = []
            for i in range(len(dex_returns)):
                for j in range(i+1, len(cex_returns)):
                    time_diff = cumulative_time[j] - cumulative_time[i]
                    if abs(time_diff - lag_minutes) < 2.5:  # Within 2.5 minutes tolerance
                        valid_pairs.append((i, j))
        else:
            # CEX leads DEX
            valid_pairs = []
            for i in range(len(cex_returns)):
                for j in range(i+1, len(dex_returns)):
                    time_diff = cumulative_time[j] - cumulative_time[i]
                    if abs(time_diff - abs(lag_minutes)) < 2.5:  # Within 2.5 minutes tolerance
                        valid_pairs.append((j, i))
        
        if len(valid_pairs) < 10:  # Need sufficient data points
            continue
        
        # Extract aligned returns
        dex_aligned = []
        cex_aligned = []
        
        for dex_idx, cex_idx in valid_pairs:
            if dex_idx < len(dex_returns) and cex_idx < len(cex_returns):
                dex_aligned.append(dex_returns[dex_idx])
                cex_aligned.append(cex_returns[cex_idx])
        
        if len(dex_aligned) < 10:
            continue
        
        # Standardize returns
        dex_std = (np.array(dex_aligned) - np.mean(dex_aligned)) / np.std(dex_aligned)
        cex_std = (np.array(cex_aligned) - np.mean(cex_aligned)) / np.std(cex_aligned)
        
        # Calculate correlation
        correlation, p_value = pearsonr(dex_std, cex_std)
        correlations[lag_minutes] = {
            'correlation': correlation,
            'p_value': p_value,
            'sample_size': len(dex_aligned)
        }
    
    if not correlations:
        return {}
    
    # Find best correlation
    best_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]['correlation']))
    best_correlation = correlations[best_lag]['correlation']
    best_p_value = correlations[best_lag]['p_value']
    sample_size = correlations[best_lag]['sample_size']
    
    return {
        'best_lag_minutes': best_lag,
        'best_correlation': best_correlation,
        'best_p_value': best_p_value,
        'sample_size': sample_size,
        'time_tolerance': 2.5,
        'note': 'P-values are optimistic due to autocorrelation. Consider HAC standard errors for significance testing.'
    }
