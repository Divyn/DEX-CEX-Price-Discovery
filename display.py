#!/usr/bin/env python3
"""
Display Module for DEX vs CEX Analysis Results
Handles all output formatting and result presentation
"""

def print_analysis_results(deviations, lead_lag, volume_analysis=None, currency="BTC", exchange="CEX"):
    """Print comprehensive analysis results"""
    print("\n" + "="*80)
    print(f"DEX vs {exchange.upper()} PRICE COMPARISON ANALYSIS - {currency}")
    print("="*80)
    
    if not deviations:
        print("No data available for analysis.")
        return
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   • Total aligned data points: {deviations['count']:,}")
    print(f"   • DEX average price: ${deviations['dex_mean_price']:,.2f}")
    print(f"   • {exchange.upper()} average price: ${deviations['cex_mean_price']:,.2f}")
    print(f"   • Average price difference ({exchange.upper()} - DEX): ${deviations['price_difference']:,.2f}")
    
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
        print(f"   • Best correlation lag: {lead_lag['best_lag_minutes']} minutes")
        print(f"   • Best correlation: {lead_lag['best_correlation']:.6f}")
        print(f"   • Best correlation p-value: {lead_lag['best_p_value']:.6f}")
        print(f"   • Sample size: {lead_lag['sample_size']}")
        print(f"   • Time tolerance: ±{lead_lag['time_tolerance']} minutes")
        
        if lead_lag['best_lag_minutes'] > 0:
            print(f"   • DEX prices lead CEX prices by {lead_lag['best_lag_minutes']} minutes")
        elif lead_lag['best_lag_minutes'] < 0:
            print(f"   • CEX prices lead DEX prices by {abs(lead_lag['best_lag_minutes'])} minutes")
        else:
            print(f"   • No significant lead-lag relationship found")
        
        print(f"   • Note: {lead_lag['note']}")
    
    if volume_analysis:
        print(f"\n📊 VOLUME ANALYSIS:")
        print(f"   • DEX total volume: ${volume_analysis['dex_total_volume']:,.2f}")
        print(f"   • {exchange.upper()} total volume: ${volume_analysis['cex_total_volume']:,.2f}")
        print(f"   • Volume ratio ({exchange.upper()}/DEX): {volume_analysis['volume_ratio']:.2f}x")
        print(f"   • DEX mean volume: ${volume_analysis['dex_mean_volume']:,.2f}")
        print(f"   • {exchange.upper()} mean volume: ${volume_analysis['cex_mean_volume']:,.2f}")
        print(f"   • Volume correlation: {volume_analysis['volume_correlation']:.6f}")
        print(f"   • Volume correlation p-value: {volume_analysis['volume_p_value']:.6f}")
        
        print(f"\n📈 VOLUME PERCENTILES:")
        print(f"   • DEX 90th percentile: ${volume_analysis['dex_volume_percentiles']['90th']:,.2f}")
        print(f"   • {exchange.upper()} 90th percentile: ${volume_analysis['cex_volume_percentiles']['90th']:,.2f}")
        print(f"   • DEX high volume periods: {volume_analysis['dex_high_volume_periods']}")
        print(f"   • {exchange.upper()} high volume periods: {volume_analysis['cex_high_volume_periods']}")
        
        print(f"\n💰 VWAP COMPARISON:")
        print(f"   • DEX VWAP: ${volume_analysis['dex_vwap']:,.2f}")
        print(f"   • {exchange.upper()} VWAP: ${volume_analysis['cex_vwap']:,.2f}")
        print(f"   • VWAP deviation: {volume_analysis['vwap_deviation']:.4f}%")
    
    print(f"\n🎯 INTERPRETATION:")
    if deviations['mean_pct_deviation'] > 0:
        print(f"   • On average, DEX prices are {deviations['mean_pct_deviation']:.4f}% higher than {exchange.upper()} prices")
    else:
        print(f"   • On average, DEX prices are {abs(deviations['mean_pct_deviation']):.4f}% lower than {exchange.upper()} prices")
    
    if deviations['correlation'] > 0.9:
        print(f"   • Very strong positive correlation between DEX and {exchange.upper()} prices")
    elif deviations['correlation'] > 0.7:
        print(f"   • Strong positive correlation between DEX and {exchange.upper()} prices")
    elif deviations['correlation'] > 0.5:
        print(f"   • Moderate positive correlation between DEX and {exchange.upper()} prices")
    else:
        print(f"   • Weak correlation between DEX and {exchange.upper()} prices")
    
    if deviations['mean_abs_pct_deviation'] < 0.5:
        print(f"   • Prices are very close (mean absolute deviation < 0.5%)")
    elif deviations['mean_abs_pct_deviation'] < 1.0:
        print(f"   • Prices are reasonably close (mean absolute deviation < 1%)")
    else:
        print(f"   • Prices show significant deviation (mean absolute deviation > 1%)")

def print_summary(all_results):
    """Print summary across all exchanges and currencies"""
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL EXCHANGES AND CURRENCIES")
    print(f"{'='*80}")
    
    for exchange_name, exchange_results in all_results.items():
        print(f"\n{exchange_name.upper()}:")
        for currency, results in exchange_results.items():
            if results and 'deviations' in results:
                print(f"  {currency}:")
                print(f"    • Correlation: {results['deviations']['correlation']:.6f}")
                print(f"    • Mean % deviation: {results['deviations']['mean_pct_deviation']:.4f}%")
                print(f"    • Mean abs % deviation: {results['deviations']['mean_abs_pct_deviation']:.4f}%")
                if results.get('lead_lag') and 'best_lag_minutes' in results['lead_lag']:
                    lag = results['lead_lag']['best_lag_minutes']
                    if lag > 0:
                        print(f"    • DEX leads CEX by {lag} minutes")
                    elif lag < 0:
                        print(f"    • CEX leads DEX by {abs(lag)} minutes")
                    else:
                        print(f"    • No significant lead-lag relationship")
                if results.get('volume_analysis') and results['volume_analysis']:
                    vol = results['volume_analysis']
                    print(f"    • Volume ratio: {vol['volume_ratio']:.2f}x")
                    print(f"    • Volume correlation: {vol['volume_correlation']:.4f}")
                    print(f"    • VWAP deviation: {vol['vwap_deviation']:.4f}%")
