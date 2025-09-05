#!/usr/bin/env python3
"""
Multi-Exchange DEX vs CEX Price Comparison Analysis
Main orchestration file that coordinates API calls, calculations, and display
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from api_client import get_dex_prices, get_cex_prices
from calculations import align_timestamps_simple, calculate_deviations, analyze_lead_lag, analyze_volume_comparison
from display import print_analysis_results, print_summary

def main():
    """Main analysis function"""
    print("Starting Multi-Exchange DEX vs CEX Price Comparison Analysis...")
    
    try:
        # Fetch data using API client
        dex_data = get_dex_prices()
        cex_data = get_cex_prices()
        
        print(f"Fetched {len(dex_data)} DEX data points")
        print(f"Fetched {len(cex_data)} CEX data points")
        
        if not dex_data or not cex_data:
            print("Error: No data available from one or both sources")
            return
        
        # Group CEX data by exchange
        cex_by_exchange = {}
        for item in cex_data:
            exchange = item.get('exchange', 'unknown')
            if exchange not in cex_by_exchange:
                cex_by_exchange[exchange] = []
            cex_by_exchange[exchange].append(item)
        
        # Process each exchange separately
        all_results = {}
        all_data_for_csv = []
        
        for exchange_name, exchange_cex_data in cex_by_exchange.items():
            print(f"\n{'='*80}")
            print(f"ANALYZING DEX vs {exchange_name.upper()}")
            print(f"{'='*80}")
            
            # Align timestamps using calculations module
            aligned_data_by_currency = align_timestamps_simple(dex_data, exchange_cex_data)
            
            if not aligned_data_by_currency:
                print(f"No aligned data for {exchange_name}")
                continue
            
            # Process each currency
            for currency, aligned_data in aligned_data_by_currency.items():
                print(f"\n{'='*60}")
                print(f"ANALYZING {currency} - DEX vs {exchange_name.upper()}")
                print(f"{'='*60}")
                
                if not aligned_data:
                    print(f"No aligned data for {currency}")
                    continue
                
                print(f"Aligned {len(aligned_data)} {currency} data points")
                
                # Perform all calculations using calculations module
                deviations = calculate_deviations(aligned_data)
                lead_lag = analyze_lead_lag(aligned_data)
                volume_analysis = analyze_volume_comparison(aligned_data)
                
                # Display results using display module
                print_analysis_results(deviations, lead_lag, volume_analysis, currency, exchange_name)
                
                # Store results
                if exchange_name not in all_results:
                    all_results[exchange_name] = {}
                all_results[exchange_name][currency] = {
                    'deviations': deviations,
                    'lead_lag': lead_lag,
                    'volume_analysis': volume_analysis
                }
                
                # Add to CSV data
                for item in aligned_data:
                    item['exchange'] = exchange_name
                    all_data_for_csv.append(item)
        
        # Save detailed data to CSV
        if all_data_for_csv:
            df = pd.DataFrame(all_data_for_csv)
            df.to_csv('multi_exchange_comparison.csv', index=False)
            print(f"\n💾 Detailed data saved to 'multi_exchange_comparison.csv'")
            
            # Print summary using display module
            print_summary(all_results)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
