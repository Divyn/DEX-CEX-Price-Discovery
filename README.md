# Price Discovery Dynamics Between Decentralized and Centralized Cryptocurrency Exchanges : Code Sample for Paper

Link to paper: https://doi.org/10.5281/zenodo.17084251

## Main Orchestration Files
`multi_exchange_comparison.py` - Main entry point
- Purpose: Main orchestration script that coordinates the entire analysis
- Function: Imports modules, fetches data, processes each exchange separately, and generates final results
- Key features: Handles multiple exchanges (Binance, Coinbase), saves results to CSV, prints comprehensive analysis

## Related Modules

`api_client.py`- Data fetching module
-   Purpose: Handles all external API calls for both DEX and CEX data
-   Data sources: Bitquery (DEX), Binance & Coinbase (CEX)

` calculations.py` - Statistical analysis engine

-   Purpose: Contains all statistical analysis functions
-   Features: Handles log returns, time lag analysis, volume correlations

  `display.py` - Results presentation module

-   Purpose: Handles all output formatting and result presentation
-   Features: Formatted output, detailed statistical summaries, interpretation guidance

  
`auth.py` - API authentication

-   Purpose: Stores API keys and sensitive configuration

-   Content: Bitquery OAuth token for DEX data access. You can generate one [here](https://account.bitquery.io/user/api_v2/access_tokens)
