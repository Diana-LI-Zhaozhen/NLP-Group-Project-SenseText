# MFIN7036 NLP Project

## Content of each folder
### 1. datasets
Major datasets:
The 10K items datasets are too large, so only the google drive links are provided.
- cleaned 10K items for 2,965 companies: 10k_info_cleaned.csv & cleaned_10k_items.pkl
- betas and valuation metrics: valuation.csv
- revenues and GICS sub-industry codes: tic_rev_ind.csv

others:
- all companies reporting to SEC: company_tickers_exchange.json
- stocks only (remove those in OTC and CBOE market): company_tickers_exchange_stock.csv
- S&P 500 companies' info: sp500_combined.csv
- non S&P 500 stocks in NYSE and NASDAQ: stocks_not_sp500.csv

### 2. codes
#### (i) data collection
First, get the list of stocks with their CIK (Central Index Key) from SEC json file that contains all reporting companies:
We initially targeted at S&P 500 companies, but later expanded to all companies listed in NYSE and NASDAQ.
- target_company_collection.py
- get_stocks_outside_sp500.py

SEC offers a free-to-use api for web scraping. All 10K files can be accessed through a highly structured url, i.e. 'https://www.sec.gov/Archives/edgar/data/' + cik + '/' + accn_cont + '/' + accn + '.txt'. accn is the accession number for each filing, following the format ##########-YY-SSSSSS. accn_cont is accn without dashes. The 10Ks are written in XML format; the url provides access to a txt file.
In order to find the accn for the latest 10K filing, we need to search the submission history of a firm. Again, there is a structured url for that, i.e. 'https://data.sec.gov/api/xbrl/companyfacts/CIK' + cik_full + '.json'. cik_full appends 0s before CIK until it becomes 10 digits.
- scrape_10k_path.py
- scrape_10k_items.py

Merge and clean datasets: remove those companies using a non-standard reporting style (items will be less than 1000 characters)
- data_cleaning_merge.py

Financial data collection using yfinance api:
- financial_data_collection.py

#### (ii) data processing (with transformation)
For item 1. business only:
- text_processing_busi_only.py

#### (iii) data analysis
For item 1. business only:
- text_processing_busi_only.py