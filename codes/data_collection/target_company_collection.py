#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target Company Collection

Created on Thu Feb 17 11:58:34 2022

@author: shiqimeng
"""
# ===================================================================================
# Get target company list with identification number (CIK)
#   output format: csv
# ===================================================================================
# First download company_tickers_exchange.json in Section \
# [CIK, ticker, and exchange associations] from
# https://www.sec.gov/os/accessing-edgar-data

# Change the json file to csv format
import os
import pandas as pd
import json

path = '/Users/shiqimeng/Downloads/NLP projects'

# Load json file
with open(path + os.sep + 'company_tickers_exchange.json') as json_data:
    ticker_data = json.load(json_data)

# Convert to dataframe
df = pd.DataFrame(ticker_data['data'], columns = ticker_data['fields'])

# cik starts with 0s and must be 10 digits
# Fill 0s before cik, create a new column called cik_full
df['cik_full'] = df['cik'].astype(str)
df['cik_full'] = df['cik_full'].str.zfill(10)

# Remove securities listed in OTC and CBOE
df_stock = df.loc[(df['exchange'] != 'OTC') * (df['exchange'] != 'CBOE')]

# Save csv
df.to_csv(path + os.sep + 'company_tickers_exchange_all.csv',index = None)
df_stock.to_csv(path + os.sep + 'company_tickers_exchange_stock.csv',index = None)

# NYSE only
df_NYSE = df_stock.loc[df_stock['exchange'] == 'NYSE']
df_NYSE.to_csv(path + os.sep + 'company_tickers_exchange_NYSE.csv',index = None)

## Further cleaning -> remove inactive firms

# ===================================================================================
# Scrape Wiki for S&P 500 composite company info
#   output format: csv
# ===================================================================================
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup

path = '/Users/shiqimeng/Downloads/NLP projects'

wikiurl = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table_class = 'wikitable sortable jquery-tablesorter'
response = requests.get(wikiurl)
print(response.status_code)  # Check status -- if 200, then OK

# Parse data from the html into a beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
sp500table = soup.find('table', {'class':"wikitable"})

# Convert Table into a dataframe
df = pd.read_html(str(sp500table))
df = pd.DataFrame(df[0])

# Drop 'SEC filings'
sp500 = df.drop(['SEC filings'], axis = 1)

# Export
sp500.to_csv(path + os.sep + 'sp500_list.csv',index = None)

# ===================================================================================
# Merge S&P 500 list and SEC company
#   input: sp500_list.csv, company_tickers_exchange_stock.csv
#   output format: sp500_combined.csv
# ===================================================================================
import pandas as pd
import os

path = '/Users/shiqimeng/Downloads/NLP projects'

all_stock = pd.read_csv(path + os.sep + 'company_tickers_exchange_stock.csv',\
                        converters={'cik_full': lambda x: str(x)})
sp500 = pd.read_csv(path + os.sep + 'sp500_list.csv')

combined = pd.merge(all_stock, sp500, left_on = 'cik', right_on = 'CIK', how = 'inner')

combined = combined.drop_duplicates(subset = ['cik'])[['cik','cik_full','name','ticker',\
                                            'GICS Sector','GICS Sub-Industry',\
                                                'Headquarters Location',\
                                                    'Date first added','Founded']]
    
combined.to_csv(path + os.sep + 'sp500_combined.csv',index = None)