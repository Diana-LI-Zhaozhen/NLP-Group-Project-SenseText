#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get a list of stocks that are not S&P 500

Created on Mon Feb 21 10:43:42 2022

@author: shiqimeng
"""
import pandas as pd
import os, csv

path = '/Users/shiqimeng/Downloads/NLP projects'

sp500 = pd.read_csv(path + os.sep + 'sp500_combined.csv',\
                    converters={'cik_full': lambda x: str(x)})

all_stocks = pd.read_csv(path + os.sep + 'company_tickers_exchange_stock.csv',\
                    converters={'cik_full': lambda x: str(x)})

sp500_list = sp500['cik'].tolist()

all_stocks_with_index = all_stocks.set_index('cik')

not_sp500 = all_stocks_with_index.drop(labels = sp500_list, axis = 0)

not_sp500.to_csv(path + os.sep + 'stocks_not_sp500.csv')