#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape 10-K path for all stocks in US

Created on Thu Feb 17 20:27:59 2022

@author: shiqimeng
"""
# ===================================================================================
# Get 10-K path for target companies
#   input: csv with company id (CIK) - sp500_combined.csv, stocks_not_sp500.csv
#   output format: f10k_path_sp500.csv, f10k_path_not_sp500.csv
# Note: The initial plan was to scrape S&P 500, later the target company set expanded
#       to include other stocks. So there are two inputs.
# ===================================================================================
import pandas as pd
import os, csv
import requests
import time
from random import uniform

path = '/Users/shiqimeng/Downloads/NLP projects'

# Import csv and get the list of CIK, full CIK, and ticker
# sp500 = pd.read_csv(path + os.sep + 'sp500_combined.csv',\
#                     converters={'cik_full': lambda x: str(x)})

# cik_ls = sp500['cik'].astype(str).tolist()
# cik_full_ls = sp500['cik_full'].tolist()
# ticker_ls = sp500['ticker'].tolist()

not_sp = pd.read_csv(path + os.sep + 'stocks_not_sp500.csv',\
                    converters={'cik_full': lambda x: str(x)})

# Remove duplicates in cik
not_sp = not_sp.drop_duplicates(subset = 'cik', keep = 'first')
    
cik_ls = not_sp['cik'].astype(str).tolist()
cik_full_ls = not_sp['cik_full'].tolist()
ticker_ls = not_sp['ticker'].tolist()

# Create a csv to store path
with open(path + os.sep + 'f10k_path_not_sp500.csv', 'w', newline = '') as output:
    writer = csv.writer(output)
    writer.writerow(['cik','cik_full','ticker','accn','form','fy', '10k_path'])

# Create a list to store problematic companies
error = []  # For checking errors

# -----------------------------------------------------------------------------
# !! # Problematic companines (no 10-K): 
# - S&P 500:
#    First Republic Bank, FRC, 1132979
#    Signature Bank Corp, SBNY, 1288784
#           It seems that banks do not file 10-K, instead they file 13G
#    Organon & Co., OGN, 1821825
#    Constellation Energy Corp, CEG, 1868275
# - ouside S&P 500:
#
# -----------------------------------------------------------------------------
# Set headers (Required by SEC)
headers = {'User-Agent':'<Your email>'}

for i in range(0, len(cik_ls)):
    try:
        # Go to company facts, get the accession number of the latest 10-K
        cik = cik_ls[i]
        cik_full = cik_full_ls[i]
        ticker = ticker_ls[i]
        companyfacts_path = 'https://data.sec.gov/api/xbrl/companyfacts/CIK' \
            + cik_full + '.json'
    
        # Get response
        r_facts = requests.get(companyfacts_path, headers = headers, timeout = 3)
        
        # Get json and flatten it into a df
        t_json = r_facts.json()
        
        # json_o = json.dumps(t_json['facts'], indent = 4)
        
        df = pd.json_normalize(t_json['facts'])
        
        # Drop None values
        df = df.dropna(axis = 1)
        
        # Start from the last column, search the list of filings for 10-K
        for i in range(1, len(df.columns)+1):
            filing_ls = df[df.columns[-i]][0]
            
            if type(filing_ls) == str:
                continue
            
            # Search from the last entry
            try:
                for counter in range(1, len(filing_ls)+1):
                    if (filing_ls[-counter]['form'] == '10-K') & \
                        (filing_ls[-counter]['fy'] > 2019):
                        accn = filing_ls[-counter]['accn']
                        form = filing_ls[-counter]['form']
                        fy = filing_ls[-counter]['fy']   # Record accn, form, fy for future check
                        break
                    else:
                        continue
            except:
                continue
    
        # Get 10-K in .txt format written in SGML
        accn_cont = accn.replace('-','')
    
        f10k_path = 'https://www.sec.gov/Archives/edgar/data/' + cik + '/' + accn_cont \
            + '/' + accn + '.txt'
        
        with open(path + os.sep + 'f10k_path_not_sp500.csv', 'a', newline = '') as output:
            writer = csv.writer(output)
            writer.writerow([cik, cik_full, ticker,accn, form, fy, f10k_path])
        
        time.sleep(uniform(1,1.3)) 
        
    except:
        error.append((ticker, companyfacts_path))
        print(ticker)
        time.sleep(uniform(1,1.3)) 
        continue
