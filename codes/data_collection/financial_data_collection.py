#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Data Collection
Date: Mar 4, 2022

Created on Wed Mar  2 13:08:21 2022

@author: shiqimeng
"""
from joblib import Parallel, delayed
import os, csv
import yfinance as yf
import pickle
from tqdm import tqdm

path = '/Users/shiqimeng/Downloads/NLP projects'

with open(path + os.sep + "company_list.pkl", "rb" ) as f:
    comp_ls = pickle.load(f)

with open(path + os.sep + 'financial_dataset.csv', 'w', newline = '') as output:
    writer = csv.writer(output)
    writer.writerow(['ticker','beta', 'trailingPE', 'priceToSalesTrailing12Months',\
                      'priceToBook', 'trailingPegRatio', 'debtToEquity', 'returnOnEquity'])

with open(path + os.sep + 'pe_only.csv', 'w', newline = '') as output:
    writer = csv.writer(output)
    writer.writerow(['ticker','pe'])

def data(i):
    comp = yf.Ticker(i) 
    try:
        beta = comp.info['beta']
    except:
        beta = ''
    try:
        pe = comp.info['trailingPE']
    except:
        pe = ''
    try:
        ps = comp.info['priceToSalesTrailing12Months']
    except:
        ps = ''
    try:
        pb = comp.info['priceToBook']
    except:
        pb = ''
    try:
        peg = comp.info['trailingPegRatio']
    except:
        peg = ''
    try:
        de = comp.info['debtToEquity']
    except:
        de = ''
    try:
        roe = comp.info['returnOnEquity']
    except:
        roe = ''  
    
    comp_info = [i, beta, pe, ps, pb, peg, de, roe]
    pe_only = [i, pe]
    
    with open(path + os.sep + 'financial_dataset.csv', 'a', newline = '') as output:
        writer = csv.writer(output)
        writer.writerow(comp_info)
    
    with open(path + os.sep + 'pe_only.csv', 'a', newline = '') as output:
        writer = csv.writer(output)
        writer.writerow(pe_only)

# Use joblib for multiprocessing
Parallel(n_jobs = 5)(delayed(data)(i) for i in tqdm(comp_ls[2351:]))

# Clean datasets
import pandas as pd
fin = pd.read_csv(path + os.sep + 'financial_dataset.csv')
pe = pd.read_csv(path + os.sep + 'pe_only.csv')

# Remove duplicates
fin = fin.drop_duplicates(subset=['ticker'])
pe = pe.drop_duplicates(subset=['ticker'])
# Remove empty values
pe = pe.dropna()

with open(path + os.sep + 'pe_only.pkl', 'wb') as f:
    pickle.dump(pe, f, pickle.HIGHEST_PROTOCOL)
