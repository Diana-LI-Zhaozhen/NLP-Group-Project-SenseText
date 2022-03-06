#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Clean and Merge

Created on Fri Feb 25 12:59:19 2022

@author: shiqimeng
"""
import pandas as pd
import os
import pickle

path = '/Users/shiqimeng/Downloads/NLP projects/10k_items'

not_sp500_raw = pd.read_csv(path + os.sep + '10k_info_not_sp500.csv')
sp500_raw = pd.read_csv(path + os.sep + '10k_info_sp500.csv')

# Concatenate the two df
merged_df = pd.concat([sp500_raw,not_sp500_raw])

# Drop any company containing NA
no_na = merged_df.dropna(axis = 0, how = 'any')

# Drop any company with insufficient text info (less than 1000 characters)
busi_clean = no_na.loc[no_na['bsusiness'].str.len() > 1000]  
    # Note there is a typo in the column name 'business' 
busi_risk_clean = busi_clean.loc[busi_clean['risk_factors'].str.len() > 1000]
clean = busi_risk_clean.loc[busi_risk_clean['MDnA'].str.len() > 1000]

# Check if all items are over 1000 characters
assert all(clean['bsusiness'].str.len() > 1000) & \
    all(clean['risk_factors'].str.len() > 1000) & \
        all(clean['MDnA'].str.len() > 1000), \
            'ERROR: Not all items are over 1000 characters.'

# Correct the typo in 'business' column
final_df = clean.rename(columns = {'bsusiness':'business'})

# Reset index
final_df = final_df.reset_index(drop = True)

# Save output
final_df.to_csv(path + os.sep + '10k_info_cleaned.csv')

with open(path + os.sep + 'cleaned_10k_items.pickle', 'wb') as f:
    pickle.dump(final_df, f, pickle.HIGHEST_PROTOCOL)

# Create a subset with only the business section
busi_only = final_df[['ticker','business']]
busi_only.to_csv(path + os.sep + 'business_info_cleaned.csv')

with open(path + os.sep + 'cleaned_business_only.pickle', 'wb') as f:
    pickle.dump(busi_only, f, pickle.HIGHEST_PROTOCOL)
