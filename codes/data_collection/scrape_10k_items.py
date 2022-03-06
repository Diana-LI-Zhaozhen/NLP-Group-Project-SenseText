#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape Item 1, 1A, 7 from 10-K

Created on Mon Feb 21 17:19:56 2022

@author: shiqimeng
"""
# ------
# The following codes are written with reference to
# https://gist.github.com/anshoomehra/ead8925ea291e233a5aa2dcaa2dc61b2
# ------

# Install lxml parser
# pip install lxml

# Import modules
from bs4 import BeautifulSoup
import re   # for REGEXes
import os, csv
import requests
import pandas as pd
import time
from random import uniform

path = '/Users/shiqimeng/Downloads/NLP projects'

# Write the output .csv
with open(path + os.sep + '10k_info_not_sp500.csv', 'w', newline = '') as output:
    writer = csv.writer(output)
    writer.writerow(['ticker','business','risk_factors','MDnA'])
    
# Import the 10k path file
f10k_df = pd.read_csv(path + os.sep + 'f10k_path_not_sp500.csv')

# Zip ticker and path into a dictionary
dict10k = dict(zip(f10k_df['ticker'],f10k_df['10k_path']))

# Error log
error_ls = []

for ticker, f10k_path in dict10k.items():
    try:
        # Get 10-K in HTML format
        headers = {'User-Agent':'iris17m@connect.hku.hk'}  # Set header
        r = requests.get(f10k_path, headers = headers)
        raw_10k = r.text
        
        # Print to check
        # print(raw_10k[0:1300])
        
        # Apply REGEXes to find items
        # ------
        # All docs are included within the <DOCUMENT> and </DOCUMENT> tags.
        # Each doc type is clearly marked by a <TYPE> tag.
        # ------
        
        # Regex to find <DOCUMENT> tags
        doc_start_pattern = re.compile(r'<DOCUMENT>')
        doc_end_pattern = re.compile(r'</DOCUMENT>')
        
        # Regex to find <TYPE> tag prceeding any characters
        # [^\n]: not followed by a new line
        # +: Matches the previous element one or more times
        type_pattern = re.compile(r'<TYPE>[^\n]+')
        
        # Find the tag index
        # Create 3 lists
        # 1. A list that holds the .end() index of each match of doc_start_pattern
        # 2. A list that holds the .start() index of each match of doc_end_pattern
        # 3. A list that holds the type of docs from each match of type_pattern
        
        doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
        doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]
        
        doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]
        
        # Get the 10-K document
        document = {}
        
        # Create a loop to go through each doc type and save only the 10-K section in the dictionary
        for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
            if doc_type == '10-K':
                document[doc_type] = raw_10k[doc_start:doc_end]
                
        # Check the content
        # document['10-K'][0:500]
        
        # Find the items
        # Write the regex
        # \s white space, | or
        # .{0,1} matches any single character after the item number
        regex = re.compile(r'(>(Item|ITEM)(\s|&#160;|&nbsp;|&#xA0;|&#160;\s)(1|1A|1a|1B|1b|7|7A|7a).{0,1})')
        # (r'(>Item(\s|&#160;|&nbsp;)(1|1A|1a|1B|1b|7|7A|7a).{0,1})|(ITEM\s(1|1A|1a|1B|1b|7|7A|7a))')
        
        # Use finditer to match the regex
        # finditer can only ** loop once **
        matches = regex.finditer(document['10-K'])
        
        # Write a for loop to check the matches
        # for match in matches:
        #     print(match)
        
        # Create a dataframe to store the position
        position_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])
        
        position_df.columns = ['item', 'start', 'end']
        position_df['item'] = position_df.item.str.lower()
        
        # Check the dataframe
        # position_df.head()
        
        # Get rid of unnesesary charcters from the dataframe
        position_df.replace('&#160;',' ', regex = True, inplace = True)
        position_df.replace('&nbsp;',' ', regex = True, inplace = True)
        position_df.replace('&#xA0;',' ', regex = True, inplace = True)
        position_df.replace('\.','', regex = True, inplace = True)
        position_df.replace('>','', regex = True, inplace = True)
        position_df.replace('<','', regex = True, inplace = True)
        position_df.replace(' ','', regex = True, inplace = True)
        
        # Drop duplicates
        condition = (position_df['item'] == 'item1') | \
                    (position_df['item'] == 'item1a') | \
                    (position_df['item'] == 'item1b') |\
                    (position_df['item'] == 'item7') |\
                    (position_df['item'] == 'item7a')
                    
        position_df = position_df.loc[condition]
        
        pos_df = position_df.sort_values('start', ascending = True)
        
        # Remove those items in the table of contents
        if (len(pos_df['item']) > 5) & (pos_df['item'][4] == 'item7a'):
            pos_df = pos_df.iloc[5:,:]
        
        # Remove any row before item1
        # for i in range(len(pos_df['item'])):
        #     if pos_df['item'][i] != 'item1':
        #         continue
        #     else:
        #         break
        
        # pos_df = pos_df.iloc[i:,:]
        
        pos_df = pos_df.drop_duplicates(subset = 'item', keep = 'first')
        
        
        # Set item as the dataframe index
        pos_df.set_index('item', inplace = True)
        
        # Save text for each section
        # Get Item 1
        item_1_raw = document['10-K'][pos_df['start'].loc['item1']:pos_df['start'].loc['item1a']]
        
        # Get Item 1a
        item_1a_raw = document['10-K'][pos_df['start'].loc['item1a']:pos_df['start'].loc['item1b']]
        
        # Get Item 7
        item_7_raw = document['10-K'][pos_df['start'].loc['item7']:pos_df['start'].loc['item7a']]
        
        # Check
        # item_1_raw[1:1000]
        
        # Apply bs to refine the result
        ### First convert the raw text BeautifulSoup object 
        item_1_content = BeautifulSoup(item_1_raw, 'lxml')
        item_1a_content = BeautifulSoup(item_1a_raw, 'lxml')
        item_7_content = BeautifulSoup(item_7_raw, 'lxml')
        
        # Check
        # print(item_1a_content.prettify()[0:1000])
        
        ### Get text
        item_1 = item_1_content.get_text()
        item_1a = item_1a_content.get_text()
        item_7 = item_7_content.get_text()
        
        ## Remove \n
        item1 = item_1.replace('\n', ' ')
        item1a = item_1a.replace('\n', ' ')
        item7 = item_7.replace('\n', ' ')
        
        # Write output into csv
        with open(path + os.sep + '10k_info_not_sp500.csv', 'a', newline = '') as output:
             writer = csv.writer(output)
             writer.writerow([ticker, item1, item1a, item7])
        
        time.sleep(uniform(1,1.3)) 
        
    except:
        error_ls.append(ticker)
        print(ticker)
        time.sleep(uniform(1,1.3)) 
        continue
