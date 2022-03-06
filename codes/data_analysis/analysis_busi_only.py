#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis

Created on Thu Mar  3 23:01:24 2022

@author: shiqimeng
"""
import pandas as pd
import os
import pickle
from gensim.corpora.dictionary import Dictionary
from gensim.models.lsimodel import LsiModel
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import scipy.sparse
import logging

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',\
                    level = logging.INFO)

# --------- Set input and output path --------------

path = '/Users/shiqimeng/Desktop/MFIN7036 NLP/mfin7036_midterm/datasets'
save_path = '/Users/shiqimeng/Desktop/MFIN7036 NLP/mfin7036_midterm/outputs'

# Load matrix
mm = scipy.sparse.load_npz(save_path + os.sep +'mm.npz')
# Load dict
d = Dictionary.load_from_text(save_path + os.sep + 'dictionary.txt')

# Dimentionality reduction (LSA-SVD)
svd = TruncatedSVD(n_components = 300, n_iter = 7, random_state = 5)
svd.fit(mm.T)

# Get V
V = svd.fit_transform(mm.T)

# Check topics with gensim
id2word = dict()
for key,value in d.token2id.items():  # Create id2word dict
    id2word[value] = key
lsi = LsiModel(corpus = mm,id2word = id2word,num_topics = 300)
lsi.print_topics(20)

# K-means
kmeans = KMeans(n_clusters = 300, random_state = 0).fit(V)
labels = pd.DataFrame(kmeans.labels_.tolist(),columns = ['label'])

# Get index, company pairs
with open(path + os.sep + "cleaned_business_only.pkl", "rb" ) as f:
    busi = pickle.load(f)

company = busi[['ticker']]

cluster = company.join(labels)
groups = cluster.groupby('label')['ticker'].apply(list).reset_index()

groups.to_csv(save_path + os.sep + 'groups.csv')

# Load GICS sub-industry data (size: 1,906)
gics = pd.read_csv(path + os.sep + 'gics.csv', dtype = str)

# Create a subset of cluster
gics_tic = gics['tic'].to_list()
cluster_s = cluster.set_index('ticker').loc[gics_tic].reset_index()  

# Check overlap between the two classification systems
kmeans_dict = dict(zip(cluster_s['ticker'],cluster_s['label']))
gics_dict = dict(zip(gics['tic'], gics['gsubind']))

same_count = 0
for a_tic, a_label in kmeans_dict.items():
    for b_tic, b_label in kmeans_dict.items():
        if (a_tic != b_tic) & (a_label == b_label):  # different comps in the same 10k industry
            if gics_dict[a_tic] == gics_dict[b_tic]:  # also in the same gics industry
                same_count += 1

# Overlap proportion
overlap = same_count / (len(kmeans_dict) * (len(kmeans_dict)-1)) * 100   # 0.15275483319620856

# Find firms with below-average PE in each 10k group
# Load PE dataset
with open(path + os.sep + "pe_only.pkl", "rb" ) as f:
    pe = pickle.load(f)

# Inner join pe and cluster by ticker
val = pd.merge(cluster, pe, on = 'ticker')

# Compute mean and std
pe_mean = val.groupby('label')['pe'].mean().reset_index().rename(columns={'pe':'mean'})
pe_std = val.groupby('label')['pe'].std().reset_index().rename(columns={'pe':'std'})

val_mean = pd.merge(val, pe_mean, how = 'left', on = 'label')
val_mv = pd.merge(val_mean, pe_std, how = 'left', on = 'label')

# Drop NA
val_mv = val_mv.dropna()

# Find company with PE below 1 std
low_pe = val_mv.loc[val_mv['pe'] < val_mv['mean'] - val_mv['std']]
# Get the list of firms
low_pe_co = low_pe['ticker'].to_list()
# Save as txt
with open(path + os.sep + 'low_pe_ticker.txt', 'w') as f:   # Input to CRSP
    for item in low_pe_co:
        f.write('{}\n'.format(item))
