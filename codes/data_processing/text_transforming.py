# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:57:19 2022

@author: silvia
"""
import pickle
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet

path = r'C:\Users\zhaoz\Desktop\Text Analysis and NLP\NLP-Group-Project-SenseText-master\datasets'
lem_words = open(path + os.sep + 'lem_words.pkl','rb')
vb = open(path + os.sep + 'b_words.pkl','rb')
vr = open(path + os.sep + 'rf_words.pkl','rb')
tk = open(path + os.sep + 'cleaned_10k_items.pickle','rb')
#load data and cleaned words frequency
lem_words = pickle.load(lem_words)
b_lemma = lem_words[0]
r_lemma = lem_words[1]
b_vector = pickle.load(vb)
rf_vector = pickle.load(vr)
cleaned_10k_items = pickle.load(tk)

#%%
#sentimental analysis
from textblob import TextBlob
sentiment = [TextBlob(mes) for mes in cleaned_10k_items['MDnA']]
polar = [blob.polarity for blob in sentiment]
subject = [blob.subjectivity for blob in sentiment]

#%%
#To compute the binary vector and the tf value vector
def binary(count,freq):
    vector = []
    tf_idf = []
    for i in list(freq):
        num = count.get(i,0)
        if num != 0:
            vector.append(1)
            tf_idf.append(num)
        else:
            vector.append(0)
            tf_idf.append(0)
    return vector,tf_idf

#get the binary vector for business
b_count = [Counter(t) for t in b_lemma]
b_temp = [binary(dict(b) ,b_vector) for b in b_count]
b_binary = list(pd.DataFrame(b_temp)[:][0]) #the binary vector for business
b_t_temp = pd.DataFrame(b_temp)[:][1]
b_idf = np.log(2965/np.sum(np.array(list(b_binary)),axis = 0)) #compute idf value
b_tfidf = [list(np.array(a) * b_idf) for a in b_t_temp]#the tf-idf vector for business

#%%
#get the binary vector for risk factor
r_count = [Counter(t) for t in r_lemma]
r_temp = [binary(dict(r) ,rf_vector) for r in r_count]
r_binary = list(pd.DataFrame(r_temp)[:][0])#the binary vector for risk factor
r_t_temp = pd.DataFrame(r_temp)[:][1]
r_idf = np.log(2965/np.sum(np.array(list(r_binary)),axis = 0))
r_tfidf = [list(np.array(a) * r_idf) for a in r_t_temp]#the tf-idf vector for risk factor

#%%
data =pd.DataFrame()
data['ticker'] = cleaned_10k_items['ticker']
data['polarity'] = polar
data['subjectivity'] = subject
data['b_binary'] = b_binary
data['b_tfidf'] = b_tfidf
data['rf_binary'] = r_binary
data['rf_tfidf'] = r_tfidf
import dump

with open(path + os.sep + 'text_analysis.pkl', 'wb') as f1:  # Python 3: open(..., 'wb')
    pickle.dump(data, f1)