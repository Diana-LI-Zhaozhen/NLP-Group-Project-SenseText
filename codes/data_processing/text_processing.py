#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:57:19 2022

@author: silvia
"""
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet

path = '/Users/silvia/Desktop/NLP/'
f = open(path + 'cleaned_10k_items.pickle','rb')
vb = open(path + 'b_words_frequency.pkl','rb')
vr = open(path + 'rf_words_frequency.pkl','rb')
#load data and cleaned words frequency
cleaned_10k_items = pickle.load(f)
b_vector = pickle.load(vb)
rf_vector = pickle.load(vr)

wnl = WordNetLemmatizer()
#To remove stop words and non alphabetic tokens 
def non_stop(word_list):
    word_list = [k for k in word_list if k.isalpha()]
    return [i for i in word_list if (i not in stopwords.words('english')) and len(i)>=2]

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
#To lemmatize words
def b_get_wordnet_pos(tag):
    if tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None
    
def rf_get_wordnet_pos(tag):
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemma_b(target_b_list):
    lem_b_words = []
    for word in target_b_list:
        if b_get_wordnet_pos(pos_tag([word])[0][1]) == None:
            continue
        else:
            tem = wnl.lemmatize(word,
                                b_get_wordnet_pos(pos_tag([word])[0][1]))
            lem_b_words.append(tem)
    return lem_b_words   

def lemma_r(target_rf_list):
    lem_rf_words = []
    for word in target_rf_list:
        if rf_get_wordnet_pos(pos_tag([word])[0][1]) == None:
            continue
        else:
           tem = wnl.lemmatize(word,
                               rf_get_wordnet_pos(pos_tag([word])[0][1]))
           lem_rf_words.append(tem)
    return lem_rf_words
#%%
#sentimental analysis
from textblob import TextBlob
sentiment = [TextBlob(mes) for mes in cleaned_10k_items['MDnA']]
polar = [blob.polarity for blob in sentiment]
subject = [blob.subjectivity for blob in sentiment]
#%%
#get the binary vector for business
b_tokens = [word_tokenize(word.lower()) for word in cleaned_10k_items['business']]
b_nonstops = [non_stop(token) for token in b_tokens]
b_lemma = [lemma_b(t) for t in b_nonstops]
b_count = [Counter(t) for t in b_lemma]
b_temp = [binary(dict(b) ,b_vector) for b in b_count]
b_binary = list(pd.DataFrame(b_temp)[:][0]) #the binary vector for business
b_t_temp = pd.DataFrame(b_temp)[:][1]
b_idf = np.log(2965/np.sum(np.array(list(b_binary)),axis = 0)) #compute idf value
b_tfidf = [list(np.array(a) * b_idf) for a in b_t_temp]#the tf-idf vector for business

#%%
#get the binary vector for risk factor
r_tokens = [word_tokenize(word.lower()) for word in cleaned_10k_items['risk_factors']]
r_nonstops = [non_stop(token) for token in r_tokens]
r_lemma = [lemma_r(t) for t in r_nonstops]
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

with open(path + 'text_analysis.pkl', 'wb') as f1:  # Python 3: open(..., 'wb')
    pickle.dump(data, f1)