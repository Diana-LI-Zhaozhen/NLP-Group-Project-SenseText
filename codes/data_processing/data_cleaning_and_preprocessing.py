# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:45:58 2022

@author: yanyu
"""

import os
import pickle

path = r'C:/Users/zhaoz/Desktop/Text Analysis and NLP/Group Project SenseText/datasets'

f = open(path + os.sep + 'cleaned_10k_items.pkl','rb')

cleaned_10k_items = pickle.load(f)

#%%
#lower case
def lower_case(target_list):
    filtered_words = []
    for i in target_list:
        tem = str.lower(i)
        filtered_words.append(tem)
    return filtered_words

lower_b_words = lower_case(cleaned_10k_items['business'])
lower_rf_words = lower_case(cleaned_10k_items['risk_factors'])

#%%
#tokenization
#stop words removal

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stop_words(target_string_list):
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for i in target_string_list:
        word_tokens = word_tokenize(i)
        word_list = []
        for w in word_tokens:
            if w not in stop_words:
                word_list.append(w)
        filtered_words.append(word_list)
    return filtered_words

no_stop_b_words = remove_stop_words(lower_b_words)
no_stop_rf_words = remove_stop_words(lower_rf_words)

#%%
#word reduction
#Drop symble and numbers, and single character    
import re

def word_reduction(target_list):
    filtered_words = []   
    for i in target_list:
        com_words = []
        for j in i:
            if j.isalpha():
                com_words.append(j)
            else:
                continue
        filtered_words.append(com_words)
    return filtered_words

f_b_words = word_reduction(no_stop_b_words)
f_rf_words = word_reduction(no_stop_rf_words)

#%%
#get word class and finish Lemmatization 
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wnl = WordNetLemmatizer()

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

def lemma(target_b_list, target_rf_list):
    lem_b_words = []
    lem_rf_words = []
    for com in target_b_list:
        com_lem = []
        for word in com:
            if b_get_wordnet_pos(pos_tag([word])[0][1]) == None:
                continue
            else:
                tem = wnl.lemmatize(word, \
                                    b_get_wordnet_pos(pos_tag([word])[0][1]))
                com_lem.append(tem)
        lem_b_words.append(com_lem)
    for com in target_rf_list:
        com_lem = []
        for word in com:
            if rf_get_wordnet_pos(pos_tag([word])[0][1]) == None:
                continue
            else:
               tem = wnl.lemmatize(word, \
                                   rf_get_wordnet_pos(pos_tag([word])[0][1]))
               com_lem.append(tem)
        lem_rf_words.append(com_lem)
    return lem_b_words, lem_rf_words

lem_words = lemma(f_b_words, f_rf_words)

lem_words = open(path + os.sep + 'lem_words.pkl','rb')
lem_words = pickle.load(lem_words)

#count the apparence frequency of all the noun 
from collections import Counter        
b_words_freq = Counter(sum(lem_words[0], []))
rf_words_freq = Counter(sum(lem_words[1], []))

#remove 
def remove_key(d):
    for key in list(d.keys()):
        if (key.isalpha() == False) or (len(key)<2):
            d.pop(key)
        if (d[key] > 20000) or (d[key] < 1200):
            del d[key]

remove_key(b_words_freq)
remove_key(rf_words_freq)

b_words = list(b_words_freq.keys())
rf_words = list(rf_words_freq.keys())

#save the output as pickle
import dump

with open(path + os.sep + 'b_words.pkl', 'wb') as f1:  # Python 3: open(..., 'wb')
    pickle.dump(b_words, f1)
with open(path + os.sep + 'rf_words.pkl', 'wb') as f2:  # Python 3: open(..., 'wb')
    pickle.dump(rf_words , f2)
with open(path + 'lem_b_words.pkl', 'wb') as f3:   
    pickle.dump(lem_b_words , f3)
with open(path + os.sep + 'lem_rf_words.pkl', 'wb') as f4:   
    pickle.dump(lem_rf_words , f4)
with open(path + os.sep + 'no_stop_b_words.pkl','wb') as f5:  # Python 3: open(..., 'wb')
    pickle.dump(no_stop_b_words, f5)  
with open(path + os.sep + 'no_stop_rf_words.pkl','wb') as f6:  # Python 3: open(..., 'wb')
    pickle.dump(no_stop_rf_words, f6)
