# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:45:58 2022

@author: yanyu
"""

import pickle
f = open(r'C:/Users/zhaoz/Desktop/Text Analysis and NLP/group project/cleaned_10k_items.pkl','rb')

cleaned_10k_items = pickle.load(f)

#tokenization
#stop words removal
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#word reduction
#Drop symble and numbers, and single character    
import re
def word_reduction(target_list):
    filtered_words = []   
    for i in target_list:
        str_count = 0
        for j in i:
            if j.isalnum():
                str_count += 1
        m = re.search(r'[a-zA-Z]', i)
        if str_count == 1:
            continue
        elif m == None:
            continue
        elif ("None" in i) or ("none" in i):
            continue 
        else:
            filtered_words.append(i)
    return filtered_words

def remove_stop_words(target_string_list):
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for i in target_string_list:
        word_tokens = word_tokenize(i)
        for w in word_tokens:
            if w not in stop_words:
                filtered_words.append(w)
    return filtered_words

f_b_words = word_reduction(cleaned_10k_items['business'])
f_rf_words = word_reduction(cleaned_10k_items['risk_factors'])

no_stop_b_words = remove_stop_words(f_b_words)
no_stop_rf_words = remove_stop_words(f_rf_words)

#lower case
def lower_case(target_list):
    filtered_words = []
    for i in target_list:
        tem = str.lower(i)
        filtered_words.append(tem)
    return filtered_words

lower_b_words = lower_case(no_stop_b_words)
lower_rf_words = lower_case(no_stop_rf_words)

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
    for word in target_b_list:
        if b_get_wordnet_pos(pos_tag([word])[0][1]) == None:
            continue
        else:
            tem = wnl.lemmatize(word,
                                b_get_wordnet_pos(pos_tag([word])[0][1]))
            lem_b_words.append(tem)
    for word in target_rf_list:
        if rf_get_wordnet_pos(pos_tag([word])[0][1]) == None:
            continue
        else:
           tem = wnl.lemmatize(word,
                               rf_get_wordnet_pos(pos_tag([word])[0][1]))
           lem_rf_words.append(tem)
    
    return lem_b_words, lem_rf_words

lem_words = lemma(lower_b_words, lower_rf_words)

#count the apparence frequency of all the noun 
from collections import Counter        
b_words_freq = Counter(lem_words[0])
rf_words_freq = Counter(lem_words[1])

def remove_key(d):
    for key in list(d.keys()):
        if (key.isalpha() == False) or (len(key)<2):
            d.pop(key)
        if (d[key] > 20000) or (d[key] <1200):
            del d[key]

remove_key(b_words_freq)
remove_key(rf_words_freq)

import dump

with open('C:/Users/zhaoz/Desktop/Text Analysis and NLP/group project/b_words_frequency.pkl', 'wb') as f1:  # Python 3: open(..., 'wb')
    pickle.dump(b_words_freq, f1)
with open('C:/Users/zhaoz/Desktop/Text Analysis and NLP/group project/rf_words_frequency.pkl', 'wb') as f2:  # Python 3: open(..., 'wb')
    pickle.dump(rf_words_freq , f2)
with open('C:/Users/zhaoz/Desktop/Text Analysis and NLP/group project/lem_b_words.pkl', 'wb') as f3:   
    pickle.dump(b_words_freq , f3)
with open('C:/Users/zhaoz/Desktop/Text Analysis and NLP/group project/lem_rf_words.pkl', 'wb') as f4:   
    pickle.dump(rf_words_freq , f4)
#f.close()
