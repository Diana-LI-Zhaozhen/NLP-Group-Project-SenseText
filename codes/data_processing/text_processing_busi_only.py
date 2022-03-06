#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Processing

Created on Thu Mar  3 17:27:51 2022

@author: shiqimeng
"""
# ---------- Import packages -------------

import pandas as pd
import os
import pickle

# For text cleaning, preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# For text transformation
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import gensim.matutils
import scipy.sparse

# --------- Set input and output path --------------

path = '/Users/shiqimeng/Desktop/MFIN7036 NLP/mfin7036_midterm/datasets'
save_path = '/Users/shiqimeng/Desktop/MFIN7036 NLP/mfin7036_midterm/outputs'

# --------- Define text cleaning functions ----------

def noCamel(tcp):
    ''' 
    Remove camel case strings
        To remove sth like "BusinessOur". 
        This type of strings occurs when section title and the 1st word of a 
             sentence are not separated by a space.
        tcp: tokenized corpus, a list of list of word tokens
        return: a list of list of clean word tokens
    '''
    tcp_copy = tcp.copy()   # Make a copy
    for doc in tcp_copy:
        doc_copy = doc.copy()       # Make a copy
        for tk in doc_copy:
            if len(re.findall(r'[A-Z]',tk)) > 1:
                doc.remove(tk)
    return tcp_copy            
    
def noUpperLower(tcp):
    ''' 
    Remove mixed upper and lower case strings
        To remove sth like "BUSINESSOverview". 
        This type of strings occurs when capitalized titles are not separated 
            from the next word.  
        tcp: tokenized corpus, a list of list of word tokens
        return: a list of list of clean word tokens
    '''
    tcp_copy = tcp.copy()       # Make a copy
    for doc in tcp_copy:
        doc_copy = doc.copy()       # Make a copy
        for tk in doc_copy:
            if (sum(1 for c in tk if c.isupper()) > 1) & (sum(1 for c in tk if c.islower()) > 1):
                doc.remove(tk)
    return tcp_copy 

def noStopWd(tcp, stop_words):
    ''' 
    Remove stop words
        tcp: tokenized corpus, a list of list of word tokens
        stop_words: a list of stop words, all lowercase
        return: a list of list of clean word tokens
    '''
    # Check whether the stop words are all lowercase
    for i in stop_words:
        assert i.islower(), '{} in stop word list is not lowercase.'.format(i)

    filtered = [[w for w in doc if not w.lower() in stop_words] for doc in tcp]
    return filtered
    
def onlyWords(tcp):
    ''' 
    Remove numbers, punctuations, and special characters including Chinese
        Words include those containing digits like 5g.
        tcp: tokenized corpus, a list of list of word tokens
        return: a list of list of clean word tokens
    '''
    tcp_copy = tcp.copy()
    for doc in tcp_copy:
        doc_copy = doc.copy()
        for w in doc_copy:
            if ((w.isnumeric()) | (not w.isalnum()) |\
                (len(re.findall(r'[a-zA-Z]+',w)) == 0) |\
                    ('ϒ' in w)):    # A special case
                doc.remove(w)
    return tcp_copy    
    
def Lower(tcp):
    ''' 
    Lowercase conversion
        tcp: tokenized corpus, a list of list of word tokens
        return: a list of list of lowercase word tokens
    '''
    lower = [[w.lower() for w in doc] for doc in tcp]
    return lower

def Lem(tcp):
    ''' 
    Lemmatization
        tcp: tokenized corpus, a list of list of lowercase word tokens
        return: a list of list of lemmas of words
    '''
    # Assume the inputs are all lowercase
    for i in tcp:
        for j in i:
            assert j.islower(), \
                'Tokens are not all lowercase. Check {} in {}.'.format(j,i)
            
    wnl = WordNetLemmatizer()
    lem = [[wnl.lemmatize(w) for w in doc] for doc in tcp]
    return lem

def noDup(tcp):
    ''' 
    Remove duplicates
        tcp: tokenized corpus, a list of list of word tokens
        return: a list of list of word tokens
    '''  
    return [list(set(i)) for i in tcp]

def ReduceSyns(tcp):
    ''' 
    Remove synonyms
        tcp: tokenized corpus, a list of list of word tokens
        return: a list of list of unique word tokens
    '''  
    unique = []
    tcp_copy = tcp.copy()
    for doc in tcp_copy:
        checked = []
        for wd in doc:
            if wd in checked:   # Skip wd already checked
                continue
            else:
                syns = []
                checked.append(wd)
                for s in wordnet.synsets(wd):
                    for lem in s.lemmas():
                        syns.append(lem.name())    # Get all syns
                for idx, item in enumerate(doc):
                    if item in syns:
                        doc[idx] = wd       # Substitute all synonyms
        unique.append(list(set(doc)))
    return unique

def GetPoS(cp, pos_list):
    ''' 
    Get target PoS tokens
        cp: text corpus, a list of documents
        pos_list = list of pos tags, str
        return: a list of list of target PoS tokens
    '''  
    for i in pos_list:
        assert i in ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ',\
                     '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$',\
                         'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS',\
                             'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', \
                                 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS'],\
            'PoS tag not valid.'

    ner = []       # List to store tokens in corpus level
    for doc in cp:
        sent = sent_tokenize(doc)  # Split doc into a list of sentence
        selected_pos = []          # List to store tokens in doc level
        for i in sent:
            token = word_tokenize(i)
            tag = nltk.pos_tag(token)
            selected_pos += [i[0] for i in tag if i[1] in pos_list]
        ner.append(selected_pos)
    return ner

# -------------------- Main codes ---------------------------

if __name__ == '__main__':
    
    # Load data
    with open( path + os.sep + "cleaned_business_only.pkl", "rb" ) as f:
        busi = pickle.load(f)
    
    # Remove >(ITEM 1.|Item 1.)\s+(Business|BUSINESS) at the beginning of the str
    no_title = []
    for i in busi['business'].to_list():
        out = re.sub(r'>(ITEM|Item)(\s+|)1.(\s+|)(Business|BUSINESS|)', '',i)
        no_title.append(out)
    
    # ----------------- Tokenization ---------------
    
    tk_cp = [word_tokenize(i) for i in no_title]
    
    # -------------------------- Cleaning --------------------------
    
    tk_cp_1 = noCamel(tk_cp)         # Remove camel case 
    tk_cp_2 = noUpperLower(tk_cp_1)  # Remove mixed case
    
    # Stop word removal 
    common_words = ['overview','description','descriptions','company','companies','business',\
                    'businesses','product','products','firm','firms','inc.','corporation','llc',\
                        'ltd.','l.c.','l.l.c.','lc','co.','incorporated','corp.','background','us',\
                            'year','quarter','january','february','march','april','may','june','july',\
                                'august','september','october','november','december','fiscal','period']
    stop_words = set(stopwords.words('english') + common_words)
    tk_cp_3 = noStopWd(tk_cp_2, stop_words)
    
    tk_cp_4 = onlyWords(tk_cp_3)    # Remove numbers, punctuations, and special characters
    tk_cp_low = Lower(tk_cp_4)      # Lower case conversion 
    tk_cp_lem = Lem(tk_cp_low)      # Lemmatization
    tk_cp_nodup = noDup(tk_cp_lem)  # Remove duplicates 
    unique = ReduceSyns(tk_cp_nodup) # Reduce synonyms
    
    # Perform NER at sentence level
    # Get NN noun, singular; NNP proper noun, singular; JJ adjective 'wearable'; 
    # CD cardinal digit '5g'; VB verb
    pos = ['NN','CD','NNP','VB','JJ'] 
    ner = GetPoS(no_title, pos)
    
    ner_low = Lower(ner)         # Lower case conversion 
    ner_w = onlyWords(ner_low)   # Remove numbers, punctuations, and special characters
    ner_lem = Lem(ner_w)         # Lemmatization     # Error: 'ifnϒ'
    ner_nodup = noDup(ner_lem)   # Remove duplicates
            
    # Intersect unique and ner_nodup -> to get desired pos in unique
    ner_cleaned = [list(set(x) & set(y)) for x, y in zip(ner_nodup, unique)]
    
    # Remove words less than 2 char
    tk_clean = [[w for w in doc if len(w) > 1] for doc in ner_cleaned]
    
    # Remove words like 2the
    tk_clean_2 = [[w for w in doc if not ((len(re.findall(r'\d',w)) > 0) and \
                                           (len(re.findall(r'[a-z]',w)) > 1))] for doc in tk_clean]
    
    # Remove day of the week, frequency measure like 'quarterly'
    del_words = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','quarterly','yearly',\
                 'weekly','daily','monthly']
    cp = noStopWd(tk_clean_2, del_words)        # Cleaned corpus
    
    # Save cleaned corpus
    with open(save_path + os.sep + 'corpus_cleaned.pkl', 'wb') as f:
        pickle.dump(cp, f, pickle.HIGHEST_PROTOCOL)
    
    # ------------------------ Transformation -----------------------

    # Build BoW
    d = Dictionary(cp)  # Each key in the dictionary is unique
    
    # Create a df containing words, word index, word document frequency
    tk_id = pd.DataFrame.from_dict(d.token2id, orient='index', columns = ['index'])
    tk_id = tk_id.reset_index().rename(columns = {'level_0':'word'})
    tk_id = tk_id.set_index(keys = 'index')
    docfreq = pd.DataFrame.from_dict(d.dfs, orient='index', columns = ['doc_freq'])
    docfreq = docfreq.join(tk_id).sort_values(by = 'doc_freq', ascending = False)
    
    # Create a df_pc column to show doc frequency in %
    docfreq['df_pc'] = docfreq['doc_freq'] / 2965 * 100
    
    # Visualize word freq
    wfreq = docfreq[['df_pc']]
    hist_all = wfreq.plot(kind = 'hist',title = 'Document Frequency Histogram', color='orange')
    hist_l20 = wfreq.loc[wfreq['df_pc'] > 20].plot(kind = 'hist',title = 'Document Frequency Histogram (>20%)')
    hist_5_20 = wfreq.loc[(wfreq['df_pc'] <= 20) & (wfreq['df_pc'] > 5)].plot(kind = 'hist',\
                                            title = 'Document Frequency Histogram (5-20%)')
    hist_s5 = wfreq.loc[wfreq['df_pc'] <= 5].plot(kind = 'hist',\
                                            title = 'Document Frequency Histogram (<=5%)')
    hist_s05 = wfreq.loc[wfreq['df_pc'] <= 0.5].plot(kind = 'hist',\
                                            title = 'Document Frequency Histogram (<=0.5%)')
    
    # Target docfreq: 10 docs to 100 docs
    # show in ten docs % freq: 10/2965*100; in 100 docs: 100/2965*100
    # Histogram
    hist_10_100 = wfreq.loc[(wfreq['df_pc'] <= 100/2965*100) & (wfreq['df_pc'] > 10/2965*100)].plot(kind = 'hist',\
                                            title = 'Target Document Frequency Histogram (0.3-3%)')
    
    target = docfreq.loc[(docfreq['df_pc'] <= 100/2965*100) & (docfreq['df_pc'] > 10/2965*100)] 
    
    # Save docfreq
    target.to_csv(save_path + os.sep + 'docfreq.csv')  # Check whether remaining words are meaningful
    
    # Clean dictionary
    d.filter_extremes(no_below = 11, no_above = 0.034)
    
    # Further clean after inspecting the dictionary
    bad = ['14e','1950s','1970s','1980s','1990s', '2000s','21e','27a','2a','3a',\
               '3b','401k','505b','7a','9a','co','vii']
    d.filter_tokens(bad_ids = [d.token2id[i] for i in bad])
    
    # Save dictionary
    d.save_as_text(save_path + os.sep + 'dictionary.txt') 
    
    # Create BoW
    bow = [d.doc2bow(doc) for doc in cp]
    
    # tf-idf weighting
    tfidf = TfidfModel(bow)
    tfidf_bow = [tfidf[bow[i]] for i in range(len(bow))] # weighted bow
    
    # Save tfidf_bow
    with open(save_path + os.sep + 'tfidf_bow.pkl', 'wb') as f:
        pickle.dump(tfidf_bow, f, pickle.HIGHEST_PROTOCOL)
    
    # tf-idf weighting to matrix
    mm = gensim.matutils.corpus2csc(tfidf_bow, num_terms = len(d))  # Use sparse matrix
    
    # Save matrix
    scipy.sparse.save_npz(save_path + os.sep +'mm.npz', mm)
