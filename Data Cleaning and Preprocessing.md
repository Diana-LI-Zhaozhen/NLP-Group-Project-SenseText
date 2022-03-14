---
Title: Data Cleaning and Preprocessing
Date: 2022-03-11 15:44
Category: Progress Report
---

By Group SenseText

Now, we have collected 2965 10k files. The Business Description, Risk Factors and Management discussion and analysis (MD&A) parts were stored as paragraphs in each grid of a dataframe. Then, we want to remove massive redundant information in the text and construct a Bag of Words based on the preprocessed information. This page of the blog records our process of further cleaning the data and establishing our Bag of Words.

Our basic workflow is (1) dropping symbols, numbers, and single characters, (2) removing stop words, (3) lowering cases, (4) tokenizing words and lemmatizing according to tags, and (5) finally select words to form corpus.

# Problem 1: Pos_tag and Lemmatization
Initially, we tried to lemmatize the raw strings first and then perform pos_tag. But this method could give us many unlemmatized words as the image shows: 
![object reference]({static}/images/pic1.jpeg)

After taking a closer look at the results, we found that most of the unlemmatized words are verbs. Then, we realized that the default setting of [```WordNetLemmatizer.lemmatize()```](https://github.com/nltk/nltk/blob/develop/nltk/stem/wordnet.py#L39) is 'N'. Hence, we wrote a function called ```lemma``` to pos tag the words in raw strings before lemmatization. To keep  subsequent analysis simpler, we only returned verb, noun, adjectives, adverbs and foreign words. 
![object reference]({static}/images/pic2.jpeg)


```python
#Return tag of words
wnl = WordNetLemmatizer()
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('F'):
        return wordnet.FW
    else:
        return None

# Lemmatize
def lemma(target_list):
    lem_words = []
    for i in range(len(target_list)):
        if get_wordnet_pos(pos_tag([target_list[i]])[0][1]) == None:
            continue
        else:
            print(i)
            tem = wnl.lemmatize(target_list[i],
                                get_wordnet_pos(pos_tag([target_list[i]])[0][1]))
            lem_words.append(tem)

    return lem_words
```

For the business description item, we think nouns are the most significant since they describes the product/service. For the risk factor item, we think verbs and adverbs should also be included since they may describes the degree of risks and the company's responses.

Thus, the codes for pos_tag and lemmatization were modified:

```python
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
```

#Problem 2: Word Selection

The returned list of words are large in size. We cannot keep all the words because (1) it could cause the problem of excessive dimension (2) some words with high frequency appeared in almost all 10k files and have no special meaning. Ideally, we need to slice the list and keep 500-1000 relatively unique words that could reflect main ideas in the text for each item.

By taking closer look at the words in ther returned lists, we decide to keep the words with frequency 1200-20000. The codes are as follows:

```python
#count the apparence frequency of all the noun 
from collections import Counter        
b_words_freq = Counter(lem_words[0])
rf_words_freq = Counter(lem_words[1])

def remove_key(d):
    for key in list(d.keys()):
        if (key.isalpha() == False) or (len(key)<2):
            d.pop(key)
        if (d[key] > 20000) or (d[key] <1200 ):
            del a[k]

remove_key(b_words_freq)
remove_key(rf_words_freq)
```

At last, we get 630 nouns in the business description bag and 993 words in the risk factor bag.