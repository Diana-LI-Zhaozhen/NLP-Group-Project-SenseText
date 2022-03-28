# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:09:40 2022

@author: zhaoz
"""

import pandas as pd
import os
import pickle

data_0 = open(r'C:\Users\zhaoz\Desktop\Text Analysis and NLP\NLP-Group-Project-SenseText-master\datasets\text_analysis.pkl', 'rb')
data_1 = pd.read_csv(r'C:\Users\zhaoz\Desktop\Text Analysis and NLP\NLP-Group-Project-SenseText-master\datasets\tic_rev_ind.csv')
data_2 = pd.read_csv(r'C:\Users\zhaoz\Desktop\Text Analysis and NLP\NLP-Group-Project-SenseText-master\datasets\valuation.csv')
b = open(r'C:\Users\zhaoz\Desktop\Text Analysis and NLP\NLP-Group-Project-SenseText-master\datasets\b_words.pkl', 'rb')
rf = open(r'C:\Users\zhaoz\Desktop\Text Analysis and NLP\NLP-Group-Project-SenseText-master\datasets\rf_words.pkl', 'rb')

data_0 = pickle.load(data_0)
b = pickle.load(b)
rf = pickle.load(rf)

# expand the text value with corresponding words
data_0=pd.concat([data_0,data_0['b_tfidf'].apply(pd.Series,index=b)],axis=1)
data_0=pd.concat([data_0,data_0['rf_tfidf'].apply(pd.Series,index=rf)],axis=1)
data_0=data_0.drop(columns=['b_binary','b_tfidf','rf_binary','rf_tfidf'])

# find the common tikers as train, incommon tickers as test 
# train 2964*1626 
# test 5*1626 ['SAIA', 'RGP', 'MSEX', 'AIHS', 'INO']
common_ticker = list(set(data_0['ticker'].tolist())^ set(data_2['ticker'].tolist()))
data_test=pd.DataFrame()
data_train=data_0
for x in common_ticker:
    data_test= data_test.append(data_0.loc[data_0['ticker']== x])
    data_train=data_train.drop(data_0.loc[data_0['ticker']== x].index)
data_test.reset_index(drop=True,inplace=True)
data_train.reset_index(drop=True,inplace=True)

# clean the dataset
import numpy as np
data_train.fillna(0,inplace = True)
data_train.replace(np.inf,-1)
print(np.any(np.isfinite(data_train.iloc[:,data_train.columns!= 'ticker'])))
print(np.any(np.isnan(data_train.iloc[:,data_train.columns!= 'ticker'])))

#%%
# PCA：
# sse 
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
score_all=[]
df=data_train.iloc[:,data_train.columns!= 'ticker']
list_1 = range(100,1000,100)
for i in list_1:
    pca=PCA(n_components=i)
    new_df= pca.fit(df)
    score= pca.explained_variance_ratio_
    score_all.append(score.sum())
plt.plot(list_1,score_all)
plt.show()
score_all

# from the float chart we think dimension should be 400
pca=PCA(n_components=300)
new_df= pca.fit(df)
score= pca.explained_variance_ratio_
print(score.sum())

trans_matric = pca.components_
trans_matric.shape

index = [col for col in data_train.columns if col not in ['ticker']]

# np.dot(data_test[np.unique(index)].to_numpy()，trans_matric.T)
data_test.fillna(0,inplace = True)
data_test.replace(np.inf,-1)
test_pca_array = np.dot(data_test[np.unique(index)].to_numpy(),trans_matric.T)
train_pca_array = np.dot(data_train[np.unique(index)].to_numpy(),trans_matric.T)

train_pca_array.shape

#%%
# K-means 
# calinski_harabasz_score
from sklearn.metrics import calinski_harabasz_score
score_all=[]
list_1 = range(2,10,1)
for i in list_1:
    kmeans=KMeans(n_clusters=i,random_state=123).fit(train_pca_array)
    score=calinski_harabasz_score(train_pca_array,kmeans.labels_)
    score_all.append(score)
    print('%d cluster has calinski_harabaz score：%f'%(i,score))
plt.plot(list_1,score_all)
plt.show()

# sse 
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# test what k is suitable
k = np.arange(1,100,10)
jarr = []
for i in k:
    model = KMeans(n_clusters=i)
    model.fit(train_pca_array)
    jarr.append(model.inertia_)
    # 给这几个点打标
    plt.annotate(str(i),(i,model.inertia_))
plt.plot(k,jarr)
plt.show()
# K=4 
# using sum of the square erors to determine the best K

# K-means 
from sklearn.metrics import calinski_harabasz_score
kmeans=KMeans(n_clusters= 144,random_state=123).fit(train_pca_array)
cluster_num =kmeans.predict(test_pca_array)

aim_list = pd.DataFrame()
for x in list(data_train[kmeans.labels_==cluster_num[0]]['ticker']):
    # print(data_2[data_2['ticker']==x])
    aim_list = aim_list.append(data_2.loc[data_2['ticker']==x])
#data_test= data_test.append(data_0.loc[data_0['ticker']== x])

#take 59 SAIA as an example
aim_list.replace(np.inf,np.NaN)
aim_list.mean(axis=0,skipna=True)
# SAIA P/E by 2022.3.15 is 27.92

result_list = pd.DataFrame()
for i in cluster_num:
    aim_list = pd.DataFrame()
    for x in list(data_train[kmeans.labels_==i]['ticker']):
        aim_list = aim_list.append(data_2.loc[data_2['ticker']==x])
    aim_list.replace(np.inf,np.NaN)
    result = aim_list.mean(axis = 0,skipna=True).to_frame().T
    result_list = result_list.append(result)
result_list.insert(column = 'ticker', value = data_test['ticker'].tolist(),loc=0)
result_list.reset_index(drop=True,inplace=True)
result_list

#%%
#euclidean distance top 5 company
col = np.unique([co for co in data_train.columns if co != 'ticker'])
distance_matrix = pd.DataFrame()
distance_matrix["distance"] = np.power(data_train[col].to_numpy() - data_test[col].to_numpy()[0], 2).sum(axis=1)
distance_matrix.insert(column = 'ticker', value = data_train['ticker'].tolist(),loc=0)

set_1 = distance_matrix.sort_values(by="distance").head(15)['ticker']

aim_list_euclidean = pd.DataFrame()
for x in list(set_1):
    aim_list_euclidean = aim_list_euclidean.append(data_2.loc[data_2['ticker']==x])
    
aim_list_euclidean.replace(np.inf,np.NaN)
aim_list_euclidean.mean(axis=0)

result_list = pd.DataFrame()
for i in range(0,len(data_test['ticker'])):
    distance_matrix = pd.DataFrame()
    distance_matrix["distance"] = np.power(data_train[col].to_numpy() - data_test[col].to_numpy()[i], 2).sum(axis=1)
    distance_matrix.insert(column = 'ticker', value = data_train['ticker'].tolist(),loc=0)
    set_1 = distance_matrix.sort_values(by="distance").head(15)['ticker']
    aim_list_euclidean = pd.DataFrame()
    for x in list(set_1):
        aim_list_euclidean = aim_list_euclidean.append(data_2.loc[data_2['ticker']==x])
    aim_list_euclidean.replace(np.inf,np.nan)
    result=aim_list_euclidean.mean(axis=0).to_frame().T
    result_list = result_list.append(result)
result_list.insert(column = 'ticker', value = data_test['ticker'].tolist(),loc=0)
result_list.reset_index(drop=True,inplace=True)

#%%
#DBSCAN
from sklearn.cluster import DBSCAN
res = []
for eps in np.arange(50,150,5):
    for min_samples in range(4,10,2):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        dbscan.fit(train_pca_array)
        # cluster number
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # abnormal points number
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        # statistic clustering sample number
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})
# store the result      
df = pd.DataFrame(res)

# fail to clustering 
# trial 0-10000,100 + 10-30,2 
#trial 1-200,10+2,10,2

#%%
#SOM

from minisom import MiniSom
cluster_size=(12,12)
som = MiniSom(cluster_size[0],cluster_size[1],train_pca_array.shape[1],sigma=5, learning_rate = .5,neighborhood_function = 'gaussian',random_seed = 10)
som.train_batch(train_pca_array,500,verbose=True)
active_layer_cordinates = np.array([som.winner(x) for x in train_pca_array]).T
cluster_index = np.ravel_multi_index(active_layer_cordinates,cluster_size)

pd.value_counts(cluster_index)

cluster_index
heatmap=som.distance_map()
plt.imshow(heatmap,cmap='bone_r')
plt.colorbar()

predict_active_cordinates = np.array([som.winner(x) for x in test_pca_array]).T
predict_index = np.ravel_multi_index(predict_active_cordinates,cluster_size)
predict_index

predict_index[0]

aim_list = pd.DataFrame()
for x in list(data_train[cluster_index==predict_index[0]]['ticker']):
    # print(data_2[data_2['ticker']==x])
    aim_list = aim_list.append(data_2.loc[data_2['ticker']==x])
#aim_list

aim_list.replace(np.inf,np.NaN)
aim_list.mean(axis=0)

result_list = pd.DataFrame()
for i in predict_index:
    aim_list = pd.DataFrame()
    for x in list(data_train[cluster_index==i]['ticker']):
        aim_list = aim_list.append(data_2.loc[data_2['ticker']==x])
    aim_list.replace(np.inf,np.NaN)
    result = aim_list.mean(axis = 0).to_frame().T
    result_list = result_list.append(result)
result_list.insert(column = 'ticker', value = data_test['ticker'].tolist(),loc=0)
result_list.reset_index(drop=True,inplace=True)
result_list