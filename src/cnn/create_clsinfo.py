# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:54:25 2018

@author: obiwen
"""

import numpy as np
import pandas as pd
import scipy
from nltk.corpus import wordnet as wn

datadirect = '../data/DatasetA_train_20180813/'
attr_txt = 'attribute_list.txt'
classattr_txt = 'attributes_per_class.txt'
classemb_txt = 'class_wordembeddings.txt'
label_txt = 'label_list.txt'

####
df_attr = pd.read_csv(datadirect + attr_txt, sep='\t', names=['attr_index', 'attr_name'], index_col=None)
df_clsattr = pd.read_csv(datadirect + classattr_txt , sep='\t', names=['class_code'] + df_attr.attr_name.tolist(), index_col=None)

####embedding dim is 300
emb_dim = 300
df_clsemb = pd.read_csv(datadirect + classemb_txt, sep='\t', names=['class_code'] + ['emb{}'.format(i+1) for i in range(emb_dim)], index_col=None)
df_clslbl = pd.read_csv(datadirect + label_txt, sep='\t', names=['class_code', 'class_name'], index_col=None)

class_names = df_clslbl.class_name.apply(lambda x: 'remote_control' if x=='remote-control' else x)
class_names = class_names.apply(lambda x: 'sports_car' if x=='sportscar' else x)

class_names = class_names.tolist()
num_classes = df_clslbl.shape[0]
adj = np.zeros((num_classes, num_classes), dtype=float)

for i in range(num_classes):
    for j in range(num_classes):
        if i == j:
            adj[i, j] = 0
        else:
            if wn.synsets(class_names[i]) and wn.synsets(class_names[j]):
                adj_l, adj_r = wn.synsets(class_names[i])[0], wn.synsets(class_names[j])[0]
                adj[i, j] = adj_l.path_similarity(adj_r)
            else:
                print(class_names[i], class_names[j])


###prepare gcn data
                
adj = scipy.sparse.csr_matrix(adj)
features = scipy.sparse.csr_matrix(df_clsemb.iloc[:, 1:].values)
y_train = 
