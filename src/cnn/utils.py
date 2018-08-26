# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 23:28:40 2018

@author: obiwen
"""
import re
import numpy as np
import pandas as pd
import tensorflow as tf

import preprocessing


####################data loaders####################
def load_label(datadirect, label_txt):
    '''
    datadirect, train_txt
    '''
    
    df_lbl = pd.read_csv(datadirect + label_txt, sep='\t', names=['label_code', 'label_name'])
    df_lbl['label_index'] = df_lbl.index
    
    return df_lbl


def load_attr(datadirect, attr_txt, lblattr_txt):
    '''
    datadirect, attr_txt, classattr_txt
    '''
    
    df_attrname = pd.read_csv(datadirect + attr_txt, sep='\t', names=['attr_index', 'attr_name'], index_col=None)
    df_lblattr = pd.read_csv(datadirect + lblattr_txt , sep='\t', names=['label_code'] + df_attrname.attr_name.tolist(), index_col=None)
     
    return df_attrname, df_lblattr
 
    
def load_pair(datadirect, train_txt):
    '''
    datadirect, train_txt
    '''
    
    df_pair = pd.read_csv(datadirect + train_txt, sep='\t', names=['image_name', 'label_code'], index_col=None)
    df_pair['image_path'] = df_pair['image_name'].apply(lambda x: datadirect + 'train' + '/' + x)
    
    return df_pair


emb_dim = 300
def load_emb(datadirect, lblemb_txt):
    '''
    datadirect, lblemb_txt
    '''
    
    df_lblemb = pd.read_csv(datadirect + lblemb_txt, sep=' ', names=['label_name'] + ['emb{}'.format(i+1) for i in range(emb_dim)], index_col=None)
    
    return df_lblemb


#################adjacent matrix creater########################
def create_adjattr(df_lblattr, num_classes):
    '''
    df_lblattr, num_classes
    '''
    
    attrcols = df_lblattr.columns[1:-2]
    df_attr = df_lblattr[attrcols]
    
    adj_attr = np.zeros((num_classes, num_classes ), dtype=np.float32) 
    for i in range(num_classes):
        for j in range(i, num_classes):
            lbl_i, lbl_j = df_lblattr.label_index.iloc[i], df_lblattr.label_index.iloc[j]
            attr_i, attr_j = df_attr.iloc[i, :], df_attr.iloc[j, :]
            sim = attr_i.dot(attr_j).sum() / np.sqrt(attr_i.dot(attr_i).sum()) / np.sqrt(attr_j.dot(attr_j).sum())
            adj_attr[lbl_i, lbl_j], adj_attr[lbl_j, lbl_i] = sim, sim
    
    return adj_attr


def create_adjemb(df_lblemb, num_classes):
    '''
    df_lblemb, num_classes
    '''
    
    embcols = [col for col in df_lblemb.columns if 'emb' in col]
    df_emb = df_lblemb[embcols]
    
    adj_emb = np.zeros((num_classes, num_classes ), dtype=np.float32) 
    for i in range(num_classes):
        for j in range(i, num_classes):
            lbl_i, lbl_j = df_lblemb.label_index.iloc[i], df_lblemb.label_index.iloc[j]
            emb_i, emb_j = df_emb.iloc[i, :], df_emb.iloc[j, :]
            sim = emb_i.dot(emb_j).sum() / np.sqrt(emb_i.dot(emb_i).sum()) / np.sqrt(emb_j.dot(emb_j).sum())
            adj_emb[lbl_i, lbl_j], adj_emb[lbl_j, lbl_i] = sim, sim
    
    return adj_emb


####################preprocessings####################
def read_image(filename, label, attr):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    img_casted = tf.cast(image_decoded, tf.float32, name=None)
    
    return img_casted, label, attr


h, w = 64, 64
def pre_processing(image, label, attr, H=h, W=w, is_training=True):
    preprocessing.preprocess_image(image, H, W, 
                                    is_training=is_training,
                                    resize_side_min=H,
                                    resize_side_max=W
                                  )
     
    return image, label, attr


##################dataset maker##########################
def make_dataset(filenames, labels, attrs, epoch, batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels, attrs))
    dataset = dataset.map(read_image)
    dataset = dataset.map(pre_processing)
#    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)

    return dataset