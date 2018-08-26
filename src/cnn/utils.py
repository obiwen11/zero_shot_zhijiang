# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 23:28:40 2018

@author: obiwen
"""
import re
import pandas as pd
import tensorflow as tf

import preprocessing


####################data loaders####################
def load_attr(datadirect, attr_txt, lblattr_txt):
    '''
    datadirect, attr_txt, classattr_txt
    '''
    
    df_attr = pd.read_csv(datadirect + attr_txt, sep='\t', names=['attr_index', 'attr_name'], index_col=None)
    df_lblattr = pd.read_csv(datadirect + lblattr_txt , sep='\t', names=['label_code'] + df_attr.attr_name.tolist(), index_col=None)
     
    return df_attr, df_lblattr
 
    
def load_pair(datadirect, train_txt, attr_txt, lblattr_txt):
    '''
    datadirect, train_txt, attr_txt, lblattr_txt
    '''
    
    imgname_list, lblcode_list = [], []
    with open(datadirect + train_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            imgname, lblcode = line.strip().split('\t')
            imgname_list.append(imgname)
            lblcode_list.append(lblcode)
        f.close()
    
    imgpath_list = [datadirect + 'train' + '/' + imgname for imgname in imgname_list]
    label_list = [int(re.sub('\D', '', lblcode)) - 1 for lblcode in lblcode_list]
    
    df_attr, df_lblattr = load_attr(datadirect, attr_txt, lblattr_txt)
    df_lblattr_ = pd.merge(pd.DataFrame(lblcode_list, columns=['label_code'], index=None), df_lblattr, on='label_code', how='left')
    
    return imgpath_list, label_list, df_lblattr_.iloc[:, 1:].values.tolist(), df_attr, df_lblattr


def load_map(datadirect, label_txt):
    '''
    datadirect, train_txt
    '''
    
    lblmap = {}
    with open(datadirect + label_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lblcode, lblname = line.strip().split('\t')
            lblmap[lblcode] = lblname
        f.close()
        
    num_classes = max([int(re.sub('\D', '', lblcode)) for lblcode in lblmap.keys()])
    return lblmap, num_classes


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