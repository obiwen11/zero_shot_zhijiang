# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 00:18:03 2018

@author: obiwen
"""

import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1

import utils

datadirect = '../../data/DatasetA_train_20180813/'
train_txt = 'train.txt'
label_txt = 'label_list.txt'
attr_txt = 'attribute_list.txt'
lblattr_txt = 'attributes_per_class.txt'
lblemb_txt = 'class_wordembeddings.txt'

##################preparation pipelines#######################
df_lbl = utils.load_label(datadirect, label_txt)
num_classes = df_lbl.shape[0]

df_pair = utils.load_pair(datadirect, train_txt)
df_pair = pd.merge(df_pair, df_lbl, on='label_code', how='left')
imgnum = df_pair.shape[0]

df_attrname, df_lblattr = utils.load_attr(datadirect, attr_txt, lblattr_txt)
df_attr = pd.merge(df_pair[['label_code']], df_lblattr, on='label_code', how='left')
attrnum = df_attrname.shape[0]

df_lblattr = pd.merge(df_lblattr, df_lbl, on='label_code', how='left')
adj_attrsim = utils.create_adjattr(df_lblattr, num_classes)

df_lblemb = utils.load_emb(datadirect, lblemb_txt)
df_lblemb = pd.merge(df_lblemb, df_lbl, on='label_name', how='left')

adj_embsim = utils.create_adjemb(df_lblemb, num_classes)

imagepath_list = df_pair['image_path'].tolist()
label_list = df_pair['label_index'].tolist()
attr_list = df_attr.iloc[:, 1:].values.tolist()

imagepath_list, label_list, attr_list = shuffle(imagepath_list, label_list, attr_list, random_state=731)

files, labels, attrs = tf.constant(imagepath_list), tf.constant(label_list), tf.constant(attr_list)
     
           
def main():
    tf.set_random_seed(731)
    
    Epoch = 200
    Batch_Size = 8
    Lr = 1e-4
    Epoch_Step = int(imgnum/Batch_Size)
    Beta1 = 3
    Beta2 = 0.1
    Val_Per = 0.2
    Patience = 5
    
    adj_attr = np.repeat(adj_attrsim[np.newaxis, :, :], Batch_Size, axis=0)
    adj_emb = np.repeat(adj_embsim[np.newaxis, :, :], Batch_Size, axis=0)
    
    dataset = utils.make_dataset(files, labels, attrs, Epoch, Batch_Size, is_training=True)
    
    iterator = dataset.make_one_shot_iterator()
    next_example, next_label, next_attr = iterator.get_next()
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_50(next_example, 1000, is_training=True)
    
    ###end_points['resnet_v1_50/logits'] (?, 1, 1, 1000)
    class_conv = slim.conv2d(tf.nn.relu(end_points['resnet_v1_50/logits']), 
                            num_classes, 
                            kernel_size=[1, 1],
                            activation_fn=None,
                            scope='class_conv')
    
    attr_conv = slim.conv2d(tf.nn.relu(end_points['resnet_v1_50/logits']), 
                            attrnum, 
                            kernel_size=[1, 1],
                            activation_fn=tf.nn.sigmoid,
                            scope='attr_conv')

    
    class_logits = tf.squeeze(class_conv, axis=[1, 2])
    attr_logits = tf.squeeze(attr_conv, axis=[1, 2])
    
    class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_label, logits=class_logits))
    attr_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(next_attr, attr_logits), 2), axis=1))
    
    class_actv = tf.squeeze(class_conv, axis=[1])
    attrsim_loss = tf.reduce_mean(tf.multiply(tf.multiply(class_actv, adj_attr), tf.transpose(class_actv, perm=[0,2,1])))
    embsim_loss = tf.reduce_mean(tf.multiply(tf.multiply(class_actv, adj_emb), tf.transpose(class_actv, perm=[0,2,1])))
    
    sim_loss = attrsim_loss + embsim_loss
    
    loss = class_loss + Beta1 * attr_loss + Beta2 * sim_loss
    
#    finetune_names = ['block3', 'block4', 'class_conv', 'attr_conv']
#    unrestore_names = ['class_conv', 'attr_conv']
#    finetune_vars = [var for var in tf.trainable_variables() if max([1 if name in var.name else 0 for name in finetune_names])>0]
#    restore_vars = [var for var in tf.trainable_variables() if max([1 if name in var.name else 0 for name in unrestore_names])<=0]
#    training_op = tf.train.AdamOptimizer(learning_rate=Lr).minimize(loss, var_list=finetune_vars)
    training_op = tf.train.AdamOptimizer(learning_rate=Lr).minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
#    print('load weights')
#    restorer = tf.train.Saver(var_list=restore_vars)
#    restorer.restore(sess, '../ckpt/resnet_v1_50.ckpt')
    saver = tf.train.Saver()
    
    print('start training')
    
    best_loss = 999.0
    loss_tracer = {}
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(Epoch):
        print('epoch:{}'.format(epoch))
        epoch_loss = {'train':[], 'val':[]}
        for step in range(Epoch_Step):
            if (step + 1) / Epoch_Step <= (1 - Val_Per):
                loss_, sim_loss_ = sess.run([loss, sim_loss])
                epoch_loss['train'].append(loss_)
                if step and not step % 1:
                    print('step:{}, loss:{}, sim_loss:{}'.format(step, loss_, sim_loss_))
                sess.run(training_op)
            else:
                loss_= sess.run(loss)
                epoch_loss['val'].append(loss_)
        train_loss, val_loss = np.asarray(epoch_loss['train']).mean(), np.asarray(epoch_loss['val']).mean()
        print('epoch:{}, train loss:{}, val loss:{}'.format(epoch, train_loss, val_loss))
        
        if val_loss < best_loss:
            best_loss = val_loss
            saver.save(sess, '../ckpt/model_{}_{}'.format(epoch + 1, '%03f' % (best_loss)))
        
        if (epoch + 1) > Patience and np.asarray(loss_tracer[-Patience:]).mean() < val_loss:
            print('early stoping at epoch:{}'.format(epoch + 1))
            break
        loss_tracer.append(val_loss)
    
    coord.request_stop()
    coord.join(threads)
    
    
if __name__ == '__main__':
    main()