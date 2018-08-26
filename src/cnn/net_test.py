# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:27:48 2018

@author: obiwen
"""
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1

inputs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
  net, end_points = resnet_v1.resnet_v1_50(inputs, 1000, global_pool=True, is_training=False)