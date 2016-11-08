"""
Created on Nov 7, 2016
Implementation of VGG network for any size images
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


SAME_PADDING = 'SAME'

def init_weights():
  """Initializes weights for VGG model
    Returns:
      weights - dictionary for weights
  """
  
  weights = {
    'conv1_wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    'conv1_wc2': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    'conv2_wc3': tf.Variable(tf.random_normal([5, 5, 46, 128])),
    'conv2_wc4': tf.Variable(tf.random_normal([5, 5, 128, 128])),
    'conv3_wc5': tf.Variable(tf.random_normal([5, 5, 128, 256])),
    'conv3_wc6': tf.Variable(tf.random_normal([5, 5, 256, 256])),
    'conv3_wc7': tf.Variable(tf.random_normal([5, 5, 256, 256])),
    'conv4_wc7': tf.Variable(tf.random_normal([5, 5, 256, 512])),
    'conv4_wc7': tf.Variable(tf.random_normal([5, 5, 512, 512])),
    'conv4_wc7': tf.Variable(tf.random_normal([5, 5, 512, 512])),
    'conv5_wc7': tf.Variable(tf.random_normal([5, 5, 512, 512])),
    'conv5_wc7': tf.Variable(tf.random_normal([5, 5, 512, 512])),
    'conv5_wc7': tf.Variable(tf.random_normal([5, 5, 512, 512])),
  }
  
  return weights

def conv2d(input_data):
  """Generates convolutional layer
    Args:
      input_data - convolution parameters
    Returns:
      net - convolutional layer with activation
  """
  (x, W, b, name) = input_data
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.nn.conv2d(net, W, filter, strides=[1, 3, 3, 1], padding=SAME_PADDING)
    net = tf.nn.bias_add(net, b)
    net = tf.nn.relu(net)
    
    return net
  
def max_pool(x, name, k=2):
  
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.nn.max_pool(net, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=SAME_PADDING)
  return net
