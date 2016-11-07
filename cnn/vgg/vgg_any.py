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


def conv2d(input_data):
  """Generates convolutional layer
    Args:
      input_data - convolution parameters
    Returns:
      net - convolutional layer with activation
  """
  (x, W, b, activation, stride, name) = input_data
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.nn.conv2d(net, W, filter, strides=[1, stride, stride, 1], padding=SAME_PADDING)
    net = tf.nn.bias_add(net, b)
    net = activation(net)
    
    return net
  
def max_pool(x, name, k=2):
  
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.nn.max_pool(net, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=SAME_PADDING)
  return net
