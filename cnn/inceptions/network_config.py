"""
Created on Jan 4, 2017

Inception V3 2 Tower model implementation

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


map1 = 32
map2 = 64
num_fc1 = 700  # 1028
num_fc2 = 10
reduce1x1 = 16
dropout = 0.5

STRIDES = [1, 1, 1, 1]
POOL_KERNEL = [1, 3, 3, 1]
PADDING = 'SAME'

def createWeight(shape, Name):
  """Creates weights matrix for convolution
    Args:
      shape - matrix shape
      Name - operation name
    Returns:
      weights matrix
  """
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=Name)
    
def createBias(shape, Name):
  """Creates biases matrix for convolution
    Args:
      shape - matrix shape
      Name - operation name
    Returns:
      biases matrix
  """
  return tf.Variable(tf.constant(0.1, shape=shape),
                      name=Name)
    
def conv2d_s1(x, W):
  """Convolution operation
    Args:
      x - layer
      W - weights
    Returns:
      convolution operation
  """
  return tf.nn.conv2d(x, W, strides=STRIDES, padding=PADDING)

def max_pool_3x3_s1(x):
  """Max-pooling operation
    Args:
      x - layer
    Returns:
      max-pooling operation
  """
  return tf.nn.max_pool(x, ksize=POOL_KERNEL, strides=STRIDES, padding=PADDING)

def inception_tower_1(x):
  """Inception Module1
    Args:
      x - input layer
    Returns:
      inception1 - concatinated Inception tower
  """
    #
    # follows input
  W_conv1_1x1_1 = createWeight([1, 1, 1, map1], 'W_conv1_1x1_1')
  b_conv1_1x1_1 = createBias([map1], 'b_conv1_1x1_1')
  
  # follows input
  W_conv1_1x1_2 = createWeight([1, 1, 1, reduce1x1], 'W_conv1_1x1_2')
  b_conv1_1x1_2 = createBias([reduce1x1], 'b_conv1_1x1_2')
  
  # follows input
  W_conv1_1x1_3 = createWeight([1, 1, 1, reduce1x1], 'W_conv1_1x1_3')
  b_conv1_1x1_3 = createBias([reduce1x1], 'b_conv1_1x1_3')
  
  # follows 1x1_2
  W_conv1_3x3 = createWeight([3, 3, reduce1x1, map1], 'W_conv1_3x3')
  b_conv1_3x3 = createBias([map1], 'b_conv1_3x3')
  
  # follows 1x1_3
  W_conv1_5x5 = createWeight([5, 5, reduce1x1, map1], 'W_conv1_5x5')
  b_conv1_5x5 = createBias([map1], 'b_conv1_5x5')
  
  # follows max pooling
  W_conv1_1x1_4 = createWeight([1, 1, 1, map1], 'W_conv1_1x1_4')
  b_conv1_1x1_4 = createBias([map1], 'b_conv1_1x1_4')
  
  conv1_1x1_1 = conv2d_s1(x, W_conv1_1x1_1) + b_conv1_1x1_1
  conv1_1x1_2 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_2) + b_conv1_1x1_2)
  conv1_1x1_3 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_3) + b_conv1_1x1_3)
  conv1_3x3 = conv2d_s1(conv1_1x1_2, W_conv1_3x3) + b_conv1_3x3
  conv1_5x5 = conv2d_s1(conv1_1x1_3, W_conv1_5x5) + b_conv1_5x5
  maxpool1 = max_pool_3x3_s1(x)
  conv1_1x1_4 = conv2d_s1(maxpool1, W_conv1_1x1_4) + b_conv1_1x1_4
  
  # concatenate all the feature maps and hit them with a relu
  inception1 = tf.nn.relu(tf.concat(3, [conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4]))
  
  return inception1

def inception_tower_2(inception_x):
  """Inception Module2
    Args:
      x - input layer
    Returns:
      inception2 - second Inception tower
  """
    #
    # follows inception1
  W_conv2_1x1_1 = createWeight([1, 1, 4 * map1, map2], 'W_conv2_1x1_1')
  b_conv2_1x1_1 = createBias([map2], 'b_conv2_1x1_1')
  
  # follows inception1
  W_conv2_1x1_2 = createWeight([1, 1, 4 * map1, reduce1x1], 'W_conv2_1x1_2')
  b_conv2_1x1_2 = createBias([reduce1x1], 'b_conv2_1x1_2')
  
  # follows inception1
  W_conv2_1x1_3 = createWeight([1, 1, 4 * map1, reduce1x1], 'W_conv2_1x1_3')
  b_conv2_1x1_3 = createBias([reduce1x1], 'b_conv2_1x1_3')
  
  # follows 1x1_2
  W_conv2_3x3 = createWeight([3, 3, reduce1x1, map2], 'W_conv2_3x3')
  b_conv2_3x3 = createBias([map2], 'b_conv2_3x3')
  
  # follows 1x1_3
  W_conv2_5x5 = createWeight([5, 5, reduce1x1, map2], 'W_conv2_5x5')
  b_conv2_5x5 = createBias([map2], 'b_conv2_5x5')
    
    # follows max pooling
  W_conv2_1x1_4 = createWeight([1, 1, 4 * map1, map2], 'W_conv2_1x1_4')
  b_conv2_1x1_4 = createBias([map2], 'b_conv2_1x1_4')
  
  conv2_1x1_1 = conv2d_s1(inception_x, W_conv2_1x1_1) + b_conv2_1x1_1
  conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception_x, W_conv2_1x1_2) + b_conv2_1x1_2)
  conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception_x, W_conv2_1x1_3) + b_conv2_1x1_3)
  conv2_3x3 = conv2d_s1(conv2_1x1_2, W_conv2_3x3) + b_conv2_3x3
  conv2_5x5 = conv2d_s1(conv2_1x1_3, W_conv2_5x5) + b_conv2_5x5
  maxpool2 = max_pool_3x3_s1(inception_x)
  conv2_1x1_4 = conv2d_s1(maxpool2, W_conv2_1x1_4) + b_conv2_1x1_4
  
  # concatenate all the feature maps and hit them with a relu
  inception2 = tf.nn.relu(tf.concat(3, [conv2_1x1_1, conv2_3x3, conv2_5x5, conv2_1x1_4]))
  
  return inception2

def dropout_layer(net, train=False):
  """Adds dropout layer if training
    Args:
      net - previous layer
      train - training flag
    Returns:
      out - next layer
  """

  if train:
      out = tf.nn.dropout(net, dropout)
  else:
      out = net
    
  return out

def model(x, size, train=False):
  """Inception model
    Args:
      x - model input
      size - image size
      train - training flag
    Returns:
      out - model output
  """
  
  # Fully connected layers
    # since padding is same, the feature map with there will be 4 28*28*map2
  W_fc1 = createWeight([size * size * (4 * map2), num_fc1], 'W_fc1')
  b_fc1 = createBias([num_fc1], 'b_fc1')
  
  W_fc2 = createWeight([num_fc1, num_fc2], 'W_fc2')
  b_fc2 = createBias([num_fc2], 'b_fc2')
  
  inception1 = inception_tower_1(x)
  inception2 = inception_tower_2(inception1)
  
# flatten features for fully connected layer
  inception2_flat = tf.reshape(inception2, [-1, size * size * 4 * map2])
  
  # Fully connected layers
  h_fc0 = tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1)
  h_fc1 = dropout_layer(h_fc0, train=train)
  out = tf.matmul(h_fc1, W_fc2) + b_fc2
  
  return out

def interface(x, size):
  """Network interface
    Args:
      x - model input
      size - image size
    Returns:
      prediction - predicted value
  """
  
  linear_pred = model(x, size)
  prediction = tf.nn.softmax(linear_pred)
  
  return prediction
