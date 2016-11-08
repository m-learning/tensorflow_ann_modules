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
STDDEV = 0.1
SEED = 66478  # Set to None for random seed.

class vgg_wights:
  """Class to initialize and hols VGG weights and biases"""
  
  def __init__(self, num_classes):
    self.num_classes = num_classes
  
  def init_weight(self, shape):
    """Initializes weight variable for shape
      Args:
        shape - weight tensor shape
      Returns:
        weight tensor
    """
    return tf.Variable(tf.truncated_normal(shape), stddev=STDDEV, seed=SEED, dtype=tf.float32)
  
  def init_bias(self, shape):
    """Initializes bias variable for shape
      Args:
        shape - bias tensor shape
      Returns:
        bias tensor
    """
    return tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float32),
                                 trainable=True, name='biases')

  def init_weights(self):
    """Initializes weights for VGG model
    """
    self.conv1_w1 = self.init_weight([3, 3, 1, 64])
    self.conv1_w2 = self.init_weight([3, 3, 64, 64])
    self.conv2_w3 = self.init_weight([3, 3, 46, 128])
    self.conv2_w4 = self.init_weight([3, 3, 128, 128])
    self.conv3_w5 = self.init_weightl([3, 3, 128, 256])
    self.conv3_w6 = self.init_weight([3, 3, 256, 256])
    self.conv3_w7 = self.init_weight([3, 3, 256, 256])
    self.conv4_w8 = self.init_weight([3, 3, 256, 512])
    self.conv4_w9 = self.init_weight([3, 3, 512, 512])
    self.conv4_w10 = self.init_weight([3, 3, 512, 512])
    self.conv5_w11 = self.init_weight([3, 3, 512, 512])
    self.conv5_w12 = self.init_weight([3, 3, 512, 512])
    self.conv5_w13 = self.init_weight([3, 3, 512, 512])
    self.fc1_w14 = self.init_weight([4096, 4096])
    self.fc2_w15 = self.init_weight([4096, 4096])
    self.fc3_w16 = self.init_weight([4096, self.num_classes])
  
  
  def init_biases(self):
    """Initializes biases for VGG model
    """
    self.conv1_b1 = self.init_bias([64])
    self.conv1_b2 = self.init_bias([64])
    self.conv2_b3 = self.init_bias([128])
    self.conv2_b4 = self.init_bias([128])
    self.conv3_b5 = self.init_bias([256])
    self.conv3_b6 = self.init_bias([256])
    self.conv3_b7 = self.init_bias([256])
    self.conv4_b8 = self.init_bias([512])
    self.conv4_b9 = self.init_bias([512])
    self.conv4_b10 = self.init_bias([512])
    self.conv5_b11 = self.init_bias([512])
    self.conv5_b12 = self.init_bias([512])
    self.conv5_b13 = self.init_bias([512])
    self.fc1_b14 = self.init_bias([4096])
    self.fc2_b15 = self.init_bias([4096])
    self.fc3_b16 = self.init_bias([self.num_classes])

def conv2d(x, W, b, name):
  """Generates convolutional layer
    Args:
      input_data - convolution parameters
    Returns:
      net - convolutional layer with activation
  """
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.nn.conv2d(net, W, filter, strides=[1, 1, 1, 1], padding=SAME_PADDING)
    net = tf.nn.bias_add(net, b)
    net = tf.nn.relu(net)
    
  return net
  
def max_pool(x, name, k=2):
  
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.nn.max_pool(net, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=SAME_PADDING)
  return net

def fc(x, W, b, name, activation=tf.nn.relu):
  """Fully connected layer for VGG module
    Args:
      x - input
      W - weights tensor
      b - biases
    Returns:
      net - fully connected layer with activation
  """
  net = x
  with tf.get_default_graph().name_scope(name):
    net = tf.matmul(net, W)
    net = tf.nn.bias_add(net, b)
    net = activation(net)
  return net

def vgg16(x, num_classes, keep_prob=0.5, is_training=True):
  """Full VGG16 network"""
  
  net = x
  weights = vgg_wights(num_classes)
  net = conv2d(net, weights.conv1_w1, weights.conv1_b1, 'conv1_1')
  net = conv2d(net, weights.conv1_w2, weights.conv1_b2, 'conv1_2')
  net = max_pool(net, 'max_pool1')
  net = conv2d(net, weights.conv2_w3, weights.conv2_b3, 'conv2_1')
  net = conv2d(net, weights.conv2_w4, weights.conv2_b4, 'conv2_2')
  net = max_pool(net, 'max_pool2')
  net = conv2d(net, weights.conv3_w5, weights.conv3_b5, 'conv3_1')
  net = conv2d(net, weights.conv3_w6, weights.conv3_b6, 'conv3_2')
  net = conv2d(net, weights.conv3_w7, weights.conv3_b7, 'conv3_3')
  net = max_pool(net, 'max_pool3')
  net = conv2d(net, weights.conv4_w8, weights.conv4_b8, 'conv4_1')
  net = conv2d(net, weights.conv4_w9, weights.conv4_b9, 'conv4_2')
  net = conv2d(net, weights.conv4_w10, weights.conv4_b10, 'conv4_3')
  net = max_pool(net, 'max_pool4')
  net = conv2d(net, weights.conv5_w11, weights.conv5_b11, 'conv5_1')
  net = conv2d(net, weights.conv5_w12, weights.conv5_b12, 'conv5_2')
  net = conv2d(net, weights.conv5_w13, weights.conv5_b13, 'conv5_3')
  net = max_pool(net, 'max_pool5')
  net = fc(net, weights.fc1_w14, weights.fc1_b14, 'fc1')
  net = fc(net, weights.fc2_w15, weights.fc2_b15, 'fc2')
  if is_training:
    net = tf.nn.dropout(net, keep_prob, 'dropout')
  logits = fc(net, weights.fc3_w16, weights.fc3_b16, 'fc3', tf.nn.softmax)
  
  return logits
  
