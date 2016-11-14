"""
Created on Jun 18, 2016

Weights and biases for network

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Network Parameters
N_INPUT = 784  # MNIST data input (img shape: 28*28)
N_CLASSES = 10  # MNIST total classes (0-9 digits)
CNN_DROPOUT = 0.75  # Dropout, probability to keep units

# Defile weights and biases
class cnn_weights(object):
  """Initializes weights and biases"""
    
  def __init__(self):
    # Store layers weights
    # 5x5 conv, 1 input, 32 outputs
    self._wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    # 5x5 conv, 32 inputs, 64 outputs
    self._wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    # fully connected, 7*7*64 inputs, 1024 outputs
    self._wd1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
    # 1024 inputs, 10 outputs (class prediction)
    self._wout = tf.Variable(tf.random_normal([1024, N_CLASSES]))
      
    # Store layers biases
    self._bc1 = tf.Variable(tf.random_normal([32]))
    self._bc2 = tf.Variable(tf.random_normal([64]))
    self._bd1 = tf.Variable(tf.random_normal([1024]))
    self._b_out = tf.Variable(tf.random_normal([N_CLASSES]))
    
    # Store layers weights
    self.weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
    }
      
    # Store layers biases
    self.biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([N_CLASSES]))
    }

  @property
  def wc1(self):
    return self._wc1
  
  @property
  def wc2(self):
    return self._wc2
  
  @property
  def wd1(self):
    return self._wd1
  
  @property
  def wout(self):
    return self._wout

  @property
  def bc1(self):
    return self._bc1

  @property
  def bc2(self):
    return self._bc2

  @property
  def bd1(self):
    return self._bd1
  
  @property
  def bout(self):
    return self._b_out    
    