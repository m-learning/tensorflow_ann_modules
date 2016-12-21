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
    
  def _init_weight(self, shape, wdc=None):
    """Initializes weight with decay
      Args:
        shape - weight tensor shape
        wdc - weights decay
      Returns:
        weight - initialized weight variable
    """
    
    weight = tf.Variable(tf.random_normal(shape))
    if wdc:
      weight_decay = tf.mul(tf.nn.l2_loss(weight), wdc, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    
    return weight
  
  def init_weights(self, wdc=None):
    """Initializes weights with decay
      Args:
        wdc - weights decay
    """
    
    # Store layers weights
    # 5x5 conv, 1 input, 32 outputs
    self._wc1 = self._init_weight([5, 5, 1, 32], wdc=wdc)
    # 5x5 conv, 32 inputs, 64 outputs
    self._wc2 = self._init_weight([5, 5, 32, 64], wdc=wdc)
    # fully connected, 7*7*64 inputs, 1024 outputs
    self._wd1 = self._init_weight([7 * 7 * 64, 1024], wdc=wdc)
    # 1024 inputs, 10 outputs (class prediction)
    self._wout = self._init_weight([1024, N_CLASSES], wdc=wdc)    
    
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
    