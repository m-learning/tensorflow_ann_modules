"""
Created on Jun 18, 2016

Initializes convolutional and pooling layers for network

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.mnist.cnn_parameters import cnn_weights
from cnn.mnist import cnn_parameters as pr
import tensorflow as tf


STRIDE = 'SAME'

class cnn_functions(object):
  """CNN network for MNIST classification"""
    
  def __init__(self):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, [None, pr.N_INPUT])
    self.y = tf.placeholder(tf.float32, [None, pr.N_CLASSES])
    self.weights = cnn_weights()
    self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

  def conv2d(self, x, W, b, strides=1):
    """ Conv2D wrapper, with bias add and ReLu activation function
      Args:
        x - input
        W - weights
        b - biases
        strides - strides for convolution
    """
    net = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=STRIDE)
    net = tf.nn.bias_add(net, b)
    net = tf.nn.relu(net)
    
    return net
  
  def maxpool2d(self, x, k=2):
    """ MaxPool2D wrapper
      Args:
        x - input
        k - pooling parameter
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=STRIDE)
    
  def conv_layers(self):
    """Convolutional network interface
      Args:
        x - input tensor
        ksize - kernel size
        strides - strides for convolve and pooling
      Returns:
        out - output from network
    """  
    
    reshaped = tf.reshape(self.x, shape=[-1, 28, 28, 1])
    conv1 = self.conv2d(reshaped, self.weights.wc1, self.weights.bc1)
    conv1 = self.maxpool2d(conv1, k=2)
    conv2 = self.conv2d(conv1, self.weights.wc2, self.weights.bc2)
    conv2 = self.maxpool2d(conv2, k=2)
    
    return conv2
  
  def conv_net(self):
    """Full network interface
      Args:
        x - input tensor
        ksize - kernel size
        strides - strides for convolve and pooling
      Returns:
        out - output from network
    """
  
    # Reshape input picture
    conv_n = self.conv_layers()

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1_pr = tf.reshape(conv_n, [-1, self.weights.wd1.get_shape().as_list()[0]])
    fc1_z = tf.add(tf.matmul(fc1_pr, self.weights.wd1), self.weights.bd1)
    fc1 = tf.nn.relu(fc1_z)
    # Apply Dropout
    drop = tf.nn.dropout(fc1, self.keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(drop, self.weights.wout), self.weights.bout)
    
    return out
  
  def cnn_pred(self):
    """Prediction function for network interface
      Returns: prediction, correct prediction, accuracy
    """
      
    # Construct model
    pred = self.conv_net()
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return (pred, correct_pred, accuracy)
