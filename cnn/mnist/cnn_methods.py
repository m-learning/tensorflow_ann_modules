'''
Created on Jun 18, 2016

Initializes convolutional and pooling layers for network

@author: Levan Tsinadze
'''

from cnn_parameters import cnn_weights
import cnn_parameters as pr
import tensorflow as tf


STRIDE = 'SAME'

class cnn_functions:
  """CNN network for MNIST classification"""
    
  def __init__(self):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, [None, pr.N_INPUT])
    self.y = tf.placeholder(tf.float32, [None, pr.N_CLASSES])
    self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

  def conv2d(self, x, W, b, strides=1):
    """ Conv2D wrapper, with bias add and ReLu activation function
      Args:
        x - input
        W - weights
        b - biases
        strides - strides for convolution
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=STRIDE)
    x = tf.nn.bias_add(x, b)
    
    return tf.nn.relu(x)
  
  def maxpool2d(self, x, k=2):
    """ MaxPool2D wrapper
      Args:
        x - input
        k - pooling parameter
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=STRIDE)
  
  
  def conv_net(self, x, weights, biases, dropout):
    """Full network interface
      Args:
        x - input tensor
        ksize - kernel size
        strides - strides for convolve and pooling
      Return:
        out - output from network
    """
  
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = self.maxpool2d(conv1, k=2)
    conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = self.maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out
  
  def cnn_pred(self):
    """Prediction function for network interface
      Return: prediction, correct prediction, accuracy
    """
      
    cnn_params = cnn_weights()
    # Construct model
    pred = self.conv_net(self.x, cnn_params.weights, cnn_params.biases, self.keep_prob)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return (pred, correct_pred, accuracy)
