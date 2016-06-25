'''
Created on Jun 18, 2016

Initializes convolutional and pooling layers for network

@author: Levan Tsinadze
'''

import tensorflow as tf
import cnn_parameters as pr
from cnn_parameters import cnn_weights

# CNN network functions
class cnn_functions:
    
    def __init__(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, pr.N_INPUT])
        self.y = tf.placeholder(tf.float32, [None, pr.N_CLASSES])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


    # Convolutional layer
    def conv2d(self, x, W, b, strides=1):
        
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        
        return tf.nn.relu(x)
    
    # Pooling layer
    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')
    
    
    # Create model
    def conv_net(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)
    
        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
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
    
    # Neural network training functions
    def cnn_pred(self):
        
        cnn_params = cnn_weights()
        # Construct model
        pred = self.conv_net(self.x, cnn_params.weights, cnn_params.biases, self.keep_prob)
        
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        return (pred, correct_pred, accuracy)
