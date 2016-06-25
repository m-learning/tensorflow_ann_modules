'''
Created on Jun 18, 2016

Weights and biases for network

@author: Levan Tsinadze
'''

import tensorflow as tf

# Network Parameters
N_INPUT = 784  # MNIST data input (img shape: 28*28)
N_CLASSES = 10  # MNIST total classes (0-9 digits)
CNN_DROPOUT = 0.75  # Dropout, probability to keep units

# File path to save trained model
PARAMETERS_FILE_CONV_PATH = '/storage/ann/digits/parameters/mt/conv_model.ckpt'
PARAMETERS_FILE_CONV_PATH_SAVED = '/storage/ann/digits/parameters_saved/mt/conv_model.ckpt'

class cnn_weights:
    
    def __init__(self):
        
        # Store layers weight & bias
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
        
        # Biases
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([N_CLASSES]))
        }
    
