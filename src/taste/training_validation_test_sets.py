'''
Created on Jun 16, 2016

@author: levan Tsinadze
'''

# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print X_train
print Y_train
print X_test
print X_test

# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(64)