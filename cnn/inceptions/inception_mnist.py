"""
Created on Jan 4, 2017

Training Inception model on MNIST dataset

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cnn.inceptions.cnn_files import training_file
import cnn.inceptions.network_config as network
import numpy as np
import pandas as pd
import tensorflow as tf


num_steps = 20000
batch_size = 50
IM_SIZE = 28

_files = training_file()

_files.get_or_init_training_set()
train_set = pd.read_csv(_files.training_set, header=None)
test_set = pd.read_csv(_files.test_set, header=None)

# get labels in own array
train_lb = np.array(train_set[0])
test_lb = np.array(test_set[0])

# one hot encode the labels
train_lb = (np.arange(10) == train_lb[:, None]).astype(np.float32)
test_lb = (np.arange(10) == test_lb[:, None]).astype(np.float32)

# drop the labels column from training dataframe
trainX = train_set.drop(0, axis=1)
testX = test_set.drop(0, axis=1)

# put in correct float32 array format
trainX = np.array(trainX).astype(np.float32)
testX = np.array(testX).astype(np.float32)

# reformat the data so it's not flat
trainX = trainX.reshape(len(trainX), 28, 28, 1)
testX = testX.reshape(len(testX), 28, 28, 1)

# get a validation set and remove it from the train set
trainX, valX, train_lb, val_lb = trainX[0:(len(trainX) - 500), :, :, :], trainX[(len(trainX) - 500):len(trainX), :, :, :], \
                            train_lb[0:(len(trainX) - 500), :], train_lb[(len(trainX) - 500):len(trainX), :]

# need to batch the test data because running low on memory
class test_batchs:
  """Tests MNIST training batches"""
  
  def __init__(self, data):
      self.data = data
      self.batch_index = 0
      
  def nextBatch(self, batch_size):
      if (batch_size + self.batch_index) > self.data.shape[0]:
          print ("batch sized is messed up")
      batch = self.data[self.batch_index:(self.batch_index + batch_size), :, :, :]
      self.batch_index = self.batch_index + batch_size
      return batch

# set the test batch size
test_batch_size = 100

# returns accuracy of model
def accuracy(target, predictions):
    return(100.0 * np.sum(np.argmax(target, 1) == np.argmax(predictions, 1)) / target.shape[0])
  
# use os to get our current working directory so we can save variable
file_path = _files.checkpoint_path

graph = tf.Graph()
with graph.as_default():
  # train data and labels
  X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
  y_ = tf.placeholder(tf.float32, shape=(batch_size, 10))
    
  # validation data
  tf_valX = tf.placeholder(tf.float32, shape=(len(valX), 28, 28, 1))
  
  # test data
  tf_testX = tf.placeholder(tf.float32, shape=(test_batch_size, 28, 28, 1))
  
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network.model(X, 28, train=True), y_))
  opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  predictions_val = network.interface(tf_valX, 28)
  predictions_test = network.interface(tf_testX, 28)
  
  # initialize variable
  init = tf.global_variables_initializer()
  
  # use to save variables so we can pick up later
  saver = tf.train.Saver()
  
num_steps = 20000
with tf.Session(graph=graph) as sess:
  
  # initialize variables
  sess.run(init)
  print("Model initialized.")
  
  # set use_previous=1 to use file_path model
  # set use_previous=0 to start model from scratch
  use_previous = 1
  
  # use the previous model or don't and initialize variables
  if use_previous and os.path.exists(file_path):
      saver.restore(sess, file_path)
      print("Model restored.")
  
  # training
  for s in range(num_steps):
      offset = (s * batch_size) % (len(trainX) - batch_size)
      batch_x, batch_y = trainX[offset:(offset + batch_size), :], train_lb[offset:(offset + batch_size), :]
      feed_dict = {X : batch_x, y_ : batch_y}
      _, loss_value = sess.run([opt, loss], feed_dict=feed_dict)
      if s % 100 == 0:
          feed_dict = {tf_valX : valX}
          preds = sess.run(predictions_val, feed_dict=feed_dict)
          
          print("step: ", str(s))
          print("validation accuracy: ", str(accuracy(val_lb, preds)))
          print(" ")
          
      # get test accuracy and save model
      if s == (num_steps - 1):
          # create an array to store the outputs for the test
          result = np.array([]).reshape(0, 10)
  
          # use the batches class
          batch_testX = test_batchs(testX)
  
          for i in range(len(testX) // test_batch_size):
              feed_dict = {tf_testX : batch_testX.nextBatch(test_batch_size)}
              preds = sess.run(predictions_test, feed_dict=feed_dict)
              result = np.concatenate((result, preds), axis=0)
          
          print ("test accuracy: ", str(accuracy(test_lb, result)))
          
          save_path = saver.save(sess, file_path)
          print("Model saved.")
