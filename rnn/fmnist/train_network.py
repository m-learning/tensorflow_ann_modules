"""
Created on Feb 10, 2017

Downloads and organizes data sen and trains network on it

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rnn.fmnist import speech_data
from rnn.fmnist import network_model as network
from rnn.fmnist.rnn_files import training_file

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

def prepare_data():
  """Prepares data set
    Returns:
      tuple of-
        word_batch - training batch
        trainX - training data
        trainY - training lanbels
        testX - test data
        testY - test labels
  """

  batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
  X, Y = next(batch)
  trainX, trainY = X, Y
  testX, testY = X, Y  # overfit for now
  
  return (word_batch, trainX, trainY, testX, testY)

def train_network():
  """Trains network model"""
  
  
  _files = training_file()
  model = network.pretrain_network()
  (_, trainX, trainY, testX, testY) = prepare_data()
  model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=10, verbose=1, validation_data=(testX, testY))
  model.save_weights(_files.model_dir)
