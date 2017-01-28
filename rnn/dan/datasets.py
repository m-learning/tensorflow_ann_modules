"""
Created on Jan 27, 2017

Utility module for DAN data sets

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import imdb
from keras.preprocessing import sequence

from rnn.dan import network_config as config


def _init_imdb_dataset():
  """Initializes training, validation and test data set
    Returns:
      tuple of -
        X_train - training set
        y_train - training labels
        X_test - test set
        y_test - test labels
  """

  print('Loading data...')
  (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=config.max_features)
  print(len(X_train), 'train sequences')
  print(len(X_test), 'test sequences')
  
  print('Pad sequences (samples x time)')
  X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
  X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)
  print('X_train shape:', X_train.shape)
  print('X_test shape:', X_test.shape)
  
  print('Build model...')
  
  return (X_train, y_train, X_test, y_test)

def init_dataset(name='imdb'):
  """Initializes data set by name"""
  
  if name == 'imdb':
    result = _init_imdb_dataset()
  else:
    result = None
  
  return result
  
