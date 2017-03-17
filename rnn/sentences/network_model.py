"""
Created on Mar 17, 2017

Simple LSTM model for sentence classification

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential

from rnn.sentences import data_logger as logger


def _init_embedding(flags, is_training=False):
  """Initializes embedding layer
    Args:
      flags - training parameters
      is_training - flag to distinguish training and evaluation
    Returns:
      network_layer - embedding layer
  """
  
  if is_training:
    network_layer = Embedding(flags.top_words, flags.embedding_vecor_length, input_length=flags.max_review_length, dropout=0.2)
  else:
    network_layer = Embedding(flags.top_words, flags.embedding_vecor_length, input_length=flags.max_review_length)
  
  return network_layer

def _init_lstm(is_training):
  """Initializes LSTM layer
    Args:
      is_training - flag to distinguish training and evaluation
    Returns:
      network_layer - LSTM layer
  """
  
  if is_training:
    network_layer = LSTM(100, dropout_W=0.2, dropout_U=0.2)
  else:
    network_layer = LSTM(100)
    
  return network_layer

def init_model(flags, is_training):
  """Initializes network model
    Args:
      flags - training parameters
      is_training - flag to distinguish training and evaluation
    Return:
      model - training model
  """
  
  model = Sequential()
  embedding_layer = _init_embedding(flags, is_training)
  model.add(embedding_layer)
  lstm_layer = _init_lstm(is_training)
  model.add(lstm_layer)
  model.add(Dense(1, activation='sigmoid'))
  
  return model

def prepare_for_train(flags, is_training=True):
  """Initializes network model
    Args:
      flags - training parameters
      is_training - flag to distinguish training and evaluation
    Return:
      model - training model
  """
  
  model = init_model(flags, is_training=is_training)
  logger.log_model(flags, model)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model
  
