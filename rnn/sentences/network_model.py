"""
Created on Mar 17, 2017

Simple LSTM model for sentence classification

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential

from rnn.sentences import data_logger as logger
from rnn.sentences import training_flags as config


def _dropout_embedding_layer(model, is_training):
  """Adds dropout layer if is training
    Args:
      model - network model
      is_training - flag to distinguish training and evaluation
  """
  
  if is_training:
    model.add(Dropout(2.0, training=is_training))

def _init_lstm_dropouts(is_training):
  """Initializes LSTM layer
    Args:
      is_training - flag to distinguish training and evaluation
    Returns:
      dropouts - tuple of -
        dropout - dropout for layer
        recurrent_dropout - dropout for recurrent sell
  """
  
  if is_training:
    dropout = 0.2 
    recurrent_dropout = 0.2
  else:
    dropout = 0
    recurrent_dropout = 0
        
  return (dropout, recurrent_dropout)

def init_model(flags, is_training):
  """Initializes network model
    Args:
      flags - training parameters
      is_training - flag to distinguish training and evaluation
    Return:
      model - training model
  """
  
  model = Sequential()
  model.add(Embedding(flags.top_words, flags.embedding_vecor_length, input_length=flags.max_review_length))
  _dropout_embedding_layer(model, is_training)
  (dropout, recurrent_dropout) = _init_lstm_dropouts(is_training)
  model.add(LSTM(100, dropout=dropout, recurrent_dropout=recurrent_dropout))
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
  loss_function = config.init_loss(flags)
  logger.log_message(flags, loss_function)
  model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
  
  return model
  
